import torch.utils
from torch.utils.data import Dataset, BatchSampler
from torch.utils.data import DataLoader
import torch.utils.data
from token_segmentor import Segmentator
import torch
from tqdm import tqdm
import numpy as np
import math
import os


class CustomDataset(Dataset):

    def __init__(self, facts, violations, tokenizer, **kwargs) -> None:
        super(CustomDataset).__init__()
        self.facts = facts
        self.violations = violations
        self.tokenizer = tokenizer
        self.kwargs = kwargs

        self.segmentator = Segmentator()

    def __len__(self):
        return len(self.facts)

    def __getitem__(self, index):
        segment_size = self.kwargs.get("max_length", 0)
        overlap = self.kwargs.get("overlap", 0.0)
        max_length_tokens = self.kwargs.get("max_length_tokens", 510)
        max_segment_qtd = max_length_tokens // segment_size
        fact = self.facts[index]
        violation = self.violations[index]
        if type(violation) is list:
            _v = violation[0]
            if type(_v) is not int:
                raise ValueError(
                    "Violation must be a list of integers.(One Hot Encoding Format)"
                )
        tokens_fact = self.tokenizer(
            fact,
            add_special_tokens=False,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        tokens_fact_after = self.segmentator.splitEvenly(
            tokens_fact, segment_size, overlap, max_segment_qtd
        )

        return {
            "fact_text": fact,
            "input_ids": tokens_fact_after["input_ids"],
            "attention_mask": tokens_fact_after["attention_mask"],
            "targets": torch.tensor(violation, dtype=torch.long),
        }

    def toDataLoader(self, **kwargs) -> torch.utils.data.DataLoader:
        return DataLoader(self, **kwargs)


def custom_collate_fn(batch):
    input_ids_stack, attention_mask_stack, targets_stack, fact_stack = [], [], [], []
    dimensions = [_["input_ids"].shape[0] for _ in batch]
    max_length = max(dimensions)
    for _ in batch:
        if _["input_ids"].shape[0] < max_length:
            padding_qtd = max_length - _["input_ids"].shape[0]
            padding = torch.zeros(
                padding_qtd, _["input_ids"].shape[1], dtype=torch.int64
            )
            for _k, _v in _.items():
                if _k in ["input_ids", "attention_mask"]:
                    _[_k] = torch.cat((_v, padding), dim=0)
        input_ids_stack.append(_["input_ids"])
        attention_mask_stack.append(_["attention_mask"])
        targets_stack.append(_["targets"])
        fact_stack.append(_["fact_text"])
    return {
        "input_ids": torch.stack(input_ids_stack),
        "attention_mask": torch.stack(attention_mask_stack),
        "targets": torch.stack(targets_stack),
        "fact_text": fact_stack,
    }


class CustomBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, org_mode, max_length_tokens):
        self.dataset = dataset
        self.dataset_sorted = sorted(
            tqdm(
                enumerate(self.dataset),
                total=len(self.dataset),
                desc=f"Sorting data for dinamic batch size",
                leave=False,
            ),
            key=lambda x: x[1]["input_ids"].shape[0],
            reverse=True,
        )
        sorted_indices, self.dataset_sorted = zip(*self.dataset_sorted)
        self.original_indices = sorted_indices
        self.org_mode = org_mode
        self.max_length_tokens = max_length_tokens
        self.batch_size = batch_size
        if org_mode == "batch_mode":
            self.batch_size = batch_size
        elif org_mode == "token_mode":
            self.batch_size = max_length_tokens

    def __iter__(self):
        batch_index = []
        batch_token_size = []
        for _i in range(0, len(self.dataset_sorted)):
            flatten_size = self.dataset_sorted[_i]["input_ids"].shape[0] * self.dataset_sorted[_i]["input_ids"].shape[1]  # type: ignore
            batch_ocupation = (
                sum(batch_token_size)
                if self.org_mode == "token_mode"
                else len(batch_index)
            )
            if batch_ocupation < self.batch_size:
                batch_index.append(self.original_indices[_i])
                batch_token_size.append(flatten_size)
            else:
                yield batch_index
                batch_index = [self.original_indices[_i]]
                batch_token_size = [flatten_size]
        if len(batch_index) > 0:
            yield batch_index

    def __len__(self):
        if self.org_mode == "batch_mode":
            return math.ceil(len(self.dataset_sorted) / self.batch_size)
        elif self.org_mode == "token_mode":
            return math.ceil(sum([_["input_ids"].shape[0] * _["input_ids"].shape[1] for _ in self.dataset_sorted]) / self.batch_size)  # type: ignore


def one_hot_encoding(labels, label_map):
    num_labels = len(label_map)
    one_hot = np.zeros(num_labels)
    for label in labels:
        label_index = label_map.get(str(label))
        if label_index is not None:
            one_hot[label_index] = 1
    return one_hot


def createDataLoader(
    data: Dataset,
    max_length,
    overlap,
    max_length_tokens,
    batch_size,
    tokenizer,
    labels_json,
    feature_text,
    feature_label,
    sub_task,
    org_mode,
):
    def filter_ln_fn(lb, sub_task, labels_json):
        if sub_task == "multi_label":
            return set(map(str, lb[1])).issubset(set(labels_json.keys()))
        elif sub_task == "multi_class":
            return str(lb[1]) in labels_json.keys()

    filtred_lb = filter(
        lambda lb: filter_ln_fn(lb, sub_task, labels_json),
        tqdm(
            zip(data[feature_text], data[feature_label]),
            desc="Filtering data per label",
            total=len(data[feature_text]),
            leave=False,
        ),
    )  # type: ignore
    text, labels = zip(*filtred_lb)
    labels_one_hot = [
        one_hot_encoding(lb if type(lb) is list else [lb], labels_json)
        for lb in tqdm(labels, desc="One Hot Encoding", total=len(labels), leave=False)
    ]
    dataset = CustomDataset(
        text,
        labels_one_hot,
        tokenizer,
        max_length=max_length,
        overlap=overlap,
        max_length_tokens=max_length_tokens,
    )
    batch_sampler = CustomBatchSampler(
        dataset,
        batch_size=batch_size,
        org_mode=org_mode,
        max_length_tokens=max_length_tokens,
    )
    dataloader = dataset.toDataLoader(
        collate_fn=custom_collate_fn, batch_sampler=batch_sampler
    )
    return dataloader


class EarlyStopping:
    def __init__(self, output, patience=7, verbose=False, delta=0, trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

        self.step = 0
        self.path_folder = f"{output}_output"
        self.models_dirs = os.makedirs(self.path_folder, exist_ok=True)

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        self.val_loss_min = val_loss
        torch.save(
            model.state_dict(),
            f"{self.path_folder}/step_{self.step}_loss:{self.val_loss_min:.2f}.pt",  # type: ignore
        )
        torch.save(
            model.state_dict(),
            f"{self.path_folder}/best.pt",  # type: ignore
        )
        self.step += 1

    def is_early_stop(self):
        return self.early_stop

    def load_best_model(self, model):
        model.load_state_dict(torch.load(f"{self.path_folder}/best.pt"))
        return model

    def reset(self):
        self.early_stop = False
        self.best_score = None
        self.counter = 0
        self.val_loss_min = np.Inf

    def get_best_score(self):
        return self.best_score

    def get_counter(self):
        return self.counter


if __name__ == "__main__":
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    facts = [
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam eget.",
        "A",
        "Nothing",
        "Warning: This is a warning message!",
    ]
    violations = [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]]
    dataset = CustomDataset(facts, violations, tokenizer, max_length=20, overlap=0.2)
    for data in dataset:
        print(data)
        break
