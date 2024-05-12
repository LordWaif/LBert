import torch.utils
from torch.utils.data import Dataset, BatchSampler
from torch.utils.data import DataLoader
import torch.utils.data
from token_segmentor import Segmentator
import torch
from tqdm import tqdm
import numpy as np
import math


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
        tokens_fact = self.segmentator.splitEvenly(
            tokens_fact, segment_size, overlap, max_segment_qtd
        )

        return {
            "fact_text": fact,
            "input_ids": tokens_fact["input_ids"],
            "attention_mask": tokens_fact["attention_mask"],
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
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataset_sorted = sorted(
            tqdm(enumerate(self.dataset), desc=f"Sorting data for dinamic batch size"),
            key=lambda x: x[1]["input_ids"].shape[0],
            reverse=True,
        )
        sorted_indices, self.dataset_sorted = zip(*self.dataset_sorted)
        self.original_indices = sorted_indices

    def __iter__(self):
        batch_index = []
        for _i in range(0, len(self.dataset_sorted)):
            if len(batch_index) < self.batch_size:
                batch_index.append(self.original_indices[_i])
            else:
                yield batch_index
                batch_index = [self.original_indices[_i]]
        if len(batch_index) > 0:
            yield batch_index

    def __len__(self):
        return math.ceil(len(self.dataset_sorted) / self.batch_size)


class EarlyStopping:
    def __init__(
        self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

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
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.path))

    def is_early_stop(self):
        return self.early_stop

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
