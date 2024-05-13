from transformers import BertTokenizer
from classifier import CustomBertClassifier
from datasets import load_dataset
from config import (
    PRE_TRAINED_MODEL_NAME,
    EPOCHS,
    PREDICT_POOLER,
    BATCH_SIZE,
    MAX_LENGTH,
    OVERLAP,
    MAX_LENGTH_TOKENS,
    LR,
    LOSS,
    OPTIMIZER,
    SUB_TASK,
    ORG_MODE,
    PATIENCE,
    ACCUMULATIVE_STEPS,
)
import torch
from dataset import createDataLoader
import json
from train_eval_fn import Trainer

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name_dataset = "ecthr_cases"
    data_dir = "alleged-violation-prediction"

    feature_text = "facts"
    feature_label = "labels"

    ecthr_cases = load_dataset(name_dataset, data_dir=data_dir, trust_remote_code=True)

    def collect_labels(labels):
        lb = set(labels)
        labels_map = {_: _i for _i, _ in enumerate(lb)}
        json.dump(labels_map, open("labels.json", "w"), ensure_ascii=False, indent=4)

    # collect_labels(ecthr_cases['train']['label']) # type: ignore
    labels_json = json.load(open("labels.json"))
    train = ecthr_cases["train"]  # type: ignore
    test = ecthr_cases["test"]  # type: ignore
    validation = ecthr_cases["validation"]  # type: ignore
    model = CustomBertClassifier(
        PRE_TRAINED_MODEL_NAME, PREDICT_POOLER, num_classes=len(labels_json)
    )
    model = model.to(device)
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    def __(dataloader):
        if name_dataset == "ecthr_cases":
            text = ["\n".join(_) for _ in dataloader["facts"]]
            labels = dataloader["labels"]
            return {"facts": text, "labels": labels}
        else:
            return dataloader

    train_dataloader, test_dataloader, validation_dataloader = map(
        lambda x: createDataLoader(
            x,  # type: ignore
            MAX_LENGTH,
            OVERLAP,
            MAX_LENGTH_TOKENS,
            BATCH_SIZE,
            tokenizer,
            labels_json,
            feature_text,
            feature_label,
            SUB_TASK,
            ORG_MODE,
        ),
        [__(train), __(test), __(validation)],
    )

    optimizer = OPTIMIZER(model.parameters(), lr=LR)
    loss_fn = LOSS().to(device)

    trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        device,
        SUB_TASK,
        labels_json,
        ACCUMULATIVE_STEPS,
        True,
        patience=PATIENCE,
    )

    trainer.train(EPOCHS, train_dataloader, validation_dataloader)  # type: ignore
    trainer.evaluate(test_dataloader)  # type: ignore

    history = trainer.history
    json.dump(history, open("history.json", "w"), ensure_ascii=False, indent=4)
