from transformers import BertTokenizer
from classifier import CustomBertClassifier
from datasets import load_dataset
from config import (
    PRE_TRAINED_MODEL_NAME,
    EPOCHS,
    PREDICT_AGREGATION,
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
    LOGIT_POOLER,
    LOGIT_POOLER_LAYER,
    LOGIT_AGREGATION,
)
import torch
from dataset import createDataLoader
import json
from train_eval_fn import Trainer
from tqdm import tqdm

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name_dataset = "lex_glue"
    data_dir = "ecthr_a"

    feature_text = "text"
    feature_label = "labels"

    ecthr_cases = load_dataset(name_dataset, data_dir=data_dir, trust_remote_code=True)

    def collect_labels(labels):
        if SUB_TASK == "multi_label":
            lb = set([__ for _ in tqdm(labels) for __ in _])
        elif SUB_TASK == "multi_class":
            lb = set(labels)
        labels_map = {_: _i for _i, _ in tqdm(enumerate(lb), desc="Collecting labels")}
        json.dump(labels_map, open("labels.json", "w"), ensure_ascii=False, indent=4)

    # collect_labels(ecthr_cases["train"][feature_label])  # type: ignore
    labels_json = json.load(open("labels.json"))
    train = ecthr_cases["train"]  # type: ignore
    test = ecthr_cases["test"]  # type: ignore
    validation = ecthr_cases["validation"]  # type: ignore
    train = {k: train[k][:200] for k in [feature_text, feature_label]}  # type: ignore
    test = {k: test[k][:100] for k in [feature_text, feature_label]}  # type: ignore
    validation = {k: validation[k][:100] for k in [feature_text, feature_label]}  # type: ignore
    model = CustomBertClassifier(
        PRE_TRAINED_MODEL_NAME,
        PREDICT_AGREGATION,
        LOGIT_AGREGATION,
        LOGIT_POOLER,
        LOGIT_POOLER_LAYER,
        num_classes=len(labels_json),
    )
    model = model.to(device)
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    print(f"Nº of parameters: {model.num_parameters}")

    def __(dataloader):
        if name_dataset == "ecthr_cases" or data_dir == "ecthr_a":
            text = ["\n".join(_) for _ in dataloader[feature_text]]
            labels = dataloader[feature_label]
            return {feature_text: text, feature_label: labels}
        else:
            return dataloader

    (
        train_dataloader,
        test_dataloader,
        validation_dataloader,
    ) = map(
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
