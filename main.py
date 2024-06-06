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
    TOKENS_AGREGATION,
    NUM_HEADS,
    DATA_MODE,
    DATA_INFO,
    COLLECT_LABELS,
    ADD_SECOND_LEVEL,
    NUM_LAYERS,
    HIDDEN_DIM,
    LAYERS_AGGREGATION,
    ADD_CONV,
    KERNEL_SIZE,
)
from dataset import createDataLoader
from train_eval_fn import Trainer
import uuid, shutil, json, torch
from utils import collect_labels

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore

    feature_text = DATA_INFO["feature_text"]
    feature_label = DATA_INFO["feature_label"]
    if DATA_MODE == "huggingface":
        name_dataset = DATA_INFO["name_dataset"]
        data_dir = DATA_INFO["data_dir"]
        dataset = load_dataset(name_dataset, data_dir=data_dir, trust_remote_code=True)
    elif DATA_MODE == "dataframe":
        import pandas as pd
        from pathlib import Path

        paths = DATA_INFO["path"].split("||")
        _df_paths = zip(["train", "validation", "test"], paths)
        dataset = {
            k: pd.read_csv(v) if Path(v).suffix == ".csv" else pd.read_pickle(v)
            for k, v in _df_paths
        }

    if COLLECT_LABELS:
        collect_labels(dataset["train"][feature_label], SUB_TASK)  # type: ignore

    labels_json = json.load(open("labels.json"))
    train = dataset["train"]  # type: ignore
    test = dataset["test"]  # type: ignore
    validation = dataset["validation"]  # type: ignore
    train = {k: train[k][:50] for k in [feature_text, feature_label]}  # type: ignore
    test = {k: test[k][:50] for k in [feature_text, feature_label]}  # type: ignore
    validation = {k: validation[k][:50] for k in [feature_text, feature_label]}  # type: ignore
    model = CustomBertClassifier(
        PRE_TRAINED_MODEL_NAME,
        PREDICT_AGREGATION,
        TOKENS_AGREGATION,
        LAYERS_AGGREGATION,
        LOGIT_POOLER,
        LOGIT_POOLER_LAYER,
        num_classes=len(labels_json),
        second_level=ADD_SECOND_LEVEL,
        conv=ADD_CONV,
        conv_args={"kernel_size": KERNEL_SIZE, "num_filters": MAX_LENGTH},
        second_level_args={
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "hidden_dim": HIDDEN_DIM,
        },
    )
    model = model.to(device)
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    print(f"Nº of parameters: {model.num_parameters}")

    def __(dataloader):
        if DATA_MODE == "huggingface" and (
            name_dataset == "ecthr_cases" or data_dir == "ecthr_a"
        ):
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

    experiment_name = f"{uuid.uuid4()}"

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
        experiment_name=experiment_name,
    )

    trainer.train(EPOCHS, train_dataloader, validation_dataloader)  # type: ignore
    trainer.evaluate(test_dataloader, load_best=True)  # type: ignore

    history = trainer.history
    json.dump(
        history,
        open(f"{experiment_name}_output/history.json", "w"),
        ensure_ascii=False,
        indent=4,
    )
    shutil.copy("config.py", f"{experiment_name}_output/config.py")
    shutil.copy("labels.json", f"{experiment_name}_output/labels.json")
