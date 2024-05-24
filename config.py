import torch
from torch import nn
from torch.optim import AdamW
import numpy as np

COLLECT_LABELS = False
DATA_MODE = "huggingface"  # "huggingface" or "dataframe"
DATA_INFO = {
    "name_dataset": "lex_glue",  # "path" : "train.csv||validation.csv||test.csv"
    "data_dir": "ecthr_a",  # ...
    "feature_text": "text",  # "feature_text": "text",
    "feature_label": "labels",  # "feature_label": "labels",
}

SUB_TASK = "multi_label"
PRE_TRAINED_MODEL_NAME = "google-bert/bert-base-cased"
EPOCHS = 15

ADD_LWAN = True
NUM_HEADS = 4

PREDICT_AGREGATION = "mean"
LOGIT_AGREGATION = "mean"

LOGIT_POOLER = "hidden_state"  # "hidden_state" or "pooler_output"
LOGIT_POOLER_LAYER = 0  # last layer: 0 , ante-penultimate: -1 ... n-penultimate: -n

BATCH_SIZE = 2  # 8 4 2
MAX_LENGTH = 128  # 512 256 128
OVERLAP = 0.25  # 0.2 0.1 0.5 0.25
MAX_LENGTH_TOKENS = 128  # 8192 4096 2048 1024 512
LR = 5e-5  # 2e-5
ACCUMULATIVE_STEPS = 1
DEBUG_MODE = False
PATIENCE = 5

LOSS = nn.BCEWithLogitsLoss
OPTIMIZER = AdamW

ORG_MODE = "batch_mode"

TRESHOLD = 0.5  # 0.4 0.6
get_multi_label_pred_fn = lambda x: (torch.sigmoid(x) > TRESHOLD).int()
get_multi_class_pred_fn = lambda x: torch.argmax(torch.tensor(x), dim=1)
