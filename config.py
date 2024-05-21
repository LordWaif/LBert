import torch
from torch import nn
from torch.optim import AdamW
import numpy as np

SUB_TASK = "multi_label"
PRE_TRAINED_MODEL_NAME = "google-bert/bert-base-cased"
EPOCHS = 15

PREDICT_AGREGATION = "mean"
LOGIT_AGREGATION = "mean"

LOGIT_POOLER = "hidden_state"  # "hidden_state" or "pooler_output"
LOGIT_POOLER_LAYER = 0  # last layer: 0 , ante-penultimate: -1 ... n-penultimate: -n

BATCH_SIZE = 2  # 8 4 2
MAX_LENGTH = 512  # 512 256 128
OVERLAP = 0.25  # 0.2 0.1 0.5 0.25
MAX_LENGTH_TOKENS = 4096  # 8192 4096 2048 1024 512
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
