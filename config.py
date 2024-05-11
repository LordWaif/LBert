import torch
from torch import nn
from transformers import AdamW
import numpy as np

SUB_TASK = 'multi_label'
PRE_TRAINED_MODEL_NAME = "google-bert/bert-base-cased"
EPOCHS = 5
PREDICT_POOLER = 'mean'
BATCH_SIZE = 32
MAX_LENGTH = 512
OVERLAP = 0.2
MAX_LENGTH_TOKENS = 2048
LR = 5e-5
ACCUMULATIVE_STEPS = 1

LOSS = nn.BCEWithLogitsLoss
OPTIMIZER = AdamW

TRESHOLD = .5
get_multi_label_pred_fn = lambda x: (torch.sigmoid(x) > TRESHOLD).int()
get_multi_class_pred_fn = lambda x: torch.argmax(torch.tensor(x),dim=1)