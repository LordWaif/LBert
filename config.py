import torch
from torch import nn
from transformers import AdamW

PRE_TRAINED_MODEL_NAME = "google-bert/bert-base-cased"
EPOCHS = 10
PREDICT_POOLER = 'mean'
BATCH_SIZE = 8
MAX_LENGTH = 512
OVERLAP = 0.2
MAX_LENGTH_TOKENS = 4096
LR = 5e-5
ACCUMULATIVE_STEPS = 4

LOSS = nn.BCEWithLogitsLoss
OPTIMIZER = AdamW

TRESHOLD = .5
get_multi_label_pred_fn = lambda x: (torch.sigmoid(x) > TRESHOLD).int()