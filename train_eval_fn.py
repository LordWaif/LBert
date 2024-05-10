from tqdm import tqdm
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from config import get_multi_label_pred_fn,get_multi_class_pred_fn
import torch
from torch import nn
import numpy as np
from collections import defaultdict
from accelerate import Accelerator
import torch.nn.functional as F

def train_epoch(
  model,
  data_loader, 
  loss_fn, 
  optimizer, 
  device, 
  scheduler,
  accelerator:Accelerator,
  sub_task,
  **kwargs
    ):
    cum_targets = []
    cum_preds = []
    losses = []
    model = model.train()
    labels_name = kwargs.get('labels_name',None)
    for batch in tqdm(data_loader,desc=f'Training'):
        with accelerator.accumulate(model):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)
            outputs= model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            if sub_task == "multi_class":
                probs = F.softmax(outputs,dim=1)
                loss = loss_fn(probs, targets.float())
                preds = get_multi_class_pred_fn(probs)
            elif sub_task == "multi_label":
                loss = loss_fn(outputs, targets.float())
                preds = get_multi_label_pred_fn(outputs)
            cum_targets.extend(targets.detach().cpu())
            cum_preds.extend(preds.detach().cpu())
            losses.append(loss.item())
            accelerator.backward(loss)

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    cr:dict = classification_report(cum_targets,cum_preds,output_dict=True,target_names=labels_name) # type: ignore
    cr['loss'] = np.mean(losses)
    return cr

def eval_model(model, data_loader, loss_fn, device, early_stopping,sub_task, **kwargs):
    model = model.eval()
    cum_targets = []
    cum_preds = []
    losses = []
    labels_name = kwargs.get('labels_name',None)

    with torch.no_grad():
        for batch in tqdm(data_loader,desc=f'Evaluating'):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            if sub_task == "multi_class":
                probs = F.softmax(outputs,dim=1)
                loss = loss_fn(probs, targets.float())
                preds = get_multi_class_pred_fn(probs)
            elif sub_task == "multi_label":
                loss = loss_fn(outputs, targets.float())
                preds = get_multi_label_pred_fn(outputs)
            cum_targets.extend(targets.detach().cpu())
            cum_preds.extend(preds.detach().cpu())
            losses.append(loss.item())

    early_stopping(np.mean(losses),model)

    cr:dict = classification_report(cum_targets,cum_preds,output_dict=True,target_names=labels_name) # type: ignore
    cr['loss'] = np.mean(losses)
    # cr['confusion_matrix'] = multilabel_confusion_matrix(cum_targets,cum_preds)
    print(classification_report(cum_targets,cum_preds,target_names=labels_name))
    return cr