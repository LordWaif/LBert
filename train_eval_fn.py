import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F

from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator
from dataset import EarlyStopping
from sklearn.metrics import classification_report
from config import get_multi_label_pred_fn, get_multi_class_pred_fn, DEBUG_MODE


def train_epoch(
    model,
    data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    accelerator: Accelerator,
    sub_task,
    **kwargs,
):
    losses = []
    model = model.train()
    labels_name = kwargs.get("labels_name", None)
    stack_target = []
    stack_preds = []
    for batch in tqdm(data_loader, desc=f"Training"):
        with accelerator.accumulate(model):  # type: ignore
            loss, targets, preds = train_eval_fn(
                model, batch, sub_task, loss_fn, device
            )

            stack_target.append(targets)
            stack_preds.append(preds)
            losses.append(loss.item())
            accelerator.backward(loss)

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    stack_target = torch.cat(stack_target, dim=0)
    stack_preds = torch.cat(stack_preds, dim=0)
    cr: dict = classification_report(stack_target, stack_preds, output_dict=True, target_names=labels_name, zero_division=0)  # type: ignore
    # print(classification_report(cum_targets,cum_preds,target_names=labels_name))
    cr["loss"] = np.mean(losses)
    return cr


def eval_model(
    model,
    data_loader,
    loss_fn,
    device,
    early_stopping=lambda x, y: None,
    sub_task="multi_label",
    **kwargs,
):
    model = model.eval()
    stack_target = []
    stack_preds = []
    losses = []
    labels_name = kwargs.get("labels_name", None)

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating"):
            loss, targets, preds = train_eval_fn(
                model, batch, sub_task, loss_fn, device
            )

            stack_target.append(targets.detach().cpu())
            stack_preds.append(preds.detach().cpu())
            losses.append(loss.item())

    early_stopping(np.mean(losses), model)  # type: ignore
    stack_target = torch.cat(stack_target, dim=0)
    stack_preds = torch.cat(stack_preds, dim=0)
    cr: dict = classification_report(stack_target, stack_preds, output_dict=True, target_names=labels_name, zero_division=0)  # type: ignore
    cr["loss"] = np.mean(losses)
    # cr['confusion_matrix'] = multilabel_confusion_matrix(cum_targets,cum_preds)
    print(
        classification_report(
            stack_target, stack_preds, target_names=labels_name, zero_division=0
        )
    )
    return cr


def train_eval_fn(model, batch, sub_task, loss_fn, device):
    def internal_train_eval_fn(model, batch, sub_task, loss_fn, device):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        if sub_task == "multi_class":
            targets = torch.argmax(targets, dim=1)
            loss = loss_fn(outputs, targets.long())
            probs = F.softmax(outputs, dim=1)
            preds = get_multi_class_pred_fn(probs)
        elif sub_task == "multi_label":
            loss = loss_fn(outputs, targets.float())
            # for t, p in zip(targets, torch.sigmoid(outputs)):
            #     print(t, p)
            preds = get_multi_label_pred_fn(outputs)
        return loss, targets.detach().cpu(), preds.detach().cpu()

    if DEBUG_MODE:
        with torch.autograd.detect_anomaly():  # type: ignore
            return internal_train_eval_fn(model, batch, sub_task, loss_fn, device)
    else:
        return internal_train_eval_fn(model, batch, sub_task, loss_fn, device)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        device,
        sub_task,
        labels_name,
        accumulate_steps,
        early_stopping,
        patience,
        experiment_name,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.sub_task = sub_task
        self.labels_name = labels_name
        self.accumulate_steps = accumulate_steps
        self.early_stopping = early_stopping
        self.patience = patience

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.accumulate_steps
        )

        if self.early_stopping:
            self.early_stopping = EarlyStopping(
                experiment_name, patience=self.patience, verbose=True
            )

        self.history = {"num_parameters": model.num_parameters}

    def save_history(self, func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            self.history[func.__name__] = result
            return result

        return wrapper

    def __getattribute__(self, name):
        attr = super().__getattribute__(name)
        if callable(attr) and name in ["train", "evaluate"]:
            return self.save_history(attr)
        return attr

    def train(
        self,
        epoch: int,
        train_dataloader: Dataset,
        validation_dataloader=None,
    ):
        total_steps = len(train_dataloader) * epoch  # type: ignore
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler
        )
        history = []
        for n_epoch in range(epoch):
            print(f"Epoch {n_epoch+1}/{epoch}")

            classification_report_train = train_epoch(
                self.model,
                train_dataloader,
                self.loss_fn,
                self.optimizer,
                self.device,
                self.scheduler,
                self.accelerator,
                self.sub_task,
                labels_name=self.labels_name,
            )
            if validation_dataloader:
                classification_report_eval = eval_model(
                    self.model,
                    validation_dataloader,
                    self.loss_fn,
                    self.device,
                    self.early_stopping,
                    self.sub_task,
                    labels_name=self.labels_name,
                )
            history.append(
                {
                    "epoch": n_epoch + 1,
                    "train_cr": classification_report_train,
                    "eval_cr": classification_report_eval,
                    "eval_loss": (
                        classification_report_eval["loss"]
                        if validation_dataloader
                        else None
                    ),
                }
            )

            print(
                f'Eval loss {classification_report_eval["loss"] if validation_dataloader else "N/A"}'
            )
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

        return history

    def evaluate(self, test_dataloader: Dataset, load_best=False):
        if load_best:
            best_model = self.early_stopping.load_best_model(self.model)
        else:
            best_model = self.model
        classification_report_test = eval_model(
            best_model,
            test_dataloader,
            self.loss_fn,
            self.device,
            sub_task=self.sub_task,
            labels_name=self.labels_name,
        )
        return classification_report_test

    def get_history(self):
        return self.history
