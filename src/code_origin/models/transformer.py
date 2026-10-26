from tqdm import tqdm

import numpy as np
import copy

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.nn.functional import cross_entropy

from transformers import get_linear_schedule_with_warmup

from code_origin.config import DEVICE
from code_origin.evaluation.evaluate import evaluate


def batches(tok, texts, labels, batch_size, max_length, shuffle):
    enc = tok(list(texts), truncation=True, max_length=max_length)
    items = [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i], "labels": int(labels[i])} for i in range(len(texts))]
    return DataLoader(items, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda b: tok.pad(b, return_tensors="pt"))


EVALS_PER_EPOCH = 5   # logging only; checkpoint selection still happens at epoch ends


@torch.no_grad()
def dev_pass(model, dev_loader, classes):
    model.eval()
    loss_sum, n = 0.0, 0
    y_true, y_pred = [], []
    for b in dev_loader:
        b = {k: v.to(DEVICE) for k, v in b.items()}
        labels = b.pop("labels")
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            logits = model(**b).logits
        loss_sum += cross_entropy(logits.float(), labels).item() * labels.size(0)
        n += labels.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(logits.argmax(dim=-1).cpu().tolist())
    return loss_sum / n, evaluate(y_true, y_pred, classes)["macro_f1"]


def fit(model, tok, train, dev, classes, a):
    train_loader = batches(tok, train.code_norm, train.y.values, a.batch_size, a.max_length, shuffle=True)
    dev_loader = batches(tok, dev.code_norm, dev.y.values, 64, a.max_length, shuffle=False)
    n_classes = len(classes)
    counts = np.bincount(train.y, minlength=n_classes)
    inv = len(train) / (n_classes * counts)
    if a.class_weight == "none":
        weight = None
    elif a.class_weight == "sqrt":
        weight = torch.tensor(np.sqrt(inv), dtype=torch.float, device=DEVICE)
    else:
        weight = torch.tensor(inv, dtype=torch.float, device=DEVICE)
    opt = AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    n_batches = len(train_loader)
    steps = a.epochs * n_batches
    sched = get_linear_schedule_with_warmup(opt, int(a.warmup_ratio * steps), steps)
    # batch indices (1-based) at which dev is evaluated; the last one is the epoch end
    eval_at = {max(1, round(n_batches * k / EVALS_PER_EPOCH)) for k in range(1, EVALS_PER_EPOCH + 1)}
    history = {"train_loss": [], "dev_loss": [], "dev_macro_f1": [],
               "curve": {"epoch": [], "train_loss": [], "dev_loss": [], "dev_macro_f1": []}}
    best_f1, best_epoch, best_state = -1, None, None
    for epoch in range(a.epochs):
        model.train()
        total_loss, n_seen = 0.0, 0
        int_loss, int_n = 0.0, 0
        for i, b in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{a.epochs}", unit="batch", colour="green", dynamic_ncols=True), 1):
            b = {k: v.to(DEVICE) for k, v in b.items()}
            labels = b.pop("labels")
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                logits = model(**b).logits
            loss = cross_entropy(logits.float(), labels, weight=weight)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            opt.step(), sched.step()
            total_loss += loss.item() * labels.size(0)
            n_seen += labels.size(0)
            int_loss += loss.item() * labels.size(0)
            int_n += labels.size(0)
            if i in eval_at:
                dev_loss, dev_f1 = dev_pass(model, dev_loader, classes)
                model.train()
                c = history["curve"]
                c["epoch"].append(epoch + i / n_batches)
                c["train_loss"].append(int_loss / int_n)
                c["dev_loss"].append(dev_loss)
                c["dev_macro_f1"].append(dev_f1)
                int_loss, int_n = 0.0, 0
        train_loss = total_loss / n_seen
        history["train_loss"].append(train_loss)
        history["dev_loss"].append(dev_loss)
        history["dev_macro_f1"].append(dev_f1)
        print(f"! Epoch {epoch + 1}/{a.epochs} | "
              f"train loss {train_loss:.4f} | "
              f"dev loss {dev_loss:.4f} | "
              f"dev macro-F1 {dev_f1:.4f}")
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    print(f"Best checkpoint: epoch {best_epoch}, dev macro-F1 {best_f1:.4f}")
    history["best_epoch"] = best_epoch
    return history


@torch.no_grad()
def predict_proba(model, loader):
    model.eval()
    out = []
    for b in loader:
        b = {k: v.to(DEVICE) for k, v in b.items() if k != "labels"}
        out.append(torch.softmax(model(**b).logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)