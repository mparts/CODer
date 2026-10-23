from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from code_origin.config import DEVICE


def batches(tok, texts, labels, batch_size, shuffle):
    enc = tok(list(texts), truncation=True, max_length=512)
    items = [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i],
              "labels": int(labels[i])} for i in range(len(texts))]
    return DataLoader(items, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda b: tok.pad(b, return_tensors="pt"))


@torch.no_grad()
def predict_proba(model, loader):
    model.eval()
    out = []
    for b in loader:
        b = {k: v.to(DEVICE) for k, v in b.items() if k != "labels"}
        out.append(torch.softmax(model(**b).logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def fit(model, tok, train, n_classes, a):
    loader = batches(tok, train.code_norm, train.y.values, a.batch_size, shuffle=True)
    counts = np.bincount(train.y, minlength=n_classes)
    weight = torch.tensor(len(train) / (n_classes * counts), dtype=torch.float, device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * len(loader)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    for epoch in range(a.epochs):
        model.train()
        total = 0
        for b in tqdm(loader, desc=f"Epoch {epoch+1}/{a.epochs}", unit="batch", ncols=100, colour="green"):
            b = {k: v.to(DEVICE) for k, v in b.items()}
            labels = b.pop("labels")
            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                logits = model(**b).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), labels, weight=weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            total += loss.item()
        print(f"  epoch {epoch + 1}/{a.epochs}  train loss {total / len(loader):.4f}")