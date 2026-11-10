"""Dev macro-F1 of the UniXcoder seed average mixed with char LogReg at several weights.

    PYTHONPATH=src python3 -m code_origin.evaluation.ensemble binary_generate [batch_size]
"""
import json, sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from code_origin.config import EXPERIMENTS_DIR

variant = sys.argv[1]
arch = EXPERIMENTS_DIR / "archive"



def runs(mode, model, keep):
    out = []
    for f in (arch / mode).glob(f"*/{variant}__{model}/dev/metrics.json"):
        if keep(json.loads(f.read_text())["args"]) and (f.parent / "proba.npy").exists():
            out.append(f.parent)
    return out

bs = int(sys.argv[2]) if len(sys.argv) > 2 else 16 # batch size of the UniXcoder runs to average
# older runs lack some args, the .get defaults are their old values
ux = runs("transformer", "unixcoder",
          lambda a: a["epochs"] == 4 and a["max_length"] == 256 and a["lr"] == 2e-5
                    and a.get("dropout", 0.1) == 0.0 and a["batch_size"] == bs
                    and a.get("weight_decay", 0.01) == 0.01)
lr = runs("baseline", "char_LogReg", lambda a: True)
print(len(ux), "unixcoder runs,", len(lr), "char LogReg runs")
pred_u = pd.read_parquet(ux[0] / "predictions.parquet")
pred_c = pd.read_parquet(lr[0] / "predictions.parquet")
assert (pred_u.uid.values == pred_c.uid.values).all()      # same rows, same order
y = pred_c.y.values
pu = np.mean([np.load(d / "proba.npy") for d in ux], axis=0)   # average over seeds
pc = np.load(lr[0] / "proba.npy")
f1 = lambda p: f1_score(y, p.argmax(1), average="macro")
print(f"unixcoder, seed average  {f1(pu):.3f}")
print(f"char LogReg              {f1(pc):.3f}")
for w in (0.3, 0.5, 0.7):
    print(f"ensemble, w_unixcoder={w}  {f1(w * pu + (1 - w) * pc):.3f}")