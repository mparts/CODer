# src/code_origin/evaluation/breakdown.py
# PYTHONPATH=src python3 -m code_origin.evaluation.breakdown multiclass
import sys, json
import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from code_origin.config import EXPERIMENTS_DIR, PROCESSED_DIR
from code_origin.helpers import log

v = sys.argv[1]
arch = EXPERIMENTS_DIR / "archive"
classes = json.loads((PROCESSED_DIR / "meta.json").read_text())[v]["classes"]
ux = sorted((arch / "transformer").glob(f"*/{v}__unixcoder/test"))
lr = sorted((arch / "baseline").glob(f"*/{v}__char_LogReg/test"))[0]
ref = pd.read_parquet(lr / "predictions.parquet")
pe = np.mean([np.load(d / "proba.npy") for d in ux], axis=0)
pc = np.load(lr / "proba.npy")
models = {"UniXcoder ens": pe.argmax(1), "char LogReg": pc.argmax(1), "mix 50/50": (.5 * pe + .5 * pc).argmax(1)}
y = ref.y.values
group = np.where(ref.meta_source == "human", "Human", ref.meta_LLM + " (" + ref.meta_source + ")")

log(f"Breakdown for {v}..", "start")
for name, pred in models.items():
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=range(len(classes)), zero_division=0)
    print(f"== {name}: per class")
    print(pd.DataFrame({"precision": p, "recall": r, "F1": f, "n": n}, index=classes).round(3).to_string())

res = {}
for name, pred in models.items():
    d = pd.DataFrame({"group": group, "correct": pred == y, "flagged AI": pred != 0})
    g = d.groupby("group")
    res[(name, "correct")], res[(name, "flagged AI")] = g.correct.mean(), g["flagged AI"].mean()
res = pd.DataFrame(res)
res.insert(0, "n", pd.Series(group).value_counts())
print("\n== by true source (share of files)")
print(res.round(3).to_string())

if len(classes) > 2:
    cm = pd.crosstab(pd.Series(np.array(classes)[y], name="true"), pd.Series(np.array(classes)[models["UniXcoder ens"]], name="predicted"), normalize="index")
    print("\n== UniXcoder ensemble, confusion matrix (share of each true class)")
    print(cm.round(2).to_string())
print("\n")