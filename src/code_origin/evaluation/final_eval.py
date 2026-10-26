# src/code_origin/evaluation/final_eval.py
# PYTHONPATH=src python3 -m code_origin.evaluation.final_eval binary_generate
import sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from code_origin.config import EXPERIMENTS_DIR
from code_origin.helpers import log

v = sys.argv[1]
arch = EXPERIMENTS_DIR / "archive"
ux = sorted((arch / "transformer").glob(f"*/{v}__unixcoder/test"))
lr = sorted((arch / "baseline").glob(f"*/{v}__char_LogReg/test"))[0]

ref = pd.read_parquet(lr / "predictions.parquet")
for d in ux:                                   # same files, same order everywhere
    assert (pd.read_parquet(d / "predictions.parquet").uid.values == ref.uid.values).all()
y, groups = ref.y.values, ref.problem_id.values
P = np.stack([np.load(d / "proba.npy") for d in ux])   # seeds x files x classes
pc = np.load(lr / "proba.npy")
pe = P.mean(0)                                          # seed ensemble
labels = np.arange(P.shape[2])
f1 = lambda yy, p: f1_score(yy, p.argmax(1), average="macro", labels=labels, zero_division=0)

single = [f1(y, p) for p in P]
log(f"{v}: {len(ux)} seeds", "start")
print(f"UniXcoder single seed   {np.mean(single):.3f} ± {np.std(single, ddof=1):.3f}")
print(f"UniXcoder seed ensemble {f1(y, pe):.3f}")
print(f"char LogReg             {f1(y, pc):.3f}")
print(f"mix 50/50 (ens + char)  {f1(y, .5 * pe + .5 * pc):.3f}")

# paired cluster bootstrap over problems (files of one problem stay together)
rng = np.random.default_rng(0)
uniq = np.unique(groups)
rows = {g: np.where(groups == g)[0] for g in uniq}
d_single, d_ens = [], []
for _ in range(1000):
    i = np.concatenate([rows[g] for g in rng.choice(uniq, len(uniq))])
    fc = f1(y[i], pc[i])
    d_ens.append(f1(y[i], pe[i]) - fc)
    d_single.append(np.mean([f1(y[i], p[i]) for p in P]) - fc)
ci = lambda x: np.percentile(x, [2.5, 97.5])
for name, d in (("seed ensemble - char LogReg", d_ens), ("single seed - char LogReg  ", d_single)):
    lo, hi = ci(d)
    print(f"{name}  {np.mean(d):+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")