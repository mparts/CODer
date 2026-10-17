"""
Quick checks on the processed dataset: composition, split leakage, formatting artifacts.
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

from code_origin.config import PROCESSED_DIR
from code_origin.helpers import log


def composition(df):
    print(pd.crosstab(df.split, df.meta_LLM, margins=True), "\n")
    print(pd.crosstab(df.meta_LLM, df.meta_status), "\n")
    print("conflict rows:", int(df.conflict.sum()), "\n")


def leakage(df):
    problems = {s: set(g.problem_id) for s, g in df.groupby("split")}
    for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        print(f"shared problems {a}/{b}:", len(problems[a] & problems[b]))
    key = df.code_raw.map(lambda c: " ".join(c.split()))
    shared = df.groupby(key).split.transform("nunique") > 1
    print("rows whose code appears in more than one split:", int(shared.sum()), "\n")


def artifacts(df):
    code = df.code_raw
    feats = pd.DataFrame({
        "ends_newline": code.str.endswith("\n"),
        "starts_with_space": code.str.match(r"\s"),
        "trailing_space": code.str.contains(r"[ \t]+(?:\n|$)"),
        "has_tab": code.str.contains("\t"),
        "has_crlf": code.str.contains("\r"),
        "mean_line_len": code.map(lambda c: np.mean([len(l) for l in c.split("\n")])),
        "n_lines": code.str.count("\n") + 1,
    }).astype(float)
    table = feats.groupby(df.meta_LLM).mean().T
    auc = [roc_auc_score(df.origin == "ai", feats[c]) for c in feats]
    table["auc_human_vs_ai"] = [max(a, 1 - a) for a in auc]
    print(table.round(3))


if __name__ == "__main__":
    log("Loading processed dataset & printing stats...")
    df = pd.read_parquet(PROCESSED_DIR / "all.parquet")
    composition(df)
    leakage(df)
    artifacts(df)
