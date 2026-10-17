"""Raw AIGCodeSet -> cleaned dataset with problem-grouped splits and task variants in data/processed.

Columns of the output files:
    code_raw, code_norm        model input
    y                          target (class names are in meta.json)
    split, fold, problem_id    train/dev/test, fold 0-9, group id
    conflict                   same code appears with both labels
    meta_*                     info that leaks the label (status, source, parent, LLM): analysis only
"""
import hashlib, json, re

import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from code_origin.config import PROCESSED_DIR, RAW_DIR, SEED, VARIANTS
from code_origin.helpers import log



def normalise(code):
    code = code.replace("\r\n", "\n").replace("\r", "\n").expandtabs(4)
    return "\n".join(line.rstrip() for line in code.strip().split("\n")) + "\n"


def code_hash(code):
    return hashlib.md5(re.sub(r"\s+", " ", code).strip().encode()).hexdigest()


def add_split(df):
    # 10 folds grouped by problem, stratified by source. folds 0-1 test, 2 dev, rest train
    fold = np.zeros(len(df), dtype=int)
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    for i, (_, idx) in enumerate(sgkf.split(df, df.LLM, df.problem_id)):
        fold[idx] = i
    df["fold"] = fold
    df["split"] = np.select([fold < 2, fold == 2], ["test", "dev"], "train")


def preprocess():
    df = pd.read_parquet(RAW_DIR / "train.parquet")
    add_split(df)

    df["code_hash"] = df.code.map(code_hash)
    df["conflict"] = df.groupby("code_hash").label.transform("nunique") > 1
    n = len(df)
    df = df.drop_duplicates(["code_hash", "label"]).reset_index(drop=True)
    print(f"{n} rows -> {len(df)} after dropping duplicate code, {int(df.conflict.sum())} conflict rows")
    # nothing may cross splits: neither problems nor identical code
    assert df.groupby("problem_id").split.nunique().max() == 1
    assert df.groupby("code_hash").split.nunique().max() == 1

    df["origin"] = df.label.map({0: "human", 1: "ai"})
    df["meta_source"] = np.where(df.label == 0, "human", np.where(df.submission_id == "unrelated", "generate", "edit"))
    df["meta_parent_id"] = df.submission_id.where(df.meta_source == "edit")
    df = df.rename(columns={"code": "code_raw", "LLM": "meta_LLM", "status_in_folder": "meta_status"})
    df["code_norm"] = df.code_raw.map(normalise)

    keep = ["uid", "problem_id", "code_raw", "code_norm", "split", "fold", "conflict",
            "meta_LLM", "meta_status", "meta_source", "meta_parent_id"]
    (PROCESSED_DIR / "variants").mkdir(parents=True, exist_ok=True)
    df[keep + ["origin"]].to_parquet(PROCESSED_DIR / "all.parquet", index=False)

    meta = {}
    for name, (query, target, classes) in VARIANTS.items():
        sub = (df.query(query) if query else df).copy()
        sub["y"] = sub[target].map({c: i for i, c in enumerate(classes)}).astype(int)
        sub[keep + ["y"]].to_parquet(PROCESSED_DIR / "variants" / f"{name}.parquet", index=False)
        counts = {s: sub[sub.split == s].y.value_counts().sort_index().tolist() for s in ("train", "dev", "test")}
        meta[name] = {"classes": classes, "counts": counts}
        print(f"{name:28s} {counts}")
    (PROCESSED_DIR / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    log("Preprocessing the dataset...", "start")
    preprocess()
    log("Dataset processed!!", "end")