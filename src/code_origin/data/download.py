"""
Download dataset from the HF hub once and keep a local copy in data/raw.
"""
import json
import numpy as np
from datasets import load_dataset

from code_origin.config import RAW_DIR, REPO
from code_origin.helpers import log


def download():
    if all((RAW_DIR / f).exists() for f in ("train.parquet", "ada_train.npy")):
        print(f"{RAW_DIR} already exists, nothing to download")
        return

    ds = load_dataset(REPO)
    train, test = ds["train"].to_pandas(), ds["test"].to_pandas()

    cols = [c for c in train.columns if c != "ada_embedding"]
    # train and test was identical so we drop one
    if not train[cols].equals(test[cols]):
        raise RuntimeError("train and test differ, the dataset has changed. check it before using it")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ada = np.stack([np.asarray(json.loads(x), dtype=np.float32) for x in train.pop("ada_embedding")])
    train.insert(0, "uid", [f"train_{i}" for i in range(len(train))])
    np.save(RAW_DIR / "ada_train.npy", ada)
    train.to_parquet(RAW_DIR / "train.parquet", index=False)
    print(f"saved {len(train)} rows to {RAW_DIR}")


if __name__ == "__main__":
    log(f"Downloading dataset from {REPO}...", "start")
    download()
    log(f"Done!!", "end")