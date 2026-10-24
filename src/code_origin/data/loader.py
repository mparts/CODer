"""
Module holding helpers for loading the downloaded datasets
"""
import numpy as np, pandas as pd
from code_origin.config import PROCESSED_DIR, RAW_DIR, SEED


def load_variant(name, clean=False, balance=False, seed=SEED, meta=False):
    """Load one task variant. clean drops label-conflict rows, balance downsamples to the smallest class per split.
    The meta_* columns reveal the label, so they are only returned with meta=True."""
    df = pd.read_parquet(PROCESSED_DIR / "variants" / f"{name}.parquet")
    if clean:
        df = df[~df.conflict]
    if balance:
        df = pd.concat([g.groupby("y").sample(g.y.value_counts().min(), random_state=seed) for _, g in df.groupby("split")])
    df = df.reset_index(drop=True)
    if not meta:
        df = df.drop(columns=[c for c in df if c.startswith("meta_")])
    return df


def load_ada(df):
    """Ada embeddings for the rows of df, in the same order."""
    ada = np.load(RAW_DIR / "ada_train.npy")
    return ada[df.uid.str.split("_").str[1].astype(int).to_numpy()]
