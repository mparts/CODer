"""
Module holding general helper functions
"""
from datetime import datetime
import json

from code_origin.config import ARTIFACTS_DIR, EXPERIMENTS_DIR, PROCESSED_DIR, MODES
from code_origin.data.loader import load_variant


def log(message, position=""):
    line = "="*100
    body = f"   [{datetime.now()}] {message}"
    if position == "start":
        print(line, flush=True), print(body, flush=True), print("\n")
    elif position == "end":
        print("\n"), print(body, flush=True), print(line, flush=True)
    else:
        print(body, flush=True)


def cache_data(a, mode):
    exp = f"{a.variant}__{a.model}"
    split = "test" if a.test else "dev"
    model_dir = ARTIFACTS_DIR / exp / "model.joblib" if mode == MODES[0] else ARTIFACTS_DIR / exp
    out = EXPERIMENTS_DIR / mode / exp / split
    out.mkdir(parents=True, exist_ok=True)
    df = load_variant(a.variant, clean=True, meta=True)
    classes = json.loads((PROCESSED_DIR / "meta.json").read_text())[a.variant]["classes"]
    return exp, split, model_dir, out, df, classes

