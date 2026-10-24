"""
Module holding general helper functions
"""
from datetime import datetime
import json, shutil

from code_origin.config import ARTIFACTS_DIR, EXPERIMENTS_DIR, PROCESSED_DIR, MODES, SEED
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


def archive_run(source, dest, a, mode):
    exp = f"{a.variant}__{a.model}"
    split = "test" if a.test else "dev"
    run_name = "_".join([SEED, a.epochs, a.max_length, 
                         a.lr, a.batch_size, a.weight_decay, 
                         a.warmup_ratio, a.class_weight]) if mode == MODES[1] else SEED
    out = EXPERIMENTS_DIR / "archive" / mode / run_name / exp / split
    print(out)
    # out.mkdir(parents=True, exist_ok=True)
    # shutil.copytree(source, dest, dirs_exist_ok=True)