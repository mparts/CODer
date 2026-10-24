"""
Module holding configs
"""
from pathlib import Path
import torch, os
ROOT = Path(__file__).resolve().parents[2]


# == Paths =====================================
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
EXPERIMENTS_DIR = ROOT / "docs" / "experiments"


# == Dataset & Preprocessing =================================
REPO = "basakdemirok/AIGCodeSet"
LLMS, BINARY = ["GEMINI", "LLAMA", "CODESTRAL"], ["human", "ai"]

VARIANTS = { # name: (pandas query, target column, classes)
    "binary": (None, "origin", BINARY),
    "binary_generate": ("meta_source in ['human', 'generate']", "origin", BINARY),
    "multiclass": (None, "meta_LLM", ["Human"] + LLMS),
    "multiclass_generate": ("meta_source in ['human', 'generate']", "meta_LLM", ["Human"] + LLMS),
}


# == MODELS ============================================
BASE_MODELS = ["word_LogReg", "char_LogReg"]
TRANSFORMER_MODELS = ["codebert", "unixcoder", "graphcodebert"]

# == General ================================================
SEED = int(os.environ.get("SEED", 0))
MODES = ["baseline", "transformer"]
ARGS = {
    "baseline_models": BASE_MODELS ,
    "transformer_models": TRANSFORMER_MODELS,
    "variant": list(VARIANTS.keys()),
    "epochs": 4,
    "max_length": 256,
    "lr": 2e-5,
    "batch_size": 16,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "class_weight": "none",
    "dropout": 0.1,
}
# summary
METRIC_COLUMNS = ["experiment", "split", "n", "accuracy", "macro_f1"]
# HYPERPARAMETERS = [
#     "seed",
#     "epochs",
#     "max_length",
#     "lr",
#     "batch_size",
#     "weight_decay",
#     "warmup_ratio",
#     "class_weight",
# ]
HYPERPARAMETERS = ["seed"] + list(ARGS.keys())[3:]

# == Plotting ====================================
COLOR = "#3b6ea5"


# == Transformer =======================================
CHECKPOINTS = {"codebert": "microsoft/codebert-base",
               "unixcoder": "microsoft/unixcoder-base",
               "graphcodebert": "microsoft/graphcodebert-base",}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST = ["seed"] + list(ARGS.keys())[3:]
