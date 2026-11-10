"""Project-wide settings: paths, dataset variants, models and hyperparameters."""
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
SEED = int(os.environ.get("SEED", 0)) # set per transformer run, data splits and baselines use the default 0
MODES = ["baseline", "transformer"]
ARGS = { # choices and defaults of the training CLIs, hyperparameters as in the final UniXcoder runs
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
    "dropout": 0.0,
}

# == Summary ====================================
METRIC_COLUMNS = ["experiment", "split", "n", "accuracy", "macro_f1"]
HYPERPARAMETERS = list(ARGS.keys())[3:] # keys from "epochs" on, archive run names join their values in this order

# == Plotting ====================================
COLOR = "#3b6ea5"

# == Baseline =======================================
REG_STRENGTH = 10 # LogReg C, the inverse of the regularization strength
MAX_ITER = 2000
LR_CLASS_WEIGHTS = "balanced"

# == Transformer =======================================
CHECKPOINTS = {"codebert": "microsoft/codebert-base",
               "unixcoder": "microsoft/unixcoder-base",
               "graphcodebert": "microsoft/graphcodebert-base",}
EVALS_PER_EPOCH = 5 # for the learning curves only, checkpoints are picked at epoch ends
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


