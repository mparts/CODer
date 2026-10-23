"""
Module holding configs
"""
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]


# == Paths =====================================
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
EXPERIMENTS_DIR = ROOT / "docs" / "experiments"


# == Dataset & Preprocessing =================================
REPO = "basakdemirok/AIGCodeSet"
SEED = 42
LLMS, BINARY = ["GEMINI", "LLAMA", "CODESTRAL"], ["human", "ai"]

VARIANTS = { # name: (pandas query, target column, classes)
    "binary": (None, "origin", BINARY),
    "binary_generate": ("meta_source in ['human', 'generate']", "origin", BINARY),
    "multiclass": (None, "meta_LLM", ["Human"] + LLMS),
    "multiclass_generate": ("meta_source in ['human', 'generate']", "meta_LLM", ["Human"] + LLMS),
}

# == MODELS ============================================
BASE_MODELS = ["word_LogReg", "char_LogReg"]
TRANSFORMER_MODELS = ["codebert"]


# == Args!!!! ================================================
ARGS = {
    "baseline_models": BASE_MODELS ,
    "transformer_models": TRANSFORMER_MODELS,
    "variant": list(VARIANTS.keys()),
}



# == Plotting ====================================
COLOR = "#3b6ea5"


# == Transformer =======================================
CHECKPOINT = "microsoft/codebert-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# == General ================================================
MODES = ["baseline", "transformer"]