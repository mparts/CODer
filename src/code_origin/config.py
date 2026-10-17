"""
Module holding configs
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]


# == Paths =====================================
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
EXPERIMENTS_DIR = ROOT / "docs" / "experiments"
BASELINE_DIR = EXPERIMENTS_DIR / "baseline"


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


# == Baseline ============================================
BASE_MODELS = ["tfidf_word", "tfidf_char"]


# == Plotting ====================================
COLOR = "#3b6ea5"