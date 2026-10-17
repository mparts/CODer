"""
Module holding configs
"""
from pathlib import Path


# == Paths =====================================
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


# == Dataset & Preprocessing =================================
REPO = "basakdemirok/AIGCodeSet"
SEED = 42
LLMS, BINARY = ["GEMINI", "LLAMA", "CODESTRAL"], ["human", "ai"]