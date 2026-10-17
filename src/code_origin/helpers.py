"""
Module holding general helper functions
"""
from datetime import datetime


def log(message):
    print("\n"), print("="*100), print(f"[{datetime.now()}] {message}", flush=True)