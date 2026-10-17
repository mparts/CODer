"""
Module holding general helper functions
"""
from datetime import datetime


def log(message, position=""):
    line = "="*100
    body = f"   [{datetime.now()}] {message}"
    if position == "start":
        print(line), print(body), print("\n")
    elif position == "end":
        print("\n"), print(body), print(line)
    else:
        print(body)