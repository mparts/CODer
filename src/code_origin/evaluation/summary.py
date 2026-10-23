import argparse
from code_origin.evaluation.evaluate import summarize
from code_origin.config import MODES

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true", help="Summarize only the baseline models")
    ap.add_argument("--transformer", action="store_true", help="Summarize only the transformer models")
    a = ap.parse_args()

    if a.baseline:
        summarize([MODES[0]])
    elif a.transformer:
        summarize([MODES[1]])
    else:
        summarize(MODES)