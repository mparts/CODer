import argparse

from code_origin.evaluation.evaluate import summarize
from code_origin.config import MODES


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--baseline", action="store_true", help="Summarize only the baseline models",)
    group.add_argument("--transformer", action="store_true", help="Summarize only the transformer models",)
    group.add_argument("--archive", action="store_true", help="Summarize all archived experiments",)
    a = ap.parse_args()

    if a.archive:
        summarize(archive=True)
    elif a.baseline:
        summarize([MODES[0]])
    elif a.transformer:
        summarize([MODES[1]])
    else:
        summarize(MODES)
