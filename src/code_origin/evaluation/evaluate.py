"""
Metrics for the classification tasks.
"""
import json
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from code_origin.config import BASELINE_DIR


def evaluate(y_true, y_pred, classes):
    labels = list(range(len(classes)))
    p, r, f, n = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f.mean()),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(n[i])}
                      for i, c in enumerate(classes)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    return result


def summarize_baseline():
    rows = []
    for f in sorted(BASELINE_DIR.glob("*/*/metrics.json")):
        m = json.loads(f.read_text())
        rows.append({"experiment": m["experiment"], "split": m["split"], "n": m["n"], "accuracy": m["accuracy"],
                     "macro_f1": m["macro_f1"]})
    if not rows:
        print(f"no experiments found in {BASELINE_DIR}")
        return
    table = pd.DataFrame(rows).round(3)
    table.to_csv(BASELINE_DIR / "summary.csv", index=False)
    print(table.to_string(index=False))