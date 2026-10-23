"""
Metrics for the classification tasks.
"""
import json
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from code_origin.config import EXPERIMENTS_DIR


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


def print_metrics(res):
    print(f"acc {res['accuracy']:.3f}  macro-F1 {res['macro_f1']:.3f}")
    for c, m in res["per_class"].items():
        print(f"  {c:10s} Precision {m['precision']:.3f}  Recall {m['recall']:.3f}  F1 {m['f1']:.3f}  n={m['support']}")


def summarize(modes: list):
    all_rows = []
    for mode in modes:
        path = EXPERIMENTS_DIR / mode
        rows = []
        for f in sorted(path.glob("*/*/metrics.json")):
            m = json.loads(f.read_text())
            row = {
                "experiment": m["experiment"],
                "split": m["split"],
                "n": m["n"],
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
            }
            rows.append(row)
        if not rows:
            print(f"no experiments found in {path}")
            continue
        table = pd.DataFrame(rows).round(3)
        table = table.sort_values(by="macro_f1", ascending=False)
        table.to_csv(path / "summary.csv", index=False)
        all_rows.extend(rows)
    if not all_rows:
        print("no experiments found")
        return
    table = pd.DataFrame(all_rows).round(3)
    table = table.sort_values(by="macro_f1", ascending=False)
    if len(modes) >= 2:
        table.to_csv(EXPERIMENTS_DIR / "summary.csv", index=False)
    print(table.to_string(index=False))
