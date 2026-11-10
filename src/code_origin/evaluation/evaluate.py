"""Metrics for the classification tasks."""
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


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

