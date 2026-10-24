"""
Metrics for the classification tasks.
"""
import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from code_origin.config import EXPERIMENTS_DIR, METRIC_COLUMNS, HYPERPARAMETERS


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



def _get_hyperparameters(run_path: Path, mode: str) -> dict:
    """Extract hyperparameters from an archived run's directory name."""
    parts = run_path.name.split("_", 7)
    if mode == "baseline":
        return {"seed": parts[0]}
    if len(parts) != len(HYPERPARAMETERS):
        raise ValueError(f"Unexpected run directory name: {run_path.name}")
    return dict(zip(HYPERPARAMETERS, parts))

def _varying_hyperparameters(runs: list[dict]) -> list[str]:
    """Return hyperparameters with more than one distinct value across runs."""
    if not runs:
        return []
    return [key for key in HYPERPARAMETERS if len({run["params"].get(key) for run in runs}) > 1]

def _collect_rows(metrics_files, params: dict) -> list[dict]:
    """Read metrics and attach the run's hyperparameters."""
    rows = []
    for f in sorted(metrics_files):
        m = json.loads(f.read_text())
        row = {key: m[key] for key in METRIC_COLUMNS}
        row = {
            **{key: row[key] for key in ("experiment", "split")},
            **params,
            **{key: row[key] for key in ("n", "accuracy", "macro_f1")},
        }
        rows.append(row)
    return rows

def _save_summary(rows: list[dict], output_path: Path, hyperparameter_columns: list[str]):
    """Sort and save a summary with the requested columns."""
    if not rows:
        return
    columns = (["experiment", "split"] + hyperparameter_columns + ["n", "accuracy", "macro_f1"])
    table = pd.DataFrame(rows).reindex(columns=columns).round(3)
    table = table.sort_values(by="macro_f1", ascending=False)
    table.to_csv(output_path, index=False)


def summarize(modes: list | None = None, init_path: Path = EXPERIMENTS_DIR, archive: bool = False):
    init_path = Path(init_path)
    if archive:
        archive_path = init_path / "archive"
        all_rows, archive_hyperparameters = [], []
        for mode in ("baseline", "transformer"):
            mode_path = archive_path / mode
            if not mode_path.exists():
                print(f"no experiments found in {mode_path}")
                continue
            runs = [{"path": p, "params": _get_hyperparameters(p, mode)}
                for p in sorted(mode_path.iterdir()) if p.is_dir()]
            varying = _varying_hyperparameters(runs)
            mode_rows = []
            for run in runs:
                run_path, params = run["path"], run["params"]
                rows = _collect_rows(run_path.glob("*/*/metrics.json"),params)
                if not rows:
                    continue
                mode_rows.extend(rows)
                all_rows.extend(rows)
                _save_summary(rows, run_path / "summary.csv", varying)
            if not mode_rows:
                print(f"no experiments found in {mode_path}")
                continue
            _save_summary(mode_rows, mode_path / "summary.csv", varying)
            archive_hyperparameters.extend(varying)
        if not all_rows:
            print("no experiments found in archive")
            return
        archive_hyperparameters = [key for key in HYPERPARAMETERS if key in archive_hyperparameters]
        _save_summary(all_rows, archive_path / "summary.csv", archive_hyperparameters,)
        columns = (["experiment", "split"] + archive_hyperparameters + ["n", "accuracy", "macro_f1"])
        table = pd.DataFrame(all_rows).reindex(columns=columns).round(3)
        table = table.sort_values(by="macro_f1", ascending=False)
        print(table.to_string(index=False))
        return
    if modes is None:
        raise ValueError("modes must be provided when archive=False")
    all_rows = []
    for mode in modes:
        path = init_path / mode
        rows = _collect_rows(path.glob("*/*/metrics.json"), {},)
        if not rows:
            print(f"no experiments found in {path}")
            continue
        _save_summary(rows, path / "summary.csv", [])
        all_rows.extend(rows)
    if not all_rows:
        print("no experiments found")
        return
    table = pd.DataFrame(all_rows).round(3)
    table = table.sort_values(by="macro_f1", ascending=False)
    if len(modes) >= 2:
        table.to_csv(init_path / "summary.csv", index=False)
    print(table.to_string(index=False))

