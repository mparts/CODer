import argparse, json, pandas as pd
from pathlib import Path
from code_origin.config import EXPERIMENTS_DIR, METRIC_COLUMNS, HYPERPARAMETERS, MODES


def _get_hyperparameters(run_path: Path, mode: str) -> dict:
    """Extract hyperparameters and seed from an archived run's directory name."""
    parts = run_path.name.split("_")
    if mode == MODES[0]:
        return {"seed": parts[0]}
    if len(parts) != len(HYPERPARAMETERS) + 1:
        raise ValueError(f"Unexpected run directory name: {run_path.name}")
    params = dict(zip(HYPERPARAMETERS, parts[:-1]))
    params["seed"] = parts[-1]
    return params

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
    columns = (["experiment", "split"] + hyperparameter_columns + ["seeds", "n", "accuracy", "macro_f1"])
    table = (pd.DataFrame(rows).rename(columns={"seed": "seeds"}).reindex(columns=columns).round(3))
    if table["macro_f1"].astype(str).str.contains(" ± ").any():
        table["_sort_f1"] = table["macro_f1"].apply(lambda x: float(str(x).split(" ± ")[0]))
        table = table.sort_values("_sort_f1", ascending=False)
        table = table.drop(columns="_sort_f1")
    else:
        table = table.sort_values("macro_f1", ascending=False)
    table.to_csv(output_path, index=False)


def _aggregate_seeds(rows: list[dict], hyperparameter_columns: list[str]) -> list[dict]:
    """Aggregate repeated seeds for each experiment, split and configuration."""
    df = pd.DataFrame(rows)
    group_columns = ["experiment", "split", *[col for col in hyperparameter_columns if col != "seed"]]
    aggregated = []
    for _, group in df.groupby(group_columns, dropna=False, sort=False):
        ranked = group.sort_values("macro_f1", ascending=False)
        row = {key: ranked.iloc[0][key] for key in group_columns}
        row["seeds"] = ",".join(str(seed) for seed in ranked["seed"].tolist())
        row["n"] = int(group["n"].iloc[0])   # same eval set size across seeds
        for metric in ("accuracy", "macro_f1"):
            row[metric] = (
                f"{group[metric].mean():.3f} "
                f"± {group[metric].std(ddof=1):.3f}"
                if len(group) > 1
                else f"{group[metric].mean():.3f} ± 0.000"
            )
        aggregated.append(row)
    return aggregated


def summarize(modes: list | None = None, init_path: Path = EXPERIMENTS_DIR, archive: bool = False):
    init_path = Path(init_path)
    if archive:
        archive_path = init_path / "archive"
        all_rows, archive_hyperparameters = [], []
        for mode in MODES:
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
            aggregated_rows = _aggregate_seeds(mode_rows, varying)
            _save_summary(aggregated_rows, mode_path / "summary.csv", varying)   
            archive_hyperparameters.extend(varying)
        if not all_rows:
            print("no experiments found in archive")
            return
        archive_hyperparameters = [key for key in HYPERPARAMETERS if key in archive_hyperparameters]
        aggregated_all_rows = _aggregate_seeds(all_rows, archive_hyperparameters)
        _save_summary(aggregated_all_rows, archive_path / "summary.csv", archive_hyperparameters)
        table = pd.DataFrame(aggregated_all_rows)
        table["_macro_f1_mean"] = table["macro_f1"].apply(lambda x: float(x.split(" ± ")[0]))
        table = table.sort_values("_macro_f1_mean", ascending=False)
        table = table.drop(columns="_macro_f1_mean")
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
