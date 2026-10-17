"""Fit a baseline on the train split of one variant, evaluate it on dev (or test) and save everything.

    PYTHONPATH=src python -m code_origin.training.train --variant human_vs_ai --model tfidf_char

dev run : fits on train, saves the model to artifacts/<experiment>/
          and metrics, predictions and plots to docs/experiments/<experiment>/dev/
test run: loads that saved model and writes docs/experiments/<experiment>/test/.
"""
import argparse, json, joblib
from code_origin.helpers import log
from code_origin.config import ARTIFACTS_DIR, BASELINE_DIR, PROCESSED_DIR, BASE_MODELS
from code_origin.data.loader import load_variant
from code_origin.evaluation import plots
from code_origin.evaluation.evaluate import evaluate
from code_origin.models.baseline import make_baseline


def train_baseline():
    exp = f"{a.variant}__{a.model}"
    split = "test" if a.test else "dev"
    model_file = ARTIFACTS_DIR / exp / "model.joblib"
    out = BASELINE_DIR / exp / split
    out.mkdir(parents=True, exist_ok=True)

    df = load_variant(a.variant, clean=True, meta=True)
    classes = json.loads((PROCESSED_DIR / "meta.json").read_text())[a.variant]["classes"]

    if a.test:
        if not model_file.exists():
            raise SystemExit(f"no saved model at {model_file}, run the dev run first")
        log(f"Testing {exp} on the test split..", "start")
        clf = joblib.load(model_file)
    else:
        log(f"Training {exp}...", "start")
        train = df[df.split == "train"]
        clf = make_baseline(a.model)
        clf.fit(train.code_norm, train.y)
        model_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, model_file)

    ev = df[df.split == split].reset_index(drop=True)
    X = ev.code_norm
    pred = clf.predict(X)
    res = evaluate(ev.y, pred, classes)

    preds = ev[["uid", "problem_id", "y", "meta_LLM", "meta_source"]].assign(pred=pred)
    if len(classes) == 2:
        preds["score"] = clf.predict_proba(X)[:, 1]
    preds.to_parquet(out / "predictions.parquet", index=False)
    (out / "metrics.json").write_text(json.dumps({"experiment": exp, "split": split, "n": len(ev), "args": vars(a), **res}, indent=2))

    plots.confusion_matrix_plot(res["confusion_matrix"], classes, out / "confusion_matrix.png")
    plots.group_accuracy_plot(preds, out / "accuracy_by_source.png")
    if len(classes) == 2:
        plots.roc_plot(preds.y, preds.score, out / "roc.png")
    if a.model.startswith("tfidf"):
        plots.top_features_plot(clf, classes, out / "top_features.png")

    print(f"acc {res['accuracy']:.3f}  macro-F1 {res['macro_f1']:.3f}")
    for c, m in res["per_class"].items():
        print(f"{c:10s} Precision {m['precision']:.3f}  Recall {m['recall']:.3f}  F1 {m['f1']:.3f}  n={m['support']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="baseline")
    ap.add_argument("--variant", default="human_vs_ai")
    ap.add_argument("--model", default="tfidf_char", choices=BASE_MODELS)
    ap.add_argument("--test", action="store_true", help="evaluate the saved model on test instead of dev")
    a = ap.parse_args()
    if a.mode == "baseline":
        train_baseline()
    else:
        pass
    log("Done!!", "end")