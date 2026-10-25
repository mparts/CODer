"""Fit a baseline on the train split of one variant, evaluate it on dev (or test) and save everything.

    PYTHONPATH=src python -m code_origin.training.train --variant human_vs_ai --model tfidf_char

dev run : fits on train, saves the model to artifacts/<experiment>/
          and metrics, predictions and plots to docs/experiments/<experiment>/dev/
test run: loads that saved model and writes docs/experiments/<experiment>/test/.
"""
import argparse, json, joblib, shutil, numpy as np

from code_origin.config import ARGS, MODES
from code_origin.helpers import log, cache_data, archive_run
from code_origin.evaluation import plots, evaluate
from code_origin.models.baseline import make_baseline


def train():
    if a.test or a.eval:
        if not model_dir.exists():
            raise SystemExit(f"no saved model at {model_dir}, run the dev run first")
        log(f"Evaluating {exp} on the {split} split..", "start")
        clf = joblib.load(model_dir)
    else:
        log(f"Training {exp}...", "start")
        train = df[df.split == "train"]
        clf = make_baseline(a.model)
        clf.fit(train.code_norm, train.y)
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, model_dir)
        if a.save_ckpt:
            (out / "checkpoint").mkdir(exist_ok=True)
            shutil.copy(model_dir, out / "checkpoint" / "model.joblib")
    return clf


def eval(clf):
    ev = df[df.split == split].reset_index(drop=True)
    X = ev.code_norm
    proba = clf.predict_proba(X)
    np.save(out / "proba.npy", proba)
    pred = clf.classes_[proba.argmax(axis=1)]
    res = evaluate.evaluate(ev.y, pred, classes)

    preds = ev[["uid", "problem_id", "y", "meta_LLM", "meta_source"]].assign(pred=pred)
    if len(classes) == 2:
        preds["score"] = proba[:, 1]
    preds.to_parquet(out / "predictions.parquet", index=False)
    (out / "metrics.json").write_text(json.dumps({"experiment": exp, "split": split, "n": len(ev), "args": vars(a), **res}, indent=2))
    return res, preds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   default=ARGS["baseline_models"][0], choices=ARGS["baseline_models"])
    ap.add_argument("--variant", default=ARGS["variant"][0],         choices=ARGS["variant"])
    ap.add_argument("--eval",      action="store_true", help="Evaluate the saved model on dev")
    ap.add_argument("--test",      action="store_true", help="Evaluate the saved model on test instead of dev")
    ap.add_argument("--save_ckpt", action="store_true", help="also copy the trained model into the run folder")

    a = ap.parse_args()
    exp, split, model_dir, out, df, classes = cache_data(a, MODES[0])

    classify = train()
    results, predictions = eval(classify)
    evaluate.print_metrics(results)

    plots.confusion_matrix_plot(results["confusion_matrix"], classes, out / "confusion_matrix.png")
    plots.group_accuracy_plot(predictions, out / "accuracy_by_source.png")
    plots.top_features_plot(classify, classes, out / "top_features.png")
    if len(classes) == 2:
        plots.roc_plot(predictions.y, predictions.score, out / "roc.png")

    archive_run(out, a, MODES[0])
    shutil.rmtree(out / "checkpoint", ignore_errors=True)
    log("Success!!", "end")