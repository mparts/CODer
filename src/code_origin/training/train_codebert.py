"""Fine-tune CodeBERT on one variant. Same outputs and folders as train.py.

    PYTHONPATH=src python -m code_origin.training.train_codebert --variant binary
    PYTHONPATH=src python -m code_origin.training.train_codebert --variant binary --test

dev run : fine-tunes on train for a fixed number of epochs (no checkpoint picking), saves the model
          to artifacts/<variant>__codebert/ and results to docs/experiments/<variant>__codebert/dev/
test run: loads the saved model (no training) and writes .../test/. Do this once, at the very end.
"""
import argparse, json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from code_origin.config import SEED, CHECKPOINT, DEVICE, ARGS, MODES
from code_origin.helpers import log, cache_data
from code_origin.models.transformer import batches, predict_proba, fit
from code_origin.evaluation import plots, evaluate


def train():
    if a.test or a.eval:
        if not model_dir.exists():
            raise SystemExit(f"no saved model at {model_dir}, run the dev run first")
        log(f"Evaluating {exp} on the {split} split..", "start")
        tok = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE)
    else:
        log(f"Training {exp}...", "start")
        torch.manual_seed(SEED)
        tok = AutoTokenizer.from_pretrained(CHECKPOINT)
        model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=len(classes)).to(DEVICE)
        fit(model, tok, df[df.split == "train"], len(classes), a)
        model.save_pretrained(model_dir)
        tok.save_pretrained(model_dir)
    return model, tok

def eval(model, tok):
    ev = df[df.split == split].reset_index(drop=True)
    proba = predict_proba(model, batches(tok, ev.code_norm, ev.y.values, 64, shuffle=False))
    pred = proba.argmax(axis=1)
    res = evaluate.evaluate(ev.y, pred, classes)

    preds = ev[["uid", "problem_id", "y", "meta_LLM", "meta_source"]].assign(pred=pred)
    if len(classes) == 2:
        preds["score"] = proba[:, 1]
    preds.to_parquet(out / "predictions.parquet", index=False)
    (out / "metrics.json").write_text(json.dumps({"experiment": exp, "split": split, "n": len(ev), "args": vars(a), **res}, indent=2))
    return res, preds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",      default=ARGS["transformer_models"][0], choices=ARGS["transformer_models"])
    ap.add_argument("--variant",    default=ARGS["variant"][0],            choices=ARGS["variant"])
    ap.add_argument("--epochs",     default=1,    type=int)
    ap.add_argument("--lr",         default=2e-5, type=float)
    ap.add_argument("--batch_size", default=16,   type=int)
    ap.add_argument("--eval", action="store_true", help="Evaluate the saved model on dev")
    ap.add_argument("--test", action="store_true", help="Evaluate the saved model on test instead of dev")
    a = ap.parse_args()
    exp, split, model_dir, out, df, classes = cache_data(a, MODES[1])

    trained_model, tokenizer = train()
    results, predictions = eval(trained_model, tokenizer)
    evaluate.print_metrics(results)

    plots.confusion_matrix_plot(results["confusion_matrix"], classes, out / "confusion_matrix.png")
    plots.group_accuracy_plot(predictions, out / "accuracy_by_source.png")
    if len(classes) == 2:
        plots.roc_plot(predictions.y, predictions.score, out / "roc.png")

    log("Success!!", "end")