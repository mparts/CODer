"""Fine-tune a transformer on one variant, evaluate it on dev or test and save the results.

    PYTHONPATH=src python3 -m code_origin.training.train_codebert --variant binary --model unixcoder
    PYTHONPATH=src python3 -m code_origin.training.train_codebert --variant binary --model unixcoder --test

dev run : fine-tunes on train, keeps the epoch with the best dev macro-F1, saves the model
          to artifacts/<variant>__<model>/ and results to docs/experiments/transformer/<variant>__<model>/dev/
test run: loads the saved model (from --model_dir if given) and writes .../test/. Do this once, at the very end.
Both runs are also copied to docs/experiments/archive/transformer/<hyperparameters>_<seed>/<variant>__<model>/.
"""
import argparse, json, numpy as np, torch, pandas as pd, shutil
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging

from code_origin.config import SEED, CHECKPOINTS, DEVICE, ARGS, MODES
from code_origin.helpers import log, cache_data, archive_run
from code_origin.models.transformer import batches, predict_proba, fit
from code_origin.evaluation import plots, evaluate


def train():
    """Fine-tune a new model, or load the saved one for --eval/--test."""
    if a.test or a.eval:
        if not model_dir.exists():
            raise SystemExit(f"no saved model at {model_dir}, run the dev run first")
        log(f"Evaluating {exp} on the {split} split..", "start")
        tok = AutoTokenizer.from_pretrained(model_dir, clean_up_tokenization_spaces=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE)
    else:
        log(f"Training {exp}...", "start")
        torch.manual_seed(SEED)
        tok = AutoTokenizer.from_pretrained(CHECKPOINTS[a.model], clean_up_tokenization_spaces=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            CHECKPOINTS[a.model], num_labels=len(classes),
            hidden_dropout_prob=a.dropout, attention_probs_dropout_prob=a.dropout).to(DEVICE)
        history = fit(model, tok, df[df.split == "train"], df[df.split == "dev"], classes, a)
        model.save_pretrained(model_dir)
        tok.save_pretrained(model_dir)
        if a.save_ckpt:
            shutil.copytree(model_dir, out / "checkpoint", dirs_exist_ok=True)
        return model, tok, history
    return model, tok, None


def eval(model, tok):
    ev = df[df.split == split].reset_index(drop=True)
    proba = predict_proba(model, batches(tok, ev.code_norm, ev.y.values, 64, a.max_length, shuffle=False))
    np.save(out / "proba.npy", proba)
    pred = proba.argmax(axis=1)
    res = evaluate.evaluate(ev.y, pred, classes)

    preds = ev[["uid", "problem_id", "y", "meta_LLM", "meta_source"]].assign(pred=pred)
    if len(classes) == 2:
        preds["score"] = proba[:, 1]
    preds.to_parquet(out / "predictions.parquet", index=False)
    (out / "metrics.json").write_text(json.dumps({"experiment": exp, "split": split, "n": len(ev), "args": vars(a), **res}, indent=2))
    return res, preds


if __name__ == "__main__":
    logging.set_verbosity_error()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",        default=ARGS["transformer_models"][0], choices=ARGS["transformer_models"])
    ap.add_argument("--variant",      default=ARGS["variant"][0],            choices=ARGS["variant"])
    ap.add_argument("--epochs",       default=ARGS["epochs"],                type=int)
    ap.add_argument("--batch_size",   default=ARGS["batch_size"],            type=int)
    ap.add_argument("--lr",           default=ARGS["lr"],                    type=float)
    ap.add_argument("--weight_decay", default=ARGS["weight_decay"],          type=float)
    ap.add_argument("--warmup_ratio", default=ARGS["warmup_ratio"],          type=float)
    ap.add_argument("--max_length",   default=ARGS["max_length"],            type=int)
    ap.add_argument("--class_weight", default=ARGS["class_weight"])
    ap.add_argument("--dropout",      default=ARGS["dropout"],               type=float)
    ap.add_argument("--eval",      action="store_true", help="Evaluate the saved model on dev")
    ap.add_argument("--test",      action="store_true", help="Evaluate the saved model on test instead of dev")
    ap.add_argument("--save_ckpt", action="store_true", help="also copy the trained model into the run folder")
    ap.add_argument("--model_dir", help="With --eval/--test: load the saved model from here instead of artifacts/")
    a = ap.parse_args()
    exp, split, model_dir, out, df, classes = cache_data(a, MODES[1])
    model_dir = Path(a.model_dir) if a.model_dir else model_dir

    trained_model, tokenizer, history = train()
    results, predictions = eval(trained_model, tokenizer)
    evaluate.print_metrics(results)

    plots.confusion_matrix_plot(results["confusion_matrix"], classes, out / "confusion_matrix.png")
    plots.group_accuracy_plot(predictions, out / "accuracy_by_source.png")
    if history is not None:
        plots.training_curves_plot(history, out / "training_curves.png")
        pd.DataFrame(history["curve"]).to_csv(out / "curve.csv", index=False)
    if len(classes) == 2:
        plots.roc_plot(predictions.y, predictions.score, out / "roc.png")

    archive_run(out, a, MODES[1])
    shutil.rmtree(out / "checkpoint", ignore_errors=True) # the checkpoint copy only stays in the archive
    log("Success!!", "end")