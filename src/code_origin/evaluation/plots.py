"""Plots saved with every experiment."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from code_origin.config import LLMS, COLOR


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def confusion_matrix_plot(cm, classes, path):
    cm = np.asarray(cm)
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    size = 1.3 * len(classes) + 2.5
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(share, vmin=0, vmax=1, cmap="Blues")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{share[i, j]:.0%}\n({cm[i, j]})", ha="center", va="center",
                    color="white" if share[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix (share of each true class)")
    _save(fig, path)


def group_accuracy_plot(preds, path):
    group = np.where(preds.meta_source == "human", "Human", preds.meta_LLM + " (" + preds.meta_source + ")")
    correct = (preds.y == preds.pred).groupby(group).agg(["mean", "size"])
    order = ["Human"] + [f"{l} ({s})" for l in LLMS for s in ("generate", "edit")]
    correct = correct.loc[[g for g in order if g in correct.index][::-1]]
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(correct) + 1.5))
    ax.barh(correct.index, correct["mean"], color=COLOR)
    for i, (m, n) in enumerate(zip(correct["mean"], correct["size"])):
        ax.text(m + 0.01, i, f"{m:.0%}  (n={n})", va="center")
    ax.axvline((preds.y == preds.pred).mean(), ls="--", color="grey")
    ax.set_xlim(0, 1.2)
    ax.set_xlabel("share classified correctly")
    ax.set_title("Accuracy by true source (dashed line = overall)")
    _save(fig, path)


def roc_plot(y, score, path):
    fpr, tpr, _ = roc_curve(y, score)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, color=COLOR)
    ax.plot([0, 1], [0, 1], ls="--", color="grey")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"ROC curve, AUC = {roc_auc_score(y, score):.3f}")
    _save(fig, path)


def top_features_plot(clf, classes, path, k=15):
    # n-grams with the largest logistic regression weight for each class
    names = clf[0].get_feature_names_out()
    coef = clf[-1].coef_
    weights = {classes[1]: coef[0], classes[0]: -coef[0]} if len(classes) == 2 else dict(zip(classes, coef))
    fig, axes = plt.subplots(1, len(weights), figsize=(4.5 * len(weights), 0.32 * k + 1.5), squeeze=False)
    for ax, (cls, w) in zip(axes[0], weights.items()):
        top = np.argsort(w)[-k:]
        ax.barh([names[i].replace("$", r"\$") for i in top], w[top], color=COLOR)
        ax.set_title(f"most indicative of {cls}")
        ax.set_xlabel("weight")
    _save(fig, path)