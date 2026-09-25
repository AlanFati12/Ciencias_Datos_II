"""Metrics, statistical tests and evaluation figures.

Every metric below is computed from the confusion matrix of the decisions
y_hat = 1[p >= 0.5] against the true labels y:
    TP = true positives, TN = true negatives, FP = false alarms, FN = missed cases.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from . import config
from .plotting import CLASS_COLORS, CRITICAL, ENSEMBLE_COLOR, plt, sns


def classification_metrics(y, p, threshold: float = 0.5) -> dict:
    """All the metrics reported for a model, with the equation behind each one.

    accuracy           = (TP + TN) / N              share of correct answers (the rubric metric)
    balanced_accuracy  = (sensitivity + specificity) / 2
                         accuracy that is not inflated when one class is rare
    precision (PPV)    = TP / (TP + FP)             when the system says "positive", how often it is right
    recall/sensitivity = TP / (TP + FN)             share of real positives the system finds
    specificity        = TN / (TN + FP)             share of real negatives it correctly clears
    F1                 = 2 * precision * recall / (precision + recall)   harmonic mean of both
    ROC-AUC            = P(p(random positive) > p(random negative))
                         ranking quality over ALL thresholds (0.5 = chance, 1 = perfect)
    PR-AUC (AP)        = sum_n (R_n - R_{n-1}) P_n  area under the precision-recall curve
    MCC                = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
                         correlation between prediction and truth (-1..1), robust to imbalance
    Brier score        = (1/N) sum_i (p_i - y_i)^2  mean squared error of the probabilities
                         (0 = perfect; it rewards honest, calibrated probabilities)
    """
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, yhat),
        "balanced_accuracy": balanced_accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall_sensitivity": recall_score(y, yhat),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y, yhat),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "mcc": matthews_corrcoef(y, yhat),
        "brier": brier_score_loss(y, p),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def bootstrap_ci(y, p, metric=accuracy_score, n: int = 1000, alpha: float = 0.05,
                 seed: int = config.SEED) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of a metric on the test set.

    Procedure: draw n = 1000 resamples of the test set with replacement,
    compute the metric on each one, and keep the 2.5 % and 97.5 % quantiles.
    What it answers: "how much could the accuracy change with another test set
    of the same size?" A narrow interval means the result is stable.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    yhat = (np.asarray(p) >= 0.5).astype(int)
    scores = [metric(y[i], yhat[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return float(np.quantile(scores, alpha / 2)), float(np.quantile(scores, 1 - alpha / 2))


def mcnemar_test(y, pred_a, pred_b) -> dict:
    """Exact McNemar test: do two classifiers make significantly different errors?

    Only the discordant cases matter: b = A right & B wrong, c = A wrong & B right.
    Under H0 (both are equally good) b ~ Binomial(b + c, 0.5), so the p-value is
    the two-sided binomial tail. p < 0.05 -> the difference is not due to chance.
    Used to check whether the stacking ensemble really beats its best base model.
    """
    y, a, b = map(np.asarray, (y, pred_a, pred_b))
    a_right_b_wrong = int(((a == y) & (b != y)).sum())
    a_wrong_b_right = int(((a != y) & (b == y)).sum())
    n = a_right_b_wrong + a_wrong_b_right
    p = binomtest(a_right_b_wrong, n, 0.5).pvalue if n else 1.0
    return {"a_right_b_wrong": a_right_b_wrong, "a_wrong_b_right": a_wrong_b_right, "p_value": float(p)}


# --------------------------------------------------------------------- figures
def plot_metric_comparison(table: pd.DataFrame, metrics=("accuracy", "precision", "recall_sensitivity",
                                                         "specificity", "f1", "roc_auc"), target: float | None = 0.95,
                           colors: dict[str, str] | None = None):
    """Figure - grouped bars: every metric for every model on the held-out test set.

    How to read it: each group is one metric, each colour one model; taller is
    better. The dashed red line is the 95 % rubric target. Compare the black
    bar (stacking) with the coloured ones (the four models alone).
    """
    data = table[list(metrics)].reset_index(names="model").melt("model", var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    palette = {m: (colors or {}).get(m, cycle[i % len(cycle)]) for i, m in enumerate(table.index)}
    sns.barplot(data=data, x="metric", y="value", hue="model", palette=palette, ax=ax)
    lo = max(0.0, data["value"].min() - 0.08)
    ax.set_ylim(lo, 1.0)
    if target:
        ax.axhline(target, ls="--", c=CRITICAL, lw=1)
        ax.text(len(metrics) - 0.5, target + 0.004, f"rubric target {target:.0%}", color=CRITICAL,
                ha="right", va="bottom", fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("Score (held-out test set)")
    ax.set_title("Base models vs. combined system")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    return fig


def plot_confusion_matrices(cms: dict[str, np.ndarray], labels=("Negative", "Positive")):
    """Figure - one 2x2 confusion matrix per model, normalised by row.

    How to read it: rows are the true class, columns the prediction. The
    diagonal (top-left, bottom-right) holds the correct answers; the bottom-left
    cell is the dangerous one (sick cases the model missed = 1 - sensitivity).
    Each cell shows the count and the percentage of its row.
    """
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.3))
    for ax, (name, cm) in zip(np.atleast_1d(axes), cms.items()):
        pct = cm / cm.sum(axis=1, keepdims=True)
        annot = np.array([[f"{cm[i, j]:,}\n{pct[i, j]:.1%}" for j in range(2)] for i in range(2)])
        sns.heatmap(pct, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=ax,
                    xticklabels=labels, yticklabels=labels)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle("Confusion matrices (row-normalised)", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_roc_pr(y, probs: dict[str, np.ndarray], colors: dict[str, str]):
    """Figure - ROC curve (left) and precision-recall curve (right).

    ROC: for every threshold t, plots TPR(t) = TP/(TP+FN) against
    FPR(t) = FP/(FP+TN). A curve hugging the top-left corner is better; the
    dotted diagonal is a coin toss. The legend gives the area under the curve.
    PR: precision against recall for every threshold; top-right is better.
    """
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        lw = 2.6 if name == "Stacking ensemble" else 1.4
        c = colors.get(name, ENSEMBLE_COLOR)
        a1.plot(fpr, tpr, lw=lw, c=c, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
        a2.plot(rec, prec, lw=lw, c=c, label=f"{name} (AP {average_precision_score(y, p):.3f})")
    a1.plot([0, 1], [0, 1], ls=":", c="grey")
    a1.set(xlabel="False positive rate (1 - specificity)", ylabel="True positive rate (sensitivity)",
           title="ROC curves")
    a2.set(xlabel="Recall (sensitivity)", ylabel="Precision", title="Precision-recall curves")
    a1.legend(fontsize=8, loc="lower right")
    a2.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig


def plot_calibration(y, probs: dict[str, np.ndarray], colors: dict[str, str]):
    """Figure - reliability (calibration) curves.

    The test cases are grouped into 10 bins by predicted probability; each point
    is (mean predicted probability, observed share of positives) of one bin.
    How to read it: on the dotted diagonal, "70 %" really means 7 of 10 such
    cases are positive. Points below the diagonal = the model over-estimates.
    """
    fig, ax = plt.subplots(figsize=(5.6, 5))
    for name, p in probs.items():
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", ms=3, lw=2.4 if name == "Stacking ensemble" else 1.2,
                c=colors.get(name, ENSEMBLE_COLOR), label=name)
    ax.plot([0, 1], [0, 1], ls=":", c="grey", label="Perfect calibration")
    ax.set(xlabel="Predicted probability", ylabel="Observed fraction of positives",
           title="Calibration (reliability) curves")
    ax.legend(fontsize=8)
    return fig


def plot_meta_weights(weights: dict, colors: dict[str, str]):
    """Figure - coefficients beta_k of the meta-model z = beta_0 + sum_k beta_k p_k.

    How to read it: a taller bar means the final answer listens more to that
    model; 0 means the meta-model ignores it because the others already carry
    its information.
    """
    names = [k for k in weights if k != "intercept"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(names, [weights[n] for n in names], color=[colors[n] for n in names])
    for i, n in enumerate(names):
        ax.text(i, weights[n], f"{weights[n]:.2f}", ha="center", va="bottom" if weights[n] >= 0 else "top")
    ax.axhline(0, c="black", lw=0.8)
    ax.margins(y=0.18)
    ax.set(ylabel="Meta-model coefficient $\\beta_k$",
           title=f"How much the system trusts each model (intercept $\\beta_0$ = {weights['intercept']:.2f})")
    return fig


def plot_model_diversity(oof: pd.DataFrame, names: list[str]):
    """Figure - correlation of the base models' OOF probabilities and their pairwise disagreement.

    Left: Pearson r between the probabilities of two models (1 = identical).
    Right: share of cases where two models give different yes/no answers.
    How to read it: an ensemble only helps if its members disagree sometimes;
    lower correlation / higher disagreement = more complementary models.
    """
    corr = oof[names].corr()
    votes = (oof[names] >= 0.5).astype(int)
    disagree = pd.DataFrame([[float((votes[a] != votes[b]).mean()) for b in names] for a in names],
                            index=names, columns=names)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", vmin=corr.values.min(), vmax=1, ax=a1, square=True)
    a1.set_title("Correlation of OOF probabilities")
    sns.heatmap(disagree, annot=True, fmt=".1%", cmap="Oranges", ax=a2, square=True)
    a2.set_title("Pairwise disagreement rate")
    fig.tight_layout()
    return fig


def plot_probability_distributions(y, p, title="Final probability by true class"):
    """Figure - histogram of the final probability, one colour per true class.

    How to read it: the two humps should sit at opposite ends (negatives near 0,
    positives near 1). Cases inside the grey band are referred to a clinician;
    overlap between the colours is where the system makes its mistakes.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for cls, label in ((0, "Negative"), (1, "Positive")):
        sns.histplot(np.asarray(p)[np.asarray(y) == cls], bins=40, stat="density", element="step",
                     color=CLASS_COLORS[cls], label=f"Actual {label}", ax=ax, alpha=0.35)
    ax.axvline(0.5, c="black", ls="--", lw=1)
    lo, hi = config.REFERRAL_BAND
    ax.axvspan(lo, hi, color="grey", alpha=0.15, label="Referral band")
    ax.set(xlabel="Predicted probability of the positive class", ylabel="Density", title=title)
    ax.legend()
    return fig


def coverage_curve(y, p) -> pd.DataFrame:
    """Accuracy when the system only answers the cases it is most sure about.

    Confidence of a case: c_i = |p_i - 0.5|. Cases are sorted from most to least
    confident; for the top n of them, coverage = n / N and accuracy = correct / n.
    """
    y, p = np.asarray(y), np.asarray(p)
    confidence = np.abs(p - 0.5)
    order = np.argsort(-confidence)
    correct = ((p >= 0.5).astype(int) == y)[order]
    n = np.arange(1, len(y) + 1)
    acc = np.cumsum(correct) / n
    idx = np.unique(np.linspace(len(y) // 50, len(y) - 1, 60).astype(int))
    return pd.DataFrame({"coverage": n[idx] / len(y), "accuracy": acc[idx]})


def plot_coverage(curves: dict[str, pd.DataFrame], colors: dict[str, str], target: float = 0.95):
    """Figure - accuracy vs. coverage (selective prediction).

    How to read it: moving left, the system answers fewer cases (only the ones
    it is surest about) and accuracy rises. Where a curve crosses the dashed
    95 % line tells which share of cases can be answered automatically at that
    accuracy; the rest would go to a clinician.
    """
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, c in curves.items():
        ax.plot(c["coverage"], c["accuracy"], lw=2.4 if name == "Stacking ensemble" else 1.2,
                c=colors.get(name, ENSEMBLE_COLOR), label=name)
    ax.axhline(target, ls="--", c=CRITICAL, lw=1, label=f"{target:.0%} target")
    ax.set(xlabel="Coverage (fraction of cases the system answers automatically)",
           ylabel="Accuracy on answered cases", title="Accuracy vs. coverage (selective prediction)")
    ax.legend(fontsize=8)
    return fig
