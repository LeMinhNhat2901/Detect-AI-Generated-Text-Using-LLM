from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import seaborn as sns
except ImportError:  # pragma: no cover - fallback for minimal environments
    sns = None
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def binary_metrics(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute the main binary-classification metrics for P(AI)."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)
    y_pred = (y_prob_arr >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_arr, y_prob_arr)),
        "pr_auc": float(average_precision_score(y_true_arr, y_prob_arr)),
    }


def threshold_sweep(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    thresholds: Optional[Iterable[float]] = None,
) -> pd.DataFrame:
    """Evaluate metrics over many thresholds for threshold tuning."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.951, 0.01), 3)
    rows = [binary_metrics(y_true, y_prob, float(t)) for t in thresholds]
    return pd.DataFrame(rows)


def prediction_frame(
    texts: Iterable[str],
    y_true: Iterable[int],
    y_prob: Iterable[float],
    threshold: float = 0.5,
    model_name: str = "model",
) -> pd.DataFrame:
    """Create a compact dataframe for metrics, errors, and qualitative review."""
    df = pd.DataFrame(
        {
            "text": list(texts),
            "label": np.asarray(y_true).astype(int),
            "prob_ai": np.asarray(y_prob).astype(float),
        }
    )
    df["pred"] = (df["prob_ai"] >= threshold).astype(int)
    df["correct"] = df["pred"] == df["label"]
    df["error_type"] = "correct"
    df.loc[(df["label"] == 0) & (df["pred"] == 1), "error_type"] = "false_positive"
    df.loc[(df["label"] == 1) & (df["pred"] == 0), "error_type"] = "false_negative"
    df["model"] = model_name
    return df


def top_errors(
    pred_df: pd.DataFrame,
    n: int = 20,
) -> pd.DataFrame:
    """Return high-confidence wrong predictions for error analysis."""
    errors = pred_df.loc[~pred_df["correct"]].copy()
    if errors.empty:
        return errors
    errors["confidence"] = np.where(errors["pred"] == 1, errors["prob_ai"], 1 - errors["prob_ai"])
    cols = ["model", "error_type", "label", "pred", "prob_ai", "confidence", "text"]
    return errors.sort_values("confidence", ascending=False)[cols].head(n)


def plot_confusion_matrix(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    threshold: float = 0.5,
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
):
    """Plot a 2x2 confusion matrix with Human/AI labels."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob).astype(float) >= threshold).astype(int)
    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    if sns is not None:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Pred Human", "Pred AI"],
            yticklabels=["True Human", "True AI"],
            ax=ax,
        )
    else:
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["Pred Human", "Pred AI"])
        ax.set_yticks([0, 1], ["True Human", "True AI"])
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                ax.text(col, row, str(cm[row, col]), ha="center", va="center", color="black")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig, ax


def plot_roc_pr_curves(
    curves: Dict[str, Dict[str, Iterable[float]]],
    save_path: Optional[Path] = None,
):
    """Plot ROC and precision-recall curves for one or more models."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for name, payload in curves.items():
        y_true = np.asarray(payload["y_true"]).astype(int)
        y_prob = np.asarray(payload["y_prob"]).astype(float)

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        axes[0].plot(fpr, tpr, linewidth=2, label=f"{name} AUC={roc_auc:.4f}")

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        axes[1].plot(recall, precision, linewidth=2, label=f"{name} AP={pr_auc:.4f}")

    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")

    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig, axes
