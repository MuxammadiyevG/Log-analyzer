"""
Model-agnostic evaluation metrics.

The whole harness pivots on one idea: a detector is only as good as the number
it can prove on labeled data. This module turns (labels, scores) into an honest
scorecard and never touches any specific model.

Convention: `y_score` is a continuous anomaly score where **higher = more
anomalous** (this matches the project's IsolationForestDetector, which inverts
sklearn's score_samples). Labels are 1 = anomaly, 0 = normal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


@dataclass
class EvalResult:
    """A single evaluation scorecard.

    Threshold-free metrics (pr_auc, roc_auc) describe ranking quality and do
    not depend on where the decision boundary sits. The operating-point metrics
    (precision/recall/f1) describe one concrete threshold: either the model's
    own predictions or the F1-optimal threshold found by a sweep.
    """

    name: str
    n_samples: int
    n_anomalies: int

    # Threshold-free (ranking) metrics
    pr_auc: float
    roc_auc: float

    # Operating-point metrics (at `threshold`)
    threshold: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int

    # Whether the operating point came from the model's own predictions or a
    # best-F1 threshold sweep over the scores.
    threshold_source: str = "sweep"
    extra: Dict[str, float] = field(default_factory=dict)

    @property
    def anomaly_rate(self) -> float:
        return self.n_anomalies / self.n_samples if self.n_samples else 0.0

    @property
    def false_positive_rate(self) -> float:
        denom = self.fp + self.tn
        return self.fp / denom if denom else 0.0

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "n_samples": self.n_samples,
            "n_anomalies": self.n_anomalies,
            "anomaly_rate": round(self.anomaly_rate, 6),
            "pr_auc": round(self.pr_auc, 6),
            "roc_auc": round(self.roc_auc, 6),
            "threshold": float(self.threshold),
            "threshold_source": self.threshold_source,
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "false_positive_rate": round(self.false_positive_rate, 6),
            "confusion": {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn},
            **{k: round(v, 6) for k, v in self.extra.items()},
        }


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn), robust to single-class edge cases."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tp), int(fp), int(fn), int(tn)


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return precision, recall, f1


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Sweep every score threshold and return (threshold, best_f1).

    This is the fairest single-number summary for an imbalanced detector: it
    reports the F1 the model *could* reach with an oracle threshold, decoupling
    ranking quality from threshold-tuning luck.
    """
    if len(np.unique(y_true)) < 2:
        return float(np.max(y_score)) if len(y_score) else 0.0, 0.0

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns len(thresholds) = len(precisions) - 1
    f1s = np.divide(
        2 * precisions * recalls,
        precisions + recalls,
        out=np.zeros_like(precisions),
        where=(precisions + recalls) > 0,
    )
    best_idx = int(np.argmax(f1s[:-1])) if len(f1s) > 1 else 0
    best_thr = float(thresholds[best_idx]) if len(thresholds) else float(np.max(y_score))
    return best_thr, float(f1s[best_idx])


def evaluate_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    name: str = "model",
    y_pred: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
) -> EvalResult:
    """Turn labels + anomaly scores into a full scorecard.

    Args:
        y_true: Ground-truth labels (1 = anomaly, 0 = normal).
        y_score: Continuous anomaly scores, higher = more anomalous.
        name: Label for this result (model/dataset).
        y_pred: Optional binary predictions from the model itself. If given,
            the operating point reflects the model's real decisions.
        threshold: Optional explicit score threshold. Overrides `y_pred`.

    Returns:
        EvalResult scorecard.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()

    if y_true.shape != y_score.shape:
        raise ValueError(
            f"y_true {y_true.shape} and y_score {y_score.shape} must match"
        )
    if y_true.size == 0:
        raise ValueError("Cannot evaluate an empty set")

    n_samples = int(y_true.size)
    n_anomalies = int(y_true.sum())

    # Threshold-free ranking metrics (guard single-class edge case)
    if len(np.unique(y_true)) < 2:
        pr_auc = float("nan")
        roc_auc = float("nan")
    else:
        pr_auc = float(average_precision_score(y_true, y_score))
        roc_auc = float(roc_auc_score(y_true, y_score))

    # Decide the operating point
    if threshold is not None:
        thr = float(threshold)
        preds = (y_score >= thr).astype(int)
        source = "explicit"
    elif y_pred is not None:
        preds = np.asarray(y_pred).astype(int).ravel()
        thr = float("nan")
        source = "model"
    else:
        thr, _ = best_f1_threshold(y_true, y_score)
        preds = (y_score >= thr).astype(int)
        source = "sweep"

    tp, fp, fn, tn = _confusion(y_true, preds)
    precision, recall, f1 = _prf(tp, fp, fn)

    return EvalResult(
        name=name,
        n_samples=n_samples,
        n_anomalies=n_anomalies,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        threshold=thr,
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        threshold_source=source,
    )
