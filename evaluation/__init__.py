"""
Evaluation harness for log anomaly detection.

Model-agnostic benchmarking: given ground-truth labels and anomaly scores,
compute honest classification metrics (Precision / Recall / F1 / PR-AUC /
ROC-AUC) with proper temporal (leakage-free) splitting.

Phase 1 of the model roadmap (see docs/MODEL_ROADMAP.md). Establishes the
baseline measurement infrastructure that every future model must beat.
"""

from evaluation.metrics import EvalResult, evaluate_scores
from evaluation.splits import random_split, temporal_split

__all__ = [
    "EvalResult",
    "evaluate_scores",
    "temporal_split",
    "random_split",
]
