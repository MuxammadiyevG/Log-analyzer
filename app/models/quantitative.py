"""
Quantitative sequence detector (Phase 3, LogAnomaly's second head).

DeepLog scores *order*: is the next event grammatical? It is strong on novel or
mis-ordered events but weak on **quantitative** anomalies — a burst of many
individually-legal events (a brute-force run of `FAIL`, a flood of one API
call). LogAnomaly's insight was to add a count-based view alongside the
sequential one.

This detector represents each session as an event-count vector (plus length and
max single-event run-length) and fits an IsolationForest on normal sessions.
A `FAIL`×12 burst has a count vector far from anything normal, so it is flagged
even though every transition is "allowed". Fused with DeepLog, the sequence tier
covers both order and volume anomalies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
from loguru import logger
from sklearn.ensemble import IsolationForest


class QuantitativeDetector:
    """IsolationForest over per-session event-count vectors."""

    def __init__(self, vocab_size: int, config: Dict[str, Any] | None = None):
        config = config or {}
        self.vocab_size = int(vocab_size)
        self.model = IsolationForest(
            n_estimators=int(config.get("n_estimators", 200)),
            contamination=float(config.get("contamination", 0.05)),
            random_state=int(config.get("random_state", 42)),
            n_jobs=-1,
        )
        self.is_trained = False
        self.model_type = "quantitative"

    def _features(self, sequences: List[Sequence[int]]) -> np.ndarray:
        rows = []
        for s in sequences:
            counts = np.bincount(np.asarray(s, dtype=int), minlength=self.vocab_size)[
                : self.vocab_size
            ].astype(float)
            # Max run-length of a single repeated event (bursts stand out).
            max_run = 1
            run = 1
            for a, b in zip(s[:-1], s[1:]):
                run = run + 1 if a == b else 1
                max_run = max(max_run, run)
            rows.append(np.concatenate([counts, [len(s), max_run]]))
        return np.vstack(rows) if rows else np.empty((0, self.vocab_size + 2))

    def fit(self, sequences: List[Sequence[int]]) -> "QuantitativeDetector":
        self.model.fit(self._features(sequences))
        self.is_trained = True
        logger.info(f"QuantitativeDetector trained on {len(sequences)} sessions")
        return self

    train = fit

    def score(self, sequences: List[Sequence[int]]) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        X = self._features(sequences)
        preds = (self.model.predict(X) == -1).astype(int)
        scores = -self.model.score_samples(X)
        return scores, preds

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "vocab_size": self.vocab_size, "is_trained": self.is_trained},
            path / "quantitative.joblib",
        )

    def load(self, path: Path) -> None:
        d = joblib.load(Path(path) / "quantitative.joblib")
        self.model = d["model"]
        self.vocab_size = d["vocab_size"]
        self.is_trained = d["is_trained"]
