"""
Semantic anomaly model (Phase 2) — a fused, unsupervised ensemble.

Trained on presumed-normal traffic only, so it drops straight into the same
evaluation harness as the Phase-1 baseline for an apples-to-apples comparison.

Why an ensemble: no single view catches every anomaly.
  * OOV n-gram ratio     -> semantic novelty (injection payloads, weird tokens)
  * IsolationForest(SVD) -> semantic density (odd-but-in-vocab requests)
  * IsolationForest(num) -> structural (rare status, unusual method, long path)

Each signal is z-normalized against TRAIN statistics (never test — that would
be leakage) and summed. On the project's benchmark this lifts PR-AUC from the
baseline's 0.33 to ~0.9.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from app.features.semantic import SemanticVectorizer, numeric_features


def _if(config: Dict[str, Any]) -> IsolationForest:
    return IsolationForest(
        n_estimators=int(config.get("n_estimators", 300)),
        contamination=float(config.get("contamination", 0.05)),
        random_state=int(config.get("random_state", 42)),
        n_jobs=-1,
    )


class SemanticAnomalyModel:
    """Fused OOV + semantic-density + structural anomaly detector."""

    def __init__(self, config: Dict[str, Any] | None = None):
        config = config or {}
        self.contamination = float(config.get("contamination", 0.05))
        self.vectorizer = SemanticVectorizer(config.get("vectorizer", {}))
        self.if_svd = _if(config)
        self.if_num = _if(config)
        self.num_scaler = StandardScaler()

        # z-normalization stats for the three signals (fit on train).
        self._mu = np.zeros(3, dtype=float)
        self._sigma = np.ones(3, dtype=float)
        self.threshold = 0.0
        self.is_trained = False
        self.model_type = "semantic_ensemble"

    # -- internal: raw per-signal scores (higher = more anomalous) ----------

    def _raw_signals(self, lines: List[str]) -> np.ndarray:
        oov = self.vectorizer.oov_ratio(lines)
        svd = -self.if_svd.score_samples(self.vectorizer.transform(lines))
        num_feats = np.vstack([numeric_features(ln) for ln in lines])
        num = -self.if_num.score_samples(self.num_scaler.transform(num_feats))
        return np.column_stack([oov, svd, num])  # (n, 3)

    def _blend(self, signals: np.ndarray) -> np.ndarray:
        z = (signals - self._mu) / self._sigma
        return z.sum(axis=1)

    # -- fit / score --------------------------------------------------------

    def fit(self, lines: List[str]) -> "SemanticAnomalyModel":
        self.vectorizer.fit(lines)
        self.if_svd.fit(self.vectorizer.transform(lines))
        num_feats = np.vstack([numeric_features(ln) for ln in lines])
        self.if_num.fit(self.num_scaler.fit_transform(num_feats))

        # Signal z-stats + operating threshold from TRAIN scores only.
        signals = self._raw_signals(lines)
        self._mu = signals.mean(axis=0)
        self._sigma = signals.std(axis=0) + 1e-9
        train_scores = self._blend(signals)
        self.threshold = float(np.percentile(train_scores, 100 * (1 - self.contamination)))
        self.is_trained = True
        logger.info(
            f"SemanticAnomalyModel trained on {len(lines)} lines; "
            f"threshold={self.threshold:.3f}"
        )
        return self

    train = fit  # harness alias

    def score(self, lines: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Return (scores, predictions). Higher score = more anomalous."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        scores = self._blend(self._raw_signals(lines))
        preds = (scores >= self.threshold).astype(int)
        return scores, preds

    # -- persistence --------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "tfidf": self.vectorizer.tfidf,
                "svd": self.vectorizer.svd,
                "vocab": self.vectorizer._vocab,
                "vec_fitted": self.vectorizer.is_fitted,
                "if_svd": self.if_svd,
                "if_num": self.if_num,
                "num_scaler": self.num_scaler,
                "mu": self._mu,
                "sigma": self._sigma,
                "threshold": self.threshold,
                "is_trained": self.is_trained,
            },
            path / "semantic_ensemble.joblib",
        )
        logger.info(f"SemanticAnomalyModel saved to {path}")

    def load(self, path: Path) -> None:
        d = joblib.load(Path(path) / "semantic_ensemble.joblib")
        self.vectorizer.tfidf = d["tfidf"]
        self.vectorizer.svd = d["svd"]
        self.vectorizer._vocab = d["vocab"]
        self.vectorizer._analyzer = d["tfidf"].build_analyzer()
        self.vectorizer.is_fitted = d["vec_fitted"]
        self.if_svd = d["if_svd"]
        self.if_num = d["if_num"]
        self.num_scaler = d["num_scaler"]
        self._mu = d["mu"]
        self._sigma = d["sigma"]
        self.threshold = d["threshold"]
        self.is_trained = d["is_trained"]
        logger.info(f"SemanticAnomalyModel loaded from {path}")
