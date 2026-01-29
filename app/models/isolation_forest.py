"""
Isolation Forest anomaly detector
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
from app.models.base import BaseAnomalyDetector
from loguru import logger
from sklearn.ensemble import IsolationForest as SklearnIsolationForest


class IsolationForestDetector(BaseAnomalyDetector):
    """Isolation Forest-based anomaly detection"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Isolation Forest detector

        Args:
            config: Configuration dictionary with IF parameters
        """
        super().__init__(config)

        # Get parameters from config
        n_estimators = config.get("n_estimators", 200)
        contamination = config.get("contamination", 0.05)
        max_samples = config.get("max_samples", 1000)
        random_state = config.get("random_state", 42)

        # Initialize sklearn Isolation Forest
        self.model = SklearnIsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,
        )

        logger.info(
            f"IsolationForest initialized with {n_estimators} estimators, "
            f"contamination={contamination}"
        )

    def train(self, X: np.ndarray) -> None:
        """
        Train Isolation Forest

        Args:
            X: Training data (n_samples, n_features)
        """
        logger.info(f"Training Isolation Forest on {X.shape[0]} samples...")

        self.model.fit(X)
        self.is_trained = True

        logger.info("Isolation Forest training complete")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies

        Args:
            X: Input data (n_samples, n_features)

        Returns:
            Tuple of (predictions, scores)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Get predictions (-1 for anomaly, 1 for normal)
        predictions = self.model.predict(X)

        # Convert to binary (1 for anomaly, 0 for normal)
        predictions = (predictions == -1).astype(int)

        # Get anomaly scores (more negative = more anomalous)
        scores = self.model.score_samples(X)

        # Invert scores so higher = more anomalous
        scores = -scores

        return predictions, scores

    def save(self, path: Path) -> None:
        """
        Save model to disk

        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_path = path / "isolation_forest.joblib"
        joblib.dump(
            {"model": self.model, "config": self.config, "is_trained": self.is_trained},
            model_path,
        )

        logger.info(f"Model saved to {model_path}")

    def load(self, path: Path) -> None:
        """
        Load model from disk

        Args:
            path: Path to load model from
        """
        path = Path(path)
        model_path = path / "isolation_forest.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        data = joblib.load(model_path)
        self.model = data["model"]
        self.config = data["config"]
        self.is_trained = data["is_trained"]

        logger.info(f"Model loaded from {model_path}")
