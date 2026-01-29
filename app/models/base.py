"""
Base class for anomaly detection models
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


class BaseAnomalyDetector(ABC):
    """Abstract base class for anomaly detection models"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize detector

        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.is_trained = False

    @abstractmethod
    def train(self, X: np.ndarray) -> None:
        """
        Train the model

        Args:
            X: Training data (n_samples, n_features)
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies

        Args:
            X: Input data (n_samples, n_features)

        Returns:
            Tuple of (predictions, scores)
            - predictions: Binary array (1 = anomaly, 0 = normal)
            - scores: Anomaly scores (higher = more anomalous)
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Save model to disk

        Args:
            path: Path to save model
        """
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        """
        Load model from disk

        Args:
            path: Path to load model from
        """
        pass

    def predict_single(self, x: np.ndarray) -> Tuple[bool, float]:
        """
        Predict single sample

        Args:
            x: Single feature vector

        Returns:
            Tuple of (is_anomaly, score)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        predictions, scores = self.predict(x)
        return bool(predictions[0]), float(scores[0])
