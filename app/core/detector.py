"""
Main anomaly detection orchestrator
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from loguru import logger

from app.core.config import settings
from app.features.extractor import FeatureExtractor
from app.models.autoencoder import AutoencoderDetector
from app.models.isolation_forest import IsolationForestDetector
from app.parsers.log_parser import LogParser


class AnomalyDetector:
    """Main anomaly detection system"""

    def __init__(self, model_type: Optional[str] = None):
        """
        Initialize anomaly detector

        Args:
            model_type: Type of model to use (isolation_forest, autoencoder)
        """
        # Load configurations
        self.parser_config = settings.parser_cfg
        self.feature_config = settings.feature_cfg
        self.threshold_config = settings.threshold_cfg

        # Initialize components
        self.parser = LogParser(self.parser_config)
        self.feature_extractor = FeatureExtractor(self.feature_config)

        # Initialize model
        self.model_type = model_type or settings.MODEL_TYPE
        self.model = self._create_model(self.model_type)

        logger.info(f"AnomalyDetector initialized with {self.model_type} model")

    def _create_model(self, model_type: str):
        """Create model based on type"""
        if model_type == "isolation_forest":
            config = settings.get_config("isolation_forest", default={})
            return IsolationForestDetector(config)
        elif model_type == "autoencoder":
            config = settings.get_config("autoencoder", default={})
            # Set input dim based on feature extractor
            config["input_dim"] = self.feature_extractor.feature_dim
            return AutoencoderDetector(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def analyze_log(self, log_line: str) -> Dict[str, Any]:
        """
        Analyze a single log line

        Args:
            log_line: Raw log line string

        Returns:
            Analysis result dictionary
        """
        # Parse log
        parsed = self.parser.parse(log_line)

        # Extract features
        features = self.feature_extractor.extract(parsed)
        features_scaled = self.feature_extractor.transform(features.reshape(1, -1))

        # Predict
        is_anomaly, score = self.model.predict_single(features_scaled[0])

        # Generate explanation
        explanation = self._explain_anomaly(parsed, features, is_anomaly, score)

        # Build result
        result = {
            "anomaly": is_anomaly,
            "score": float(score),
            "model": self.model_type,
            "reason": explanation,
            "parsed_log": {
                "timestamp": str(parsed.get("timestamp")),
                "ip": parsed.get("ip"),
                "method": parsed.get("method"),
                "path": parsed.get("path"),
                "status": parsed.get("status"),
                "template_id": parsed.get("template_id"),
            },
        }

        return result

    def _explain_anomaly(
        self,
        parsed: Dict[str, Any],
        features: np.ndarray,
        is_anomaly: bool,
        score: float,
    ) -> str:
        """Generate human-readable explanation"""
        if not is_anomaly:
            return "Normal behavior detected"

        reasons = []

        # Check template
        template_id = parsed.get("template_id", -1)
        if template_id == -1 or template_id > 1000:
            reasons.append("rare log template")

        # Check status code
        status = parsed.get("status", 200)
        if status >= 400:
            if status >= 500:
                reasons.append("server error (5xx)")
            else:
                reasons.append("client error (4xx)")

        # Check path
        path = parsed.get("path", "/")
        suspicious_patterns = self.feature_config.get("suspicious_patterns", [])
        found_patterns = [p for p in suspicious_patterns if p.lower() in path.lower()]
        if found_patterns:
            reasons.append(f"suspicious pattern: {found_patterns[0]}")

        if len(path) > 200:
            reasons.append("unusually long path")

        # Check method
        method = parsed.get("method", "GET")
        if method in ["DELETE", "PUT", "PATCH"]:
            reasons.append(f"uncommon method: {method}")

        if not reasons:
            reasons.append(f"high anomaly score ({score:.2f})")

        return " + ".join(reasons)

    def train(self, log_lines: list) -> Dict[str, Any]:
        """
        Train the model on historical logs

        Args:
            log_lines: List of raw log lines

        Returns:
            Training statistics
        """
        logger.info(f"Training on {len(log_lines)} log lines...")

        # Parse logs
        parsed_logs = []
        for line in log_lines:
            try:
                parsed = self.parser.parse(line)
                if parsed and parsed.get("status", 0) > 0:
                    parsed_logs.append(parsed)
            except Exception as e:
                logger.warning(f"Failed to parse log: {e}")
                continue

        logger.info(f"Successfully parsed {len(parsed_logs)} logs")

        # Extract features
        features = self.feature_extractor.extract_batch(parsed_logs)

        # Fit scaler and transform
        features_scaled = self.feature_extractor.fit_transform(features)

        # Train model
        self.model.train(features_scaled)

        return {
            "total_logs": len(log_lines),
            "parsed_logs": len(parsed_logs),
            "features_shape": features.shape,
            "model_type": self.model_type,
            "status": "trained",
        }

    def save(self, path: Optional[Path] = None) -> None:
        """Save detector components"""
        if path is None:
            path = Path(settings.MODEL_PATH)

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save(path)

        # Save feature scaler
        import joblib

        scaler_path = path / "feature_scaler.joblib"
        joblib.dump(
            {
                "scaler": self.feature_extractor.scaler,
                "is_fitted": self.feature_extractor.is_fitted,
            },
            scaler_path,
        )

        logger.info(f"Detector saved to {path}")

    def load(self, path: Optional[Path] = None) -> None:
        """Load detector components"""
        if path is None:
            path = Path(settings.MODEL_PATH)

        path = Path(path)

        # Load model
        self.model.load(path)

        # Load feature scaler
        import joblib

        scaler_path = path / "feature_scaler.joblib"
        if scaler_path.exists():
            data = joblib.load(scaler_path)
            self.feature_extractor.scaler = data["scaler"]
            self.feature_extractor.is_fitted = data["is_fitted"]

        logger.info(f"Detector loaded from {path}")
