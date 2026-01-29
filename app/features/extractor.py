"""
Feature extraction for anomaly detection
"""

import hashlib
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
from loguru import logger
from sklearn.preprocessing import StandardScaler


class FeatureExtractor:
    """Extract numerical features from parsed logs"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature extractor

        Args:
            config: Feature configuration dictionary
        """
        self.config = config
        self.enabled_features = config.get("enabled", [])
        self.suspicious_patterns = config.get("suspicious_patterns", [])
        self.ip_window = config.get("ip_window_minutes", 60)

        # Tracking for IP statistics
        self.ip_requests = defaultdict(lambda: deque(maxlen=1000))
        self.ip_paths = defaultdict(set)

        # Scalers for normalization
        self.scaler = StandardScaler()
        self.is_fitted = False

        # Feature dimension
        self.feature_dim = self._calculate_feature_dim()

        logger.info(f"FeatureExtractor initialized with {self.feature_dim} features")

    def _calculate_feature_dim(self) -> int:
        """Calculate total feature dimension"""
        dim = 0
        if "template_id" in self.enabled_features:
            dim += 1
        if "status_code_category" in self.enabled_features:
            dim += 3  # 2xx, 4xx, 5xx
        if "path_length" in self.enabled_features:
            dim += 1
        if "suspicious_tokens" in self.enabled_features:
            dim += 1
        if "http_method" in self.enabled_features:
            dim += 7  # GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS
        if "time_features" in self.enabled_features:
            dim += 2  # hour, weekday
        if "ip_statistics" in self.enabled_features:
            dim += 3  # frequency, path_diversity, request_rate
        if "user_agent_hash" in self.enabled_features:
            dim += 1
        return dim

    def extract(self, parsed_log: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from parsed log

        Args:
            parsed_log: Dictionary with parsed log fields

        Returns:
            Feature vector as numpy array
        """
        features = []

        if "template_id" in self.enabled_features:
            features.append(self._extract_template_id(parsed_log))

        if "status_code_category" in self.enabled_features:
            features.extend(self._extract_status_category(parsed_log))

        if "path_length" in self.enabled_features:
            features.append(self._extract_path_length(parsed_log))

        if "suspicious_tokens" in self.enabled_features:
            features.append(self._extract_suspicious_score(parsed_log))

        if "http_method" in self.enabled_features:
            features.extend(self._extract_http_method(parsed_log))

        if "time_features" in self.enabled_features:
            features.extend(self._extract_time_features(parsed_log))

        if "ip_statistics" in self.enabled_features:
            features.extend(self._extract_ip_statistics(parsed_log))

        if "user_agent_hash" in self.enabled_features:
            features.append(self._extract_user_agent_hash(parsed_log))

        return np.array(features, dtype=np.float32)

    def extract_batch(self, parsed_logs: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extract features from multiple logs

        Args:
            parsed_logs: List of parsed log dictionaries

        Returns:
            Feature matrix (n_samples, n_features)
        """
        features = [self.extract(log) for log in parsed_logs]
        return np.array(features, dtype=np.float32)

    def fit_scaler(self, features: np.ndarray):
        """
        Fit the feature scaler

        Args:
            features: Feature matrix to fit on
        """
        self.scaler.fit(features)
        self.is_fitted = True
        logger.info("Feature scaler fitted")

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted scaler

        Args:
            features: Raw feature matrix

        Returns:
            Scaled feature matrix
        """
        if not self.is_fitted:
            logger.warning("Scaler not fitted, returning raw features")
            return features
        return self.scaler.transform(features)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit_scaler(features)
        return self.transform(features)

    # Feature extraction methods

    def _extract_template_id(self, log: Dict[str, Any]) -> float:
        """Extract log template ID"""
        return float(log.get("template_id", -1))

    def _extract_status_category(self, log: Dict[str, Any]) -> List[float]:
        """One-hot encode status code category"""
        status = log.get("status", 200)
        return [
            1.0 if 200 <= status < 300 else 0.0,  # 2xx
            1.0 if 400 <= status < 500 else 0.0,  # 4xx
            1.0 if 500 <= status < 600 else 0.0,  # 5xx
        ]

    def _extract_path_length(self, log: Dict[str, Any]) -> float:
        """Extract request path length"""
        path = log.get("path", "/")
        return float(len(path))

    def _extract_suspicious_score(self, log: Dict[str, Any]) -> float:
        """Count suspicious patterns in path"""
        path = log.get("path", "").lower()
        raw = log.get("raw", "").lower()

        score = 0
        for pattern in self.suspicious_patterns:
            if pattern.lower() in path or pattern.lower() in raw:
                score += 1

        return float(score)

    def _extract_http_method(self, log: Dict[str, Any]) -> List[float]:
        """One-hot encode HTTP method"""
        method = log.get("method", "GET").upper()
        methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
        return [1.0 if method == m else 0.0 for m in methods]

    def _extract_time_features(self, log: Dict[str, Any]) -> List[float]:
        """Extract time-based features"""
        timestamp = log.get("timestamp", datetime.now())
        if not isinstance(timestamp, datetime):
            timestamp = datetime.now()

        return [float(timestamp.hour), float(timestamp.weekday())]

    def _extract_ip_statistics(self, log: Dict[str, Any]) -> List[float]:
        """Extract IP behavior statistics"""
        ip = log.get("ip", "unknown")
        path = log.get("path", "/")
        timestamp = log.get("timestamp", datetime.now())

        if not isinstance(timestamp, datetime):
            timestamp = datetime.now()

        # Update tracking
        self.ip_requests[ip].append(timestamp)
        self.ip_paths[ip].add(path)

        # Calculate statistics
        # 1. Request frequency (requests in last N minutes)
        cutoff = timestamp - timedelta(minutes=self.ip_window)
        recent_requests = [t for t in self.ip_requests[ip] if t > cutoff]
        frequency = len(recent_requests)

        # 2. Path diversity (unique paths)
        path_diversity = len(self.ip_paths[ip])

        # 3. Request rate (requests per minute)
        if len(recent_requests) > 1:
            time_span = (recent_requests[-1] - recent_requests[0]).total_seconds() / 60
            request_rate = len(recent_requests) / max(time_span, 1)
        else:
            request_rate = 0

        return [float(frequency), float(path_diversity), float(request_rate)]

    def _extract_user_agent_hash(self, log: Dict[str, Any]) -> float:
        """Hash user agent to numeric value"""
        user_agent = log.get("user_agent", "-")
        if user_agent == "-" or not user_agent:
            return 0.0

        # Simple hash to numeric
        hash_val = int(hashlib.md5(user_agent.encode()).hexdigest()[:8], 16)
        return float(hash_val % 10000)  # Modulo to keep value reasonable

    def get_feature_names(self) -> List[str]:
        """Get feature names in order"""
        names = []

        if "template_id" in self.enabled_features:
            names.append("template_id")

        if "status_code_category" in self.enabled_features:
            names.extend(["status_2xx", "status_4xx", "status_5xx"])

        if "path_length" in self.enabled_features:
            names.append("path_length")

        if "suspicious_tokens" in self.enabled_features:
            names.append("suspicious_score")

        if "http_method" in self.enabled_features:
            names.extend(
                [
                    "method_GET",
                    "method_POST",
                    "method_PUT",
                    "method_DELETE",
                    "method_PATCH",
                    "method_HEAD",
                    "method_OPTIONS",
                ]
            )

        if "time_features" in self.enabled_features:
            names.extend(["hour", "weekday"])

        if "ip_statistics" in self.enabled_features:
            names.extend(["ip_frequency", "ip_path_diversity", "ip_request_rate"])

        if "user_agent_hash" in self.enabled_features:
            names.append("user_agent_hash")

        return names
