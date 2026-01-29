"""
Model training script
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.core.detector import AnomalyDetector


def load_logs(file_path: str) -> list:
    """
    Load logs from file

    Args:
        file_path: Path to log file

    Returns:
        List of log lines
    """
    logger.info(f"Loading logs from {file_path}...")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {file_path}")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        logs = [line.strip() for line in f if line.strip()]

    logger.info(f"Loaded {len(logs)} log lines")
    return logs


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train anomaly detection model")
    parser.add_argument(
        "--data",
        type=str,
        default="data/sample_logs.txt",
        help="Path to training data file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["isolation_forest", "autoencoder"],
        help="Model type to train",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output directory for trained model"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to use for training",
    )

    args = parser.parse_args()

    # Load configuration
    training_config = settings.training_cfg
    data_path = args.data or training_config.get("data_path", "data/sample_logs.txt")
    model_type = args.model or settings.MODEL_TYPE
    output_path = args.output or settings.MODEL_PATH
    max_samples = args.max_samples or training_config.get("max_samples")

    logger.info("=" * 60)
    logger.info("Log Anomaly Detection - Model Training")
    logger.info("=" * 60)
    logger.info(f"Data path: {data_path}")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Output path: {output_path}")

    # Load data
    try:
        logs = load_logs(data_path)
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)

    # Limit samples if specified
    if max_samples and len(logs) > max_samples:
        logger.info(f"Limiting to {max_samples} samples")
        import random

        random.seed(42)
        logs = random.sample(logs, max_samples)

    # Check minimum samples
    min_samples = training_config.get("min_samples", 1000)
    if len(logs) < min_samples:
        logger.error(
            f"Insufficient data. Need at least {min_samples} logs, got {len(logs)}"
        )
        sys.exit(1)

    # Initialize detector
    logger.info(f"Initializing {model_type} detector...")
    detector = AnomalyDetector(model_type=model_type)

    # Train
    try:
        stats = detector.train(logs)

        logger.info("=" * 60)
        logger.info("Training Statistics:")
        logger.info("=" * 60)
        for key, value in stats.items():
            logger.info(f"{key}: {value}")

        # Save model
        detector.save(Path(output_path))

        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {output_path}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
