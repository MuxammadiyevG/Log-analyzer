"""
Data preparation utilities for training
"""

import random
import sys
from pathlib import Path
from typing import List, Tuple

from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_and_split_logs(
    file_path: str, validation_split: float = 0.2, seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Load logs and split into train/validation

    Args:
        file_path: Path to log file
        validation_split: Fraction for validation set
        seed: Random seed

    Returns:
        Tuple of (train_logs, val_logs)
    """
    logger.info(f"Loading logs from {file_path}...")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        logs = [line.strip() for line in f if line.strip()]

    logger.info(f"Loaded {len(logs)} log lines")

    # Shuffle
    random.seed(seed)
    random.shuffle(logs)

    # Split
    split_idx = int(len(logs) * (1 - validation_split))
    train_logs = logs[:split_idx]
    val_logs = logs[split_idx:]

    logger.info(f"Split: {len(train_logs)} train, {len(val_logs)} validation")

    return train_logs, val_logs


def filter_valid_logs(logs: List[str], min_length: int = 20) -> List[str]:
    """
    Filter out invalid or too short logs

    Args:
        logs: List of log lines
        min_length: Minimum log line length

    Returns:
        Filtered log lines
    """
    filtered = [log for log in logs if len(log) >= min_length]
    logger.info(f"Filtered: {len(logs)} -> {len(filtered)} logs")
    return filtered


def deduplicate_logs(logs: List[str]) -> List[str]:
    """
    Remove duplicate log lines

    Args:
        logs: List of log lines

    Returns:
        Deduplicated log lines
    """
    unique_logs = list(dict.fromkeys(logs))
    logger.info(f"Deduplicated: {len(logs)} -> {len(unique_logs)} logs")
    return unique_logs


def sample_logs(logs: List[str], n_samples: int, seed: int = 42) -> List[str]:
    """
    Randomly sample logs

    Args:
        logs: List of log lines
        n_samples: Number of samples to take
        seed: Random seed

    Returns:
        Sampled log lines
    """
    if len(logs) <= n_samples:
        return logs

    random.seed(seed)
    sampled = random.sample(logs, n_samples)
    logger.info(f"Sampled {n_samples} logs from {len(logs)}")
    return sampled


def prepare_training_data(
    file_path: str,
    max_samples: int = None,
    validation_split: float = 0.2,
    deduplicate: bool = True,
    min_length: int = 20,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Complete data preparation pipeline

    Args:
        file_path: Path to log file
        max_samples: Maximum number of samples (None for all)
        validation_split: Validation set fraction
        deduplicate: Whether to remove duplicates
        min_length: Minimum log line length
        seed: Random seed

    Returns:
        Tuple of (train_logs, val_logs)
    """
    logger.info("=" * 60)
    logger.info("Data Preparation Pipeline")
    logger.info("=" * 60)

    # Load
    train_logs, val_logs = load_and_split_logs(file_path, validation_split, seed)

    # Filter
    train_logs = filter_valid_logs(train_logs, min_length)
    val_logs = filter_valid_logs(val_logs, min_length)

    # Deduplicate
    if deduplicate:
        train_logs = deduplicate_logs(train_logs)
        val_logs = deduplicate_logs(val_logs)

    # Sample
    if max_samples and len(train_logs) > max_samples:
        train_logs = sample_logs(train_logs, max_samples, seed)

    logger.info("=" * 60)
    logger.info(f"Final: {len(train_logs)} train, {len(val_logs)} validation")
    logger.info("=" * 60)

    return train_logs, val_logs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--input", type=str, required=True, help="Input log file")
    parser.add_argument("--output-train", type=str, default="data/train_logs.txt")
    parser.add_argument("--output-val", type=str, default="data/val_logs.txt")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--validation-split", type=float, default=0.2)

    args = parser.parse_args()

    train_logs, val_logs = prepare_training_data(
        args.input, max_samples=args.max_samples, validation_split=args.validation_split
    )

    # Save
    Path(args.output_train).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_train, "w") as f:
        f.write("\n".join(train_logs))

    with open(args.output_val, "w") as f:
        f.write("\n".join(val_logs))

    logger.info(f"Saved to {args.output_train} and {args.output_val}")
