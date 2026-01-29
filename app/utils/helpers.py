"""
Helper utilities
"""

from datetime import datetime
from typing import List, Tuple

import numpy as np


def calculate_percentile_threshold(
    scores: np.ndarray, percentile: float = 95.0
) -> float:
    """
    Calculate threshold based on percentile

    Args:
        scores: Array of anomaly scores
        percentile: Percentile value (0-100)

    Returns:
        Threshold value
    """
    return np.percentile(scores, percentile)


def calculate_mad_threshold(scores: np.ndarray, n_mad: float = 3.5) -> float:
    """
    Calculate threshold using Median Absolute Deviation (MAD)

    Args:
        scores: Array of anomaly scores
        n_mad: Number of MADs from median

    Returns:
        Threshold value
    """
    median = np.median(scores)
    mad = np.median(np.abs(scores - median))
    return median + n_mad * mad


def calculate_iqr_threshold(scores: np.ndarray, multiplier: float = 1.5) -> float:
    """
    Calculate threshold using Interquartile Range (IQR)

    Args:
        scores: Array of anomaly scores
        multiplier: IQR multiplier

    Returns:
        Threshold value
    """
    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1
    return q3 + multiplier * iqr


def batch_process(items: List, batch_size: int = 100):
    """
    Process items in batches

    Args:
        items: List of items to process
        batch_size: Size of each batch

    Yields:
        Batch of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def format_timestamp(dt: datetime) -> str:
    """Format datetime to string"""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division with default value

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division fails

    Returns:
        Division result or default
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (ZeroDivisionError, TypeError):
        return default


def moving_average(values: List[float], window: int = 5) -> List[float]:
    """
    Calculate moving average

    Args:
        values: List of values
        window: Window size

    Returns:
        List of moving averages
    """
    if len(values) < window:
        return values

    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        end = i + 1
        result.append(sum(values[start:end]) / (end - start))

    return result


def calculate_entropy(values: List) -> float:
    """
    Calculate Shannon entropy

    Args:
        values: List of values

    Returns:
        Entropy value
    """
    if not values:
        return 0.0

    from collections import Counter

    counts = Counter(values)
    total = len(values)

    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)

    return entropy
