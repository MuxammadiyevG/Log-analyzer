"""
Train/test splitting — with the leakage trap built in as a first-class warning.

The single most cited evaluation mistake in log-anomaly research ("How Far Are
We?", ICSE 2022): splitting logs *randomly* lets future events leak into the
training set, so the reported F1 is fantasy. Honest numbers require a
**chronological (temporal)** split. Both are provided here; the random one logs
a loud warning every time it is used.
"""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple, TypeVar

from loguru import logger

T = TypeVar("T")


def temporal_split(
    items: Sequence[T],
    *,
    test_fraction: float = 0.3,
    already_sorted: bool = False,
    key=None,
) -> Tuple[List[T], List[T]]:
    """Chronological split: earliest events train, latest events test.

    This is the only leakage-free protocol for time-ordered logs.

    Args:
        items: Sequence of records, ideally each carrying a timestamp.
        test_fraction: Fraction of the *tail* held out for testing.
        already_sorted: Set True if `items` is already time-ordered.
        key: Optional sort key (e.g. lambda r: r["timestamp"]).

    Returns:
        (train, test) preserving chronological order.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")

    ordered = list(items)
    if not already_sorted and key is not None:
        ordered.sort(key=key)

    split_idx = int(len(ordered) * (1.0 - test_fraction))
    train, test = ordered[:split_idx], ordered[split_idx:]
    logger.info(
        f"Temporal split: {len(train)} train (past) / {len(test)} test (future)"
    )
    return train, test


def random_split(
    items: Sequence[T],
    *,
    test_fraction: float = 0.3,
    seed: int = 42,
) -> Tuple[List[T], List[T]]:
    """Random shuffle split — DEMONSTRATION / ablation ONLY.

    Random splitting of time-ordered logs causes data leakage and inflates
    metrics. It exists here so the harness can *quantify* that inflation
    (temporal vs random on the same data). Never report a random-split number
    as a real result.
    """
    logger.warning(
        "random_split used — this leaks future data and INFLATES metrics. "
        "Use temporal_split for any number you intend to report."
    )
    ordered = list(items)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    split_idx = int(len(ordered) * (1.0 - test_fraction))
    return ordered[:split_idx], ordered[split_idx:]
