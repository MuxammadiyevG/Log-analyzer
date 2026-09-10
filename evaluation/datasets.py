"""
Benchmark datasets.

Two sources:

1. WebLogBenchmark — the project's OWN Apache/Nginx logs, made into a labeled
   benchmark by injecting synthetic attacks into the (future) test window. This
   is where the current IsolationForest model actually lives, so it gives an
   honest baseline for *this* project today.

2. LoghubHDFS — scaffold loader for the public HDFS dataset (session windows by
   block id). The current point-wise model cannot consume sessions, but every
   SOTA sequence model in the roadmap (DeepLog, LogAnomaly, LogLLM) can, so the
   loader is here and ready for Phase 3+.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

from evaluation.splits import temporal_split
from evaluation.synthetic import inject_anomalies

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEBLOG = PROJECT_ROOT / "data" / "sample_logs.log"

_TS_RE = re.compile(r"\[(\d{2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2})")
_BLK_RE = re.compile(r"(blk_-?\d+)")


@dataclass
class Benchmark:
    """A ready-to-score benchmark: normal-only train + labeled test."""

    name: str
    train_lines: List[str]
    test_lines: List[str]
    test_labels: np.ndarray
    meta: Dict[str, object] = field(default_factory=dict)

    def summary(self) -> str:
        n = len(self.test_labels)
        pos = int(self.test_labels.sum())
        return (
            f"{self.name}: train={len(self.train_lines)} (normal only), "
            f"test={n} ({pos} anomalies, {pos / n:.2%})"
        )


def _extract_ts(line: str) -> Optional[datetime]:
    m = _TS_RE.search(line)
    if not m:
        return None
    try:
        naive = datetime.strptime(m.group(1), "%d/%b/%Y:%H:%M:%S")
        return naive.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def build_weblog_benchmark(
    path: Path = DEFAULT_WEBLOG,
    *,
    test_fraction: float = 0.3,
    anomaly_fraction: float = 0.05,
    seed: int = 42,
) -> Benchmark:
    """Build a labeled benchmark from the project's own web-server logs.

    Protocol (unsupervised, leakage-free):
      1. Read real lines, order them chronologically.
      2. Hold out the latest `test_fraction` as the test window; the earlier
         lines are the (presumed-normal) training set.
      3. Inject synthetic labeled attacks into the test window's time range.
      4. Test set = held-out real lines (label 0) + injected attacks (label 1).

    The model only ever trains on past, presumed-normal traffic — exactly how
    an unsupervised detector runs in production.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Web log not found: {path}. Point --data at an Apache/Nginx combined log."
        )

    raw = [ln.strip() for ln in path.read_text(errors="ignore").splitlines() if ln.strip()]
    logger.info(f"Loaded {len(raw)} real log lines from {path}")

    # Order chronologically; lines without a parseable timestamp keep file order.
    with_ts = [(ln, _extract_ts(ln)) for ln in raw]
    parseable = [t for _, t in with_ts if t is not None]
    if parseable:
        base = min(parseable)
        ordered = sorted(
            with_ts, key=lambda p: p[1] if p[1] is not None else base
        )
    else:
        ordered = with_ts

    train_pairs, test_pairs = temporal_split(
        ordered, test_fraction=test_fraction, already_sorted=True
    )
    train_lines = [ln for ln, _ in train_pairs]
    test_normal = [ln for ln, _ in test_pairs]

    # Determine the test-window time range for injected anomalies.
    test_ts = [t for _, t in test_pairs if t is not None]
    if test_ts:
        time_lo, time_hi = min(test_ts), max(test_ts)
    else:
        now = datetime.now(timezone.utc)
        time_lo, time_hi = now, now

    n_anom = max(int(len(test_normal) * anomaly_fraction), 25)
    injected = inject_anomalies(time_lo, time_hi, n_anom, seed=seed)
    logger.info(f"Injected {len(injected)} labeled anomalies into test window")

    test_lines = list(test_normal) + [ln for ln, _ in injected]
    test_labels = np.array([0] * len(test_normal) + [1] * len(injected), dtype=int)

    categories: Dict[str, int] = {}
    for _, cat in injected:
        categories[cat] = categories.get(cat, 0) + 1

    return Benchmark(
        name="weblog",
        train_lines=train_lines,
        test_lines=test_lines,
        test_labels=test_labels,
        meta={
            "source": str(path),
            "time_lo": time_lo.isoformat(),
            "time_hi": time_hi.isoformat(),
            "injected_categories": categories,
        },
    )


# --- Loghub HDFS scaffold (for future sequence models) ----------------------


@dataclass
class SessionDataset:
    """HDFS-style sessions: each session is a sequence of raw log lines."""

    name: str
    sessions: List[List[str]]
    labels: np.ndarray
    meta: Dict[str, object] = field(default_factory=dict)


def load_hdfs_sessions(
    log_path: Path,
    label_path: Path,
) -> SessionDataset:
    """Load the public HDFS dataset grouped into session windows by block id.

    Download (full dataset, ~1.5 GB) from Loghub:
        https://github.com/logpai/loghub  ->  HDFS_v1
    You need two files:
        HDFS.log            (raw log lines)
        anomaly_label.csv   (columns: BlockId, Label)

    The current point-wise IsolationForest cannot consume sessions; this loader
    exists for the sequence models in Phase 3+ (DeepLog / LogAnomaly / LogLLM).
    """
    log_path, label_path = Path(log_path), Path(label_path)
    if not log_path.exists() or not label_path.exists():
        raise FileNotFoundError(
            "HDFS files missing. Get HDFS.log + anomaly_label.csv from "
            "https://github.com/logpai/loghub (HDFS_v1) and pass their paths."
        )

    # BlockId -> label (1 = anomaly)
    label_map: Dict[str, int] = {}
    for i, row in enumerate(label_path.read_text().splitlines()):
        if i == 0 and "BlockId" in row:
            continue
        parts = row.split(",")
        if len(parts) >= 2:
            label_map[parts[0].strip()] = 1 if parts[1].strip().lower() == "anomaly" else 0

    # Group lines by block id, preserving first-seen order (temporal proxy).
    sessions: Dict[str, List[str]] = {}
    order: List[str] = []
    for line in log_path.read_text(errors="ignore").splitlines():
        m = _BLK_RE.search(line)
        if not m:
            continue
        blk = m.group(1)
        if blk not in sessions:
            sessions[blk] = []
            order.append(blk)
        sessions[blk].append(line.strip())

    seqs, labs = [], []
    for blk in order:
        if blk in label_map:
            seqs.append(sessions[blk])
            labs.append(label_map[blk])

    logger.info(f"HDFS: {len(seqs)} sessions, {int(sum(labs))} anomalous")
    return SessionDataset(
        name="hdfs",
        sessions=seqs,
        labels=np.array(labs, dtype=int),
        meta={"n_blocks": len(seqs)},
    )
