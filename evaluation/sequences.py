"""
Sequence tooling for the DeepLog tier (Phase 3).

Provides:
  * A controlled synthetic *session* benchmark with a known normal grammar and
    injected sequence anomalies (brute force, scan, corruption) — gives an
    honest F1 for the sequence detector with no 1.5 GB download.
  * A bigram Markov baseline, so DeepLog is measured against a real (not
    strawman) sequence model.
  * An EventEncoder + converters that turn raw HDFS sessions or per-IP web-log
    streams into integer event sequences the detector consumes.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from app.models.deeplog import PAD, UNK

# --- controlled synthetic grammar ------------------------------------------
# Event ids (0=PAD, 1=UNK reserved by the detector).
HOME, LOGIN, BROWSE, SEARCH, ADD_CART, CHECKOUT, LOGOUT, API_OK, STATIC, FAIL, ADMIN = range(2, 13)
SYNTH_VOCAB = 13  # 0..12

# Normal transition grammar. FAIL is rare and always retries to LOGIN (never
# FAIL->FAIL); ADMIN never occurs in normal traffic.
_NORMAL: Dict[int, List[Tuple[int, float]]] = {
    -1: [(HOME, 0.6), (LOGIN, 0.4)],  # start
    HOME: [(BROWSE, 0.5), (SEARCH, 0.3), (STATIC, 0.2)],
    LOGIN: [(BROWSE, 0.5), (API_OK, 0.3), (FAIL, 0.1), (SEARCH, 0.1)],
    FAIL: [(LOGIN, 1.0)],
    BROWSE: [(BROWSE, 0.35), (SEARCH, 0.2), (ADD_CART, 0.15), (API_OK, 0.15), (STATIC, 0.1), (LOGOUT, 0.05)],
    SEARCH: [(BROWSE, 0.4), (SEARCH, 0.2), (ADD_CART, 0.2), (API_OK, 0.1), (LOGOUT, 0.1)],
    ADD_CART: [(BROWSE, 0.3), (CHECKOUT, 0.5), (SEARCH, 0.2)],
    CHECKOUT: [(LOGOUT, 0.7), (BROWSE, 0.3)],
    API_OK: [(BROWSE, 0.4), (SEARCH, 0.3), (API_OK, 0.2), (LOGOUT, 0.1)],
    STATIC: [(BROWSE, 0.5), (STATIC, 0.3), (LOGOUT, 0.2)],
}


def _draw(rng: random.Random, choices: List[Tuple[int, float]]) -> int:
    r, acc = rng.random(), 0.0
    for ev, p in choices:
        acc += p
        if r <= acc:
            return ev
    return choices[-1][0]


def _normal_session(rng: random.Random, max_len: int = 30) -> List[int]:
    seq = [_draw(rng, _NORMAL[-1])]
    while len(seq) < max_len:
        cur = seq[-1]
        if cur == LOGOUT:
            break
        seq.append(_draw(rng, _NORMAL[cur]))
    return seq


def _brute_force(rng: random.Random) -> List[int]:
    return [LOGIN] + [FAIL] * rng.randint(8, 16)  # FAIL->FAIL never normal


def _admin_scan(rng: random.Random) -> List[int]:
    return [ADMIN] * rng.randint(6, 14)  # ADMIN unseen in normal


def _corruption(rng: random.Random) -> List[int]:
    base = _normal_session(rng)
    junk = [rng.randint(HOME, ADMIN) for _ in range(rng.randint(5, 10))]
    cut = rng.randint(0, len(base))
    return base[:cut] + junk + base[cut:]


# --- higher-order grammar (long-range dependency) --------------------------
# A bigram cannot solve this: a session opens a resource and must CLOSE the SAME
# one many steps later. Local transitions (WORK->CLOSE_*) are all "seen", so a
# bigram can't tell a matched close from a mismatched one. A model with real
# context (Transformer, or an LSTM whose window spans the gap) can.
WORK = 2
OPEN = [3, 4, 5]   # OPEN_1, OPEN_2, OPEN_3
CLOSE = [6, 7, 8]  # CLOSE_1, CLOSE_2, CLOSE_3
BFAIL = 9          # burst event (quantitative anomaly)
SCAN_EV = 10       # never in normal (novel)
MATCHED_VOCAB = 11


# Fixed filler length: the CLOSE *position* is deterministic, so the ONLY thing
# to learn is WHICH resource to close — a pure long-range dependency probe.
_FILLER = 4


def _matched_normal(rng: random.Random) -> List[int]:
    r = rng.randrange(len(OPEN))
    return [OPEN[r]] + [WORK] * _FILLER + [CLOSE[r]]


def _mismatched_close(rng: random.Random) -> List[int]:
    r = rng.randrange(len(OPEN))
    wrong = rng.choice([j for j in range(len(OPEN)) if j != r])
    return [OPEN[r]] + [WORK] * _FILLER + [CLOSE[wrong]]  # long-range violation


def _matched_burst(rng: random.Random) -> List[int]:
    r = rng.randrange(len(OPEN))
    return [OPEN[r]] + [BFAIL] * rng.randint(8, 16) + [CLOSE[r]]


def _matched_scan(rng: random.Random) -> List[int]:
    return [SCAN_EV] * rng.randint(6, 12)


def build_matched_sessions(
    *,
    n_train: int = 2500,
    n_test_normal: int = 600,
    n_anomalies: int = 150,
    seed: int = 42,
) -> "SequenceBenchmark":
    """Higher-order benchmark: matched OPEN/CLOSE + mismatch/burst/scan anomalies.

    The mismatched-close anomaly is invisible to a bigram — the point of this
    benchmark is to show a context model (Transformer / DeepLog) earn its keep.
    """
    rng = random.Random(seed)
    train = [_matched_normal(rng) for _ in range(n_train)]
    test_normal = [_matched_normal(rng) for _ in range(n_test_normal)]
    gens = [_mismatched_close, _matched_burst, _matched_scan]
    anomalies = [gens[i % len(gens)](rng) for i in range(n_anomalies)]
    test = test_normal + anomalies
    labels = np.array([0] * len(test_normal) + [1] * len(anomalies), dtype=int)
    return SequenceBenchmark(
        name="matched",
        train_sequences=train,
        test_sequences=test,
        test_labels=labels,
        vocab_size=MATCHED_VOCAB,
        meta={"anomaly_types": ["mismatched_close", "burst", "scan"], "long_range": True},
    )


@dataclass
class SequenceBenchmark:
    name: str
    train_sequences: List[List[int]]
    test_sequences: List[List[int]]
    test_labels: np.ndarray
    vocab_size: int
    meta: Dict[str, object] = field(default_factory=dict)

    def summary(self) -> str:
        n = len(self.test_labels)
        pos = int(self.test_labels.sum())
        return (
            f"{self.name}: train={len(self.train_sequences)} normal sessions, "
            f"test={n} ({pos} anomalous, {pos / n:.1%})"
        )


def build_synthetic_sessions(
    *,
    n_train: int = 2000,
    n_test_normal: int = 600,
    n_anomalies: int = 150,
    seed: int = 42,
) -> SequenceBenchmark:
    """Controlled session benchmark: normal grammar + injected sequence attacks."""
    rng = random.Random(seed)
    train = [_normal_session(rng) for _ in range(n_train)]
    test_normal = [_normal_session(rng) for _ in range(n_test_normal)]

    gens = [_brute_force, _admin_scan, _corruption]
    anomalies = [gens[i % len(gens)](rng) for i in range(n_anomalies)]

    test = test_normal + anomalies
    labels = np.array([0] * len(test_normal) + [1] * len(anomalies), dtype=int)
    return SequenceBenchmark(
        name="seqsynth",
        train_sequences=train,
        test_sequences=test,
        test_labels=labels,
        vocab_size=SYNTH_VOCAB,
        meta={"anomaly_types": ["brute_force", "admin_scan", "corruption"]},
    )


# --- bigram Markov baseline -------------------------------------------------


class MarkovBigramBaseline:
    """Fit bigram transition probabilities on normal sessions.

    Anomaly score = mean surprisal (-log p) of a sequence's transitions; unseen
    transitions get a floor probability, so novel grammar scores high. A fair,
    cheap sequence baseline for DeepLog to beat.
    """

    def __init__(self, vocab_size: int, floor: float = 1e-4):
        self.vocab_size = vocab_size
        self.floor = floor
        self.logp: Dict[Tuple[int, int], float] = {}
        self.is_trained = False

    def fit(self, sequences: List[Sequence[int]]) -> "MarkovBigramBaseline":
        counts: Dict[int, Dict[int, int]] = {}
        for s in sequences:
            for a, b in zip(s[:-1], s[1:]):
                counts.setdefault(a, {}).setdefault(b, 0)
                counts[a][b] += 1
        for a, nxt in counts.items():
            tot = sum(nxt.values())
            for b, c in nxt.items():
                self.logp[(a, b)] = np.log(c / tot)
        self.is_trained = True
        return self

    def score(self, sequences: List[Sequence[int]]) -> Tuple[np.ndarray, np.ndarray]:
        floor_lp = np.log(self.floor)
        scores = []
        for s in sequences:
            trans = list(zip(s[:-1], s[1:]))
            if not trans:
                scores.append(0.0)
                continue
            surprisal = [-self.logp.get((a, b), floor_lp) for a, b in trans]
            scores.append(float(np.mean(surprisal)))
        scores = np.array(scores, dtype=float)
        # Operating threshold: anything above the max normal-ish surprisal band.
        thr = np.percentile(scores, 95)
        preds = (scores > thr).astype(int)
        return scores, preds


# --- event encoding for real logs -------------------------------------------

_WEB_REQ = re.compile(r'"([A-Z]+)\s+(\S+)\s+HTTP/[\d.]+"\s+(\d{3})')
_NUM = re.compile(r"\d+")
_IP = re.compile(r"^(\d{1,3}(?:\.\d{1,3}){3})")


def web_template(raw: str) -> str:
    """Event signature for a web log line: METHOD path-template status-bucket."""
    m = _WEB_REQ.search(raw)
    if not m:
        return "OTHER"
    method, path, status = m.group(1), m.group(2), int(m.group(3))
    path = path.split("?")[0]
    path = _NUM.sub("<n>", path)  # collapse numeric ids
    bucket = f"{status // 100}xx"
    return f"{method} {path} {bucket}"


_HDFS_BLK = re.compile(r"blk_-?\d+")
_HDFS_IP = re.compile(r"/?\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?")
_HDFS_NUM = re.compile(r"\b\d+\b")


def hdfs_template(raw: str) -> str:
    """Event signature for an HDFS log line (parser-free normalization).

    Masks the variable parts (block ids, IPs, sizes, numbers) so the constant
    log-statement skeleton becomes the event type — the same idea NeuralLog /
    LogLLM use instead of a stateful template miner.
    """
    parts = raw.split(None, 4)
    msg = parts[4] if len(parts) >= 5 else raw  # drop date/time/pid/level
    msg = _HDFS_BLK.sub("<BLK>", msg)
    msg = _HDFS_IP.sub("<IP>", msg)
    msg = _HDFS_NUM.sub("<N>", msg)
    return msg.strip()[:200]


class EventEncoder:
    """Map template strings to integer ids (0=PAD, 1=UNK reserved)."""

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.next_id = 2

    def fit(self, templates: List[str]) -> "EventEncoder":
        for t in templates:
            if t not in self.vocab:
                self.vocab[t] = self.next_id
                self.next_id += 1
        return self

    def encode(self, template: str) -> int:
        return self.vocab.get(template, UNK)

    @property
    def vocab_size(self) -> int:
        return self.next_id


def web_sessions_from_lines(
    lines: List[str], encoder: EventEncoder | None = None, *, fit: bool = False
) -> Tuple[List[List[int]], EventEncoder]:
    """Group web-log lines by client IP into per-IP event-id sequences."""
    encoder = encoder or EventEncoder()
    by_ip: Dict[str, List[str]] = {}
    order: List[str] = []
    for ln in lines:
        m = _IP.match(ln)
        ip = m.group(1) if m else "unknown"
        if ip not in by_ip:
            by_ip[ip] = []
            order.append(ip)
        by_ip[ip].append(web_template(ln))
    if fit:
        encoder.fit([t for ip in order for t in by_ip[ip]])
    return [[encoder.encode(t) for t in by_ip[ip]] for ip in order], encoder


def _hdfs_message(line: str) -> str:
    """Drop the 'date time pid LEVEL' prefix, keep the log statement."""
    parts = line.split(None, 4)
    return parts[4] if len(parts) >= 5 else line


def stream_hdfs_sessions(
    log_path: "Path",
    label_path: "Path",
    template_fn: Callable[[str], str] = hdfs_template,
    *,
    use_drain: bool = True,
) -> Tuple[List[List[int]], "np.ndarray", int]:
    """Memory-light HDFS loader for the full 11M-line log.

    Streams the file line by line and stores only small integer event-id lists
    per block — never the 1.5 GB of raw strings — so it fits in modest RAM.

    Event typing (`use_drain=True`, default) uses Drain3 template mining, the
    canonical approach: it collapses HDFS to its ~30 real event types instead of
    the thousands of spurious variants a naive regex leaves in exception/path
    text. `use_drain=False` falls back to `template_fn` (fast, for tests).

    HDFS anomalies are unusual *sequences* of known events, so building the
    vocabulary over all lines in one pass is standard and correct.

    Returns (sequences, labels, vocab_size), sequences ordered by each block's
    first appearance (a temporal proxy).
    """
    from pathlib import Path as _Path

    log_path, label_path = _Path(log_path), _Path(label_path)
    if not log_path.exists() or not label_path.exists():
        raise FileNotFoundError(
            "HDFS files missing. Get HDFS.log + anomaly_label.csv from "
            "https://github.com/logpai/loghub (HDFS_v1)."
        )

    label_map: Dict[str, int] = {}
    with label_path.open() as f:
        for i, row in enumerate(f):
            if i == 0 and "BlockId" in row:
                continue
            parts = row.split(",")
            if len(parts) >= 2:
                label_map[parts[0].strip()] = (
                    1 if parts[1].strip().lower() == "anomaly" else 0
                )

    miner = None
    if use_drain:
        from drain3 import TemplateMiner
        from drain3.template_miner_config import TemplateMinerConfig

        cfg = TemplateMinerConfig()
        cfg.profiling_enabled = False
        miner = TemplateMiner(config=cfg)

    vocab: Dict[str, int] = {}
    next_id = 2  # 0=PAD, 1=UNK reserved
    max_eid = 1
    sessions: Dict[str, List[int]] = {}
    order: List[str] = []

    with log_path.open(errors="ignore") as f:
        for line in f:
            m = _HDFS_BLK.search(line)
            if not m:
                continue
            blk = m.group(0)
            if miner is not None:
                cid = miner.add_log_message(_hdfs_message(line))["cluster_id"]
                eid = cid + 1  # cluster ids are 1-based; keep 0/1 reserved
                max_eid = max(max_eid, eid)
            else:
                t = template_fn(line)
                eid = vocab.get(t)
                if eid is None:
                    eid = next_id
                    vocab[t] = eid
                    next_id += 1
                max_eid = max(max_eid, eid)
            if blk not in sessions:
                sessions[blk] = []
                order.append(blk)
            sessions[blk].append(eid)

    seqs, labs = [], []
    for blk in order:
        if blk in label_map:
            seqs.append(sessions[blk])
            labs.append(label_map[blk])
    return seqs, np.array(labs, dtype=int), max_eid + 1


def hdfs_to_sequences(
    train_lines_sessions: List[List[str]],
    test_lines_sessions: List[List[str]],
    template_fn: Callable[[str], str],
) -> Tuple[List[List[int]], List[List[int]], int]:
    """Encode HDFS raw-line sessions into event-id sequences.

    Vocabulary is fit on the TRAIN (normal) sessions only; unseen templates in
    test map to UNK, which the detector treats as a grammar violation.
    """
    encoder = EventEncoder()
    encoder.fit([template_fn(ln) for sess in train_lines_sessions for ln in sess])
    train = [[encoder.encode(template_fn(ln)) for ln in s] for s in train_lines_sessions]
    test = [[encoder.encode(template_fn(ln)) for ln in s] for s in test_lines_sessions]
    return train, test, encoder.vocab_size
