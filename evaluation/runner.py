"""
Benchmark runner — wires a concrete model to the model-agnostic metrics.

Phase 1 baseline: the project's existing IsolationForest (point-wise, web-log
domain). Trained fresh on the past/normal split, then scored on the
future+injected test set. Reports two numbers:

  * operating point  — the model's own decisions (sklearn contamination)
  * best-F1 sweep    — the F1 the ranking could reach with an oracle threshold

The gap between them tells you whether the problem is the model or just the
threshold.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger

# Make `app` importable when run as a module from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.detector import AnomalyDetector  # noqa: E402
from app.models.deeplog import DeepLogDetector  # noqa: E402
from app.models.log_transformer import LogTransformerDetector  # noqa: E402
from app.models.quantitative import QuantitativeDetector  # noqa: E402
from app.models.semantic_detector import SemanticAnomalyModel  # noqa: E402
from evaluation.datasets import Benchmark, build_weblog_benchmark  # noqa: E402
from evaluation.metrics import EvalResult, evaluate_scores  # noqa: E402
from evaluation.sequences import (  # noqa: E402
    MarkovBigramBaseline,
    SequenceBenchmark,
    build_matched_sessions,
    build_synthetic_sessions,
    hdfs_template,
    stream_hdfs_sessions,
)


def train_baseline_detector(
    train_lines: List[str], model_type: str = "isolation_forest"
) -> AnomalyDetector:
    """Train a fresh detector on presumed-normal training logs."""
    detector = AnomalyDetector(model_type=model_type)
    stats = detector.train(train_lines)
    logger.info(f"Trained baseline: {stats}")
    return detector


def score_lines(
    detector: AnomalyDetector, lines: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Score each test line. Returns (scores, model_predictions).

    IP-statistics tracking is reset before scoring so training-time state does
    not leak into inference (a fresh production start). This also surfaces the
    known stateful-feature caveat: single-line inference sees frequency ~= 1.
    """
    fe = detector.feature_extractor
    fe.ip_requests.clear()
    fe.ip_paths.clear()

    scores = np.empty(len(lines), dtype=float)
    preds = np.empty(len(lines), dtype=int)
    for i, line in enumerate(lines):
        result = detector.analyze_log(line)
        scores[i] = result["score"]
        preds[i] = 1 if result["anomaly"] else 0
    return scores, preds


def evaluate_benchmark(
    bench: Benchmark,
    detector: AnomalyDetector,
) -> Dict[str, EvalResult]:
    """Score a benchmark's test set and return both scorecards."""
    scores, preds = score_lines(detector, bench.test_lines)
    y = bench.test_labels

    op = evaluate_scores(
        y, scores, name=f"{bench.name}:{detector.model_type}:operating", y_pred=preds
    )
    sweep = evaluate_scores(
        y, scores, name=f"{bench.name}:{detector.model_type}:best-f1"
    )
    return {"operating_point": op, "best_f1_sweep": sweep}


def _scorecards(
    bench: Benchmark, scores: np.ndarray, preds: np.ndarray, prefix: str
) -> Dict[str, EvalResult]:
    """Build the operating-point + best-F1 pair from raw scores."""
    y = bench.test_labels
    return {
        f"{prefix}_operating": evaluate_scores(
            y, scores, name=f"{prefix}:operating", y_pred=preds
        ),
        f"{prefix}_best_f1": evaluate_scores(y, scores, name=f"{prefix}:best-f1"),
    }


def evaluate_semantic(
    bench: Benchmark, model: SemanticAnomalyModel
) -> Dict[str, EvalResult]:
    scores, preds = model.score(bench.test_lines)
    cards = _scorecards(bench, scores, preds, "semantic")
    return {
        "operating_point": cards["semantic_operating"],
        "best_f1_sweep": cards["semantic_best_f1"],
    }


def run_weblog(
    data_path: Path,
    *,
    model_type: str = "isolation_forest",
    test_fraction: float = 0.3,
    anomaly_fraction: float = 0.05,
    seed: int = 42,
) -> Tuple[Benchmark, Dict[str, EvalResult]]:
    """End-to-end web-log baseline for a given model type.

    model_type: "isolation_forest" / "autoencoder" (Phase-1 baseline) or
    "semantic" (Phase-2 semantic model).
    """
    bench = build_weblog_benchmark(
        data_path,
        test_fraction=test_fraction,
        anomaly_fraction=anomaly_fraction,
        seed=seed,
    )
    logger.info(bench.summary())

    if model_type == "semantic":
        model = SemanticAnomalyModel().fit(bench.train_lines)
        results = evaluate_semantic(bench, model)
    else:
        detector = train_baseline_detector(bench.train_lines, model_type=model_type)
        results = evaluate_benchmark(bench, detector)
    return bench, results


def run_weblog_compare(
    data_path: Path,
    *,
    test_fraction: float = 0.3,
    anomaly_fraction: float = 0.05,
    seed: int = 42,
) -> Tuple[Benchmark, Dict[str, EvalResult]]:
    """Train baseline + semantic on the SAME benchmark and return one table.

    This is the honest Phase-1 vs Phase-2 comparison: identical train/test
    split, identical injected anomalies, only the model changes.
    """
    bench = build_weblog_benchmark(
        data_path,
        test_fraction=test_fraction,
        anomaly_fraction=anomaly_fraction,
        seed=seed,
    )
    logger.info(bench.summary())

    baseline = train_baseline_detector(bench.train_lines, model_type="isolation_forest")
    base_scores, base_preds = score_lines(baseline, bench.test_lines)

    semantic = SemanticAnomalyModel().fit(bench.train_lines)
    sem_scores, sem_preds = semantic.score(bench.test_lines)

    results: Dict[str, EvalResult] = {}
    results.update(_scorecards(bench, base_scores, base_preds, "baseline"))
    results.update(_scorecards(bench, sem_scores, sem_preds, "semantic"))
    return bench, results


def _seq_scorecards(
    y: np.ndarray, scores: np.ndarray, preds: np.ndarray, prefix: str
) -> Dict[str, EvalResult]:
    return {
        f"{prefix}_operating": evaluate_scores(
            y, scores, name=f"{prefix}:operating", y_pred=preds
        ),
        f"{prefix}_best_f1": evaluate_scores(y, scores, name=f"{prefix}:best-f1"),
    }


def run_sequence_synth(
    *,
    seed: int = 42,
    epochs: int = 30,
) -> Tuple[SequenceBenchmark, Dict[str, EvalResult]]:
    """Phase-3 sequence benchmark: DeepLog vs a bigram Markov baseline.

    Controlled synthetic sessions (known normal grammar + injected brute-force /
    scan / corruption). Both models train on normal sessions only.
    """
    bench = build_synthetic_sessions(seed=seed)
    logger.info(bench.summary())
    y = bench.test_labels

    # top_k must cover the normal grammar's branching factor (~6 here) but stay
    # well below the vocabulary, or every event lands in top-k and nothing is a
    # violation. This is DeepLog's key hyperparameter (the candidate set g).
    deeplog = DeepLogDetector(
        bench.vocab_size, {"epochs": epochs, "top_k": 6, "window": 10}
    ).fit(bench.train_sequences)
    dl_scores, dl_preds = deeplog.score(bench.test_sequences)

    quant = QuantitativeDetector(bench.vocab_size).fit(bench.train_sequences)
    q_scores, q_preds = quant.score(bench.test_sequences)

    markov = MarkovBigramBaseline(bench.vocab_size).fit(bench.train_sequences)
    mk_scores, mk_preds = markov.score(bench.test_sequences)

    # Fuse DeepLog (order/novel) + Quantitative (volume/burst) — the LogAnomaly
    # two-head idea. z-normalize each on TRAIN scores (never test = no leakage),
    # sum for ranking; union the operating predictions.
    dl_tr, _ = deeplog.score(bench.train_sequences)
    q_tr, _ = quant.score(bench.train_sequences)

    def _z(x, ref):
        return (np.asarray(x) - ref.mean()) / (ref.std() + 1e-9)

    fused_scores = _z(dl_scores, dl_tr) + _z(q_scores, q_tr)
    fused_preds = ((dl_preds == 1) | (q_preds == 1)).astype(int)

    results: Dict[str, EvalResult] = {}
    results.update(_seq_scorecards(y, mk_scores, mk_preds, "markov"))
    results.update(_seq_scorecards(y, dl_scores, dl_preds, "deeplog"))
    results.update(_seq_scorecards(y, q_scores, q_preds, "quantitative"))
    results.update(_seq_scorecards(y, fused_scores, fused_preds, "fused"))
    return bench, results


def run_sequence_hard(
    *,
    seed: int = 42,
    epochs: int = 40,
) -> Tuple[SequenceBenchmark, Dict[str, EvalResult]]:
    """Phase-4 higher-order benchmark: Transformer vs DeepLog vs Markov.

    The matched OPEN/CLOSE grammar has a long-range dependency a bigram cannot
    represent, so this is where the context models finally earn their keep over
    the Phase-3 bigram baseline.
    """
    bench = build_matched_sessions(seed=seed)
    logger.info(bench.summary())
    y = bench.test_labels

    markov = MarkovBigramBaseline(bench.vocab_size).fit(bench.train_sequences)
    mk_s, mk_p = markov.score(bench.test_sequences)

    # top_k=2 so not all three CLOSE_* fit the candidate set: a mismatched close
    # is only "allowed" if the model actually tracked which resource was opened.
    deeplog = DeepLogDetector(
        bench.vocab_size, {"epochs": epochs, "top_k": 2, "window": 8}
    ).fit(bench.train_sequences)
    dl_s, dl_p = deeplog.score(bench.test_sequences)

    xf = LogTransformerDetector(
        bench.vocab_size, {"epochs": epochs, "top_k": 2, "window": 8}
    ).fit(bench.train_sequences)
    xf_s, xf_p = xf.score(bench.test_sequences)

    results: Dict[str, EvalResult] = {}
    results.update(_seq_scorecards(y, mk_s, mk_p, "markov"))
    results.update(_seq_scorecards(y, dl_s, dl_p, "deeplog"))
    results.update(_seq_scorecards(y, xf_s, xf_p, "transformer"))
    return bench, results


def natural_ratio_estimate(
    result: EvalResult, p: float = 0.0293
) -> Dict[str, float]:
    """Project precision/F1 to the natural anomaly rate `p`.

    The test set caps the (cheap) normal class for tractable CPU scoring, so its
    base rate is not natural. Recall and FPR are base-rate independent, so the
    honest, literature-comparable precision/F1 at the true HDFS rate (2.93%)
    follow directly from them. This is an unbiased estimate, not a re-measurement.
    """
    r = result.recall
    fpr = result.false_positive_rate
    denom = r * p + fpr * (1 - p)
    prec = r * p / denom if denom > 0 else 0.0
    f1 = 2 * prec * r / (prec + r) if (prec + r) > 0 else 0.0
    return {"precision": round(prec, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def run_hdfs_deeplog(
    log_path: Path,
    label_path: Path,
    *,
    epochs: int = 40,
    top_k: int = 15,
    max_train_sessions: int = 10000,
    max_test_normal: int = 50000,
    seed: int = 42,
) -> Tuple[Dict[str, object], Dict[str, EvalResult]]:
    """Real HDFS benchmark (the canonical DeepLog setting).

    Sessions are grouped by block id (see load_hdfs_sessions) and encoded
    parser-free (hdfs_template). Following the DeepLog protocol, the model
    trains on a modest number of NORMAL sessions (the earliest ones — a
    leakage-free temporal choice) and is tested on held-out sessions. To stay
    tractable on CPU, the test set keeps ALL held-out anomalies plus a sample of
    normals (`max_test_normal`); this preserves recall exactly and only
    subsamples the easy negative class.

    Get the data from https://github.com/logpai/loghub (HDFS_v1): HDFS.log +
    anomaly_label.csv.
    """
    import random as _random

    # Memory-light streaming loader (handles the full 11M-line log).
    seqs, labels, vocab = stream_hdfs_sessions(
        Path(log_path), Path(label_path), hdfs_template
    )
    n = len(seqs)
    normal_idx = [i for i in range(n) if labels[i] == 0]
    anom_idx = [i for i in range(n) if labels[i] == 1]

    train_idx = normal_idx[:max_train_sessions]  # earliest normals
    train_set = set(train_idx)

    rng = _random.Random(seed)
    test_normal_pool = [i for i in normal_idx if i not in train_set]
    if len(test_normal_pool) > max_test_normal:
        test_normal_pool = rng.sample(test_normal_pool, max_test_normal)
    test_idx = sorted(test_normal_pool + anom_idx)  # keep every anomaly

    train_seqs = [seqs[i] for i in train_idx]
    test_seqs = [seqs[i] for i in test_idx]
    y = labels[test_idx]
    logger.info(
        f"HDFS: {n} sessions total; train={len(train_seqs)} normal, "
        f"test={len(test_seqs)} ({int(y.sum())} anomalies, {y.mean():.2%}), vocab={vocab}"
    )

    deeplog = DeepLogDetector(vocab, {"epochs": epochs, "top_k": top_k}).fit(train_seqs)
    dl_s, dl_p = deeplog.score(test_seqs)
    quant = QuantitativeDetector(vocab).fit(train_seqs)
    q_s, q_p = quant.score(test_seqs)

    dl_tr, _ = deeplog.score(train_seqs)
    q_tr, _ = quant.score(train_seqs)

    def _z(x, ref):
        return (np.asarray(x) - ref.mean()) / (ref.std() + 1e-9)

    fused_s = _z(dl_s, dl_tr) + _z(q_s, q_tr)
    fused_p = ((dl_p == 1) | (q_p == 1)).astype(int)

    results: Dict[str, EvalResult] = {}
    results.update(_seq_scorecards(y, dl_s, dl_p, "deeplog"))
    results.update(_seq_scorecards(y, q_s, q_p, "quantitative"))
    results.update(_seq_scorecards(y, fused_s, fused_p, "fused"))
    meta = {
        "dataset": "hdfs",
        "vocab": vocab,
        "train_sessions": len(train_seqs),
        "test_sessions": len(test_seqs),
        "test_anomalies": int(y.sum()),
        "test_anomaly_rate": round(float(y.mean()), 4),
        "top_k": top_k,
        # Literature-comparable estimates projected to the natural 2.93% rate.
        "deeplog_natural_2.93pct": natural_ratio_estimate(results["deeplog_operating"]),
        "fused_natural_2.93pct": natural_ratio_estimate(results["fused_operating"]),
    }
    return meta, results
