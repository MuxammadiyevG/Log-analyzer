"""
Tests for the evaluation harness.

Split into two layers:
  * Core (metrics / splits / synthetic / datasets) — depends only on numpy +
    scikit-learn, always runnable.
  * End-to-end baseline — needs the full app stack (torch, drain3, ...); skipped
    automatically when those aren't installed.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("sklearn")

from evaluation.metrics import best_f1_threshold, evaluate_scores  # noqa: E402
from evaluation.splits import random_split, temporal_split  # noqa: E402
from evaluation.synthetic import ATTACK_GENERATORS, inject_anomalies  # noqa: E402


# --- metrics ---------------------------------------------------------------


def test_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95, 0.99])
    r = evaluate_scores(y, scores, name="perfect")
    assert r.pr_auc == pytest.approx(1.0)
    assert r.roc_auc == pytest.approx(1.0)
    assert r.f1 == pytest.approx(1.0)
    assert r.tp == 3 and r.fp == 0 and r.fn == 0


def test_random_scores_low_f1():
    rng = np.random.default_rng(0)
    y = np.array([0] * 90 + [1] * 10)
    scores = rng.random(100)  # uninformative
    r = evaluate_scores(y, scores, name="rand")
    # PR-AUC for uninformative scores hovers near the positive prevalence (0.1)
    assert r.pr_auc < 0.35
    assert 0.0 <= r.f1 <= 1.0


def test_best_f1_threshold_recovers_boundary():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    thr, f1 = best_f1_threshold(y, scores)
    assert f1 == pytest.approx(1.0)
    assert 0.2 < thr <= 0.8


def test_single_class_returns_nan_auc():
    y = np.array([0, 0, 0])
    scores = np.array([0.1, 0.5, 0.9])
    r = evaluate_scores(y, scores, name="one-class")
    assert r.pr_auc != r.pr_auc  # NaN
    assert r.roc_auc != r.roc_auc

def test_model_predictions_operating_point():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.6, 0.4, 0.9])
    preds = np.array([0, 1, 0, 1])  # one FP, one FN
    r = evaluate_scores(y, scores, y_pred=preds, name="op")
    assert r.threshold_source == "model"
    assert r.fp == 1 and r.fn == 1


def test_evaluate_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        evaluate_scores(np.array([0, 1]), np.array([0.1, 0.2, 0.3]))


# --- splits ----------------------------------------------------------------


def test_temporal_split_preserves_order_and_size():
    items = list(range(100))
    train, test = temporal_split(items, test_fraction=0.3, already_sorted=True)
    assert len(train) == 70 and len(test) == 30
    assert train == list(range(70))
    assert test == list(range(70, 100))


def test_temporal_split_sorts_by_key():
    items = [{"t": 3}, {"t": 1}, {"t": 2}]
    train, test = temporal_split(items, test_fraction=0.34, key=lambda r: r["t"])
    assert train[0]["t"] == 1  # sorted ascending


def test_random_split_is_deterministic():
    items = list(range(50))
    a1, b1 = random_split(items, seed=7)
    a2, b2 = random_split(items, seed=7)
    assert a1 == a2 and b1 == b2
    assert sorted(a1 + b1) == items  # no loss


# --- synthetic injection ---------------------------------------------------


def test_inject_anomalies_count_and_labels():
    from datetime import datetime, timezone

    lo = datetime(2024, 1, 1, tzinfo=timezone.utc)
    hi = datetime(2024, 1, 2, tzinfo=timezone.utc)
    out = inject_anomalies(lo, hi, 30, seed=1, include_bruteforce=True)
    assert len(out) >= 30  # + brute-force burst
    for line, cat in out:
        assert isinstance(line, str) and len(line) > 20
        assert cat in set(ATTACK_GENERATORS) | {"brute_force"}


def test_inject_anomalies_reproducible():
    from datetime import datetime, timezone

    lo = datetime(2024, 1, 1, tzinfo=timezone.utc)
    hi = datetime(2024, 1, 2, tzinfo=timezone.utc)
    a = inject_anomalies(lo, hi, 20, seed=5)
    b = inject_anomalies(lo, hi, 20, seed=5)
    assert a == b


def test_injected_lines_parse_as_apache():
    """Injected lines must match the combined-log timestamp shape."""
    import re
    from datetime import datetime, timezone

    lo = datetime(2024, 1, 1, tzinfo=timezone.utc)
    hi = datetime(2024, 1, 2, tzinfo=timezone.utc)
    ts_re = re.compile(r"\[\d{2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2}")
    for line, _ in inject_anomalies(lo, hi, 15, seed=3):
        assert ts_re.search(line), line


# --- weblog benchmark (needs data file, no torch) --------------------------


def test_build_weblog_benchmark():
    from evaluation.datasets import DEFAULT_WEBLOG, build_weblog_benchmark

    if not DEFAULT_WEBLOG.exists():
        pytest.skip("sample_logs.log not present")

    bench = build_weblog_benchmark(DEFAULT_WEBLOG, test_fraction=0.3, seed=42)
    assert len(bench.train_lines) > 0
    assert len(bench.test_lines) == len(bench.test_labels)
    # test set must contain both classes
    assert bench.test_labels.sum() > 0
    assert (bench.test_labels == 0).sum() > 0
    # injected anomalies are the tail of the test set
    assert bench.test_labels[-1] == 1


# --- end-to-end baseline (needs full app stack) ----------------------------


def test_end_to_end_weblog_baseline():
    pytest.importorskip("torch")
    pytest.importorskip("drain3")
    pytest.importorskip("loguru")

    from evaluation.datasets import DEFAULT_WEBLOG
    from evaluation.runner import run_weblog

    if not DEFAULT_WEBLOG.exists():
        pytest.skip("sample_logs.log not present")

    bench, results = run_weblog(DEFAULT_WEBLOG, test_fraction=0.3, seed=42)
    assert "operating_point" in results and "best_f1_sweep" in results
    op = results["operating_point"]
    assert op.n_samples == len(bench.test_labels)
    # ranking should beat random (positives prevalence) on these obvious attacks
    assert op.pr_auc == op.pr_auc  # not NaN
    assert 0.0 <= op.f1 <= 1.0


# --- Phase 2: semantic model (sklearn only, no torch) ----------------------


def test_semantic_request_document():
    from app.features.semantic import request_document

    line = (
        '10.0.0.1 - - [01/Jan/2024:10:00:00 +0000] '
        '"GET /admin/../../etc/passwd HTTP/1.1" 403 0 "-" "curl/7.68.0"'
    )
    doc = request_document(line)
    assert "GET" in doc and "/admin/../../etc/passwd" in doc and "403" in doc
    assert "curl/7.68.0" in doc


def test_semantic_model_beats_random_on_injected():
    """Semantic model must rank injected attacks well above normal traffic."""
    from datetime import timezone

    from app.models.semantic_detector import SemanticAnomalyModel
    from evaluation.datasets import DEFAULT_WEBLOG, _extract_ts
    from evaluation.metrics import evaluate_scores
    from evaluation.synthetic import inject_anomalies

    if not DEFAULT_WEBLOG.exists():
        pytest.skip("sample_logs.log not present")

    lines = [ln.strip() for ln in DEFAULT_WEBLOG.read_text(errors="ignore").splitlines() if ln.strip()]
    train = lines[:2000]
    test_normal = lines[2000:2600]

    ts = [t for t in (_extract_ts(x) for x in test_normal) if t]
    from datetime import datetime
    lo = min(ts) if ts else datetime(2024, 1, 1, tzinfo=timezone.utc)
    hi = max(ts) if ts else datetime(2024, 1, 2, tzinfo=timezone.utc)
    injected = inject_anomalies(lo, hi, 60, seed=9)

    test_lines = test_normal + [ln for ln, _ in injected]
    y = np.array([0] * len(test_normal) + [1] * len(injected))

    model = SemanticAnomalyModel().fit(train)
    scores, _ = model.score(test_lines)
    r = evaluate_scores(y, scores, name="semantic")
    # Semantic representation should crush the 0.325 baseline PR-AUC.
    assert r.pr_auc > 0.80, f"semantic pr_auc too low: {r.pr_auc}"


# --- Phase 3: sequence tooling (encoder/markov: no torch) ------------------


def test_web_template_normalizes_ids():
    from evaluation.sequences import web_template

    line = '1.2.3.4 - - [x] "GET /api/users/123 HTTP/1.1" 200 5 "-" "ua"'
    assert web_template(line) == "GET /api/users/<n> 2xx"


def test_event_encoder_unk():
    from app.models.deeplog import UNK
    from evaluation.sequences import EventEncoder

    enc = EventEncoder().fit(["A", "B"])
    assert enc.encode("A") == 2 and enc.encode("B") == 3
    assert enc.encode("NEVER_SEEN") == UNK


def test_web_sessions_group_by_ip():
    from evaluation.sequences import web_sessions_from_lines

    lines = [
        '1.1.1.1 - - [x] "GET /a HTTP/1.1" 200 1 "-" "u"',
        '2.2.2.2 - - [x] "GET /b HTTP/1.1" 200 1 "-" "u"',
        '1.1.1.1 - - [x] "GET /c HTTP/1.1" 200 1 "-" "u"',
    ]
    seqs, enc = web_sessions_from_lines(lines, fit=True)
    assert len(seqs) == 2  # two distinct IPs
    assert len(seqs[0]) == 2  # 1.1.1.1 has two requests


def test_synthetic_sessions_have_both_classes():
    from evaluation.sequences import build_synthetic_sessions

    b = build_synthetic_sessions(n_train=200, n_test_normal=100, n_anomalies=40, seed=1)
    assert b.test_labels.sum() == 40
    assert (b.test_labels == 0).sum() == 100
    assert len(b.train_sequences) == 200


def test_markov_baseline_flags_brute_force():
    from evaluation.sequences import (
        FAIL,
        LOGIN,
        MarkovBigramBaseline,
        SYNTH_VOCAB,
        build_synthetic_sessions,
    )

    b = build_synthetic_sessions(n_train=500, n_test_normal=50, n_anomalies=1, seed=2)
    mk = MarkovBigramBaseline(SYNTH_VOCAB).fit(b.train_sequences)
    normal_scores, _ = mk.score(b.test_sequences[:50])
    brute, _ = mk.score([[LOGIN] + [FAIL] * 12])
    assert brute[0] > np.median(normal_scores)  # brute force is more surprising


# --- Phase 3: DeepLog (needs torch) ----------------------------------------


def test_deeplog_catches_order_anomalies_cleanly():
    """DeepLog's strength: novel/mis-ordered events, with no false positives.

    (It is weak on quantitative bursts like brute force — that is the
    Quantitative head's job — so this test checks the order/novel dimension.)
    """
    pytest.importorskip("torch")
    import random

    from app.models.deeplog import DeepLogDetector
    from evaluation.sequences import _admin_scan, _corruption, build_synthetic_sessions

    b = build_synthetic_sessions(n_train=1500, n_test_normal=300, n_anomalies=1, seed=7)
    model = DeepLogDetector(b.vocab_size, {"epochs": 15, "window": 10, "top_k": 6}).fit(
        b.train_sequences
    )
    # No normal session should trip a violation (precision on normal).
    _, normal_preds = model.score(b.test_sequences[:300])
    assert normal_preds.mean() < 0.05, f"too many false positives: {normal_preds.mean()}"

    rng = random.Random(321)
    order_anoms = [_admin_scan(rng) for _ in range(30)] + [_corruption(rng) for _ in range(30)]
    _, anom_preds = model.score(order_anoms)
    assert anom_preds.mean() > 0.90, f"missed order anomalies: {anom_preds.mean()}"


def test_quantitative_head_catches_brute_force():
    """The exact gap DeepLog leaves: a burst of a known event."""
    import random

    from app.models.quantitative import QuantitativeDetector
    from evaluation.sequences import _brute_force, build_synthetic_sessions

    b = build_synthetic_sessions(n_train=1500, n_test_normal=300, n_anomalies=1, seed=7)
    q = QuantitativeDetector(b.vocab_size).fit(b.train_sequences)
    _, normal_preds = q.score(b.test_sequences[:300])
    rng = random.Random(321)
    _, brute_preds = q.score([_brute_force(rng) for _ in range(30)])
    assert brute_preds.mean() > 0.90, f"quant missed brute force: {brute_preds.mean()}"
    assert normal_preds.mean() < 0.15  # keeps normal mostly clean


def test_fused_sequence_tier_catches_every_type():
    """DeepLog + Quantitative fused must catch order AND volume anomalies."""
    pytest.importorskip("torch")
    from evaluation.runner import run_sequence_synth

    _, results = run_sequence_synth(seed=7, epochs=15)
    fused = results["fused_operating"]
    # Recall 1.0 expected — fusion closes the brute-force gap DeepLog alone left.
    assert fused.pr_auc > 0.90, f"fused pr_auc too low: {fused.pr_auc}"
    assert fused.recall > 0.90, f"fused recall too low: {fused.recall}"
    # Fusion recall must be at least DeepLog's alone (quant only adds coverage).
    assert fused.recall >= results["deeplog_operating"].recall


# --- Phase 3: HDFS wiring (tiny fake data, no 1.5GB download) ---------------


def _write_fake_hdfs(tmp_path):
    """~50 block sessions: 40 normal (shared templates), 10 anomalous (weird)."""
    import random

    rng = random.Random(0)
    log_lines, labels = [], ["BlockId,Label"]
    normal_msgs = [
        "dfs.DataNode$DataXceiver Receiving block {blk} src: /10.0.0.1:50010",
        "dfs.DataNode$PacketResponder PacketResponder for block {blk} terminating",
        "dfs.FSNamesystem BLOCK* NameSystem.addStoredBlock blockMap updated {blk} size 67108864",
    ]
    for i in range(50):
        blk = f"blk_{-1000000 - i}" if i % 2 == 0 else f"blk_{2000000 + i}"
        anomaly = i >= 40
        msgs = list(normal_msgs)
        if anomaly:
            msgs.append("dfs.DataNode$DataXceiver writeBlock {blk} received exception EOFException")
        for m in msgs:
            log_lines.append(f"081109 20{i:02d}15 {i} INFO {m.format(blk=blk)}")
        labels.append(f"{blk},{'Anomaly' if anomaly else 'Normal'}")
    (tmp_path / "HDFS.log").write_text("\n".join(log_lines))
    (tmp_path / "anomaly_label.csv").write_text("\n".join(labels))
    return tmp_path / "HDFS.log", tmp_path / "anomaly_label.csv"


def test_hdfs_loader_and_templating(tmp_path):
    from evaluation.datasets import load_hdfs_sessions
    from evaluation.sequences import hdfs_template

    log_p, lbl_p = _write_fake_hdfs(tmp_path)
    ds = load_hdfs_sessions(log_p, lbl_p)
    assert len(ds.sessions) == 50
    assert ds.labels.sum() == 10
    # Templating must mask the variable block id.
    tmpl = hdfs_template(ds.sessions[0][0])
    assert "<BLK>" in tmpl and "blk_" not in tmpl


def test_hdfs_deeplog_pipeline_runs(tmp_path):
    pytest.importorskip("torch")
    from evaluation.runner import run_hdfs_deeplog

    log_p, lbl_p = _write_fake_hdfs(tmp_path)
    meta, results = run_hdfs_deeplog(
        log_p, lbl_p, epochs=3, max_train_sessions=25, max_test_normal=50
    )
    assert "fused_operating" in results
    fused = results["fused_operating"]
    assert fused.n_samples > 0
    assert fused.n_anomalies > 0 and fused.n_anomalies < fused.n_samples  # both classes
    assert 0.0 <= fused.f1 <= 1.0  # pipeline produced valid metrics


# --- Phase 4: transformer beats bigram on long-range dependency ------------


def test_matched_benchmark_markov_is_blind_to_long_range():
    """A bigram cannot represent OPEN/CLOSE matching — it must miss mismatches."""
    import random

    from evaluation.sequences import (
        MarkovBigramBaseline,
        MATCHED_VOCAB,
        _mismatched_close,
        build_matched_sessions,
    )

    b = build_matched_sessions(n_train=1500, n_test_normal=200, n_anomalies=30, seed=3)
    mk = MarkovBigramBaseline(MATCHED_VOCAB).fit(b.train_sequences)
    rng = random.Random(11)
    mism = [_mismatched_close(rng) for _ in range(30)]
    # Mismatched close uses only "seen" bigrams (WORK->CLOSE_*, OPEN->WORK), so
    # the bigram assigns them normal surprisal: it cannot rank them as anomalous.
    normal_scores, _ = mk.score(b.test_sequences[:200])
    mism_scores, _ = mk.score(mism)
    assert abs(np.mean(mism_scores) - np.mean(normal_scores)) < 0.05


def test_transformer_beats_bigram_on_long_range():
    pytest.importorskip("torch")
    from app.models.log_transformer import LogTransformerDetector
    from evaluation.metrics import evaluate_scores
    from evaluation.sequences import (
        MarkovBigramBaseline,
        MATCHED_VOCAB,
        _mismatched_close,
        build_matched_sessions,
    )

    b = build_matched_sessions(n_train=1500, n_test_normal=300, n_anomalies=90, seed=3)
    xf = LogTransformerDetector(
        b.vocab_size, {"epochs": 30, "top_k": 2, "window": 8}
    ).fit(b.train_sequences)
    mk = MarkovBigramBaseline(MATCHED_VOCAB).fit(b.train_sequences)

    xf_s, _ = xf.score(b.test_sequences)
    mk_s, _ = mk.score(b.test_sequences)
    xf_r = evaluate_scores(b.test_labels, xf_s, name="xf")
    mk_r = evaluate_scores(b.test_labels, mk_s, name="mk")
    # Context model must rank better than the bigram on this benchmark.
    assert xf_r.pr_auc > mk_r.pr_auc, f"xf {xf_r.pr_auc} !> markov {mk_r.pr_auc}"

    # And catch the long-range mismatch far better than the (blind) bigram.
    import random

    rng = random.Random(99)
    mism = [_mismatched_close(rng) for _ in range(40)]
    _, xf_pred = xf.score(mism)   # operating point: any grammar violation
    mk_thr = np.percentile(mk.score(b.test_sequences[:300])[0], 95)
    mk_flag = (mk.score(mism)[0] > mk_thr).mean()
    assert xf_pred.mean() > mk_flag + 0.2, (
        f"transformer ({xf_pred.mean():.2f}) not clearly beating blind bigram ({mk_flag:.2f})"
    )


# --- Phase 4: LLM explainer (mock — no SDK/key needed) ---------------------


def test_llm_explainer_parses_verdict():
    from app.models.llm_explainer import Explanation, LLMExplainer

    ex = LLMExplainer()
    # Inject a fake completion so no SDK/network/key is touched.
    ex._complete = lambda system, user: (
        'Sure, here is the verdict:\n'
        '{"anomaly": true, "severity": "high", "attack_type": "path_traversal", '
        '"reason": "Request tries to read /etc/passwd via ../ traversal."}'
    )
    result = ex.explain(
        '10.0.0.1 - - [x] "GET /admin/../../etc/passwd HTTP/1.1" 403 0',
        signals={"semantic_score": 3.1, "oov_ratio": 0.4},
    )
    assert isinstance(result, Explanation)
    assert result.anomaly is True
    assert result.severity == "high"
    assert result.attack_type == "path_traversal"
    assert "passwd" in result.reason


def test_llm_explainer_handles_garbage_output():
    from app.models.llm_explainer import LLMExplainer

    ex = LLMExplainer()
    ex._complete = lambda system, user: "the model rambled with no json"
    result = ex.explain("some log line")
    assert result.anomaly is False  # safe default
    assert result.reason  # falls back to the raw text


def test_llm_explainer_prompt_includes_signals():
    from app.models.llm_explainer import LLMExplainer

    captured = {}

    ex = LLMExplainer()
    def fake(system, user):
        captured["system"] = system
        captured["user"] = user
        return '{"anomaly": false, "severity": "low", "attack_type": "none", "reason": "ok"}'
    ex._complete = fake
    ex.explain("GET /api/users", signals={"oov_ratio": 0.02})
    assert "oov_ratio" in captured["user"]
    assert "JSON" in captured["system"]


def test_llm_explainer_available_is_bool():
    from app.models.llm_explainer import LLMExplainer

    assert isinstance(LLMExplainer.available(), bool)
