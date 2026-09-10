# Phase 3 — Sequence Tier (DeepLog + Quantitative)

Phases 1–2 score each line in isolation, so they are blind to anomalies that
live in **order and volume**: a brute-force burst is a stream of individually
normal `POST /login 401` lines. Phase 3 adds a sequence tier.

## Models

| Model | File | Detects | Idea |
|-------|------|---------|------|
| **DeepLog** | `app/models/deeplog.py` | Order / novel events | LSTM predicts the next event id from a sliding window; if the real next event is outside the top-k prediction, the grammar was violated. Trained on normal only. |
| **Quantitative** | `app/models/quantitative.py` | Volume / bursts | IsolationForest over per-session event-count vectors (+ length, max run-length). A `FAIL`×12 burst has a count vector far from normal. |
| **Fused** | `evaluation/runner.py` | Both | z-normalize each score on **train** stats, sum; union the operating predictions. This is LogAnomaly's two-head idea. |
| Markov bigram | `evaluation/sequences.py` | (baseline) | Transition surprisal — a fair, cheap sequence baseline. |

Why two heads: DeepLog is strong on order/novelty but weak on repeated **known**
events (the well-documented "quantitative" blind spot). The Quantitative head
covers exactly that gap. Neither alone is enough; fused, they cover both axes.

## Results — synthetic session benchmark

Controlled sessions with a known normal grammar + injected brute-force / scan /
corruption anomalies. Both models train on normal sessions only.

Run: `python -m evaluation.run --dataset seqsynth`

| model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|-------|--------|---------|-----------|--------|----|
| Markov bigram (best-F1) | 1.000 | 1.000 | 0.980 | 0.993 | 0.997 |
| DeepLog | 0.931 | 0.957 | 1.000 | 0.913 | 0.955 |
| Quantitative | 0.712 | 0.924 | 0.696 | 0.780 | 0.736 |
| **Fused (best-F1)** | **0.983** | **0.988** | 0.954 | 0.973 | **0.964** |

- DeepLog: **zero false positives**, catches all novel/mis-ordered anomalies,
  misses some brute-force bursts (its known blind spot).
- Quantitative: catches the brute-force bursts DeepLog misses.
- **Fused: F1 0.96, recall 0.97** — catches every anomaly type, closing the
  brute-force gap that Phases 1–2 could not touch.

### Honest caveat

The Markov bigram scores ~1.0 here **because the synthetic grammar is a pure
first-order process** — a bigram is near-optimal on it by construction. This
benchmark proves the sequence tier *works* and closes the brute-force gap; it
does **not** show DeepLog's advantage over a bigram, which only appears with
long-range dependencies. For that, run the real HDFS benchmark (below), where
DeepLog reaches ~0.96 F1 in the literature and a bigram does not.

## Real HDFS benchmark (the canonical DeepLog setting)

Wired and tested (`run_hdfs_deeplog`), runs when you supply the data:

```bash
# Get HDFS.log + anomaly_label.csv from https://github.com/logpai/loghub (HDFS_v1)
python -m evaluation.run --dataset hdfs --model deeplog \
    --hdfs-log HDFS.log --hdfs-label anomaly_label.csv \
    --report reports/hdfs.md
```

Sessions are grouped by `block_id`, split chronologically, and encoded
parser-free (`hdfs_template` masks block ids / IPs / numbers — the NeuralLog /
LogLLM approach). The pipeline is verified end-to-end on tiny fake HDFS data in
the test suite, so it is ready for the full download.

## Where this sits

Phase 3 delivers **Tier 2** of the roadmap's tiered architecture. Point/semantic
(Tier 1, Phases 1–2) filter fast; the sequence tier catches order/volume
anomalies they miss. Next: Tier 3 (transformer / LLM) for peak accuracy on the
hardest cases.
