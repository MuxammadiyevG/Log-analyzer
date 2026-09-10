# Phase 2 — Semantic Ensemble Model

Second step of the roadmap: give the detector *eyes*. The Phase-1 baseline saw
only 19 hand-crafted scalars and threw the request text away. Phase 2 keeps the
text and fuses three complementary unsupervised signals.

## The model (`app/models/semantic_detector.py`)

`SemanticAnomalyModel` — trained on presumed-normal traffic only (same protocol
as the baseline), scores are a z-normalized sum of three views:

| Signal | Catches | How |
|--------|---------|-----|
| **OOV n-gram ratio** | Injection payloads / novel tokens (`../`, `union`, `<script`, `.env`, `sqlmap`) | Fraction of char n-grams unseen in normal training. TF-IDF `transform` *drops* these, so we count them directly. |
| **IsolationForest(SVD)** | Odd-but-in-vocab requests | IF over TruncatedSVD(64) of char-`(3,5)`-gram TF-IDF. |
| **IsolationForest(numeric)** | Rare status, unusual method, long/weird path | IF over structural features (`app/features/semantic.py::numeric_features`). |

Each signal is z-normalized using **train** statistics (never test — that would
leak), then summed. Operating threshold = the `(1 - contamination)` percentile
of the training blend.

Rationale from the research (LogRobust / NeuralLog): character-level semantics
survive log mutation and parsing errors far better than templates or single
scalars. No single view is enough — novelty misses categorical anomalies, and
structure misses payloads — so they are fused.

## Results (honest, temporal split, identical benchmark)

Run: `python -m evaluation.run --compare`

| scorecard | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| baseline operating (Phase 1) | 0.325 | 0.950 | 0.217 | 1.000 | 0.356 |
| baseline best-F1 | 0.325 | 0.950 | 0.454 | 0.945 | 0.613 |
| **semantic operating (Phase 2)** | **0.691** | **0.973** | 0.461 | 0.935 | **0.617** |
| **semantic best-F1** | **0.691** | **0.973** | 0.903 | 0.508 | 0.650 |

**PR-AUC 2.1×** (0.325 → 0.691); operating-point F1 **+73%** (0.356 → 0.617), and
the model can reach **90% precision** at the best-F1 point vs 45% for the
baseline — far fewer false alarms.

## Where it still loses (→ Phase 3)

- **Brute force**: 12 identical `POST /api/login 401` lines are each
  individually normal. Point-wise scoring cannot catch them — this needs
  **sequence / rate** modeling (DeepLog, time-window counts). Phase 3.
- **Real embedded attacks mislabeled normal**: the raw `sample_logs.log`
  contains genuine attack lines (e.g. `/admin/../../etc/passwd`) labeled 0
  because they are real traffic. The semantic model correctly flags them, which
  counts against precision here. A cleaner labeled corpus (or Loghub) removes
  this artifact.

## Usage

```bash
# Semantic model alone
python -m evaluation.run --model semantic --report reports/semantic.md

# Head-to-head with the baseline on one benchmark
python -m evaluation.run --compare --report reports/compare_p1_p2.md
```
