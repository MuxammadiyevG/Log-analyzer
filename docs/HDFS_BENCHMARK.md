# Real HDFS Benchmark (Loghub HDFS_v1)

The sequence tier (Phase 3) run on the **real, full HDFS dataset** — the
canonical DeepLog benchmark — not a synthetic stand-in.

## Setup

- **Data**: Loghub HDFS_v1 (`HDFS.log`, 11,175,629 lines, 1.5 GB) +
  `anomaly_label.csv`. Downloaded from Zenodo (record 3227177).
- **Parsing**: Drain3 template mining → **47 event types** (canonical HDFS set;
  parser-free regex was rejected — it left 1600+ spurious templates in
  exception/path text).
- **Sessions**: grouped by `block_id` → **575,061 sessions**, 16,838 anomalous
  (2.93%).
- **Protocol** (DeepLog, unsupervised, leakage-free): train on the earliest
  **10,000 normal** sessions; test on held-out sessions.
- **Test set**: all **16,838 anomalies** + a uniform sample of **50,000
  normals** (66,838 total). Capping the cheap normal class keeps CPU scoring
  tractable; recall and FPR are base-rate independent, so precision/F1 are
  **projected to the true 2.93% rate** (an unbiased estimate, not a
  re-measurement).

Reproduce:

```bash
python -m evaluation.run --dataset hdfs --model deeplog \
    --hdfs-log HDFS.log --hdfs-label anomaly_label.csv \
    --report reports/hdfs_deeplog.md
```

## Results

| model | precision | recall | F1 | FPR |
|-------|-----------|--------|----|-----|
| DeepLog (measured, 25% test ratio) | 0.992 | 0.673 | 0.802 | 0.0018 |
| **DeepLog (projected to natural 2.93%)** | **0.919** | **0.673** | **0.777** | 0.0018 |
| Quantitative head | 0.24 | 0.76 | 0.37 | 0.80 |

**DeepLog reaches F1 ≈ 0.78 at the natural rate — precision 0.92, recall 0.67,
FPR 0.18%.** It flags almost no normal session (FPR 0.0018) while catching two
thirds of anomalies.

## Honest gap vs the literature (~0.96 F1)

The published DeepLog HDFS number is ~0.96 F1; this run reaches ~0.78. The gap is
entirely **recall** (0.67 vs ~0.96), for three concrete reasons:

1. **DeepLog's quantitative blind spot.** A `top_k=6` sweep (flagging ~70% of
   *everything*) still tops out at **recall ≈ 0.80** — so ~20–30% of HDFS
   anomalies contain no out-of-order event at all. They are **count/volume**
   anomalies that next-event prediction structurally cannot see. Closing this
   needs a real quantitative model (LogAnomaly's count-LSTM). Our simple
   IsolationForest count head is too noisy on HDFS (FPR 0.80) to fuse usefully —
   union fusion floods false positives, so it is not recommended here.
2. **Training size capped by RAM.** This machine OOM-killed a 40k-session
   training run; the model was trained on 10k sessions. More normal data
   calibrates the per-event model better (lower FPR at lower `top_k` → higher
   recall at fixed precision).
3. **Reproduction variance.** Independent studies ("How Far Are We?", ICSE 2022)
   report the canonical 0.96 needs careful tuning and a validation-tuned
   threshold, and that naive reproductions land lower — DeepLog recall in the
   0.6–0.9 range is commonly reported.

This is a legitimate, honestly-measured real-world result, not the paper's
best-case. The `top_k` operating point is tunable: `top_k=18` gives precision
0.96 at recall 0.57; `top_k=13` gives recall 0.71 at precision 0.66 — pick per
the cost of a missed anomaly vs a false alarm.

## To push toward the literature number (needs more resources)

- Train on ≫10k normal sessions (needs more RAM/GPU).
- Add a proper quantitative-sequence model (LogAnomaly count-LSTM), not a plain
  count IsolationForest, and fuse.
- Or go supervised (LogRobust) if labels are available — the literature's top
  HDFS numbers are supervised.
