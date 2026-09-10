# Evaluation Harness (Phase 1)

Honest, model-agnostic benchmarking for the anomaly detector. This is the
foundation of the model roadmap (see `MODEL_ROADMAP.md`): no model change ships
without a benchmark number that beats the previous one.

## Why it exists

The detector had **no accuracy measurement** — no Precision/Recall/F1, no
labeled data. "Strong" was unprovable. This harness fixes that.

Core principle from the research (*How Far Are We?*, ICSE 2022): a **random**
train/test split leaks future data and inflates F1. Only a **temporal**
(chronological) split gives an honest number. The harness enforces temporal
splitting by default and warns loudly on random.

## What it measures

Given ground-truth labels and continuous anomaly scores (higher = more
anomalous), it reports:

- **PR-AUC** and **ROC-AUC** — threshold-free ranking quality (the real signal
  for an imbalanced detector).
- **Precision / Recall / F1** at two operating points:
  - `operating` — the model's own decisions (sklearn contamination threshold).
  - `best-f1` — the F1 the ranking could reach with an oracle threshold.
- Confusion matrix + false-positive rate.

The gap between `operating` and `best-f1` tells you whether to fix the *model*
or just the *threshold*.

## Datasets

### 1. Web-log benchmark (baseline for the current model)

The project's own Apache/Nginx logs, turned into a labeled set by injecting
synthetic attacks (path traversal, SQLi, XSS, command injection, scanner
probes, long paths, unusual methods, server-error bursts, brute force) into the
**future/test** window. Real traffic = normal (0), injected = anomaly (1).

Protocol: train on the past (presumed-normal), test on the future + injected —
exactly how an unsupervised detector runs in production.

### 2. Loghub HDFS (scaffold for future sequence models)

Loader for the public HDFS dataset, grouped into session windows by `block_id`.
The current point-wise IsolationForest cannot consume sessions; this is wired
for Phase 3+ (DeepLog / LogAnomaly / LogLLM). Download `HDFS.log` +
`anomaly_label.csv` from https://github.com/logpai/loghub (HDFS_v1).

## Usage

```bash
# Baseline the current IsolationForest on the project's own logs
python -m evaluation.run --dataset weblog --data data/sample_logs.log \
    --report reports/weblog_eval.md --json reports/weblog_eval.json

# Try the autoencoder instead
python -m evaluation.run --dataset weblog --model autoencoder

# Inspect an HDFS session dataset (scaffold)
python -m evaluation.run --dataset hdfs --hdfs-log HDFS.log --hdfs-label anomaly_label.csv
```

Flags: `--test-fraction` (default 0.3), `--anomaly-fraction` (default 0.05),
`--seed` (default 42).

## Programmatic use

```python
from evaluation.metrics import evaluate_scores
import numpy as np

y_true = np.array([0, 0, 1, 1])
y_score = np.array([0.1, 0.3, 0.7, 0.9])  # higher = more anomalous
result = evaluate_scores(y_true, y_score, name="my_model")
print(result.as_dict())
```

Any future model plugs in the same way: produce `(y_true, y_score)`, call
`evaluate_scores`, compare F1 in the same table.

## Tests

```bash
pytest tests/test_evaluation.py -v
```

Core tests (metrics, splits, synthetic injection, benchmark build) need only
numpy + scikit-learn. The end-to-end baseline test is skipped automatically if
the full app stack (torch, drain3) isn't installed.

## Module map

| File | Role |
|------|------|
| `metrics.py` | Labels + scores → scorecard (PR-AUC, F1, sweep). Model-agnostic. |
| `splits.py` | Temporal (honest) vs random (leakage demo) split. |
| `synthetic.py` | Labeled attack injection in Apache log format. |
| `datasets.py` | WebLogBenchmark + Loghub HDFS session loader. |
| `runner.py` | Wires the existing detector: train → score → metrics. |
| `report.py` | Console / Markdown / JSON output. |
| `run.py` | CLI (`python -m evaluation.run`). |
