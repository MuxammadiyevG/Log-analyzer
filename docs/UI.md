# Web UI — upload logs, get results (any format)

A browser UI served by the FastAPI app: drop a log file or paste lines, get a
ranked list of anomalies with a reason. Works on **any line-oriented log**.

## Run

```bash
./venv/bin/python -m uvicorn app.main:app --port 8000
# open http://localhost:8000
```

The UI is served at `/`; the JSON API lives under `/api/v1` (docs at `/docs`).
No Claude, no torch — scikit-learn only.

## What it does

1. **Detects the format** of each line — web (Apache/Nginx combined), syslog
   (RFC 3164 / 5424), JSON / Elastic Common Schema (ECS), or `generic` for
   anything else.
2. **Scores every line** with three combined signals:
   - **signature** — deterministic attack patterns (path traversal, SQLi, XSS,
     command injection, scanner probes, scanner user-agents);
   - **severity** — normalized badness from the log level / HTTP status, plus
     error-keyword density (`failed`, `denied`, `exception`, `out of memory`, …)
     for formats without an explicit level;
   - **statistical** — an IsolationForest outlier model **fit on the uploaded
     batch itself** (char-TF-IDF + generic features), so it needs no
     pre-training and finds "what's unusual within these logs". Runs only when
     the batch has ≥ 30 lines; smaller pastes use signatures + severity alone.
3. Returns a summary (line count, anomaly count/rate, formats detected, whether
   the statistical stage ran) and a per-line table: severity badge, score,
   format, reason, and the raw line — anomalies first.

## Why fit-on-upload

There is no labeled "normal" corpus for arbitrary customer logs. Fitting the
outlier model on each uploaded batch treats the bulk as normal and surfaces the
outliers — the honest, format-agnostic way to answer "find the anomalies in
*this* log", while the signature + severity rules catch known-bad lines
regardless of batch size or how common they are.

## Endpoints

- `POST /api/v1/analyze/text` — body `{"text": "<log lines>"}`
- `POST /api/v1/analyze/file` — multipart upload (`file`)

Both return: `{total, anomalies, anomaly_rate, formats, statistical, results[]}`.

## Code

| File | Role |
|------|------|
| `app/parsers/formats.py` | Format detect + parse + severity + generic features + explanation. |
| `app/models/multiformat_detector.py` | Fit-on-batch analyzer (signature + severity + statistical). |
| `app/static/index.html` | The UI (upload/paste, summary, results table). |
| `app/api/routes.py` | `/analyze/text` and `/analyze/file`. |

## Scope / honesty

- Known formats get proper structured parsing; unknown formats fall back to
  `generic` (still scored via char n-grams + generic features + signatures).
- The statistical stage assumes the batch is mostly normal (standard outlier
  assumption). A batch that is *mostly* anomalous will under-flag statistically
  — signatures/severity still fire.
- This is point-in-time, per-line detection. Sequence/rate anomalies (brute
  force across many lines) are the sequence tier's job (`docs/PHASE3_SEQUENCE.md`),
  not wired into this UI yet.
