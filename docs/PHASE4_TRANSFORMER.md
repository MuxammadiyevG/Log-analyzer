# Phase 4 — Transformer Tier + LLM Explainability

Tier 3 of the roadmap: peak accuracy on the hardest sequences, plus a
human-readable explanation for the on-call engineer.

## 1. LogTransformer — self-attention sequence model

`app/models/log_transformer.py`. Same detection contract as DeepLog (next event
outside top-k ⇒ violation), but self-attention can look at any earlier position
directly, so it captures **long-range dependencies** an LSTM window or a bigram
cannot. This is the CPU-trainable stand-in for the transformer tier (LogBERT /
NeuralLog family); full LogLLM (BERT+Llama+QLoRA) needs a GPU and is out of scope
for this environment.

### The honest test: a benchmark a bigram cannot solve

Phase 3 noted that on a first-order grammar a bigram baseline is near-optimal, so
it did not show a deep model's advantage. Phase 4 fixes that with a **long-range
benchmark** (`build_matched_sessions`): a session opens a resource and must
CLOSE the *same* one several steps later. Local transitions (`WORK→CLOSE_*`) are
all "seen", so a bigram assigns a mismatched close normal surprisal — it is
structurally blind to the dependency.

Run: `python -m evaluation.run --dataset seqmatched`

| model | PR-AUC | ROC-AUC | best-F1 | catches mismatch? |
|-------|--------|---------|---------|-------------------|
| Markov bigram | 0.756 | 0.819 | 0.800 | **No** (recall ceiling 0.667) |
| DeepLog (LSTM) | 0.806 | 0.918 | 0.814 | Mostly |
| **LogTransformer** | **0.856** | **0.960** | **0.851** | **Yes** (recall → 1.0) |

The bigram **cannot exceed 0.667 recall** — it never flags the mismatched close.
The context models do, and the Transformer ranks best (PR-AUC 0.856 vs 0.756).
This is the demonstration Phase 3 lacked: on a long-range dependency, attention
earns its cost over a bigram.

(The low *operating-point* precision on this benchmark is a `top_k=2` artifact —
deliberately strict to force the long-range test. The threshold-free PR-AUC /
ROC-AUC and best-F1 are the fair comparison, and the Transformer wins all three.)

## 2. LLM explainer — Tier 3 peak / explainability

`app/models/llm_explainer.py`. For the handful of sessions the cheap tiers flag,
`LLMExplainer` asks Claude for a zero-shot second opinion **and** a plain-language
explanation an engineer can act on — the project's "explainable" goal.

```python
from app.models.llm_explainer import LLMExplainer

ex = LLMExplainer()  # model="claude-opus-5", effort="low"
if LLMExplainer.available():           # SDK importable + API key present
    result = ex.explain(
        '10.0.0.1 - - [x] "GET /admin/../../etc/passwd HTTP/1.1" 403 0',
        signals={"semantic_score": 3.1, "oov_ratio": 0.4},
    )
    print(result.as_dict())
    # {"anomaly": True, "severity": "high",
    #  "attack_type": "path_traversal",
    #  "reason": "Request tries to read /etc/passwd via ../ traversal."}
```

Why this and not fine-tuned LogLLM here: prompt-based Claude is **zero-shot** (no
training, no GPU) and reaches F1 ~0.82–0.91 on log anomaly detection in the
literature. It is the right peak tier for this environment and doubles as the
explanation layer.

Design:
- The `anthropic` SDK is imported **lazily** and the client is built on first
  call, so importing the module and running the whole test suite needs **no SDK
  and no API key**.
- `LLMExplainer.available()` reports whether a live call is possible.
- Tests inject a fake `_complete` to verify parsing/prompt without the network.
- Live use needs `pip install anthropic` and `ANTHROPIC_API_KEY` (or `ant auth
  login`). Default model `claude-opus-5` at `effort="low"` (cheap per-anomaly
  triage); override in the constructor.

## Where this sits

The tiered detector is now complete end to end:

```
Tier 0  rules/signatures        (deterministic attacks)
Tier 1  point + semantic        (Phases 1–2: fast filter, F1 0.62)
Tier 2  sequence: DeepLog+Quant  (Phase 3: order + volume anomalies)
Tier 3  Transformer + LLM        (Phase 4: long-range + explanation)
```

Cheap tiers filter the stream; the expensive tiers only see what they flag. Next
steps (bigger data / GPU): run the real HDFS benchmark (`run_hdfs_deeplog`, wired
in Phase 3) and, with a GPU, fine-tune the full LogLLM.
