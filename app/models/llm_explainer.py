"""
LLM explainer — Tier 3 peak / explainability (Phase 4).

The lower tiers say *that* something is anomalous and give a numeric score. This
tier uses Claude to (a) give a zero-shot second opinion and (b) explain the
anomaly in plain language an on-call engineer can act on — the project's stated
"explainable" goal. It is meant for the handful of sessions the cheap tiers
flag, not every log line.

This is the realistic LLM tier for this environment: full LogLLM
(BERT+Llama+QLoRA fine-tuning) needs a GPU and large downloads, whereas a
prompt-based call to Claude is zero-shot (no training) and, per the literature,
reaches F1 ~0.82–0.91 on log anomaly detection with no labels.

Design notes:
  * The `anthropic` SDK is imported lazily, so importing this module — and the
    whole test suite — works with no SDK and no API key installed.
  * `available()` reports whether a live call is possible.
  * Tests inject a fake `_complete` to exercise parsing without the network.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger

# Default model per the claude-api guidance; effort kept low because this is a
# high-volume per-anomaly triage call, not a deep reasoning task.
DEFAULT_MODEL = "claude-opus-5"

_SYSTEM = (
    "You are a security log-analysis assistant. You are given a web/server log "
    "sample that an automated detector flagged, plus the detector's signals. "
    "Decide whether it is genuinely anomalous and explain why in one or two "
    "plain sentences an on-call engineer can act on. Respond with ONLY a JSON "
    'object: {"anomaly": true|false, "severity": "low"|"medium"|"high"|"critical", '
    '"attack_type": "<short label or none>", "reason": "<one or two sentences>"}.'
)


@dataclass
class Explanation:
    anomaly: bool
    severity: str
    attack_type: str
    reason: str
    raw: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "anomaly": self.anomaly,
            "severity": self.severity,
            "attack_type": self.attack_type,
            "reason": self.reason,
        }


class LLMExplainer:
    """Zero-shot Claude explainer/confirmer for flagged log samples."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: Optional[str] = None,
        effort: str = "low",
        max_tokens: int = 512,
    ):
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self._api_key = api_key
        self._client = None  # created lazily on first live call

    # -- availability -------------------------------------------------------

    @staticmethod
    def available() -> bool:
        """True if a live call could run (SDK importable + a key present)."""
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))

    def _get_client(self):
        if self._client is None:
            import anthropic  # lazy

            self._client = (
                anthropic.Anthropic(api_key=self._api_key)
                if self._api_key
                else anthropic.Anthropic()
            )
        return self._client

    # -- prompt / call ------------------------------------------------------

    @staticmethod
    def _build_user_prompt(sample: str, signals: Optional[Dict[str, Any]]) -> str:
        parts = ["Log sample(s):", sample.strip()]
        if signals:
            parts.append("\nDetector signals:")
            for k, v in signals.items():
                parts.append(f"- {k}: {v}")
        return "\n".join(parts)

    def _complete(self, system: str, user: str) -> str:
        """One Claude call returning the raw assistant text. Override in tests."""
        client = self._get_client()
        resp = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

    # -- public -------------------------------------------------------------

    def explain(
        self, sample: str, *, signals: Optional[Dict[str, Any]] = None
    ) -> Explanation:
        """Classify + explain one flagged sample. Zero-shot, no training."""
        text = self._complete(_SYSTEM, self._build_user_prompt(sample, signals))
        return self._parse(text)

    @staticmethod
    def _parse(text: str) -> Explanation:
        """Parse the JSON verdict, tolerant of stray prose around it."""
        m = re.search(r"\{.*\}", text, re.DOTALL)
        blob = m.group(0) if m else text
        try:
            data = json.loads(blob)
        except (json.JSONDecodeError, ValueError):
            logger.warning("LLMExplainer: could not parse JSON, defaulting to unknown")
            data = {}
        return Explanation(
            anomaly=bool(data.get("anomaly", False)),
            severity=str(data.get("severity", "unknown")),
            attack_type=str(data.get("attack_type", "none")),
            reason=str(data.get("reason", text.strip()[:300] or "no explanation")),
            raw=text,
        )

    def explain_flagged(
        self, samples: List[str], *, signals: Optional[List[Dict[str, Any]]] = None
    ) -> List[Explanation]:
        """Explain a batch of flagged samples (call the lower tiers first)."""
        out = []
        for i, s in enumerate(samples):
            sig = signals[i] if signals and i < len(signals) else None
            out.append(self.explain(s, signals=sig))
        return out
