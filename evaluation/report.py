"""
Render evaluation scorecards to a console table, Markdown, and JSON.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from evaluation.metrics import EvalResult


def _fmt(x: float) -> str:
    return "n/a" if x != x else f"{x:.4f}"  # x != x catches NaN


def render_console(results: Dict[str, EvalResult]) -> str:
    lines = []
    header = f"{'scorecard':<20} {'PR-AUC':>8} {'ROC-AUC':>8} {'Prec':>7} {'Recall':>7} {'F1':>7}"
    lines.append(header)
    lines.append("-" * len(header))
    for key, r in results.items():
        lines.append(
            f"{key:<20} {_fmt(r.pr_auc):>8} {_fmt(r.roc_auc):>8} "
            f"{_fmt(r.precision):>7} {_fmt(r.recall):>7} {_fmt(r.f1):>7}"
        )
    return "\n".join(lines)


def render_markdown(
    results: Dict[str, EvalResult], *, title: str = "Evaluation report", meta: dict | None = None
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out = [f"# {title}", "", f"_Generated: {ts}_", ""]

    if meta:
        out.append("## Dataset")
        out.append("")
        for k, v in meta.items():
            out.append(f"- **{k}**: `{v}`")
        out.append("")

    out.append("## Scorecards")
    out.append("")
    out.append(
        "| scorecard | samples | anomalies | PR-AUC | ROC-AUC | threshold | precision | recall | F1 | FPR |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for key, r in results.items():
        out.append(
            f"| {key} | {r.n_samples} | {r.n_anomalies} | {_fmt(r.pr_auc)} | "
            f"{_fmt(r.roc_auc)} | {r.threshold_source} | {_fmt(r.precision)} | "
            f"{_fmt(r.recall)} | {_fmt(r.f1)} | {_fmt(r.false_positive_rate)} |"
        )
    out.append("")
    out.append("## Confusion (per scorecard)")
    out.append("")
    for key, r in results.items():
        out.append(f"- **{key}**: TP={r.tp} FP={r.fp} FN={r.fn} TN={r.tn}")
    out.append("")
    out.append(
        "> PR-AUC / ROC-AUC are threshold-free ranking quality. The gap between "
        "the `operating` and `best-f1` scorecards is pure threshold headroom."
    )
    return "\n".join(out)


def save_json(results: Dict[str, EvalResult], path: Path, *, meta: dict | None = None) -> None:
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "meta": meta or {},
        "results": {k: r.as_dict() for k, r in results.items()},
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def save_markdown(
    results: Dict[str, EvalResult], path: Path, *, title: str = "Evaluation report", meta: dict | None = None
) -> None:
    Path(path).write_text(render_markdown(results, title=title, meta=meta))
