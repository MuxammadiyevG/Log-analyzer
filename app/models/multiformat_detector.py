"""
Multi-format anomaly detector — the engine behind the web UI.

Works on ANY line-oriented log (web, syslog, JSON/ECS, or unknown → generic),
with no pre-training: it fits an outlier model on the uploaded batch itself
("what's unusual within these logs?") and combines that with deterministic
signature + severity rules, so it is useful even on a small paste where the
statistical model can't run.

Three signals per line, combined:
  * signature  — deterministic attack patterns (traversal, SQLi, XSS, …)
  * severity   — normalized log level / HTTP status / error-keyword badness
  * statistical — a *robust* IsolationForest outlier over char-TF-IDF(SVD) +
                  generic features, fit on the batch (only when big enough)

Design guardrails (learned from a false-positive report on benign router logs):
  * The statistical score is normalized with the median/MAD, not mean/std, and
    is zero unless a line is a STRONG outlier — so a benign, homogeneous batch
    flags nothing instead of exploding to "everything critical".
  * A statistical-only outlier is never "critical"/"high" — it is a rare pattern
    to review, not a known threat. Critical/high are reserved for signatures and
    genuinely high-severity log levels.

scikit-learn only. No Claude, no torch.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from loguru import logger
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from app.parsers.formats import (
    detect_and_parse,
    document,
    explain,
    generic_features,
    suspicious_score,
)

MIN_FIT = 30            # below this the statistical model can't be trusted
STAT_FLAG = 0.3         # statistical score (robust) needed to flag a line
_CRITICAL_LABELS = {
    "SQL injection", "command injection", "shell payload",
    "cross-site scripting (XSS)", "path traversal", "path traversal (encoded)",
    "sensitive file access",
}
_STAT_REASON = "unusual pattern vs the batch (review — not a known threat)"


def _severity(label: str, sev: float, stat: float, is_anom: bool) -> str:
    if not is_anom:
        return "normal"
    if label in _CRITICAL_LABELS:
        return "critical"
    if label:                       # scanner probe / secret probe / scanner UA
        return "high"
    if sev >= 0.8:
        return "high"
    if sev >= 0.6:
        return "medium"
    # statistical-only: rare, not a known threat -> cap at medium/low
    return "medium" if stat >= 0.7 else "low"


class MultiFormatDetector:
    """Fit-on-batch outlier detection + signature/severity rules, any format."""

    def __init__(self, contamination: float = 0.02, seed: int = 42):
        self.contamination = contamination
        self.seed = seed

    def analyze(self, lines: List[str]) -> Dict[str, Any]:
        recs = [detect_and_parse(ln) for ln in lines if ln.strip()]
        if not recs:
            return {"total": 0, "anomalies": 0, "anomaly_rate": 0.0,
                    "formats": {}, "statistical": False, "results": []}

        n = len(recs)
        sig_res = [suspicious_score(r) for r in recs]
        sig = np.array([s for s, _ in sig_res], dtype=float)
        sev = np.array([r.severity for r in recs], dtype=float)

        stat = np.zeros(n, dtype=float)
        used_stat = False
        if n >= MIN_FIT:
            try:
                stat = self._statistical(recs)
                used_stat = True
            except Exception as e:
                logger.warning(f"MultiFormatDetector: statistical stage skipped ({e})")

        results = []
        anomalies = 0
        for i, rec in enumerate(recs):
            label = sig_res[i][1]
            stat_flag = stat[i] >= STAT_FLAG
            rule_flag = sig[i] >= 1.0 or sev[i] >= 0.6
            is_anom = bool(rule_flag or stat_flag)
            score01 = float(max(sig[i], sev[i], stat[i]))
            if is_anom:
                anomalies += 1
                reason = explain(rec)
                if not rule_flag and stat_flag:  # statistical-only
                    reason = _STAT_REASON
            else:
                reason = "normal"
            results.append(
                {
                    "line": rec.raw if len(rec.raw) <= 500 else rec.raw[:500] + "…",
                    "format": rec.fmt,
                    "anomaly": is_anom,
                    "score": round(score01, 3),
                    "severity": _severity(label, sev[i], stat[i], is_anom),
                    "reason": reason,
                }
            )
        results.sort(key=lambda r: (not r["anomaly"], -r["score"]))

        formats: Dict[str, int] = {}
        for r in recs:
            formats[r.fmt] = formats.get(r.fmt, 0) + 1

        return {
            "total": n,
            "anomalies": anomalies,
            "anomaly_rate": round(anomalies / n, 4),
            "formats": formats,
            "statistical": used_stat,
            "results": results,
        }

    def _statistical(self, recs) -> np.ndarray:
        """Robust IsolationForest outlier score in [0,1].

        Returns 0 for the bulk of the batch; only STRONG outliers (robust
        z > 3 via median/MAD) rise above 0, so a homogeneous benign batch
        produces no statistical anomalies instead of blowing up.
        """
        docs = [document(r) for r in recs]
        tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, lowercase=True)
        X = tfidf.fit_transform(docs)
        n_comp = max(2, min(48, X.shape[1] - 1, X.shape[0] - 1))
        svd = TruncatedSVD(n_components=n_comp, random_state=self.seed).fit_transform(X)
        feats = np.vstack([generic_features(r) for r in recs])
        combined = StandardScaler().fit_transform(np.hstack([svd, feats]))
        iso = IsolationForest(
            n_estimators=200, contamination=self.contamination,
            random_state=self.seed, n_jobs=-1,
        ).fit(combined)
        raw = -iso.score_samples(combined)  # higher = more anomalous

        med = float(np.median(raw))
        mad = float(np.median(np.abs(raw - med)))
        if mad < 1e-6:                      # near-constant → nothing stands out
            return np.zeros(len(recs))
        rz = (raw - med) / (1.4826 * mad)   # robust z-score
        # rz <= 3 -> 0 (normal spread); rz in (3, 8] -> (0, 1]
        return np.clip((rz - 3.0) / 5.0, 0.0, 1.0)
