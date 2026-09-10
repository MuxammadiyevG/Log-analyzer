"""
Multi-format anomaly detector — the engine behind the web UI.

Works on ANY line-oriented log (web, syslog, JSON/ECS, or unknown → generic),
with no pre-training: it fits an outlier model on the uploaded batch itself
("what's unusual within these logs?") and combines that with deterministic
signature + severity rules, so it is useful even on a small paste where the
statistical model can't run.

Three signals per line, combined:
  * signature  — deterministic attack patterns (traversal, SQLi, XSS, …)
  * severity   — normalized log level / HTTP status badness
  * statistical — IsolationForest outlier over char-TF-IDF(SVD) + generic
                  features, fit on the batch (only when the batch is big enough)

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

MIN_FIT = 30  # below this, the statistical model can't be trusted → rules only


def _severity_bucket(score: float, is_anomaly: bool) -> str:
    if not is_anomaly:
        return "normal"
    if score >= 0.9:
        return "critical"
    if score >= 0.7:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


class MultiFormatDetector:
    """Fit-on-batch outlier detection + signature/severity rules, any format."""

    def __init__(self, contamination: float = 0.05, seed: int = 42):
        self.contamination = contamination
        self.seed = seed

    def analyze(self, lines: List[str]) -> Dict[str, Any]:
        recs = [detect_and_parse(ln) for ln in lines if ln.strip()]
        if not recs:
            return {"total": 0, "anomalies": 0, "anomaly_rate": 0.0,
                    "formats": {}, "statistical": False, "results": []}

        n = len(recs)
        sig = np.array([suspicious_score(r)[0] for r in recs], dtype=float)
        sev = np.array([r.severity for r in recs], dtype=float)

        # Statistical outlier score (only on a big-enough batch).
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
            is_anom = bool(sig[i] >= 1.0 or sev[i] >= 0.6 or stat[i] >= 0.5)
            score01 = float(max(sig[i], sev[i], stat[i]))
            if is_anom:
                anomalies += 1
            results.append(
                {
                    "line": rec.raw if len(rec.raw) <= 500 else rec.raw[:500] + "…",
                    "format": rec.fmt,
                    "anomaly": is_anom,
                    "score": round(score01, 3),
                    "severity": _severity_bucket(score01, is_anom),
                    "reason": explain(rec) if is_anom else "normal",
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
        """IsolationForest over char-TF-IDF(SVD) + generic features. Returns 0..1."""
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
        # z-normalize to a 0..1 positive-outlier band
        mu, sd = raw.mean(), raw.std() + 1e-9
        z = (raw - mu) / sd
        return np.clip(z / 4.0, 0.0, 1.0)
