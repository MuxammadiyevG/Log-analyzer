"""
Semantic vectorization of web-server log lines (Phase 2).

The Phase-1 baseline collapsed the whole request into a single scalar
"suspicious_score", throwing away the actual text. This module keeps the text:
it turns the request line into character n-gram TF-IDF vectors, so tokens like
`../`, `union select`, `<script>`, `.env`, `sqlmap`, `%2f` become first-class
features. A TruncatedSVD projection then compresses the sparse space into a
dense vector any detector can consume.

Character n-grams (not word tokens) are deliberate: attacks mutate word
boundaries (`un/**/ion`, `..%2f`, `<scr ipt>`) but the character substrings
survive, which is exactly why char-level TF-IDF is the standard robust choice
for log/URL anomaly work (LogRobust/NeuralLog use the same intuition with
FastText).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

import numpy as np
from loguru import logger
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# "METHOD path PROTO" ... status ... "referrer" "user-agent"
_REQ_RE = re.compile(r'"(?P<method>[A-Z]+)\s+(?P<path>\S+)\s+HTTP/[\d.]+"\s+(?P<status>\d{3})')
_UA_RE = re.compile(r'"[^"]*"\s+"(?P<ua>[^"]*)"\s*$')

_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
# Status codes that are structurally rare / suspicious for normal web traffic.
_RARE_STATUS = {418, 451, 226, 414, 599, 431, 511}
NUMERIC_DIM = 3 + len(_METHODS) + 4  # path stats + method one-hot + status buckets


def numeric_features(raw: str) -> np.ndarray:
    """Structural (non-semantic) features that catch *categorical* anomalies.

    Semantic novelty alone misses things like an unusual method or a rare
    status code, which carry no strange tokens. These numeric features cover
    that axis; the model fuses both.
    """
    m = _REQ_RE.search(raw)
    method = m.group("method") if m else "GET"
    path = m.group("path") if m else "/"
    status = int(m.group("status")) if m else 200

    feats = [
        float(len(path)),
        float(sum(c.isdigit() for c in path)),
        float(sum(not c.isalnum() for c in path)),  # special-char density
        *(1.0 if method == x else 0.0 for x in _METHODS),
        1.0 if 200 <= status < 300 else 0.0,
        1.0 if 400 <= status < 500 else 0.0,
        1.0 if status >= 500 else 0.0,
        1.0 if status in _RARE_STATUS else 0.0,
    ]
    return np.asarray(feats, dtype=np.float32)


def request_document(raw: str) -> str:
    """Flatten a raw log line into a text document for TF-IDF.

    Combines method, path (verbatim — the attack lives here), status, and user
    agent. Falls back to the whole line when the request can't be parsed.
    """
    m = _REQ_RE.search(raw)
    ua_m = _UA_RE.search(raw)
    ua = ua_m.group("ua") if ua_m else "-"
    if m:
        return f"{m.group('method')} {m.group('path')} {m.group('status')} {ua}"
    return raw


class SemanticVectorizer:
    """Raw log lines -> dense semantic vectors.

    Pipeline: request document -> char n-gram TF-IDF -> TruncatedSVD (dense).
    Fit on presumed-normal traffic only.
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        config = config or {}
        self.ngram_range = tuple(config.get("ngram_range", (3, 5)))
        self.max_features = int(config.get("max_features", 20000))
        self.min_df = int(config.get("min_df", 2))
        self.n_components = int(config.get("n_components", 48))

        self.tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            lowercase=True,
        )
        self.svd: TruncatedSVD | None = None
        self._analyzer = None
        self._vocab: set = set()
        self.is_fitted = False

    def fit(self, lines: List[str]) -> "SemanticVectorizer":
        docs = [request_document(ln) for ln in lines]
        X = self.tfidf.fit_transform(docs)

        # Keep SVD components strictly below the rank of the TF-IDF matrix.
        n_comp = min(self.n_components, X.shape[1] - 1, max(X.shape[0] - 1, 1))
        n_comp = max(n_comp, 2)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.svd.fit(X)

        self._analyzer = self.tfidf.build_analyzer()
        self._vocab = set(self.tfidf.vocabulary_)
        self.is_fitted = True
        logger.info(
            f"SemanticVectorizer fitted: vocab={len(self._vocab)}, "
            f"svd_dim={n_comp}, explained_var={self.svd.explained_variance_ratio_.sum():.3f}"
        )
        return self

    def transform(self, lines: List[str]) -> np.ndarray:
        """Dense SVD projection of the char-n-gram TF-IDF."""
        if not self.is_fitted or self.svd is None:
            raise RuntimeError("SemanticVectorizer not fitted")
        docs = [request_document(ln) for ln in lines]
        X = self.tfidf.transform(docs)
        return self.svd.transform(X).astype(np.float32)

    def oov_ratio(self, lines: List[str]) -> np.ndarray:
        """Fraction of char n-grams unseen in normal training.

        This is the single strongest novelty signal: injection payloads
        (`passwd`, `union`, `<script`, `.env`, `sqlmap`) are full of n-grams
        that never appear in normal traffic, and TF-IDF's transform silently
        *drops* them — so we measure them directly instead of losing them.
        """
        if not self.is_fitted or self._analyzer is None:
            raise RuntimeError("SemanticVectorizer not fitted")
        out = np.empty(len(lines), dtype=np.float32)
        for i, ln in enumerate(lines):
            grams = self._analyzer(request_document(ln))
            if not grams:
                out[i] = 0.0
            else:
                out[i] = sum(1 for g in grams if g not in self._vocab) / len(grams)
        return out

    def fit_transform(self, lines: List[str]) -> np.ndarray:
        self.fit(lines)
        return self.transform(lines)
