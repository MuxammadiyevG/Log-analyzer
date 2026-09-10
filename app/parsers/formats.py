"""
Multi-format log parsing for the analyzer UI.

Detects and normalizes the common line-oriented log formats — web
(Apache/Nginx combined), syslog (RFC 3164 and 5424), JSON / Elastic Common
Schema (ECS), and a generic fallback — into one `Record` with a unified,
format-agnostic feature vector and a human-readable explanation.

The point: the anomaly engine can then run on ANY of these formats, because the
features (severity, error-keyword density, suspicious tokens, character
statistics) are computed the same way regardless of syntax, while each format
still contributes its own structured fields (HTTP status, syslog severity, JSON
log.level).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

# --- suspicious signatures (deterministic, format-independent) --------------
SIGNATURES = [
    ("../", "path traversal"),
    ("..%2f", "path traversal (encoded)"),
    ("%2e%2e", "path traversal (encoded)"),
    ("/etc/passwd", "sensitive file access"),
    ("/etc/shadow", "sensitive file access"),
    ("union select", "SQL injection"),
    ("union%20select", "SQL injection"),
    ("' or ", "SQL injection"),
    (" or 1=1", "SQL injection"),
    ("drop table", "SQL injection"),
    ("<script", "cross-site scripting (XSS)"),
    ("onerror=", "cross-site scripting (XSS)"),
    ("onload=", "cross-site scripting (XSS)"),
    (";cat ", "command injection"),
    ("$(", "command injection"),
    ("|nc ", "command injection"),
    ("/bin/sh", "shell payload"),
    ("/.env", "config/secret probe"),
    ("/.git", "source-repo probe"),
    ("wp-login", "scanner probe"),
    ("phpmyadmin", "scanner probe"),
    ("/actuator", "actuator probe"),
]
_SCANNER_UAS = ("sqlmap", "nikto", "nmap", "masscan", "nuclei", "python-requests", "curl/")
_ERROR_KW = re.compile(
    r"\b(error|fail(ed|ure)?|denied|refus(ed|e)|exception|fatal|panic|unauthori[sz]ed|"
    r"invalid|timeout|timed out|unable|cannot|segfault|oom|out of memory|killed|"
    r"reject(ed)?|malformed|traceback|stacktrace|forbidden|breach|attack|malicious)\b",
    re.IGNORECASE,
)
_IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")

# --- format patterns --------------------------------------------------------
_WEB = re.compile(
    r'^(?P<ip>\d{1,3}(?:\.\d{1,3}){3})\s+\S+\s+\S+\s+\[(?P<ts>[^\]]+)\]\s+'
    r'"(?P<method>[A-Z]+)\s+(?P<path>\S+)\s+HTTP/[\d.]+"\s+(?P<status>\d{3})'
    r'(?:\s+\S+\s+"[^"]*"\s+"(?P<ua>[^"]*)")?'
)
_SYSLOG5424 = re.compile(
    r"^<(?P<pri>\d{1,3})>1\s+(?P<ts>\S+)\s+(?P<host>\S+)\s+(?P<app>\S+)\s+(?P<pid>\S+)\s+\S+\s+(?:-\s+)?(?P<msg>.*)$"
)
_SYSLOG3164 = re.compile(
    r"^(?:<(?P<pri>\d{1,3})>)?(?P<ts>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+"
    r"(?P<host>\S+)\s+(?P<tag>[^:\[]+)(?:\[(?P<pid>\d+)\])?:\s*(?P<msg>.*)$"
)

_SYSLOG_SEV = {  # RFC5424 severity (pri % 8) -> badness 0..1
    0: 1.0, 1: 0.95, 2: 0.9, 3: 0.8, 4: 0.55, 5: 0.3, 6: 0.1, 7: 0.0,
}
_JSON_LEVEL = {
    "emergency": 1.0, "alert": 0.95, "critical": 0.9, "crit": 0.9, "fatal": 0.9,
    "error": 0.8, "err": 0.8, "warning": 0.55, "warn": 0.55, "notice": 0.3,
    "info": 0.1, "information": 0.1, "debug": 0.0, "trace": 0.0,
}

NUMERIC_DIM = 9


@dataclass
class Record:
    fmt: str
    raw: str
    message: str
    severity: float = 0.0           # normalized badness 0..1
    level_name: str = ""
    source: str = ""                # host / app / tag
    status: int = 0                 # web only
    method: str = ""
    path: str = ""
    ua: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


_SEV_HIGH = re.compile(
    r"\b(out of memory|oom|killed|kernel panic|panic|fatal|segfault|core dump(ed)?|"
    r"emergency|unauthori[sz]ed|forbidden|breach|malicious|attack|exploit|"
    r"exception|traceback|stack ?trace)\b", re.IGNORECASE)
_SEV_MED = re.compile(
    r"\b(fail(ed|ure)?|denied|refus(ed|e)|error|invalid|reject(ed)?|"
    r"timeout|timed out|unable|cannot|malformed)\b", re.IGNORECASE)


def _keyword_severity(text: str) -> float:
    """Derive a badness score from error keywords (for formats without a level)."""
    if _SEV_HIGH.search(text):
        return 0.85
    if _SEV_MED.search(text):
        return 0.6
    return 0.0


def _http_severity(status: int) -> float:
    if status >= 500:
        return 0.9
    if status in (401, 403):
        return 0.7
    if status >= 400:
        return 0.6
    if status >= 300:
        return 0.2
    return 0.0


def detect_and_parse(line: str) -> Record:
    """Detect the format of one line and parse it into a Record."""
    s = line.strip()
    if not s:
        return Record(fmt="empty", raw=line, message="")

    # JSON / ECS
    if s[0] == "{":
        try:
            obj = json.loads(s)
            msg = str(obj.get("message") or obj.get("msg") or obj.get("log", {}).get("message") if isinstance(obj.get("log"), dict) else obj.get("message") or obj.get("msg") or "")
            if not msg:
                msg = s
            level = str(
                obj.get("log.level") or obj.get("level")
                or (obj.get("log", {}) if isinstance(obj.get("log"), dict) else {}).get("level")
                or obj.get("severity") or ""
            ).lower()
            sev = _JSON_LEVEL.get(level, 0.1 if level else 0.0)
            sev = max(sev, _keyword_severity(msg))
            src = str(obj.get("host", {}).get("name") if isinstance(obj.get("host"), dict) else obj.get("host") or obj.get("service", {}).get("name") if isinstance(obj.get("service"), dict) else obj.get("service") or "")
            return Record(fmt="json", raw=line, message=msg, severity=sev,
                          level_name=level, source=src, extra={"json": True})
        except (json.JSONDecodeError, ValueError, AttributeError):
            pass

    # syslog RFC5424
    m = _SYSLOG5424.match(s)
    if m:
        sev = max(_SYSLOG_SEV.get(int(m.group("pri")) % 8, 0.1), _keyword_severity(m.group("msg")))
        return Record(fmt="syslog5424", raw=line, message=m.group("msg"),
                      severity=sev, source=m.group("app"),
                      extra={"host": m.group("host"), "pid": m.group("pid")})

    # web (Apache/Nginx combined)
    m = _WEB.match(s)
    if m:
        status = int(m.group("status"))
        return Record(fmt="web", raw=line, message=f'{m.group("method")} {m.group("path")}',
                      severity=_http_severity(status), status=status,
                      method=m.group("method"), path=m.group("path"),
                      ua=m.group("ua") or "-", source=m.group("ip"))

    # syslog RFC3164
    m = _SYSLOG3164.match(s)
    if m:
        pri = m.group("pri")
        base = _SYSLOG_SEV.get(int(pri) % 8, 0.0) if pri else 0.0
        sev = max(base, _keyword_severity(m.group("msg")))
        return Record(fmt="syslog", raw=line, message=m.group("msg"),
                      severity=sev, source=(m.group("tag") or "").strip(),
                      extra={"host": m.group("host")})

    # generic
    return Record(fmt="generic", raw=line, message=s, severity=_keyword_severity(s))


def document(rec: Record) -> str:
    """Text fed to char n-gram TF-IDF (format-agnostic)."""
    if rec.fmt == "web":
        return f"{rec.method} {rec.path} {rec.status} {rec.ua}"
    return (rec.source + " " + rec.message).strip() or rec.raw


def suspicious_score(rec: Record) -> tuple[float, str]:
    """Deterministic signature match over the raw line. Returns (score, label)."""
    low = rec.raw.lower()
    for needle, label in SIGNATURES:
        if needle in low:
            return 1.0, label
    ua = rec.ua.lower()
    if ua and any(t in ua for t in _SCANNER_UAS):
        return 0.6, "scanner/tool user-agent"
    return 0.0, ""


def generic_features(rec: Record) -> np.ndarray:
    """Fixed-length, format-agnostic numeric features (dim = NUMERIC_DIM)."""
    text = rec.message or rec.raw
    n = max(len(text), 1)
    digits = sum(c.isdigit() for c in text) / n
    special = sum(not c.isalnum() and not c.isspace() for c in text) / n
    upper = sum(c.isupper() for c in text) / n
    tokens = len(text.split())
    err = len(_ERROR_KW.findall(text))
    sus, _ = suspicious_score(rec)
    has_ip = 1.0 if _IP.search(rec.raw) else 0.0
    return np.array(
        [
            math.log1p(len(text)),
            digits,
            special,
            upper,
            math.log1p(tokens),
            float(min(err, 5)),
            rec.severity,
            sus,
            has_ip,
        ],
        dtype=np.float32,
    )


def explain(rec: Record) -> str:
    """Format-aware human explanation."""
    reasons = []
    sus, label = suspicious_score(rec)
    if label:
        reasons.append(label)

    if rec.fmt == "web":
        if rec.status >= 500:
            reasons.append(f"server error ({rec.status})")
        elif rec.status >= 400:
            reasons.append(f"client error ({rec.status})")
        if rec.method in ("DELETE", "PUT", "PATCH", "TRACE", "CONNECT"):
            reasons.append(f"uncommon method {rec.method}")
        if len(rec.path) > 200:
            reasons.append("unusually long path")
    else:
        if rec.severity >= 0.8:
            reasons.append(f"high-severity log ({rec.level_name or 'error-level'})")
        elif rec.severity >= 0.5:
            reasons.append(f"warning-level log ({rec.level_name or 'warning'})")
        kws = _ERROR_KW.findall(rec.message or rec.raw)
        if kws:
            uniq = list(dict.fromkeys(k if isinstance(k, str) else k[0] for k in kws))[:3]
            reasons.append("error keywords: " + ", ".join(uniq))

    return " + ".join(dict.fromkeys(r for r in reasons if r)) or "statistical outlier vs the batch"
