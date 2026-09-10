"""
Tests for multi-format log parsing + the fit-on-batch analyzer.

scikit-learn only — no torch, no Claude, no network.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("sklearn")

from app.parsers.formats import (  # noqa: E402
    detect_and_parse,
    explain,
    generic_features,
    suspicious_score,
    NUMERIC_DIM,
)


# --- format detection ------------------------------------------------------


def test_detect_web():
    r = detect_and_parse(
        '1.2.3.4 - - [01/Jan/2024:10:00:00 +0000] "GET /a HTTP/1.1" 200 5 "-" "UA"'
    )
    assert r.fmt == "web" and r.status == 200 and r.method == "GET"


def test_detect_syslog3164():
    r = detect_and_parse("Aug 10 12:00:06 web01 kernel: Out of memory: Killed process 1")
    assert r.fmt == "syslog"
    assert r.severity >= 0.8  # keyword-derived


def test_detect_syslog5424():
    r = detect_and_parse("<34>1 2024-01-01T10:00:00Z host app 111 ID - disk failure")
    assert r.fmt == "syslog5424"
    assert r.severity >= 0.8  # pri 34 % 8 == 2 (critical)


def test_detect_json_ecs():
    r = detect_and_parse('{"log.level":"error","message":"boom exception"}')
    assert r.fmt == "json" and r.level_name == "error"
    assert r.severity >= 0.8


def test_detect_generic():
    r = detect_and_parse("just some free-form text without structure")
    assert r.fmt == "generic" and r.severity == 0.0


def test_json_info_low_severity():
    r = detect_and_parse('{"level":"info","message":"user logged in"}')
    assert r.severity < 0.3


# --- signatures + features -------------------------------------------------


def test_signature_detection():
    assert suspicious_score(detect_and_parse('x "GET /a/../../etc/passwd HTTP/1.1" 403'))[0] == 1.0
    score, label = suspicious_score(detect_and_parse("q=1 UNION SELECT pw FROM users"))
    assert score == 1.0 and "SQL" in label


def test_generic_features_dim():
    r = detect_and_parse("Aug 10 12:00:06 web01 kernel: Out of memory")
    assert generic_features(r).shape == (NUMERIC_DIM,)


def test_explain_is_string():
    r = detect_and_parse('1.2.3.4 - - [x] "GET /admin/../../etc/passwd HTTP/1.1" 403 0 "-" "curl/7.0"')
    assert "traversal" in explain(r)


# --- analyzer (rules path, small batch) ------------------------------------


def test_analyzer_small_batch_rules_only():
    from app.models.multiformat_detector import MultiFormatDetector

    lines = [
        'Aug 10 12:00:01 web01 sshd[1]: Accepted password for admin from 10.0.0.2',
        'Aug 10 12:00:05 web01 sshd[2]: Failed password for invalid user root from 45.9.1.7',
        'Aug 10 12:00:06 web01 kernel: Out of memory: Killed process 1899',
        '{"level":"info","message":"ok"}',
        '1.2.3.4 - - [x] "GET /admin/../../etc/passwd HTTP/1.1" 403 0 "-" "curl/7.0"',
    ]
    out = MultiFormatDetector().analyze(lines)
    assert out["statistical"] is False  # < MIN_FIT
    assert out["anomalies"] == 3  # Failed, OOM, traversal
    assert set(out["formats"]) >= {"syslog", "json", "web"}
    # traversal must be the top (critical) result
    assert out["results"][0]["severity"] == "critical"
    # the benign "Accepted password" / info line stay normal
    normals = [r for r in out["results"] if not r["anomaly"]]
    assert any("Accepted password" in r["line"] for r in normals)


# --- analyzer (statistical path, big batch) --------------------------------


def test_analyzer_big_batch_statistical():
    from app.models.multiformat_detector import MultiFormatDetector

    # 60 near-identical normal lines + 1 blatant outlier
    normal = [f'10.0.0.{i%20} - - [x] "GET /api/items?p={i} HTTP/1.1" 200 {100+i} "-" "Mozilla/5.0"'
              for i in range(60)]
    outlier = '9.9.9.9 - - [x] "GET /x?q=1 UNION SELECT pw FROM users-- HTTP/1.1" 500 0 "-" "sqlmap/1"'
    out = MultiFormatDetector().analyze(normal + [outlier])
    assert out["statistical"] is True  # >= MIN_FIT
    assert out["anomalies"] >= 1
    assert out["results"][0]["anomaly"] is True


def test_benign_syslog_batch_not_over_flagged():
    """Regression: benign router/syslog logs must NOT come back all-critical."""
    import random

    from app.models.multiformat_detector import MultiFormatDetector

    r = random.Random(3)
    macs = ["ee:b1:d2:bb:cb:e4", "98:9e:63:2a:e9:10", "a6:81:2b:eb:68:0f"]
    lines = []
    for i in range(120):
        ts = f"Apr 28 {r.randint(10,23):02d}:{r.randint(0,59):02d}:{r.randint(0,59):02d}"
        mac = r.choice(macs); ip = f"192.168.{r.randint(20,23)}.{r.randint(2,254)}"
        kind = r.choice(["DHCPREQUEST", "DHCPACK"])
        lines.append(f"<30>{ts} UDM-Pro dnsmasq-dhcp[7003]: {kind}(br0) {ip} {mac}")
    out = MultiFormatDetector().analyze(lines)
    # benign batch: no critical/high, and few (if any) anomalies
    assert not any(x["severity"] in ("critical", "high") for x in out["results"])
    assert out["anomaly_rate"] < 0.1
