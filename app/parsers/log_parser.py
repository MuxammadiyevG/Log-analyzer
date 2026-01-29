"""
Log parsing with Drain3 template mining and regex fallback
"""

import re
from datetime import datetime
from typing import Any, Dict, Optional

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from loguru import logger


class LogParser:
    """Universal log parser with multiple format support"""

    # Common log patterns
    APACHE_COMBINED = re.compile(
        r"(?P<ip>[\d\.]+) - (?P<user>\S+) \[(?P<timestamp>[^\]]+)\] "
        r'"(?P<method>\S+) (?P<path>\S+) (?P<protocol>\S+)" '
        r'(?P<status>\d+) (?P<size>\S+) "(?P<referrer>[^"]*)" "(?P<user_agent>[^"]*)"'
    )

    NGINX_COMBINED = re.compile(
        r"(?P<ip>[\d\.]+) - (?P<user>\S+) \[(?P<timestamp>[^\]]+)\] "
        r'"(?P<method>\S+) (?P<path>\S+) (?P<protocol>\S+)" '
        r'(?P<status>\d+) (?P<size>\d+) "(?P<referrer>[^"]*)" "(?P<user_agent>[^"]*)"'
    )

    COMMON_LOG = re.compile(
        r"(?P<ip>[\d\.]+) - (?P<user>\S+) \[(?P<timestamp>[^\]]+)\] "
        r'"(?P<method>\S+) (?P<path>\S+) (?P<protocol>\S+)" '
        r"(?P<status>\d+) (?P<size>\S+)"
    )

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize log parser

        Args:
            config: Parser configuration dictionary
        """
        self.config = config
        self.format = config.get("format", "auto")

        # Initialize Drain3 for template mining
        drain_config = TemplateMinerConfig()
        drain_config.load(config.get("drain3_config", {}))
        drain_config.profiling_enabled = False

        self.template_miner = TemplateMiner(config=drain_config)

        logger.info(f"LogParser initialized with format: {self.format}")

    def parse(self, log_line: str) -> Dict[str, Any]:
        """
        Parse a single log line

        Args:
            log_line: Raw log line string

        Returns:
            Dictionary with parsed fields
        """
        if not log_line or not log_line.strip():
            return self._empty_result()

        # Try format-specific parsing first
        if self.format == "auto":
            result = self._auto_parse(log_line)
        elif self.format == "apache_combined":
            result = self._parse_with_pattern(log_line, self.APACHE_COMBINED)
        elif self.format == "nginx":
            result = self._parse_with_pattern(log_line, self.NGINX_COMBINED)
        elif self.format == "common":
            result = self._parse_with_pattern(log_line, self.COMMON_LOG)
        else:
            result = self._simple_parse(log_line)

        # Extract template using Drain3
        if result:
            template_result = self.template_miner.add_log_message(log_line)
            result["template_id"] = template_result["cluster_id"]
            result["template"] = template_result["template_mined"]

        return result or self._empty_result()

    def _auto_parse(self, log_line: str) -> Optional[Dict[str, Any]]:
        """Try multiple patterns automatically"""
        patterns = [self.APACHE_COMBINED, self.NGINX_COMBINED, self.COMMON_LOG]

        for pattern in patterns:
            result = self._parse_with_pattern(log_line, pattern)
            if result:
                return result

        return self._simple_parse(log_line)

    def _parse_with_pattern(
        self, log_line: str, pattern: re.Pattern
    ) -> Optional[Dict[str, Any]]:
        """Parse using a specific regex pattern"""
        match = pattern.match(log_line)
        if not match:
            return None

        data = match.groupdict()

        # Normalize fields
        return {
            "timestamp": self._parse_timestamp(data.get("timestamp", "")),
            "ip": data.get("ip", "unknown"),
            "user": data.get("user", "-"),
            "method": data.get("method", "GET"),
            "path": data.get("path", "/"),
            "protocol": data.get("protocol", "HTTP/1.1"),
            "status": int(data.get("status", 0)),
            "size": self._parse_size(data.get("size", "0")),
            "referrer": data.get("referrer", "-"),
            "user_agent": data.get("user_agent", "-"),
            "raw": log_line,
        }

    def _simple_parse(self, log_line: str) -> Dict[str, Any]:
        """Fallback simple parsing"""
        parts = log_line.split()

        return {
            "timestamp": datetime.now(),
            "ip": parts[0] if len(parts) > 0 and self._is_ip(parts[0]) else "unknown",
            "user": "-",
            "method": self._extract_method(log_line),
            "path": self._extract_path(log_line),
            "protocol": "HTTP/1.1",
            "status": self._extract_status(log_line),
            "size": 0,
            "referrer": "-",
            "user_agent": "-",
            "raw": log_line,
        }

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp string"""
        formats = ["%d/%b/%Y:%H:%M:%S %z", "%Y-%m-%d %H:%M:%S", "%d/%b/%Y:%H:%M:%S"]

        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except (ValueError, AttributeError):
                continue

        return datetime.now()

    def _parse_size(self, size_str: str) -> int:
        """Parse response size"""
        if size_str == "-" or not size_str:
            return 0
        try:
            return int(size_str)
        except ValueError:
            return 0

    def _is_ip(self, s: str) -> bool:
        """Check if string is an IP address"""
        parts = s.split(".")
        if len(parts) != 4:
            return False
        try:
            return all(0 <= int(part) <= 255 for part in parts)
        except ValueError:
            return False

    def _extract_method(self, log_line: str) -> str:
        """Extract HTTP method"""
        methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
        for method in methods:
            if method in log_line:
                return method
        return "GET"

    def _extract_path(self, log_line: str) -> str:
        """Extract request path"""
        match = re.search(r'"\w+ ([^\s]+) ', log_line)
        return match.group(1) if match else "/"

    def _extract_status(self, log_line: str) -> int:
        """Extract status code"""
        match = re.search(r'" (\d{3}) ', log_line)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        return 200

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            "timestamp": datetime.now(),
            "ip": "unknown",
            "user": "-",
            "method": "GET",
            "path": "/",
            "protocol": "HTTP/1.1",
            "status": 0,
            "size": 0,
            "referrer": "-",
            "user_agent": "-",
            "template_id": -1,
            "template": "",
            "raw": "",
        }
