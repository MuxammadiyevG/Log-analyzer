"""
API tests
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app

client = TestClient(app)


def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health():
    """Test health check"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_type" in data
    assert data["status"] == "healthy"


def test_stats():
    """Test statistics endpoint"""
    response = client.get("/api/v1/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "anomalies_detected" in data
    assert "anomaly_rate" in data


def test_analyze_without_training():
    """Test analyze endpoint before model training"""
    response = client.post(
        "/api/v1/analyze",
        json={
            "log": '192.168.1.1 - - [01/Jan/2024:12:00:00 +0000] "GET /api/users HTTP/1.1" 200 1234'
        },
    )
    # Should return 503 if model not trained
    assert response.status_code in [200, 503]


def test_analyze_batch():
    """Test batch analysis endpoint"""
    logs = [
        '192.168.1.1 - - [01/Jan/2024:12:00:00 +0000] "GET /api/users HTTP/1.1" 200 1234',
        '192.168.1.2 - - [01/Jan/2024:12:00:01 +0000] "POST /api/login HTTP/1.1" 401 567',
    ]

    response = client.post("/api/v1/analyze/batch", json={"logs": logs})
    # Should return 200 or 503
    assert response.status_code in [200, 503]


def test_invalid_request():
    """Test invalid request"""
    response = client.post("/api/v1/analyze", json={})
    assert response.status_code == 422  # Validation error


def test_reset_stats():
    """Test reset statistics"""
    response = client.post("/api/v1/reset-stats")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data


@pytest.mark.parametrize(
    "log_line,expected_status",
    [
        (
            '192.168.1.1 - - [01/Jan/2024:12:00:00 +0000] "GET /api/users HTTP/1.1" 200 1234',
            200,
        ),
        ("invalid log line", 200),  # Should still process
        ("", 200),  # Empty should be handled
    ],
)
def test_analyze_various_logs(log_line, expected_status):
    """Test various log formats"""
    response = client.post("/api/v1/analyze", json={"log": log_line})
    assert response.status_code in [expected_status, 503]
