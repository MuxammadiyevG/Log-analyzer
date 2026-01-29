"""
FastAPI routes for anomaly detection API
"""

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.detector import AnomalyDetector


# Request/Response models
class LogRequest(BaseModel):
    """Single log analysis request"""

    log: str = Field(..., description="Raw log line to analyze")

    class Config:
        json_schema_extra = {
            "example": {
                "log": '192.168.1.1 - - [01/Jan/2024:12:00:00 +0000] "GET /api/users HTTP/1.1" 200 1234 "-" "Mozilla/5.0"'
            }
        }


class BatchLogRequest(BaseModel):
    """Batch log analysis request"""

    logs: List[str] = Field(..., description="List of raw log lines")

    class Config:
        json_schema_extra = {
            "example": {
                "logs": [
                    '192.168.1.1 - - [01/Jan/2024:12:00:00 +0000] "GET /api/users HTTP/1.1" 200 1234',
                    '192.168.1.2 - - [01/Jan/2024:12:00:01 +0000] "POST /api/login HTTP/1.1" 401 567',
                ]
            }
        }


class AnomalyResponse(BaseModel):
    """Anomaly detection response"""

    anomaly: bool = Field(..., description="Whether log is anomalous")
    score: float = Field(..., description="Anomaly score")
    model: str = Field(..., description="Model used for detection")
    reason: str = Field(..., description="Explanation of why it's anomalous")
    parsed_log: Dict[str, Any] = Field(..., description="Parsed log fields")


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    detector_type: str
    detector_trained: bool

    model_config = {"protected_namespaces": ()}


class StatsResponse(BaseModel):
    """Statistics response"""

    total_requests: int
    anomalies_detected: int
    anomaly_rate: float


# Router
router = APIRouter()

# Global detector instance
detector: AnomalyDetector = None
request_stats = {"total": 0, "anomalies": 0}


def get_detector() -> AnomalyDetector:
    """Get or create detector instance"""
    global detector
    if detector is None:
        detector = AnomalyDetector()
        # Try to load existing model
        try:
            detector.load()
            logger.info("Loaded existing model")
        except Exception as e:
            logger.warning(f"No existing model found: {e}")
    return detector


@router.post("/analyze", response_model=AnomalyResponse)
async def analyze_log(request: LogRequest) -> AnomalyResponse:
    """
    Analyze a single log line for anomalies

    Args:
        request: Log analysis request

    Returns:
        Anomaly detection result
    """
    try:
        det = get_detector()

        if not det.model.is_trained:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not trained. Please train the model first.",
            )

        result = det.analyze_log(request.log)

        # Update stats
        request_stats["total"] += 1
        if result["anomaly"]:
            request_stats["anomalies"] += 1

        return AnomalyResponse(**result)

    except Exception as e:
        logger.error(f"Error analyzing log: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/analyze/batch", response_model=List[AnomalyResponse])
async def analyze_batch(request: BatchLogRequest) -> List[AnomalyResponse]:
    """
    Analyze multiple log lines

    Args:
        request: Batch log analysis request

    Returns:
        List of anomaly detection results
    """
    try:
        det = get_detector()

        if not det.model.is_trained:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not trained. Please train the model first.",
            )

        results = []
        for log_line in request.logs:
            try:
                result = det.analyze_log(log_line)
                results.append(AnomalyResponse(**result))

                # Update stats
                request_stats["total"] += 1
                if result["anomaly"]:
                    request_stats["anomalies"] += 1
            except Exception as e:
                logger.warning(f"Error analyzing log line: {e}")
                continue

        return results

    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint

    Returns:
        Service health status
    """
    det = get_detector()

    return HealthResponse(
        status="healthy",
        detector_type=det.model_type,
        detector_trained=det.model.is_trained,
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    """
    Get detection statistics

    Returns:
        Request and anomaly statistics
    """
    total = request_stats["total"]
    anomalies = request_stats["anomalies"]
    rate = anomalies / total if total > 0 else 0.0

    return StatsResponse(
        total_requests=total, anomalies_detected=anomalies, anomaly_rate=rate
    )


@router.post("/reset-stats")
async def reset_stats() -> Dict[str, str]:
    """Reset statistics"""
    request_stats["total"] = 0
    request_stats["anomalies"] = 0
    return {"message": "Statistics reset successfully"}
