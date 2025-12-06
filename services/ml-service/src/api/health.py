"""
Health Check Routes.

@SENTRY - Health and readiness endpoints for Kubernetes.
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.config import settings

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    version: str
    service: str
    details: dict[str, Any] = {}


class ReadinessResponse(BaseModel):
    """Readiness check response model."""
    ready: bool
    timestamp: datetime
    checks: dict[str, bool] = {}


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Basic health check endpoint.
    
    Returns service health status.
    Used by Kubernetes liveness probes.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="0.2.0",
        service="ml-service",
        details={
            "environment": settings.environment,
        },
    )


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_check() -> ReadinessResponse:
    """
    Readiness check endpoint.
    
    Verifies all dependencies are ready.
    Used by Kubernetes readiness probes.
    """
    checks = {
        "config_loaded": True,
    }
    
    # Check PostgreSQL
    try:
        from src.db.postgres import get_db_session
        async with get_db_session() as session:
            await session.execute("SELECT 1")
        checks["postgres"] = True
    except Exception:
        checks["postgres"] = False
    
    # Check Redis
    try:
        from src.db.redis import get_redis
        redis = await get_redis()
        await redis.client.ping()
        checks["redis"] = True
    except Exception:
        checks["redis"] = False
    
    ready = all(checks.values())
    
    return ReadinessResponse(
        ready=ready,
        timestamp=datetime.utcnow(),
        checks=checks,
    )


@router.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    
    @SENTRY - Metrics for observability.
    """
    # In production, use prometheus_client to generate metrics
    return {
        "requests_total": 0,
        "requests_in_flight": 0,
        "llm_tokens_total": 0,
        "embeddings_generated": 0,
        "training_jobs_total": 0,
    }


@router.get("/inference-metrics")
async def inference_metrics():
    """
    Inference performance metrics endpoint.
    
    @VELOCITY @SENTRY - Real-time inference optimization metrics.
    
    Returns:
        - avg_latency_ms: Average inference latency
        - min/max_latency_ms: Latency bounds
        - batches_processed: Total batches processed
        - avg_batch_size: Average requests per batch
        - cache_hit_rate: Cache efficiency
        - total_errors: Error count
        - retries: Retry count
    """
    from src.services.inference_optimizer import get_inference_metrics
    
    metrics = get_inference_metrics()
    return metrics.to_dict()
