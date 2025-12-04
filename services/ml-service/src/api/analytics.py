"""
Analytics API Endpoints

@ORACLE @PRISM - REST API for predictive analytics, forecasting, and insights.

Copyright (c) 2025 NEURECTOMY. All Rights Reserved.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
import structlog

from src.services.analytics import (
    PredictiveAnalyticsService,
    ResourceUsagePredictor,
    AgentPerformancePredictor,
    MetricType,
    ForecastMethod,
    MetricDataPoint,
    AnomalyResult,
    ForecastResult,
    TrendAnalysis,
    AnalyticsInsight,
    CostOptimizationRecommendation,
)


logger = structlog.get_logger()
router = APIRouter(prefix="/analytics", tags=["analytics"])

# Initialize services
analytics_service = PredictiveAnalyticsService()
resource_predictor = ResourceUsagePredictor(analytics_service)
agent_predictor = AgentPerformancePredictor(analytics_service)


# ==============================================================================
# Request/Response Models
# ==============================================================================

class RecordMetricRequest(BaseModel):
    """Request to record a metric."""
    metric_type: MetricType
    value: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RecordMetricResponse(BaseModel):
    """Response from recording a metric."""
    success: bool
    anomaly_detected: bool = False
    anomaly: Optional[AnomalyResult] = None


class ForecastRequest(BaseModel):
    """Request for forecast."""
    metric_type: MetricType
    periods: int = Field(default=10, ge=1, le=100)
    method: Optional[ForecastMethod] = None


class TrendRequest(BaseModel):
    """Request for trend analysis."""
    metric_type: MetricType
    period_hours: int = Field(default=24, ge=1, le=168)


class AgentMetricRequest(BaseModel):
    """Request to record agent metric."""
    agent_id: str
    success: bool
    latency_ms: float
    tokens_used: int


class DashboardResponse(BaseModel):
    """Dashboard data response."""
    generated_at: str
    metrics_summary: Dict[str, Any]
    forecasts: Dict[str, Any]
    anomalies: List[Dict[str, Any]]
    trends: Dict[str, Any]
    recommendations: List[Dict[str, Any]]


# ==============================================================================
# Endpoints
# ==============================================================================

@router.on_event("startup")
async def startup():
    """Initialize analytics service on startup."""
    await analytics_service.initialize()


@router.post("/metrics", response_model=RecordMetricResponse)
async def record_metric(request: RecordMetricRequest):
    """
    Record a metric value.
    
    Automatically checks for anomalies and returns detection result.
    """
    try:
        anomaly = await analytics_service.record_metric(
            metric_type=request.metric_type,
            value=request.value,
            metadata=request.metadata
        )
        
        return RecordMetricResponse(
            success=True,
            anomaly_detected=anomaly is not None and anomaly.detected,
            anomaly=anomaly
        )
    except Exception as e:
        logger.error("Failed to record metric", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics/batch")
async def record_metrics_batch(
    metrics: List[RecordMetricRequest]
) -> Dict[str, Any]:
    """Record multiple metrics at once."""
    results = []
    anomalies_detected = 0
    
    for metric in metrics:
        try:
            anomaly = await analytics_service.record_metric(
                metric_type=metric.metric_type,
                value=metric.value,
                metadata=metric.metadata
            )
            
            if anomaly and anomaly.detected:
                anomalies_detected += 1
                results.append({
                    "metric_type": metric.metric_type.value,
                    "anomaly": anomaly.model_dump()
                })
        except Exception as e:
            logger.error("Failed to record metric", metric=metric.metric_type, error=str(e))
    
    return {
        "success": True,
        "metrics_recorded": len(metrics),
        "anomalies_detected": anomalies_detected,
        "anomaly_details": results
    }


@router.get("/forecast/{metric_type}", response_model=ForecastResult)
async def get_forecast(
    metric_type: MetricType,
    periods: int = Query(default=10, ge=1, le=100),
    method: Optional[ForecastMethod] = None
):
    """
    Get forecast for a specific metric.
    
    Args:
        metric_type: Type of metric to forecast
        periods: Number of periods to forecast
        method: Forecasting method (optional)
    """
    try:
        result = await analytics_service.get_forecast(
            metric_type=metric_type,
            periods=periods,
            method=method
        )
        return result
    except Exception as e:
        logger.error("Failed to generate forecast", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trend/{metric_type}", response_model=TrendAnalysis)
async def analyze_trend(
    metric_type: MetricType,
    period_hours: int = Query(default=24, ge=1, le=168)
):
    """
    Analyze trend for a specific metric.
    
    Args:
        metric_type: Type of metric to analyze
        period_hours: Number of hours to analyze (max 168 = 1 week)
    """
    try:
        result = await analytics_service.analyze_trend(
            metric_type=metric_type,
            period_hours=period_hours
        )
        return result
    except Exception as e:
        logger.error("Failed to analyze trend", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/insights", response_model=List[AnalyticsInsight])
async def get_insights(
    limit: int = Query(default=10, ge=1, le=100)
):
    """Get recent analytics insights."""
    try:
        return await analytics_service.get_insights(limit=limit)
    except Exception as e:
        logger.error("Failed to get insights", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations", response_model=List[CostOptimizationRecommendation])
async def get_cost_recommendations():
    """Get cost optimization recommendations."""
    try:
        return await analytics_service.get_cost_recommendations()
    except Exception as e:
        logger.error("Failed to get recommendations", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard")
async def get_dashboard():
    """
    Get complete dashboard data.
    
    Includes metrics summaries, forecasts, trends, and recommendations.
    """
    try:
        data = await analytics_service.generate_dashboard_data()
        return data
    except Exception as e:
        logger.error("Failed to generate dashboard", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# Resource Usage Endpoints
# ==============================================================================

@router.get("/resources/prediction")
async def predict_resource_usage(
    hours_ahead: int = Query(default=24, ge=1, le=168)
) -> Dict[str, Any]:
    """
    Predict resource usage for capacity planning.
    
    Args:
        hours_ahead: Number of hours to predict
    """
    try:
        return await resource_predictor.predict_resource_needs(hours_ahead)
    except Exception as e:
        logger.error("Failed to predict resources", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/resources/cost-estimate")
async def estimate_cost(
    hours_ahead: int = Query(default=24, ge=1, le=720)
) -> Dict[str, Any]:
    """
    Estimate costs for the specified period.
    
    Args:
        hours_ahead: Number of hours to estimate (max 720 = 30 days)
    """
    try:
        return await resource_predictor.estimate_cost(hours_ahead)
    except Exception as e:
        logger.error("Failed to estimate cost", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# Agent Performance Endpoints
# ==============================================================================

@router.post("/agents/metrics")
async def record_agent_metric(request: AgentMetricRequest) -> Dict[str, Any]:
    """
    Record agent performance metric.
    
    Tracks success rate, latency, and token usage per agent.
    """
    try:
        await agent_predictor.record_agent_metric(
            agent_id=request.agent_id,
            success=request.success,
            latency_ms=request.latency_ms,
            tokens_used=request.tokens_used
        )
        return {"success": True, "agent_id": request.agent_id}
    except Exception as e:
        logger.error("Failed to record agent metric", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_id}/prediction")
async def predict_agent_performance(agent_id: str) -> Dict[str, Any]:
    """
    Predict agent performance metrics.
    
    Args:
        agent_id: ID of the agent to predict
    """
    try:
        return await agent_predictor.predict_agent_performance(agent_id)
    except Exception as e:
        logger.error("Failed to predict agent performance", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# Health & Status
# ==============================================================================

@router.get("/status")
async def analytics_status() -> Dict[str, Any]:
    """Get analytics service status."""
    metric_counts = {
        metric_type.value: len(analytics_service.metrics.get(metric_type.value, []))
        for metric_type in MetricType
    }
    
    return {
        "status": "healthy" if analytics_service._initialized else "initializing",
        "initialized": analytics_service._initialized,
        "metric_counts": metric_counts,
        "insights_count": len(analytics_service.insights),
        "history_size": analytics_service.history_size
    }
