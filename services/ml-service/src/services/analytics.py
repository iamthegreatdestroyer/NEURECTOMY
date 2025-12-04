"""
Predictive Analytics Service

@ORACLE @PRISM - Advanced analytics for agent performance forecasting,
resource usage prediction, and anomaly detection.

Copyright (c) 2025 NEURECTOMY. All Rights Reserved.

Features:
- Time series forecasting for resource usage
- Agent performance prediction
- Anomaly detection using statistical methods
- Trend analysis and insights
- Cost optimization recommendations
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import math
import statistics
import structlog

from pydantic import BaseModel, Field


logger = structlog.get_logger()


# ==============================================================================
# Analytics Enums and Types
# ==============================================================================

class MetricType(str, Enum):
    """Types of metrics that can be tracked."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    TOKEN_USAGE = "token_usage"
    COST = "cost"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    CACHE_HIT_RATE = "cache_hit_rate"
    AGENT_SUCCESS_RATE = "agent_success_rate"


class AnomalyType(str, Enum):
    """Types of anomalies detected."""
    SPIKE = "spike"
    DROP = "drop"
    TREND_CHANGE = "trend_change"
    PATTERN_BREAK = "pattern_break"
    THRESHOLD_BREACH = "threshold_breach"


class ForecastMethod(str, Enum):
    """Forecasting methods available."""
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    LINEAR_REGRESSION = "linear_regression"
    HOLT_WINTERS = "holt_winters"


# ==============================================================================
# Analytics Models
# ==============================================================================

class MetricDataPoint(BaseModel):
    """Single data point for a metric."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    value: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TimeSeriesData(BaseModel):
    """Time series data for analytics."""
    metric_type: MetricType
    data_points: List[MetricDataPoint] = Field(default_factory=list)
    aggregation_interval: str = "1m"  # 1m, 5m, 1h, 1d


class AnomalyResult(BaseModel):
    """Result of anomaly detection."""
    detected: bool
    anomaly_type: Optional[AnomalyType] = None
    severity: float = 0.0  # 0-1 scale
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metric_type: MetricType
    actual_value: float
    expected_value: float
    deviation: float
    message: str = ""


class ForecastResult(BaseModel):
    """Result of forecasting."""
    metric_type: MetricType
    method: ForecastMethod
    predictions: List[Tuple[datetime, float]] = Field(default_factory=list)
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    accuracy_score: float = 0.0
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class TrendAnalysis(BaseModel):
    """Trend analysis result."""
    metric_type: MetricType
    trend: str  # "increasing", "decreasing", "stable", "volatile"
    slope: float
    correlation: float
    period_start: datetime
    period_end: datetime
    insights: List[str] = Field(default_factory=list)


class AnalyticsInsight(BaseModel):
    """High-level insight from analytics."""
    category: str
    title: str
    description: str
    severity: str  # "info", "warning", "critical"
    recommendation: str
    metric_type: Optional[MetricType] = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class CostOptimizationRecommendation(BaseModel):
    """Cost optimization recommendation."""
    category: str
    current_cost: float
    potential_savings: float
    implementation: str
    impact: str  # "low", "medium", "high"
    confidence: float


# ==============================================================================
# Statistical Utilities
# ==============================================================================

class StatisticalEngine:
    """Statistical computation engine for analytics."""
    
    @staticmethod
    def moving_average(data: List[float], window: int = 5) -> List[float]:
        """Calculate moving average."""
        if len(data) < window:
            return data
        
        result = []
        for i in range(len(data) - window + 1):
            avg = sum(data[i:i + window]) / window
            result.append(avg)
        return result
    
    @staticmethod
    def exponential_smoothing(
        data: List[float], 
        alpha: float = 0.3
    ) -> List[float]:
        """Apply exponential smoothing."""
        if not data:
            return []
        
        result = [data[0]]
        for i in range(1, len(data)):
            smoothed = alpha * data[i] + (1 - alpha) * result[-1]
            result.append(smoothed)
        return result
    
    @staticmethod
    def linear_regression(
        x: List[float], 
        y: List[float]
    ) -> Tuple[float, float, float]:
        """
        Calculate linear regression.
        Returns: (slope, intercept, r_squared)
        """
        if len(x) != len(y) or len(x) < 2:
            return (0.0, 0.0, 0.0)
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi ** 2 for xi in x)
        sum_y2 = sum(yi ** 2 for yi in y)
        
        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return (0.0, sum_y / n if n > 0 else 0.0, 0.0)
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        
        # Calculate R-squared
        ss_tot = sum((yi - sum_y / n) ** 2 for yi in y)
        ss_res = sum((y[i] - (slope * x[i] + intercept)) ** 2 for i in range(n))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return (slope, intercept, r_squared)
    
    @staticmethod
    def z_score(value: float, mean: float, std: float) -> float:
        """Calculate Z-score for anomaly detection."""
        if std == 0:
            return 0.0
        return (value - mean) / std
    
    @staticmethod
    def mad(data: List[float]) -> float:
        """Calculate Median Absolute Deviation."""
        if not data:
            return 0.0
        median = statistics.median(data)
        return statistics.median([abs(x - median) for x in data])
    
    @staticmethod
    def modified_z_score(value: float, data: List[float]) -> float:
        """Calculate modified Z-score using MAD (more robust)."""
        if not data:
            return 0.0
        median = statistics.median(data)
        mad = StatisticalEngine.mad(data)
        if mad == 0:
            return 0.0
        return 0.6745 * (value - median) / mad


# ==============================================================================
# Anomaly Detector
# ==============================================================================

class AnomalyDetector:
    """
    Anomaly detection engine using multiple statistical methods.
    
    Methods:
    - Z-score based detection
    - Modified Z-score (MAD-based)
    - IQR-based detection
    - Threshold-based detection
    """
    
    def __init__(
        self,
        z_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        sensitivity: float = 0.5
    ):
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
        self.sensitivity = sensitivity
        self.baselines: Dict[str, Dict[str, float]] = {}
    
    def set_baseline(
        self, 
        metric_type: MetricType, 
        mean: float, 
        std: float,
        min_val: float,
        max_val: float
    ) -> None:
        """Set baseline statistics for a metric."""
        self.baselines[metric_type.value] = {
            "mean": mean,
            "std": std,
            "min": min_val,
            "max": max_val
        }
    
    def detect(
        self, 
        metric_type: MetricType,
        value: float,
        historical_data: List[float]
    ) -> AnomalyResult:
        """
        Detect anomaly in a single value given historical context.
        """
        if not historical_data or len(historical_data) < 3:
            return AnomalyResult(
                detected=False,
                metric_type=metric_type,
                actual_value=value,
                expected_value=value,
                deviation=0.0,
                message="Insufficient historical data"
            )
        
        mean = statistics.mean(historical_data)
        std = statistics.stdev(historical_data) if len(historical_data) > 1 else 0.0
        
        # Z-score detection
        z_score = StatisticalEngine.z_score(value, mean, std)
        
        # Modified Z-score (more robust to outliers)
        mod_z_score = StatisticalEngine.modified_z_score(value, historical_data)
        
        # IQR-based detection
        q1 = statistics.quantiles(historical_data, n=4)[0]
        q3 = statistics.quantiles(historical_data, n=4)[2]
        iqr = q3 - q1
        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr
        
        # Determine if anomaly
        is_z_anomaly = abs(z_score) > self.z_threshold
        is_mod_z_anomaly = abs(mod_z_score) > self.z_threshold
        is_iqr_anomaly = value < lower_bound or value > upper_bound
        
        # Combined detection (majority voting)
        anomaly_votes = sum([is_z_anomaly, is_mod_z_anomaly, is_iqr_anomaly])
        is_anomaly = anomaly_votes >= 2  # At least 2 methods agree
        
        if is_anomaly:
            # Determine anomaly type
            if value > mean:
                anomaly_type = AnomalyType.SPIKE
            else:
                anomaly_type = AnomalyType.DROP
            
            severity = min(1.0, abs(z_score) / (self.z_threshold * 2))
            
            return AnomalyResult(
                detected=True,
                anomaly_type=anomaly_type,
                severity=severity,
                metric_type=metric_type,
                actual_value=value,
                expected_value=mean,
                deviation=value - mean,
                message=f"Anomaly detected: {anomaly_type.value} with Z-score {z_score:.2f}"
            )
        
        return AnomalyResult(
            detected=False,
            metric_type=metric_type,
            actual_value=value,
            expected_value=mean,
            deviation=value - mean,
            message="No anomaly detected"
        )
    
    def detect_trend_change(
        self,
        metric_type: MetricType,
        data: List[float],
        window_size: int = 10
    ) -> AnomalyResult:
        """Detect sudden trend changes."""
        if len(data) < window_size * 2:
            return AnomalyResult(
                detected=False,
                metric_type=metric_type,
                actual_value=data[-1] if data else 0,
                expected_value=0,
                deviation=0,
                message="Insufficient data for trend analysis"
            )
        
        # Compare recent window to previous window
        recent = data[-window_size:]
        previous = data[-window_size*2:-window_size]
        
        recent_mean = statistics.mean(recent)
        previous_mean = statistics.mean(previous)
        previous_std = statistics.stdev(previous) if len(previous) > 1 else 1
        
        change_ratio = abs(recent_mean - previous_mean) / max(previous_std, 0.001)
        
        if change_ratio > self.z_threshold:
            return AnomalyResult(
                detected=True,
                anomaly_type=AnomalyType.TREND_CHANGE,
                severity=min(1.0, change_ratio / (self.z_threshold * 2)),
                metric_type=metric_type,
                actual_value=recent_mean,
                expected_value=previous_mean,
                deviation=recent_mean - previous_mean,
                message=f"Trend change detected: {change_ratio:.2f}x deviation"
            )
        
        return AnomalyResult(
            detected=False,
            metric_type=metric_type,
            actual_value=recent_mean,
            expected_value=previous_mean,
            deviation=recent_mean - previous_mean,
            message="No trend change detected"
        )


# ==============================================================================
# Forecaster
# ==============================================================================

class Forecaster:
    """
    Time series forecasting engine.
    
    Methods:
    - Simple Moving Average
    - Exponential Smoothing
    - Linear Regression
    - Holt-Winters (simplified)
    """
    
    def __init__(self, default_method: ForecastMethod = ForecastMethod.EXPONENTIAL_SMOOTHING):
        self.default_method = default_method
    
    def forecast(
        self,
        metric_type: MetricType,
        data: List[float],
        periods: int = 10,
        method: Optional[ForecastMethod] = None
    ) -> ForecastResult:
        """Generate forecast for future periods."""
        method = method or self.default_method
        
        if not data or len(data) < 3:
            return ForecastResult(
                metric_type=metric_type,
                method=method,
                predictions=[],
                confidence_interval=(0.0, 0.0),
                accuracy_score=0.0
            )
        
        if method == ForecastMethod.MOVING_AVERAGE:
            predictions = self._moving_average_forecast(data, periods)
        elif method == ForecastMethod.EXPONENTIAL_SMOOTHING:
            predictions = self._exponential_smoothing_forecast(data, periods)
        elif method == ForecastMethod.LINEAR_REGRESSION:
            predictions = self._linear_regression_forecast(data, periods)
        else:
            predictions = self._exponential_smoothing_forecast(data, periods)
        
        # Calculate confidence interval
        std = statistics.stdev(data) if len(data) > 1 else 0
        mean_pred = statistics.mean([p[1] for p in predictions]) if predictions else 0
        ci_low = mean_pred - 1.96 * std
        ci_high = mean_pred + 1.96 * std
        
        # Calculate simple accuracy score based on recent data fit
        accuracy = self._calculate_accuracy(data, method)
        
        return ForecastResult(
            metric_type=metric_type,
            method=method,
            predictions=predictions,
            confidence_interval=(ci_low, ci_high),
            accuracy_score=accuracy
        )
    
    def _moving_average_forecast(
        self, 
        data: List[float], 
        periods: int,
        window: int = 5
    ) -> List[Tuple[datetime, float]]:
        """Forecast using moving average."""
        if len(data) < window:
            window = len(data)
        
        last_ma = sum(data[-window:]) / window
        now = datetime.utcnow()
        
        predictions = []
        for i in range(periods):
            future_time = now + timedelta(minutes=i + 1)
            predictions.append((future_time, last_ma))
        
        return predictions
    
    def _exponential_smoothing_forecast(
        self,
        data: List[float],
        periods: int,
        alpha: float = 0.3
    ) -> List[Tuple[datetime, float]]:
        """Forecast using exponential smoothing."""
        smoothed = StatisticalEngine.exponential_smoothing(data, alpha)
        last_smoothed = smoothed[-1] if smoothed else data[-1]
        
        now = datetime.utcnow()
        predictions = []
        
        for i in range(periods):
            future_time = now + timedelta(minutes=i + 1)
            predictions.append((future_time, last_smoothed))
        
        return predictions
    
    def _linear_regression_forecast(
        self,
        data: List[float],
        periods: int
    ) -> List[Tuple[datetime, float]]:
        """Forecast using linear regression."""
        x = list(range(len(data)))
        slope, intercept, _ = StatisticalEngine.linear_regression(x, data)
        
        now = datetime.utcnow()
        predictions = []
        
        for i in range(periods):
            future_x = len(data) + i
            future_y = slope * future_x + intercept
            future_time = now + timedelta(minutes=i + 1)
            predictions.append((future_time, future_y))
        
        return predictions
    
    def _calculate_accuracy(
        self, 
        data: List[float], 
        method: ForecastMethod
    ) -> float:
        """Calculate accuracy using holdout validation."""
        if len(data) < 10:
            return 0.5
        
        # Use last 20% as test set
        split = int(len(data) * 0.8)
        train = data[:split]
        test = data[split:]
        
        # Generate predictions for test period
        forecast_result = self.forecast(
            MetricType.LATENCY,  # Dummy, doesn't affect calculation
            train,
            periods=len(test),
            method=method
        )
        
        if not forecast_result.predictions:
            return 0.5
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        predictions = [p[1] for p in forecast_result.predictions][:len(test)]
        mape = 0.0
        for actual, predicted in zip(test, predictions):
            if actual != 0:
                mape += abs((actual - predicted) / actual)
        
        mape = mape / len(test) if test else 1.0
        
        # Convert MAPE to accuracy (1 - MAPE, capped at 0-1)
        accuracy = max(0.0, min(1.0, 1.0 - mape))
        return accuracy


# ==============================================================================
# Predictive Analytics Service
# ==============================================================================

class PredictiveAnalyticsService:
    """
    Main service for predictive analytics.
    
    Provides:
    - Real-time metric ingestion
    - Anomaly detection
    - Forecasting
    - Trend analysis
    - Cost optimization insights
    """
    
    def __init__(
        self,
        history_size: int = 1000,
        anomaly_threshold: float = 3.0
    ):
        self.history_size = history_size
        self.metrics: Dict[str, deque] = {}
        self.anomaly_detector = AnomalyDetector(z_threshold=anomaly_threshold)
        self.forecaster = Forecaster()
        self.insights: List[AnalyticsInsight] = []
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the analytics service."""
        logger.info("Initializing Predictive Analytics Service...")
        
        # Initialize metric storage for each type
        for metric_type in MetricType:
            self.metrics[metric_type.value] = deque(maxlen=self.history_size)
        
        self._initialized = True
        logger.info("✅ Predictive Analytics Service initialized")
    
    async def record_metric(
        self,
        metric_type: MetricType,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[AnomalyResult]:
        """
        Record a metric value and check for anomalies.
        
        Returns anomaly result if anomaly detected.
        """
        if not self._initialized:
            await self.initialize()
        
        data_point = MetricDataPoint(
            timestamp=datetime.utcnow(),
            value=value,
            metadata=metadata or {}
        )
        
        self.metrics[metric_type.value].append(data_point)
        
        # Check for anomaly if we have enough history
        history = [dp.value for dp in self.metrics[metric_type.value]]
        if len(history) >= 10:
            anomaly = self.anomaly_detector.detect(metric_type, value, history[:-1])
            if anomaly.detected:
                await self._generate_anomaly_insight(anomaly)
                return anomaly
        
        return None
    
    async def get_forecast(
        self,
        metric_type: MetricType,
        periods: int = 10,
        method: Optional[ForecastMethod] = None
    ) -> ForecastResult:
        """Get forecast for a metric."""
        history = [dp.value for dp in self.metrics.get(metric_type.value, [])]
        return self.forecaster.forecast(metric_type, history, periods, method)
    
    async def analyze_trend(
        self,
        metric_type: MetricType,
        period_hours: int = 24
    ) -> TrendAnalysis:
        """Analyze trend for a metric over a period."""
        now = datetime.utcnow()
        cutoff = now - timedelta(hours=period_hours)
        
        # Filter data points within period
        data_points = [
            dp for dp in self.metrics.get(metric_type.value, [])
            if dp.timestamp >= cutoff
        ]
        
        if len(data_points) < 5:
            return TrendAnalysis(
                metric_type=metric_type,
                trend="unknown",
                slope=0.0,
                correlation=0.0,
                period_start=cutoff,
                period_end=now,
                insights=["Insufficient data for trend analysis"]
            )
        
        # Calculate trend using linear regression
        values = [dp.value for dp in data_points]
        x = list(range(len(values)))
        slope, intercept, r_squared = StatisticalEngine.linear_regression(x, values)
        
        # Determine trend direction
        if abs(slope) < 0.01 * statistics.mean(values):
            trend = "stable"
        elif slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
        
        # Check for volatility
        std = statistics.stdev(values) if len(values) > 1 else 0
        mean = statistics.mean(values)
        cv = std / mean if mean != 0 else 0  # Coefficient of variation
        
        if cv > 0.3:
            trend = "volatile"
        
        # Generate insights
        insights = []
        if trend == "increasing":
            insights.append(f"{metric_type.value} is trending upward by {slope:.4f} per data point")
        elif trend == "decreasing":
            insights.append(f"{metric_type.value} is trending downward by {abs(slope):.4f} per data point")
        elif trend == "volatile":
            insights.append(f"{metric_type.value} shows high volatility (CV: {cv:.2%})")
        else:
            insights.append(f"{metric_type.value} is stable")
        
        if r_squared > 0.8:
            insights.append(f"Strong linear trend (R²: {r_squared:.2f})")
        
        return TrendAnalysis(
            metric_type=metric_type,
            trend=trend,
            slope=slope,
            correlation=math.sqrt(r_squared) if r_squared >= 0 else 0,
            period_start=cutoff,
            period_end=now,
            insights=insights
        )
    
    async def get_cost_recommendations(self) -> List[CostOptimizationRecommendation]:
        """Generate cost optimization recommendations."""
        recommendations = []
        
        # Analyze token usage
        token_history = [dp.value for dp in self.metrics.get(MetricType.TOKEN_USAGE.value, [])]
        if token_history:
            avg_tokens = statistics.mean(token_history)
            max_tokens = max(token_history)
            
            if max_tokens > avg_tokens * 2:
                recommendations.append(CostOptimizationRecommendation(
                    category="Token Usage",
                    current_cost=avg_tokens * 0.00002,  # Rough estimate
                    potential_savings=0.15,  # 15% potential savings
                    implementation="Implement prompt compression or caching for repeated queries",
                    impact="medium",
                    confidence=0.7
                ))
        
        # Analyze cache hit rate
        cache_history = [dp.value for dp in self.metrics.get(MetricType.CACHE_HIT_RATE.value, [])]
        if cache_history:
            avg_hit_rate = statistics.mean(cache_history)
            
            if avg_hit_rate < 0.5:
                recommendations.append(CostOptimizationRecommendation(
                    category="Caching",
                    current_cost=0.0,
                    potential_savings=0.25,  # 25% potential savings
                    implementation="Increase cache TTL or implement semantic caching",
                    impact="high",
                    confidence=0.8
                ))
        
        # Analyze error rate
        error_history = [dp.value for dp in self.metrics.get(MetricType.ERROR_RATE.value, [])]
        if error_history:
            avg_error_rate = statistics.mean(error_history)
            
            if avg_error_rate > 0.05:  # > 5% error rate
                recommendations.append(CostOptimizationRecommendation(
                    category="Error Reduction",
                    current_cost=avg_error_rate * 100,  # Estimate wasted cost
                    potential_savings=0.10,
                    implementation="Implement retry logic with exponential backoff",
                    impact="medium",
                    confidence=0.75
                ))
        
        return recommendations
    
    async def get_insights(
        self,
        limit: int = 10
    ) -> List[AnalyticsInsight]:
        """Get recent analytics insights."""
        return self.insights[-limit:]
    
    async def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for analytics dashboard."""
        dashboard = {
            "generated_at": datetime.utcnow().isoformat(),
            "metrics_summary": {},
            "forecasts": {},
            "anomalies": [],
            "trends": {},
            "recommendations": []
        }
        
        # Generate summaries for each metric
        for metric_type in MetricType:
            history = [dp.value for dp in self.metrics.get(metric_type.value, [])]
            if history:
                dashboard["metrics_summary"][metric_type.value] = {
                    "current": history[-1],
                    "mean": statistics.mean(history),
                    "min": min(history),
                    "max": max(history),
                    "std": statistics.stdev(history) if len(history) > 1 else 0,
                    "count": len(history)
                }
        
        # Generate forecasts for key metrics
        key_metrics = [MetricType.LATENCY, MetricType.TOKEN_USAGE, MetricType.COST]
        for metric_type in key_metrics:
            forecast = await self.get_forecast(metric_type, periods=10)
            if forecast.predictions:
                dashboard["forecasts"][metric_type.value] = {
                    "predictions": [
                        {"time": p[0].isoformat(), "value": p[1]}
                        for p in forecast.predictions
                    ],
                    "confidence_interval": forecast.confidence_interval,
                    "accuracy": forecast.accuracy_score
                }
        
        # Get trends
        for metric_type in MetricType:
            trend = await self.analyze_trend(metric_type, period_hours=24)
            dashboard["trends"][metric_type.value] = {
                "direction": trend.trend,
                "slope": trend.slope,
                "insights": trend.insights
            }
        
        # Get recommendations
        recommendations = await self.get_cost_recommendations()
        dashboard["recommendations"] = [
            {
                "category": r.category,
                "potential_savings": f"{r.potential_savings:.0%}",
                "implementation": r.implementation,
                "impact": r.impact
            }
            for r in recommendations
        ]
        
        return dashboard
    
    async def _generate_anomaly_insight(self, anomaly: AnomalyResult) -> None:
        """Generate insight from detected anomaly."""
        severity_map = {
            (0, 0.3): "info",
            (0.3, 0.7): "warning",
            (0.7, 1.0): "critical"
        }
        
        severity = "info"
        for (low, high), level in severity_map.items():
            if low <= anomaly.severity < high:
                severity = level
                break
        
        insight = AnalyticsInsight(
            category="Anomaly Detection",
            title=f"{anomaly.anomaly_type.value.title() if anomaly.anomaly_type else 'Anomaly'} Detected in {anomaly.metric_type.value}",
            description=anomaly.message,
            severity=severity,
            recommendation=self._get_anomaly_recommendation(anomaly),
            metric_type=anomaly.metric_type
        )
        
        self.insights.append(insight)
        
        # Keep only recent insights
        if len(self.insights) > 100:
            self.insights = self.insights[-100:]
    
    def _get_anomaly_recommendation(self, anomaly: AnomalyResult) -> str:
        """Get recommendation based on anomaly type."""
        recommendations = {
            MetricType.LATENCY: "Check for resource contention or slow queries",
            MetricType.ERROR_RATE: "Review error logs and implement retry mechanisms",
            MetricType.TOKEN_USAGE: "Review prompts for optimization opportunities",
            MetricType.MEMORY_USAGE: "Check for memory leaks or increase resources",
            MetricType.CPU_USAGE: "Consider scaling or optimizing compute-intensive operations",
            MetricType.COST: "Review usage patterns and implement cost controls",
        }
        
        return recommendations.get(
            anomaly.metric_type, 
            "Review system metrics and logs for root cause"
        )


# ==============================================================================
# Resource Usage Predictor
# ==============================================================================

class ResourceUsagePredictor:
    """
    Specialized predictor for resource usage and capacity planning.
    """
    
    def __init__(self, analytics_service: PredictiveAnalyticsService):
        self.analytics = analytics_service
    
    async def predict_resource_needs(
        self,
        hours_ahead: int = 24
    ) -> Dict[str, Any]:
        """Predict resource needs for the specified period."""
        predictions = {}
        
        # CPU prediction
        cpu_forecast = await self.analytics.get_forecast(
            MetricType.CPU_USAGE,
            periods=hours_ahead * 6,  # 10-min intervals
            method=ForecastMethod.EXPONENTIAL_SMOOTHING
        )
        
        if cpu_forecast.predictions:
            max_predicted_cpu = max(p[1] for p in cpu_forecast.predictions)
            predictions["cpu"] = {
                "max_predicted": max_predicted_cpu,
                "recommendation": "scale_up" if max_predicted_cpu > 80 else "stable",
                "confidence": cpu_forecast.accuracy_score
            }
        
        # Memory prediction
        memory_forecast = await self.analytics.get_forecast(
            MetricType.MEMORY_USAGE,
            periods=hours_ahead * 6,
            method=ForecastMethod.LINEAR_REGRESSION
        )
        
        if memory_forecast.predictions:
            max_predicted_memory = max(p[1] for p in memory_forecast.predictions)
            predictions["memory"] = {
                "max_predicted": max_predicted_memory,
                "recommendation": "scale_up" if max_predicted_memory > 85 else "stable",
                "confidence": memory_forecast.accuracy_score
            }
        
        return predictions
    
    async def estimate_cost(
        self,
        hours_ahead: int = 24
    ) -> Dict[str, Any]:
        """Estimate costs for the specified period."""
        cost_forecast = await self.analytics.get_forecast(
            MetricType.COST,
            periods=hours_ahead,
            method=ForecastMethod.LINEAR_REGRESSION
        )
        
        if not cost_forecast.predictions:
            return {"estimated_cost": 0, "confidence": 0}
        
        total_cost = sum(p[1] for p in cost_forecast.predictions)
        
        return {
            "estimated_cost": total_cost,
            "hourly_average": total_cost / hours_ahead,
            "confidence": cost_forecast.accuracy_score,
            "confidence_interval": cost_forecast.confidence_interval
        }


# ==============================================================================
# Agent Performance Predictor
# ==============================================================================

class AgentPerformancePredictor:
    """
    Specialized predictor for agent performance metrics.
    """
    
    def __init__(self, analytics_service: PredictiveAnalyticsService):
        self.analytics = analytics_service
        self.agent_metrics: Dict[str, deque] = {}
    
    async def record_agent_metric(
        self,
        agent_id: str,
        success: bool,
        latency_ms: float,
        tokens_used: int
    ) -> None:
        """Record agent performance metric."""
        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = deque(maxlen=1000)
        
        self.agent_metrics[agent_id].append({
            "timestamp": datetime.utcnow(),
            "success": success,
            "latency_ms": latency_ms,
            "tokens_used": tokens_used
        })
        
        # Also record to global analytics
        await self.analytics.record_metric(
            MetricType.AGENT_SUCCESS_RATE,
            1.0 if success else 0.0
        )
        await self.analytics.record_metric(MetricType.LATENCY, latency_ms)
        await self.analytics.record_metric(MetricType.TOKEN_USAGE, tokens_used)
    
    async def predict_agent_performance(
        self,
        agent_id: str
    ) -> Dict[str, Any]:
        """Predict agent performance metrics."""
        if agent_id not in self.agent_metrics:
            return {"error": "No data for agent"}
        
        metrics = list(self.agent_metrics[agent_id])
        if len(metrics) < 10:
            return {"error": "Insufficient data"}
        
        # Calculate success rate trend
        success_rates = [1.0 if m["success"] else 0.0 for m in metrics]
        latencies = [m["latency_ms"] for m in metrics]
        
        # Predict future success rate
        x = list(range(len(success_rates)))
        slope, intercept, r2 = StatisticalEngine.linear_regression(x, success_rates)
        
        predicted_success_rate = min(1.0, max(0.0, slope * (len(metrics) + 10) + intercept))
        
        return {
            "agent_id": agent_id,
            "current_success_rate": statistics.mean(success_rates[-10:]),
            "predicted_success_rate": predicted_success_rate,
            "avg_latency_ms": statistics.mean(latencies),
            "latency_trend": "increasing" if slope > 0 else "decreasing",
            "confidence": r2,
            "total_executions": len(metrics)
        }
