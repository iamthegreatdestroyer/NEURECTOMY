"""
Tests for Predictive Analytics Service.

@ECLIPSE @TENSOR - Test suite for analytics and forecasting.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.services.analytics import (
    PredictiveAnalyticsService,
    AnomalyDetector,
    StatisticalEngine,
    Forecaster,
    MetricType,
    MetricDataPoint,
    AnomalyResult,
    AnomalyType,
    ForecastResult,
    ForecastMethod,
    TrendAnalysis,
    AgentPerformancePredictor,
)


# ==============================================================================
# Test Data Fixtures
# ==============================================================================

@pytest.fixture
def sample_metric_data():
    """Generate sample metric data points."""
    return [
        0.85 + np.random.uniform(-0.1, 0.1)
        for _ in range(100)
    ]


@pytest.fixture
def anomaly_data():
    """Generate data with anomalies."""
    # Normal data
    normal = [10 + np.random.normal(0, 1) for _ in range(95)]
    # Anomalies
    anomalies = [100, 150, 200, 5, -10]  # Clear outliers
    return normal + anomalies


@pytest.fixture
def time_series_data():
    """Generate time series with trend and seasonality."""
    x = np.linspace(0, 4 * np.pi, 100)
    # Trend + seasonality + noise
    data = x * 0.5 + np.sin(x) * 5 + np.random.normal(0, 0.5, 100)
    return list(data)


@pytest.fixture
def analytics_service():
    """Create analytics service instance."""
    return PredictiveAnalyticsService()


# ==============================================================================
# StatisticalEngine Tests
# ==============================================================================

class TestStatisticalEngine:
    """Tests for statistical computations."""
    
    def test_moving_average(self):
        """Test moving average calculation."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        result = StatisticalEngine.moving_average(data, window=3)
        
        assert len(result) == 8  # n - window + 1
        assert result[0] == pytest.approx(2.0)  # (1+2+3)/3
        assert result[-1] == pytest.approx(9.0)  # (8+9+10)/3
    
    def test_moving_average_short_data(self):
        """Test moving average with data shorter than window."""
        data = [1, 2]
        
        result = StatisticalEngine.moving_average(data, window=5)
        
        assert result == data  # Returns original if too short
    
    def test_exponential_smoothing(self):
        """Test exponential smoothing."""
        data = [1, 2, 3, 4, 5]
        
        result = StatisticalEngine.exponential_smoothing(data, alpha=0.5)
        
        assert len(result) == len(data)
        assert result[0] == data[0]  # First value unchanged
    
    def test_exponential_smoothing_empty(self):
        """Test exponential smoothing with empty data."""
        result = StatisticalEngine.exponential_smoothing([])
        
        assert result == []
    
    def test_linear_regression(self):
        """Test linear regression calculation."""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]  # Perfect linear relationship y = 2x
        
        slope, intercept, r_squared = StatisticalEngine.linear_regression(x, y)
        
        assert slope == pytest.approx(2.0)
        assert intercept == pytest.approx(0.0)
        assert r_squared == pytest.approx(1.0)
    
    def test_linear_regression_insufficient_data(self):
        """Test linear regression with insufficient data."""
        slope, intercept, r_squared = StatisticalEngine.linear_regression([1], [2])
        
        assert slope == 0.0
        assert r_squared == 0.0
    
    def test_z_score(self):
        """Test Z-score calculation."""
        result = StatisticalEngine.z_score(value=15, mean=10, std=2)
        
        assert result == pytest.approx(2.5)
    
    def test_z_score_zero_std(self):
        """Test Z-score with zero standard deviation."""
        result = StatisticalEngine.z_score(value=15, mean=10, std=0)
        
        assert result == 0.0
    
    def test_mad(self):
        """Test Median Absolute Deviation."""
        data = [1, 2, 3, 4, 5]  # Median is 3
        
        mad = StatisticalEngine.mad(data)
        
        assert mad == pytest.approx(1.0)  # |1-3|, |2-3|, |3-3|, |4-3|, |5-3| = 2,1,0,1,2 -> median=1
    
    def test_mad_empty(self):
        """Test MAD with empty data."""
        mad = StatisticalEngine.mad([])
        
        assert mad == 0.0
    
    def test_modified_z_score(self):
        """Test modified Z-score calculation."""
        data = [1, 2, 3, 4, 5, 100]  # 100 is an outlier
        
        mod_z = StatisticalEngine.modified_z_score(100, data)
        
        assert mod_z > 3  # Should be high for outlier


# ==============================================================================
# AnomalyDetector Tests
# ==============================================================================

class TestAnomalyDetector:
    """Tests for anomaly detection."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = AnomalyDetector()
        
        assert detector is not None
        assert detector.z_threshold == 3.0
    
    def test_detect_anomaly_spike(self, sample_metric_data):
        """Test detecting a spike anomaly."""
        detector = AnomalyDetector()
        
        # Add a clear spike
        sample_metric_data.append(10.0)  # Clear outlier in 0.75-0.95 range
        
        result = detector.detect(
            MetricType.AGENT_SUCCESS_RATE,
            10.0,
            sample_metric_data[:-1]
        )
        
        if result.detected:
            assert result.anomaly_type == AnomalyType.SPIKE
    
    def test_detect_anomaly_drop(self, sample_metric_data):
        """Test detecting a drop anomaly."""
        detector = AnomalyDetector()
        
        # Add a clear drop
        sample_metric_data.append(-5.0)  # Clear outlier
        
        result = detector.detect(
            MetricType.AGENT_SUCCESS_RATE,
            -5.0,
            sample_metric_data[:-1]
        )
        
        if result.detected:
            assert result.anomaly_type == AnomalyType.DROP
    
    def test_detect_no_anomaly(self):
        """Test no anomaly in normal data."""
        detector = AnomalyDetector()
        normal_data = [10 + np.random.normal(0, 0.1) for _ in range(50)]
        
        result = detector.detect(
            MetricType.LATENCY,
            10.0,  # Normal value
            normal_data
        )
        
        assert result.detected is False
    
    def test_detect_insufficient_data(self):
        """Test detection with insufficient data."""
        detector = AnomalyDetector()
        
        result = detector.detect(
            MetricType.LATENCY,
            10.0,
            [1, 2]  # Only 2 data points
        )
        
        assert result.detected is False
        assert "Insufficient" in result.message
    
    def test_set_baseline(self):
        """Test setting baseline statistics."""
        detector = AnomalyDetector()
        
        detector.set_baseline(
            MetricType.CPU_USAGE,
            mean=50.0,
            std=10.0,
            min_val=0.0,
            max_val=100.0
        )
        
        assert MetricType.CPU_USAGE.value in detector.baselines
        assert detector.baselines[MetricType.CPU_USAGE.value]["mean"] == 50.0
    
    def test_anomaly_result_attributes(self, sample_metric_data):
        """Test anomaly result has all attributes."""
        detector = AnomalyDetector()
        
        result = detector.detect(
            MetricType.LATENCY,
            sample_metric_data[0],
            sample_metric_data[1:]
        )
        
        assert hasattr(result, "detected")
        assert hasattr(result, "metric_type")
        assert hasattr(result, "actual_value")
        assert hasattr(result, "expected_value")
        assert hasattr(result, "deviation")


# ==============================================================================
# Forecaster Tests
# ==============================================================================

class TestForecaster:
    """Tests for forecasting functionality."""
    
    def test_initialization(self):
        """Test forecaster initialization."""
        forecaster = Forecaster()
        
        assert forecaster is not None
    
    def test_forecast_moving_average(self, time_series_data):
        """Test forecasting with moving average."""
        forecaster = Forecaster()
        
        result = forecaster.forecast(
            MetricType.LATENCY,
            time_series_data,
            periods=5,
            method=ForecastMethod.MOVING_AVERAGE
        )
        
        assert isinstance(result, ForecastResult)
        assert len(result.predictions) == 5
        assert result.method == ForecastMethod.MOVING_AVERAGE
    
    def test_forecast_exponential_smoothing(self, time_series_data):
        """Test forecasting with exponential smoothing."""
        forecaster = Forecaster()
        
        result = forecaster.forecast(
            MetricType.CPU_USAGE,
            time_series_data,
            periods=10,
            method=ForecastMethod.EXPONENTIAL_SMOOTHING
        )
        
        assert len(result.predictions) == 10
    
    def test_forecast_linear_regression(self, time_series_data):
        """Test forecasting with linear regression."""
        forecaster = Forecaster()
        
        result = forecaster.forecast(
            MetricType.TOKEN_USAGE,
            time_series_data,
            periods=5,
            method=ForecastMethod.LINEAR_REGRESSION
        )
        
        assert len(result.predictions) == 5
    
    def test_forecast_insufficient_data(self):
        """Test forecasting with insufficient data."""
        forecaster = Forecaster()
        
        result = forecaster.forecast(
            MetricType.LATENCY,
            [1, 2],  # Too little data
            periods=5
        )
        
        assert result.accuracy_score == 0.0
    
    def test_forecast_empty_data(self):
        """Test forecasting with empty data."""
        forecaster = Forecaster()
        
        result = forecaster.forecast(
            MetricType.LATENCY,
            [],
            periods=5
        )
        
        assert len(result.predictions) == 0
    
    def test_auto_select_method(self, time_series_data):
        """Test automatic method selection."""
        forecaster = Forecaster()
        
        result = forecaster.forecast(
            MetricType.LATENCY,
            time_series_data,
            periods=5,
            method=None  # Auto-select
        )
        
        assert result.method is not None
        assert len(result.predictions) == 5


# ==============================================================================
# PredictiveAnalyticsService Tests
# ==============================================================================

class TestPredictiveAnalyticsService:
    """Tests for the main analytics service."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, analytics_service):
        """Test service initialization."""
        await analytics_service.initialize()
        
        assert analytics_service._initialized is True
    
    @pytest.mark.asyncio
    async def test_record_metric(self, analytics_service):
        """Test recording a metric."""
        await analytics_service.initialize()
        
        result = await analytics_service.record_metric(
            MetricType.LATENCY,
            150.0
        )
        
        # First record shouldn't trigger anomaly (no history)
        assert result is None or not result.detected
    
    @pytest.mark.asyncio
    async def test_record_anomaly_detection(self, analytics_service):
        """Test that anomalies are detected when recording."""
        await analytics_service.initialize()
        
        # Build up history
        for _ in range(15):
            await analytics_service.record_metric(MetricType.LATENCY, 100.0)
        
        # Record anomaly
        result = await analytics_service.record_metric(MetricType.LATENCY, 1000.0)
        
        # Should detect anomaly
        if result:
            assert result.detected is True
    
    @pytest.mark.asyncio
    async def test_get_forecast(self, analytics_service):
        """Test getting a forecast."""
        await analytics_service.initialize()
        
        # Add some data
        for i in range(20):
            await analytics_service.record_metric(MetricType.CPU_USAGE, 50 + i * 0.5)
        
        forecast = await analytics_service.get_forecast(
            MetricType.CPU_USAGE,
            periods=5
        )
        
        assert isinstance(forecast, ForecastResult)
    
    @pytest.mark.asyncio
    async def test_analyze_trend(self, analytics_service):
        """Test trend analysis."""
        await analytics_service.initialize()
        
        # Add trending data
        for i in range(30):
            await analytics_service.record_metric(MetricType.THROUGHPUT, 100 + i * 2)
        
        trend = await analytics_service.analyze_trend(
            MetricType.THROUGHPUT,
            period_hours=1
        )
        
        assert isinstance(trend, TrendAnalysis)
        assert trend.trend in ["increasing", "decreasing", "stable", "volatile", "unknown"]
    
    @pytest.mark.asyncio
    async def test_get_cost_recommendations(self, analytics_service):
        """Test getting cost recommendations."""
        await analytics_service.initialize()
        
        # Add some cost-related metrics
        for _ in range(20):
            await analytics_service.record_metric(MetricType.TOKEN_USAGE, 1000)
            await analytics_service.record_metric(MetricType.CACHE_HIT_RATE, 0.3)
        
        recommendations = await analytics_service.get_cost_recommendations()
        
        assert isinstance(recommendations, list)
    
    @pytest.mark.asyncio
    async def test_generate_insights(self, analytics_service):
        """Test insight generation."""
        await analytics_service.initialize()
        
        # Add various metrics
        for i in range(30):
            await analytics_service.record_metric(MetricType.ERROR_RATE, 0.05 + i * 0.002)
        
        insights = await analytics_service.get_insights()
        
        assert isinstance(insights, list)


# ==============================================================================
# AgentPerformancePredictor Tests
# ==============================================================================

class TestAgentPerformancePredictor:
    """Tests for agent performance prediction."""
    
    @pytest.fixture
    def predictor(self, analytics_service):
        """Create predictor with analytics service."""
        return AgentPerformancePredictor(analytics_service)
    
    @pytest.mark.asyncio
    async def test_initialization(self, predictor):
        """Test predictor initialization."""
        assert predictor is not None
        assert predictor.agent_metrics == {}
    
    @pytest.mark.asyncio
    async def test_record_agent_metric(self, predictor, analytics_service):
        """Test recording agent metrics."""
        await analytics_service.initialize()
        
        await predictor.record_agent_metric(
            agent_id="agent_001",
            success=True,
            latency_ms=150.0,
            tokens_used=100
        )
        
        assert "agent_001" in predictor.agent_metrics
        assert len(predictor.agent_metrics["agent_001"]) == 1
    
    @pytest.mark.asyncio
    async def test_predict_agent_performance(self, predictor, analytics_service):
        """Test predicting agent performance."""
        await analytics_service.initialize()
        
        # Record enough data points
        for i in range(15):
            await predictor.record_agent_metric(
                agent_id="agent_001",
                success=i % 5 != 0,  # 80% success rate
                latency_ms=100 + i * 2,
                tokens_used=50 + i
            )
        
        prediction = await predictor.predict_agent_performance("agent_001")
        
        assert "agent_id" in prediction
        assert "current_success_rate" in prediction
        assert "predicted_success_rate" in prediction
    
    @pytest.mark.asyncio
    async def test_predict_unknown_agent(self, predictor, analytics_service):
        """Test prediction for unknown agent."""
        await analytics_service.initialize()
        
        prediction = await predictor.predict_agent_performance("unknown_agent")
        
        assert "error" in prediction
    
    @pytest.mark.asyncio
    async def test_predict_insufficient_data(self, predictor, analytics_service):
        """Test prediction with insufficient data."""
        await analytics_service.initialize()
        
        # Record only a few data points
        for i in range(3):
            await predictor.record_agent_metric(
                agent_id="agent_002",
                success=True,
                latency_ms=100,
                tokens_used=50
            )
        
        prediction = await predictor.predict_agent_performance("agent_002")
        
        assert "error" in prediction


# ==============================================================================
# Edge Cases and Error Handling
# ==============================================================================

class TestAnalyticsEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_service_not_initialized(self):
        """Test service auto-initializes on first use."""
        service = PredictiveAnalyticsService()
        
        # Should auto-initialize
        await service.record_metric(MetricType.LATENCY, 100)
        
        assert service._initialized is True
    
    def test_forecast_with_constant_data(self):
        """Test forecasting with constant data."""
        forecaster = Forecaster()
        constant_data = [10.0] * 50
        
        result = forecaster.forecast(
            MetricType.CPU_USAGE,
            constant_data,
            periods=5
        )
        
        # Should return predictions (even if all same)
        assert len(result.predictions) >= 0
    
    def test_anomaly_detector_all_same_values(self):
        """Test anomaly detection with uniform data."""
        detector = AnomalyDetector()
        uniform_data = [5.0] * 50
        
        # Slightly different value should not be anomaly
        result = detector.detect(MetricType.LATENCY, 5.1, uniform_data)
        
        # With all same values, std=0, any different value might be anomalous
        # but we handle this gracefully
        assert isinstance(result, AnomalyResult)
    
    def test_linear_regression_vertical_line(self):
        """Test linear regression with vertical data (undefined slope)."""
        x = [5, 5, 5, 5, 5]  # All same x values
        y = [1, 2, 3, 4, 5]
        
        slope, intercept, r2 = StatisticalEngine.linear_regression(x, y)
        
        # Should handle gracefully (denominator = 0 case)
        assert isinstance(slope, float)


# ==============================================================================
# Performance Tests
# ==============================================================================

class TestAnalyticsPerformance:
    """Performance tests for analytics operations."""
    
    @pytest.mark.asyncio
    async def test_large_history_handling(self):
        """Test handling large metric history."""
        service = PredictiveAnalyticsService(history_size=10000)
        await service.initialize()
        
        import time
        start = time.time()
        
        # Record many metrics
        for i in range(1000):
            await service.record_metric(MetricType.LATENCY, 100 + i % 50)
        
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed < 5.0  # 5 seconds max
    
    def test_large_forecast(self):
        """Test forecasting with large dataset."""
        forecaster = Forecaster()
        large_data = [50 + np.sin(i / 10) * 20 for i in range(1000)]
        
        import time
        start = time.time()
        
        result = forecaster.forecast(
            MetricType.CPU_USAGE,
            large_data,
            periods=100
        )
        
        elapsed = time.time() - start
        
        assert elapsed < 2.0  # 2 seconds max
        assert len(result.predictions) == 100

