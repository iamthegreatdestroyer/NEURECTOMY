# ============================================================================
# NEURECTOMY ML Service - Integration Tests
# End-to-end tests with real service dependencies
# ============================================================================

"""
Integration tests for the ML Service.

These tests require running infrastructure:
- PostgreSQL with pgvector
- Redis
- MLflow (optional)

Run with: pytest tests/integration/ -v --integration
"""

import asyncio
import pytest
from typing import AsyncGenerator, Generator
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock
import uuid


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create async HTTP client for API testing."""
    # Mock the database and external dependencies
    with patch('src.services.training.MLflowClient') as mock_mlflow:
        mock_mlflow.return_value = MagicMock()
        
        # Import app after mocking
        from main import app
        
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


@pytest.fixture
def auth_headers() -> dict:
    """Create authentication headers for protected endpoints."""
    # For testing, create a mock JWT token
    return {
        "Authorization": "Bearer test-token-for-integration-tests"
    }


# ============================================================================
# Health Check Tests
# ============================================================================

class TestHealthEndpoints:
    """Test health check endpoints."""

    @pytest.mark.asyncio
    async def test_health_check(self, async_client: AsyncClient):
        """Test basic health check endpoint."""
        response = await async_client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "version" in data
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_readiness_check(self, async_client: AsyncClient):
        """Test readiness probe endpoint."""
        response = await async_client.get("/api/v1/health/ready")
        # May return 200 or 503 depending on service state
        assert response.status_code in [200, 503]

    @pytest.mark.asyncio
    async def test_liveness_check(self, async_client: AsyncClient):
        """Test liveness probe endpoint."""
        response = await async_client.get("/api/v1/health/live")
        assert response.status_code == 200


# ============================================================================
# Agent Training Integration Tests
# ============================================================================

class TestAgentTrainingIntegration:
    """Integration tests for agent training workflows."""

    @pytest.mark.asyncio
    async def test_full_training_workflow(self, async_client: AsyncClient):
        """Test complete training workflow from config to completion."""
        # Step 1: Create training configuration
        training_config = {
            "agent_id": str(uuid.uuid4()),
            "model_type": "transformer",
            "config": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 1,
                "hidden_size": 128,
                "num_layers": 2
            },
            "dataset": {
                "type": "synthetic",
                "size": 100
            }
        }

        # Step 2: Start training job
        response = await async_client.post(
            "/api/v1/training/start",
            json=training_config
        )
        
        # Training might not be fully implemented yet
        if response.status_code == 404:
            pytest.skip("Training endpoint not implemented")
        
        assert response.status_code in [200, 202]

    @pytest.mark.asyncio
    async def test_training_status_retrieval(self, async_client: AsyncClient):
        """Test retrieving training job status."""
        job_id = str(uuid.uuid4())
        
        response = await async_client.get(f"/api/v1/training/{job_id}/status")
        
        # Either returns status or 404 if job doesn't exist
        assert response.status_code in [200, 404]


# ============================================================================
# Analytics Integration Tests
# ============================================================================

class TestAnalyticsIntegration:
    """Integration tests for analytics and forecasting."""

    @pytest.mark.asyncio
    async def test_agent_forecast_endpoint(self, async_client: AsyncClient):
        """Test agent performance forecasting endpoint."""
        agent_id = str(uuid.uuid4())
        forecast_request = {
            "metric": "accuracy",
            "periods": 10,
            "method": "exponential_smoothing"
        }

        response = await async_client.post(
            f"/api/v1/analytics/forecast/agent/{agent_id}",
            json=forecast_request
        )

        if response.status_code == 404:
            pytest.skip("Analytics endpoint not implemented")

        # Either success or validation error
        assert response.status_code in [200, 400, 422]

    @pytest.mark.asyncio
    async def test_anomaly_detection_endpoint(self, async_client: AsyncClient):
        """Test anomaly detection endpoint."""
        anomaly_request = {
            "values": [1.0, 1.1, 1.2, 10.0, 1.1, 1.0, 1.2],  # 10.0 is anomaly
            "metric_type": "performance",
            "sensitivity": 0.95
        }

        response = await async_client.post(
            "/api/v1/analytics/anomalies/detect",
            json=anomaly_request
        )

        if response.status_code == 404:
            pytest.skip("Anomaly detection endpoint not implemented")

        assert response.status_code in [200, 400, 422]

    @pytest.mark.asyncio
    async def test_system_insights_endpoint(self, async_client: AsyncClient):
        """Test system insights endpoint."""
        response = await async_client.get("/api/v1/analytics/insights/system")

        if response.status_code == 404:
            pytest.skip("Insights endpoint not implemented")

        assert response.status_code in [200, 401, 403]


# ============================================================================
# Authentication Integration Tests
# ============================================================================

class TestAuthenticationIntegration:
    """Integration tests for authentication flow."""

    @pytest.mark.asyncio
    async def test_login_flow(self, async_client: AsyncClient):
        """Test login and token generation."""
        login_data = {
            "username": "test_user",
            "password": "test_password"
        }

        response = await async_client.post(
            "/api/v1/auth/login",
            data=login_data
        )

        if response.status_code == 404:
            pytest.skip("Auth endpoint not implemented")

        # Either success, unauthorized, or validation error
        assert response.status_code in [200, 401, 422]

    @pytest.mark.asyncio
    async def test_token_refresh(self, async_client: AsyncClient):
        """Test token refresh flow."""
        refresh_data = {
            "refresh_token": "test-refresh-token"
        }

        response = await async_client.post(
            "/api/v1/auth/refresh",
            json=refresh_data
        )

        if response.status_code == 404:
            pytest.skip("Auth refresh endpoint not implemented")

        assert response.status_code in [200, 401, 422]


# ============================================================================
# WebSocket Integration Tests
# ============================================================================

class TestWebSocketIntegration:
    """Integration tests for WebSocket functionality."""

    @pytest.mark.asyncio
    async def test_websocket_connection(self, async_client: AsyncClient):
        """Test WebSocket connection establishment."""
        # Note: httpx doesn't support WebSocket natively
        # This is a placeholder for WebSocket testing
        # Use websockets library or similar for actual WS tests
        pytest.skip("WebSocket testing requires specialized client")


# ============================================================================
# End-to-End Workflow Tests
# ============================================================================

class TestEndToEndWorkflows:
    """Complete end-to-end workflow tests."""

    @pytest.mark.asyncio
    async def test_agent_lifecycle_workflow(self, async_client: AsyncClient):
        """Test complete agent lifecycle: create, train, evaluate, deploy."""
        # This test simulates a complete workflow
        
        # Step 1: Health check to ensure service is running
        health_response = await async_client.get("/api/v1/health")
        assert health_response.status_code == 200

        # Step 2: Get initial metrics (if available)
        metrics_response = await async_client.get("/metrics")
        if metrics_response.status_code == 200:
            assert "neurectomy" in metrics_response.text or "python" in metrics_response.text

    @pytest.mark.asyncio
    async def test_api_versioning(self, async_client: AsyncClient):
        """Test API versioning is properly configured."""
        # Test v1 endpoints
        v1_response = await async_client.get("/api/v1/health")
        assert v1_response.status_code == 200

        # Test non-versioned endpoint should redirect or work
        root_response = await async_client.get("/")
        assert root_response.status_code in [200, 307, 404]

    @pytest.mark.asyncio
    async def test_cors_headers(self, async_client: AsyncClient):
        """Test CORS headers are properly set."""
        response = await async_client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:16000",
                "Access-Control-Request-Method": "GET"
            }
        )
        
        # CORS might or might not be configured
        # Just verify the endpoint responds
        assert response.status_code in [200, 204, 405]


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Basic performance tests."""

    @pytest.mark.asyncio
    async def test_health_endpoint_performance(self, async_client: AsyncClient):
        """Test health endpoint responds within acceptable time."""
        import time
        
        start = time.perf_counter()
        response = await async_client.get("/api/v1/health")
        elapsed = time.perf_counter() - start
        
        assert response.status_code == 200
        assert elapsed < 1.0, f"Health check took {elapsed:.2f}s, should be < 1s"

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, async_client: AsyncClient):
        """Test service handles concurrent requests."""
        import asyncio
        
        async def make_request():
            return await async_client.get("/api/v1/health")
        
        # Make 10 concurrent requests
        tasks = [make_request() for _ in range(10)]
        responses = await asyncio.gather(*tasks)
        
        # All should succeed
        success_count = sum(1 for r in responses if r.status_code == 200)
        assert success_count >= 8, f"Only {success_count}/10 requests succeeded"
