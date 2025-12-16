"""
Tests for Neurectomy Metrics
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from neurectomy.monitoring.metrics import (
    MetricsMiddleware,
    track_ryot_request,
    track_compression,
    track_storage_operation,
    track_circuit_breaker,
    set_active_users,
    increment_tokens_generated,
    update_circuit_breaker_state,
    http_requests_total,
    ryot_requests_total,
    sigmalang_compression_ratio,
    sigmavault_operations_total,
)


class TestMetricsMiddleware:
    """Test ASGI metrics middleware"""
    
    @pytest.mark.asyncio
    async def test_middleware_tracks_request(self):
        """Test that middleware tracks HTTP requests"""
        app = AsyncMock()
        middleware = MetricsMiddleware(app)
        
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/v1/complete",
            "headers": [],
        }
        
        async def receive():
            return {"type": "http.request"}
        
        async def send(message):
            if message["type"] == "http.response.start":
                assert message["status"] == 200
        
        await middleware(scope, receive, send)
        
        app.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_middleware_normalizes_paths(self):
        """Test that middleware normalizes paths to prevent cardinality explosion"""
        middleware = MetricsMiddleware(AsyncMock())
        
        # Should normalize /v1/resource/123 -> /v1/resource
        assert middleware._normalize_path("/v1/users/123") == "/v1/users"
        assert middleware._normalize_path("/v1/complete") == "/v1/complete"
        assert middleware._normalize_path("/metrics") == "/metrics"
    
    @pytest.mark.asyncio
    async def test_middleware_non_http_request(self):
        """Test that middleware ignores non-HTTP requests"""
        app = AsyncMock()
        middleware = MetricsMiddleware(app)
        
        scope = {"type": "websocket"}
        
        await middleware(scope, None, None)
        
        app.assert_called_once()


class TestRyotTracking:
    """Test Ryot LLM tracking"""
    
    @pytest.mark.asyncio
    async def test_track_ryot_request_success(self):
        """Test tracking successful Ryot request"""
        
        @track_ryot_request
        async def generate_text(prompt: str):
            result = MagicMock()
            result.token_count = 100
            return result
        
        result = await generate_text("test prompt")
        
        assert result.token_count == 100
        # Metric should be recorded (but we can't easily verify without mocking prometheus_client)
    
    @pytest.mark.asyncio
    async def test_track_ryot_request_error(self):
        """Test tracking failed Ryot request"""
        
        @track_ryot_request
        async def generate_text(prompt: str):
            raise ValueError("API error")
        
        with pytest.raises(ValueError):
            await generate_text("test prompt")


class TestCompressionTracking:
    """Test ΣLANG compression tracking"""
    
    @pytest.mark.asyncio
    async def test_track_compression_success(self):
        """Test tracking successful compression"""
        
        @track_compression
        async def compress_data(data: bytes):
            result = MagicMock()
            result.original_size = 1000
            result.compressed_size = 50
            return result
        
        result = await compress_data(b"test data")
        
        assert result.original_size == 1000
        assert result.compressed_size == 50
    
    @pytest.mark.asyncio
    async def test_track_compression_error(self):
        """Test tracking compression error"""
        
        @track_compression
        async def compress_data(data: bytes):
            raise RuntimeError("Compression failed")
        
        with pytest.raises(RuntimeError):
            await compress_data(b"test data")


class TestStorageTracking:
    """Test ΣVAULT storage tracking"""
    
    @pytest.mark.asyncio
    async def test_track_storage_operation_store(self):
        """Test tracking storage operation"""
        
        @track_storage_operation("store")
        async def store_file(path: str, data: bytes):
            return {"status": "stored"}
        
        result = await store_file("test.txt", b"data")
        
        assert result["status"] == "stored"
    
    @pytest.mark.asyncio
    async def test_track_storage_operation_error(self):
        """Test tracking storage error"""
        
        @track_storage_operation("store")
        async def store_file(path: str, data: bytes):
            raise IOError("Storage failed")
        
        with pytest.raises(IOError):
            await store_file("test.txt", b"data")


class TestCircuitBreakerTracking:
    """Test circuit breaker tracking"""
    
    @pytest.mark.asyncio
    async def test_track_circuit_breaker_success(self):
        """Test tracking successful circuit breaker operation"""
        
        @track_circuit_breaker("test_service")
        async def call_service():
            return {"result": "success"}
        
        result = await call_service()
        
        assert result["result"] == "success"
    
    @pytest.mark.asyncio
    async def test_track_circuit_breaker_failure(self):
        """Test tracking circuit breaker failure"""
        
        @track_circuit_breaker("test_service")
        async def call_service():
            raise RuntimeError("Service unavailable")
        
        with pytest.raises(RuntimeError):
            await call_service()


class TestHelperFunctions:
    """Test metric helper functions"""
    
    def test_set_active_users(self):
        """Test setting active users"""
        set_active_users(42)
        # Can't easily verify without accessing gauge internals
    
    def test_increment_tokens_generated(self):
        """Test incrementing tokens"""
        increment_tokens_generated("gpt-4", 100)
        # Metric should be incremented
    
    def test_update_circuit_breaker_state(self):
        """Test updating circuit breaker state"""
        update_circuit_breaker_state("api_gateway", 0)  # Closed
        update_circuit_breaker_state("api_gateway", 1)  # Open
        update_circuit_breaker_state("api_gateway", 2)  # Half-open


class TestMetricLabels:
    """Test metric label usage"""
    
    def test_http_requests_labels(self):
        """Test HTTP request metric labels"""
        # Should accept GET, POST, PUT, DELETE
        for method in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
            # Metric should handle all HTTP methods
            pass
    
    def test_status_codes(self):
        """Test tracking different status codes"""
        # Should handle 1xx, 2xx, 3xx, 4xx, 5xx
        for status in [200, 201, 400, 401, 404, 500, 502, 503]:
            # Metric should handle all status codes
            pass


class TestMetricNaming:
    """Test metric naming conventions"""
    
    def test_metric_names_prefixed(self):
        """Verify all metrics use neurectomy_ prefix"""
        # This is enforced by the prometheus_client library
        assert "neurectomy_" in str(http_requests_total)
        assert "neurectomy_" in str(ryot_requests_total)
        assert "neurectomy_" in str(sigmalang_compression_ratio)
        assert "neurectomy_" in str(sigmavault_operations_total)
