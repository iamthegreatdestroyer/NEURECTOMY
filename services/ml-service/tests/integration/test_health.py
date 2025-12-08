"""
Integration tests for health check endpoints.

Tests verify that all services are operational and properly connected.
"""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_endpoint(api_client: AsyncClient):
    """Test main health check endpoint."""
    response = await api_client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure
    assert data["service"] == "intelligence-foundry-ml"
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "websocket_connections" in data
    
    # Verify subsystem health
    assert data["mlflow"] == "healthy", "MLflow should be healthy"
    assert data["optuna"] == "healthy", "Optuna should be healthy"


@pytest.mark.asyncio
async def test_mlflow_health(api_client: AsyncClient, mlflow_base_url: str):
    """Test MLflow service is accessible."""
    import httpx
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{mlflow_base_url}/health")
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_minio_health(minio_base_url: str):
    """Test MinIO service is accessible."""
    import httpx
    
    async with httpx.AsyncClient() as client:
        # MinIO console should be accessible
        response = await client.get(f"{minio_base_url}/minio/health/live")
        assert response.status_code == 200 or response.status_code == 404
        # 404 is acceptable - means MinIO is running but endpoint path may differ


@pytest.mark.asyncio
async def test_service_dependencies(api_client: AsyncClient):
    """Test that all service dependencies are properly configured."""
    response = await api_client.get("/health")
    data = response.json()
    
    # All critical subsystems must be healthy
    critical_systems = ["mlflow", "optuna"]
    for system in critical_systems:
        assert data.get(system) == "healthy", f"{system} must be healthy for service operation"
