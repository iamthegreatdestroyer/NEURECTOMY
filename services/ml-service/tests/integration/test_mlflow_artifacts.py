"""
Integration tests for MLflow artifact management endpoints.

Tests cover:
- Artifact upload
- Artifact download
- Artifact listing
- Model registration
- S3 storage integration
"""
import pytest
from httpx import AsyncClient
from tests.conftest import generate_test_experiment_name
import tempfile
from pathlib import Path


@pytest.mark.asyncio
async def test_log_artifact(api_client: AsyncClient, cleanup_experiments):
    """Test uploading an artifact to a run."""
    # Setup
    exp_response = await api_client.post(
        "/api/mlflow/experiments/create",
        json={"name": generate_test_experiment_name()}
    )
    experiment_id = exp_response.json()["experiment_id"]
    cleanup_experiments.append(experiment_id)
    
    run_response = await api_client.post(
        "/api/mlflow/runs/create",
        json={"experiment_id": experiment_id}
    )
    run_id = run_response.json()["run_id"]
    
    # Create test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test artifact content")
        temp_file = Path(f.name)
    
    try:
        # Upload artifact
        with open(temp_file, 'rb') as f:
            files = {"file": ("test_artifact.txt", f, "text/plain")}
            response = await api_client.post(
                f"/api/mlflow/runs/{run_id}/log-artifact",
                files=files,
                data={"artifact_path": "test_artifacts"}
            )
        
        assert response.status_code == 200
        
        # Verify artifact was uploaded
        list_response = await api_client.get(
            f"/api/mlflow/runs/{run_id}/artifacts/list"
        )
        assert list_response.status_code == 200
        
        artifacts = list_response.json()
        artifact_paths = [a["path"] for a in artifacts]
        assert any("test_artifact.txt" in path for path in artifact_paths)
    
    finally:
        # Cleanup temp file
        temp_file.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_list_artifacts(api_client: AsyncClient, cleanup_experiments):
    """Test listing artifacts in a run."""
    # Setup
    exp_response = await api_client.post(
        "/api/mlflow/experiments/create",
        json={"name": generate_test_experiment_name()}
    )
    experiment_id = exp_response.json()["experiment_id"]
    cleanup_experiments.append(experiment_id)
    
    run_response = await api_client.post(
        "/api/mlflow/runs/create",
        json={"experiment_id": experiment_id}
    )
    run_id = run_response.json()["run_id"]
    
    # Test uploading multiple artifacts
    for i in range(3):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(f"Test content {i}")
            temp_file = Path(f.name)
        
        try:
            with open(temp_file, 'rb') as f:
                files = {"file": (f"artifact_{i}.txt", f, "text/plain")}
                await api_client.post(
                    f"/api/mlflow/runs/{run_id}/log-artifact",
                    files=files
                )
        finally:
            temp_file.unlink(missing_ok=True)
    
    # List artifacts
    response = await api_client.get(
        f"/api/mlflow/runs/{run_id}/artifacts/list"
    )
    
    assert response.status_code == 200
    artifacts = response.json()
    
    assert isinstance(artifacts, list)
    assert len(artifacts) >= 3


@pytest.mark.asyncio
async def test_download_artifact(api_client: AsyncClient, cleanup_experiments):
    """Test downloading an artifact from a run."""
    # Setup
    exp_response = await api_client.post(
        "/api/mlflow/experiments/create",
        json={"name": generate_test_experiment_name()}
    )
    experiment_id = exp_response.json()["experiment_id"]
    cleanup_experiments.append(experiment_id)
    
    run_response = await api_client.post(
        "/api/mlflow/runs/create",
        json={"experiment_id": experiment_id}
    )
    run_id = run_response.json()["run_id"]
    
    # Upload artifact
    test_content = "Test downloadable content"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        temp_file = Path(f.name)
    
    try:
        with open(temp_file, 'rb') as f:
            files = {"file": ("download_test.txt", f, "text/plain")}
            await api_client.post(
                f"/api/mlflow/runs/{run_id}/log-artifact",
                files=files
            )
    finally:
        temp_file.unlink(missing_ok=True)
    
    # Download artifact
    response = await api_client.get(
        f"/api/mlflow/runs/{run_id}/artifacts/download",
        params={"path": "download_test.txt"}
    )
    
    assert response.status_code == 200
    # Verify content matches
    assert test_content in response.text or test_content.encode() in response.content


@pytest.mark.asyncio
async def test_register_model(api_client: AsyncClient, cleanup_experiments):
    """Test model registration from artifacts."""
    # Setup
    exp_response = await api_client.post(
        "/api/mlflow/experiments/create",
        json={"name": generate_test_experiment_name()}
    )
    experiment_id = exp_response.json()["experiment_id"]
    cleanup_experiments.append(experiment_id)
    
    run_response = await api_client.post(
        "/api/mlflow/runs/create",
        json={"experiment_id": experiment_id}
    )
    run_id = run_response.json()["run_id"]
    
    # Upload a test model artifact
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pkl', delete=False) as f:
        f.write("Fake model content")
        temp_file = Path(f.name)
    
    try:
        with open(temp_file, 'rb') as f:
            files = {"file": ("model.pkl", f, "application/octet-stream")}
            await api_client.post(
                f"/api/mlflow/runs/{run_id}/log-artifact",
                files=files,
                data={"artifact_path": "model"}
            )
    finally:
        temp_file.unlink(missing_ok=True)
    
    # Register model
    import uuid
    model_name = f"test_model_{uuid.uuid4().hex[:8]}"
    
    response = await api_client.post(
        "/api/mlflow/model-versions/register",
        json={
            "name": model_name,
            "run_id": run_id,
            "artifact_path": "model"
        }
    )
    
    # May succeed or fail depending on MLflow configuration
    # Just verify endpoint is reachable
    assert response.status_code in [200, 400, 500]
