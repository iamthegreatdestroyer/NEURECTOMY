"""
Integration tests for MLflow run management endpoints.

Tests cover:
- Run creation
- Logging metrics, parameters, tags
- Run updates
- Run search
- Run retrieval
"""
import pytest
from httpx import AsyncClient
from tests.conftest import (
    generate_test_experiment_name,
    generate_test_run_name
)


@pytest.mark.asyncio
async def test_create_run(api_client: AsyncClient, cleanup_experiments):
    """Test run creation within an experiment."""
    # Create experiment first
    experiment_name = generate_test_experiment_name()
    exp_response = await api_client.post(
        "/api/mlflow/experiments/create",
        json={"name": experiment_name}
    )
    experiment_id = exp_response.json()["experiment_id"]
    cleanup_experiments.append(experiment_id)
    
    # Create run
    run_name = generate_test_run_name()
    response = await api_client.post(
        "/api/mlflow/runs/create",
        json={
            "experiment_id": experiment_id,
            "run_name": run_name,
            "tags": {
                "test": "true",
                "model": "test_model"
            }
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "run_id" in data
    assert "run_uuid" in data
    
    # Verify run was created
    run_id = data["run_id"]
    get_response = await api_client.get(f"/api/mlflow/runs/{run_id}")
    assert get_response.status_code == 200
    
    run_data = get_response.json()
    assert run_data["info"]["run_name"] == run_name
    assert run_data["info"]["experiment_id"] == experiment_id


@pytest.mark.asyncio
async def test_log_metric(api_client: AsyncClient, cleanup_experiments):
    """Test logging metrics to a run."""
    # Setup: create experiment and run
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
    
    # Log metric
    response = await api_client.post(
        f"/api/mlflow/runs/{run_id}/log-metric",
        json={
            "key": "accuracy",
            "value": 0.95,
            "timestamp": 1700000000000,
            "step": 10
        }
    )
    
    assert response.status_code == 200
    
    # Verify metric was logged
    get_response = await api_client.get(f"/api/mlflow/runs/{run_id}")
    run_data = get_response.json()
    
    assert "metrics" in run_data["data"]
    assert "accuracy" in run_data["data"]["metrics"]
    assert run_data["data"]["metrics"]["accuracy"] == 0.95


@pytest.mark.asyncio
async def test_log_parameter(api_client: AsyncClient, cleanup_experiments):
    """Test logging parameters to a run."""
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
    
    # Log parameter
    response = await api_client.post(
        f"/api/mlflow/runs/{run_id}/log-parameter",
        json={
            "key": "learning_rate",
            "value": "0.001"
        }
    )
    
    assert response.status_code == 200
    
    # Verify parameter
    get_response = await api_client.get(f"/api/mlflow/runs/{run_id}")
    run_data = get_response.json()
    
    assert "params" in run_data["data"]
    assert "learning_rate" in run_data["data"]["params"]
    assert run_data["data"]["params"]["learning_rate"] == "0.001"


@pytest.mark.asyncio
async def test_log_batch(api_client: AsyncClient, cleanup_experiments):
    """Test batch logging of metrics, parameters, and tags."""
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
    
    # Log batch
    response = await api_client.post(
        f"/api/mlflow/runs/{run_id}/log-batch",
        json={
            "metrics": [
                {"key": "loss", "value": 0.5, "timestamp": 1700000000000, "step": 1},
                {"key": "loss", "value": 0.3, "timestamp": 1700000001000, "step": 2}
            ],
            "params": [
                {"key": "batch_size", "value": "32"},
                {"key": "epochs", "value": "10"}
            ],
            "tags": [
                {"key": "model_type", "value": "cnn"},
                {"key": "dataset", "value": "cifar10"}
            ]
        }
    )
    
    assert response.status_code == 200
    
    # Verify all data was logged
    get_response = await api_client.get(f"/api/mlflow/runs/{run_id}")
    run_data = get_response.json()
    
    assert "loss" in run_data["data"]["metrics"]
    assert "batch_size" in run_data["data"]["params"]
    assert "epochs" in run_data["data"]["params"]
    assert "model_type" in run_data["data"]["tags"]


@pytest.mark.asyncio
async def test_update_run(api_client: AsyncClient, cleanup_experiments):
    """Test updating run status."""
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
    
    # Update run to FINISHED
    response = await api_client.post(
        f"/api/mlflow/runs/{run_id}/update",
        json={
            "status": "FINISHED",
            "end_time": 1700000002000
        }
    )
    
    assert response.status_code == 200
    
    # Verify update
    get_response = await api_client.get(f"/api/mlflow/runs/{run_id}")
    run_data = get_response.json()
    
    assert run_data["info"]["status"] == "FINISHED"


@pytest.mark.asyncio
async def test_search_runs(api_client: AsyncClient, cleanup_experiments):
    """Test run search with filters."""
    # Create experiment with multiple runs
    exp_response = await api_client.post(
        "/api/mlflow/experiments/create",
        json={"name": generate_test_experiment_name()}
    )
    experiment_id = exp_response.json()["experiment_id"]
    cleanup_experiments.append(experiment_id)
    
    # Create runs with different metrics
    for i in range(3):
        run_response = await api_client.post(
            "/api/mlflow/runs/create",
            json={
                "experiment_id": experiment_id,
                "run_name": f"run_{i}"
            }
        )
        run_id = run_response.json()["run_id"]
        
        # Log metric
        await api_client.post(
            f"/api/mlflow/runs/{run_id}/log-metric",
            json={
                "key": "accuracy",
                "value": 0.8 + (i * 0.05),
                "timestamp": 1700000000000,
                "step": 0
            }
        )
    
    # Search for runs with accuracy > 0.85
    response = await api_client.post(
        "/api/mlflow/runs/search",
        json={
            "experiment_ids": [experiment_id],
            "filter_string": "metrics.accuracy > 0.85",
            "max_results": 10
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "runs" in data
    # Should find runs with accuracy 0.85 and 0.90
    assert len(data["runs"]) >= 2
