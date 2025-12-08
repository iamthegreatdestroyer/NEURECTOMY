"""
Integration tests for MLflow experiment management endpoints.

Tests cover:
- Experiment creation
- Experiment listing
- Experiment retrieval
- Experiment search
- Experiment deletion
"""
import pytest
from httpx import AsyncClient
from tests.conftest import generate_test_experiment_name


@pytest.mark.asyncio
async def test_create_experiment(api_client: AsyncClient, cleanup_experiments):
    """Test experiment creation."""
    experiment_name = generate_test_experiment_name()
    
    response = await api_client.post(
        "/api/mlflow/experiments/create",
        json={
            "name": experiment_name,
            "artifact_location": f"s3://mlflow-artifacts/{experiment_name}",
            "tags": {
                "test": "true",
                "purpose": "integration_test"
            }
        }
    )
    
    assert response.status_code == 201  # 201 Created for resource creation
    data = response.json()
    
    assert "experiment_id" in data
    experiment_id = data["experiment_id"]
    
    # Register for cleanup
    cleanup_experiments.append(experiment_id)
    
    # Verify experiment was created
    get_response = await api_client.get(f"/api/mlflow/experiments/{experiment_id}")
    assert get_response.status_code == 200
    
    experiment_data = get_response.json()
    assert experiment_data["name"] == experiment_name
    assert experiment_data["artifact_location"] == f"s3://mlflow-artifacts/{experiment_name}"


@pytest.mark.asyncio
async def test_list_experiments(api_client: AsyncClient):
    """Test experiment listing."""
    response = await api_client.get("/api/mlflow/experiments/list")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "experiments" in data
    assert isinstance(data["experiments"], list)
    
    # Should have at least the default experiment (id=0)
    assert len(data["experiments"]) >= 1


@pytest.mark.asyncio
async def test_get_experiment_by_id(api_client: AsyncClient, cleanup_experiments):
    """Test retrieving experiment by ID."""
    # Create experiment
    experiment_name = generate_test_experiment_name()
    create_response = await api_client.post(
        "/api/mlflow/experiments/create",
        json={"name": experiment_name}
    )
    experiment_id = create_response.json()["experiment_id"]
    cleanup_experiments.append(experiment_id)
    
    # Get experiment
    response = await api_client.get(f"/api/mlflow/experiments/{experiment_id}")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["experiment_id"] == experiment_id
    assert data["name"] == experiment_name
    assert "lifecycle_stage" in data
    assert "creation_time" in data


@pytest.mark.asyncio
async def test_get_experiment_by_name(api_client: AsyncClient, cleanup_experiments):
    """Test retrieving experiment by name."""
    # Create experiment
    experiment_name = generate_test_experiment_name()
    create_response = await api_client.post(
        "/api/mlflow/experiments/create",
        json={"name": experiment_name}
    )
    experiment_id = create_response.json()["experiment_id"]
    cleanup_experiments.append(experiment_id)
    
    # Get experiment by name
    response = await api_client.get(
        f"/api/mlflow/experiments/by-name/{experiment_name}"
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["name"] == experiment_name
    assert data["experiment_id"] == experiment_id


@pytest.mark.asyncio
async def test_search_experiments(api_client: AsyncClient, cleanup_experiments):
    """Test experiment search with filters."""
    # Create test experiments
    experiment_name_1 = generate_test_experiment_name()
    experiment_name_2 = generate_test_experiment_name()
    
    response_1 = await api_client.post(
        "/api/mlflow/experiments/create",
        json={
            "name": experiment_name_1,
            "tags": {"test": "true", "type": "classification"}
        }
    )
    response_2 = await api_client.post(
        "/api/mlflow/experiments/create",
        json={
            "name": experiment_name_2,
            "tags": {"test": "true", "type": "regression"}
        }
    )
    
    exp_id_1 = response_1.json()["experiment_id"]
    exp_id_2 = response_2.json()["experiment_id"]
    cleanup_experiments.append(exp_id_1)
    cleanup_experiments.append(exp_id_2)
    
    # Search for experiments with specific tag
    response = await api_client.post(
        "/api/mlflow/experiments/search",
        json={
            "filter_string": "tags.test = 'true'",
            "max_results": 100
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "experiments" in data
    experiment_names = [exp["name"] for exp in data["experiments"]]
    assert experiment_name_1 in experiment_names
    assert experiment_name_2 in experiment_names


@pytest.mark.asyncio
async def test_create_duplicate_experiment_fails(api_client: AsyncClient, cleanup_experiments):
    """Test that creating duplicate experiment fails."""
    experiment_name = generate_test_experiment_name()
    
    # Create first experiment
    response_1 = await api_client.post(
        "/api/mlflow/experiments/create",
        json={"name": experiment_name}
    )
    assert response_1.status_code == 201  # 201 Created for resource creation
    cleanup_experiments.append(response_1.json()["experiment_id"])
    
    # Try to create duplicate
    response_2 = await api_client.post(
        "/api/mlflow/experiments/create",
        json={"name": experiment_name}
    )
    
    # Should fail with 4xx error
    assert response_2.status_code >= 400
