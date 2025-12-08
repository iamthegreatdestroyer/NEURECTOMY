"""
Integration tests for Optuna study management endpoints.

Tests cover:
- Study creation
- Study listing
- Study retrieval
- Trial creation
- Trial listing
- Best trial retrieval
"""
import pytest
from httpx import AsyncClient
from tests.conftest import generate_test_study_name


@pytest.mark.asyncio
async def test_create_study(api_client: AsyncClient, cleanup_studies):
    """Test study creation."""
    study_name = generate_test_study_name()
    
    response = await api_client.post(
        "/api/optuna/studies/create",
        json={
            "study_name": study_name,
            "direction": "minimize",
            "sampler": "tpe",
            "pruner": "median"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["study_name"] == study_name
    cleanup_studies.append(study_name)
    
    # Verify study was created
    get_response = await api_client.get(f"/api/optuna/studies/{study_name}")
    assert get_response.status_code == 200


@pytest.mark.asyncio
async def test_list_studies(api_client: AsyncClient):
    """Test study listing."""
    response = await api_client.get("/api/optuna/studies/list")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "studies" in data
    assert isinstance(data["studies"], list)


@pytest.mark.asyncio
async def test_get_study(api_client: AsyncClient, cleanup_studies):
    """Test retrieving study by name."""
    study_name = generate_test_study_name()
    
    # Create study
    create_response = await api_client.post(
        "/api/optuna/studies/create",
        json={
            "study_name": study_name,
            "direction": "maximize"
        }
    )
    assert create_response.status_code == 200
    cleanup_studies.append(study_name)
    
    # Get study
    response = await api_client.get(f"/api/optuna/studies/{study_name}")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["study_name"] == study_name
    assert data["direction"] == "maximize"


@pytest.mark.asyncio
async def test_create_trial(api_client: AsyncClient, cleanup_studies):
    """Test manual trial creation."""
    study_name = generate_test_study_name()
    
    # Create study
    await api_client.post(
        "/api/optuna/studies/create",
        json={"study_name": study_name, "direction": "minimize"}
    )
    cleanup_studies.append(study_name)
    
    # Create trial
    response = await api_client.post(
        f"/api/optuna/studies/{study_name}/trials/create",
        json={
            "params": {
                "learning_rate": 0.001,
                "batch_size": 32
            },
            "value": 0.15
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "trial_id" in data
    assert data["params"]["learning_rate"] == 0.001
    assert data["value"] == 0.15


@pytest.mark.asyncio
async def test_list_trials(api_client: AsyncClient, cleanup_studies):
    """Test listing trials in a study."""
    study_name = generate_test_study_name()
    
    # Create study
    await api_client.post(
        "/api/optuna/studies/create",
        json={"study_name": study_name, "direction": "minimize"}
    )
    cleanup_studies.append(study_name)
    
    # Create multiple trials
    for i in range(3):
        await api_client.post(
            f"/api/optuna/studies/{study_name}/trials/create",
            json={
                "params": {"x": float(i)},
                "value": float(i ** 2)
            }
        )
    
    # List trials
    response = await api_client.get(
        f"/api/optuna/studies/{study_name}/trials/list"
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "trials" in data
    assert len(data["trials"]) >= 3


@pytest.mark.asyncio
async def test_get_best_trial(api_client: AsyncClient, cleanup_studies):
    """Test retrieving best trial from a study."""
    study_name = generate_test_study_name()
    
    # Create study (minimize)
    await api_client.post(
        "/api/optuna/studies/create",
        json={"study_name": study_name, "direction": "minimize"}
    )
    cleanup_studies.append(study_name)
    
    # Create trials with different values
    await api_client.post(
        f"/api/optuna/studies/{study_name}/trials/create",
        json={"params": {"x": 1.0}, "value": 5.0}
    )
    await api_client.post(
        f"/api/optuna/studies/{study_name}/trials/create",
        json={"params": {"x": 2.0}, "value": 2.0}  # Best
    )
    await api_client.post(
        f"/api/optuna/studies/{study_name}/trials/create",
        json={"params": {"x": 3.0}, "value": 8.0}
    )
    
    # Get best trial
    response = await api_client.get(
        f"/api/optuna/studies/{study_name}/best-trial"
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["value"] == 2.0
    assert data["params"]["x"] == 2.0


@pytest.mark.asyncio
async def test_get_best_params(api_client: AsyncClient, cleanup_studies):
    """Test retrieving best parameters from a study."""
    study_name = generate_test_study_name()
    
    # Create study (maximize)
    await api_client.post(
        "/api/optuna/studies/create",
        json={"study_name": study_name, "direction": "maximize"}
    )
    cleanup_studies.append(study_name)
    
    # Create trials
    await api_client.post(
        f"/api/optuna/studies/{study_name}/trials/create",
        json={
            "params": {"lr": 0.001, "dropout": 0.2},
            "value": 0.85
        }
    )
    await api_client.post(
        f"/api/optuna/studies/{study_name}/trials/create",
        json={
            "params": {"lr": 0.01, "dropout": 0.3},
            "value": 0.92  # Best
        }
    )
    
    # Get best params
    response = await api_client.get(
        f"/api/optuna/studies/{study_name}/best-params"
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["lr"] == 0.01
    assert data["dropout"] == 0.3
