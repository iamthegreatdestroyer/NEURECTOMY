"""
Pytest Configuration and Shared Fixtures

Copyright (c) 2025 NEURECTOMY. All Rights Reserved.

This module provides:
- Async fixtures for database connections
- Mock fixtures for external AI services
- Test data factories
- Common test utilities
"""

import asyncio
from datetime import datetime
from typing import AsyncGenerator, Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio
from pydantic import BaseModel
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport


# ==============================================================================
# Test Configuration
# ==============================================================================

class TestSettings:
    """Test-specific settings that override production config."""
    
    DATABASE_URL = "postgresql+asyncpg://test:test@localhost:5434/test_neurectomy"
    REDIS_URL = "redis://localhost:6379/1"  # Use different DB for tests
    MLFLOW_TRACKING_URI = "http://localhost:5000"
    
    # Mock API keys
    OPENAI_API_KEY = "test-openai-key"
    ANTHROPIC_API_KEY = "test-anthropic-key"
    
    # Test embedding config
    EMBEDDING_DIM = 384
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ==============================================================================
# Mock Responses
# ==============================================================================

MOCK_EMBEDDING = [0.1] * 384  # 384-dim embedding vector

MOCK_LLM_RESPONSE = {
    "content": "This is a mock LLM response for testing purposes.",
    "model": "test-model",
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
    },
}

MOCK_OLLAMA_RESPONSE = {
    "model": "llama3.2",
    "created_at": "2024-01-01T00:00:00Z",
    "response": "This is a mock Ollama response.",
    "done": True,
    "context": [],
    "total_duration": 1000000000,
    "load_duration": 100000000,
    "prompt_eval_count": 10,
    "prompt_eval_duration": 200000000,
    "eval_count": 20,
    "eval_duration": 400000000,
}


# ==============================================================================
# Session-Scoped Fixtures
# ==============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings() -> TestSettings:
    """Provide test settings."""
    return TestSettings()


# ==============================================================================
# FastAPI Test Client Fixtures
# ==============================================================================

@pytest.fixture
def mock_app():
    """Create a mock FastAPI app for testing."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    
    app = FastAPI(
        title="NEURECTOMY ML Service - Test",
        version="0.2.0-test",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mock state objects
    app.state.llm_service = MagicMock()
    app.state.embedding_service = MagicMock()
    app.state.training_orchestrator = MagicMock()
    app.state.mlflow_tracker = MagicMock()
    app.state.agent_intelligence = MagicMock()
    
    # Health endpoints
    @app.get("/health")
    async def health():
        return {"status": "healthy", "service": "ml-service"}
    
    @app.get("/ready")
    async def ready():
        return {"status": "ready", "dependencies": {"postgres": True, "redis": True, "mlflow": True}}
    
    @app.get("/metrics")
    async def metrics():
        return {"requests_total": 100, "latency_avg_ms": 50}
    
    # LLM endpoints
    @app.post("/api/v1/llm/chat")
    async def chat(request: dict):
        return {
            "content": "Mock response",
            "model": request.get("model", "test"),
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        }
    
    @app.get("/api/v1/llm/models")
    async def list_models():
        return {"models": [{"id": "llama3.2", "name": "Llama 3.2"}]}
    
    # Embeddings endpoints
    @app.post("/api/v1/embeddings/generate")
    async def generate_embeddings(request: dict):
        texts = request.get("texts", [])
        if not texts:
            from fastapi import HTTPException
            raise HTTPException(status_code=422, detail="texts cannot be empty")
        return {
            "embeddings": [[0.1] * 384 for _ in texts],
            "model": "test-embed",
            "dimensions": 384
        }
    
    @app.post("/api/v1/embeddings/search")
    async def semantic_search(request: dict):
        return {"results": [], "total": 0}
    
    # Training endpoints
    @app.post("/api/v1/training/jobs")
    async def create_training_job(request: dict):
        return {"job_id": "test-job-123", "status": "pending"}
    
    @app.get("/api/v1/training/jobs/{job_id}")
    async def get_training_job(job_id: str):
        return {"job_id": job_id, "status": "running"}
    
    @app.delete("/api/v1/training/jobs/{job_id}")
    async def cancel_training_job(job_id: str):
        return {"job_id": job_id, "status": "cancelled"}
    
    @app.get("/api/v1/training/experiments")
    async def list_experiments():
        return {"experiments": []}
    
    # Agent endpoints
    @app.post("/api/v1/agents")
    async def create_agent(request: dict):
        return {"agent_id": "test-agent-123", "name": request.get("name", "TestAgent")}
    
    @app.get("/api/v1/agents/{agent_id}")
    async def get_agent(agent_id: str):
        return {"agent_id": agent_id, "name": "TestAgent"}
    
    @app.put("/api/v1/agents/{agent_id}")
    async def update_agent(agent_id: str, request: dict):
        return {"agent_id": agent_id, **request}
    
    @app.delete("/api/v1/agents/{agent_id}")
    async def delete_agent(agent_id: str):
        return {"deleted": True}
    
    @app.post("/api/v1/agents/{agent_id}/chat")
    async def agent_chat(agent_id: str, request: dict):
        return {"response": "Mock agent response", "agent_id": agent_id}
    
    return app


@pytest.fixture
def test_client(mock_app):
    """Provide synchronous test client."""
    return TestClient(mock_app)


@pytest_asyncio.fixture
async def async_test_client(mock_app):
    """Provide async test client."""
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# ==============================================================================
# Database Fixtures
# ==============================================================================

@pytest_asyncio.fixture
async def mock_postgres_pool():
    """Mock PostgreSQL connection pool."""
    pool = AsyncMock()
    pool.acquire = AsyncMock()
    pool.release = AsyncMock()
    pool.close = AsyncMock()
    
    # Mock connection
    connection = AsyncMock()
    connection.execute = AsyncMock(return_value="OK")
    connection.fetch = AsyncMock(return_value=[])
    connection.fetchone = AsyncMock(return_value=None)
    connection.fetchval = AsyncMock(return_value=None)
    
    # Context manager for acquire
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=connection)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    
    yield pool


@pytest_asyncio.fixture
async def mock_redis():
    """Mock Redis client."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.setex = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=1)
    redis.exists = AsyncMock(return_value=0)
    redis.expire = AsyncMock(return_value=True)
    redis.ttl = AsyncMock(return_value=-2)
    redis.keys = AsyncMock(return_value=[])
    redis.mget = AsyncMock(return_value=[])
    redis.pipeline = MagicMock()
    redis.close = AsyncMock()
    
    yield redis


@pytest_asyncio.fixture
async def mock_vector_store():
    """Mock vector store for embeddings."""
    store = AsyncMock()
    store.add_vectors = AsyncMock(return_value=True)
    store.search = AsyncMock(return_value=[])
    store.delete = AsyncMock(return_value=True)
    store.get_stats = AsyncMock(return_value={"total_vectors": 0})
    
    yield store


# ==============================================================================
# AI Service Fixtures
# ==============================================================================

@pytest_asyncio.fixture
async def mock_ollama_client():
    """Mock Ollama client for local LLM testing."""
    client = AsyncMock()
    
    # Mock generate
    client.generate = AsyncMock(return_value=MOCK_OLLAMA_RESPONSE)
    
    # Mock embeddings
    client.embeddings = AsyncMock(return_value={"embedding": MOCK_EMBEDDING})
    
    # Mock chat
    client.chat = AsyncMock(return_value={
        "model": "llama3.2",
        "message": {"role": "assistant", "content": "Mock chat response"},
        "done": True,
    })
    
    # Mock list models
    client.list = AsyncMock(return_value={
        "models": [
            {"name": "llama3.2", "size": 1000000000},
            {"name": "nomic-embed-text", "size": 500000000},
        ]
    })
    
    yield client


@pytest_asyncio.fixture
async def mock_openai_client():
    """Mock OpenAI client."""
    client = AsyncMock()
    
    # Mock chat completion
    completion = MagicMock()
    completion.choices = [
        MagicMock(
            message=MagicMock(content="Mock OpenAI response"),
            finish_reason="stop",
        )
    ]
    completion.usage = MagicMock(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
    )
    completion.model = "gpt-4-turbo-preview"
    
    client.chat.completions.create = AsyncMock(return_value=completion)
    
    # Mock embeddings
    embedding_response = MagicMock()
    embedding_response.data = [MagicMock(embedding=MOCK_EMBEDDING)]
    embedding_response.usage = MagicMock(total_tokens=5)
    
    client.embeddings.create = AsyncMock(return_value=embedding_response)
    
    yield client


@pytest_asyncio.fixture
async def mock_anthropic_client():
    """Mock Anthropic client."""
    client = AsyncMock()
    
    # Mock message creation
    message = MagicMock()
    message.content = [MagicMock(text="Mock Claude response")]
    message.model = "claude-3-opus-20240229"
    message.usage = MagicMock(input_tokens=10, output_tokens=20)
    message.stop_reason = "end_turn"
    
    client.messages.create = AsyncMock(return_value=message)
    
    yield client


@pytest_asyncio.fixture
async def mock_sentence_transformer():
    """Mock sentence transformer for embeddings."""
    model = MagicMock()
    model.encode = MagicMock(return_value=[MOCK_EMBEDDING])
    model.get_sentence_embedding_dimension = MagicMock(return_value=384)
    
    yield model


# ==============================================================================
# MLflow Fixtures
# ==============================================================================

@pytest.fixture
def mock_mlflow():
    """Mock MLflow tracking."""
    with patch("mlflow.start_run") as mock_start, \
         patch("mlflow.log_param") as mock_param, \
         patch("mlflow.log_metric") as mock_metric, \
         patch("mlflow.log_artifact") as mock_artifact, \
         patch("mlflow.end_run") as mock_end, \
         patch("mlflow.set_tracking_uri") as mock_uri, \
         patch("mlflow.create_experiment") as mock_create_exp, \
         patch("mlflow.set_experiment") as mock_set_exp:
        
        mock_start.return_value.__enter__ = MagicMock(
            return_value=MagicMock(info=MagicMock(run_id="test-run-id"))
        )
        mock_start.return_value.__exit__ = MagicMock(return_value=None)
        mock_create_exp.return_value = "test-experiment-id"
        
        yield {
            "start_run": mock_start,
            "log_param": mock_param,
            "log_metric": mock_metric,
            "log_artifact": mock_artifact,
            "end_run": mock_end,
            "set_tracking_uri": mock_uri,
            "create_experiment": mock_create_exp,
            "set_experiment": mock_set_exp,
        }


# ==============================================================================
# Test Data Factories
# ==============================================================================

@pytest.fixture
def agent_data_factory():
    """Factory for creating test agent data."""
    def _create(
        agent_id: str = None,
        name: str = "TestAgent",
        agent_type: str = "research",
        description: str = "A test agent",
        capabilities: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        return {
            "id": agent_id or str(uuid4()),
            "name": name,
            "type": agent_type,
            "description": description,
            "capabilities": capabilities or ["search", "analyze", "summarize"],
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "status": "active",
            "config": kwargs.get("config", {}),
            "memory": kwargs.get("memory", {"short_term": [], "long_term": []}),
            "metrics": kwargs.get("metrics", {"tasks_completed": 0, "success_rate": 0.0}),
        }
    
    return _create


@pytest.fixture
def document_data_factory():
    """Factory for creating test document data."""
    def _create(
        doc_id: str = None,
        content: str = "Test document content for RAG pipeline testing.",
        metadata: Dict[str, Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        return {
            "id": doc_id or str(uuid4()),
            "content": content,
            "metadata": metadata or {"source": "test", "type": "text"},
            "embedding": kwargs.get("embedding", MOCK_EMBEDDING),
            "created_at": datetime.utcnow().isoformat(),
            "chunk_index": kwargs.get("chunk_index", 0),
            "total_chunks": kwargs.get("total_chunks", 1),
        }
    
    return _create


@pytest.fixture
def training_job_factory():
    """Factory for creating test training job data."""
    def _create(
        job_id: str = None,
        model_type: str = "fine-tune",
        status: str = "pending",
        **kwargs
    ) -> Dict[str, Any]:
        return {
            "id": job_id or str(uuid4()),
            "model_type": model_type,
            "status": status,
            "config": kwargs.get("config", {
                "epochs": 3,
                "batch_size": 8,
                "learning_rate": 2e-5,
            }),
            "metrics": kwargs.get("metrics", {}),
            "created_at": datetime.utcnow().isoformat(),
            "started_at": kwargs.get("started_at"),
            "completed_at": kwargs.get("completed_at"),
            "error": kwargs.get("error"),
        }
    
    return _create


@pytest.fixture
def conversation_factory():
    """Factory for creating test conversation data."""
    def _create(
        conversation_id: str = None,
        agent_id: str = None,
        messages: List[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        return {
            "id": conversation_id or str(uuid4()),
            "agent_id": agent_id or str(uuid4()),
            "messages": messages or [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi! How can I help?"},
            ],
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "metadata": kwargs.get("metadata", {}),
        }
    
    return _create


# ==============================================================================
# HTTP Client Fixtures
# ==============================================================================

@pytest_asyncio.fixture
async def mock_httpx_client():
    """Mock httpx async client for API testing."""
    client = AsyncMock()
    
    # Default successful response
    response = AsyncMock()
    response.status_code = 200
    response.json = MagicMock(return_value={"status": "ok"})
    response.text = "OK"
    response.raise_for_status = MagicMock()
    
    client.get = AsyncMock(return_value=response)
    client.post = AsyncMock(return_value=response)
    client.put = AsyncMock(return_value=response)
    client.delete = AsyncMock(return_value=response)
    client.patch = AsyncMock(return_value=response)
    
    yield client


# ==============================================================================
# Utility Fixtures
# ==============================================================================

@pytest.fixture
def sample_text_chunks():
    """Provide sample text chunks for RAG testing."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are inspired by biological neurons.",
        "Deep learning uses multiple layers of neural networks.",
        "Natural language processing enables computers to understand text.",
    ]


@pytest.fixture
def sample_embeddings():
    """Provide sample embeddings for vector search testing."""
    import random
    random.seed(42)
    
    return [
        [random.random() for _ in range(384)]
        for _ in range(5)
    ]


@pytest.fixture
def sample_conversation_history():
    """Provide sample conversation for context testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a type of AI..."},
        {"role": "user", "content": "Can you give an example?"},
    ]


# ==============================================================================
# Cleanup Fixtures
# ==============================================================================

@pytest_asyncio.fixture(autouse=True)
async def cleanup_after_test():
    """Cleanup fixture that runs after each test."""
    yield
    # Any cleanup logic here
    await asyncio.sleep(0)  # Allow pending tasks to complete


# ==============================================================================
# Integration Test Fixtures (Task 5 - Intelligence Foundry)
# ==============================================================================

@pytest_asyncio.fixture
async def api_client() -> AsyncGenerator[AsyncClient, None]:
    """
    HTTP client for Intelligence Foundry API testing.
    
    Points to http://localhost:8002 (Intelligence Foundry FastAPI service).
    """
    async with AsyncClient(base_url="http://localhost:8002", timeout=30.0) as client:
        yield client


@pytest.fixture
def mlflow_base_url() -> str:
    """MLflow Tracking Server base URL."""
    return "http://localhost:5000"


@pytest.fixture
def minio_base_url() -> str:
    """MinIO S3 storage base URL."""
    return "http://localhost:9001"


@pytest.fixture
def optuna_base_url() -> str:
    """Optuna Dashboard base URL."""
    return "http://localhost:8085"


@pytest_asyncio.fixture
async def cleanup_experiments(api_client: AsyncClient):
    """
    Cleanup MLflow experiments created during tests.
    
    Yields test_experiment_ids list that tests can append to.
    After test completes, deletes all experiments in the list.
    """
    test_experiment_ids: List[str] = []
    
    yield test_experiment_ids
    
    # Cleanup: Delete all test experiments
    for experiment_id in test_experiment_ids:
        try:
            response = await api_client.post(
                f"/api/mlflow/experiments/{experiment_id}/delete"
            )
            if response.status_code not in [200, 404]:
                print(f"Warning: Failed to cleanup experiment {experiment_id}: {response.status_code}")
        except Exception as e:
            print(f"Warning: Cleanup exception for experiment {experiment_id}: {e}")


@pytest_asyncio.fixture
async def cleanup_studies(api_client: AsyncClient):
    """
    Cleanup Optuna studies created during tests.
    
    Yields test_study_names list that tests can append to.
    After test completes, deletes all studies in the list.
    """
    test_study_names: List[str] = []
    
    yield test_study_names
    
    # Cleanup: Delete all test studies
    for study_name in test_study_names:
        try:
            response = await api_client.delete(
                f"/api/optuna/studies/{study_name}"
            )
            if response.status_code not in [200, 404]:
                print(f"Warning: Failed to cleanup study {study_name}: {response.status_code}")
        except Exception as e:
            print(f"Warning: Cleanup exception for study {study_name}: {e}")


def generate_test_experiment_name() -> str:
    """Generate unique experiment name for testing."""
    return f"test_exp_{uuid4().hex[:8]}"


def generate_test_study_name() -> str:
    """Generate unique study name for testing."""
    return f"test_study_{uuid4().hex[:8]}"


def generate_test_run_name() -> str:
    """Generate unique run name for testing."""
    return f"test_run_{uuid4().hex[:8]}"


# ==============================================================================
# Markers Configuration
# ==============================================================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests (require services)")
    config.addinivalue_line("markers", "slow: Slow tests (>1s execution)")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU/DirectML")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "mlflow: Tests for MLflow integration")
    config.addinivalue_line("markers", "optuna: Tests for Optuna integration")
    config.addinivalue_line("markers", "websocket: Tests for WebSocket functionality")
