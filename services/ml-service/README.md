# NEURECTOMY ML Service

Intelligence Foundry - Machine Learning service for the NEURECTOMY platform.

## Features

- **LLM Integration**: vLLM + Ollama hybrid inference
- **Embedding Pipeline**: Real-time embedding generation with caching
- **RAG System**: Retrieval-Augmented Generation with hybrid search
- **Agent Intelligence**: Behavior modeling, memory systems, learning pipelines
- **Analytics Engine**: Predictive analytics, time-series analysis, anomaly detection
- **MLOps**: MLflow experiment tracking, model versioning

## Setup

### Prerequisites

- Python 3.11+
- Docker (for dependencies)
- CUDA-capable GPU (optional, for accelerated inference)

### Installation

```bash
# Start dependencies
docker-compose up -d postgres redis mlflow

# Install package
pip install -e ".[dev]"

# Run service
uvicorn main:app --reload
```

## API Endpoints

- `POST /api/v1/embeddings/generate` - Generate embeddings
- `POST /api/v1/llm/generate` - LLM text generation
- `POST /api/v1/rag/query` - RAG query
- `POST /api/v1/training/start` - Start training job
- `GET /api/v1/training/{job_id}/status` - Get training status
- `POST /api/v1/analytics/forecast` - Generate forecast
- `POST /api/v1/analytics/anomaly-detection` - Detect anomalies

## Configuration

Set environment variables or use `.env` file:

```env
OPENAI_API_KEY=your_key
OLLAMA_HOST=http://localhost:11434
MLFLOW_TRACKING_URI=http://localhost:5001
DATABASE_URL=postgresql://neurectomy:neurectomy@localhost:5434/neurectomy
REDIS_URL=redis://localhost:6379
```

## Development

```bash
# Run tests
pytest

# Type checking
mypy src

# Linting
ruff check src
```

## Architecture

```
ml-service/
├── src/
│   ├── config.py           # Configuration management
│   ├── services/           # Core services
│   │   ├── llm_service.py
│   │   ├── embedding_service.py
│   │   ├── training_orchestrator.py
│   │   ├── agent_intelligence.py
│   │   └── analytics_engine.py
│   ├── pipelines/          # ML pipelines
│   │   └── rag_pipeline.py
│   └── testing/            # ML testing framework
│       └── ml_testing_framework.py
└── main.py                 # FastAPI application
```

## License

Proprietary - NEURECTOMY Project
