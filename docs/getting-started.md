# Getting Started

## Installation

```bash
# Clone repository
git clone https://github.com/neurectomy/neurectomy.git
cd neurectomy

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-api.txt

# Run tests
pytest tests/ -v
```

## Quick Start

### Using the Orchestrator

```python
from neurectomy import NeurectomyOrchestrator

# Create orchestrator
orchestrator = NeurectomyOrchestrator()

# Generate text
result = orchestrator.generate(
    prompt="Explain machine learning",
    max_tokens=100,
    temperature=0.7,
)

print(result.generated_text)
print(f"Tokens: {result.tokens_generated}")
print(f"Latency: {result.execution_time_ms}ms")
```

### Using the SDK

```python
from neurectomy.sdk import NeurectomyClient

client = NeurectomyClient()

# Simple generation
response = client.generate("Hello!")
print(response["text"])

# Streaming
for chunk in client.stream("Tell me a story"):
    print(chunk, end="", flush=True)
```

### Using Elite Agents

```python
from neurectomy.elite import EliteCollective
from neurectomy.core.types import TaskRequest

collective = EliteCollective()

# List agents
print(f"Available: {len(collective.list_agents())} agents")

# Execute task
request = TaskRequest(
    task_id="example_001",
    task_type="summarize",
    payload={"text": "Long document..."},
)

result = collective.execute(request)
print(result.output)
```

## Configuration

```python
from neurectomy import NeurectomyOrchestrator, OrchestratorConfig

config = OrchestratorConfig(
    max_concurrent_tasks=10,
    enable_compression=True,
    enable_rsu=True,
)

orchestrator = NeurectomyOrchestrator(config)
```

## Running the API Server

```bash
# Start API server
python -m neurectomy.api.app

# Test health endpoint
curl http://localhost:8000/health

# Generate text via API
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello!", "max_tokens": 50}'
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# With coverage
pytest tests/ --cov=neurectomy --cov-report=html
```

## Next Steps

- [Architecture Overview](architecture.md)
- [API Documentation](api/rest-api.md)
- [Tutorials](tutorials/basic-generation.md)
- [Examples](../examples/)
