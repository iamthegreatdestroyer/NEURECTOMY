# Python SDK Reference

## Installation

```bash
pip install neurectomy
```

## Client

### NeurectomyClient

```python
from neurectomy.sdk import NeurectomyClient, NeurectomyConfig

# Default configuration
client = NeurectomyClient()

# Custom configuration
config = NeurectomyConfig(
    base_url="http://localhost:8000",
    api_key="your-api-key",
    timeout=60.0,
)
client = NeurectomyClient(config)
```

## Methods

### generate()

Generate text from a prompt.

```python
response = client.generate(
    prompt="Your prompt",
    max_tokens=256,
    temperature=0.7,
)
print(response["text"])
print(response["tokens"])
```

### stream()

Stream tokens as they're generated.

```python
for chunk in client.stream(
    prompt="Tell me a story",
    max_tokens=500,
):
    print(chunk, end="", flush=True)
```

### execute_task()

Execute an agent task.

```python
result = client.execute_task(
    task_type="summarize",
    payload={"text": "Long document..."},
    team="analysis",
)
print(result["output"])
```

### list_agents()

List available agents.

```python
agents = client.list_agents()
print(f"Available agents: {len(agents)}")
```

### list_teams()

List agent teams.

```python
teams = client.list_teams()
print(f"Teams: {teams}")
```

### health()

Check system health.

```python
health = client.health()
print(health["status"])
```

## Async Client

```python
from neurectomy.sdk import AsyncNeurectomyClient

async with AsyncNeurectomyClient() as client:
    response = await client.generate("Hello!")
    print(response["text"])
```

### Async Methods

```python
# Generate
response = await client.generate("Prompt", max_tokens=100)

# Stream
async for chunk in client.stream("Prompt", max_tokens=100):
    print(chunk, end="")

# Execute task
result = await client.execute_task(
    task_type="summarize",
    payload={"text": "..."},
    team="analysis",
)

# Health
health = await client.health()
```

## Configuration Options

```python
config = NeurectomyConfig(
    base_url="http://localhost:8000",  # API endpoint
    api_key="your-key",                 # Authentication
    timeout=60.0,                       # Request timeout
    max_retries=3,                      # Retry attempts
    retry_delay=1.0,                    # Retry delay (seconds)
)
```

## Error Handling

```python
from neurectomy.sdk.errors import (
    NeurectomyError,
    APIError,
    TimeoutError,
    RateLimitError,
)

try:
    response = client.generate("prompt")
except RateLimitError:
    print("Rate limited, retry later")
except APIError as e:
    print(f"API error: {e}")
except NeurectomyError as e:
    print(f"Error: {e}")
```

## Examples

### Simple Generation

```python
from neurectomy.sdk import NeurectomyClient

client = NeurectomyClient()
response = client.generate("What is machine learning?")
print(response["text"])
```

### Streaming

```python
for chunk in client.stream("Write a story about robots"):
    print(chunk, end="", flush=True)
```

### Agent Task

```python
result = client.execute_task(
    task_type="code",
    payload={
        "language": "python",
        "description": "Function to calculate factorial"
    },
    team="synthesis"
)
print(result["output"])
```

### Batch Processing

```python
prompts = [
    "Explain gravity",
    "What is photosynthesis?",
    "Describe quantum computing",
]

for prompt in prompts:
    response = client.generate(prompt, max_tokens=100)
    print(f"{prompt} -> {response['text'][:50]}...")
```

### Context Management

```python
context = """
You are a helpful Python tutor.
The user will ask Python-related questions.
Provide clear, concise answers.
"""

response = client.generate(
    prompt=f"{context}\nUser: How do I create a list in Python?",
    max_tokens=100,
)
```

### With Context Manager

```python
with NeurectomyClient() as client:
    response = client.generate("Hello!")
    print(response["text"])
# Automatically closed when exiting context
```
