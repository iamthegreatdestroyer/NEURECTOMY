# Tutorial: Using Elite Agents

## Overview

Learn to leverage the 40 Elite Agents for specialized tasks.

## Available Teams

| Team        | Count | Focus                     | Example Tasks           |
| ----------- | ----- | ------------------------- | ----------------------- |
| Inference   | 8     | Generation & reasoning    | Text completion, Q&A    |
| Compression | 8     | Optimization & efficiency | Context compression     |
| Storage     | 8     | Data & retrieval          | RSU management, search  |
| Analysis    | 8     | Understanding & insight   | Summarization, NER      |
| Synthesis   | 8     | Creation & innovation     | Code generation, design |

## Basic Agent Task

Execute a task with the collective:

```python
from neurectomy.elite import EliteCollective
from neurectomy.core.types import TaskRequest

collective = EliteCollective()

# Summarization task (Analysis team)
request = TaskRequest(
    task_id="summary_001",
    task_type="summarize",
    payload={
        "text": "Long document to summarize...",
        "max_length": 100,
    },
)

result = collective.execute(request)
print(f"Summary: {result.output}")
```

## Task Types

### Summarization

Summarize long texts:

```python
request = TaskRequest(
    task_id="sum_001",
    task_type="summarize",
    payload={
        "text": "Your long text here...",
        "max_length": 150,
        "style": "bullet_points",  # or "paragraph"
    },
)

result = collective.execute(request)
```

### Code Generation

Generate code:

```python
request = TaskRequest(
    task_id="code_001",
    task_type="code",
    payload={
        "language": "python",
        "description": "Function to calculate factorial",
        "requirements": ["recursive", "with memoization"],
    },
)

result = collective.execute(request)
print(result.output)
```

### Analysis

Analyze text:

```python
request = TaskRequest(
    task_id="analysis_001",
    task_type="analyze",
    payload={
        "text": "I love this product! Excellent quality.",
        "analysis_type": "sentiment",  # sentiment, entities, topics, etc.
    },
)

result = collective.execute(request)
```

### Compression

Compress context:

```python
request = TaskRequest(
    task_id="compress_001",
    task_type="compress",
    payload={
        "text": "Very long document...",
        "target_ratio": 10,  # Compress 10x
    },
)

result = collective.execute(request)
```

## Targeting Specific Teams

Force tasks to specific teams:

```python
from neurectomy.core.types import AgentCapability

# Force analysis team
request = TaskRequest(
    task_id="analysis_001",
    task_type="analyze",
    payload={"text": "Text to analyze..."},
    required_capabilities=[AgentCapability.ANALYSIS],
    team="analysis",  # Optional: specify team
)

result = collective.execute(request)
```

## Multi-Agent Workflow

Chain multiple agent tasks:

```python
# Step 1: Compress context
compress_request = TaskRequest(
    task_id="step_1",
    task_type="compress",
    payload={"text": long_document, "target_ratio": 5},
)
compressed = collective.execute(compress_request)

# Step 2: Analyze compressed content
analyze_request = TaskRequest(
    task_id="step_2",
    task_type="analyze",
    payload={"text": compressed.output, "analysis_type": "sentiment"},
)
analysis = collective.execute(analyze_request)

# Step 3: Generate summary
summary_request = TaskRequest(
    task_id="step_3",
    task_type="summarize",
    payload={"text": analysis.output, "max_length": 100},
)
summary = collective.execute(summary_request)

print(f"Final result: {summary.output}")
```

## Parallel Execution

Execute multiple tasks in parallel:

```python
import concurrent.futures

tasks = [
    TaskRequest("task_1", "summarize", {"text": text1}),
    TaskRequest("task_2", "summarize", {"text": text2}),
    TaskRequest("task_3", "summarize", {"text": text3}),
]

results = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(executor.map(collective.execute, tasks))

for i, result in enumerate(results):
    print(f"Task {i+1}: {result.output}")
```

## Agent Capabilities

List available capabilities:

```python
from neurectomy.core.types import AgentCapability

capabilities = [
    AgentCapability.GENERATION,
    AgentCapability.SUMMARIZATION,
    AgentCapability.CODE_GENERATION,
    AgentCapability.ANALYSIS,
    AgentCapability.COMPRESSION,
    AgentCapability.STORAGE,
    AgentCapability.SEARCH,
]

print(f"Available capabilities: {capabilities}")
```

## Team Collaboration Example

Example of teams working together:

```python
# Step 1: Inference team generates content
gen_request = TaskRequest(
    task_id="gen_001",
    task_type="generate",
    payload={"prompt": "Write a technical article about AI"},
    team="inference",
)
article = collective.execute(gen_request)

# Step 2: Compression team optimizes it
compress_request = TaskRequest(
    task_id="comp_001",
    task_type="compress",
    payload={"text": article.output},
    team="compression",
)
compressed = collective.execute(compress_request)

# Step 3: Storage team stores it
store_request = TaskRequest(
    task_id="store_001",
    task_type="store",
    payload={"content": compressed.output, "type": "article"},
    team="storage",
)
stored = collective.execute(store_request)

# Step 4: Analysis team generates metadata
analyze_request = TaskRequest(
    task_id="analyze_001",
    task_type="analyze",
    payload={
        "text": article.output,
        "analysis_type": "metadata",
    },
    team="analysis",
)
metadata = collective.execute(analyze_request)

print(f"Article stored with ID: {stored.resource_id}")
print(f"Metadata: {metadata.output}")
```

## Advanced: Custom Capabilities

Create custom task types:

```python
request = TaskRequest(
    task_id="custom_001",
    task_type="custom_task",
    payload={
        "input": "custom data",
        "params": {"param1": "value1"},
    },
    custom_capabilities=["custom_cap_1", "custom_cap_2"],
)

result = collective.execute(request)
```

## Next Steps

- [Basic Generation Tutorial](basic-generation.md)
- [Architecture Overview](../architecture.md)
- [Examples](../../examples/)
