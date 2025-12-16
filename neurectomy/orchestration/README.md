# Orchestration Module

DAG-based workflow orchestration engine and pub/sub event system for Neurectomy.

## Overview

The Orchestration Module provides two core components:

1. **Workflow Engine**: DAG-based task orchestration with dependency resolution
2. **Event Bus**: Pub/Sub event system for asynchronous communication

## Features

- **DAG Validation**: Automatic detection of cycles and missing dependencies
- **Topological Execution**: Tasks execute in optimal dependency order
- **Error Handling**: Graceful handling of task failures with skipping
- **Execution Metrics**: Track execution time, status, and results
- **Flexible Handlers**: Register custom handlers for different task types
- **Async Support**: Full async/await support for concurrent operations

## Task States

```
PENDING     → Initial state
RUNNING     → Task execution started
COMPLETED   → Task finished successfully
FAILED      → Task encountered error
SKIPPED     → Task skipped due to failed dependency
```

## Usage

### Basic Example

```python
from neurectomy.orchestration.workflow_engine import (
    WorkflowEngine, Workflow, Task
)

# Create engine and register handlers
engine = WorkflowEngine()

async def fetch_data(config, workflow):
    """Fetch data handler"""
    return {"data": "fetched"}

async def process_data(config, workflow):
    """Process data handler"""
    return {"status": "processed"}

engine.register_handler("fetch", fetch_data)
engine.register_handler("process", process_data)

# Define workflow
workflow = Workflow(
    workflow_id="data_pipeline",
    name="Data Processing Pipeline",
    tasks={
        "fetch_task": Task(
            task_id="fetch_task",
            name="Fetch Data",
            task_type="fetch",
            config={"source": "api"},
            dependencies=[],
        ),
        "process_task": Task(
            task_id="process_task",
            name="Process Data",
            task_type="process",
            config={"method": "aggregation"},
            dependencies=["fetch_task"],
        ),
    }
)

# Execute workflow
result = await engine.execute_workflow(workflow)
print(result.to_dict())
```

### Advanced Example with Multiple Stages

```python
workflow = Workflow(
    workflow_id="ml_pipeline",
    name="ML Pipeline",
    tasks={
        # Stage 1: Data preparation
        "load": Task(
            task_id="load",
            task_type="load",
            name="Load Data",
            config={"path": "data.csv"},
            dependencies=[],
        ),
        "validate": Task(
            task_id="validate",
            task_type="validate",
            name="Validate Data",
            config={"checks": ["nulls", "types"]},
            dependencies=["load"],
        ),
        # Stage 2: Processing
        "clean": Task(
            task_id="clean",
            task_type="clean",
            name="Clean Data",
            config={"method": "standard"},
            dependencies=["validate"],
        ),
        "feature_eng": Task(
            task_id="feature_eng",
            task_type="feature_eng",
            name="Feature Engineering",
            config={"features": ["scaled", "encoded"]},
            dependencies=["clean"],
        ),
        # Stage 3: Training
        "split": Task(
            task_id="split",
            task_type="split",
            name="Train/Test Split",
            config={"ratio": 0.8},
            dependencies=["feature_eng"],
        ),
        "train": Task(
            task_id="train",
            task_type="train",
            name="Train Model",
            config={"model": "random_forest"},
            dependencies=["split"],
        ),
        # Stage 4: Evaluation
        "evaluate": Task(
            task_id="evaluate",
            task_type="evaluate",
            name="Evaluate Model",
            config={"metrics": ["accuracy", "f1"]},
            dependencies=["train"],
        ),
    }
)

result = await engine.execute_workflow(workflow)
```

## API Reference

### WorkflowEngine

#### Methods

**`__init__(max_concurrent=1)`**

- Initialize engine
- `max_concurrent`: Maximum concurrent tasks (default: 1 for sequential)

**`register_handler(task_type, handler)`**

- Register handler for task type
- `task_type`: String identifier for task type
- `handler`: Async callable(config, workflow) → result

**`async execute_workflow(workflow) → WorkflowResult`**

- Execute workflow
- Returns WorkflowResult with execution details

**`get_task_metrics(result) → dict`**

- Calculate execution metrics
- Returns dict with timing and status statistics

### Workflow

#### Methods

**`validate() → (bool, Optional[str])`**

- Validate workflow DAG
- Returns (is_valid, error_message)

**`get_execution_order() → List[str]`**

- Get tasks in topological order
- Returns list of task IDs

**`get_task_levels() → Dict[str, int]`**

- Get depth level for each task
- Returns mapping of task_id → level

### Task

#### Attributes

- `task_id`: Unique identifier
- `name`: Human-readable name
- `task_type`: Handler type
- `config`: Task configuration
- `dependencies`: List of task IDs this depends on
- `status`: TaskStatus enum
- `result`: Task output
- `error`: Error message if failed
- `start_time`: Execution start time
- `end_time`: Execution end time

#### Methods

**`get_duration() → Optional[float]`**

- Get execution duration in seconds

### WorkflowResult

#### Methods

**`to_dict() → dict`**

- Convert result to dictionary
- Includes all task results and metrics

## Error Handling

### Task Failures

When a task fails:

1. Task status set to FAILED
2. Error message captured
3. Dependent tasks are SKIPPED
4. Workflow continues with independent branches

### Workflow Validation Errors

- Empty workflow
- Unknown dependencies
- Cyclic dependencies
- Missing handlers

## Examples

### Document Processing Pipeline

```python
async def fetch_doc(config, workflow):
    # Fetch document from URL
    return {"content": "document content"}

async def extract_text(config, workflow):
    # Extract text from document
    return {"text": "extracted text"}

async def compress_text(config, workflow):
    # Compress extracted text
    return {"compressed": "compressed data"}

async def store_result(config, workflow):
    # Store result
    return {"stored": True}

engine = WorkflowEngine()
engine.register_handler("fetch", fetch_doc)
engine.register_handler("extract", extract_text)
engine.register_handler("compress", compress_text)
engine.register_handler("store", store_result)

workflow = Workflow(
    workflow_id="doc_processing",
    name="Document Processing",
    tasks={
        "fetch": Task(...),
        "extract": Task(..., dependencies=["fetch"]),
        "compress": Task(..., dependencies=["extract"]),
        "store": Task(..., dependencies=["compress"]),
    }
)

result = await engine.execute_workflow(workflow)
```

### Parallel Processing

```python
# Multiple independent tasks (parallel execution possible)
workflow = Workflow(
    workflow_id="parallel",
    name="Parallel Processing",
    tasks={
        "prep": Task(...),
        "task_a": Task(..., dependencies=["prep"]),
        "task_b": Task(..., dependencies=["prep"]),
        "task_c": Task(..., dependencies=["prep"]),
        "merge": Task(..., dependencies=["task_a", "task_b", "task_c"]),
    }
)
```

## Execution Metrics

- `total_duration`: Total workflow time
- `total_tasks`: Number of tasks
- `completed_tasks`: Successfully executed
- `failed_tasks`: Failed execution
- `skipped_tasks`: Skipped due to dependency failure
- `avg_task_duration`: Average task execution time
- `max_task_duration`: Longest task
- `min_task_duration`: Shortest task

## Integration with Neurectomy

The Workflow Engine integrates with Neurectomy components:

```python
# Inference workflow
engine.register_handler("inference", agent_complete)
engine.register_handler("compression", compress)
engine.register_handler("storage", store_file)

# Agent-based workflow
workflow_tasks = {
    "analyze": Task("analyze", "analyze", "inference", {}),
    "compress": Task("compress", "compress", "compression",
                    {}, dependencies=["analyze"]),
    "store": Task("store", "store", "storage",
                 {}, dependencies=["compress"]),
}
```

## Testing

```python
import pytest

@pytest.mark.asyncio
async def test_workflow():
    engine = WorkflowEngine()
    engine.register_handler("test", test_handler)

    workflow = Workflow(...)
    result = await engine.execute_workflow(workflow)

    assert result.status == "completed"
```

## Performance Considerations

- **Sequential Execution**: Set `max_concurrent=1` (default)
- **Parallel Execution**: Set `max_concurrent=N` for N concurrent tasks
- **Task Isolation**: Each task gets its own execution context
- **Error Recovery**: Failed tasks don't block independent branches

## See Also

- [PHASE-17B: Event Bus](../event_bus/)
- [PHASE-17C: Task Delegation](../task_delegation/)
- [Agent Supervisor](../../agents/supervisor.py)
