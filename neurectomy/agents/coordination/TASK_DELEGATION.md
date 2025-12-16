# Task Delegation System

Dynamic task delegation across the Elite Agent Collective based on capabilities and load.

## Overview

The Task Delegation System intelligently assigns tasks to agents based on:

- **Capability Matching**: Tasks only assigned to capable agents
- **Load Balancing**: Prefers agents with lower utilization
- **Specialization**: Tracks and uses agent specialization scores
- **Success Rates**: Learns from past performance
- **Priority Handling**: Critical tasks get preferential placement
- **Pending Queue**: Manages tasks awaiting available agents

## Features

- **Capability-Based Matching**: Filter agents by required capabilities
- **Load-Aware Selection**: Prefer less loaded agents
- **Specialization Scoring**: Track proficiency per task type
- **Success Rate Tracking**: Learn from historical performance
- **Pending Task Queue**: Buffer tasks when no agents available
- **Delegation History**: Track all delegation decisions
- **Status Monitoring**: Real-time visibility into agent and collective status

## Architecture

### Scoring Algorithm

Each agent receives a score based on:

1. **Load Score** (60% weight)
   - Utilization: `active_tasks / max_concurrent_tasks`
   - Score: `1.0 - utilization`
   - Prefers agents with available capacity

2. **Specialization Score** (30% weight)
   - Per-task-type proficiency (0.0 - 1.0)
   - Default: 1.0 for registered capabilities
   - Trained through explicit configuration

3. **Success Rate** (10% weight)
   - Historical success rate per task type
   - Updated via exponential moving average
   - Default: 1.0 (100% success)

4. **Priority Multiplier**
   - Applied to combined score
   - Formula: `1.0 + (priority_value * 0.15)`
   - Ranges from 1.15 (LOW) to 1.60 (CRITICAL)

### Combined Score

```
total_score = (
    load_score * 0.6 +
    specialization_score * 0.3 +
    success_rate * 0.1
) * priority_multiplier
```

## Usage

### Basic Setup

```python
from neurectomy.agents.coordination.task_delegation import (
    TaskDelegator,
    AgentCapability,
    Task,
    TaskPriority,
)

# Create delegator
delegator = TaskDelegator()

# Register agents
inference_agent = AgentCapability(
    agent_id="TENSOR",
    capabilities={"inference", "training", "optimization"},
    max_concurrent_tasks=3,
    specialization_score={
        "inference": 1.0,
        "training": 0.9,
        "optimization": 0.8,
    },
)
delegator.register_agent(inference_agent)

compression_agent = AgentCapability(
    agent_id="VELOCITY",
    capabilities={"compression", "decompression"},
    max_concurrent_tasks=5,
    specialization_score={
        "compression": 1.0,
        "decompression": 1.0,
    },
)
delegator.register_agent(compression_agent)

# Delegate task
task = Task(
    task_id="task_001",
    task_type="inference",
    priority=TaskPriority.HIGH,
    payload={"model": "gpt-4", "prompt": "Hello"},
    required_capabilities={"inference"},
)

result = await delegator.delegate_task(task)
print(f"Task assigned to: {result.agent_id}")
print(f"Score: {result.score:.3f}")
```

### Register Elite Agents

```python
# Register multiple specialized agents
agents_config = {
    "APEX": {
        "capabilities": {"code_review", "architecture", "optimization"},
        "max_concurrent": 2,
        "specialization": {
            "code_review": 1.0,
            "architecture": 0.95,
            "optimization": 0.85,
        },
    },
    "CIPHER": {
        "capabilities": {"security_audit", "encryption", "threat_modeling"},
        "max_concurrent": 3,
        "specialization": {
            "security_audit": 1.0,
            "encryption": 0.98,
            "threat_modeling": 0.90,
        },
    },
    "TENSOR": {
        "capabilities": {"inference", "training", "optimization"},
        "max_concurrent": 4,
        "specialization": {
            "inference": 1.0,
            "training": 0.95,
            "optimization": 0.85,
        },
    },
}

for agent_id, config in agents_config.items():
    agent = AgentCapability(
        agent_id=agent_id,
        capabilities=set(config["capabilities"]),
        max_concurrent_tasks=config["max_concurrent"],
        specialization_score=config["specialization"],
    )
    delegator.register_agent(agent)
```

### Monitor Collective

```python
# Get status of entire collective
status = delegator.get_collective_status()
print(f"Agents: {status['total_agents']}")
print(f"Capacity: {status['total_capacity']}")
print(f"Active: {status['active_tasks']}")
print(f"Utilization: {status['utilization']:.1%}")

# Get specific agent status
agent_status = delegator.get_agent_status("TENSOR")
print(f"Agent: {agent_status['agent_id']}")
print(f"Load: {agent_status['active_tasks']}/{agent_status['max_tasks']}")
print(f"Capabilities: {agent_status['capabilities']}")
```

### Handle Task Completion

```python
# Task completes
await delegator.mark_task_complete(
    task_id="task_001",
    agent_id="TENSOR",
    success=True,
    task_type="inference"
)

# If agents now have capacity, pending tasks are processed
print(f"Pending tasks: {len(delegator.pending_tasks)}")
```

## API Reference

### TaskDelegator

#### Methods

**`register_agent(agent: AgentCapability)`**

- Register an agent with capabilities
- Tracks max concurrent tasks and specialization

**`async delegate_task(task: Task) -> DelegationResult`**

- Delegate task to best available agent
- Returns DelegationResult with assignment details
- Queues task if no agents available

**`async mark_task_complete(task_id, agent_id, success, task_type)`**

- Mark task as complete
- Frees agent capacity
- Updates success rates
- Processes pending tasks

**`get_agent_status(agent_id: str) -> Dict`**

- Get detailed status of single agent
- Includes load, capabilities, specialization

**`get_collective_status() -> Dict`**

- Get aggregated status of all agents
- Includes total capacity, utilization, pending

**`get_delegation_history(limit, agent_id, success_only) -> List`**

- Get historical delegation decisions
- Optional filtering by agent, success status

**`get_stats() -> Dict`**

- Get comprehensive statistics
- Success rates, average scores

### AgentCapability

#### Attributes

- `agent_id`: Unique agent identifier
- `capabilities`: Set of task types agent can handle
- `max_concurrent_tasks`: Maximum tasks simultaneously
- `active_tasks`: Current task count
- `specialization_score`: Per-task-type proficiency (0.0-1.0)
- `success_rate`: Per-task-type success rate (0.0-1.0)

#### Methods

- `get_utilization()`: Current capacity usage
- `has_capacity()`: Check if can accept more tasks
- `can_handle(required_capabilities)`: Check capability match

### Task

#### Attributes

- `task_id`: Unique task identifier
- `task_type`: Task classification (e.g., "inference")
- `priority`: TaskPriority level
- `payload`: Task-specific data
- `required_capabilities`: Set of needed capabilities
- `created_at`: Creation timestamp
- `deadline`: Optional task deadline

### DelegationResult

#### Attributes

- `task_id`: Task that was delegated
- `agent_id`: Agent assigned (None if failed)
- `success`: Whether delegation succeeded
- `score`: Agent's delegation score
- `reason`: Failure reason if applicable
- `timestamp`: Delegation timestamp

## Examples

### Specialized Task Routing

```python
async def handle_document_processing():
    """Route document tasks to specialized agents"""

    # Register document processing specialists
    ocr_agent = AgentCapability(
        agent_id="OCR_SPECIALIST",
        capabilities={"ocr", "text_extraction"},
        max_concurrent_tasks=2,
        specialization_score={"ocr": 1.0, "text_extraction": 0.95},
    )
    delegator.register_agent(ocr_agent)

    compression_agent = AgentCapability(
        agent_id="COMPRESSION_SPECIALIST",
        capabilities={"compression", "decompression"},
        max_concurrent_tasks=4,
        specialization_score={"compression": 1.0},
    )
    delegator.register_agent(compression_agent)

    # Delegate OCR task
    ocr_task = Task(
        task_id="ocr_001",
        task_type="ocr",
        priority=TaskPriority.HIGH,
        payload={"document": "scan.pdf"},
        required_capabilities={"ocr"},
    )
    result = await delegator.delegate_task(ocr_task)
    print(f"OCR task: {result.agent_id}")

    # Delegate compression task
    compress_task = Task(
        task_id="compress_001",
        task_type="compression",
        priority=TaskPriority.MEDIUM,
        payload={"file_size": 500000},
        required_capabilities={"compression"},
    )
    result = await delegator.delegate_task(compress_task)
    print(f"Compression task: {result.agent_id}")
```

### Load Balancing

```python
async def handle_inference_workload():
    """Distribute inference tasks across multiple agents"""

    # Register multiple inference agents
    for i in range(1, 4):
        agent = AgentCapability(
            agent_id=f"INFERENCE_{i}",
            capabilities={"inference"},
            max_concurrent_tasks=5,
        )
        delegator.register_agent(agent)

    # Create multiple tasks
    tasks = [
        Task(
            task_id=f"inference_{i}",
            task_type="inference",
            priority=TaskPriority.MEDIUM,
            payload={"input": f"data_{i}"},
            required_capabilities={"inference"},
        )
        for i in range(1, 6)
    ]

    # Delegate all tasks
    for task in tasks:
        result = await delegator.delegate_task(task)
        print(f"Task {task.task_id} → {result.agent_id}")

    # Check distribution
    status = delegator.get_collective_status()
    print(f"Utilization: {status['utilization']:.1%}")
```

### Priority-Based Routing

```python
async def handle_critical_task():
    """Route critical tasks preferentially"""

    agent = AgentCapability(
        agent_id="GENERAL_AGENT",
        capabilities={"processing"},
        max_concurrent_tasks=2,
    )
    delegator.register_agent(agent)

    # Low priority task (goes to pending)
    low_task = Task(
        task_id="low_001",
        task_type="processing",
        priority=TaskPriority.LOW,
        payload={},
        required_capabilities={"processing"},
    )

    # Critical task (gets better score)
    critical_task = Task(
        task_id="critical_001",
        task_type="processing",
        priority=TaskPriority.CRITICAL,
        payload={},
        required_capabilities={"processing"},
    )

    # First low task takes slot
    result1 = await delegator.delegate_task(low_task)

    # Critical task still processed (score helps if slot opens)
    result2 = await delegator.delegate_task(critical_task)
```

## Performance Considerations

1. **Scoring Efficiency**
   - O(n) where n = number of candidate agents
   - Minimal computation per delegation

2. **Pending Queue**
   - Default: 1000 maximum pending tasks
   - Configurable at delegator creation

3. **History Retention**
   - All delegations tracked indefinitely
   - Consider pruning for long-running systems

4. **Agent Overhead**
   - Minimal memory per agent (~100 bytes base)
   - Scales to thousands of agents

## Testing

```python
import pytest

@pytest.mark.asyncio
async def test_specialized_delegation():
    delegator = TaskDelegator()

    # Setup
    agent = AgentCapability(
        agent_id="specialist",
        capabilities={"inference"},
        specialization_score={"inference": 1.0},
    )
    delegator.register_agent(agent)

    # Test
    task = Task(
        task_id="task_1",
        task_type="inference",
        priority=TaskPriority.HIGH,
        payload={},
        required_capabilities={"inference"},
    )
    result = await delegator.delegate_task(task)

    assert result.success
    assert result.agent_id == "specialist"
```

## Integration with Workflow Engine

```python
from neurectomy.orchestration import WorkflowEngine, Event, EventType

async def delegate_workflow_tasks(workflow, delegator):
    """Delegate workflow tasks to agents"""

    for task in workflow.tasks.values():
        # Convert workflow task to delegation task
        delegation_task = Task(
            task_id=task.task_id,
            task_type=task.task_type,
            priority=TaskPriority.HIGH,
            payload=task.config,
            required_capabilities={task.task_type},
        )

        # Delegate
        result = await delegator.delegate_task(delegation_task)

        # Emit event
        await bus.emit(Event(
            EventType.TASK_STARTED,
            workflow.workflow_id,
            {"task_id": task.task_id, "agent": result.agent_id}
        ))
```

## See Also

- [Workflow Engine](../orchestration/workflow_engine.md)
- [Event Bus](../orchestration/event_bus.md)
- [Agent Supervisor](supervisor.py)
