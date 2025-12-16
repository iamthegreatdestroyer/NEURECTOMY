# Event Bus: Pub/Sub Event System

Publish-Subscribe event system for asynchronous communication between Neurectomy components.

## Overview

The Event Bus provides a decoupled, event-driven communication layer for the Neurectomy system. Components publish events, and other components subscribe to events matching specific patterns.

## Features

- **Pub/Sub Pattern**: Decouple event producers from consumers
- **Pattern Matching**: Subscribe to events using patterns (e.g., "task._", "agent._")
- **Custom Filters**: Add filter functions for fine-grained control
- **Async Handlers**: Full async/await support for event processing
- **Event History**: Track all events with configurable history size
- **Statistics**: Monitor events and subscriptions
- **Global Bus**: Singleton global event bus instance

## Event Types

```python
EventType.TASK_STARTED = "task.started"
EventType.TASK_COMPLETED = "task.completed"
EventType.TASK_FAILED = "task.failed"
EventType.TASK_SKIPPED = "task.skipped"

EventType.WORKFLOW_STARTED = "workflow.started"
EventType.WORKFLOW_COMPLETED = "workflow.completed"
EventType.WORKFLOW_FAILED = "workflow.failed"

EventType.AGENT_HEARTBEAT = "agent.heartbeat"
EventType.AGENT_FAILED = "agent.failed"
EventType.AGENT_RECOVERED = "agent.recovered"
EventType.AGENT_REGISTERED = "agent.registered"

EventType.COMPRESSION_STARTED = "compression.started"
EventType.COMPRESSION_COMPLETED = "compression.completed"

EventType.STORAGE_UPLOADED = "storage.uploaded"
EventType.STORAGE_RETRIEVED = "storage.retrieved"
EventType.STORAGE_DELETED = "storage.deleted"

EventType.ERROR_OCCURRED = "error.occurred"
EventType.WARNING_ISSUED = "warning.issued"
EventType.INFO_LOGGED = "info.logged"
```

## Usage

### Basic Subscribe and Publish

```python
from neurectomy.orchestration.event_bus import EventBus, Event, EventType

# Create event bus
bus = EventBus()

# Define handler
async def on_task_completed(event):
    print(f"Task completed: {event.data['task_id']}")
    print(f"Result: {event.data['result']}")

# Subscribe to events
bus.subscribe("task.completed", on_task_completed)

# Emit event
event = Event(
    event_type=EventType.TASK_COMPLETED,
    source="workflow_1",
    data={"task_id": "t1", "result": "success"}
)
await bus.emit(event)
```

### Pattern Matching

```python
# Match specific event
bus.subscribe("task.completed", handler1)

# Match all task events
bus.subscribe("task.*", handler2)

# Match all events
bus.subscribe("*", handler3)
```

### Custom Filters

```python
# Subscribe with filter function
def only_critical(event):
    """Only process critical tasks"""
    return event.data.get("priority") == "critical"

async def on_critical_task_failed(event):
    print(f"CRITICAL: Task failed: {event.data['task_id']}")

bus.subscribe("task.failed", on_critical_task_failed, filter_fn=only_critical)
```

### Global Event Bus

```python
from neurectomy.orchestration.event_bus import EventBusGlobal

# Get global bus instance
bus = EventBusGlobal.get_instance()

# Use like regular bus
bus.subscribe("*", my_handler)
```

## API Reference

### EventBus

#### Methods

**`subscribe(pattern, handler, filter_fn=None) -> str`**

- Subscribe to events matching pattern
- Returns subscription ID

**`unsubscribe(subscription_id) -> bool`**

- Unsubscribe from events
- Returns True if found and removed

**`async emit(event)`**

- Emit single event to subscribers
- Calls all matching handlers

**`async emit_many(events)`**

- Emit multiple events
- Iterates and emits each event

**`get_history(pattern=None, source=None, limit=None) -> List[Event]`**

- Get event history with optional filters
- Returns matching events

**`get_subscription_count(pattern=None) -> int`**

- Get number of active subscriptions
- Optional pattern filter

**`get_stats() -> Dict`**

- Get event bus statistics
- Includes event counts and subscription info

**`clear_history()`**

- Clear all event history

**`reset_stats()`**

- Reset statistics counters

### Event

#### Attributes

- `event_type`: EventType enum value
- `source`: String identifier for event source
- `data`: Dictionary with event data
- `timestamp`: When event was created
- `event_id`: Unique event identifier

#### Methods

**`matches(pattern) -> bool`**

- Check if event matches pattern
- Supports exact, prefix wildcard, and "\*"

**`to_dict() -> Dict`**

- Convert event to dictionary
- Includes all fields in serializable format

## Examples

### Task Completion Workflow

```python
from neurectomy.orchestration import EventBusGlobal, Event, EventType

bus = EventBusGlobal.get_instance()

# 1. Log all task events
async def log_task_event(event):
    print(f"[{event.timestamp}] {event.event_type.value}: {event.source}")

bus.subscribe("task.*", log_task_event)

# 2. Send alerts on failures
async def alert_on_failure(event):
    if event.data.get("critical"):
        print(f"ALERT: Critical task failed - {event.data['reason']}")

bus.subscribe("task.failed", alert_on_failure)

# 3. Update statistics
async def update_stats(event):
    print(f"Task completed in {event.data['duration']}s")

bus.subscribe("task.completed", update_stats)

# Emit events
await bus.emit(Event(
    EventType.TASK_STARTED,
    "workflow_1",
    {"task_id": "t1"}
))

await bus.emit(Event(
    EventType.TASK_COMPLETED,
    "workflow_1",
    {"task_id": "t1", "duration": 5.2}
))
```

### Agent Monitoring

```python
async def on_agent_heartbeat(event):
    """Process agent heartbeat"""
    agent_id = event.data["agent_id"]
    status = event.data["status"]
    print(f"Agent {agent_id} reported: {status}")

async def on_agent_failed(event):
    """Alert when agent fails"""
    agent_id = event.data["agent_id"]
    print(f"ALERT: Agent {agent_id} failed!")

    # Trigger recovery
    await trigger_agent_recovery(agent_id)

bus.subscribe("agent.heartbeat", on_agent_heartbeat)
bus.subscribe("agent.failed", on_agent_failed)
```

### Event History Analysis

```python
# Get all events in last minute
history = bus.get_history()

# Get all task failures
failures = bus.get_history(pattern="task.failed")

# Get events from specific workflow
workflow_events = bus.get_history(source="workflow_1")

# Get last 10 events
recent = bus.get_history(limit=10)

# Complex filtering
critical_failures = [
    e for e in bus.get_history(pattern="task.failed")
    if e.data.get("priority") == "critical"
]
```

### Statistics and Monitoring

```python
stats = bus.get_stats()

print(f"Total events emitted: {stats['total_events_emitted']}")
print(f"Active subscriptions: {stats['total_subscriptions']}")
print(f"Event breakdown: {stats['event_counts']}")

# Monitor specific patterns
task_subs = bus.get_subscription_count("task.*")
agent_subs = bus.get_subscription_count("agent.*")
```

## Integration with Workflow Engine

```python
from neurectomy.orchestration import WorkflowEngine, EventBusGlobal, Event, EventType

engine = WorkflowEngine()
bus = EventBusGlobal.get_instance()

# Emit events during workflow execution
async def on_task_start(task_id, workflow_id):
    await bus.emit(Event(
        EventType.TASK_STARTED,
        workflow_id,
        {"task_id": task_id}
    ))

async def on_task_complete(task_id, workflow_id, result):
    await bus.emit(Event(
        EventType.TASK_COMPLETED,
        workflow_id,
        {"task_id": task_id, "result": result}
    ))

async def on_task_fail(task_id, workflow_id, error):
    await bus.emit(Event(
        EventType.TASK_FAILED,
        workflow_id,
        {"task_id": task_id, "error": error}
    ))
```

## Integration with Agent Supervisor

```python
from neurectomy.agents.supervisor import AgentSupervisor
from neurectomy.orchestration import EventBusGlobal, Event, EventType

supervisor = AgentSupervisor()
bus = EventBusGlobal.get_instance()

async def monitor_agents():
    """Emit agent events based on supervisor state"""
    while True:
        for agent_id, health in supervisor.get_all_health().items():
            # Emit heartbeat
            await bus.emit(Event(
                EventType.AGENT_HEARTBEAT,
                "supervisor",
                {
                    "agent_id": agent_id,
                    "status": health.status.value,
                    "last_heartbeat": health.last_heartbeat.isoformat()
                }
            ))

            # Emit failure alerts
            if health.status.value == "failed":
                await bus.emit(Event(
                    EventType.AGENT_FAILED,
                    "supervisor",
                    {"agent_id": agent_id, "error": health.last_error}
                ))

        await asyncio.sleep(10)  # Check every 10 seconds
```

## Best Practices

### 1. Use Specific Event Types

```python
# Good - specific
await bus.emit(Event(EventType.TASK_COMPLETED, ...))

# Avoid - generic
await bus.emit(Event("custom.event", ...))
```

### 2. Include Relevant Data

```python
# Good - complete information
data = {
    "task_id": "t1",
    "duration": 5.2,
    "result": result,
    "status": "success",
    "timestamp": datetime.now().isoformat()
}

# Avoid - minimal data
data = {"done": True}
```

### 3. Use Pattern Subscriptions

```python
# Good - pattern
bus.subscribe("task.*", handler)
bus.subscribe("agent.*", handler)

# Avoid - too many subscriptions
bus.subscribe("task.completed", handler1)
bus.subscribe("task.failed", handler2)
bus.subscribe("task.started", handler3)
bus.subscribe("task.skipped", handler4)
```

### 4. Filter at Handler Level

```python
# Good - filter in handler
async def handler(event):
    if event.data.get("priority") != "critical":
        return
    # Process critical events

# Better - use filter function
def critical_only(event):
    return event.data.get("priority") == "critical"

bus.subscribe("task.failed", handler, filter_fn=critical_only)
```

## Testing

```python
import pytest

@pytest.mark.asyncio
async def test_task_completion():
    bus = EventBus()
    events = []

    async def handler(event):
        events.append(event)

    bus.subscribe("task.completed", handler)

    event = Event(EventType.TASK_COMPLETED, "test")
    await bus.emit(event)

    assert len(events) == 1
    assert events[0].event_type == EventType.TASK_COMPLETED
```

## Performance Considerations

- **Subscribers**: Use pattern matching to reduce unnecessary calls
- **History Size**: Set appropriate max_history based on memory constraints
- **Async Handlers**: Keep handlers fast; use background tasks for long operations
- **Event Size**: Keep event.data reasonably small

## See Also

- [Workflow Engine](workflow_engine.md)
- [Task Delegation](task_delegation.md)
- [Agent Supervisor](../agents/supervisor.py)
