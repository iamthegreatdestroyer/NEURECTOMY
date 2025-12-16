"""
Example: Real-time Event Monitoring
Demonstrates event bus monitoring and filtering
"""

import asyncio
from datetime import datetime
from neurectomy.orchestration.event_bus import (
    EventBus,
    Event,
    EventType,
    EventBusGlobal,
)


# Handler implementations
async def log_all_events(event: Event):
    """Log all events"""
    print(f"[LOG] {event.timestamp.strftime('%H:%M:%S')} - "
          f"{event.event_type.value} from {event.source}")


async def alert_on_failure(event: Event):
    """Send alert on task failure"""
    task_id = event.data.get("task_id", "unknown")
    error = event.data.get("error", "unknown error")
    print(f"\n⚠️  ALERT: Task {task_id} failed!")
    print(f"   Error: {error}")
    print(f"   Source: {event.source}\n")


async def track_performance(event: Event):
    """Track task performance"""
    duration = event.data.get("duration", 0)
    task_id = event.data.get("task_id", "unknown")
    
    if duration > 5:
        print(f"⏱️  SLOW TASK: {task_id} took {duration:.2f}s")


async def agent_status_monitor(event: Event):
    """Monitor agent status"""
    agent_id = event.data.get("agent_id", "unknown")
    status = event.data.get("status", "unknown")
    print(f"🤖 Agent {agent_id}: {status}")


async def update_metrics(event: Event):
    """Update system metrics"""
    print(f"📊 Metric update: {event.event_type.value}")


async def main():
    """Run event bus monitoring example"""
    print("=" * 60)
    print("EVENT BUS MONITORING SYSTEM")
    print("=" * 60)
    print()
    
    # Get global bus
    bus = EventBusGlobal.get_instance()
    
    # Register handlers
    print("📝 Registering event handlers...")
    bus.subscribe("*", log_all_events)
    bus.subscribe("task.failed", alert_on_failure)
    bus.subscribe("task.completed", track_performance)
    bus.subscribe("agent.*", agent_status_monitor)
    bus.subscribe("task.*", update_metrics)
    print(f"   Total subscriptions: {bus.get_subscription_count()}\n")
    
    # Simulate task events
    print("Starting task events...")
    print("-" * 60)
    
    await bus.emit(Event(
        EventType.TASK_STARTED,
        "workflow_1",
        {"task_id": "task_001"}
    ))
    await asyncio.sleep(0.5)
    
    await bus.emit(Event(
        EventType.TASK_COMPLETED,
        "workflow_1",
        {"task_id": "task_001", "duration": 2.5, "result": "success"}
    ))
    await asyncio.sleep(0.5)
    
    await bus.emit(Event(
        EventType.TASK_STARTED,
        "workflow_1",
        {"task_id": "task_002"}
    ))
    await asyncio.sleep(0.5)
    
    # Simulate slow task
    await bus.emit(Event(
        EventType.TASK_COMPLETED,
        "workflow_1",
        {"task_id": "task_002", "duration": 7.3, "result": "success"}
    ))
    await asyncio.sleep(0.5)
    
    # Simulate task failure
    await bus.emit(Event(
        EventType.TASK_FAILED,
        "workflow_1",
        {"task_id": "task_003", "error": "Connection timeout", "priority": "high"}
    ))
    await asyncio.sleep(0.5)
    
    # Simulate agent events
    print("\nStarting agent events...")
    print("-" * 60)
    
    await bus.emit(Event(
        EventType.AGENT_REGISTERED,
        "supervisor",
        {"agent_id": "APEX", "status": "active"}
    ))
    await asyncio.sleep(0.3)
    
    await bus.emit(Event(
        EventType.AGENT_HEARTBEAT,
        "supervisor",
        {"agent_id": "CIPHER", "status": "healthy"}
    ))
    await asyncio.sleep(0.3)
    
    await bus.emit(Event(
        EventType.AGENT_HEARTBEAT,
        "supervisor",
        {"agent_id": "ARCHITECT", "status": "healthy"}
    ))
    await asyncio.sleep(0.3)
    
    await bus.emit(Event(
        EventType.AGENT_FAILED,
        "supervisor",
        {"agent_id": "VELOCITY", "status": "degraded"}
    ))
    await asyncio.sleep(0.3)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("EVENT STATISTICS")
    print("=" * 60)
    
    stats = bus.get_stats()
    print(f"Total events emitted: {stats['total_events_emitted']}")
    print(f"Total subscriptions: {stats['total_subscriptions']}")
    print(f"Active patterns: {', '.join(stats['patterns'])}")
    print()
    
    print("Event breakdown:")
    for event_type, count in sorted(stats['event_counts'].items()):
        print(f"  {event_type}: {count}")
    print()
    
    # Analyze history
    print("=" * 60)
    print("EVENT HISTORY ANALYSIS")
    print("=" * 60)
    print()
    
    # All events
    all_events = bus.get_history()
    print(f"Total events in history: {len(all_events)}")
    print()
    
    # Failed tasks
    failed_tasks = bus.get_history(pattern="task.failed")
    print(f"Failed tasks: {len(failed_tasks)}")
    for event in failed_tasks:
        print(f"  - {event.data['task_id']}: {event.data['error']}")
    print()
    
    # Agent events
    agent_events = bus.get_history(pattern="agent.*")
    print(f"Agent events: {len(agent_events)}")
    for event in agent_events:
        print(f"  - {event.event_type.value}: {event.data['agent_id']}")
    print()
    
    # Workflow 1 events
    workflow_events = bus.get_history(source="workflow_1")
    print(f"Events from workflow_1: {len(workflow_events)}")
    for event in workflow_events:
        print(f"  - {event.event_type.value}: {event.data.get('task_id', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(main())
