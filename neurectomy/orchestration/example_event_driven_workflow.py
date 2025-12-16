"""
Example: Event-Driven Workflow Processing
Demonstrates integration of event bus with workflow engine
"""

import asyncio
from neurectomy.orchestration.workflow_engine import (
    WorkflowEngine,
    Workflow,
    Task,
)
from neurectomy.orchestration.event_bus import (
    EventBus,
    Event,
    EventType,
)


# Workflow task handlers
async def fetch_data(config: dict, workflow: Workflow):
    """Fetch data task"""
    await asyncio.sleep(0.3)
    return {"data": "fetched data"}


async def process_data(config: dict, workflow: Workflow):
    """Process data task"""
    await asyncio.sleep(0.5)
    return {"processed": "processed data"}


async def save_results(config: dict, workflow: Workflow):
    """Save results task"""
    await asyncio.sleep(0.2)
    return {"saved": True}


# Event handlers
async def on_workflow_started(event: Event):
    """Handle workflow start"""
    print(f"✅ Workflow started: {event.data['workflow_id']}")


async def on_task_started(event: Event):
    """Handle task start"""
    task_id = event.data.get("task_id")
    print(f"  → Starting task: {task_id}")


async def on_task_completed(event: Event):
    """Handle task completion"""
    task_id = event.data.get("task_id")
    duration = event.data.get("duration", 0)
    print(f"  ✓ Completed: {task_id} ({duration:.2f}s)")


async def on_task_failed(event: Event):
    """Handle task failure"""
    task_id = event.data.get("task_id")
    error = event.data.get("error")
    print(f"  ✗ Failed: {task_id} - {error}")


async def on_workflow_completed(event: Event):
    """Handle workflow completion"""
    workflow_id = event.data['workflow_id']
    duration = event.data.get('duration', 0)
    print(f"✅ Workflow completed: {workflow_id} ({duration:.2f}s)")


async def main():
    """Run event-driven workflow example"""
    print("=" * 60)
    print("EVENT-DRIVEN WORKFLOW PROCESSING")
    print("=" * 60)
    print()
    
    # Create components
    engine = WorkflowEngine()
    bus = EventBus()
    
    # Register workflow handlers
    engine.register_handler("fetch", fetch_data)
    engine.register_handler("process", process_data)
    engine.register_handler("save", save_results)
    
    # Register event handlers
    bus.subscribe("workflow.started", on_workflow_started)
    bus.subscribe("task.started", on_task_started)
    bus.subscribe("task.completed", on_task_completed)
    bus.subscribe("task.failed", on_task_failed)
    bus.subscribe("workflow.completed", on_workflow_completed)
    
    print("📝 Registered event handlers")
    print(f"   Total subscriptions: {bus.get_subscription_count()}\n")
    
    # Define workflow
    workflow = Workflow(
        workflow_id="data_processing_flow",
        name="Data Processing Pipeline",
        tasks={
            "fetch": Task(
                task_id="fetch",
                name="Fetch Data",
                task_type="fetch",
                config={"source": "api"},
                dependencies=[],
            ),
            "process": Task(
                task_id="process",
                name="Process Data",
                task_type="process",
                config={"method": "aggregation"},
                dependencies=["fetch"],
            ),
            "save": Task(
                task_id="save",
                name="Save Results",
                task_type="save",
                config={"destination": "database"},
                dependencies=["process"],
            ),
        },
    )
    
    # Emit workflow started event
    await bus.emit(Event(
        EventType.WORKFLOW_STARTED,
        "event_driven_example",
        {"workflow_id": workflow.workflow_id}
    ))
    print()
    
    # Execute workflow
    print("Running workflow...")
    print("-" * 60)
    
    workflow_start = asyncio.get_event_loop().time()
    
    try:
        result = await engine.execute_workflow(workflow)
        
        workflow_end = asyncio.get_event_loop().time()
        workflow_duration = workflow_end - workflow_start
        
        # Emit task events during execution
        for task_id, task_info in result.tasks.items():
            if task_info["status"] == "completed":
                await bus.emit(Event(
                    EventType.TASK_COMPLETED,
                    workflow.workflow_id,
                    {
                        "task_id": task_id,
                        "duration": task_info["duration"],
                        "result": task_info["result"]
                    }
                ))
        
        # Emit workflow completed event
        await bus.emit(Event(
            EventType.WORKFLOW_COMPLETED,
            "event_driven_example",
            {
                "workflow_id": workflow.workflow_id,
                "duration": workflow_duration,
                "status": result.status
            }
        ))
        
        print()
        print("=" * 60)
        print("WORKFLOW EXECUTION SUMMARY")
        print("=" * 60)
        print()
        
        # Print metrics
        metrics = engine.get_task_metrics(result)
        print(f"Status: {result.status}")
        print(f"Total duration: {metrics['total_duration']:.2f}s")
        print(f"Tasks completed: {metrics['completed_tasks']}/{metrics['total_tasks']}")
        print()
        
        # Print event statistics
        stats = bus.get_stats()
        print("=" * 60)
        print("EVENT STATISTICS")
        print("=" * 60)
        print()
        print(f"Total events emitted: {stats['total_events_emitted']}")
        print(f"Active subscriptions: {stats['total_subscriptions']}")
        print()
        
        print("Event breakdown:")
        for event_type, count in sorted(stats['event_counts'].items()):
            print(f"  {event_type}: {count}")
        print()
        
        # Print event history
        history = bus.get_history()
        print("=" * 60)
        print("EVENT HISTORY")
        print("=" * 60)
        print()
        
        for i, event in enumerate(history, 1):
            print(f"{i}. {event.event_type.value}")
            print(f"   Source: {event.source}")
            print(f"   Data: {event.data}")
        
    except Exception as e:
        print(f"Workflow execution failed: {e}")
        
        # Emit workflow failed event
        await bus.emit(Event(
            EventType.WORKFLOW_FAILED,
            "event_driven_example",
            {
                "workflow_id": workflow.workflow_id,
                "error": str(e)
            }
        ))


if __name__ == "__main__":
    asyncio.run(main())
