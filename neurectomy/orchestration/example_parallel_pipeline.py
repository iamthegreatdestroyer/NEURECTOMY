"""
Example: Parallel Processing Workflow
Demonstrates parallel task execution with shared dependencies
"""

import asyncio
from neurectomy.orchestration.workflow_engine import (
    WorkflowEngine,
    Workflow,
    Task,
)


# Handler implementations
async def prepare_data(config: dict, workflow: Workflow):
    """Prepare data for processing"""
    print("Preparing data...")
    await asyncio.sleep(0.3)
    return {"data": [1, 2, 3, 4, 5]}


async def process_task_a(config: dict, workflow: Workflow):
    """Process task A (parallel)"""
    print("Running task A...")
    await asyncio.sleep(0.4)
    return {"result_a": "processed_a"}


async def process_task_b(config: dict, workflow: Workflow):
    """Process task B (parallel)"""
    print("Running task B...")
    await asyncio.sleep(0.3)
    return {"result_b": "processed_b"}


async def process_task_c(config: dict, workflow: Workflow):
    """Process task C (parallel)"""
    print("Running task C...")
    await asyncio.sleep(0.5)
    return {"result_c": "processed_c"}


async def merge_results(config: dict, workflow: Workflow):
    """Merge results from parallel tasks"""
    print("Merging results...")
    await asyncio.sleep(0.2)
    return {
        "merged": {
            "task_a": "processed_a",
            "task_b": "processed_b",
            "task_c": "processed_c",
        }
    }


async def main():
    """Run parallel processing workflow"""
    # Create engine
    engine = WorkflowEngine(max_concurrent=1)
    
    # Register handlers
    engine.register_handler("prepare", prepare_data)
    engine.register_handler("process_a", process_task_a)
    engine.register_handler("process_b", process_task_b)
    engine.register_handler("process_c", process_task_c)
    engine.register_handler("merge", merge_results)
    
    # Define workflow with parallel tasks
    workflow = Workflow(
        workflow_id="parallel_processing",
        name="Parallel Processing Pipeline",
        tasks={
            # Stage 1: Preparation
            "prep": Task(
                task_id="prep",
                name="Prepare Data",
                task_type="prepare",
                config={"format": "json"},
                dependencies=[],
            ),
            # Stage 2: Parallel processing (all depend on prep)
            "task_a": Task(
                task_id="task_a",
                name="Process Task A",
                task_type="process_a",
                config={"method": "a"},
                dependencies=["prep"],
            ),
            "task_b": Task(
                task_id="task_b",
                name="Process Task B",
                task_type="process_b",
                config={"method": "b"},
                dependencies=["prep"],
            ),
            "task_c": Task(
                task_id="task_c",
                name="Process Task C",
                task_type="process_c",
                config={"method": "c"},
                dependencies=["prep"],
            ),
            # Stage 3: Merge results (depends on all parallel tasks)
            "merge": Task(
                task_id="merge",
                name="Merge Results",
                task_type="merge",
                config={"strategy": "combine"},
                dependencies=["task_a", "task_b", "task_c"],
            ),
        },
    )
    
    # Print execution plan
    print("=== Workflow Execution Plan ===")
    print(f"Workflow: {workflow.name}")
    print(f"Execution order: {workflow.get_execution_order()}")
    print()
    
    task_levels = workflow.get_task_levels()
    print("Task levels (parallelizable at same level):")
    for level in sorted(set(task_levels.values())):
        tasks_at_level = [tid for tid, lvl in task_levels.items() if lvl == level]
        print(f"  Level {level}: {tasks_at_level}")
    print()
    
    # Execute workflow
    print("Starting parallel processing workflow...")
    print()
    
    try:
        result = await engine.execute_workflow(workflow)
        
        print("\n=== Workflow Results ===")
        print(f"Status: {result.status}")
        print(f"Total Duration: {result.get_total_duration():.2f}s")
        print()
        
        # Print detailed results
        for task_id, task_info in result.tasks.items():
            print(f"[{task_info['status'].upper()}] {task_id}")
            if task_info['duration']:
                print(f"  Duration: {task_info['duration']:.2f}s")
            if task_info['result']:
                print(f"  Result: {task_info['result']}")
        
        # Print metrics
        print()
        metrics = engine.get_task_metrics(result)
        print("=== Performance Metrics ===")
        print(f"Total tasks: {metrics['total_tasks']}")
        print(f"Completed: {metrics['completed_tasks']}")
        print(f"Total workflow time: {metrics['total_duration']:.2f}s")
        if 'avg_task_duration' in metrics:
            print(f"Average task time: {metrics['avg_task_duration']:.2f}s")
            print(f"Sum of all task times: {sum(t['duration'] for t in result.tasks.values() if t['duration']):.2f}s")
            print(f"Parallelization opportunity: {sum(t['duration'] for t in result.tasks.values() if t['duration']) - metrics['total_duration']:.2f}s")
        
    except Exception as e:
        print(f"Workflow execution failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
