"""
Example: Document Processing Workflow
Demonstrates workflow engine with realistic document processing pipeline
"""

import asyncio
from neurectomy.orchestration.workflow_engine import (
    WorkflowEngine,
    Workflow,
    Task,
)


# Handler implementations
async def fetch_document(config: dict, workflow: Workflow):
    """Fetch document from source"""
    print(f"Fetching document from: {config.get('url', 'unknown')}")
    await asyncio.sleep(0.5)  # Simulate network delay
    return {
        "document_id": "doc_12345",
        "content": "This is sample document content " * 100,
        "size": 3500,
    }


async def extract_text(config: dict, workflow: Workflow):
    """Extract text from document"""
    print("Extracting text from document...")
    await asyncio.sleep(0.3)  # Simulate extraction
    return {
        "extracted_text": "Extracted content from document",
        "pages": 5,
        "confidence": 0.95,
    }


async def compress_content(config: dict, workflow: Workflow):
    """Compress extracted content"""
    target_ratio = config.get("target_ratio", 0.1)
    print(f"Compressing content to {target_ratio} ratio...")
    await asyncio.sleep(0.2)  # Simulate compression
    return {
        "compressed_data": "compressed_binary_data",
        "original_size": 3500,
        "compressed_size": 350,
        "ratio": target_ratio,
    }


async def store_result(config: dict, workflow: Workflow):
    """Store result in vault"""
    print("Storing result in ΣVAULT...")
    await asyncio.sleep(0.1)
    return {
        "object_id": "obj_abc123",
        "path": "processed_docs/doc_12345",
        "timestamp": "2025-01-16T10:30:00Z",
    }


async def send_notification(config: dict, workflow: Workflow):
    """Send completion notification"""
    print("Sending notification...")
    await asyncio.sleep(0.1)
    return {"notification_sent": True}


async def main():
    """Run document processing workflow"""
    # Create engine
    engine = WorkflowEngine(max_concurrent=1)
    
    # Register handlers
    engine.register_handler("fetch", fetch_document)
    engine.register_handler("extract", extract_text)
    engine.register_handler("compress", compress_content)
    engine.register_handler("store", store_result)
    engine.register_handler("notify", send_notification)
    
    # Define workflow
    workflow = Workflow(
        workflow_id="doc_processing_001",
        name="Document Processing Pipeline",
        tasks={
            "fetch": Task(
                task_id="fetch",
                name="Fetch Document",
                task_type="fetch",
                config={"url": "https://example.com/docs/sample.pdf"},
                dependencies=[],
            ),
            "extract": Task(
                task_id="extract",
                name="Extract Text",
                task_type="extract",
                config={"language": "en", "ocr": True},
                dependencies=["fetch"],
            ),
            "compress": Task(
                task_id="compress",
                name="Compress Content",
                task_type="compress",
                config={"target_ratio": 0.1, "algorithm": "lz4"},
                dependencies=["extract"],
            ),
            "store": Task(
                task_id="store",
                name="Store Result",
                task_type="store",
                config={"vault": "main", "retention": 30},
                dependencies=["compress"],
            ),
            "notify": Task(
                task_id="notify",
                name="Send Notification",
                task_type="notify",
                config={"recipient": "admin@example.com"},
                dependencies=["store"],
            ),
        },
    )
    
    # Validate workflow
    is_valid, error = workflow.validate()
    if not is_valid:
        print(f"Workflow validation failed: {error}")
        return
    
    print("Starting document processing workflow...")
    print(f"Execution order: {workflow.get_execution_order()}")
    print()
    
    # Execute workflow
    try:
        result = await engine.execute_workflow(workflow)
        
        print("\n=== Workflow Execution Results ===")
        print(f"Status: {result.status}")
        print(f"Duration: {result.get_total_duration():.2f}s")
        print()
        
        # Print task results
        for task_id, task_info in result.tasks.items():
            print(f"Task: {task_id}")
            print(f"  Status: {task_info['status']}")
            print(f"  Duration: {task_info['duration']:.2f}s" if task_info['duration'] else "  Duration: N/A")
            if task_info['result']:
                print(f"  Result: {task_info['result']}")
            if task_info['error']:
                print(f"  Error: {task_info['error']}")
            print()
        
        # Print metrics
        metrics = engine.get_task_metrics(result)
        print("=== Execution Metrics ===")
        print(f"Total tasks: {metrics['total_tasks']}")
        print(f"Completed: {metrics['completed_tasks']}")
        print(f"Failed: {metrics['failed_tasks']}")
        print(f"Skipped: {metrics['skipped_tasks']}")
        if 'avg_task_duration' in metrics:
            print(f"Average task time: {metrics['avg_task_duration']:.2f}s")
            print(f"Max task time: {metrics['max_task_duration']:.2f}s")
            print(f"Min task time: {metrics['min_task_duration']:.2f}s")
        
    except Exception as e:
        print(f"Workflow execution failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
