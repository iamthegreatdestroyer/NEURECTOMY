#!/usr/bin/env python3
"""Multi-agent collaboration example."""

from neurectomy.elite import EliteCollective
from neurectomy.core.types import TaskRequest


def main():
    """Run multi-agent collaboration example."""
    collective = EliteCollective()

    print("=" * 60)
    print("  NEURECTOMY: Multi-Agent Collaboration")
    print("=" * 60)
    print()

    print(f"Elite Collective: {len(collective.list_agents())} agents")
    print(f"Teams: {collective.list_teams()}")
    print()

    # Task 1: Analysis - Sentiment Analysis
    print("Task 1: Sentiment Analysis")
    print("-" * 60)
    request = TaskRequest(
        task_id="sentiment_001",
        task_type="analyze",
        payload={
            "text": "I absolutely love this product! Best purchase ever.",
        },
    )
    result = collective.execute(request)
    print(f"Input: {request.payload['text']}")
    print(f"Agent: {result.executing_agent}")
    print(f"Result: {result.output}")
    print()

    # Task 2: Code Generation
    print("Task 2: Code Generation")
    print("-" * 60)
    request = TaskRequest(
        task_id="code_001",
        task_type="code",
        payload={
            "description": "Python function to reverse a string",
            "language": "python",
        },
    )
    result = collective.execute(request)
    print(f"Task: {request.payload['description']}")
    print(f"Agent: {result.executing_agent}")
    print(f"Result:\n{result.output}")
    print()

    # Task 3: Summarization
    print("Task 3: Summarization")
    print("-" * 60)
    long_text = (
        "The history of artificial intelligence spans centuries. "
        "From ancient automata to modern neural networks, "
        "AI has evolved dramatically. Today's machine learning systems "
        "can process vast amounts of data and perform complex tasks. "
        "The future of AI holds tremendous potential for solving "
        "real-world problems." * 3
    )
    request = TaskRequest(
        task_id="summary_001",
        task_type="summarize",
        payload={
            "text": long_text,
            "max_length": 100,
        },
    )
    result = collective.execute(request)
    print(f"Original length: {len(long_text)} chars")
    print(f"Agent: {result.executing_agent}")
    print(f"Summary: {result.output}")
    print()

    # Task 4: Context Compression
    print("Task 4: Context Compression")
    print("-" * 60)
    request = TaskRequest(
        task_id="compress_001",
        task_type="compress",
        payload={
            "text": long_text,
            "target_ratio": 5,
        },
    )
    result = collective.execute(request)
    compressed_text = result.output if hasattr(result, 'output') else str(result)
    print(f"Original: {len(long_text)} chars")
    print(f"Compressed: {len(str(compressed_text))} chars")
    print(f"Agent: {result.executing_agent}")
    print(f"Result: {compressed_text[:100]}...")
    print()

    print("=" * 60)
    print("  All tasks completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
