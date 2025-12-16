#!/usr/bin/env python3
"""Simple text generation example."""

from neurectomy import NeurectomyOrchestrator


def main():
    """Run simple generation examples."""
    # Create orchestrator
    orchestrator = NeurectomyOrchestrator()

    # Generate text
    prompts = [
        "Explain quantum computing in simple terms.",
        "Write a haiku about programming.",
        "What are the benefits of exercise?",
    ]

    print("=" * 60)
    print("  NEURECTOMY: Simple Text Generation")
    print("=" * 60)
    print()

    for prompt in prompts:
        print(f"Prompt: {prompt}")
        print("-" * 60)

        result = orchestrator.generate(prompt, max_tokens=100)

        print(f"Response: {result.generated_text}")
        print(f"Tokens: {result.tokens_generated}")
        print(f"Latency: {result.execution_time_ms:.1f}ms")
        print()


if __name__ == "__main__":
    main()
