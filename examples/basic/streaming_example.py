#!/usr/bin/env python3
"""Streaming generation example."""

from neurectomy import NeurectomyOrchestrator


def main():
    """Run streaming generation example."""
    orchestrator = NeurectomyOrchestrator()

    print("=" * 60)
    print("  NEURECTOMY: Streaming Text Generation")
    print("=" * 60)
    print()

    prompts = [
        "Tell me a short story about a robot learning to paint.",
        "Explain how photosynthesis works.",
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"Example {i}: {prompt}")
        print("-" * 60)
        print("Streaming response:\n")

        try:
            for chunk in orchestrator.stream_generate(
                prompt,
                max_tokens=200,
            ):
                print(chunk, end="", flush=True)
        except AttributeError:
            # Fallback if stream_generate not available
            result = orchestrator.generate(prompt, max_tokens=200)
            print(result.generated_text, flush=True)

        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
