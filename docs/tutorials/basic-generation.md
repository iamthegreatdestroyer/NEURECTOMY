# Tutorial: Basic Text Generation

## Overview

Learn how to generate text using Neurectomy.

## Prerequisites

- Neurectomy installed (`pip install neurectomy`)
- API server running (optional for SDK-only usage)
- Python 3.8+

## Step 1: Simple Generation

Start with the simplest example:

```python
from neurectomy import NeurectomyOrchestrator

orchestrator = NeurectomyOrchestrator()

result = orchestrator.generate(
    prompt="What is artificial intelligence?",
    max_tokens=100,
)

print(result.generated_text)
```

Output:

```
Artificial intelligence (AI) is the simulation of human intelligence processes by machines,
especially computer systems. These processes include learning, reasoning, problem-solving,
perception, and language understanding...
```

## Step 2: Adjusting Parameters

Control the output with different parameters:

```python
# More creative output
result = orchestrator.generate(
    prompt="Write a creative story about robots",
    max_tokens=200,
    temperature=0.9,  # Higher = more creative (0.0-1.0)
)

# More focused/deterministic output
result = orchestrator.generate(
    prompt="Explain photosynthesis",
    max_tokens=150,
    temperature=0.3,  # Lower = more focused
)

# Very strict/factual
result = orchestrator.generate(
    prompt="List the planets in order",
    max_tokens=100,
    temperature=0.1,  # Very low for factual content
)
```

## Step 3: Using Context

Provide context for better responses:

```python
context = """
The following is a conversation about Python programming.

User: How do I create a list?
Assistant: You can create a list using square brackets: my_list = [1, 2, 3]

User: How do I add items?
"""

result = orchestrator.generate(
    prompt=context + "Assistant:",
    max_tokens=100,
)

print(result.generated_text)
# Output: You can use the append() method: my_list.append(4)
```

## Step 4: Multiple Generations

Generate multiple completions:

```python
prompts = [
    "Write a haiku about programming",
    "Explain machine learning",
    "What are best practices for Python?",
]

for prompt in prompts:
    print(f"\nPrompt: {prompt}")
    print("-" * 50)

    result = orchestrator.generate(
        prompt=prompt,
        max_tokens=150,
    )

    print(f"Response: {result.generated_text}")
    print(f"Tokens: {result.tokens_generated}")
    print(f"Latency: {result.execution_time_ms:.1f}ms")
```

## Step 5: Using the Python SDK

Use the SDK for API-based access:

```python
from neurectomy.sdk import NeurectomyClient

# Create client (connects to running API server)
client = NeurectomyClient()

# Generate text
response = client.generate(
    prompt="Explain quantum computing",
    max_tokens=200,
)

print(response["text"])
print(f"Tokens: {response['tokens']}")
```

## Step 6: Streaming

Get responses token-by-token:

```python
print("Streaming response:")
print("-" * 40)

for chunk in orchestrator.stream_generate(
    prompt="Tell me a short story about an astronaut",
    max_tokens=200,
):
    print(chunk, end="", flush=True)

print("\n" + "-" * 40)
print("Done!")
```

## Step 7: Error Handling

Handle errors gracefully:

```python
from neurectomy.sdk.errors import NeurectomyError

try:
    response = client.generate(
        prompt="Your prompt",
        max_tokens=100,
    )
    print(response["text"])
except NeurectomyError as e:
    print(f"Error: {e}")
```

## Common Parameters

| Parameter     | Type  | Default | Description          |
| ------------- | ----- | ------- | -------------------- |
| `prompt`      | str   | -       | Input prompt         |
| `max_tokens`  | int   | 100     | Max output tokens    |
| `temperature` | float | 0.7     | Creativity (0.0-1.0) |
| `top_p`       | float | 0.9     | Nucleus sampling     |
| `stream`      | bool  | False   | Stream output        |

## Tips & Tricks

### Better Prompts

```python
# ❌ Vague
result = orchestrator.generate("Tell me about Python")

# ✅ Specific
result = orchestrator.generate(
    "Explain Python decorators with examples",
    max_tokens=300,
    temperature=0.3,
)
```

### Few-Shot Examples

```python
prompt = """
Examples:
Q: What's 2+2?
A: 4

Q: What's 5*3?
A: 15

Q: What's 100/5?
A:"""

result = orchestrator.generate(prompt)
```

### System Prompts

```python
system = "You are a helpful Python programming assistant."
prompt = "How do I read a file?"

result = orchestrator.generate(
    prompt=f"{system}\n\nUser: {prompt}\nAssistant:",
)
```

## Next Steps

- [Streaming Tutorial](streaming-responses.md)
- [Agent Tasks Tutorial](agent-tasks.md)
- [API Reference](../api/rest-api.md)
- [Examples](../../examples/)
