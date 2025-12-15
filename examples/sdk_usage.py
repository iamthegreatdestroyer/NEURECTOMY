"""
Neurectomy SDK Usage Examples
==============================

Examples of using the Neurectomy Python SDK.
"""

# =============================================================================
# BASIC USAGE
# =============================================================================

from neurectomy.sdk import NeurectomyClient

# Create client
client = NeurectomyClient()

# Simple text generation
response = client.generate("What is artificial intelligence?")
print(response["text"])
print(f"Tokens: {response['total_tokens']}")
print(f"Latency: {response['latency_ms']:.1f}ms")


# =============================================================================
# STREAMING GENERATION
# =============================================================================

# Stream tokens as they're generated
for chunk in client.stream("Tell me a story about AI"):
    print(chunk, end="", flush=True)
print()


# =============================================================================
# AGENT TASKS
# =============================================================================

# List available agents
agents = client.list_agents()
print(f"Total agents: {agents['total']}")
print(f"Teams: {list(agents['teams'].keys())}")

# Get specific agent details
agent = client.get_agent("apex_01")
print(f"Agent: {agent['name']}")
print(f"Capabilities: {agent['capabilities']}")

# Execute agent task
result = client.execute_task(
    task_type="summarize",
    payload={
        "text": "Long document to summarize...",
        "max_length": 100,
    },
    team="analysis",
)
print(f"Summary: {result['result']}")


# =============================================================================
# CONVERSATIONS
# =============================================================================

# Chat with conversation context
conversation_id = "conv_001"

response = client.chat(
    conversation_id=conversation_id,
    messages=[
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi! How can I help?"},
        {"role": "user", "content": "What's the weather?"},
    ],
    max_tokens=100,
)

print(f"Assistant: {response['message']}")


# =============================================================================
# HEALTH & MONITORING
# =============================================================================

# Check system health
health = client.health()
print(f"Status: {health['status']}")
print(f"Uptime: {health['uptime_seconds']:.0f}s")

# Get metrics
metrics = client.metrics()
print(f"Inference latency: {metrics['inference']['latency_ms']}")
print(f"Compression ratio: {metrics['compression']['ratio']}")

# Simple health check
if client.is_healthy():
    print("✓ System is healthy")
else:
    print("✗ System is unhealthy")


# =============================================================================
# CONTEXT MANAGER
# =============================================================================

# Auto-close client
with NeurectomyClient() as client:
    response = client.generate("Test prompt")
    print(response["text"])
# Client automatically closed


# =============================================================================
# ASYNC USAGE
# =============================================================================

import asyncio
from neurectomy.sdk import AsyncNeurectomyClient


async def async_example():
    async with AsyncNeurectomyClient() as client:
        # Async generation
        response = await client.generate("Async test")
        print(response["text"])
        
        # Async health check
        health = await client.health()
        print(f"Status: {health['status']}")


# Run async example
# asyncio.run(async_example())


# =============================================================================
# ADVANCED CONFIGURATION
# =============================================================================

from neurectomy.sdk import NeurectomyConfig

# Custom configuration
config = NeurectomyConfig(
    base_url="http://api.example.com",
    api_key="your-api-key",
    timeout=120.0,
    max_retries=5,
)

client = NeurectomyClient(config)


# =============================================================================
# ERROR HANDLING
# =============================================================================

try:
    response = client.generate("Test prompt")
except Exception as e:
    print(f"Error: {e}")


# =============================================================================
# CUSTOM PARAMETERS
# =============================================================================

# Generation with custom parameters
response = client.generate(
    prompt="Explain quantum computing",
    max_tokens=500,
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    use_compression=True,
    compression_level=2,
)

print(response["text"])
print(f"Compression ratio: {response['compression_ratio']:.1f}x")
