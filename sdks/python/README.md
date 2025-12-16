# Neurectomy Python SDK

Production-ready Python SDK for the Neurectomy API.

## Features

- **Type Hints**: Full type hints for better IDE support
- **Dataclasses**: Clean, type-safe response objects
- **Retry Logic**: Automatic retry with exponential backoff
- **Session Management**: Persistent HTTP sessions with connection pooling
- **Error Handling**: Comprehensive error types
- **Timeout Support**: Configurable request timeouts

## Installation

### From PyPI

```bash
pip install neurectomy-sdk
```

### From source

```bash
git clone https://github.com/iamthegreatdestroyer/NEURECTOMY
cd sdks/python
pip install -e .
```

## Quick Start

```python
from neurectomy import NeurectomyClient

# Create client
client = NeurectomyClient(api_key="your-api-key")

# Generate text completion
response = client.complete(
    prompt="Explain quantum computing",
    max_tokens=200,
    temperature=0.7
)
print(response.text)

# Compress text
compressed = client.compress(
    text="Long text...",
    target_ratio=0.1
)
print(f"Compressed {compressed.compression_ratio}x")

# Store and retrieve files
stored = client.store_file("docs/file.txt", "base64-data")
retrieved = client.retrieve_file(stored.object_id)
```

## Configuration

```python
client = NeurectomyClient(
    api_key="your-api-key",
    base_url="https://api.neurectomy.ai",  # Optional
    timeout=30,                              # Optional
    retry_on_failure=True,                   # Optional
    max_retries=3                            # Optional
)
```

## API Methods

### Text Completion

```python
response = client.complete(
    prompt="Your prompt",
    max_tokens=100,
    temperature=0.7,
    model="ryot-bitnet-7b",
    top_p=1.0,
    frequency_penalty=0,
    presence_penalty=0
)

# response.text - Generated text
# response.tokens_generated - Number of tokens generated
# response.finish_reason - Reason completion stopped
# response.usage - Token usage details (if provided)
```

### Text Compression

```python
response = client.compress(
    text="Text to compress",
    target_ratio=0.1,
    compression_level=5,
    algorithm="lz4"
)

# response.compressed_data - Compressed data (base64)
# response.compression_ratio - Achieved ratio
# response.original_size - Original size in bytes
# response.compressed_size - Compressed size in bytes
# response.algorithm - Algorithm used
```

### File Storage

```python
# Store file
stored = client.store_file(
    path="path/to/file.txt",
    data="base64-encoded-data",
    metadata={"description": "My file"}
)

# Retrieve file
retrieved = client.retrieve_file(stored.object_id)

# Delete file
client.delete_file(stored.object_id)
```

### API Status

```python
status = client.get_status()
print(f"API Status: {status.status} (v{status.version})")
```

## Error Handling

```python
from neurectomy import NeurectomyClient, APIError, NeurectomyError

client = NeurectomyClient(api_key="key")

try:
    response = client.complete("test")
except APIError as e:
    print(f"API error {e.code}: {e.message}")
except NeurectomyError as e:
    print(f"SDK error: {e}")
```

## Error Types

- `ConfigError` - Configuration error (e.g., missing API key)
- `APIError` - API returned error response
- `NeurectomyError` - General SDK error (network, timeout, etc.)

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black neurectomy tests

# Type checking
mypy neurectomy

# Lint
flake8 neurectomy
```

## License

MIT

## Support

For issues and support: https://github.com/iamthegreatdestroyer/NEURECTOMY
