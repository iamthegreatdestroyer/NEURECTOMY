# Neurectomy Rust SDK

A production-ready, type-safe Rust SDK for the Neurectomy API. Built with async/await support using Tokio.

## Features

- **Type-Safe**: Full type definitions for all API endpoints
- **Async/Await**: Modern async Rust with Tokio
- **Error Handling**: Comprehensive error types with `thiserror`
- **Configuration**: Flexible configuration with builder pattern
- **Timeout Support**: Configurable request timeouts
- **Serialization**: Full serde support for JSON

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
neurectomy-sdk = "1.0.0"
tokio = { version = "1", features = ["full"] }
```

## Quick Start

```rust
use neurectomy_sdk::NeurectomyClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create client
    let client = NeurectomyClient::new("your-api-key".to_string())?;

    // Generate text completion
    let response = client.complete(
        "Explain quantum computing".to_string(),
        Some(200),
        Some(0.7),
    ).await?;

    println!("{}", response.text);

    // Compress text
    let compressed = client.compress(
        "Large text...".to_string(),
        Some(0.1),
        Some(5),
    ).await?;

    println!("Compressed {}x", compressed.compression_ratio);

    // Store file
    let stored = client.store_file(
        "path/to/file.txt".to_string(),
        "base64-encoded-data".to_string(),
    ).await?;

    println!("Stored with ID: {}", stored.object_id);

    Ok(())
}
```

## Configuration

```rust
use neurectomy_sdk::{NeurectomyClient, NeurectomyConfig};

let config = NeurectomyConfig::new("api-key".to_string())
    .with_base_url("https://custom.api.com".to_string())
    .with_timeout(60);

let client = NeurectomyClient::with_config(config)?;
```

## API Methods

### Text Completion

```rust
let response = client.complete(
    prompt,
    max_tokens,    // Option<u32>
    temperature,   // Option<f32>
).await?;
```

### Text Compression

```rust
let response = client.compress(
    text,
    target_ratio,       // Option<f32>
    compression_level,  // Option<u8>
).await?;
```

### File Storage

```rust
// Store file
let response = client.store_file(path, data).await?;

// Retrieve file
let file = client.retrieve_file(object_id).await?;

// Delete file
client.delete_file(object_id).await?;
```

### API Status

```rust
let status = client.get_status().await?;
```

## Error Handling

```rust
use neurectomy_sdk::NeurectomyError;

match client.complete("test".to_string(), None, None).await {
    Ok(response) => println!("{}", response.text),
    Err(NeurectomyError::ApiError { code, message }) => {
        eprintln!("API error {}: {}", code, message);
    }
    Err(NeurectomyError::HttpError(e)) => eprintln!("HTTP error: {}", e),
    Err(e) => eprintln!("Error: {}", e),
}
```

## Building

```bash
# Build the library
cargo build --release

# Run tests
cargo test

# Run examples
cargo run --example basic_usage

# Generate documentation
cargo doc --open
```

## Type Definitions

The SDK provides comprehensive type definitions:

- `NeurectomyClient` - Main API client
- `NeurectomyConfig` - Configuration builder
- `CompletionRequest/Response` - Text completion
- `CompressionRequest/Response` - Text compression
- `StorageResponse` - File storage response
- `RetrievedFile` - Retrieved file data
- `NeurectomyError` - All error types

## License

MIT

## Support

For issues and support: https://github.com/iamthegreatdestroyer/NEURECTOMY
