# Neurectomy Go SDK

Production-ready Go SDK for the Neurectomy API. Provides type-safe access to all Neurectomy services with idiomatic Go patterns.

## Features

- **Type Safe**: Full type definitions for all API endpoints
- **Idiomatic Go**: Follows Go best practices and conventions
- **Error Handling**: Proper Go error handling patterns
- **HTTP Client**: Built-in timeout and error handling
- **JSON Support**: Full JSON serialization/deserialization
- **Lightweight**: Minimal dependencies

## Installation

```bash
go get github.com/iamthegreatdestroyer/neurectomy-go-sdk
```

## Quick Start

```go
package main

import (
	"fmt"
	"log"

	"github.com/iamthegreatdestroyer/neurectomy-go-sdk"
)

func main() {
	// Create client
	client, err := neurectomy.NewClient("your-api-key")
	if err != nil {
		log.Fatal(err)
	}

	// Generate text completion
	resp, err := client.Complete("Explain quantum computing", 200, 0.7)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(resp.Text)

	// Compress text
	compressed, err := client.Compress("Large text...", 0.1, 5)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Compressed %.2fx\n", compressed.CompressionRatio)

	// Store file
	stored, err := client.StoreFile("docs/file.txt", "base64-data")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Stored with ID: %s\n", stored.ObjectID)

	// Retrieve file
	retrieved, err := client.RetrieveFile(stored.ObjectID)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(retrieved.Data)

	// Check status
	status, err := client.GetStatus()
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("API Status: %s (v%s)\n", status.Status, status.Version)
}
```

## Configuration

```go
opts := &neurectomy.ClientOptions{
	BaseURL: "https://custom.api.com",
	Timeout: 60 * time.Second,
}

client, err := neurectomy.NewClientWithOptions("api-key", opts)
```

## API Methods

### Text Completion

```go
resp, err := client.Complete(
	prompt,     // string
	maxTokens,  // int (0 = 100)
	temperature // float32 (0 = 0.7)
)

// resp.Text - Generated text
// resp.TokensGenerated - Number of tokens
// resp.FinishReason - Reason completion stopped
// resp.Usage - Token usage (if provided)
```

### Text Compression

```go
resp, err := client.Compress(
	text,                // string
	targetRatio,         // float32 (0 = 0.1)
	compressionLevel,    // int (0 = 2)
)

// resp.CompressedData - Compressed data (base64)
// resp.CompressionRatio - Achieved ratio
// resp.OriginalSize - Original size in bytes
// resp.CompressedSize - Compressed size in bytes
// resp.Algorithm - Algorithm used
```

### File Storage

```go
// Store file
stored, err := client.StoreFile(path, data)

// Retrieve file
retrieved, err := client.RetrieveFile(stored.ObjectID)

// Delete file
err := client.DeleteFile(stored.ObjectID)
```

### API Status

```go
status, err := client.GetStatus()
```

## Error Handling

```go
import "errors"

resp, err := client.Complete("test", 0, 0)
if err != nil {
	if errors.Is(err, neurectomy.ErrMissingAPIKey) {
		log.Fatal("API key required")
	}
	log.Printf("API error: %v", err)
}
```

## Types

- `Client` - Main API client
- `CompletionRequest/Response` - Text completion
- `CompressionRequest/Response` - Text compression
- `StorageRequest/Response` - File storage
- `RetrievedFile` - Retrieved file data
- `StatusResponse` - API status
- `TokenUsage` - Token usage information

## Development

```bash
# Run tests
go test ./...

# Run tests with coverage
go test -cover ./...

# Format code
go fmt ./...

# Run linter
go vet ./...
```

## License

MIT

## Support

For issues and support: https://github.com/iamthegreatdestroyer/NEURECTOMY
