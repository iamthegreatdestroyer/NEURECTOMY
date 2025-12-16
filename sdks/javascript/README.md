# Neurectomy JavaScript/TypeScript SDK

Official TypeScript SDK for the Neurectomy API. Provides type-safe access to all Neurectomy services including text completion, compression, and file storage.

## Installation

```bash
npm install @neurectomy/sdk
# or
yarn add @neurectomy/sdk
```

## Quick Start

```typescript
import NeurectomyClient from '@neurectomy/sdk';

const client = new NeurectomyClient({
  apiKey: process.env.NEURECTOMY_API_KEY,
});

// Generate text completion
const completion = await client.complete({
  prompt: 'Explain quantum computing',
  maxTokens: 200,
  temperature: 0.7,
});
console.log(completion.text);

// Compress text
const compressed = await client.compress({
  text: 'Long document with lots of text...',
  targetRatio: 0.1,
  compressionLevel: 5,
});
console.log(`Compressed to ${compressed.compressionRatio}x`);

// Store file
const stored = await client.storeFile('documents/my-file.txt', 'base64-encoded-data', {
  description: 'Important document',
});
console.log(`Stored with ID: ${stored.objectId}`);

// Retrieve file
const retrieved = await client.retrieveFile(stored.objectId);
console.log(retrieved.data);

// Check API status
const status = await client.getStatus();
console.log(`API Status: ${status.status} (v${status.version})`);
```

## Configuration

```typescript
const client = new NeurectomyClient({
  apiKey: 'your-api-key', // Required
  baseURL: 'https://api.neurectomy.ai', // Optional, default shown
  timeout: 30000, // Optional, milliseconds
  retryOnFailure: true, // Optional, default: true
  maxRetries: 3, // Optional, default: 3
});
```

## API Methods

### Text Completion

```typescript
const response = await client.complete({
  prompt: 'Your prompt here',
  maxTokens: 100,
  temperature: 0.7,
  topP: 1.0,
  frequencyPenalty: 0,
  presencePenalty: 0,
  model: 'ryot-bitnet-7b',
});
```

### Text Compression

```typescript
const response = await client.compress({
  text: 'Text to compress',
  targetRatio: 0.1, // 10% of original size
  compressionLevel: 2, // 1-9
  algorithm: 'lz4', // Compression algorithm
});
```

### File Storage

```typescript
// Store file
const stored = await client.storeFile(
  'path/to/file.txt',
  'base64-encoded-data',
  { metadata: 'value' } // Optional
);

// Retrieve file
const retrieved = await client.retrieveFile(stored.objectId);

// Delete file
const deleted = await client.deleteFile(stored.objectId);
```

### API Status

```typescript
const status = await client.getStatus();
// Returns: { status: string, version: string }
```

## Error Handling

```typescript
try {
  const completion = await client.complete({
    prompt: 'test',
  });
} catch (error) {
  if (error instanceof Error) {
    console.error('Error:', error.message);
    // Error has additional properties:
    // - code: error code
    // - details: additional details
    // - status: HTTP status code
  }
}
```

## Retry Logic

The SDK automatically retries failed requests with exponential backoff:

- Retries on network errors
- Retries on rate limit (429)
- Retries on server errors (5xx)
- Exponential backoff: 2s, 4s, 8s
- Default max retries: 3

Configure retries:

```typescript
const client = new NeurectomyClient({
  apiKey: 'your-key',
  retryOnFailure: true,
  maxRetries: 5,
});
```

## Type Definitions

Full TypeScript support with complete type definitions:

```typescript
import {
  NeurectomyClient,
  NeurectomyConfig,
  CompletionRequest,
  CompletionResponse,
  CompressionRequest,
  CompressionResponse,
  StorageRequest,
  StorageResponse,
  RetrievedFile,
  ErrorResponse,
} from '@neurectomy/sdk';
```

## Development

```bash
# Install dependencies
npm install

# Build TypeScript
npm run build

# Run tests
npm test

# Watch mode
npm run dev

# Lint code
npm run lint

# Format code
npm run format
```

## Build Output

The build process generates:

- `dist/index.js` - Compiled JavaScript
- `dist/index.d.ts` - Type definitions
- `dist/index.js.map` - Source maps

## License

MIT

## Support

For issues and support, visit: https://github.com/iamthegreatdestroyer/NEURECTOMY
