# @neurectomy/core API Reference

Core utilities, schemas, and error types for the NEURECTOMY platform.

## Installation

```bash
pnpm add @neurectomy/core
```

## Error Classes

### NeurectomyError

Base error class for all NEURECTOMY errors.

```typescript
import { NeurectomyError } from "@neurectomy/core";

// Create an error with context
const error = new NeurectomyError("Operation failed", "OPERATION_FAILED", {
  userId: "123",
  attemptedAction: "delete",
});

// Access error properties
console.log(error.message); // 'Operation failed'
console.log(error.code); // 'OPERATION_FAILED'
console.log(error.context); // { userId: '123', attemptedAction: 'delete' }

// Serialize to JSON
const json = error.toJSON();
// {
//   name: 'NeurectomyError',
//   message: 'Operation failed',
//   code: 'OPERATION_FAILED',
//   context: { ... },
//   timestamp: '2024-...',
//   stack: '...'
// }
```

### ValidationError

Error for schema validation failures.

```typescript
import { ValidationError } from "@neurectomy/core";

const error = new ValidationError("Validation failed", [
  { field: "email", message: "Invalid email format" },
  { field: "password", message: "Must be at least 8 characters" },
]);

// Access validation errors
console.log(error.errors);
// [
//   { field: 'email', message: 'Invalid email format' },
//   { field: 'password', message: 'Must be at least 8 characters' }
// ]
```

### NetworkError

Error for network/HTTP failures.

```typescript
import { NetworkError } from "@neurectomy/core";

const error = new NetworkError(
  "Request failed",
  503,
  "https://api.example.com/endpoint"
);

console.log(error.statusCode); // 503
console.log(error.url); // 'https://api.example.com/endpoint'
```

### ConfigurationError

Error for configuration/setup issues.

```typescript
import { ConfigurationError } from "@neurectomy/core";

const error = new ConfigurationError("Invalid configuration", "database.host");

console.log(error.configKey); // 'database.host'
```

## Schemas

Zod schemas for validating agent configurations, tasks, and workflows.

### agentConfigSchema

```typescript
import { agentConfigSchema } from "@neurectomy/core";

const config = {
  name: "MyAgent",
  type: "ai",
  version: "1.0.0",
  description: "A helpful AI agent",
  capabilities: ["chat", "analysis"],
  enabled: true,
  maxRetries: 3,
};

// Validate
const result = agentConfigSchema.safeParse(config);
if (result.success) {
  console.log("Valid config:", result.data);
} else {
  console.log("Errors:", result.error.issues);
}

// Parse (throws on invalid)
const validated = agentConfigSchema.parse(config);
```

**Schema Fields:**

| Field          | Type                                          | Required | Default | Description                    |
| -------------- | --------------------------------------------- | -------- | ------- | ------------------------------ |
| `id`           | `string`                                      | No       | -       | Unique identifier              |
| `name`         | `string`                                      | Yes      | -       | Agent name (1-256 chars)       |
| `type`         | `'ai' \| 'tool' \| 'composite' \| 'workflow'` | Yes      | -       | Agent type                     |
| `version`      | `string`                                      | No       | -       | Semver version (e.g., "1.0.0") |
| `description`  | `string`                                      | No       | -       | Description (max 4096 chars)   |
| `capabilities` | `string[]`                                    | No       | -       | List of capabilities           |
| `parameters`   | `Record<string, unknown>`                     | No       | -       | Custom parameters              |
| `metadata`     | `Record<string, string>`                      | No       | -       | Metadata tags                  |
| `enabled`      | `boolean`                                     | No       | `true`  | Whether agent is enabled       |
| `timeout`      | `number`                                      | No       | -       | Timeout in ms                  |
| `maxRetries`   | `number`                                      | No       | `3`     | Max retry attempts             |
| `rateLimit`    | `object`                                      | No       | -       | Rate limiting config           |

### taskDefinitionSchema

```typescript
import { taskDefinitionSchema } from '@neurectomy/core';

const task = {
  name: 'ProcessData',
  agentId: 'agent-123',
  input: { data: [...] },
  priority: 75,
  timeout: 30000,
};

const validated = taskDefinitionSchema.parse(task);
```

### workflowSchema

```typescript
import { workflowSchema } from "@neurectomy/core";

const workflow = {
  name: "DataPipeline",
  version: "1.0.0",
  tasks: [
    { name: "Fetch", agentId: "fetcher-agent" },
    { name: "Process", agentId: "processor-agent", dependencies: ["Fetch"] },
  ],
  triggers: [{ type: "schedule", config: { cron: "0 * * * *" } }],
};

const validated = workflowSchema.parse(workflow);
```

## Utility Functions

### formatters

```typescript
import { formatBytes, formatDuration, formatDate } from '@neurectomy/core';

// Format bytes
formatBytes(1024);           // '1.00 KB'
formatBytes(1536, 0);        // '2 KB'
formatBytes(1073741824);     // '1.00 GB'

// Format duration
formatDuration(5000);        // '5s'
formatDuration(65000);       // '1m 5s'
formatDuration(3665000);     // '1h 1m 5s'

// Format date
formatDate(new Date());                  // '12/5/2024, 3:30:00 PM'
formatDate(new Date(), 'en-US', {...});  // Custom options
```

### identifiers

```typescript
import { generateId, generateUUID, isValidId } from "@neurectomy/core";

// Generate prefixed ID
const agentId = generateId("agent"); // 'agent_abc123xyz...'
const taskId = generateId("task"); // 'task_def456uvw...'

// Generate UUID
const uuid = generateUUID(); // '550e8400-e29b-41d4-a716-446655440000'

// Validate ID format
isValidId("agent_abc123"); // true
isValidId("invalid"); // false
```

### objects

```typescript
import { deepMerge, pick, omit, isPlainObject } from "@neurectomy/core";

// Deep merge objects
const merged = deepMerge({ a: 1, b: { c: 2 } }, { b: { d: 3 }, e: 4 });
// { a: 1, b: { c: 2, d: 3 }, e: 4 }

// Pick properties
const picked = pick({ a: 1, b: 2, c: 3 }, ["a", "c"]);
// { a: 1, c: 3 }

// Omit properties
const omitted = omit({ a: 1, b: 2, c: 3 }, ["b"]);
// { a: 1, c: 3 }

// Check if plain object
isPlainObject({}); // true
isPlainObject([]); // false
isPlainObject(null); // false
```

### functions

```typescript
import { debounce, throttle, retry } from "@neurectomy/core";

// Debounce function calls
const debouncedSave = debounce(save, 300);
debouncedSave(data); // Only executes after 300ms of no calls

// Throttle function calls
const throttledScroll = throttle(onScroll, 100);
// Maximum one call per 100ms

// Retry with exponential backoff
const result = await retry(() => fetchData(), {
  maxRetries: 3,
  baseDelayMs: 1000,
  shouldRetry: (error) => error.status >= 500,
});
```

## Constants

```typescript
import {
  DEFAULT_TIMEOUT,
  MAX_RETRIES,
  RATE_LIMIT_WINDOW,
  API_VERSION,
} from "@neurectomy/core";
```

## TypeScript Types

All schemas export inferred types:

```typescript
import type {
  AgentConfigInput,
  AgentConfigOutput,
  TaskDefinitionInput,
  TaskDefinitionOutput,
  WorkflowInput,
  WorkflowOutput,
} from "@neurectomy/core";
```
