# @neurectomy/api-client API Reference

HTTP and GraphQL client for communicating with NEURECTOMY services.

## Installation

```bash
pnpm add @neurectomy/api-client
```

## REST Client

### Creating a Client

```typescript
import { createRestClient, RestClient } from "@neurectomy/api-client/rest";

// Using factory function
const client = createRestClient({
  baseUrl: "https://api.neurectomy.io",
  headers: {
    "X-API-Key": "your-api-key",
  },
  timeout: 30000,
});

// Using class directly
const client = new RestClient({
  baseUrl: "https://api.neurectomy.io",
});
```

### Configuration Options

| Option    | Type                     | Required | Default | Description                      |
| --------- | ------------------------ | -------- | ------- | -------------------------------- |
| `baseUrl` | `string`                 | Yes      | -       | Base URL for API requests        |
| `headers` | `Record<string, string>` | No       | -       | Default headers for all requests |
| `timeout` | `number`                 | No       | `30000` | Request timeout in milliseconds  |

### Making Requests

#### GET Request

```typescript
// Simple GET
const users = await client.get<User[]>("/users");

// GET with query parameters
const user = await client.get<User>("/users", {
  page: 1,
  limit: 10,
  sort: "createdAt",
});
// Results in: GET /users?page=1&limit=10&sort=createdAt
```

#### POST Request

```typescript
const newUser = await client.post<User>("/users", {
  name: "John Doe",
  email: "john@example.com",
});
```

#### PUT Request

```typescript
const updated = await client.put<User>("/users/123", {
  name: "Jane Doe",
});
```

#### PATCH Request

```typescript
const patched = await client.patch<User>("/users/123", {
  status: "active",
});
```

#### DELETE Request

```typescript
await client.delete("/users/123");
```

### Authentication

```typescript
// Set auth token
client.setAuthToken("eyJhbGciOiJIUzI1NiIs...");
// Adds: Authorization: Bearer eyJhbGciOiJIUzI1NiIs...

// Clear auth token
client.clearAuthToken();
```

## Error Handling

### RestError

```typescript
import { RestError } from "@neurectomy/api-client/rest";

try {
  await client.get("/protected");
} catch (error) {
  if (error instanceof RestError) {
    console.log(error.statusCode); // 401
    console.log(error.message); // 'HTTP 401: Unauthorized'
    console.log(error.responseBody); // '{"error":"Invalid token"}'
  }
}
```

### handleApiError

Normalize any error into a consistent format:

```typescript
import { handleApiError } from "@neurectomy/api-client";

try {
  await client.get("/endpoint");
} catch (error) {
  const apiError = handleApiError(error);
  console.log(apiError);
  // {
  //   code: 'HTTP_404',
  //   message: 'Not Found',
  //   statusCode: 404,
  //   details: { error: 'Resource not found' }
  // }
}
```

## Retry Logic

### retryRequest

Retry failed requests with exponential backoff:

```typescript
import { retryRequest, isRetryableError } from "@neurectomy/api-client";

const result = await retryRequest(() => client.get("/flaky-endpoint"), {
  maxRetries: 3,
  baseDelayMs: 1000,
  maxDelayMs: 10000,
  shouldRetry: isRetryableError,
});
```

### Options

| Option        | Type                 | Default | Description                |
| ------------- | -------------------- | ------- | -------------------------- |
| `maxRetries`  | `number`             | `3`     | Maximum retry attempts     |
| `baseDelayMs` | `number`             | `1000`  | Base delay between retries |
| `maxDelayMs`  | `number`             | `10000` | Maximum delay cap          |
| `shouldRetry` | `(error) => boolean` | -       | Custom retry predicate     |

### isRetryableError

Built-in predicate for identifying retryable errors:

```typescript
import { isRetryableError } from "@neurectomy/api-client";

// Returns true for:
// - HTTP 5xx errors (server errors)
// - HTTP 429 (rate limited)
// - Network/fetch errors

isRetryableError(new RestError("Error", 503, "")); // true
isRetryableError(new RestError("Error", 429, "")); // true
isRetryableError(new RestError("Error", 404, "")); // false
```

## GraphQL Client

### Setup

```typescript
import { GraphQLClient } from "@neurectomy/api-client/graphql";

const client = new GraphQLClient({
  endpoint: "https://api.neurectomy.io/graphql",
  headers: {
    Authorization: "Bearer token",
  },
});
```

### Queries

```typescript
const query = `
  query GetAgent($id: ID!) {
    agent(id: $id) {
      id
      name
      status
    }
  }
`;

const data = await client.request(query, { id: "agent-123" });
```

### Mutations

```typescript
const mutation = `
  mutation CreateAgent($input: AgentInput!) {
    createAgent(input: $input) {
      id
      name
    }
  }
`;

const data = await client.request(mutation, {
  input: {
    name: "New Agent",
    type: "ai",
  },
});
```

### Subscriptions (WebSocket)

```typescript
import { createWebSocketClient } from "@neurectomy/api-client/graphql";

const wsClient = createWebSocketClient({
  url: "wss://api.neurectomy.io/graphql",
  connectionParams: {
    authToken: "token",
  },
});

const subscription = `
  subscription OnAgentStatus($id: ID!) {
    agentStatus(id: $id) {
      status
      lastActivity
    }
  }
`;

const unsubscribe = wsClient.subscribe(
  { query: subscription, variables: { id: "agent-123" } },
  {
    next: (data) => console.log("Update:", data),
    error: (error) => console.error("Error:", error),
    complete: () => console.log("Complete"),
  }
);

// Cleanup
unsubscribe();
wsClient.dispose();
```

## React Query Integration

### Setup Provider

```typescript
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5000,
      retry: 3,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <YourApp />
    </QueryClientProvider>
  );
}
```

### Using Generated Hooks

```typescript
import { useAgentQuery, useCreateAgentMutation } from '@neurectomy/api-client';

function AgentView({ id }: { id: string }) {
  const { data, isLoading, error } = useAgentQuery({ id });

  if (isLoading) return <Loading />;
  if (error) return <Error error={error} />;

  return <AgentDetails agent={data.agent} />;
}

function CreateAgentForm() {
  const mutation = useCreateAgentMutation();

  const handleSubmit = (data: AgentInput) => {
    mutation.mutate({ input: data });
  };

  return (
    <form onSubmit={handleSubmit}>
      {mutation.isLoading && <Spinner />}
      {mutation.isError && <Error error={mutation.error} />}
      {/* form fields */}
    </form>
  );
}
```

## TypeScript Types

```typescript
import type { RestClientConfig, ApiError } from "@neurectomy/api-client";
```
