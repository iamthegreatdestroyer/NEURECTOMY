# @neurectomy/test-config

Shared test configuration, utilities, and mocks for the NEURECTOMY monorepo.

## Installation

```bash
pnpm add -D @neurectomy/test-config vitest
```

## Usage

### Vitest Configuration

Create a `vitest.config.ts` in your package:

```typescript
// For Node.js packages
import { defineConfig, mergeConfig } from "vitest/config";
import { baseConfig } from "@neurectomy/test-config/vitest";

export default mergeConfig(
  baseConfig,
  defineConfig({
    test: {
      // Package-specific overrides
    },
  })
);
```

```typescript
// For React packages
import { defineConfig, mergeConfig } from "vitest/config";
import { reactConfig } from "@neurectomy/test-config/vitest";

export default mergeConfig(
  reactConfig,
  defineConfig({
    test: {
      // Package-specific overrides
    },
  })
);
```

### Available Configurations

| Config              | Environment | Use Case                              |
| ------------------- | ----------- | ------------------------------------- |
| `baseConfig`        | Node.js     | Core packages, utilities, API clients |
| `reactConfig`       | jsdom       | React components, UI packages         |
| `integrationConfig` | Node.js     | Cross-package integration tests       |
| `benchmarkConfig`   | Node.js     | Performance benchmarks                |

## Mocks

### API Mocks (MSW)

```typescript
import {
  startMockServer,
  stopMockServer,
  mockRestEndpoint,
  mockGraphQLQuery,
} from "@neurectomy/test-config/mocks";
import { beforeAll, afterAll, afterEach, describe, it, expect } from "vitest";

describe("API Tests", () => {
  beforeAll(() => startMockServer());
  afterAll(() => stopMockServer());
  afterEach(() => resetMockHandlers());

  it("fetches agents", async () => {
    mockRestEndpoint("get", "/api/agents", {
      agents: [{ id: "1", name: "Test Agent" }],
    });

    const response = await fetch("/api/agents");
    const data = await response.json();

    expect(data.agents).toHaveLength(1);
  });

  it("handles GraphQL queries", async () => {
    mockGraphQLQuery("GetAgents", {
      data: { agents: { nodes: [] } },
    });

    // Your GraphQL client code here
  });
});
```

### Storage Mocks

```typescript
import {
  createMockStore,
  createMockCache,
  createMockIndexedDB,
} from "@neurectomy/test-config/mocks";

// Simple key-value store
const store = createMockStore<string>();
store.set("key", "value");
expect(store.get("key")).toBe("value");

// Cache with TTL
const cache = createMockCache();
await cache.set("key", "value", 60); // 60 second TTL
expect(await cache.get("key")).toBe("value");

// IndexedDB
const idb = createMockIndexedDB();
const request = idb.open("test-db", 1);
request.onupgradeneeded = (event) => {
  const db = event.target.result;
  db.createObjectStore("items", { keyPath: "id" });
};
```

### Event Mocks

```typescript
import {
  createMockEventEmitter,
  createMockMessageQueue,
  createMockJetStream,
} from "@neurectomy/test-config/mocks";

// Type-safe event emitter
interface MyEvents {
  userCreated: { id: string; name: string };
  userDeleted: { id: string };
}

const emitter = createMockEventEmitter<MyEvents>();

emitter.on("userCreated", (data) => {
  console.log(data.name); // TypeScript knows this is string
});

await emitter.emit("userCreated", { id: "1", name: "Test" });

// Message queue
const queue = createMockMessageQueue<{ type: string; payload: unknown }>();
queue.subscribe((msg) => console.log(msg.type));
await queue.publish({ type: "task", payload: { id: 1 } });

// NATS JetStream
const js = createMockJetStream();
js.addStream("AGENTS", ["agents.>"]);
await js.publish("agents.created", { id: "123" });
```

## Test Utilities

### Test Factories

```typescript
import {
  createTestAgent,
  createTestWorkflow,
  createTestUser,
  testUuid,
  relativeDate,
} from "@neurectomy/test-config/utils";

// Create test data with defaults
const agent = createTestAgent({ name: "My Agent" });
const workflow = createTestWorkflow({ status: "active" });
const user = createTestUser({ role: "admin" });

// Generate unique IDs
const id = testUuid(); // 'a1b2c3d4-...'

// Create relative dates
const yesterday = relativeDate({ days: -1 });
const inOneHour = relativeDate({ hours: 1 });
```

### Async Helpers

```typescript
import {
  wait,
  retry,
  pollUntil,
  withTimeout,
  createDeferred,
} from "@neurectomy/test-config/utils";

// Wait for duration
await wait(100);

// Retry on failure
const result = await retry(() => unreliableOperation(), {
  maxAttempts: 3,
  delay: 100,
  backoff: "exponential",
});

// Poll until condition
const data = await pollUntil(
  () => fetchStatus(),
  (status) => status === "ready",
  { interval: 100, timeout: 5000 }
);

// Add timeout to promise
const result = await withTimeout(
  longRunningOperation(),
  5000,
  "Operation timed out"
);

// Deferred promise for controlling flow
const deferred = createDeferred<string>();
someAsyncCode(deferred.resolve);
await deferred.promise;
```

### Custom Assertions

```typescript
import {
  assertDefined,
  assertResolves,
  assertRejects,
  assertEventually,
  assertSameElements,
} from "@neurectomy/test-config/utils";

// Type-safe assertions
const maybeValue: string | undefined = getValue();
assertDefined(maybeValue);
console.log(maybeValue.length); // TypeScript knows it's defined

// Promise assertions
await assertResolves(fetchData(), { id: 1 });
await assertRejects(failingOperation(), "Expected error message");

// Eventual consistency
await assertEventually(() => document.querySelector(".loaded") !== null, {
  timeout: 5000,
  message: "Element never appeared",
});

// Array comparison (order-independent)
assertSameElements([1, 2, 3], [3, 2, 1]); // Passes
```

## Setup Files

### Node.js Tests

Tests automatically use `@neurectomy/test-config/setup` which:

- Sets `NODE_ENV=test`
- Mocks console methods to suppress noise
- Clears mocks between tests
- Provides global test utilities

### DOM Tests

React tests automatically use `@neurectomy/test-config/setup-dom` which:

- Includes all Node.js setup
- Adds `@testing-library/jest-dom` matchers
- Mocks browser APIs (ResizeObserver, IntersectionObserver, etc.)
- Mocks localStorage/sessionStorage
- Cleans up React Testing Library between tests

## Coverage Thresholds

Default thresholds (can be overridden per-package):

| Metric     | Threshold |
| ---------- | --------- |
| Branches   | 70%       |
| Functions  | 70%       |
| Lines      | 70%       |
| Statements | 70%       |

Integration tests have lower thresholds (50%) by default.

## Best Practices

1. **Use typed event emitters** for better IDE support
2. **Prefer MSW over fetch mocks** for realistic API testing
3. **Use factories** instead of inline test data
4. **Reset mocks** in `afterEach` to prevent test pollution
5. **Use `pollUntil`** instead of arbitrary `wait()` calls
6. **Add timeouts** to async operations to catch hanging tests

## API Reference

See individual mock/utility files for detailed JSDoc documentation.
