# @neurectomy/test-config API Reference

Shared test configuration, utilities, and mocks for the NEURECTOMY monorepo.

## Installation

```bash
pnpm add -D @neurectomy/test-config vitest
```

## Vitest Configurations

### baseConfig

Base configuration for Node.js packages (non-React).

```typescript
import { defineConfig, mergeConfig } from "vitest/config";
import { baseConfig } from "@neurectomy/test-config/vitest";

export default mergeConfig(
  baseConfig,
  defineConfig({
    test: {
      // package-specific overrides
    },
  })
);
```

**Features:**

- Environment: `node`
- Include patterns: `src/**/*.test.ts`, `src/**/*.spec.ts`
- Coverage: v8 provider with 70% thresholds
- Test timeout: 10 seconds
- Pool: `forks` for isolation

---

### reactConfig

Configuration for React/DOM component testing.

```typescript
import { defineConfig, mergeConfig } from "vitest/config";
import { reactConfig } from "@neurectomy/test-config/vitest";

export default mergeConfig(
  reactConfig,
  defineConfig({
    test: {
      // component-specific overrides
    },
  })
);
```

**Features:**

- Environment: `jsdom`
- Include patterns: `*.test.tsx`, `*.spec.tsx`
- Setup files: `@neurectomy/test-config/setup-dom`
- CSS modules support
- Excludes: `*.stories.tsx`

---

### integrationConfig

Configuration for cross-package integration tests.

```typescript
import { defineConfig, mergeConfig } from "vitest/config";
import { integrationConfig } from "@neurectomy/test-config/vitest";

export default mergeConfig(
  integrationConfig,
  defineConfig({
    test: {
      // integration test overrides
    },
  })
);
```

**Features:**

- Include: `tests/integration/**/*.test.ts`, `tests/e2e/**/*.test.ts`
- Timeout: 30 seconds
- Max concurrency: 1 (sequential)
- Shuffle: disabled
- Retry: 1
- Coverage threshold: 50%

---

### benchmarkConfig

Configuration for performance benchmarks.

```typescript
import { defineConfig, mergeConfig } from "vitest/config";
import { benchmarkConfig } from "@neurectomy/test-config/vitest";

export default mergeConfig(
  benchmarkConfig,
  defineConfig({
    test: {
      // benchmark overrides
    },
  })
);
```

**Features:**

- Include: `src/**/*.bench.ts`, `tests/bench/**/*.ts`
- Output: `./benchmarks/results.json`
- Reporters: default + JSON

---

## Setup Files

### setup-dom.ts

DOM setup for React testing, auto-imported by `reactConfig`.

```typescript
// Imported automatically by reactConfig
// Provides:
// - @testing-library/jest-dom matchers
// - global cleanup
// - window/document mocks
```

### setup.ts

Base setup for all test environments.

```typescript
// Imported by baseConfig
// Provides:
// - Extended expect matchers
// - Console suppression (optional)
// - Global test utilities
```

---

## Mocks

### MSW (Mock Service Worker)

#### startMockServer()

Start the MSW server for API mocking.

```typescript
import { startMockServer, stopMockServer } from "@neurectomy/test-config/mocks";
import { beforeAll, afterAll } from "vitest";

beforeAll(() => startMockServer());
afterAll(() => stopMockServer());
```

#### mockRestEndpoint(method, path, response, options?)

Mock a REST API endpoint.

```typescript
import { mockRestEndpoint } from "@neurectomy/test-config/mocks";

// Simple mock
mockRestEndpoint("get", "/api/agents", {
  agents: [{ id: "1", name: "Test" }],
});

// With status code
mockRestEndpoint("post", "/api/agents", { id: "2" }, { status: 201 });

// With headers
mockRestEndpoint(
  "get",
  "/api/user",
  { user: {} },
  {
    headers: { "X-Custom": "value" },
  }
);

// Error response
mockRestEndpoint("get", "/api/error", { error: "Not found" }, { status: 404 });
```

**Parameters:**

| Param             | Type                                              | Description                     |
| ----------------- | ------------------------------------------------- | ------------------------------- |
| `method`          | `'get' \| 'post' \| 'put' \| 'patch' \| 'delete'` | HTTP method                     |
| `path`            | `string`                                          | API path (relative or absolute) |
| `response`        | `unknown`                                         | Response body                   |
| `options.status`  | `number`                                          | HTTP status code (default: 200) |
| `options.headers` | `Record<string, string>`                          | Response headers                |

#### mockGraphQLQuery(operationName, response)

Mock a GraphQL query.

```typescript
import { mockGraphQLQuery } from "@neurectomy/test-config/mocks";

mockGraphQLQuery("GetAgents", {
  data: {
    agents: {
      nodes: [{ id: "1", name: "Test Agent" }],
      pageInfo: { hasNextPage: false },
    },
  },
});
```

#### mockGraphQLMutation(operationName, response)

Mock a GraphQL mutation.

```typescript
import { mockGraphQLMutation } from "@neurectomy/test-config/mocks";

mockGraphQLMutation("CreateAgent", {
  data: {
    createAgent: { id: "new-id", name: "New Agent" },
  },
});
```

#### resetMockHandlers()

Reset all mock handlers to initial state.

```typescript
import { resetMockHandlers } from "@neurectomy/test-config/mocks";
import { afterEach } from "vitest";

afterEach(() => {
  resetMockHandlers();
});
```

---

### Storage Mocks

#### createMockStore(initial?)

Create an in-memory key-value store mock.

```typescript
import { createMockStore } from "@neurectomy/test-config/mocks";

const store = createMockStore({ key: "value" });

await store.get("key"); // 'value'
await store.set("new", 123);
await store.delete("key");
await store.clear();
```

#### createMockCache(options?)

Create an in-memory cache mock with TTL support.

```typescript
import { createMockCache } from "@neurectomy/test-config/mocks";

const cache = createMockCache({ defaultTTL: 60000 });

await cache.get("key");
await cache.set("key", "value", 30000); // 30s TTL
await cache.has("key");
await cache.delete("key");
await cache.clear();
```

---

## Test Utilities

### utils/render.tsx

React rendering utilities wrapping @testing-library/react.

```typescript
import { render, screen, userEvent } from '@neurectomy/test-config/utils';

const { container, rerender } = render(<Button>Click me</Button>);

// Query elements
const button = screen.getByRole('button');
expect(button).toHaveTextContent('Click me');

// User interactions
const user = userEvent.setup();
await user.click(button);
```

### utils/factories.ts

Test data factories.

```typescript
import {
  createMockAgent,
  createMockWorkflow,
  createMockUser,
  createMockContainer,
} from "@neurectomy/test-config/utils";

// Create with defaults
const agent = createMockAgent();

// Override specific fields
const customAgent = createMockAgent({
  name: "Custom Agent",
  status: "running",
});

// Create multiple
const agents = Array.from({ length: 5 }, () => createMockAgent());
```

### utils/wait.ts

Async waiting utilities.

```typescript
import {
  waitFor,
  waitForCondition,
  sleep,
} from "@neurectomy/test-config/utils";

// Wait for assertion
await waitFor(() => {
  expect(screen.getByText("Loaded")).toBeInTheDocument();
});

// Wait for condition
await waitForCondition(() => store.isReady, { timeout: 5000 });

// Simple delay
await sleep(100);
```

---

## Configuration Reference

### Coverage Settings

Default coverage configuration:

```typescript
coverage: {
  provider: 'v8',
  reporter: ['text', 'json', 'html', 'lcov'],
  reportsDirectory: './coverage',
  exclude: [
    'node_modules/**',
    'dist/**',
    '**/*.d.ts',
    '**/*.test.ts',
    '**/*.spec.ts',
    '**/index.ts',
    '**/types.ts',
  ],
  thresholds: {
    global: {
      branches: 70,
      functions: 70,
      lines: 70,
      statements: 70,
    },
  },
}
```

### Test Patterns

| Config              | Include Patterns                         |
| ------------------- | ---------------------------------------- |
| `baseConfig`        | `src/**/*.test.ts`, `src/**/*.spec.ts`   |
| `reactConfig`       | `src/**/*.test.tsx`, `src/**/*.spec.tsx` |
| `integrationConfig` | `tests/integration/**/*.test.ts`         |
| `benchmarkConfig`   | `src/**/*.bench.ts`                      |

---

## Examples

### Full Test File Example

```typescript
import {
  describe,
  it,
  expect,
  beforeAll,
  afterAll,
  afterEach,
  vi,
} from "vitest";
import {
  startMockServer,
  stopMockServer,
  resetMockHandlers,
  mockRestEndpoint,
  createMockAgent,
} from "@neurectomy/test-config/mocks";
import { waitFor } from "@neurectomy/test-config/utils";

describe("AgentService", () => {
  beforeAll(() => startMockServer());
  afterAll(() => stopMockServer());
  afterEach(() => {
    resetMockHandlers();
    vi.clearAllMocks();
  });

  it("fetches agents from API", async () => {
    const mockAgent = createMockAgent({ name: "Test Agent" });
    mockRestEndpoint("get", "/api/agents", { agents: [mockAgent] });

    const service = new AgentService();
    const agents = await service.fetchAll();

    expect(agents).toHaveLength(1);
    expect(agents[0].name).toBe("Test Agent");
  });

  it("handles API errors gracefully", async () => {
    mockRestEndpoint(
      "get",
      "/api/agents",
      { error: "Server Error" },
      { status: 500 }
    );

    const service = new AgentService();

    await expect(service.fetchAll()).rejects.toThrow("Server Error");
  });
});
```

### React Component Test Example

```typescript
import { describe, it, expect, vi } from 'vitest';
import { render, screen, userEvent } from '@neurectomy/test-config/utils';
import { Button } from '@neurectomy/ui';

describe('Button', () => {
  it('calls onClick when clicked', async () => {
    const onClick = vi.fn();
    const user = userEvent.setup();

    render(<Button onClick={onClick}>Click me</Button>);

    await user.click(screen.getByRole('button'));

    expect(onClick).toHaveBeenCalledOnce();
  });

  it('is disabled when disabled prop is true', () => {
    render(<Button disabled>Disabled</Button>);

    expect(screen.getByRole('button')).toBeDisabled();
  });
});
```

---

## Peer Dependencies

- `vitest` >= 1.0.0
- `@testing-library/react` >= 14.0.0 (for React tests)
- `@testing-library/user-event` >= 14.0.0 (for React tests)
- `msw` >= 2.0.0 (for API mocking)
- `jsdom` >= 24.0.0 (for DOM tests)
