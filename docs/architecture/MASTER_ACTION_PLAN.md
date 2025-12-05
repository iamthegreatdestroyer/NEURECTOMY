# 🧠 NEURECTOMY Master Architecture Action Plan

> **Document Version:** 1.0  
> **Created:** December 5, 2025  
> **Author:** @ARCHITECT Agent  
> **Status:** Active

---

## Executive Summary

This document outlines a comprehensive action plan to address architectural issues identified in the NEURECTOMY repository. The plan is organized into phases with clear deliverables, dependencies, and success criteria.

**Current Architecture Score: 7.1/10**  
**Target Architecture Score: 9.0/10**

---

## 📋 Issue Summary

| ID  | Issue                                      | Severity | Phase |
| --- | ------------------------------------------ | -------- | ----- |
| A1  | Inconsistent build outputs across packages | High     | 1     |
| A2  | Missing shared TypeScript configuration    | High     | 1     |
| A3  | Inconsistent dependency declarations       | Medium   | 1     |
| A4  | Test coverage gaps in core packages        | High     | 2     |
| A5  | Cross-service type sharing gap             | Medium   | 3     |
| A6  | Missing package documentation              | Medium   | 4     |
| A7  | No API versioning strategy                 | Low      | 5     |
| A8  | No event sourcing for agent state          | Low      | 5     |

---

## 🚀 Phase 1: Build System Standardization (Week 1-2)

### Objective

Establish consistent, reliable build infrastructure across all packages.

### 1.1 Create Shared TypeScript Configuration

**Task:** Create base TypeScript configurations for different package types.

**Deliverables:**

```
packages/
├── typescript-config/
│   ├── package.json
│   ├── base.json           # Base config for all packages
│   ├── library.json        # For publishable library packages
│   ├── react-library.json  # For React component packages
│   ├── node.json           # For Node.js services
│   └── README.md
```

**Implementation:**

```jsonc
// packages/typescript-config/base.json
{
  "$schema": "https://json.schemastore.org/tsconfig",
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "noUncheckedIndexedAccess": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
  },
}
```

```jsonc
// packages/typescript-config/library.json
{
  "$schema": "https://json.schemastore.org/tsconfig",
  "extends": "./base.json",
  "compilerOptions": {
    "composite": true,
    "outDir": "./dist",
    "rootDir": "./src",
  },
}
```

```jsonc
// packages/typescript-config/react-library.json
{
  "$schema": "https://json.schemastore.org/tsconfig",
  "extends": "./library.json",
  "compilerOptions": {
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "jsx": "react-jsx",
  },
}
```

**Package.json:**

```json
{
  "name": "@neurectomy/typescript-config",
  "version": "0.1.0",
  "private": true,
  "files": ["*.json"],
  "exports": {
    "./base": "./base.json",
    "./library": "./library.json",
    "./react-library": "./react-library.json",
    "./node": "./node.json"
  }
}
```

---

### 1.2 Create Shared ESLint Configuration

**Deliverables:**

```
packages/
├── eslint-config/
│   ├── package.json
│   ├── base.js
│   ├── typescript.js
│   ├── react.js
│   └── README.md
```

**Implementation:**

```javascript
// packages/eslint-config/base.js
module.exports = {
  extends: ["eslint:recommended", "prettier"],
  env: {
    es2022: true,
    node: true,
  },
  parserOptions: {
    ecmaVersion: "latest",
    sourceType: "module",
  },
  rules: {
    "no-console": ["warn", { allow: ["warn", "error"] }],
    "no-unused-vars": "off", // Handled by TypeScript
    "prefer-const": "error",
    "no-var": "error",
  },
};
```

```javascript
// packages/eslint-config/typescript.js
module.exports = {
  extends: [
    "./base.js",
    "plugin:@typescript-eslint/recommended",
    "plugin:@typescript-eslint/recommended-requiring-type-checking",
  ],
  parser: "@typescript-eslint/parser",
  plugins: ["@typescript-eslint"],
  rules: {
    "@typescript-eslint/no-unused-vars": ["error", { argsIgnorePattern: "^_" }],
    "@typescript-eslint/explicit-function-return-type": "off",
    "@typescript-eslint/explicit-module-boundary-types": "off",
    "@typescript-eslint/no-explicit-any": "warn",
    "@typescript-eslint/no-floating-promises": "error",
    "@typescript-eslint/no-misused-promises": "error",
  },
};
```

```javascript
// packages/eslint-config/react.js
module.exports = {
  extends: [
    "./typescript.js",
    "plugin:react/recommended",
    "plugin:react-hooks/recommended",
    "plugin:jsx-a11y/recommended",
  ],
  plugins: ["react", "react-hooks", "jsx-a11y"],
  settings: {
    react: {
      version: "detect",
    },
  },
  rules: {
    "react/react-in-jsx-scope": "off",
    "react/prop-types": "off",
    "react-hooks/rules-of-hooks": "error",
    "react-hooks/exhaustive-deps": "warn",
  },
};
```

---

### 1.3 Standardize Package Build Configuration

**Task:** Migrate all packages to consistent tsup build configuration.

**Packages to Update:**

| Package                               | Current               | Target      |
| ------------------------------------- | --------------------- | ----------- |
| `@neurectomy/core`                    | tsc (raw .ts exports) | tsup → dist |
| `@neurectomy/types`                   | tsup ✓                | Keep        |
| `@neurectomy/ui`                      | tsc (raw .ts exports) | tsup → dist |
| `@neurectomy/api-client`              | tsc (raw .ts exports) | tsup → dist |
| `@neurectomy/3d-engine`               | tsup ✓                | Keep        |
| `@neurectomy/container-command`       | tsup ✓                | Keep        |
| `@neurectomy/discovery-engine`        | tsup ✓                | Keep        |
| `@neurectomy/continuous-intelligence` | tsup ✓                | Keep        |
| `@neurectomy/deployment-orchestrator` | tsup ✓                | Keep        |
| `@neurectomy/performance-engine`      | tsup ✓                | Keep        |

**Standard tsup.config.ts Template:**

```typescript
import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["esm", "cjs"],
  dts: true,
  sourcemap: true,
  clean: true,
  treeshake: true,
  splitting: false,
  minify: false,
  external: [
    // Add peer dependencies here
  ],
});
```

**Standard package.json exports:**

```json
{
  "main": "./dist/index.cjs",
  "module": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "exports": {
    ".": {
      "types": "./dist/index.d.ts",
      "import": "./dist/index.js",
      "require": "./dist/index.cjs"
    }
  },
  "files": ["dist", "README.md"]
}
```

---

### 1.4 Standardize Dependency Declarations

**Task:** Establish clear rules for dependency types.

**Rules:**

```
1. @neurectomy/* packages used at RUNTIME → dependencies
2. @neurectomy/* packages used only for TYPES → peerDependencies
3. Build tools, test frameworks → devDependencies
4. React, Three.js (host-provided) → peerDependencies
```

**Packages to Fix:**

| Package                               | Change                                             |
| ------------------------------------- | -------------------------------------------------- |
| `@neurectomy/discovery-engine`        | Move `@neurectomy/types` from peer to dependencies |
| `@neurectomy/continuous-intelligence` | Move `@neurectomy/types` from peer to dependencies |
| `@neurectomy/performance-engine`      | Keep consistent with others                        |

---

### 1.5 Success Criteria - Phase 1

- [ ] All packages extend shared TypeScript config
- [ ] All packages use tsup for building
- [ ] All packages produce dist/ artifacts with .js, .cjs, .d.ts
- [ ] `pnpm build` succeeds for all packages
- [ ] No raw .ts exports in any package
- [ ] ESLint config shared across all packages
- [ ] Dependency declarations follow documented rules

---

## 🧪 Phase 2: Test Infrastructure (Week 3-4)

### Objective

Establish comprehensive testing across all packages.

### 2.1 Create Shared Test Configuration

**Deliverables:**

```
packages/
├── test-config/
│   ├── package.json
│   ├── vitest.shared.ts
│   ├── setup.ts
│   ├── mocks/
│   │   ├── docker.ts
│   │   ├── kubernetes.ts
│   │   └── fetch.ts
│   └── README.md
```

**Implementation:**

```typescript
// packages/test-config/vitest.shared.ts
import { defineConfig } from "vitest/config";

export const createVitestConfig = (options?: {
  coverage?: boolean;
  react?: boolean;
}) =>
  defineConfig({
    test: {
      globals: true,
      environment: options?.react ? "jsdom" : "node",
      setupFiles: options?.react
        ? ["@neurectomy/test-config/setup-react"]
        : ["@neurectomy/test-config/setup"],
      coverage: options?.coverage
        ? {
            provider: "v8",
            reporter: ["text", "json", "html"],
            exclude: [
              "node_modules/",
              "dist/",
              "**/*.d.ts",
              "**/*.test.ts",
              "**/index.ts",
            ],
            thresholds: {
              statements: 80,
              branches: 75,
              functions: 80,
              lines: 80,
            },
          }
        : undefined,
      include: ["src/**/*.test.ts", "src/**/*.spec.ts"],
      exclude: ["node_modules", "dist"],
    },
  });
```

```typescript
// packages/test-config/setup.ts
import { vi } from "vitest";

// Global mocks
vi.mock("pino", () => ({
  default: () => ({
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    debug: vi.fn(),
  }),
}));

// Reset mocks between tests
beforeEach(() => {
  vi.clearAllMocks();
});
```

---

### 2.2 Add Tests to Untested Packages

**Priority Order:**

#### 2.2.1 `@neurectomy/core` Tests

```typescript
// packages/core/src/__tests__/utils/identifiers.test.ts
import { describe, it, expect } from "vitest";
import { generateId, createTimestamp } from "../../utils/identifiers";

describe("identifiers", () => {
  describe("generateId", () => {
    it("should generate unique IDs", () => {
      const id1 = generateId();
      const id2 = generateId();
      expect(id1).not.toBe(id2);
    });

    it("should generate valid UUID format", () => {
      const id = generateId();
      expect(id).toMatch(
        /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i
      );
    });
  });

  describe("createTimestamp", () => {
    it("should return current timestamp", () => {
      const before = Date.now();
      const timestamp = createTimestamp();
      const after = Date.now();
      expect(timestamp.getTime()).toBeGreaterThanOrEqual(before);
      expect(timestamp.getTime()).toBeLessThanOrEqual(after);
    });
  });
});
```

```typescript
// packages/core/src/__tests__/schemas/agent.test.ts
import { describe, it, expect } from "vitest";
import { agentConfigSchema } from "../../schemas/agent";

describe("agentConfigSchema", () => {
  it("should validate valid agent config", () => {
    const validConfig = {
      name: "Test Agent",
      codename: "TEST",
      tier: "foundational",
      description: "A test agent",
      philosophy: "Testing is important",
      capabilities: ["test", "validate"],
    };
    expect(() => agentConfigSchema.parse(validConfig)).not.toThrow();
  });

  it("should reject invalid tier", () => {
    const invalidConfig = {
      name: "Test Agent",
      codename: "TEST",
      tier: "invalid-tier",
      description: "A test agent",
      philosophy: "Testing is important",
      capabilities: ["test"],
    };
    expect(() => agentConfigSchema.parse(invalidConfig)).toThrow();
  });

  it("should require all mandatory fields", () => {
    const incompleteConfig = {
      name: "Test Agent",
    };
    expect(() => agentConfigSchema.parse(incompleteConfig)).toThrow();
  });
});
```

#### 2.2.2 `@neurectomy/types` Tests

```typescript
// packages/types/src/__tests__/index.test.ts
import { describe, it, expect, expectTypeOf } from "vitest";
import type { Agent, AgentStatus, AgentTier, Workflow } from "../index";

describe("Type Definitions", () => {
  describe("Agent Types", () => {
    it("should have correct AgentStatus values", () => {
      const statuses: AgentStatus[] = [
        "idle",
        "running",
        "paused",
        "error",
        "completed",
        "active",
      ];
      expect(statuses).toHaveLength(6);
    });

    it("should have correct AgentTier values", () => {
      const tiers: AgentTier[] = [
        "foundational",
        "specialist",
        "innovator",
        "meta",
        "domain",
        "emerging",
        "human-centric",
        "enterprise",
      ];
      expect(tiers).toHaveLength(8);
    });
  });

  describe("Type Structure", () => {
    it("Agent should have required properties", () => {
      const agent: Agent = {
        id: "123",
        name: "Test",
        codename: "TEST",
        tier: "foundational",
        status: "idle",
        description: "desc",
        philosophy: "phil",
        capabilities: [],
        version: "1.0.0",
        createdAt: new Date(),
        updatedAt: new Date(),
        metadata: {},
      };
      expect(agent.id).toBe("123");
    });
  });
});
```

#### 2.2.3 `@neurectomy/ui` Tests

```typescript
// packages/ui/src/__tests__/components/Button.test.tsx
import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { Button } from '../../components/Button';

describe('Button', () => {
  it('renders with children', () => {
    render(<Button>Click me</Button>);
    expect(screen.getByText('Click me')).toBeInTheDocument();
  });

  it('calls onClick when clicked', () => {
    const handleClick = vi.fn();
    render(<Button onClick={handleClick}>Click</Button>);
    fireEvent.click(screen.getByText('Click'));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('applies variant styles', () => {
    render(<Button variant="destructive">Delete</Button>);
    const button = screen.getByText('Delete');
    expect(button).toHaveClass('destructive');
  });

  it('is disabled when disabled prop is true', () => {
    render(<Button disabled>Disabled</Button>);
    expect(screen.getByText('Disabled')).toBeDisabled();
  });
});
```

#### 2.2.4 `@neurectomy/api-client` Tests

```typescript
// packages/api-client/src/__tests__/graphql/client.test.ts
import { describe, it, expect, vi, beforeEach } from "vitest";
import { GraphQLClient } from "graphql-request";
import { createApiClient } from "../../graphql";

vi.mock("graphql-request");

describe("API Client", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("creates client with correct endpoint", () => {
    const client = createApiClient({
      endpoint: "http://localhost:4000/graphql",
    });
    expect(GraphQLClient).toHaveBeenCalledWith(
      "http://localhost:4000/graphql",
      expect.any(Object)
    );
  });

  it("includes auth header when token provided", () => {
    const client = createApiClient({
      endpoint: "http://localhost:4000/graphql",
      token: "test-token",
    });
    expect(GraphQLClient).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: "Bearer test-token",
        }),
      })
    );
  });
});
```

---

### 2.3 Add Integration Test Infrastructure

**Deliverables:**

```
tests/
├── integration/
│   ├── vitest.config.ts
│   ├── setup.ts
│   ├── fixtures/
│   │   ├── agents.ts
│   │   ├── workflows.ts
│   │   └── containers.ts
│   ├── rust-ml-integration.test.ts
│   ├── container-workflow.test.ts
│   └── agent-lifecycle.test.ts
├── e2e/
│   ├── playwright.config.ts
│   ├── fixtures/
│   └── specs/
└── README.md
```

**Integration Test Example:**

```typescript
// tests/integration/agent-lifecycle.test.ts
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { createApiClient } from "@neurectomy/api-client";

describe("Agent Lifecycle Integration", () => {
  let client: ReturnType<typeof createApiClient>;
  let testAgentId: string;

  beforeAll(async () => {
    client = createApiClient({
      endpoint: process.env.GRAPHQL_ENDPOINT || "http://localhost:4000/graphql",
    });
  });

  afterAll(async () => {
    // Cleanup test agent
    if (testAgentId) {
      await client.deleteAgent({ id: testAgentId });
    }
  });

  it("should create, start, and stop an agent", async () => {
    // Create
    const created = await client.createAgent({
      name: "Integration Test Agent",
      codename: "INT_TEST",
      tier: "foundational",
      description: "Created by integration test",
      philosophy: "Testing ensures reliability",
      capabilities: ["test"],
    });
    testAgentId = created.id;
    expect(created.status).toBe("idle");

    // Start
    const started = await client.startAgent({ id: testAgentId });
    expect(started.status).toBe("running");

    // Stop
    const stopped = await client.stopAgent({ id: testAgentId });
    expect(stopped.status).toBe("idle");
  });
});
```

---

### 2.4 Configure CI/CD Test Pipeline

**GitHub Actions Workflow:**

```yaml
# .github/workflows/test.yml
name: Test

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node-version: [20.x]
    steps:
      - uses: actions/checkout@v4

      - uses: pnpm/action-setup@v2
        with:
          version: 8

      - name: Use Node.js ${{ matrix.node-version }}
        uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
          cache: "pnpm"

      - name: Install dependencies
        run: pnpm install --frozen-lockfile

      - name: Build packages
        run: pnpm build

      - name: Run unit tests
        run: pnpm test

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage/lcov.info
          fail_ci_if_error: true

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    services:
      postgres:
        image: pgvector/pgvector:pg16
        env:
          POSTGRES_USER: neurectomy
          POSTGRES_PASSWORD: neurectomy
          POSTGRES_DB: neurectomy_test
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
    steps:
      - uses: actions/checkout@v4

      - uses: pnpm/action-setup@v2
        with:
          version: 8

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: 20.x
          cache: "pnpm"

      - name: Install dependencies
        run: pnpm install --frozen-lockfile

      - name: Build packages
        run: pnpm build

      - name: Run integration tests
        run: pnpm test:integration
        env:
          DATABASE_URL: postgres://neurectomy:neurectomy@localhost:5432/neurectomy_test
          REDIS_URL: redis://localhost:6379
```

---

### 2.5 Success Criteria - Phase 2

- [ ] All packages have vitest configuration
- [ ] `@neurectomy/core` has >80% test coverage
- [ ] `@neurectomy/types` has type validation tests
- [ ] `@neurectomy/ui` has component tests with React Testing Library
- [ ] `@neurectomy/api-client` has client tests with mocked responses
- [ ] Integration test suite runs against local services
- [ ] GitHub Actions runs tests on every PR
- [ ] Coverage reports uploaded to Codecov
- [ ] Test failures block merges

---

## 🔗 Phase 3: Cross-Service Type Synchronization (Month 2)

### Objective

Establish single source of truth for types across TypeScript, Rust, and Python.

### 3.1 Implement Schema-First Type Generation

**Strategy Options:**

| Option           | Pros                                | Cons                    | Recommendation           |
| ---------------- | ----------------------------------- | ----------------------- | ------------------------ |
| Protocol Buffers | Mature, efficient binary, all langs | Verbose, learning curve | For high-throughput APIs |
| JSON Schema      | Human-readable, wide support        | No RPC support          | For REST/GraphQL types   |
| TypeSpec         | Microsoft-backed, OpenAPI native    | Newer, less tooling     | For OpenAPI-first        |

**Recommended Approach:** JSON Schema + Custom Codegen

**Deliverables:**

```
schemas/
├── definitions/
│   ├── agent.schema.json
│   ├── workflow.schema.json
│   ├── container.schema.json
│   └── common.schema.json
├── generators/
│   ├── typescript.ts
│   ├── rust.ts
│   └── python.ts
├── generated/
│   ├── typescript/
│   ├── rust/
│   └── python/
└── codegen.config.ts
```

**JSON Schema Example:**

```json
// schemas/definitions/agent.schema.json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://neurectomy.io/schemas/agent.json",
  "title": "Agent",
  "description": "NEURECTOMY Agent definition",
  "type": "object",
  "required": ["id", "name", "codename", "tier", "status", "capabilities"],
  "properties": {
    "id": {
      "type": "string",
      "format": "uuid",
      "description": "Unique identifier"
    },
    "name": {
      "type": "string",
      "minLength": 1,
      "maxLength": 100
    },
    "codename": {
      "type": "string",
      "pattern": "^[A-Z][A-Z0-9_]*$"
    },
    "tier": {
      "$ref": "#/definitions/AgentTier"
    },
    "status": {
      "$ref": "#/definitions/AgentStatus"
    },
    "capabilities": {
      "type": "array",
      "items": { "type": "string" },
      "minItems": 1
    },
    "metadata": {
      "$ref": "#/definitions/AgentMetadata"
    }
  },
  "definitions": {
    "AgentTier": {
      "type": "string",
      "enum": [
        "foundational",
        "specialist",
        "innovator",
        "meta",
        "domain",
        "emerging",
        "human-centric",
        "enterprise"
      ]
    },
    "AgentStatus": {
      "type": "string",
      "enum": ["idle", "running", "paused", "error", "completed", "active"]
    },
    "AgentMetadata": {
      "type": "object",
      "properties": {
        "icon": { "type": "string" },
        "color": { "type": "string" },
        "tags": {
          "type": "array",
          "items": { "type": "string" }
        }
      }
    }
  }
}
```

**Codegen Script:**

```typescript
// schemas/codegen.config.ts
import { compileFromFile } from "json-schema-to-typescript";
import { writeFileSync, mkdirSync } from "fs";
import { join } from "path";

const SCHEMAS = [
  "agent.schema.json",
  "workflow.schema.json",
  "container.schema.json",
];

async function generateTypeScript() {
  for (const schema of SCHEMAS) {
    const ts = await compileFromFile(join(__dirname, "definitions", schema), {
      bannerComment:
        "// AUTO-GENERATED - DO NOT EDIT\n// Source: schemas/definitions/" +
        schema,
    });

    const outPath = join(
      __dirname,
      "generated/typescript",
      schema.replace(".schema.json", ".ts")
    );
    mkdirSync(join(__dirname, "generated/typescript"), { recursive: true });
    writeFileSync(outPath, ts);
  }
}

async function generateRust() {
  // Use typify or schemafy for Rust generation
  const { compileSchema } = await import("typify");

  for (const schema of SCHEMAS) {
    const rust = await compileSchema({
      input: join(__dirname, "definitions", schema),
      derive: ["Debug", "Clone", "Serialize", "Deserialize"],
    });

    const outPath = join(
      __dirname,
      "generated/rust",
      schema.replace(".schema.json", ".rs")
    );
    mkdirSync(join(__dirname, "generated/rust"), { recursive: true });
    writeFileSync(outPath, rust);
  }
}

async function generatePython() {
  // Use datamodel-code-generator for Python
  const { generate } = await import("datamodel-code-generator");

  for (const schema of SCHEMAS) {
    const python = await generate({
      input: join(__dirname, "definitions", schema),
      inputFileType: "jsonschema",
      output: join(
        __dirname,
        "generated/python",
        schema.replace(".schema.json", ".py")
      ),
    });
  }
}

// Run all generators
Promise.all([generateTypeScript(), generateRust(), generatePython()]).then(() =>
  console.log("✅ All types generated")
);
```

**Package.json Script:**

```json
{
  "scripts": {
    "codegen": "ts-node schemas/codegen.config.ts",
    "codegen:watch": "chokidar 'schemas/definitions/**/*.json' -c 'pnpm codegen'"
  }
}
```

---

### 3.2 GraphQL Schema as Source of Truth (Alternative)

If you prefer GraphQL-first approach:

```graphql
# schemas/graphql/schema.graphql
enum AgentTier {
  FOUNDATIONAL
  SPECIALIST
  INNOVATOR
  META
  DOMAIN
  EMERGING
  HUMAN_CENTRIC
  ENTERPRISE
}

enum AgentStatus {
  IDLE
  RUNNING
  PAUSED
  ERROR
  COMPLETED
  ACTIVE
}

type Agent {
  id: ID!
  name: String!
  codename: String!
  tier: AgentTier!
  status: AgentStatus!
  description: String!
  philosophy: String!
  capabilities: [String!]!
  version: String!
  createdAt: DateTime!
  updatedAt: DateTime!
  metadata: AgentMetadata
}
```

**Generate types from GraphQL:**

- TypeScript: `@graphql-codegen/typescript`
- Rust: `graphql-client` crate
- Python: `ariadne-codegen` or `strawberry-graphql`

---

### 3.3 Success Criteria - Phase 3

- [ ] Single source of truth for all shared types (JSON Schema or GraphQL)
- [ ] TypeScript types auto-generated
- [ ] Rust types auto-generated
- [ ] Python types auto-generated (Pydantic models)
- [ ] Codegen runs in CI on schema changes
- [ ] Type mismatches caught at build time
- [ ] Documentation generated from schemas

---

## 📚 Phase 4: Documentation (Month 3)

### Objective

Comprehensive documentation for all packages and services.

### 4.1 Package README Standards

Every package must have a README with:

```markdown
# @neurectomy/[package-name]

> Brief description

## Installation

\`\`\`bash
pnpm add @neurectomy/[package-name]
\`\`\`

## Quick Start

\`\`\`typescript
// Minimal working example
\`\`\`

## API Reference

### Functions

#### `functionName(params): ReturnType`

Description of function.

**Parameters:**

- `param1` (Type) - Description

**Returns:** Description of return value

**Example:**
\`\`\`typescript
const result = functionName({ param1: 'value' });
\`\`\`

## Architecture

Brief explanation of internal architecture.

## Contributing

Link to CONTRIBUTING.md

## License

Proprietary - See LICENSE
```

---

### 4.2 API Documentation with TypeDoc

**Setup:**

```bash
pnpm add -D typedoc typedoc-plugin-markdown
```

**typedoc.json:**

```json
{
  "entryPoints": ["packages/*/src/index.ts"],
  "out": "docs/api",
  "plugin": ["typedoc-plugin-markdown"],
  "excludePrivate": true,
  "excludeInternal": true,
  "readme": "none",
  "githubPages": false
}
```

**Package.json:**

```json
{
  "scripts": {
    "docs:api": "typedoc",
    "docs:serve": "typedoc --watch"
  }
}
```

---

### 4.3 Architecture Decision Records (ADRs)

**Template:**

```markdown
# ADR-[NUMBER]: [TITLE]

## Status

[Proposed | Accepted | Deprecated | Superseded]

## Context

What is the issue we're seeing that motivates this decision?

## Decision

What is the change we're proposing?

## Consequences

What becomes easier or harder because of this change?

## Alternatives Considered

What other options were evaluated?
```

**Initial ADRs to Create:**

| ADR     | Title                                  |
| ------- | -------------------------------------- |
| ADR-001 | Monorepo Structure with Turborepo      |
| ADR-002 | Polyglot Architecture (TS/Rust/Python) |
| ADR-003 | GraphQL as Primary API                 |
| ADR-004 | Event-Driven with NATS JetStream       |
| ADR-005 | WebGPU-First 3D Rendering              |
| ADR-006 | Type Generation Strategy               |

---

### 4.4 Success Criteria - Phase 4

- [ ] All packages have comprehensive README.md
- [ ] TypeDoc generates API documentation
- [ ] ADRs document all major decisions
- [ ] Architecture diagrams (C4 model) created
- [ ] Getting started guide updated
- [ ] Troubleshooting guide expanded
- [ ] All docs reviewed for accuracy

---

## 🔮 Phase 5: Advanced Architecture (Month 4+)

### Objective

Implement advanced patterns for long-term scalability.

### 5.1 API Versioning Strategy

**GraphQL Approach:**

```graphql
# Deprecate fields, don't remove
type Agent {
  id: ID!
  name: String!
  displayName: String! @deprecated(reason: "Use name instead")
}

# Add new capabilities via directives
type Query {
  agents: [Agent!]!
  agentsV2(filter: AgentFilter): AgentConnection! @since(version: "2.0")
}
```

**REST Approach:**

```
/api/v1/agents  # Current stable
/api/v2/agents  # Next version (beta)
```

**Version Header:**

```
Accept: application/vnd.neurectomy.v2+json
```

---

### 5.2 Event Sourcing for Agent State

**NATS Subjects:**

```
neurectomy.agents.{agentId}.events
neurectomy.agents.{agentId}.commands
neurectomy.workflows.{workflowId}.events
neurectomy.containers.{containerId}.events
```

**Event Types:**

```typescript
// packages/events/src/types.ts
interface AgentEvent {
  type: "AGENT_CREATED" | "AGENT_STARTED" | "AGENT_STOPPED" | "AGENT_ERROR";
  agentId: string;
  timestamp: Date;
  payload: unknown;
  metadata: {
    correlationId: string;
    causationId?: string;
    userId?: string;
  };
}

interface AgentCreatedEvent extends AgentEvent {
  type: "AGENT_CREATED";
  payload: {
    name: string;
    codename: string;
    tier: AgentTier;
    capabilities: string[];
  };
}
```

**Event Store:**

```typescript
// packages/events/src/store.ts
export class EventStore {
  constructor(private nats: NatsConnection) {}

  async append(stream: string, event: AgentEvent): Promise<void> {
    const js = this.nats.jetstream();
    await js.publish(stream, JSON.stringify(event));
  }

  async *read(
    stream: string,
    fromSequence?: number
  ): AsyncGenerator<AgentEvent> {
    const js = this.nats.jetstream();
    const consumer = await js.consumers.get(stream);

    for await (const msg of consumer.consume()) {
      yield JSON.parse(msg.data.toString()) as AgentEvent;
      msg.ack();
    }
  }

  async replay(
    stream: string,
    handler: (event: AgentEvent) => Promise<void>
  ): Promise<void> {
    for await (const event of this.read(stream)) {
      await handler(event);
    }
  }
}
```

---

### 5.3 Feature Flags System

```typescript
// packages/feature-flags/src/index.ts
import { z } from "zod";

const FeatureFlagSchema = z.object({
  WEBGPU_RENDERING: z.boolean().default(true),
  FIRECRACKER_RUNTIME: z.boolean().default(false),
  EXPERIMENTAL_AGENTS: z.boolean().default(false),
  EVENT_SOURCING: z.boolean().default(false),
  MULTI_REGION: z.boolean().default(false),
});

type FeatureFlags = z.infer<typeof FeatureFlagSchema>;

class FeatureFlagService {
  private flags: FeatureFlags;

  constructor() {
    this.flags = FeatureFlagSchema.parse({
      WEBGPU_RENDERING: process.env.ENABLE_WEBGPU === "true",
      FIRECRACKER_RUNTIME: process.env.ENABLE_FIRECRACKER === "true",
      EXPERIMENTAL_AGENTS: process.env.ENABLE_EXPERIMENTAL === "true",
      EVENT_SOURCING: process.env.ENABLE_EVENT_SOURCING === "true",
      MULTI_REGION: process.env.ENABLE_MULTI_REGION === "true",
    });
  }

  isEnabled(flag: keyof FeatureFlags): boolean {
    return this.flags[flag];
  }

  getAll(): FeatureFlags {
    return { ...this.flags };
  }
}

export const featureFlags = new FeatureFlagService();
```

---

### 5.4 Success Criteria - Phase 5

- [ ] API versioning documented and implemented
- [ ] Event sourcing infrastructure deployed
- [ ] Feature flags system operational
- [ ] NATS JetStream event streams configured
- [ ] Event replay capability tested
- [ ] Migration path documented for major versions

---

## 📊 Overall Progress Tracking

### Phase Completion Checklist

| Phase                  | Status         | Target Date | Completion |
| ---------------------- | -------------- | ----------- | ---------- |
| Phase 1: Build System  | 🔴 Not Started | Week 2      | 0%         |
| Phase 2: Testing       | 🔴 Not Started | Week 4      | 0%         |
| Phase 3: Type Sync     | 🔴 Not Started | Month 2     | 0%         |
| Phase 4: Documentation | 🔴 Not Started | Month 3     | 0%         |
| Phase 5: Advanced      | 🔴 Not Started | Month 4+    | 0%         |

### Architecture Score Targets

| Dimension     | Current | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 |
| ------------- | ------- | ------- | ------- | ------- | ------- | ------- |
| Modularity    | 9/10    | 9/10    | 9/10    | 9/10    | 9/10    | 10/10   |
| Type Safety   | 8/10    | 8/10    | 8/10    | 10/10   | 10/10   | 10/10   |
| Build System  | 7/10    | 9/10    | 9/10    | 9/10    | 9/10    | 9/10    |
| Testing       | 5/10    | 5/10    | 9/10    | 9/10    | 9/10    | 9/10    |
| Documentation | 4/10    | 4/10    | 5/10    | 6/10    | 9/10    | 9/10    |
| Observability | 9/10    | 9/10    | 9/10    | 9/10    | 9/10    | 10/10   |
| Scalability   | 8/10    | 8/10    | 8/10    | 8/10    | 8/10    | 10/10   |
| DX            | 7/10    | 8/10    | 9/10    | 9/10    | 9/10    | 9/10    |
| **TOTAL**     | **7.1** | **7.5** | **8.0** | **8.6** | **9.0** | **9.5** |

---

## 🛠️ Quick Reference: Commands

```bash
# Phase 1
pnpm create-config         # Create shared configs
pnpm build                  # Build all packages
pnpm lint                   # Lint all packages

# Phase 2
pnpm test                   # Run all tests
pnpm test:coverage          # Run with coverage
pnpm test:integration       # Run integration tests

# Phase 3
pnpm codegen                # Generate types from schemas
pnpm codegen:watch          # Watch mode

# Phase 4
pnpm docs:api               # Generate API docs
pnpm docs:serve             # Serve docs locally

# General
pnpm clean                  # Clean all build artifacts
pnpm typecheck              # TypeScript type checking
pnpm format                 # Format all files
```

---

## 📞 Support & Questions

- **Architecture Questions:** @ARCHITECT agent
- **Build Issues:** @FLUX agent
- **Testing Issues:** @ECLIPSE agent
- **Type Generation:** @SYNAPSE agent
- **Documentation:** @SCRIBE agent

---

**Document End**

_Last Updated: December 5, 2025_  
_Next Review: January 5, 2026_
