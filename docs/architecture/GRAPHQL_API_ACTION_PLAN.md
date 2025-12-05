# 🔷 NEURECTOMY GraphQL API Enhancement Action Plan

> **Document Version:** 1.0  
> **Created:** December 5, 2025  
> **Authors:** @ARCHITECT + @SYNAPSE Agents  
> **Status:** Active  
> **Parent Document:** [MASTER_ACTION_PLAN.md](./MASTER_ACTION_PLAN.md)

---

## Executive Summary

This document provides a comprehensive action plan to enhance the NEURECTOMY GraphQL API. The plan addresses five key areas: interface abstractions, unified mutations, type generation, subscription resilience, and security directives.

**Current GraphQL API Score: 7.0/10**  
**Target GraphQL API Score: 9.5/10**

---

## 📋 Issue Summary

| ID    | Issue                                   | Severity | Phase | Est. Effort |
| ----- | --------------------------------------- | -------- | ----- | ----------- |
| GQL1  | Missing interface abstractions (Node)   | High     | 1     | 2 days      |
| GQL2  | Inconsistent mutation response patterns | High     | 2     | 3 days      |
| GQL3  | No type generation pipeline             | High     | 3     | 4 days      |
| GQL4  | Missing pagination standardization      | Medium   | 1     | 1 day       |
| GQL5  | Subscription reconnection gaps          | Medium   | 4     | 3 days      |
| GQL6  | No field-level authorization            | High     | 5     | 4 days      |
| GQL7  | Missing query complexity analysis       | Medium   | 5     | 2 days      |
| GQL8  | No persisted queries support            | Low      | 3     | 2 days      |
| GQL9  | Subscription filtering limitations      | Medium   | 4     | 2 days      |
| GQL10 | Missing deprecation strategy            | Low      | 2     | 1 day       |

---

## 📁 Deliverables Overview

```
packages/
├── graphql-schema/                    # NEW: Unified schema package
│   ├── package.json
│   ├── codegen.yml
│   ├── schema/
│   │   ├── schema.graphql            # Master schema
│   │   ├── directives.graphql        # Custom directives
│   │   ├── interfaces.graphql        # Node, Timestamped, etc.
│   │   ├── scalars.graphql           # UUID, DateTime, JSON
│   │   ├── enums.graphql             # All enums
│   │   ├── types/
│   │   │   ├── agent.graphql
│   │   │   ├── conversation.graphql
│   │   │   ├── workflow.graphql
│   │   │   └── user.graphql
│   │   ├── inputs/
│   │   │   ├── agent.input.graphql
│   │   │   ├── filters.input.graphql
│   │   │   └── pagination.input.graphql
│   │   ├── mutations/
│   │   │   ├── agent.mutation.graphql
│   │   │   ├── auth.mutation.graphql
│   │   │   └── workflow.mutation.graphql
│   │   └── subscriptions/
│   │       ├── agent.subscription.graphql
│   │       └── system.subscription.graphql
│   └── generated/
│       ├── typescript/
│       ├── rust/
│       └── python/
├── api-client/
│   └── src/
│       └── graphql/
│           ├── __generated__/        # Auto-generated types
│           ├── client.ts             # Enhanced client
│           ├── subscriptions.ts      # Subscription manager
│           └── persisted-queries.ts  # Query hashes
```

---

## 🚀 Phase 1: Interface Abstractions (Week 1)

### Objective

Establish DRY schema patterns using GraphQL interfaces for consistent type definitions.

### 1.1 Core Interfaces

**File:** `packages/graphql-schema/schema/interfaces.graphql`

```graphql
"""
Relay-compliant Node interface for global object identification.
All entities that can be fetched by ID must implement this interface.
"""
interface Node {
  "Globally unique identifier"
  id: UUID!
}

"""
Interface for entities with audit timestamps.
"""
interface Timestamped {
  "When the entity was created"
  createdAt: DateTime!
  "When the entity was last updated"
  updatedAt: DateTime!
}

"""
Interface for entities owned by a user.
"""
interface Owned {
  "The user who owns this entity"
  owner: User!
}

"""
Interface for soft-deletable entities.
"""
interface SoftDeletable {
  "Whether the entity is deleted"
  isDeleted: Boolean!
  "When the entity was deleted"
  deletedAt: DateTime
}

"""
Interface for entities with a status lifecycle.
"""
interface Stateful {
  "Current status of the entity"
  status: String!
}

"""
Combined interface for standard CRUD entities.
"""
interface Entity implements Node & Timestamped {
  id: UUID!
  createdAt: DateTime!
  updatedAt: DateTime!
}
```

### 1.2 Updated Type Definitions

**Before (Current):**

```graphql
type Agent {
  id: UUID!
  name: String!
  status: AgentStatus!
  owner: User!
  createdAt: DateTime!
  updatedAt: DateTime!
}

type Workflow {
  id: UUID!
  name: String!
  status: WorkflowStatus!
  owner: User!
  createdAt: DateTime!
  updatedAt: DateTime!
}
```

**After (With Interfaces):**

```graphql
type Agent implements Node & Timestamped & Owned & Stateful {
  id: UUID!
  name: String!
  description: String
  status: AgentStatus!
  systemPrompt: String!
  model: ModelConfig!
  config: JSON!
  tools: [Tool!]!
  owner: User!
  createdAt: DateTime!
  updatedAt: DateTime!
}

type Workflow implements Node & Timestamped & Owned & Stateful {
  id: UUID!
  name: String!
  description: String
  status: WorkflowStatus!
  definition: JSON!
  triggers: [WorkflowTrigger!]!
  owner: User!
  createdAt: DateTime!
  updatedAt: DateTime!
}
```

### 1.3 Relay Connection Interfaces

**File:** `packages/graphql-schema/schema/connections.graphql`

```graphql
"""
Standard Relay PageInfo for cursor-based pagination.
"""
type PageInfo {
  "When paginating forwards, are there more items?"
  hasNextPage: Boolean!
  "When paginating backwards, are there more items?"
  hasPreviousPage: Boolean!
  "Cursor of the first edge"
  startCursor: String
  "Cursor of the last edge"
  endCursor: String
}

"""
Generic connection interface for pagination.
"""
interface Connection {
  "Pagination information"
  pageInfo: PageInfo!
  "Total count of items"
  totalCount: Int!
}

"""
Generic edge interface.
"""
interface Edge {
  "Cursor for this edge"
  cursor: String!
}
```

### 1.4 Success Criteria - Phase 1

- [ ] `interfaces.graphql` created with Node, Timestamped, Owned, Stateful
- [ ] All entity types implement appropriate interfaces
- [ ] Connection types follow Relay specification
- [ ] Schema validates without errors
- [ ] Backward compatibility maintained

---

## 🔄 Phase 2: Unified Mutation Patterns (Week 2)

### Objective

Standardize all mutation responses for predictable client-side handling and error management.

### 2.1 Standard Mutation Payload Interface

**File:** `packages/graphql-schema/schema/mutations/base.graphql`

```graphql
"""
User-facing error returned from mutations.
"""
type UserError {
  "Path to the input field that caused the error"
  field: [String!]
  "Human-readable error message"
  message: String!
  "Machine-readable error code"
  code: ErrorCode!
}

"""
Error codes for mutation failures.
"""
enum ErrorCode {
  INVALID_INPUT
  NOT_FOUND
  UNAUTHORIZED
  FORBIDDEN
  CONFLICT
  RATE_LIMITED
  INTERNAL_ERROR
  VALIDATION_FAILED
  DEPENDENCY_ERROR
}

"""
Base interface for all mutation payloads.
"""
interface MutationPayload {
  "Whether the mutation succeeded"
  success: Boolean!
  "Human-readable message"
  message: String
  "List of user-facing errors"
  userErrors: [UserError!]!
}
```

### 2.2 Entity-Specific Payloads

```graphql
# Agent Mutations
type CreateAgentPayload implements MutationPayload {
  success: Boolean!
  message: String
  userErrors: [UserError!]!
  agent: Agent
}

type UpdateAgentPayload implements MutationPayload {
  success: Boolean!
  message: String
  userErrors: [UserError!]!
  agent: Agent
}

type DeleteAgentPayload implements MutationPayload {
  success: Boolean!
  message: String
  userErrors: [UserError!]!
  deletedId: UUID
  deletedAt: DateTime
}

# Batch Operations
type BatchAgentPayload implements MutationPayload {
  success: Boolean!
  message: String
  userErrors: [UserError!]!
  agents: [Agent!]
  failedIds: [UUID!]
  successCount: Int!
  failureCount: Int!
}
```

### 2.3 Mutation Definitions with Payloads

```graphql
type Mutation {
  # Agent CRUD
  createAgent(input: CreateAgentInput!): CreateAgentPayload!
  updateAgent(id: UUID!, input: UpdateAgentInput!): UpdateAgentPayload!
  deleteAgent(id: UUID!): DeleteAgentPayload!

  # Agent Batch Operations
  createAgents(inputs: [CreateAgentInput!]!): BatchAgentPayload!
  deleteAgents(ids: [UUID!]!): BatchAgentPayload!
  updateAgentStatuses(updates: [AgentStatusUpdateInput!]!): BatchAgentPayload!

  # Agent State Changes
  startAgent(id: UUID!): UpdateAgentPayload!
  stopAgent(id: UUID!): UpdateAgentPayload!
  restartAgent(id: UUID!): UpdateAgentPayload!
}
```

### 2.4 Client-Side Handling Pattern

```typescript
// packages/api-client/src/graphql/mutations.ts

interface MutationResult<T> {
  success: boolean;
  message?: string;
  userErrors: UserError[];
  data?: T;
}

export async function executeMutation<T, V>(
  client: GraphQLClient,
  mutation: string,
  variables: V
): Promise<MutationResult<T>> {
  const result = await client.mutate<{ [key: string]: MutationPayload & T }>(
    mutation,
    variables
  );

  const payload = Object.values(result)[0];

  if (!payload.success || payload.userErrors.length > 0) {
    return {
      success: false,
      message: payload.message,
      userErrors: payload.userErrors,
    };
  }

  return {
    success: true,
    message: payload.message,
    userErrors: [],
    data: payload as T,
  };
}
```

### 2.5 Success Criteria - Phase 2

- [ ] `MutationPayload` interface defined
- [ ] All mutations return `*Payload` types
- [ ] `UserError` type with error codes implemented
- [ ] Batch operations for agents, workflows, documents
- [ ] Client-side mutation helper created
- [ ] Migration guide for existing clients

---

## ⚙️ Phase 3: Type Generation Pipeline (Week 3-4)

### Objective

Implement end-to-end type safety with automated code generation for TypeScript, Rust, and Python.

### 3.1 GraphQL Codegen Configuration

**File:** `packages/graphql-schema/codegen.yml`

```yaml
schema: "./schema/**/*.graphql"
generates:
  # TypeScript Types
  ./generated/typescript/types.ts:
    plugins:
      - typescript
      - typescript-operations
    config:
      strictScalars: true
      scalars:
        UUID: string
        DateTime: string
        JSON: Record<string, unknown>
      enumsAsTypes: false
      avoidOptionals: false
      immutableTypes: true

  # TypeScript React Hooks
  ./generated/typescript/hooks.ts:
    preset: client
    plugins:
      - typescript
      - typescript-operations
      - typescript-react-query
    config:
      fetcher: graphql-request
      exposeQueryKeys: true
      exposeFetcher: true

  # TypeScript Document Nodes
  ./generated/typescript/operations.ts:
    plugins:
      - typed-document-node
    config:
      documentMode: documentNode

  # Schema AST for runtime validation
  ./generated/typescript/schema.json:
    plugins:
      - introspection
```

### 3.2 Rust Type Generation

**File:** `services/rust-core/build.rs`

```rust
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=../../packages/graphql-schema/schema/");

    // Generate Rust types from GraphQL schema
    let output = Command::new("graphql-client")
        .args([
            "generate",
            "--schema-path", "../../packages/graphql-schema/generated/typescript/schema.json",
            "--output", "src/graphql/generated.rs",
            "--custom-scalars-module", "crate::scalars",
        ])
        .output()
        .expect("Failed to generate GraphQL types");

    if !output.status.success() {
        panic!("GraphQL codegen failed: {}", String::from_utf8_lossy(&output.stderr));
    }
}
```

**File:** `services/rust-core/src/scalars.rs`

```rust
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

pub type UUID = Uuid;
pub type DateTimeScalar = DateTime<Utc>;
pub type JSON = serde_json::Value;
```

### 3.3 Python Type Generation

**File:** `services/ml-service/codegen.py`

```python
# Generate Pydantic models from GraphQL schema
from ariadne_codegen import generate

generate(
    schema_path="../../packages/graphql-schema/schema/",
    output_dir="./src/graphql/generated/",
    config={
        "client_name": "NeurectomyClient",
        "async_client": True,
        "generate_pydantic_models": True,
        "scalars": {
            "UUID": "uuid.UUID",
            "DateTime": "datetime.datetime",
            "JSON": "dict[str, Any]",
        },
    },
)
```

### 3.4 CI/CD Integration

**File:** `.github/workflows/graphql-codegen.yml`

```yaml
name: GraphQL Codegen

on:
  push:
    paths:
      - "packages/graphql-schema/schema/**"
  pull_request:
    paths:
      - "packages/graphql-schema/schema/**"

jobs:
  codegen:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: "20"

      - name: Install dependencies
        run: pnpm install

      - name: Validate schema
        run: pnpm --filter @neurectomy/graphql-schema validate

      - name: Generate types
        run: pnpm --filter @neurectomy/graphql-schema codegen

      - name: Check for changes
        run: |
          git diff --exit-code packages/graphql-schema/generated/ || \
          (echo "Generated types are out of date. Run 'pnpm codegen'" && exit 1)
```

### 3.5 Persisted Queries

**File:** `packages/api-client/src/graphql/persisted-queries.ts`

```typescript
import { createHash } from "crypto";

export interface PersistedQuery {
  id: string;
  hash: string;
  document: string;
}

const queries = new Map<string, PersistedQuery>();

export function registerQuery(name: string, document: string): PersistedQuery {
  const hash = createHash("sha256").update(document).digest("hex");
  const query: PersistedQuery = { id: name, hash, document };
  queries.set(name, query);
  return query;
}

export function getPersistedQuery(name: string): PersistedQuery | undefined {
  return queries.get(name);
}

// Pre-register common queries
export const PERSISTED = {
  GetAgent: registerQuery(
    "GetAgent",
    `query GetAgent($id: UUID!) { agent(id: $id) { ...AgentFields } }`
  ),
  ListAgents: registerQuery(
    "ListAgents",
    `query ListAgents($first: Int) { agents(first: $first) { ...ConnectionFields } }`
  ),
  // ... more queries
};
```

### 3.6 Success Criteria - Phase 3

- [ ] `codegen.yml` configured for TypeScript
- [ ] Rust codegen integrated in `build.rs`
- [ ] Python Pydantic models generated
- [ ] CI validates generated types match schema
- [ ] Persisted queries implemented
- [ ] Type mismatches caught at build time

---

## 📡 Phase 4: Enhanced Subscriptions (Week 5)

### Objective

Build production-resilient WebSocket subscriptions with reconnection, filtering, and backpressure handling.

### 4.1 Enhanced Subscription Manager

**File:** `packages/api-client/src/graphql/subscriptions.ts`

```typescript
import { Client, createClient } from "graphql-ws";

export interface SubscriptionManagerConfig {
  url: string;
  connectionParams?: () => Record<string, unknown>;
  maxReconnectAttempts?: number;
  reconnectDelay?: number;
  heartbeatInterval?: number;
}

export class SubscriptionManager {
  private client: Client | null = null;
  private reconnectAttempts = 0;
  private readonly maxReconnectAttempts: number;
  private readonly reconnectDelay: number;
  private subscriptions = new Map<string, () => void>();
  private connectionState: "disconnected" | "connecting" | "connected" =
    "disconnected";

  constructor(private config: SubscriptionManagerConfig) {
    this.maxReconnectAttempts = config.maxReconnectAttempts ?? 5;
    this.reconnectDelay = config.reconnectDelay ?? 1000;
  }

  async connect(): Promise<void> {
    this.connectionState = "connecting";

    this.client = createClient({
      url: this.config.url,
      connectionParams: this.config.connectionParams,
      retryAttempts: this.maxReconnectAttempts,
      retryWait: async (retries) => {
        const delay = Math.min(
          this.reconnectDelay * Math.pow(2, retries),
          30000
        );
        await new Promise((resolve) => setTimeout(resolve, delay));
      },
      on: {
        connected: () => {
          this.connectionState = "connected";
          this.reconnectAttempts = 0;
          this.resubscribeAll();
        },
        closed: () => {
          this.connectionState = "disconnected";
        },
        error: (error) => {
          console.error("WebSocket error:", error);
        },
      },
    });
  }

  subscribe<TData>(
    id: string,
    query: string,
    variables: Record<string, unknown>,
    handlers: {
      onData: (data: TData) => void;
      onError?: (error: Error) => void;
      onComplete?: () => void;
    }
  ): () => void {
    if (!this.client) {
      throw new Error("SubscriptionManager not connected");
    }

    const unsubscribe = this.client.subscribe(
      { query, variables },
      {
        next: (result) => {
          if (result.data) handlers.onData(result.data as TData);
        },
        error: (err) =>
          handlers.onError?.(
            err instanceof Error ? err : new Error(String(err))
          ),
        complete: () => handlers.onComplete?.(),
      }
    );

    this.subscriptions.set(id, unsubscribe);
    return () => {
      unsubscribe();
      this.subscriptions.delete(id);
    };
  }

  private resubscribeAll(): void {
    // Resubscribe logic after reconnection
  }

  dispose(): void {
    this.subscriptions.forEach((unsub) => unsub());
    this.subscriptions.clear();
    this.client?.dispose();
  }
}
```

### 4.2 Enhanced Subscription Types

**File:** `packages/graphql-schema/schema/subscriptions/enhanced.graphql`

```graphql
"""
Subscription filter options.
"""
input SubscriptionFilter {
  "Filter by log level"
  level: LogLevel
  "Filter by text content"
  contains: String
  "Only events after this time"
  since: DateTime
}

"""
Throttle strategy for high-volume subscriptions.
"""
enum ThrottleStrategy {
  "Drop oldest events when buffer full"
  DROP_OLDEST
  "Drop newest events when buffer full"
  DROP_NEWEST
  "Sample events at interval"
  SAMPLE
  "No throttling (default)"
  NONE
}

"""
Enhanced agent logs subscription with filtering.
"""
type Subscription {
  agentLogs(
    agentId: UUID!
    filter: SubscriptionFilter
    throttle: ThrottleStrategy = NONE
    bufferSize: Int = 100
  ): AgentLogEvent!

  """
  Subscribe to connection state changes.
  """
  connectionState: ConnectionStateEvent!

  """
  Subscribe with automatic reconnection tracking.
  """
  agentUpdatesReliable(
    agentId: UUID!
    "Resume from this event ID after reconnection"
    lastEventId: String
  ): AgentUpdateEvent!
}

type ConnectionStateEvent {
  status: ConnectionStatus!
  connectedAt: DateTime
  lastHeartbeat: DateTime
  subscriptionCount: Int!
  reconnectAttempts: Int!
}

enum ConnectionStatus {
  CONNECTED
  CONNECTING
  DISCONNECTED
  RECONNECTING
  FAILED
}

type AgentUpdateEvent {
  "Unique event ID for resumption"
  eventId: String!
  "Event timestamp"
  timestamp: DateTime!
  "The updated agent data"
  agent: Agent!
  "Type of update"
  updateType: AgentUpdateType!
}

enum AgentUpdateType {
  STATUS_CHANGE
  CONFIG_UPDATE
  METRICS_UPDATE
  ERROR
}
```

### 4.3 Success Criteria - Phase 4

- [ ] `SubscriptionManager` class with auto-reconnect
- [ ] Exponential backoff implemented
- [ ] Subscription filtering (level, contains, since)
- [ ] Throttle strategies for high-volume streams
- [ ] Connection state subscription
- [ ] Event ID-based resumption after reconnect
- [ ] Heartbeat/keepalive implemented

---

## 🔐 Phase 5: Security Directives (Week 6)

### Objective

Implement field-level authorization, rate limiting, and query complexity analysis.

### 5.1 Custom Directives

**File:** `packages/graphql-schema/schema/directives.graphql`

```graphql
"""
Requires authentication to access field or type.
"""
directive @auth(
  "Required role to access"
  requires: UserRole = USER
) on FIELD_DEFINITION | OBJECT

"""
Rate limit for field or mutation.
"""
directive @rateLimit(
  "Maximum requests allowed"
  max: Int!
  "Time window (e.g., '1m', '1h', '1d')"
  window: String!
  "Rate limit scope"
  scope: RateLimitScope = USER
) on FIELD_DEFINITION

"""
Query complexity cost for field.
"""
directive @complexity(
  "Base complexity cost"
  value: Int!
  "Fields that multiply the cost"
  multipliers: [String!]
) on FIELD_DEFINITION

"""
Marks a field as deprecated with migration info.
"""
directive @deprecated(
  "Reason for deprecation"
  reason: String!
  "Removal version"
  removeIn: String
  "Replacement field"
  replacedBy: String
) on FIELD_DEFINITION | ENUM_VALUE

"""
Audit logging for sensitive operations.
"""
directive @audit(
  "Log level for this operation"
  level: AuditLevel = INFO
) on FIELD_DEFINITION

enum RateLimitScope {
  USER
  IP
  API_KEY
  GLOBAL
}

enum AuditLevel {
  DEBUG
  INFO
  WARN
  CRITICAL
}
```

### 5.2 Applying Security Directives

```graphql
type Query {
  # Public - no auth required
  systemHealth: SystemHealth!

  # Requires authentication
  me: User! @auth

  # Requires specific role
  users: UserConnection! @auth(requires: ADMIN)

  # With rate limiting
  agents(first: Int, after: String): AgentConnection!
    @auth
    @rateLimit(max: 100, window: "1m")
    @complexity(value: 5, multipliers: ["first"])
}

type Mutation {
  # Standard auth
  createAgent(input: CreateAgentInput!): CreateAgentPayload!
    @auth
    @rateLimit(max: 10, window: "1m")
    @audit(level: INFO)

  # Admin only with strict rate limit
  deleteUser(id: UUID!): DeleteUserPayload!
    @auth(requires: SUPERADMIN)
    @rateLimit(max: 5, window: "1h", scope: GLOBAL)
    @audit(level: CRITICAL)
}

type User {
  id: UUID!
  email: String! @auth # Only visible to authenticated users
  # Only visible to admins
  apiKeys: [ApiKey!]! @auth(requires: ADMIN)

  # Deprecated field
  config: JSON
    @deprecated(
      reason: "Use settings field instead"
      removeIn: "v2.0"
      replacedBy: "settings"
    )
  settings: UserSettings!
}
```

### 5.3 Complexity Analysis Implementation

```typescript
// packages/api-server/src/graphql/complexity.ts

import {
  getComplexity,
  simpleEstimator,
  fieldExtensionsEstimator,
} from "graphql-query-complexity";

export const MAX_COMPLEXITY = 1000;

export function calculateComplexity(
  schema: GraphQLSchema,
  query: DocumentNode,
  variables: Record<string, unknown>
): number {
  return getComplexity({
    schema,
    query,
    variables,
    estimators: [
      fieldExtensionsEstimator(),
      simpleEstimator({ defaultComplexity: 1 }),
    ],
  });
}

export function validateQueryComplexity(
  schema: GraphQLSchema,
  query: DocumentNode,
  variables: Record<string, unknown>
): void {
  const complexity = calculateComplexity(schema, query, variables);

  if (complexity > MAX_COMPLEXITY) {
    throw new GraphQLError(
      `Query complexity ${complexity} exceeds maximum ${MAX_COMPLEXITY}`,
      {
        extensions: {
          code: "QUERY_TOO_COMPLEX",
          complexity,
          max: MAX_COMPLEXITY,
        },
      }
    );
  }
}
```

### 5.4 Success Criteria - Phase 5

- [ ] `@auth` directive with role-based access
- [ ] `@rateLimit` directive with configurable windows
- [ ] `@complexity` directive for cost analysis
- [ ] Query complexity validation middleware
- [ ] `@audit` directive for sensitive operations
- [ ] `@deprecated` directive with migration info
- [ ] Security documentation updated

---

## 📊 Progress Tracking

### Phase Completion Checklist

| Phase | Description              | Status         | Target   | Completion |
| ----- | ------------------------ | -------------- | -------- | ---------- |
| 1     | Interface Abstractions   | 🔴 Not Started | Week 1   | 0%         |
| 2     | Unified Mutations        | 🔴 Not Started | Week 2   | 0%         |
| 3     | Type Generation Pipeline | 🔴 Not Started | Week 3-4 | 0%         |
| 4     | Enhanced Subscriptions   | 🔴 Not Started | Week 5   | 0%         |
| 5     | Security Directives      | 🔴 Not Started | Week 6   | 0%         |

### GraphQL API Score Targets

| Dimension         | Current | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 |
| ----------------- | ------- | ------- | ------- | ------- | ------- | ------- |
| Schema Design     | 7/10    | 9/10    | 9/10    | 9/10    | 9/10    | 9/10    |
| Type Safety       | 6/10    | 7/10    | 8/10    | 10/10   | 10/10   | 10/10   |
| Mutation Patterns | 5/10    | 5/10    | 9/10    | 9/10    | 9/10    | 9/10    |
| Subscriptions     | 7/10    | 7/10    | 7/10    | 7/10    | 9/10    | 9/10    |
| Security          | 6/10    | 6/10    | 6/10    | 6/10    | 6/10    | 9/10    |
| DX                | 7/10    | 8/10    | 9/10    | 9/10    | 9/10    | 10/10   |
| **TOTAL**         | **6.3** | **7.0** | **8.0** | **8.3** | **8.7** | **9.3** |

---

## 🛠️ Quick Reference: Commands

```bash
# Schema Management
pnpm --filter @neurectomy/graphql-schema validate    # Validate schema
pnpm --filter @neurectomy/graphql-schema codegen     # Generate types
pnpm --filter @neurectomy/graphql-schema codegen:watch  # Watch mode

# Testing
pnpm --filter @neurectomy/api-client test:graphql    # Test GraphQL client
pnpm --filter @neurectomy/api-server test:resolvers  # Test resolvers

# Development
pnpm --filter @neurectomy/api-server dev             # Start dev server
pnpm --filter @neurectomy/graphql-schema introspect  # Generate schema.json
```

---

## 📞 Support & Questions

- **Schema Design:** @ARCHITECT agent
- **API Patterns:** @SYNAPSE agent
- **Type Generation:** @APEX agent
- **Security:** @CIPHER agent
- **Testing:** @ECLIPSE agent

---

**Document End**

_Last Updated: December 5, 2025_  
_Next Review: December 19, 2025_
