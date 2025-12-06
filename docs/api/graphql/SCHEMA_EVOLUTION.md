# Schema Evolution & Migration Guide

> **@SCRIBE + @ARCHITECT Implementation**
> Comprehensive documentation for NEURECTOMY GraphQL schema versioning, migration strategies, and governance best practices.

## Table of Contents

1. [Overview](#overview)
2. [Versioning Strategy](#versioning-strategy)
3. [Deprecation Lifecycle](#deprecation-lifecycle)
4. [Migration Process](#migration-process)
5. [Breaking Changes](#breaking-changes)
6. [Compatibility Checking](#compatibility-checking)
7. [Best Practices](#best-practices)
8. [API Reference](#api-reference)
9. [Troubleshooting](#troubleshooting)

---

## Overview

NEURECTOMY employs a robust schema governance system that enables:

- **Semantic Versioning** for GraphQL schemas
- **Graceful Deprecation** with sunset periods
- **Backward Compatibility** validation
- **Automated Migration** tooling
- **Breaking Change Detection**

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Schema Governance System                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │   Schema    │───▶│  Schema         │───▶│  Migration      │ │
│  │   Registry  │    │  Validator      │    │  Tools          │ │
│  └─────────────┘    └─────────────────┘    └─────────────────┘ │
│         │                   │                       │           │
│         ▼                   ▼                       ▼           │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │ Deprecation │    │  Compatibility  │    │  Changelog      │ │
│  │ Tracker     │    │  Reports        │    │  Generator      │ │
│  └─────────────┘    └─────────────────┘    └─────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Versioning Strategy

### Semantic Versioning for GraphQL

NEURECTOMY uses a modified semantic versioning scheme:

```
MAJOR.MINOR.PATCH[-PRERELEASE]

Examples:
- 1.0.0       (Initial stable release)
- 1.1.0       (New fields added, backward compatible)
- 1.1.1       (Bug fixes, documentation updates)
- 2.0.0       (Breaking changes)
- 2.0.0-beta.1 (Pre-release version)
```

#### Version Bump Rules

| Change Type            | Version Bump | Example               |
| ---------------------- | ------------ | --------------------- |
| Breaking change        | MAJOR        | Removing a field      |
| New feature (additive) | MINOR        | Adding a new query    |
| Bug fix                | PATCH        | Fixing resolver logic |
| Deprecation notice     | PATCH        | Adding @deprecated    |

### Registering Schema Versions

```typescript
import { SchemaRegistry, createSchemaRegistry } from "@neurectomy/api-client";

const registry = createSchemaRegistry({
  currentVersion: "1.2.0",
  supportedVersions: ["1.0.0", "1.1.0", "1.2.0"],
  deprecationPolicy: {
    warningPeriod: 30, // days
    sunsetPeriod: 90, // days
  },
});

// Register a new version
await registry.registerVersion({
  version: "1.3.0",
  releaseDate: new Date("2025-01-15"),
  changelog: [
    { type: "ADDED", description: "New WorkflowInstance type" },
    { type: "DEPRECATED", description: "Legacy workflow fields" },
  ],
  schema: schemaSDL,
});
```

---

## Deprecation Lifecycle

### Deprecation States

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   ACTIVE     │────▶│   WARNING    │────▶│   SUNSET     │
│              │     │   (30 days)  │     │   (90 days)  │
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
                                                 ▼
                                          ┌──────────────┐
                                          │   REMOVED    │
                                          └──────────────┘
```

### Marking Fields Deprecated

In your GraphQL schema:

```graphql
type Agent {
  id: ID!
  name: String!

  # Deprecated field with migration guidance
  status: AgentStatus
    @deprecated(
      reason: "Use `state` field instead for more detailed status information"
    )

  # New field replacing deprecated one
  state: AgentState!
}
```

### Tracking Deprecation Usage

```typescript
import {
  DeprecationTracker,
  createDeprecationTracker,
} from "@neurectomy/api-client";

const tracker = createDeprecationTracker({
  reportingEndpoint: "/api/deprecation-usage",
  sampleRate: 0.1, // Report 10% of requests
  enableWarnings: true,
});

// Track field usage
tracker.trackFieldUsage({
  field: "Agent.status",
  operationName: "GetAgentDetails",
  clientVersion: "1.2.0",
});

// Get deprecation report
const report = tracker.getUsageReport();
console.log(report);
// {
//   'Agent.status': {
//     usageCount: 1250,
//     lastUsed: '2025-12-01T10:30:00Z',
//     uniqueClients: 15,
//     deprecatedSince: '2025-09-01',
//     sunsetDate: '2025-12-01',
//     daysTillSunset: -4
//   }
// }
```

### Client Notifications

Deprecation warnings are communicated through:

1. **GraphQL Response Extensions**

```json
{
  "data": { ... },
  "extensions": {
    "deprecations": [
      {
        "field": "Agent.status",
        "reason": "Use `state` field instead",
        "sunsetDate": "2025-12-01"
      }
    ]
  }
}
```

2. **HTTP Headers**

```
Deprecation: true
Sunset: Sat, 01 Dec 2025 00:00:00 GMT
Link: </docs/migration/agent-status>; rel="deprecation"
```

---

## Migration Process

### Step-by-Step Migration

#### 1. Analyze Current Usage

```typescript
import { createMigrationAnalyzer } from "@neurectomy/api-client";

const analyzer = createMigrationAnalyzer(registry);

// Analyze client queries against schema changes
const analysis = await analyzer.analyzeClientQueries(clientQueries, {
  fromVersion: "1.2.0",
  toVersion: "2.0.0",
});

console.log(analysis);
// {
//   breakingChanges: [
//     { type: 'FIELD_REMOVED', path: 'Agent.status', affectedQueries: 12 }
//   ],
//   deprecationWarnings: [...],
//   recommendations: [...]
// }
```

#### 2. Generate Migration Guide

```typescript
const guide = await registry.generateMigrationGuide({
  fromVersion: "1.2.0",
  toVersion: "2.0.0",
  format: "markdown",
});

// Outputs comprehensive migration documentation
console.log(guide);
```

#### 3. Apply Schema Transformations

```typescript
import { createSchemaMigrator } from "@neurectomy/api-client";

const migrator = createSchemaMigrator({
  transformations: [
    {
      type: "RENAME_FIELD",
      from: "Agent.status",
      to: "Agent.state",
      coercion: (status) => mapStatusToState(status),
    },
    {
      type: "ADD_REQUIRED_FIELD",
      field: "Agent.version",
      defaultValue: "1.0.0",
    },
  ],
});

// Transform queries automatically
const transformedQuery = migrator.transformQuery(oldQuery);
```

### Automated Query Rewriting

For seamless client upgrades, NEURECTOMY supports automatic query rewriting:

```typescript
const rewriter = createQueryRewriter({
  rules: [
    {
      match: { field: "Agent.status" },
      replace: { field: "Agent.state", transform: statusToStateTransform },
    },
  ],
});

// Enable in your GraphQL server
app.use(
  "/graphql",
  graphqlHTTP({
    schema,
    extensions: ({ document }) => ({
      rewrittenQuery: rewriter.rewrite(document),
    }),
  })
);
```

---

## Breaking Changes

### Types of Breaking Changes

| Category             | Example                       | Severity  |
| -------------------- | ----------------------------- | --------- |
| Field Removal        | Removing `Agent.status`       | 🔴 High   |
| Type Change          | `count: Int` → `count: Float` | 🔴 High   |
| Required Field Added | New required argument         | 🟠 Medium |
| Enum Value Removed   | Removing `PENDING` status     | 🔴 High   |
| Interface Change     | Removing interface field      | 🔴 High   |
| Union Change         | Removing union member         | 🟠 Medium |

### Breaking Change Detection

```typescript
import { detectBreakingChanges } from "@neurectomy/api-client";

const changes = detectBreakingChanges(oldSchema, newSchema);

changes.forEach((change) => {
  console.log(`${change.severity}: ${change.description}`);
  console.log(`  Path: ${change.path}`);
  console.log(`  Migration: ${change.migrationHint}`);
});
```

### Safe Schema Evolution Patterns

#### ✅ Safe: Adding Optional Fields

```graphql
# v1.0.0
type Agent {
  id: ID!
  name: String!
}

# v1.1.0 - Safe addition
type Agent {
  id: ID!
  name: String!
  description: String # New optional field
}
```

#### ✅ Safe: Adding New Types

```graphql
# v1.1.0 - Safe addition
type AgentMetrics {
  cpu: Float!
  memory: Float!
}

type Agent {
  id: ID!
  metrics: AgentMetrics # New optional field with new type
}
```

#### ⚠️ Caution: Changing Nullability

```graphql
# Changing from nullable to non-null is BREAKING
name: String   # v1.0.0
name: String!  # v2.0.0 - BREAKING

# Changing from non-null to nullable is SAFE
name: String!  # v1.0.0
name: String   # v1.1.0 - Safe (more permissive)
```

---

## Compatibility Checking

### Compatibility Levels

| Level      | Description                                 |
| ---------- | ------------------------------------------- |
| `NONE`     | No compatibility - breaking changes present |
| `BACKWARD` | New schema accepts old queries              |
| `FORWARD`  | Old schema accepts new queries              |
| `FULL`     | Bidirectional compatibility                 |

### Running Compatibility Checks

```typescript
import { checkSchemaCompatibility } from "@neurectomy/api-client";

const report = await checkSchemaCompatibility({
  baseSchema: oldSchemaSDL,
  compareSchema: newSchemaSDL,
  strictMode: true,
});

console.log(report);
// {
//   compatible: false,
//   level: 'NONE',
//   breakingChanges: [...],
//   warnings: [...],
//   suggestions: [...]
// }
```

### CI/CD Integration

```yaml
# .github/workflows/schema-check.yml
name: Schema Compatibility Check

on:
  pull_request:
    paths:
      - "packages/graphql-schema/**"

jobs:
  check-compatibility:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Check Schema Compatibility
        run: |
          npx neurectomy-schema check \
            --base main \
            --compare HEAD \
            --fail-on-breaking
```

---

## Best Practices

### 1. Version Early, Version Often

```typescript
// Register schema changes immediately
registry.registerVersion({
  version: incrementVersion(currentVersion, "minor"),
  changelog: currentChanges,
  releaseDate: new Date(),
});
```

### 2. Use Additive Changes

```graphql
# Instead of modifying existing fields, add new ones
type Agent {
  # Keep the old field (deprecated)
  status: AgentStatus @deprecated(reason: "Use stateV2")

  # Add new field with enhanced functionality
  stateV2: AgentStateV2!
}
```

### 3. Set Reasonable Deprecation Periods

```typescript
const deprecationPolicy = {
  // Minimum time to warn users
  warningPeriod: 30, // days

  // Total time before removal
  sunsetPeriod: 90, // days

  // Notify these teams
  notificationChannels: ["#api-consumers", "#frontend"],
};
```

### 4. Document Everything

```graphql
"""
Agent represents an AI entity in the NEURECTOMY system.

@since 1.0.0
@see AgentState for status information
"""
type Agent {
  """
  Unique identifier for the agent.
  Format: UUID v4
  """
  id: ID!

  """
  Human-readable name of the agent.
  Maximum length: 100 characters.
  """
  name: String!
}
```

### 5. Test Migration Paths

```typescript
describe("Schema Migration", () => {
  it("should migrate v1 queries to v2", async () => {
    const v1Query = `query { agent(id: "1") { status } }`;
    const migrated = migrator.migrate(v1Query, "v1", "v2");

    expect(migrated).toContain("state");
    expect(migrated).not.toContain("status");
  });
});
```

---

## API Reference

### SchemaRegistry

```typescript
interface SchemaRegistry {
  // Version management
  registerVersion(version: VersionInfo): Promise<void>;
  getVersion(version: string): VersionInfo | undefined;
  getCurrentVersion(): string;
  getSupportedVersions(): string[];

  // Compatibility
  checkCompatibility(from: string, to: string): CompatibilityReport;

  // Migration
  generateMigrationGuide(options: MigrationOptions): Promise<MigrationGuide>;

  // Changelog
  getChangelog(options?: ChangelogOptions): ChangelogEntry[];
}
```

### DeprecationTracker

```typescript
interface DeprecationTracker {
  // Tracking
  trackFieldUsage(usage: FieldUsage): void;
  trackOperationUsage(usage: OperationUsage): void;

  // Reporting
  getUsageReport(): DeprecationReport;
  getFieldReport(field: string): FieldReport;

  // Lifecycle
  markDeprecated(field: string, options: DeprecationOptions): void;
  markSunset(field: string): void;
  markRemoved(field: string): void;
}
```

### MigrationTools

```typescript
interface MigrationTools {
  // Analysis
  analyzeBreakingChanges(from: Schema, to: Schema): BreakingChange[];
  analyzeClientImpact(
    queries: string[],
    changes: BreakingChange[]
  ): ImpactReport;

  // Transformation
  transformQuery(query: string, transformations: Transformation[]): string;
  generateCodemods(changes: BreakingChange[]): Codemod[];

  // Validation
  validateMigration(from: string, to: string): ValidationResult;
}
```

---

## Troubleshooting

### Common Issues

#### "Schema version not found"

```typescript
// Ensure version is registered before use
if (!registry.getVersion(targetVersion)) {
  await registry.registerVersion({
    version: targetVersion,
    schema: schemaSDL,
    releaseDate: new Date(),
  });
}
```

#### "Breaking change detected in CI"

1. Check if the change is intentional
2. If intentional, bump MAJOR version
3. Add migration guide
4. Update deprecation notices

```bash
# Allow breaking changes with explicit flag
npx neurectomy-schema check --allow-breaking --reason "Planned v2 migration"
```

#### "Deprecation warnings not showing"

```typescript
// Ensure tracker is properly configured
const tracker = createDeprecationTracker({
  enableWarnings: true,
  warningHeader: "X-Deprecation-Warning",
  responseExtensions: true,
});

// Register with your GraphQL server
server.use(tracker.middleware());
```

### Debug Mode

```typescript
// Enable debug logging
const registry = createSchemaRegistry({
  debug: true,
  logLevel: "verbose",
  onEvent: (event) => console.log("[Schema]", event),
});
```

---

## Resources

- [GraphQL Schema Design Best Practices](https://graphql.org/learn/best-practices/)
- [Principled GraphQL](https://principledgraphql.com/)
- [Apollo Schema Changelog](https://www.apollographql.com/docs/graphos/schema-management/)
- [NEURECTOMY API Documentation](/docs/api/graphql/)

---

_Last updated: December 2025_
_Schema Governance Version: 1.0.0_
