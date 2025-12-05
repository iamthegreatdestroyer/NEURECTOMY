# @neurectomy/schema-codegen

Generate JSON schemas and TypeScript type guards from Zod schemas.

## Features

- Convert Zod schemas to JSON Schema format
- Generate TypeScript type guards and parse functions
- CLI for schema generation and validation
- Programmatic API for custom workflows

## Installation

This package is internal to the NEURECTOMY monorepo.

## CLI Usage

### Generate Schemas

```bash
# Generate all schemas to ./generated
pnpm neurectomy-codegen generate

# Generate to custom directory
pnpm neurectomy-codegen generate -o ./src/schemas

# Generate only JSON schemas
pnpm neurectomy-codegen generate -f json

# Generate only TypeScript guards
pnpm neurectomy-codegen generate -f typescript
```

### Validate Files

```bash
# Validate an agent config file
pnpm neurectomy-codegen validate agentConfig ./config/my-agent.json

# Validate a workflow file
pnpm neurectomy-codegen validate workflow ./workflows/pipeline.json
```

## Programmatic API

```typescript
import {
  generateSchemas,
  generateJsonSchema,
} from "@neurectomy/schema-codegen";
import { z } from "zod";

// Define custom schema
const mySchema = z.object({
  name: z.string(),
  value: z.number(),
});

// Generate JSON schema
const jsonSchema = generateJsonSchema(mySchema, "MySchema");

// Generate multiple schemas to files
await generateSchemas({
  outputDir: "./generated",
  schemas: {
    mySchema,
  },
  format: "both",
  prettyPrint: true,
});
```

## Generated Files

The generator creates the following files:

- `{schemaName}.schema.json` - JSON Schema file
- `{schemaName}.guards.ts` - TypeScript type guards
- `index.ts` - Barrel export file

### Example Generated Type Guard

```typescript
/**
 * Type guard for AgentConfig
 */
export function isAgentConfig(value: unknown): value is AgentConfig {
  const result = AgentConfigSchema.safeParse(value);
  return result.success;
}

/**
 * Parse and validate AgentConfig
 */
export function parseAgentConfig(value: unknown): AgentConfig {
  return AgentConfigSchema.parse(value);
}

/**
 * Safe parse AgentConfig (returns result object)
 */
export function safeParseAgentConfig(
  value: unknown
): z.SafeParseReturnType<unknown, AgentConfig> {
  return AgentConfigSchema.safeParse(value);
}
```

## Available Schemas

From `@neurectomy/core`:

- `agentConfig` - Agent configuration schema
- `taskDefinition` - Task definition schema
- `workflow` - Workflow definition schema
