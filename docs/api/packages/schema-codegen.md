# @neurectomy/schema-codegen API Reference

Generate JSON schemas and TypeScript type guards from Zod schemas.

## Installation

This package is internal to the NEURECTOMY monorepo.

```bash
# Install in a workspace package
pnpm add @neurectomy/schema-codegen
```

## CLI Reference

### neurectomy-codegen generate

Generate schemas from Zod definitions.

```bash
neurectomy-codegen generate [options]
```

**Options:**

| Option     | Alias | Description                                 | Default       |
| ---------- | ----- | ------------------------------------------- | ------------- |
| `--output` | `-o`  | Output directory                            | `./generated` |
| `--format` | `-f`  | Output format: `json`, `typescript`, `both` | `both`        |
| `--pretty` | `-p`  | Pretty print JSON                           | `true`        |

**Examples:**

```bash
# Generate all schemas to ./generated
pnpm neurectomy-codegen generate

# Custom output directory
pnpm neurectomy-codegen generate -o ./src/schemas

# JSON only
pnpm neurectomy-codegen generate -f json

# TypeScript only
pnpm neurectomy-codegen generate -f typescript

# Minified JSON
pnpm neurectomy-codegen generate --no-pretty
```

---

### neurectomy-codegen validate

Validate a JSON file against a schema.

```bash
neurectomy-codegen validate <schema> <file>
```

**Arguments:**

| Argument | Description                                              |
| -------- | -------------------------------------------------------- |
| `schema` | Schema name: `agentConfig`, `taskDefinition`, `workflow` |
| `file`   | Path to JSON file to validate                            |

**Examples:**

```bash
# Validate agent configuration
pnpm neurectomy-codegen validate agentConfig ./config/agent.json

# Validate workflow definition
pnpm neurectomy-codegen validate workflow ./workflows/pipeline.json

# Validate task definition
pnpm neurectomy-codegen validate taskDefinition ./tasks/process.json
```

**Exit Codes:**

| Code | Meaning               |
| ---- | --------------------- |
| `0`  | Validation successful |
| `1`  | Validation failed     |

---

## Programmatic API

### generateJsonSchema(schema, name)

Convert a Zod schema to JSON Schema format.

```typescript
import { generateJsonSchema } from "@neurectomy/schema-codegen";
import { z } from "zod";

const mySchema = z.object({
  name: z.string().min(1),
  count: z.number().int().positive(),
  tags: z.array(z.string()).optional(),
});

const jsonSchema = generateJsonSchema(mySchema, "MyType");
```

**Parameters:**

| Param    | Type        | Description                           |
| -------- | ----------- | ------------------------------------- |
| `schema` | `z.ZodType` | Zod schema to convert                 |
| `name`   | `string`    | Schema name (used in JSON Schema $id) |

**Returns:** `object` - JSON Schema object

**Output Example:**

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "MyType",
  "type": "object",
  "properties": {
    "name": { "type": "string", "minLength": 1 },
    "count": { "type": "integer", "exclusiveMinimum": 0 },
    "tags": { "type": "array", "items": { "type": "string" } }
  },
  "required": ["name", "count"]
}
```

---

### generateTypeGuard(schema, typeName)

Generate TypeScript type guard functions.

```typescript
import { generateTypeGuard } from "@neurectomy/schema-codegen";
import { z } from "zod";

const schema = z.object({ name: z.string() });
const code = generateTypeGuard(schema, "MyType");
```

**Parameters:**

| Param      | Type        | Description                       |
| ---------- | ----------- | --------------------------------- |
| `schema`   | `z.ZodType` | Zod schema                        |
| `typeName` | `string`    | TypeScript type name (PascalCase) |

**Returns:** `string` - TypeScript code

**Output Example:**

```typescript
/**
 * Type guard for MyType
 */
export function isMyType(value: unknown): value is MyType {
  const result = MyTypeSchema.safeParse(value);
  return result.success;
}

/**
 * Parse and validate MyType
 */
export function parseMyType(value: unknown): MyType {
  return MyTypeSchema.parse(value);
}

/**
 * Safe parse MyType (returns result object)
 */
export function safeParseMyType(
  value: unknown
): z.SafeParseReturnType<unknown, MyType> {
  return MyTypeSchema.safeParse(value);
}
```

---

### generateSchemas(options)

Generate multiple schemas to files.

```typescript
import { generateSchemas } from "@neurectomy/schema-codegen";
import { z } from "zod";

const userSchema = z.object({
  id: z.string().uuid(),
  email: z.string().email(),
  name: z.string(),
});

const configSchema = z.object({
  apiUrl: z.string().url(),
  timeout: z.number().default(5000),
});

const results = await generateSchemas({
  outputDir: "./generated",
  schemas: {
    user: userSchema,
    config: configSchema,
  },
  format: "both",
  prettyPrint: true,
});
```

**Parameters:**

```typescript
interface SchemaGeneratorOptions {
  outputDir: string; // Output directory path
  schemas: Record<string, z.ZodType>; // Map of schema name to Zod schema
  format?: "json" | "typescript" | "both"; // Output format (default: 'both')
  prettyPrint?: boolean; // Pretty print JSON (default: true)
}
```

**Returns:** `Promise<GeneratedSchema[]>`

```typescript
interface GeneratedSchema {
  name: string; // Schema name
  jsonSchema: object; // Generated JSON Schema
  typescript?: string; // Generated TypeScript code (if format includes TS)
}
```

**Generated Files:**

| File Pattern         | Description            |
| -------------------- | ---------------------- |
| `{name}.schema.json` | JSON Schema file       |
| `{name}.guards.ts`   | TypeScript type guards |
| `index.ts`           | Barrel export file     |

---

## Type Exports

### SchemaGeneratorOptions

```typescript
interface SchemaGeneratorOptions {
  outputDir: string;
  schemas: Record<string, z.ZodType>;
  format?: "json" | "typescript" | "both";
  prettyPrint?: boolean;
}
```

### GeneratedSchema

```typescript
interface GeneratedSchema {
  name: string;
  jsonSchema: object;
  typescript?: string;
}
```

---

## Re-exports

The package re-exports commonly used utilities:

```typescript
// Re-exported from zod
export { z } from "zod";

// Re-exported from zod-to-json-schema
export { zodToJsonSchema } from "zod-to-json-schema";
```

---

## Usage Examples

### Basic Schema Generation

```typescript
import { generateSchemas, z } from "@neurectomy/schema-codegen";

const agentSchema = z.object({
  id: z.string(),
  name: z.string().min(1).max(100),
  status: z.enum(["idle", "running", "error"]),
  capabilities: z.array(z.string()),
  config: z.record(z.unknown()).optional(),
});

await generateSchemas({
  outputDir: "./src/generated",
  schemas: { agent: agentSchema },
  format: "both",
});

// Creates:
// - ./src/generated/agent.schema.json
// - ./src/generated/agent.guards.ts
// - ./src/generated/index.ts
```

### Using Generated Guards

```typescript
// After generation
import { isAgent, parseAgent, safeParseAgent } from "./generated";

// Type guard
if (isAgent(data)) {
  console.log(data.name); // TypeScript knows this is valid
}

// Strict parsing (throws on invalid)
try {
  const agent = parseAgent(input);
} catch (error) {
  console.error("Invalid agent:", error);
}

// Safe parsing (returns result)
const result = safeParseAgent(input);
if (result.success) {
  console.log(result.data);
} else {
  console.error(result.error.issues);
}
```

### Custom JSON Schema Options

```typescript
import { zodToJsonSchema } from "@neurectomy/schema-codegen";
import { z } from "zod";

const schema = z.object({ name: z.string() });

// With custom options
const jsonSchema = zodToJsonSchema(schema, {
  name: "CustomSchema",
  $refStrategy: "root",
  target: "openApi3",
  errorMessages: true,
});
```

### Validating External Files

```typescript
import { z } from "zod";
import * as fs from "fs/promises";

const configSchema = z.object({
  apiKey: z.string(),
  baseUrl: z.string().url(),
});

async function validateConfigFile(path: string) {
  const content = await fs.readFile(path, "utf-8");
  const data = JSON.parse(content);

  const result = configSchema.safeParse(data);

  if (!result.success) {
    console.error("Validation errors:");
    result.error.issues.forEach((issue) => {
      console.error(`  - ${issue.path.join(".")}: ${issue.message}`);
    });
    return null;
  }

  return result.data;
}
```

---

## Dependencies

- `zod` ^3.22.0 - Schema definition
- `zod-to-json-schema` ^3.22.0 - JSON Schema conversion
- `commander` ^12.0.0 - CLI framework

## Peer Dependencies

None - all dependencies are bundled.
