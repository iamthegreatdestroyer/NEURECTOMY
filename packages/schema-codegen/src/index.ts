import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";
import * as fs from "fs/promises";
import * as path from "path";

export interface SchemaGeneratorOptions {
  outputDir: string;
  schemas: Record<string, z.ZodType>;
  format?: "json" | "typescript" | "both";
  prettyPrint?: boolean;
}

export interface GeneratedSchema {
  name: string;
  jsonSchema: object;
  typescript?: string;
}

/**
 * Generate JSON schemas from Zod schemas.
 */
export function generateJsonSchema(schema: z.ZodType, name: string): object {
  return zodToJsonSchema(schema, {
    name,
    $refStrategy: "none",
    target: "jsonSchema7",
  });
}

/**
 * Generate TypeScript type guard from Zod schema.
 */
export function generateTypeGuard(schema: z.ZodType, typeName: string): string {
  return `
/**
 * Type guard for ${typeName}
 */
export function is${typeName}(value: unknown): value is ${typeName} {
  const result = ${typeName}Schema.safeParse(value);
  return result.success;
}

/**
 * Parse and validate ${typeName}
 */
export function parse${typeName}(value: unknown): ${typeName} {
  return ${typeName}Schema.parse(value);
}

/**
 * Safe parse ${typeName} (returns result object)
 */
export function safeParse${typeName}(value: unknown): z.SafeParseReturnType<unknown, ${typeName}> {
  return ${typeName}Schema.safeParse(value);
}
`.trim();
}

/**
 * Generate all schemas to files.
 */
export async function generateSchemas(
  options: SchemaGeneratorOptions
): Promise<GeneratedSchema[]> {
  const { outputDir, schemas, format = "both", prettyPrint = true } = options;

  await fs.mkdir(outputDir, { recursive: true });

  const results: GeneratedSchema[] = [];

  for (const [name, schema] of Object.entries(schemas)) {
    const jsonSchema = generateJsonSchema(schema, name);
    const result: GeneratedSchema = { name, jsonSchema };

    // Write JSON schema
    if (format === "json" || format === "both") {
      const jsonPath = path.join(outputDir, `${name}.schema.json`);
      const jsonContent = prettyPrint
        ? JSON.stringify(jsonSchema, null, 2)
        : JSON.stringify(jsonSchema);
      await fs.writeFile(jsonPath, jsonContent, "utf-8");
    }

    // Generate TypeScript helpers
    if (format === "typescript" || format === "both") {
      const tsContent = generateTypeGuard(schema, pascalCase(name));
      result.typescript = tsContent;

      const tsPath = path.join(outputDir, `${name}.guards.ts`);
      await fs.writeFile(tsPath, tsContent, "utf-8");
    }

    results.push(result);
  }

  // Generate index file
  if (format === "typescript" || format === "both") {
    const indexContent = Object.keys(schemas)
      .map((name) => `export * from './${name}.guards';`)
      .join("\n");
    await fs.writeFile(path.join(outputDir, "index.ts"), indexContent, "utf-8");
  }

  return results;
}

/**
 * Convert string to PascalCase.
 */
function pascalCase(str: string): string {
  return str
    .replace(/[-_](.)/g, (_, char) => char.toUpperCase())
    .replace(/^(.)/, (_, char) => char.toUpperCase());
}

export { z } from "zod";
export { zodToJsonSchema } from "zod-to-json-schema";
