#!/usr/bin/env node

import { Command } from "commander";
import * as path from "path";
import * as fs from "fs/promises";
import {
  agentConfigSchema,
  taskDefinitionSchema,
  workflowSchema,
} from "@neurectomy/core";
import { generateSchemas } from "./index";

const program = new Command();

program
  .name("neurectomy-codegen")
  .description(
    "Generate JSON schemas and TypeScript type guards from Zod schemas"
  )
  .version("0.1.0");

program
  .command("generate")
  .description("Generate schemas from @neurectomy/core")
  .option("-o, --output <dir>", "Output directory", "./generated")
  .option(
    "-f, --format <format>",
    "Output format: json, typescript, both",
    "both"
  )
  .option("--no-pretty", "Disable pretty printing")
  .action(async (options) => {
    const outputDir = path.resolve(options.output);

    console.log(`🔄 Generating schemas to ${outputDir}...`);

    const schemas = {
      agentConfig: agentConfigSchema,
      taskDefinition: taskDefinitionSchema,
      workflow: workflowSchema,
    };

    const results = await generateSchemas({
      outputDir,
      schemas,
      format: options.format as "json" | "typescript" | "both",
      prettyPrint: options.pretty,
    });

    console.log(`✅ Generated ${results.length} schemas:`);
    results.forEach((r) => console.log(`   - ${r.name}`));
  });

program
  .command("validate")
  .description("Validate a JSON file against a schema")
  .argument("<schema>", "Schema name (agentConfig, taskDefinition, workflow)")
  .argument("<file>", "JSON file to validate")
  .action(async (schemaName: string, file: string) => {
    const schemaMap: Record<string, typeof agentConfigSchema> = {
      agentConfig: agentConfigSchema,
      taskDefinition: taskDefinitionSchema,
      workflow: workflowSchema,
    };

    const schema = schemaMap[schemaName];
    if (!schema) {
      console.error(`❌ Unknown schema: ${schemaName}`);
      console.error(`   Available: ${Object.keys(schemaMap).join(", ")}`);
      process.exit(1);
    }

    const filePath = path.resolve(file);
    const content = await fs.readFile(filePath, "utf-8");
    const data = JSON.parse(content);

    const result = schema.safeParse(data);

    if (result.success) {
      console.log("✅ Validation passed");
    } else {
      console.error("❌ Validation failed:");
      result.error.issues.forEach((issue) => {
        console.error(`   - ${issue.path.join(".")}: ${issue.message}`);
      });
      process.exit(1);
    }
  });

program.parse();
