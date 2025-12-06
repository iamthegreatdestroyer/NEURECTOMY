#!/usr/bin/env node

/**
 * =============================================================================
 * GraphQL Schema Validation Script
 * =============================================================================
 * Comprehensive validation for GraphQL schema files including:
 * - Syntax validation
 * - Type reference checking
 * - Directive validation
 * - Deprecation checks
 * - Naming convention enforcement
 * - Complexity analysis
 * =============================================================================
 */

import { readFileSync, readdirSync, statSync } from "fs";
import { join, resolve, basename } from "path";
import { buildSchema, parse, validate, validateSchema } from "graphql";

// =============================================================================
// Configuration
// =============================================================================

const CONFIG = {
  schemaDir: resolve(process.cwd(), "schema"),
  maxComplexity: 100,
  maxDepth: 10,
  namingConventions: {
    types: /^[A-Z][a-zA-Z0-9]*$/,
    fields: /^[a-z][a-zA-Z0-9]*$/,
    enums: /^[A-Z][A-Z0-9_]*$/,
    arguments: /^[a-z][a-zA-Z0-9]*$/,
    directives: /^[a-z][a-zA-Z0-9]*$/,
  },
  requiredDirectives: ["deprecated"],
  deprecationPeriodDays: 90,
};

// =============================================================================
// Types
// =============================================================================

/**
 * @typedef {Object} ValidationResult
 * @property {boolean} valid
 * @property {Array<ValidationError>} errors
 * @property {Array<ValidationWarning>} warnings
 * @property {SchemaStats} stats
 */

/**
 * @typedef {Object} ValidationError
 * @property {string} type
 * @property {string} message
 * @property {string} [file]
 * @property {number} [line]
 * @property {string} [location]
 */

/**
 * @typedef {Object} ValidationWarning
 * @property {string} type
 * @property {string} message
 * @property {string} [file]
 * @property {string} [suggestion]
 */

/**
 * @typedef {Object} SchemaStats
 * @property {number} types
 * @property {number} queries
 * @property {number} mutations
 * @property {number} subscriptions
 * @property {number} fields
 * @property {number} enums
 * @property {number} interfaces
 * @property {number} unions
 * @property {number} inputTypes
 * @property {number} directives
 * @property {number} deprecatedFields
 */

// =============================================================================
// Utilities
// =============================================================================

/**
 * Recursively find all GraphQL files in a directory.
 * @param {string} dir
 * @returns {string[]}
 */
function findGraphQLFiles(dir) {
  const files = [];

  const entries = readdirSync(dir);
  for (const entry of entries) {
    const fullPath = join(dir, entry);
    const stat = statSync(fullPath);

    if (stat.isDirectory()) {
      files.push(...findGraphQLFiles(fullPath));
    } else if (entry.endsWith(".graphql")) {
      files.push(fullPath);
    }
  }

  return files;
}

/**
 * Load and concatenate all schema files.
 * @param {string[]} files
 * @returns {{content: string, fileMap: Map<number, string>}}
 */
function loadSchemaFiles(files) {
  let content = "";
  const fileMap = new Map();
  let lineCount = 0;

  for (const file of files) {
    const fileContent = readFileSync(file, "utf-8");
    const fileLines = fileContent.split("\n").length;

    // Map line numbers to files
    for (let i = 0; i < fileLines; i++) {
      fileMap.set(lineCount + i + 1, file);
    }

    content += fileContent + "\n";
    lineCount += fileLines;
  }

  return { content, fileMap };
}

// =============================================================================
// Validators
// =============================================================================

/**
 * Validate GraphQL syntax.
 * @param {string} schemaContent
 * @returns {ValidationError[]}
 */
function validateSyntax(schemaContent) {
  const errors = [];

  try {
    parse(schemaContent);
  } catch (error) {
    errors.push({
      type: "SYNTAX_ERROR",
      message: error.message,
      line: error.locations?.[0]?.line,
    });
  }

  return errors;
}

/**
 * Validate schema is well-formed.
 * @param {string} schemaContent
 * @returns {ValidationError[]}
 */
function validateSchemaStructure(schemaContent) {
  const errors = [];

  try {
    const schema = buildSchema(schemaContent);
    const schemaErrors = validateSchema(schema);

    for (const error of schemaErrors) {
      errors.push({
        type: "SCHEMA_ERROR",
        message: error.message,
        location: error.nodes?.[0]?.loc?.source?.body?.substring(
          error.nodes[0].loc.start,
          error.nodes[0].loc.end
        ),
      });
    }
  } catch (error) {
    errors.push({
      type: "BUILD_ERROR",
      message: error.message,
    });
  }

  return errors;
}

/**
 * Validate naming conventions.
 * @param {string} schemaContent
 * @returns {ValidationWarning[]}
 */
function validateNamingConventions(schemaContent) {
  const warnings = [];

  try {
    const schema = buildSchema(schemaContent);
    const typeMap = schema.getTypeMap();

    for (const [typeName, type] of Object.entries(typeMap)) {
      // Skip built-in types
      if (typeName.startsWith("__")) continue;

      // Check type naming
      if (!CONFIG.namingConventions.types.test(typeName)) {
        warnings.push({
          type: "NAMING_CONVENTION",
          message: `Type "${typeName}" should use PascalCase`,
          suggestion: toPascalCase(typeName),
        });
      }

      // Check field naming
      if (type.getFields) {
        const fields = type.getFields();
        for (const [fieldName] of Object.entries(fields)) {
          if (!CONFIG.namingConventions.fields.test(fieldName)) {
            warnings.push({
              type: "NAMING_CONVENTION",
              message: `Field "${typeName}.${fieldName}" should use camelCase`,
              suggestion: toCamelCase(fieldName),
            });
          }
        }
      }

      // Check enum value naming
      if (type.getValues) {
        const values = type.getValues();
        for (const value of values) {
          if (!CONFIG.namingConventions.enums.test(value.name)) {
            warnings.push({
              type: "NAMING_CONVENTION",
              message: `Enum value "${typeName}.${value.name}" should use SCREAMING_SNAKE_CASE`,
              suggestion: toScreamingSnakeCase(value.name),
            });
          }
        }
      }
    }
  } catch {
    // Schema building failed, skip naming validation
  }

  return warnings;
}

/**
 * Validate deprecations.
 * @param {string} schemaContent
 * @returns {ValidationWarning[]}
 */
function validateDeprecations(schemaContent) {
  const warnings = [];

  try {
    const schema = buildSchema(schemaContent);
    const typeMap = schema.getTypeMap();

    for (const [typeName, type] of Object.entries(typeMap)) {
      if (typeName.startsWith("__")) continue;

      if (type.getFields) {
        const fields = type.getFields();
        for (const [fieldName, field] of Object.entries(fields)) {
          if (field.deprecationReason) {
            warnings.push({
              type: "DEPRECATION",
              message: `Field "${typeName}.${fieldName}" is deprecated: ${field.deprecationReason}`,
              suggestion:
                "Ensure deprecation is documented and migration path is clear",
            });
          }
        }
      }
    }
  } catch {
    // Schema building failed, skip deprecation validation
  }

  return warnings;
}

/**
 * Analyze schema complexity.
 * @param {string} schemaContent
 * @returns {{warnings: ValidationWarning[], stats: SchemaStats}}
 */
function analyzeComplexity(schemaContent) {
  const warnings = [];
  const stats = {
    types: 0,
    queries: 0,
    mutations: 0,
    subscriptions: 0,
    fields: 0,
    enums: 0,
    interfaces: 0,
    unions: 0,
    inputTypes: 0,
    directives: 0,
    deprecatedFields: 0,
  };

  try {
    const schema = buildSchema(schemaContent);
    const typeMap = schema.getTypeMap();

    for (const [typeName, type] of Object.entries(typeMap)) {
      if (typeName.startsWith("__")) continue;

      stats.types++;

      // Count by type kind
      const typeDef = type.astNode;
      if (typeDef) {
        switch (typeDef.kind) {
          case "EnumTypeDefinition":
            stats.enums++;
            break;
          case "InterfaceTypeDefinition":
            stats.interfaces++;
            break;
          case "UnionTypeDefinition":
            stats.unions++;
            break;
          case "InputObjectTypeDefinition":
            stats.inputTypes++;
            break;
        }
      }

      // Count fields
      if (type.getFields) {
        const fields = Object.values(type.getFields());
        stats.fields += fields.length;

        // Count deprecated
        for (const field of fields) {
          if (field.deprecationReason) {
            stats.deprecatedFields++;
          }
        }

        // Warn on types with too many fields
        if (fields.length > 50) {
          warnings.push({
            type: "COMPLEXITY",
            message: `Type "${typeName}" has ${fields.length} fields, consider splitting`,
            suggestion: "Split into multiple related types or use interfaces",
          });
        }
      }
    }

    // Count operations
    const queryType = schema.getQueryType();
    if (queryType?.getFields) {
      stats.queries = Object.keys(queryType.getFields()).length;
    }

    const mutationType = schema.getMutationType();
    if (mutationType?.getFields) {
      stats.mutations = Object.keys(mutationType.getFields()).length;
    }

    const subscriptionType = schema.getSubscriptionType();
    if (subscriptionType?.getFields) {
      stats.subscriptions = Object.keys(subscriptionType.getFields()).length;
    }

    // Count directives
    const directives = schema.getDirectives();
    stats.directives = directives.filter(
      (d) => !d.name.startsWith("__")
    ).length;
  } catch {
    // Schema building failed, skip complexity analysis
  }

  return { warnings, stats };
}

/**
 * Check for circular references that exceed max depth.
 * @param {string} schemaContent
 * @returns {ValidationWarning[]}
 */
function checkCircularReferences(schemaContent) {
  const warnings = [];

  try {
    const schema = buildSchema(schemaContent);
    const typeMap = schema.getTypeMap();
    const visited = new Set();

    function checkDepth(type, path, depth) {
      if (depth > CONFIG.maxDepth) {
        warnings.push({
          type: "CIRCULAR_REFERENCE",
          message: `Deep nesting detected: ${path.join(" -> ")}`,
          suggestion: `Consider flattening the structure (max depth: ${CONFIG.maxDepth})`,
        });
        return;
      }

      const typeName = type.name;
      if (visited.has(typeName)) return;
      visited.add(typeName);

      if (type.getFields) {
        const fields = type.getFields();
        for (const [fieldName, field] of Object.entries(fields)) {
          const fieldType = getNamedType(field.type);
          if (fieldType && typeMap[fieldType.name]) {
            checkDepth(
              typeMap[fieldType.name],
              [...path, fieldName],
              depth + 1
            );
          }
        }
      }

      visited.delete(typeName);
    }

    for (const [typeName, type] of Object.entries(typeMap)) {
      if (typeName.startsWith("__")) continue;
      checkDepth(type, [typeName], 0);
    }
  } catch {
    // Schema building failed
  }

  return warnings;
}

// =============================================================================
// Helper Functions
// =============================================================================

function getNamedType(type) {
  while (type.ofType) {
    type = type.ofType;
  }
  return type;
}

function toPascalCase(str) {
  return str
    .replace(/[-_](.)/g, (_, c) => c.toUpperCase())
    .replace(/^(.)/, (_, c) => c.toUpperCase());
}

function toCamelCase(str) {
  return str
    .replace(/[-_](.)/g, (_, c) => c.toUpperCase())
    .replace(/^(.)/, (_, c) => c.toLowerCase());
}

function toScreamingSnakeCase(str) {
  return str
    .replace(/([a-z])([A-Z])/g, "$1_$2")
    .replace(/[-\s]/g, "_")
    .toUpperCase();
}

// =============================================================================
// Main Validation Function
// =============================================================================

/**
 * Run all validations.
 * @param {boolean} strict
 * @returns {ValidationResult}
 */
function runValidation(strict = false) {
  console.log("🔍 Starting GraphQL schema validation...\n");

  const files = findGraphQLFiles(CONFIG.schemaDir);
  console.log(`📁 Found ${files.length} GraphQL files\n`);

  if (files.length === 0) {
    return {
      valid: false,
      errors: [{ type: "NO_FILES", message: "No GraphQL files found" }],
      warnings: [],
      stats: {},
    };
  }

  const { content, fileMap } = loadSchemaFiles(files);

  // Run validations
  const syntaxErrors = validateSyntax(content);
  if (syntaxErrors.length > 0) {
    // Enhance errors with file info
    for (const error of syntaxErrors) {
      if (error.line) {
        error.file = fileMap.get(error.line);
      }
    }
    return {
      valid: false,
      errors: syntaxErrors,
      warnings: [],
      stats: {},
    };
  }

  const schemaErrors = validateSchemaStructure(content);
  const namingWarnings = validateNamingConventions(content);
  const deprecationWarnings = validateDeprecations(content);
  const { warnings: complexityWarnings, stats } = analyzeComplexity(content);
  const circularWarnings = checkCircularReferences(content);

  const allWarnings = [
    ...namingWarnings,
    ...deprecationWarnings,
    ...complexityWarnings,
    ...circularWarnings,
  ];

  const valid =
    schemaErrors.length === 0 && (!strict || allWarnings.length === 0);

  return {
    valid,
    errors: schemaErrors,
    warnings: allWarnings,
    stats,
  };
}

// =============================================================================
// CLI Entry Point
// =============================================================================

function main() {
  const args = process.argv.slice(2);
  const strict = args.includes("--strict");
  const json = args.includes("--json");

  const result = runValidation(strict);

  if (json) {
    console.log(JSON.stringify(result, null, 2));
    process.exit(result.valid ? 0 : 1);
    return;
  }

  // Print errors
  if (result.errors.length > 0) {
    console.log("❌ Errors:\n");
    for (const error of result.errors) {
      const location = error.file
        ? `[${basename(error.file)}:${error.line || "?"}]`
        : "";
      console.log(`  ${location} ${error.type}: ${error.message}`);
    }
    console.log("");
  }

  // Print warnings
  if (result.warnings.length > 0) {
    console.log(`⚠️  Warnings (${result.warnings.length}):\n`);
    for (const warning of result.warnings.slice(0, 20)) {
      console.log(`  ${warning.type}: ${warning.message}`);
      if (warning.suggestion) {
        console.log(`    💡 ${warning.suggestion}`);
      }
    }
    if (result.warnings.length > 20) {
      console.log(`  ... and ${result.warnings.length - 20} more warnings`);
    }
    console.log("");
  }

  // Print stats
  if (result.stats && Object.keys(result.stats).length > 0) {
    console.log("📊 Schema Statistics:\n");
    console.log(`  Types:         ${result.stats.types}`);
    console.log(`  Queries:       ${result.stats.queries}`);
    console.log(`  Mutations:     ${result.stats.mutations}`);
    console.log(`  Subscriptions: ${result.stats.subscriptions}`);
    console.log(`  Fields:        ${result.stats.fields}`);
    console.log(`  Enums:         ${result.stats.enums}`);
    console.log(`  Interfaces:    ${result.stats.interfaces}`);
    console.log(`  Unions:        ${result.stats.unions}`);
    console.log(`  Input Types:   ${result.stats.inputTypes}`);
    console.log(`  Directives:    ${result.stats.directives}`);
    console.log(`  Deprecated:    ${result.stats.deprecatedFields}`);
    console.log("");
  }

  // Final status
  if (result.valid) {
    console.log("✅ Schema validation passed!\n");
    process.exit(0);
  } else {
    console.log("❌ Schema validation failed!\n");
    process.exit(1);
  }
}

main();
