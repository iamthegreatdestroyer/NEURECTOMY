#!/usr/bin/env node

/**
 * =============================================================================
 * GraphQL Schema Documentation Coverage Checker
 * =============================================================================
 * Analyzes GraphQL schema files to ensure all types, fields, and arguments
 * have proper documentation (descriptions).
 * =============================================================================
 */

import { readFileSync, readdirSync, statSync } from "fs";
import { join, resolve, basename } from "path";
import { buildSchema } from "graphql";

// =============================================================================
// Configuration
// =============================================================================

const CONFIG = {
  schemaDir: resolve(process.cwd(), "schema"),
  // Minimum documentation coverage percentage
  minCoverage: 80,
  // Types to exclude from coverage (usually built-in or internal)
  excludePatterns: [
    /^__/, // Built-in introspection types
    /^PageInfo$/, // Standard relay types
    /Connection$/, // Connection types often self-documenting
    /Edge$/, // Edge types often self-documenting
  ],
};

// =============================================================================
// Types
// =============================================================================

/**
 * @typedef {Object} CoverageReport
 * @property {number} totalItems
 * @property {number} documentedItems
 * @property {number} coverage
 * @property {boolean} passing
 * @property {Array<UndocumentedItem>} undocumented
 * @property {Object} byCategory
 */

/**
 * @typedef {Object} UndocumentedItem
 * @property {string} category
 * @property {string} name
 * @property {string} [parent]
 * @property {string} location
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

  try {
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
  } catch (error) {
    console.error(`Error reading directory ${dir}:`, error.message);
  }

  return files;
}

/**
 * Load and concatenate all schema files.
 * @param {string[]} files
 * @returns {string}
 */
function loadSchemaFiles(files) {
  let content = "";

  for (const file of files) {
    content += readFileSync(file, "utf-8") + "\n";
  }

  return content;
}

/**
 * Check if a type name should be excluded.
 * @param {string} name
 * @returns {boolean}
 */
function shouldExclude(name) {
  return CONFIG.excludePatterns.some((pattern) => pattern.test(name));
}

// =============================================================================
// Coverage Analysis
// =============================================================================

/**
 * Analyze documentation coverage.
 * @param {string} schemaContent
 * @returns {CoverageReport}
 */
function analyzeCoverage(schemaContent) {
  const undocumented = [];
  const byCategory = {
    types: { total: 0, documented: 0 },
    fields: { total: 0, documented: 0 },
    arguments: { total: 0, documented: 0 },
    enumValues: { total: 0, documented: 0 },
    directives: { total: 0, documented: 0 },
  };

  try {
    const schema = buildSchema(schemaContent);
    const typeMap = schema.getTypeMap();

    // Analyze types
    for (const [typeName, type] of Object.entries(typeMap)) {
      if (shouldExclude(typeName)) continue;

      byCategory.types.total++;

      // Check type description
      if (type.description) {
        byCategory.types.documented++;
      } else {
        undocumented.push({
          category: "type",
          name: typeName,
          location: `Type "${typeName}"`,
        });
      }

      // Analyze fields
      if (type.getFields) {
        const fields = type.getFields();
        for (const [fieldName, field] of Object.entries(fields)) {
          byCategory.fields.total++;

          if (field.description) {
            byCategory.fields.documented++;
          } else {
            undocumented.push({
              category: "field",
              name: fieldName,
              parent: typeName,
              location: `${typeName}.${fieldName}`,
            });
          }

          // Analyze arguments
          if (field.args) {
            for (const arg of field.args) {
              byCategory.arguments.total++;

              if (arg.description) {
                byCategory.arguments.documented++;
              } else {
                undocumented.push({
                  category: "argument",
                  name: arg.name,
                  parent: `${typeName}.${fieldName}`,
                  location: `${typeName}.${fieldName}(${arg.name})`,
                });
              }
            }
          }
        }
      }

      // Analyze enum values
      if (type.getValues) {
        const values = type.getValues();
        for (const value of values) {
          byCategory.enumValues.total++;

          if (value.description) {
            byCategory.enumValues.documented++;
          } else {
            undocumented.push({
              category: "enumValue",
              name: value.name,
              parent: typeName,
              location: `${typeName}.${value.name}`,
            });
          }
        }
      }
    }

    // Analyze directives
    const directives = schema.getDirectives();
    for (const directive of directives) {
      if (directive.name.startsWith("__")) continue;

      byCategory.directives.total++;

      if (directive.description) {
        byCategory.directives.documented++;
      } else {
        undocumented.push({
          category: "directive",
          name: directive.name,
          location: `@${directive.name}`,
        });
      }
    }
  } catch (error) {
    console.error("Error building schema:", error.message);
    return {
      totalItems: 0,
      documentedItems: 0,
      coverage: 0,
      passing: false,
      undocumented: [],
      byCategory: {},
      error: error.message,
    };
  }

  // Calculate totals
  const totalItems = Object.values(byCategory).reduce(
    (sum, cat) => sum + cat.total,
    0
  );
  const documentedItems = Object.values(byCategory).reduce(
    (sum, cat) => sum + cat.documented,
    0
  );
  const coverage = totalItems > 0 ? (documentedItems / totalItems) * 100 : 100;

  return {
    totalItems,
    documentedItems,
    coverage,
    passing: coverage >= CONFIG.minCoverage,
    undocumented,
    byCategory,
  };
}

// =============================================================================
// Report Generation
// =============================================================================

/**
 * Generate coverage report.
 * @param {CoverageReport} report
 * @returns {string}
 */
function generateReport(report) {
  const lines = [];

  lines.push("# GraphQL Schema Documentation Coverage Report");
  lines.push("");
  lines.push(
    `Coverage: ${report.coverage.toFixed(1)}% (${report.documentedItems}/${report.totalItems})`
  );
  lines.push(`Minimum Required: ${CONFIG.minCoverage}%`);
  lines.push(`Status: ${report.passing ? "✅ PASSING" : "❌ FAILING"}`);
  lines.push("");

  // Category breakdown
  lines.push("## Coverage by Category");
  lines.push("");
  lines.push("| Category | Documented | Total | Coverage |");
  lines.push("|----------|------------|-------|----------|");

  for (const [category, stats] of Object.entries(report.byCategory)) {
    const catCoverage =
      stats.total > 0
        ? ((stats.documented / stats.total) * 100).toFixed(1)
        : "100.0";
    const status = parseFloat(catCoverage) >= CONFIG.minCoverage ? "✅" : "⚠️";
    lines.push(
      `| ${category} | ${stats.documented} | ${stats.total} | ${status} ${catCoverage}% |`
    );
  }

  lines.push("");

  // Undocumented items
  if (report.undocumented.length > 0) {
    lines.push("## Undocumented Items");
    lines.push("");

    // Group by category
    const byCategory = {};
    for (const item of report.undocumented) {
      if (!byCategory[item.category]) {
        byCategory[item.category] = [];
      }
      byCategory[item.category].push(item);
    }

    for (const [category, items] of Object.entries(byCategory)) {
      lines.push(
        `### ${category.charAt(0).toUpperCase() + category.slice(1)}s (${items.length})`
      );
      lines.push("");

      // Show up to 20 items per category
      const displayItems = items.slice(0, 20);
      for (const item of displayItems) {
        lines.push(`- \`${item.location}\``);
      }

      if (items.length > 20) {
        lines.push(`- ... and ${items.length - 20} more`);
      }

      lines.push("");
    }
  }

  // Recommendations
  lines.push("## Recommendations");
  lines.push("");

  if (report.byCategory.types?.total > report.byCategory.types?.documented) {
    lines.push(
      "- Add descriptions to all type definitions using triple-quoted strings"
    );
  }
  if (report.byCategory.fields?.total > report.byCategory.fields?.documented) {
    lines.push("- Document all fields, especially query/mutation root fields");
  }
  if (
    report.byCategory.arguments?.total > report.byCategory.arguments?.documented
  ) {
    lines.push(
      "- Add descriptions to field arguments, especially required ones"
    );
  }
  if (
    report.byCategory.enumValues?.total >
    report.byCategory.enumValues?.documented
  ) {
    lines.push("- Document enum values to clarify their meaning");
  }

  lines.push("");
  lines.push("### Documentation Example");
  lines.push("");
  lines.push("```graphql");
  lines.push('"""');
  lines.push("Represents a user in the system.");
  lines.push('"""');
  lines.push("type User {");
  lines.push('  """The unique identifier for the user."""');
  lines.push("  id: ID!");
  lines.push("");
  lines.push('  """');
  lines.push("  The user's display name.");
  lines.push("  This is shown in the UI and can be changed by the user.");
  lines.push('  """');
  lines.push("  displayName: String!");
  lines.push("}");
  lines.push("```");

  return lines.join("\n");
}

// =============================================================================
// CLI Entry Point
// =============================================================================

function main() {
  const args = process.argv.slice(2);
  const json = args.includes("--json");
  const markdown = args.includes("--markdown");
  const minCoverageArg = args.find((arg) => arg.startsWith("--min-coverage="));

  if (minCoverageArg) {
    const value = parseInt(minCoverageArg.split("=")[1], 10);
    if (!isNaN(value)) {
      CONFIG.minCoverage = value;
    }
  }

  console.log("📚 Checking GraphQL schema documentation coverage...\n");

  const files = findGraphQLFiles(CONFIG.schemaDir);

  if (files.length === 0) {
    console.error("❌ No GraphQL files found in", CONFIG.schemaDir);
    process.exit(1);
  }

  console.log(`📁 Found ${files.length} GraphQL files\n`);

  const schemaContent = loadSchemaFiles(files);
  const report = analyzeCoverage(schemaContent);

  if (json) {
    console.log(JSON.stringify(report, null, 2));
  } else if (markdown) {
    console.log(generateReport(report));
  } else {
    // Console output
    console.log(`📊 Documentation Coverage: ${report.coverage.toFixed(1)}%`);
    console.log(
      `   Documented: ${report.documentedItems}/${report.totalItems} items`
    );
    console.log(`   Minimum Required: ${CONFIG.minCoverage}%`);
    console.log("");

    console.log("📈 Coverage by Category:");
    for (const [category, stats] of Object.entries(report.byCategory)) {
      const catCoverage =
        stats.total > 0
          ? ((stats.documented / stats.total) * 100).toFixed(1)
          : "100.0";
      const bar = generateProgressBar(parseFloat(catCoverage));
      console.log(
        `   ${category.padEnd(12)} ${bar} ${catCoverage}% (${stats.documented}/${stats.total})`
      );
    }
    console.log("");

    if (report.undocumented.length > 0) {
      console.log(`⚠️  Undocumented items: ${report.undocumented.length}`);

      // Show top undocumented by category
      const byCategory = {};
      for (const item of report.undocumented) {
        byCategory[item.category] = (byCategory[item.category] || 0) + 1;
      }

      for (const [category, count] of Object.entries(byCategory)) {
        console.log(`   - ${count} ${category}(s)`);
      }
      console.log("");

      // Show first few undocumented
      console.log("   Top undocumented items:");
      for (const item of report.undocumented.slice(0, 10)) {
        console.log(`   - ${item.location}`);
      }
      if (report.undocumented.length > 10) {
        console.log(`   - ... and ${report.undocumented.length - 10} more`);
      }
      console.log("");
    }

    if (report.passing) {
      console.log("✅ Documentation coverage check PASSED!\n");
    } else {
      console.log("❌ Documentation coverage check FAILED!");
      console.log(
        `   Coverage is ${report.coverage.toFixed(1)}%, minimum required is ${CONFIG.minCoverage}%\n`
      );
    }
  }

  process.exit(report.passing ? 0 : 1);
}

/**
 * Generate a progress bar string.
 * @param {number} percentage
 * @returns {string}
 */
function generateProgressBar(percentage) {
  const width = 20;
  const filled = Math.round((percentage / 100) * width);
  const empty = width - filled;

  const filledChar = percentage >= CONFIG.minCoverage ? "█" : "▓";
  const emptyChar = "░";

  return `[${filledChar.repeat(filled)}${emptyChar.repeat(empty)}]`;
}

main();
