/**
 * @fileoverview License Compatibility Checker
 * @module @neurectomy/legal-fortress/license/compatibility
 *
 * @agents @AEGIS @AXIOM - Compliance + Mathematics Specialists
 *
 * Analyzes license compatibility using:
 * - Graph-based compatibility matrices
 * - SPDX expression parsing
 * - Transitive compatibility analysis
 * - Risk assessment algorithms
 */

import { LicenseType, LicenseRisk, DetectedLicense } from "../types";
import {
  LicenseDefinition,
  LICENSE_DATABASE,
  LicenseDetectionEngine,
} from "./detection";

// ============================================================================
// COMPATIBILITY TYPES (@AEGIS)
// ============================================================================

/**
 * Compatibility level between licenses
 */
export type CompatibilityLevel =
  | "compatible" // Can be combined freely
  | "conditional" // Can be combined with conditions
  | "one-way" // Only one direction works
  | "incompatible" // Cannot be combined
  | "unknown"; // Compatibility unknown

/**
 * Compatibility check result
 */
export interface CompatibilityResult {
  source: string;
  target: string;
  level: CompatibilityLevel;
  direction?: "source-to-target" | "target-to-source" | "bidirectional";
  conditions?: string[];
  explanation: string;
  riskLevel: LicenseRisk;
}

/**
 * Project license analysis result
 */
export interface ProjectLicenseAnalysis {
  projectLicense: string;
  dependencies: { name: string; license: string }[];
  compatibilityMatrix: CompatibilityResult[];
  overallRisk: LicenseRisk;
  issues: LicenseIssue[];
  recommendations: string[];
}

/**
 * License issue found in analysis
 */
export interface LicenseIssue {
  severity: "error" | "warning" | "info";
  type:
    | "incompatible"
    | "unknown"
    | "copyleft-conflict"
    | "attribution-required"
    | "patent-clause";
  description: string;
  affectedPackages: string[];
  remediation?: string;
}

/**
 * SPDX expression node
 */
export interface SPDXExpressionNode {
  type: "license" | "and" | "or" | "with" | "exception";
  license?: string;
  left?: SPDXExpressionNode;
  right?: SPDXExpressionNode;
  exception?: string;
}

// ============================================================================
// COMPATIBILITY MATRIX (@AXIOM)
// ============================================================================

/**
 * License compatibility rules
 * Based on SPDX compatibility guidelines and legal analysis
 * @agent @AXIOM - Mathematical license relationships
 */
export const COMPATIBILITY_MATRIX: Record<
  string,
  Record<string, CompatibilityLevel>
> = {
  // MIT is compatible with almost everything
  MIT: {
    MIT: "compatible",
    "Apache-2.0": "compatible",
    "BSD-2-Clause": "compatible",
    "BSD-3-Clause": "compatible",
    ISC: "compatible",
    Unlicense: "compatible",
    "CC0-1.0": "compatible",
    "GPL-2.0": "one-way",
    "GPL-3.0": "one-way",
    "LGPL-3.0": "conditional",
    "AGPL-3.0": "one-way",
    "MPL-2.0": "compatible",
    "EPL-2.0": "conditional",
    PROPRIETARY: "conditional",
  },

  // Apache 2.0 specifics
  "Apache-2.0": {
    MIT: "compatible",
    "Apache-2.0": "compatible",
    "BSD-2-Clause": "compatible",
    "BSD-3-Clause": "compatible",
    ISC: "compatible",
    Unlicense: "compatible",
    "CC0-1.0": "compatible",
    "GPL-2.0": "incompatible", // Patent clause conflict
    "GPL-3.0": "one-way",
    "LGPL-3.0": "conditional",
    "AGPL-3.0": "one-way",
    "MPL-2.0": "compatible",
    "EPL-2.0": "conditional",
    PROPRIETARY: "conditional",
  },

  // GPL-3.0 is restrictive
  "GPL-3.0": {
    MIT: "compatible",
    "Apache-2.0": "compatible",
    "BSD-2-Clause": "compatible",
    "BSD-3-Clause": "compatible",
    ISC: "compatible",
    Unlicense: "compatible",
    "CC0-1.0": "compatible",
    "GPL-2.0": "incompatible", // Version conflict
    "GPL-3.0": "compatible",
    "LGPL-3.0": "compatible",
    "AGPL-3.0": "compatible",
    "MPL-2.0": "compatible",
    "EPL-2.0": "incompatible",
    PROPRIETARY: "incompatible",
  },

  // GPL-2.0 is more restrictive
  "GPL-2.0": {
    MIT: "compatible",
    "Apache-2.0": "incompatible",
    "BSD-2-Clause": "compatible",
    "BSD-3-Clause": "compatible",
    ISC: "compatible",
    Unlicense: "compatible",
    "CC0-1.0": "compatible",
    "GPL-2.0": "compatible",
    "GPL-3.0": "incompatible",
    "LGPL-3.0": "incompatible",
    "AGPL-3.0": "incompatible",
    "MPL-2.0": "conditional",
    "EPL-2.0": "incompatible",
    PROPRIETARY: "incompatible",
  },

  // AGPL-3.0 is most restrictive
  "AGPL-3.0": {
    MIT: "compatible",
    "Apache-2.0": "compatible",
    "BSD-2-Clause": "compatible",
    "BSD-3-Clause": "compatible",
    ISC: "compatible",
    Unlicense: "compatible",
    "CC0-1.0": "compatible",
    "GPL-2.0": "incompatible",
    "GPL-3.0": "compatible",
    "LGPL-3.0": "compatible",
    "AGPL-3.0": "compatible",
    "MPL-2.0": "compatible",
    "EPL-2.0": "incompatible",
    PROPRIETARY: "incompatible",
  },

  // LGPL-3.0 weak copyleft
  "LGPL-3.0": {
    MIT: "compatible",
    "Apache-2.0": "compatible",
    "BSD-2-Clause": "compatible",
    "BSD-3-Clause": "compatible",
    ISC: "compatible",
    Unlicense: "compatible",
    "CC0-1.0": "compatible",
    "GPL-2.0": "incompatible",
    "GPL-3.0": "one-way",
    "LGPL-3.0": "compatible",
    "AGPL-3.0": "one-way",
    "MPL-2.0": "compatible",
    "EPL-2.0": "conditional",
    PROPRIETARY: "conditional", // Dynamic linking allowed
  },

  // MPL-2.0 file-level copyleft
  "MPL-2.0": {
    MIT: "compatible",
    "Apache-2.0": "compatible",
    "BSD-2-Clause": "compatible",
    "BSD-3-Clause": "compatible",
    ISC: "compatible",
    Unlicense: "compatible",
    "CC0-1.0": "compatible",
    "GPL-2.0": "one-way",
    "GPL-3.0": "one-way",
    "LGPL-3.0": "compatible",
    "AGPL-3.0": "one-way",
    "MPL-2.0": "compatible",
    "EPL-2.0": "conditional",
    PROPRIETARY: "conditional",
  },

  // Proprietary defaults
  PROPRIETARY: {
    MIT: "compatible",
    "Apache-2.0": "compatible",
    "BSD-2-Clause": "compatible",
    "BSD-3-Clause": "compatible",
    ISC: "compatible",
    Unlicense: "compatible",
    "CC0-1.0": "compatible",
    "GPL-2.0": "incompatible",
    "GPL-3.0": "incompatible",
    "LGPL-3.0": "conditional",
    "AGPL-3.0": "incompatible",
    "MPL-2.0": "conditional",
    "EPL-2.0": "conditional",
    PROPRIETARY: "compatible",
  },
};

// ============================================================================
// SPDX EXPRESSION PARSER (@AXIOM)
// ============================================================================

/**
 * Parse SPDX license expression
 * @agent @AXIOM - Formal expression parsing
 */
export function parseSPDXExpression(expression: string): SPDXExpressionNode {
  const tokens = tokenizeSPDX(expression);
  return parseExpression(tokens, 0).node;
}

/**
 * Tokenize SPDX expression
 */
function tokenizeSPDX(expression: string): string[] {
  const normalized = expression
    .replace(/\(/g, " ( ")
    .replace(/\)/g, " ) ")
    .replace(/\s+/g, " ")
    .trim();

  return normalized.split(" ").filter((t) => t.length > 0);
}

/**
 * Parse expression tokens
 */
function parseExpression(
  tokens: string[],
  pos: number
): { node: SPDXExpressionNode; nextPos: number } {
  let { node: left, nextPos } = parsePrimary(tokens, pos);

  while (nextPos < tokens.length) {
    const token = tokens[nextPos]!.toUpperCase();

    if (token === "AND" || token === "OR") {
      nextPos++;
      const { node: right, nextPos: afterRight } = parsePrimary(
        tokens,
        nextPos
      );
      left = {
        type: token.toLowerCase() as "and" | "or",
        left,
        right,
      };
      nextPos = afterRight;
    } else if (token === "WITH") {
      nextPos++;
      const exception = tokens[nextPos];
      left = {
        type: "with",
        left,
        exception,
      };
      nextPos++;
    } else {
      break;
    }
  }

  return { node: left, nextPos };
}

/**
 * Parse primary expression (license or grouped expression)
 */
function parsePrimary(
  tokens: string[],
  pos: number
): { node: SPDXExpressionNode; nextPos: number } {
  const token = tokens[pos]!;

  if (token === "(") {
    const { node, nextPos } = parseExpression(tokens, pos + 1);
    // Skip closing paren
    return { node, nextPos: nextPos + 1 };
  }

  // License identifier
  return {
    node: { type: "license", license: token },
    nextPos: pos + 1,
  };
}

/**
 * Evaluate SPDX expression to list of valid license choices
 */
export function evaluateSPDXExpression(node: SPDXExpressionNode): string[][] {
  if (node.type === "license") {
    return [[node.license!]];
  }

  if (node.type === "and") {
    const leftChoices = evaluateSPDXExpression(node.left!);
    const rightChoices = evaluateSPDXExpression(node.right!);

    // AND requires both - combine all possibilities
    const result: string[][] = [];
    for (const left of leftChoices) {
      for (const right of rightChoices) {
        result.push([...left, ...right]);
      }
    }
    return result;
  }

  if (node.type === "or") {
    const leftChoices = evaluateSPDXExpression(node.left!);
    const rightChoices = evaluateSPDXExpression(node.right!);

    // OR allows any choice
    return [...leftChoices, ...rightChoices];
  }

  if (node.type === "with") {
    const baseChoices = evaluateSPDXExpression(node.left!);
    // WITH adds exception to each choice
    return baseChoices.map((choice) =>
      choice.map((l) => `${l} WITH ${node.exception}`)
    );
  }

  return [[]];
}

// ============================================================================
// LICENSE COMPATIBILITY CHECKER (@AEGIS @AXIOM)
// ============================================================================

/**
 * License Compatibility Checker
 * @agent @AEGIS @AXIOM - Complete compatibility analysis
 */
export class LicenseCompatibilityChecker {
  private licenseEngine: LicenseDetectionEngine;

  constructor(licenseEngine?: LicenseDetectionEngine) {
    this.licenseEngine = licenseEngine ?? new LicenseDetectionEngine();
  }

  /**
   * Check compatibility between two licenses
   */
  checkCompatibility(
    sourceLicense: string,
    targetLicense: string
  ): CompatibilityResult {
    const sourceNorm = this.normalizeLicense(sourceLicense);
    const targetNorm = this.normalizeLicense(targetLicense);

    // Check direct compatibility
    const sourceRow = COMPATIBILITY_MATRIX[sourceNorm];
    if (sourceRow && sourceRow[targetNorm]) {
      return this.buildResult(sourceNorm, targetNorm, sourceRow[targetNorm]!);
    }

    // Check reverse compatibility
    const targetRow = COMPATIBILITY_MATRIX[targetNorm];
    if (targetRow && targetRow[sourceNorm]) {
      const level = targetRow[sourceNorm]!;
      if (level === "one-way") {
        return this.buildResult(
          sourceNorm,
          targetNorm,
          "incompatible",
          "target-to-source"
        );
      }
      return this.buildResult(sourceNorm, targetNorm, level, "bidirectional");
    }

    // Unknown combination
    return this.buildResult(sourceNorm, targetNorm, "unknown");
  }

  /**
   * Analyze project license compatibility
   * @agent @AXIOM - Graph analysis
   */
  analyzeProject(
    projectLicense: string,
    dependencies: { name: string; license: string }[]
  ): ProjectLicenseAnalysis {
    const compatibilityMatrix: CompatibilityResult[] = [];
    const issues: LicenseIssue[] = [];
    let highestRisk: LicenseRisk = "low";

    // Check each dependency against project license
    for (const dep of dependencies) {
      const result = this.checkCompatibility(projectLicense, dep.license);
      compatibilityMatrix.push(result);

      // Track highest risk
      if (this.riskLevel(result.riskLevel) > this.riskLevel(highestRisk)) {
        highestRisk = result.riskLevel;
      }

      // Generate issues
      if (result.level === "incompatible") {
        issues.push({
          severity: "error",
          type: "incompatible",
          description: `License ${dep.license} is incompatible with project license ${projectLicense}`,
          affectedPackages: [dep.name],
          remediation: `Replace ${dep.name} with a compatible alternative or change project license`,
        });
      } else if (result.level === "unknown") {
        issues.push({
          severity: "warning",
          type: "unknown",
          description: `Unknown license compatibility between ${projectLicense} and ${dep.license}`,
          affectedPackages: [dep.name],
          remediation: "Review license terms manually or consult legal counsel",
        });
      } else if (result.level === "conditional") {
        issues.push({
          severity: "info",
          type: "attribution-required",
          description: `Conditional compatibility with ${dep.license}`,
          affectedPackages: [dep.name],
          remediation:
            result.conditions?.join(", ") ?? "Review specific conditions",
        });
      }
    }

    // Check for copyleft conflicts
    const copyleftDeps = dependencies.filter((d) => {
      const def = this.licenseEngine.getLicenseDefinition(d.license);
      return def?.copyleft;
    });

    if (copyleftDeps.length > 0) {
      const projectDef =
        this.licenseEngine.getLicenseDefinition(projectLicense);
      if (projectDef && !projectDef.copyleft) {
        issues.push({
          severity: "warning",
          type: "copyleft-conflict",
          description:
            "Copyleft dependencies may require releasing your code under the same license",
          affectedPackages: copyleftDeps.map((d) => d.name),
          remediation:
            "Review copyleft requirements or choose alternative packages",
        });
      }
    }

    // Generate recommendations
    const recommendations = this.generateRecommendations(
      projectLicense,
      dependencies,
      issues
    );

    return {
      projectLicense,
      dependencies,
      compatibilityMatrix,
      overallRisk: highestRisk,
      issues,
      recommendations,
    };
  }

  /**
   * Check SPDX expression compatibility
   */
  checkSPDXCompatibility(
    projectExpression: string,
    dependencyExpression: string
  ): CompatibilityResult[] {
    const projectChoices = evaluateSPDXExpression(
      parseSPDXExpression(projectExpression)
    );
    const depChoices = evaluateSPDXExpression(
      parseSPDXExpression(dependencyExpression)
    );

    const results: CompatibilityResult[] = [];

    // Find best compatible combination
    for (const projectChoice of projectChoices) {
      for (const depChoice of depChoices) {
        // Check if all licenses in combination are compatible
        let allCompatible = true;
        for (const projLicense of projectChoice) {
          for (const depLicense of depChoice) {
            const result = this.checkCompatibility(projLicense, depLicense);
            results.push(result);
            if (result.level === "incompatible") {
              allCompatible = false;
            }
          }
        }

        if (allCompatible) {
          // Found compatible combination
          return results.filter((r) => r.level !== "incompatible");
        }
      }
    }

    return results;
  }

  /**
   * Get license type (copyleft, permissive, etc)
   */
  getLicenseType(spdxId: string): LicenseType | "unknown" {
    const def = this.licenseEngine.getLicenseDefinition(spdxId);
    return def?.type ?? "unknown";
  }

  /**
   * Get license risk level
   */
  getLicenseRisk(spdxId: string): LicenseRisk {
    const def = this.licenseEngine.getLicenseDefinition(spdxId);
    return def?.risk ?? "unknown";
  }

  // ============================================================================
  // PRIVATE METHODS
  // ============================================================================

  /**
   * Normalize license identifier
   */
  private normalizeLicense(license: string): string {
    // Handle common variations
    const normalized = license
      .replace(/\s+/g, "-")
      .replace(/v(\d)/, "-$1")
      .replace("only", "")
      .replace("or-later", "+")
      .trim();

    // Map common aliases
    const aliases: Record<string, string> = {
      BSD: "BSD-3-Clause",
      GPL: "GPL-3.0",
      LGPL: "LGPL-3.0",
      Apache: "Apache-2.0",
    };

    return aliases[normalized] ?? normalized;
  }

  /**
   * Build compatibility result
   */
  private buildResult(
    source: string,
    target: string,
    level: CompatibilityLevel,
    direction:
      | "source-to-target"
      | "target-to-source"
      | "bidirectional" = "bidirectional"
  ): CompatibilityResult {
    const sourceDef = this.licenseEngine.getLicenseDefinition(source);
    const targetDef = this.licenseEngine.getLicenseDefinition(target);

    let explanation: string;
    let conditions: string[] | undefined;
    let riskLevel: LicenseRisk;

    switch (level) {
      case "compatible":
        explanation = `${source} and ${target} are compatible and can be freely combined`;
        riskLevel = "low";
        break;

      case "conditional":
        explanation = `${source} and ${target} can be combined with specific conditions`;
        conditions = this.getConditions(source, target);
        riskLevel = "medium";
        break;

      case "one-way":
        explanation =
          direction === "source-to-target"
            ? `Code under ${source} can use ${target} code, but not vice versa`
            : `Code under ${target} can use ${source} code, but not vice versa`;
        riskLevel = "medium";
        break;

      case "incompatible":
        explanation = `${source} and ${target} have incompatible terms and cannot be combined`;
        riskLevel = "critical";
        break;

      default:
        explanation = `Compatibility between ${source} and ${target} is unknown - manual review required`;
        riskLevel = "unknown";
    }

    return {
      source,
      target,
      level,
      direction,
      conditions,
      explanation,
      riskLevel,
    };
  }

  /**
   * Get specific conditions for conditional compatibility
   */
  private getConditions(source: string, target: string): string[] {
    const conditions: string[] = [];

    // LGPL requires dynamic linking
    if (target === "LGPL-3.0" || source === "LGPL-3.0") {
      conditions.push(
        "Dynamic linking only - static linking requires LGPL licensing"
      );
    }

    // MPL requires separate files
    if (target === "MPL-2.0" || source === "MPL-2.0") {
      conditions.push("MPL-licensed files must remain under MPL");
    }

    // Attribution requirements
    const needsAttribution = [
      "MIT",
      "BSD-2-Clause",
      "BSD-3-Clause",
      "Apache-2.0",
    ];
    if (needsAttribution.includes(target)) {
      conditions.push(`Must include ${target} license notice and attribution`);
    }

    // Patent clauses
    if (target === "Apache-2.0" || source === "Apache-2.0") {
      conditions.push(
        "Apache-2.0 includes patent grant - be aware of patent implications"
      );
    }

    return conditions;
  }

  /**
   * Convert risk to numeric level for comparison
   */
  private riskLevel(risk: LicenseRisk): number {
    const levels: Record<LicenseRisk, number> = {
      low: 1,
      medium: 2,
      high: 3,
      critical: 4,
      unknown: 5,
    };
    return levels[risk] ?? 0;
  }

  /**
   * Generate recommendations based on analysis
   */
  private generateRecommendations(
    projectLicense: string,
    dependencies: { name: string; license: string }[],
    issues: LicenseIssue[]
  ): string[] {
    const recommendations: string[] = [];

    const hasErrors = issues.some((i) => i.severity === "error");
    const hasCopyleft = issues.some((i) => i.type === "copyleft-conflict");
    const hasUnknown = issues.some((i) => i.type === "unknown");

    if (hasErrors) {
      recommendations.push("Address license incompatibilities before release");
      recommendations.push(
        "Consider alternative packages with compatible licenses"
      );
    }

    if (hasCopyleft) {
      recommendations.push(
        "Review copyleft requirements - may need to release source code"
      );
      const projectDef =
        this.licenseEngine.getLicenseDefinition(projectLicense);
      if (
        projectDef?.type === "proprietary" ||
        projectDef?.type === "commercial"
      ) {
        recommendations.push(
          "Consider using LGPL-compatible alternatives for copyleft dependencies"
        );
      }
    }

    if (hasUnknown) {
      recommendations.push(
        "Consult legal counsel for unknown license combinations"
      );
      recommendations.push("Document license review decisions in your project");
    }

    // General recommendations
    recommendations.push("Maintain an up-to-date SBOM for compliance tracking");
    recommendations.push("Regularly audit dependencies for license changes");

    return recommendations;
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export {
  COMPATIBILITY_MATRIX,
  parseSPDXExpression,
  evaluateSPDXExpression,
  LicenseCompatibilityChecker,
};

export type {
  CompatibilityLevel,
  CompatibilityResult,
  ProjectLicenseAnalysis,
  LicenseIssue,
  SPDXExpressionNode,
};
