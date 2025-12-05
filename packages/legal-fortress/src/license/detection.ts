/**
 * @fileoverview License Detection Engine
 * @module @neurectomy/legal-fortress/license/detection
 *
 * @agents @AEGIS @LINGUA - Compliance + NLP Specialists
 *
 * Detects and classifies software licenses using:
 * - Pattern matching for known licenses
 * - NLP-based text analysis
 * - Fuzzy matching for license variants
 * - SPDX identifier mapping
 */

import { DetectedLicense, LicenseType, LicenseRisk } from "../types";
import { v4 as uuidv4 } from "uuid";

// ============================================================================
// LICENSE DATABASE (@AEGIS)
// ============================================================================

/**
 * License definition with patterns and metadata
 */
export interface LicenseDefinition {
  spdxId: string;
  name: string;
  type: LicenseType;
  risk: LicenseRisk;
  osiApproved: boolean;
  copyleft: boolean;
  patterns: RegExp[];
  keywords: string[];
  url: string;
}

/**
 * Comprehensive license database
 * @agent @AEGIS - Compliance expertise
 */
export const LICENSE_DATABASE: LicenseDefinition[] = [
  // Permissive Licenses
  {
    spdxId: "MIT",
    name: "MIT License",
    type: "permissive",
    risk: "low",
    osiApproved: true,
    copyleft: false,
    patterns: [
      /permission\s+is\s+hereby\s+granted,?\s+free\s+of\s+charge/i,
      /MIT\s+License/i,
      /SPDX-License-Identifier:\s*MIT/i,
    ],
    keywords: ["mit", "permission", "without restriction", "sublicense"],
    url: "https://opensource.org/licenses/MIT",
  },
  {
    spdxId: "Apache-2.0",
    name: "Apache License 2.0",
    type: "permissive",
    risk: "low",
    osiApproved: true,
    copyleft: false,
    patterns: [
      /Apache\s+License,?\s+Version\s+2\.0/i,
      /SPDX-License-Identifier:\s*Apache-2\.0/i,
      /Licensed under the Apache License/i,
    ],
    keywords: ["apache", "contributor", "patent grant", "reproduction"],
    url: "https://www.apache.org/licenses/LICENSE-2.0",
  },
  {
    spdxId: "BSD-3-Clause",
    name: "BSD 3-Clause License",
    type: "permissive",
    risk: "low",
    osiApproved: true,
    copyleft: false,
    patterns: [
      /BSD\s+3-Clause/i,
      /SPDX-License-Identifier:\s*BSD-3-Clause/i,
      /Redistributions of source code must retain/i,
    ],
    keywords: ["bsd", "redistributions", "binary form", "neither the name"],
    url: "https://opensource.org/licenses/BSD-3-Clause",
  },
  {
    spdxId: "BSD-2-Clause",
    name: 'BSD 2-Clause "Simplified" License',
    type: "permissive",
    risk: "low",
    osiApproved: true,
    copyleft: false,
    patterns: [/BSD\s+2-Clause/i, /SPDX-License-Identifier:\s*BSD-2-Clause/i],
    keywords: ["bsd", "redistributions", "two clause", "simplified"],
    url: "https://opensource.org/licenses/BSD-2-Clause",
  },
  {
    spdxId: "ISC",
    name: "ISC License",
    type: "permissive",
    risk: "low",
    osiApproved: true,
    copyleft: false,
    patterns: [/ISC\s+License/i, /SPDX-License-Identifier:\s*ISC/i],
    keywords: ["isc", "permission to use", "as is"],
    url: "https://opensource.org/licenses/ISC",
  },
  {
    spdxId: "Unlicense",
    name: "The Unlicense",
    type: "public-domain",
    risk: "low",
    osiApproved: true,
    copyleft: false,
    patterns: [
      /This is free and unencumbered software/i,
      /SPDX-License-Identifier:\s*Unlicense/i,
      /released into the public domain/i,
    ],
    keywords: ["unlicense", "public domain", "unencumbered"],
    url: "https://unlicense.org/",
  },
  {
    spdxId: "CC0-1.0",
    name: "Creative Commons Zero v1.0",
    type: "public-domain",
    risk: "low",
    osiApproved: false,
    copyleft: false,
    patterns: [
      /CC0\s+1\.0/i,
      /SPDX-License-Identifier:\s*CC0-1\.0/i,
      /Creative\s+Commons\s+Zero/i,
    ],
    keywords: ["cc0", "creative commons", "zero", "public domain"],
    url: "https://creativecommons.org/publicdomain/zero/1.0/",
  },

  // Copyleft Licenses
  {
    spdxId: "GPL-3.0",
    name: "GNU General Public License v3.0",
    type: "copyleft",
    risk: "high",
    osiApproved: true,
    copyleft: true,
    patterns: [
      /GNU\s+GENERAL\s+PUBLIC\s+LICENSE\s*Version\s+3/i,
      /SPDX-License-Identifier:\s*GPL-3\.0/i,
      /Licensed under the GNU GPL v3/i,
    ],
    keywords: ["gpl", "gnu", "copyleft", "free software", "version 3"],
    url: "https://www.gnu.org/licenses/gpl-3.0.html",
  },
  {
    spdxId: "GPL-2.0",
    name: "GNU General Public License v2.0",
    type: "copyleft",
    risk: "high",
    osiApproved: true,
    copyleft: true,
    patterns: [
      /GNU\s+GENERAL\s+PUBLIC\s+LICENSE\s*Version\s+2/i,
      /SPDX-License-Identifier:\s*GPL-2\.0/i,
    ],
    keywords: ["gpl", "gnu", "version 2", "free software"],
    url: "https://www.gnu.org/licenses/gpl-2.0.html",
  },
  {
    spdxId: "LGPL-3.0",
    name: "GNU Lesser General Public License v3.0",
    type: "weak-copyleft",
    risk: "medium",
    osiApproved: true,
    copyleft: true,
    patterns: [
      /GNU\s+LESSER\s+GENERAL\s+PUBLIC\s+LICENSE/i,
      /SPDX-License-Identifier:\s*LGPL-3\.0/i,
    ],
    keywords: ["lgpl", "lesser", "library", "gnu"],
    url: "https://www.gnu.org/licenses/lgpl-3.0.html",
  },
  {
    spdxId: "AGPL-3.0",
    name: "GNU Affero General Public License v3.0",
    type: "copyleft",
    risk: "critical",
    osiApproved: true,
    copyleft: true,
    patterns: [
      /GNU\s+AFFERO\s+GENERAL\s+PUBLIC\s+LICENSE/i,
      /SPDX-License-Identifier:\s*AGPL-3\.0/i,
    ],
    keywords: ["agpl", "affero", "network", "gnu"],
    url: "https://www.gnu.org/licenses/agpl-3.0.html",
  },
  {
    spdxId: "MPL-2.0",
    name: "Mozilla Public License 2.0",
    type: "weak-copyleft",
    risk: "medium",
    osiApproved: true,
    copyleft: true,
    patterns: [
      /Mozilla\s+Public\s+License\s+Version\s+2\.0/i,
      /SPDX-License-Identifier:\s*MPL-2\.0/i,
    ],
    keywords: ["mpl", "mozilla", "file-level", "copyleft"],
    url: "https://www.mozilla.org/en-US/MPL/2.0/",
  },
  {
    spdxId: "EPL-2.0",
    name: "Eclipse Public License 2.0",
    type: "weak-copyleft",
    risk: "medium",
    osiApproved: true,
    copyleft: true,
    patterns: [
      /Eclipse\s+Public\s+License/i,
      /SPDX-License-Identifier:\s*EPL-2\.0/i,
    ],
    keywords: ["eclipse", "epl", "contributor"],
    url: "https://www.eclipse.org/legal/epl-2.0/",
  },

  // Proprietary/Commercial
  {
    spdxId: "PROPRIETARY",
    name: "Proprietary License",
    type: "proprietary",
    risk: "critical",
    osiApproved: false,
    copyleft: false,
    patterns: [
      /All\s+rights\s+reserved/i,
      /Proprietary\s+and\s+Confidential/i,
      /may\s+not\s+be\s+copied.*without.*permission/i,
    ],
    keywords: [
      "proprietary",
      "confidential",
      "all rights reserved",
      "trade secret",
    ],
    url: "",
  },
  {
    spdxId: "COMMERCIAL",
    name: "Commercial License",
    type: "commercial",
    risk: "critical",
    osiApproved: false,
    copyleft: false,
    patterns: [
      /Commercial\s+License/i,
      /requires?\s+a\s+commercial\s+license/i,
      /purchase.*license/i,
    ],
    keywords: ["commercial", "purchase", "license fee", "subscription"],
    url: "",
  },
];

// ============================================================================
// LICENSE DETECTION (@LINGUA)
// ============================================================================

/**
 * Detection result with confidence scoring
 */
export interface DetectionResult {
  detected: DetectedLicense[];
  unknownText?: string;
  analysisTime: number;
}

/**
 * Detection options
 */
export interface DetectionOptions {
  minConfidence?: number;
  checkKeywords?: boolean;
  fuzzyMatch?: boolean;
  maxResults?: number;
}

const DEFAULT_DETECTION_OPTIONS: DetectionOptions = {
  minConfidence: 0.5,
  checkKeywords: true,
  fuzzyMatch: true,
  maxResults: 5,
};

/**
 * License Detection Engine
 * @agent @AEGIS @LINGUA - Complete detection implementation
 */
export class LicenseDetectionEngine {
  private licenseDb: LicenseDefinition[];

  constructor(customLicenses?: LicenseDefinition[]) {
    this.licenseDb = [...LICENSE_DATABASE, ...(customLicenses ?? [])];
  }

  /**
   * Detect licenses in text content
   * @agent @LINGUA - NLP analysis
   */
  detect(
    text: string,
    options: DetectionOptions = DEFAULT_DETECTION_OPTIONS
  ): DetectionResult {
    const startTime = Date.now();
    const opts = { ...DEFAULT_DETECTION_OPTIONS, ...options };
    const detectedLicenses: DetectedLicense[] = [];

    // Normalize text for analysis
    const normalizedText = this.normalizeText(text);

    for (const license of this.licenseDb) {
      const score = this.calculateConfidence(normalizedText, license, opts);

      if (score >= (opts.minConfidence ?? 0.5)) {
        detectedLicenses.push({
          id: uuidv4(),
          spdxId: license.spdxId,
          name: license.name,
          confidence: score,
          location: this.findLocation(text, license),
          sourceFile: "",
          detectedAt: new Date(),
        });
      }
    }

    // Sort by confidence
    detectedLicenses.sort((a, b) => b.confidence - a.confidence);

    // Limit results
    const limited = detectedLicenses.slice(0, opts.maxResults);

    return {
      detected: limited,
      unknownText:
        limited.length === 0 ? this.extractUnknownLicense(text) : undefined,
      analysisTime: Date.now() - startTime,
    };
  }

  /**
   * Detect license from SPDX identifier
   */
  detectFromSPDX(identifier: string): DetectedLicense | null {
    const license = this.licenseDb.find(
      (l) => l.spdxId.toLowerCase() === identifier.toLowerCase()
    );

    if (!license) {
      return null;
    }

    return {
      id: uuidv4(),
      spdxId: license.spdxId,
      name: license.name,
      confidence: 1.0,
      location: { line: 0, column: 0 },
      sourceFile: "",
      detectedAt: new Date(),
    };
  }

  /**
   * Extract license from package.json
   */
  detectFromPackageJson(
    packageJson: string | Record<string, unknown>
  ): DetectedLicense | null {
    const pkg =
      typeof packageJson === "string" ? JSON.parse(packageJson) : packageJson;

    const licenseField = pkg["license"] || pkg["License"];

    if (typeof licenseField === "string") {
      // Simple SPDX expression
      return this.detectFromSPDX(licenseField);
    }

    if (typeof licenseField === "object" && licenseField !== null) {
      // Object format: { type: "MIT", url: "..." }
      const licenseObj = licenseField as Record<string, unknown>;
      const type = licenseObj["type"] as string | undefined;
      if (type) {
        return this.detectFromSPDX(type);
      }
    }

    return null;
  }

  /**
   * Detect licenses from file headers
   */
  detectFromHeader(sourceCode: string, maxLines: number = 50): DetectionResult {
    // Extract header (first N lines or until code starts)
    const lines = sourceCode.split("\n").slice(0, maxLines);
    const headerEnd = lines.findIndex((line) => {
      // Detect end of header comment
      const trimmed = line.trim();
      return (
        trimmed.length > 0 &&
        !trimmed.startsWith("//") &&
        !trimmed.startsWith("/*") &&
        !trimmed.startsWith("*") &&
        !trimmed.startsWith("#") &&
        !trimmed.startsWith("--")
      );
    });

    const headerLines = headerEnd > 0 ? lines.slice(0, headerEnd) : lines;
    const headerText = headerLines.join("\n");

    return this.detect(headerText);
  }

  /**
   * Get license definition by SPDX ID
   */
  getLicenseDefinition(spdxId: string): LicenseDefinition | null {
    return (
      this.licenseDb.find(
        (l) => l.spdxId.toLowerCase() === spdxId.toLowerCase()
      ) ?? null
    );
  }

  /**
   * Get all known license IDs
   */
  getKnownLicenses(): string[] {
    return this.licenseDb.map((l) => l.spdxId);
  }

  // ============================================================================
  // PRIVATE METHODS
  // ============================================================================

  /**
   * Normalize text for comparison
   */
  private normalizeText(text: string): string {
    return text
      .toLowerCase()
      .replace(/\s+/g, " ")
      .replace(/[^\w\s]/g, " ")
      .trim();
  }

  /**
   * Calculate confidence score for a license
   * @agent @LINGUA - Fuzzy matching
   */
  private calculateConfidence(
    normalizedText: string,
    license: LicenseDefinition,
    options: DetectionOptions
  ): number {
    let score = 0;
    let maxScore = 0;

    // Pattern matching (highest weight)
    for (const pattern of license.patterns) {
      maxScore += 40;
      if (pattern.test(normalizedText)) {
        score += 40;
      }
    }

    // Keyword matching
    if (options.checkKeywords) {
      for (const keyword of license.keywords) {
        maxScore += 10;
        const keywordNormalized = keyword.toLowerCase();

        if (options.fuzzyMatch) {
          // Fuzzy match - allow some variation
          if (this.fuzzyContains(normalizedText, keywordNormalized)) {
            score += 10;
          }
        } else {
          if (normalizedText.includes(keywordNormalized)) {
            score += 10;
          }
        }
      }
    }

    return maxScore > 0 ? score / maxScore : 0;
  }

  /**
   * Fuzzy string containment check
   */
  private fuzzyContains(haystack: string, needle: string): boolean {
    // Simple fuzzy match - could be enhanced with Levenshtein distance
    if (haystack.includes(needle)) {
      return true;
    }

    // Check for words in needle appearing in haystack
    const needleWords = needle.split(" ").filter((w) => w.length > 2);
    const matchedWords = needleWords.filter((w) => haystack.includes(w));

    return matchedWords.length / needleWords.length >= 0.7;
  }

  /**
   * Find location of license text
   */
  private findLocation(
    text: string,
    license: LicenseDefinition
  ): { line: number; column: number } {
    const lines = text.split("\n");

    for (let i = 0; i < lines.length; i++) {
      for (const pattern of license.patterns) {
        const match = lines[i]!.match(pattern);
        if (match) {
          return {
            line: i + 1,
            column: (match.index ?? 0) + 1,
          };
        }
      }
    }

    return { line: 0, column: 0 };
  }

  /**
   * Extract potential unknown license text
   */
  private extractUnknownLicense(text: string): string {
    // Look for common license indicators
    const indicators = [
      /license/i,
      /copyright/i,
      /permission/i,
      /granted/i,
      /redistribute/i,
    ];

    const lines = text.split("\n");
    const relevantLines: string[] = [];

    for (const line of lines) {
      if (indicators.some((ind) => ind.test(line))) {
        relevantLines.push(line.trim());
      }
    }

    return relevantLines.slice(0, 10).join("\n");
  }
}

// ============================================================================
// BATCH DETECTION (@AEGIS)
// ============================================================================

/**
 * Batch license detection for multiple files
 * @agent @AEGIS - Compliance scanning
 */
export class BatchLicenseDetector {
  private engine: LicenseDetectionEngine;

  constructor(engine?: LicenseDetectionEngine) {
    this.engine = engine ?? new LicenseDetectionEngine();
  }

  /**
   * Detect licenses in multiple files
   */
  detectInFiles(
    files: { path: string; content: string }[],
    options?: DetectionOptions
  ): Map<string, DetectionResult> {
    const results = new Map<string, DetectionResult>();

    for (const file of files) {
      // Determine detection strategy based on file type
      const fileName = file.path.toLowerCase();
      let result: DetectionResult;

      if (
        fileName.endsWith("license") ||
        fileName.endsWith("license.txt") ||
        fileName.endsWith("license.md")
      ) {
        result = this.engine.detect(file.content, options);
      } else if (fileName.endsWith("package.json")) {
        const detected = this.engine.detectFromPackageJson(file.content);
        result = {
          detected: detected ? [detected] : [],
          analysisTime: 0,
        };
      } else {
        // Source code - check header
        result = this.engine.detectFromHeader(file.content);
      }

      // Update source file
      for (const license of result.detected) {
        license.sourceFile = file.path;
      }

      results.set(file.path, result);
    }

    return results;
  }

  /**
   * Aggregate all detected licenses
   */
  aggregateLicenses(results: Map<string, DetectionResult>): {
    uniqueLicenses: Map<string, { license: DetectedLicense; files: string[] }>;
    unknownFiles: string[];
    summary: { spdxId: string; count: number; risk: LicenseRisk }[];
  } {
    const uniqueLicenses = new Map<
      string,
      { license: DetectedLicense; files: string[] }
    >();
    const unknownFiles: string[] = [];

    for (const [path, result] of results) {
      if (result.detected.length === 0) {
        unknownFiles.push(path);
        continue;
      }

      // Take highest confidence license per file
      const bestMatch = result.detected[0]!;

      if (uniqueLicenses.has(bestMatch.spdxId)) {
        uniqueLicenses.get(bestMatch.spdxId)!.files.push(path);
      } else {
        uniqueLicenses.set(bestMatch.spdxId, {
          license: bestMatch,
          files: [path],
        });
      }
    }

    // Build summary
    const summary: { spdxId: string; count: number; risk: LicenseRisk }[] = [];
    for (const [spdxId, data] of uniqueLicenses) {
      const definition = this.engine.getLicenseDefinition(spdxId);
      summary.push({
        spdxId,
        count: data.files.length,
        risk: definition?.risk ?? "unknown",
      });
    }

    // Sort by risk level
    const riskOrder: Record<string, number> = {
      critical: 0,
      high: 1,
      medium: 2,
      low: 3,
      unknown: 4,
    };
    summary.sort((a, b) => (riskOrder[a.risk] ?? 4) - (riskOrder[b.risk] ?? 4));

    return { uniqueLicenses, unknownFiles, summary };
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export {
  LICENSE_DATABASE,
  LicenseDetectionEngine,
  BatchLicenseDetector,
  DEFAULT_DETECTION_OPTIONS,
};

export type { LicenseDefinition, DetectionResult, DetectionOptions };
