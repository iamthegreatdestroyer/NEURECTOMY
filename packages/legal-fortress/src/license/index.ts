/**
 * @fileoverview License Compliance Module
 * @module @neurectomy/legal-fortress/license
 *
 * Comprehensive license compliance system:
 * - License detection with NLP analysis
 * - SBOM generation (SPDX, CycloneDX)
 * - License compatibility checking
 */

// Detection (@AEGIS @LINGUA)
export {
  LICENSE_DATABASE,
  LicenseDetectionEngine,
  BatchLicenseDetector,
  DEFAULT_DETECTION_OPTIONS,
} from "./detection";

export type {
  LicenseDefinition,
  DetectionResult,
  DetectionOptions,
} from "./detection";

// SBOM Generation (@AEGIS @FORGE)
export { SBOMGenerator, SBOMValidator, DEFAULT_SBOM_OPTIONS } from "./sbom";

export type {
  PackageInfo,
  SBOMOptions,
  SPDXDocument,
  SPDXPackage,
  SPDXRelationship,
  CycloneDXDocument,
  CycloneDXComponent,
  ValidationResult,
} from "./sbom";

// Compatibility (@AEGIS @AXIOM)
export {
  COMPATIBILITY_MATRIX,
  parseSPDXExpression,
  evaluateSPDXExpression,
  LicenseCompatibilityChecker,
} from "./compatibility";

export type {
  CompatibilityLevel,
  CompatibilityResult,
  ProjectLicenseAnalysis,
  LicenseIssue,
  SPDXExpressionNode,
} from "./compatibility";
