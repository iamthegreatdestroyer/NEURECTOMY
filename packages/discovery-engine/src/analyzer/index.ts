/**
 * NEURECTOMY Discovery Engine - Analyzer Module
 *
 * Dependency analysis and vulnerability detection
 *
 * @packageDocumentation
 */

export {
  DependencyAnalyzer,
  createDependencyAnalyzer,
} from "./dependency-analyzer";

// Re-export types for convenience
export type {
  Dependency,
  DependencyType,
  PackageEcosystem,
  DependencyNode,
  DependencyAnalysis,
  OutdatedDependency,
  VulnerableDependency,
  DeprecatedDependency,
  DuplicateDependency,
  Vulnerability,
  VulnerabilitySeverity,
  VulnerabilitySource,
  UpgradeUrgency,
  AnalyzerConfig,
} from "../types";
