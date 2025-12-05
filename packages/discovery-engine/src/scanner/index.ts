/**
 * NEURECTOMY Discovery Engine - Scanner Module
 *
 * Repository scanning with GitHub API integration
 *
 * @packageDocumentation
 */

export {
  RepositoryScanner,
  createRepositoryScanner,
} from "./repository-scanner";

// Re-export types for convenience
export type {
  RepositoryInfo,
  RepositoryHealth,
  HealthFactor,
  HealthCategory,
  LicenseInfo,
  ScannerConfig,
  RepositorySearchQuery,
  RepositorySearchResult,
} from "../types";
