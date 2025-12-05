/**
 * NEURECTOMY Discovery Engine
 *
 * @VANGUARD @PRISM @ORACLE - Comprehensive open source discovery platform
 *
 * Features:
 * - Repository scanning with GitHub API integration
 * - Dependency analysis with vulnerability detection
 * - ML-powered library recommendations
 * - Technology radar generation
 *
 * @packageDocumentation
 */

// Scanner
export { RepositoryScanner, createRepositoryScanner } from "./scanner";

// Analyzer
export { DependencyAnalyzer, createDependencyAnalyzer } from "./analyzer";

// Recommendations
export {
  RecommendationEngine,
  createRecommendationEngine,
} from "./recommendations";

// Types
export * from "./types";
