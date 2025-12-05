/**
 * NEURECTOMY Discovery Engine - Recommendations Module
 *
 * ML-powered library recommendations and technology radar
 *
 * @packageDocumentation
 */

export {
  RecommendationEngine,
  createRecommendationEngine,
} from "./recommendation-engine";

// Re-export types for convenience
export type {
  LibraryRecommendation,
  PackageInfo,
  PackageComparison,
  ComparisonMetric,
  RecommendationReason,
  MigrationDifficulty,
  TypeScriptSupport,
  RecommendationConfig,
  PackageSearchQuery,
  PackageSearchResult,
  TechnologyRadar,
  RadarEntry,
  RadarQuadrant,
  RadarRing,
  RadarMovement,
} from "../types";
