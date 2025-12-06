/**
 * @fileoverview Optimizer Module Exports
 * @module @neurectomy/performance-engine/optimizer
 *
 * Agent Assignment: @VELOCITY @VERTEX @NEURAL
 *
 * Comprehensive query and auto-optimization systems:
 * - SQL/GraphQL/NoSQL query optimization
 * - Index recommendations
 * - Self-learning optimization engine
 * - Pattern recognition and prediction
 *
 * @author NEURECTOMY Phase 5 - Performance Excellence
 * @version 1.0.0
 */

// Query Optimization (@VELOCITY @VERTEX)
export { default as QueryOptimizer } from "./query-optimizer";
export type {
  QueryType,
  QueryAnalysis,
  QueryIssue,
  TableAccess,
  JoinInfo,
  IndexRecommendation,
} from "./query-optimizer";

// Auto-Optimization (@VELOCITY @NEURAL)
export { default as AutoOptimizer } from "./auto-optimizer";
export type {
  AutoOptimizerConfig,
  OptimizationAction,
  OptimizationHistory,
  LearningModel,
  ContextPattern,
  FailurePattern,
} from "./auto-optimizer";

// ============================================================================
// Unified Optimizer Factory
// ============================================================================

import QueryOptimizer from "./query-optimizer";
import AutoOptimizer, { type AutoOptimizerConfig } from "./auto-optimizer";

/**
 * Create a query optimizer instance
 */
export function createQueryOptimizer(): QueryOptimizer {
  return new QueryOptimizer();
}

/**
 * Create an auto-optimizer instance
 */
export function createAutoOptimizer(
  config?: Partial<AutoOptimizerConfig>
): AutoOptimizer {
  return new AutoOptimizer(config);
}

/**
 * Create both optimizers as a bundle
 */
export function createOptimizerBundle(config?: {
  auto?: Partial<AutoOptimizerConfig>;
}): {
  queryOptimizer: QueryOptimizer;
  autoOptimizer: AutoOptimizer;
} {
  return {
    queryOptimizer: new QueryOptimizer(),
    autoOptimizer: new AutoOptimizer(config?.auto),
  };
}
