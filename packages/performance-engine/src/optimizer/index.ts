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
export {
  QueryOptimizer,
  PlanAnalyzer,
  IndexAdvisor,
  QueryRewriter,
  CostEstimator,
  type QueryType,
  type QueryPlan,
  type QueryPlanNode,
  type IndexRecommendation,
  type QueryOptimizationResult,
  type QueryOptimizerConfig,
  DEFAULT_QUERY_OPTIMIZER_CONFIG,
} from "./query-optimizer";

// Auto-Optimization (@VELOCITY @NEURAL)
export {
  AutoOptimizer,
  PatternLearner,
  OptimizationEngine,
  ResourcePredictor,
  type OptimizationPattern,
  type PatternMatch,
  type AutoOptimizationResult,
  type ResourcePrediction,
  type AutoOptimizerConfig,
  DEFAULT_AUTO_OPTIMIZER_CONFIG,
} from "./auto-optimizer";

// ============================================================================
// Unified Optimizer Factory
// ============================================================================

import { QueryOptimizer, QueryOptimizerConfig } from "./query-optimizer";
import { AutoOptimizer, AutoOptimizerConfig } from "./auto-optimizer";

/**
 * Create a query optimizer instance
 */
export function createQueryOptimizer(
  config?: Partial<QueryOptimizerConfig>
): QueryOptimizer {
  return new QueryOptimizer(config);
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
  query?: Partial<QueryOptimizerConfig>;
  auto?: Partial<AutoOptimizerConfig>;
}): {
  queryOptimizer: QueryOptimizer;
  autoOptimizer: AutoOptimizer;
} {
  return {
    queryOptimizer: new QueryOptimizer(config?.query),
    autoOptimizer: new AutoOptimizer(config?.auto),
  };
}
