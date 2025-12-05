/**
 * @fileoverview Performance Engine - Main Package Export
 * @module @neurectomy/performance-engine
 *
 * @elite-agent-collective @VELOCITY @CORE @NEURAL @VERTEX
 *
 * Enterprise-grade performance optimization engine providing:
 * - CPU and Memory Profiling with hot path analysis
 * - Query Optimization for SQL/GraphQL/NoSQL
 * - Auto-optimization with ML-driven learning
 * - Multi-tier intelligent caching
 * - Memory pooling for allocation efficiency
 *
 * @author NEURECTOMY Phase 5 - Performance Excellence
 * @version 1.0.0
 */

// ============================================================================
// Types Export
// ============================================================================

export type {
  ProfilerConfig,
  ProfilerResult,
  OptimizationConfig,
  OptimizationResult,
  CacheConfig,
  CacheStrategy,
  PoolingConfig,
  PerformanceMetrics,
  PerformanceThresholds,
  PerformanceBudget,
  PerformanceReport,
  PerformanceReportSection,
  PerformanceRecommendation,
  PerformanceAlert,
  AlertSeverity,
  PerformanceBaseline,
  BaselineComparison,
} from "./types.js";

// ============================================================================
// Profiler Module (@VELOCITY @CORE)
// ============================================================================

export {
  // CPU Profiler
  CPUProfiler,
  HotPathAnalyzer,
  type ProfilerOptions,
  type StackFrame,
  type ProfileNode,
  type ProfileSession,
  type PathFrame,
  type ExecutionPath,
  type DetectedPattern,
  type OptimizationRecommendation as CPUOptimizationRecommendation,
  type HotPathAnalysis,

  // Memory Profiler
  MemoryProfiler,
  type MemoryProfilerOptions,
  type AllocationSite,
  type HeapDiff,
} from "./profiler/index.js";

// ============================================================================
// Optimizer Module (@VELOCITY @VERTEX @NEURAL)
// ============================================================================

export {
  // Query Optimizer
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

  // Auto Optimizer
  AutoOptimizer,
  PatternRecognizer,
  MLPredictor,
  OptimizationScheduler,
  type OptimizationLevel,
  type AutoOptimizerConfig,
  type LearningPattern,
  type PatternCategory,
  type PatternRecognizerConfig,
  type RecognizedPattern,
  type PatternRecommendation,
  type PredictionModel,
  type ModelType,
  type PredictorConfig,
  type Prediction,
  type FeatureVector,
  type SchedulePriority,
  type SchedulerConfig,
  type ScheduledOptimization,
  type OptimizationTask,
  DEFAULT_AUTO_OPTIMIZER_CONFIG,
  DEFAULT_PATTERN_RECOGNIZER_CONFIG,
  DEFAULT_PREDICTOR_CONFIG,
  DEFAULT_SCHEDULER_CONFIG,
} from "./optimizer/index.js";

// ============================================================================
// Caching Module (@VELOCITY @CORE @NEURAL)
// ============================================================================

export {
  // Cache Manager
  CacheManager,
  BloomFilter,
  type EvictionPolicy,
  type CacheEntry,
  type CacheStats,
  type CacheTierConfig,
  type CacheManagerConfig,
  type CacheEventType,
  type CacheEvent,
  DEFAULT_CACHE_CONFIG,

  // Memory Pool
  ObjectPool,
  BufferPool,
  SlabAllocator,
  ArenaAllocator,
  type PoolStats,
  type PoolConfig,
  type Poolable,
  type BufferPoolConfig,
  type SlabConfig,
  type SlabStats,
  type ArenaConfig,
  type ArenaStats,
  DEFAULT_POOL_CONFIG,
  DEFAULT_BUFFER_POOL_CONFIG,
  DEFAULT_SLAB_CONFIG,
  DEFAULT_ARENA_CONFIG,
} from "./caching/index.js";

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create a full-featured performance engine instance
 */
export function createPerformanceEngine(options?: {
  profiler?: Partial<import("./profiler/index.js").ProfilerOptions>;
  queryOptimizer?: Partial<import("./optimizer/index.js").QueryOptimizerConfig>;
  autoOptimizer?: Partial<import("./optimizer/index.js").AutoOptimizerConfig>;
  cache?: Partial<import("./caching/index.js").CacheManagerConfig>;
}) {
  const { CPUProfiler } = require("./profiler/index.js");
  const { MemoryProfiler } = require("./profiler/index.js");
  const { QueryOptimizer, AutoOptimizer } = require("./optimizer/index.js");
  const { CacheManager } = require("./caching/index.js");

  return {
    cpuProfiler: new CPUProfiler(options?.profiler),
    memoryProfiler: new MemoryProfiler(options?.profiler),
    queryOptimizer: new QueryOptimizer(options?.queryOptimizer),
    autoOptimizer: new AutoOptimizer(options?.autoOptimizer),
    cacheManager: new CacheManager(options?.cache),
  };
}

/**
 * Create a lightweight profiling-only engine
 */
export function createProfiler(options?: {
  cpu?: Partial<import("./profiler/index.js").ProfilerOptions>;
  memory?: Partial<import("./profiler/index.js").MemoryProfilerOptions>;
}) {
  const { CPUProfiler, MemoryProfiler } = require("./profiler/index.js");

  return {
    cpu: new CPUProfiler(options?.cpu),
    memory: new MemoryProfiler(options?.memory),
  };
}

/**
 * Create an optimization engine
 */
export function createOptimizer(options?: {
  query?: Partial<import("./optimizer/index.js").QueryOptimizerConfig>;
  auto?: Partial<import("./optimizer/index.js").AutoOptimizerConfig>;
}) {
  const { QueryOptimizer, AutoOptimizer } = require("./optimizer/index.js");

  return {
    query: new QueryOptimizer(options?.query),
    auto: new AutoOptimizer(options?.auto),
  };
}

/**
 * Create a caching subsystem
 */
export function createCacheSystem<T>(options?: {
  manager?: Partial<import("./caching/index.js").CacheManagerConfig>;
  pool?: Partial<import("./caching/index.js").PoolConfig>;
}) {
  const { CacheManager, ObjectPool } = require("./caching/index.js");

  return {
    cache: new CacheManager<T>(options?.manager),
    pool: (factory: () => T & import("./caching/index.js").Poolable) =>
      new ObjectPool(factory, options?.pool),
  };
}
