/**
 * @neurectomy/performance-engine - Type Definitions
 *
 * @elite-agent-collective @VELOCITY @AXIOM @CORE
 *
 * Comprehensive type system for performance monitoring, profiling,
 * optimization, memory management, and caching systems.
 */

import { z } from "zod";

// ============================================================================
// CORE PERFORMANCE TYPES
// ============================================================================

/**
 * Performance metric categories
 */
export const MetricCategorySchema = z.enum([
  "cpu",
  "memory",
  "io",
  "network",
  "latency",
  "throughput",
  "error_rate",
  "cache",
  "database",
  "api",
  "render",
  "bundle",
  "custom",
]);

export type MetricCategory = z.infer<typeof MetricCategorySchema>;

/**
 * Time resolution for metrics
 */
export const TimeResolutionSchema = z.enum([
  "microsecond",
  "millisecond",
  "second",
  "minute",
  "hour",
  "day",
]);

export type TimeResolution = z.infer<typeof TimeResolutionSchema>;

/**
 * Aggregation methods for metrics
 */
export const AggregationMethodSchema = z.enum([
  "sum",
  "avg",
  "min",
  "max",
  "p50",
  "p90",
  "p95",
  "p99",
  "p999",
  "count",
  "rate",
  "histogram",
]);

export type AggregationMethod = z.infer<typeof AggregationMethodSchema>;

/**
 * Performance severity levels
 */
export const SeverityLevelSchema = z.enum([
  "info",
  "warning",
  "critical",
  "fatal",
]);

export type SeverityLevel = z.infer<typeof SeverityLevelSchema>;

// ============================================================================
// METRIC SCHEMAS
// ============================================================================

/**
 * Base metric data point
 */
export const MetricDataPointSchema = z.object({
  timestamp: z.number(),
  value: z.number(),
  labels: z.record(z.string()).optional(),
  metadata: z.record(z.unknown()).optional(),
});

export type MetricDataPoint = z.infer<typeof MetricDataPointSchema>;

/**
 * Metric definition
 */
export const MetricDefinitionSchema = z.object({
  name: z.string(),
  description: z.string(),
  category: MetricCategorySchema,
  unit: z.string(),
  type: z.enum(["counter", "gauge", "histogram", "summary"]),
  labels: z.array(z.string()).optional(),
  buckets: z.array(z.number()).optional(),
  objectives: z.record(z.number()).optional(),
});

export type MetricDefinition = z.infer<typeof MetricDefinitionSchema>;

/**
 * Metric time series
 */
export const MetricTimeSeriesSchema = z.object({
  metric: MetricDefinitionSchema,
  dataPoints: z.array(MetricDataPointSchema),
  startTime: z.number(),
  endTime: z.number(),
  resolution: TimeResolutionSchema,
  aggregations: z.record(z.number()).optional(),
});

export type MetricTimeSeries = z.infer<typeof MetricTimeSeriesSchema>;

// ============================================================================
// PROFILER TYPES
// ============================================================================

/**
 * CPU profile sample
 */
export const CPUProfileSampleSchema = z.object({
  nodeId: z.number(),
  hitCount: z.number(),
  children: z.array(z.number()),
  functionName: z.string(),
  scriptName: z.string(),
  lineNumber: z.number(),
  columnNumber: z.number(),
  bailoutReason: z.string().optional(),
  selfTime: z.number(),
  totalTime: z.number(),
});

export type CPUProfileSample = z.infer<typeof CPUProfileSampleSchema>;

/**
 * CPU profile result
 */
export const CPUProfileResultSchema = z.object({
  id: z.string(),
  startTime: z.number(),
  endTime: z.number(),
  duration: z.number(),
  samples: z.array(CPUProfileSampleSchema),
  hotFunctions: z.array(
    z.object({
      functionName: z.string(),
      scriptName: z.string(),
      selfTime: z.number(),
      totalTime: z.number(),
      percentage: z.number(),
      callCount: z.number(),
    })
  ),
  flamegraphData: z.string().optional(),
  summary: z.object({
    totalSamples: z.number(),
    totalTime: z.number(),
    idlePercentage: z.number(),
    activePercentage: z.number(),
    gcPercentage: z.number(),
  }),
});

export type CPUProfileResult = z.infer<typeof CPUProfileResultSchema>;

/**
 * Memory allocation record
 */
export const MemoryAllocationSchema = z.object({
  id: z.string(),
  timestamp: z.number(),
  size: z.number(),
  type: z.string(),
  stackTrace: z.array(z.string()),
  retained: z.boolean(),
  retainedSize: z.number().optional(),
});

export type MemoryAllocation = z.infer<typeof MemoryAllocationSchema>;

/**
 * Heap snapshot node
 */
export const HeapSnapshotNodeSchema = z.object({
  nodeId: z.number(),
  name: z.string(),
  type: z.enum([
    "hidden",
    "array",
    "string",
    "object",
    "code",
    "closure",
    "regexp",
    "number",
    "native",
    "synthetic",
    "concatenated string",
    "sliced string",
    "symbol",
    "bigint",
  ]),
  selfSize: z.number(),
  retainedSize: z.number(),
  edgeCount: z.number(),
  childrenIds: z.array(z.number()),
  dominatorId: z.number().optional(),
});

export type HeapSnapshotNode = z.infer<typeof HeapSnapshotNodeSchema>;

/**
 * Heap snapshot result
 */
export const HeapSnapshotResultSchema = z.object({
  id: z.string(),
  timestamp: z.number(),
  totalSize: z.number(),
  totalNodes: z.number(),
  nodes: z.array(HeapSnapshotNodeSchema),
  dominatorTree: z.record(z.array(z.number())),
  retainerPaths: z.record(z.array(z.string())),
  summary: z.object({
    objectCount: z.number(),
    stringCount: z.number(),
    arrayCount: z.number(),
    closureCount: z.number(),
    codeCount: z.number(),
    externalSize: z.number(),
    nativeSize: z.number(),
  }),
  leakSuspects: z.array(
    z.object({
      nodeId: z.number(),
      name: z.string(),
      retainedSize: z.number(),
      reason: z.string(),
      confidence: z.number(),
    })
  ),
});

export type HeapSnapshotResult = z.infer<typeof HeapSnapshotResultSchema>;

/**
 * Trace event for execution tracing
 */
export const TraceEventSchema = z.object({
  name: z.string(),
  category: z.string(),
  phase: z.enum(["B", "E", "X", "I", "C", "b", "n", "e", "s", "t", "f"]),
  timestamp: z.number(),
  duration: z.number().optional(),
  pid: z.number(),
  tid: z.number(),
  args: z.record(z.unknown()).optional(),
  id: z.string().optional(),
});

export type TraceEvent = z.infer<typeof TraceEventSchema>;

/**
 * Execution trace result
 */
export const ExecutionTraceSchema = z.object({
  id: z.string(),
  name: z.string(),
  startTime: z.number(),
  endTime: z.number(),
  events: z.array(TraceEventSchema),
  criticalPath: z.array(
    z.object({
      name: z.string(),
      duration: z.number(),
      percentage: z.number(),
    })
  ),
  asyncTasks: z.array(
    z.object({
      name: z.string(),
      startTime: z.number(),
      endTime: z.number(),
      status: z.enum(["completed", "pending", "failed"]),
    })
  ),
});

export type ExecutionTrace = z.infer<typeof ExecutionTraceSchema>;

// ============================================================================
// OPTIMIZER TYPES
// ============================================================================

/**
 * Optimization opportunity
 */
export const OptimizationOpportunitySchema = z.object({
  id: z.string(),
  type: z.enum([
    "algorithm",
    "data_structure",
    "caching",
    "batching",
    "parallelization",
    "lazy_loading",
    "compression",
    "indexing",
    "pooling",
    "memoization",
    "code_splitting",
    "tree_shaking",
    "dead_code_elimination",
  ]),
  location: z.object({
    file: z.string(),
    line: z.number(),
    column: z.number().optional(),
    function: z.string().optional(),
  }),
  currentMetrics: z.object({
    time: z.number().optional(),
    memory: z.number().optional(),
    calls: z.number().optional(),
  }),
  estimatedImprovement: z.object({
    timeReduction: z.number().optional(),
    memoryReduction: z.number().optional(),
    description: z.string(),
  }),
  priority: z.enum(["low", "medium", "high", "critical"]),
  effort: z.enum(["trivial", "small", "medium", "large", "major"]),
  risk: z.enum(["none", "low", "medium", "high"]),
  suggestion: z.string(),
  codeExample: z.string().optional(),
  references: z.array(z.string()).optional(),
});

export type OptimizationOpportunity = z.infer<
  typeof OptimizationOpportunitySchema
>;

/**
 * Auto-optimization rule
 */
export const AutoOptimizationRuleSchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string(),
  trigger: z.object({
    metricName: z.string(),
    condition: z.enum(["gt", "gte", "lt", "lte", "eq", "neq"]),
    threshold: z.number(),
    duration: z.number().optional(),
  }),
  action: z.object({
    type: z.enum([
      "adjust_cache_size",
      "adjust_pool_size",
      "enable_compression",
      "enable_batching",
      "scale_workers",
      "trigger_gc",
      "clear_cache",
      "custom",
    ]),
    parameters: z.record(z.unknown()),
  }),
  cooldown: z.number(),
  enabled: z.boolean(),
  lastTriggered: z.number().optional(),
});

export type AutoOptimizationRule = z.infer<typeof AutoOptimizationRuleSchema>;

/**
 * Query optimization suggestion
 */
export const QueryOptimizationSchema = z.object({
  originalQuery: z.string(),
  optimizedQuery: z.string(),
  queryType: z.enum(["sql", "graphql", "mongodb", "elasticsearch", "custom"]),
  improvements: z.array(
    z.object({
      type: z.string(),
      description: z.string(),
      impact: z.enum(["low", "medium", "high"]),
    })
  ),
  estimatedSpeedup: z.number(),
  indexSuggestions: z.array(
    z.object({
      table: z.string(),
      columns: z.array(z.string()),
      type: z.enum(["btree", "hash", "gin", "gist", "brin"]),
      reason: z.string(),
    })
  ),
  warnings: z.array(z.string()),
});

export type QueryOptimization = z.infer<typeof QueryOptimizationSchema>;

/**
 * Bundle optimization result
 */
export const BundleOptimizationSchema = z.object({
  id: z.string(),
  timestamp: z.number(),
  originalSize: z.number(),
  optimizedSize: z.number(),
  compressionRatio: z.number(),
  chunks: z.array(
    z.object({
      name: z.string(),
      originalSize: z.number(),
      optimizedSize: z.number(),
      modules: z.array(z.string()),
      isAsync: z.boolean(),
      loadPriority: z.enum(["critical", "high", "medium", "low"]),
    })
  ),
  treeShaking: z.object({
    removedExports: z.number(),
    removedBytes: z.number(),
    unusedDependencies: z.array(z.string()),
  }),
  codeSplitting: z.object({
    suggestedSplits: z.array(
      z.object({
        module: z.string(),
        reason: z.string(),
        estimatedSavings: z.number(),
      })
    ),
    currentChunks: z.number(),
    optimalChunks: z.number(),
  }),
  duplicates: z.array(
    z.object({
      module: z.string(),
      occurrences: z.number(),
      wastedBytes: z.number(),
    })
  ),
});

export type BundleOptimization = z.infer<typeof BundleOptimizationSchema>;

// ============================================================================
// MEMORY MANAGEMENT TYPES
// ============================================================================

/**
 * Memory pool configuration
 */
export const MemoryPoolConfigSchema = z.object({
  name: z.string(),
  objectType: z.string(),
  initialSize: z.number(),
  maxSize: z.number(),
  growthFactor: z.number(),
  shrinkThreshold: z.number(),
  ttl: z.number().optional(),
  onAcquire: z.function().optional(),
  onRelease: z.function().optional(),
});

export type MemoryPoolConfig = z.infer<typeof MemoryPoolConfigSchema>;

/**
 * Memory pool statistics
 */
export const MemoryPoolStatsSchema = z.object({
  name: z.string(),
  totalObjects: z.number(),
  availableObjects: z.number(),
  acquiredObjects: z.number(),
  totalAcquisitions: z.number(),
  totalReleases: z.number(),
  growthEvents: z.number(),
  shrinkEvents: z.number(),
  missCount: z.number(),
  hitRate: z.number(),
  avgAcquireTime: z.number(),
  avgReleaseTime: z.number(),
  memoryUsage: z.number(),
});

export type MemoryPoolStats = z.infer<typeof MemoryPoolStatsSchema>;

/**
 * Garbage collection event
 */
export const GCEventSchema = z.object({
  timestamp: z.number(),
  type: z.enum(["scavenge", "mark_sweep", "incremental", "full"]),
  duration: z.number(),
  usedHeapBefore: z.number(),
  usedHeapAfter: z.number(),
  freedMemory: z.number(),
  totalHeapSize: z.number(),
  externalMemory: z.number(),
});

export type GCEvent = z.infer<typeof GCEventSchema>;

/**
 * Memory pressure level
 */
export const MemoryPressureLevelSchema = z.enum([
  "normal",
  "moderate",
  "critical",
  "out_of_memory",
]);

export type MemoryPressureLevel = z.infer<typeof MemoryPressureLevelSchema>;

/**
 * Memory statistics
 */
export const MemoryStatsSchema = z.object({
  timestamp: z.number(),
  heapUsed: z.number(),
  heapTotal: z.number(),
  heapLimit: z.number(),
  external: z.number(),
  arrayBuffers: z.number(),
  rss: z.number(),
  pressureLevel: MemoryPressureLevelSchema,
  gcStats: z.object({
    totalCollections: z.number(),
    totalPauseTime: z.number(),
    avgPauseTime: z.number(),
    maxPauseTime: z.number(),
    lastCollection: z.number().optional(),
  }),
  pools: z.array(MemoryPoolStatsSchema),
});

export type MemoryStats = z.infer<typeof MemoryStatsSchema>;

/**
 * Memory leak detection result
 */
export const MemoryLeakDetectionSchema = z.object({
  id: z.string(),
  timestamp: z.number(),
  detected: z.boolean(),
  confidence: z.number(),
  suspects: z.array(
    z.object({
      type: z.string(),
      location: z.string(),
      retainedSize: z.number(),
      retainerPath: z.array(z.string()),
      growthRate: z.number(),
      firstSeen: z.number(),
      evidence: z.array(z.string()),
    })
  ),
  recommendations: z.array(z.string()),
  memoryTimeline: z.array(
    z.object({
      timestamp: z.number(),
      heapUsed: z.number(),
      heapTotal: z.number(),
    })
  ),
});

export type MemoryLeakDetection = z.infer<typeof MemoryLeakDetectionSchema>;

// ============================================================================
// CACHE TYPES
// ============================================================================

/**
 * Cache eviction policy
 */
export const CacheEvictionPolicySchema = z.enum([
  "lru",
  "lfu",
  "fifo",
  "lifo",
  "ttl",
  "arc",
  "slru",
  "custom",
]);

export type CacheEvictionPolicy = z.infer<typeof CacheEvictionPolicySchema>;

/**
 * Cache tier
 */
export const CacheTierSchema = z.enum([
  "l1_memory",
  "l2_memory",
  "l3_disk",
  "l4_distributed",
]);

export type CacheTier = z.infer<typeof CacheTierSchema>;

/**
 * Cache configuration
 */
export const CacheConfigSchema = z.object({
  name: z.string(),
  tier: CacheTierSchema,
  maxSize: z.number(),
  maxItems: z.number().optional(),
  evictionPolicy: CacheEvictionPolicySchema,
  ttl: z.number().optional(),
  updateAgeOnGet: z.boolean().optional(),
  allowStale: z.boolean().optional(),
  staleMaxAge: z.number().optional(),
  compression: z.boolean().optional(),
  compressionThreshold: z.number().optional(),
  serializer: z.enum(["json", "msgpack", "protobuf", "custom"]).optional(),
  keyPrefix: z.string().optional(),
});

export type CacheConfig = z.infer<typeof CacheConfigSchema>;

/**
 * Cache entry metadata
 */
export const CacheEntryMetadataSchema = z.object({
  key: z.string(),
  size: z.number(),
  createdAt: z.number(),
  accessedAt: z.number(),
  updatedAt: z.number(),
  accessCount: z.number(),
  ttl: z.number().optional(),
  expiresAt: z.number().optional(),
  tags: z.array(z.string()).optional(),
  priority: z.number().optional(),
});

export type CacheEntryMetadata = z.infer<typeof CacheEntryMetadataSchema>;

/**
 * Cache statistics
 */
export const CacheStatsSchema = z.object({
  name: z.string(),
  tier: CacheTierSchema,
  hits: z.number(),
  misses: z.number(),
  hitRate: z.number(),
  sets: z.number(),
  deletes: z.number(),
  evictions: z.number(),
  expirations: z.number(),
  currentSize: z.number(),
  maxSize: z.number(),
  utilizationRate: z.number(),
  avgGetTime: z.number(),
  avgSetTime: z.number(),
  itemCount: z.number(),
  staleHits: z.number().optional(),
  compressionRatio: z.number().optional(),
});

export type CacheStats = z.infer<typeof CacheStatsSchema>;

/**
 * Multi-tier cache configuration
 */
export const MultiTierCacheConfigSchema = z.object({
  name: z.string(),
  tiers: z.array(CacheConfigSchema),
  writePolicy: z.enum(["write_through", "write_back", "write_around"]),
  readPolicy: z.enum(["read_through", "refresh_ahead"]),
  promotionThreshold: z.number().optional(),
  demotionThreshold: z.number().optional(),
});

export type MultiTierCacheConfig = z.infer<typeof MultiTierCacheConfigSchema>;

/**
 * Cache warming configuration
 */
export const CacheWarmingConfigSchema = z.object({
  enabled: z.boolean(),
  strategy: z.enum(["eager", "lazy", "predictive"]),
  priorityKeys: z.array(z.string()).optional(),
  warmingQueries: z
    .array(
      z.object({
        name: z.string(),
        query: z.string(),
        priority: z.number(),
        refreshInterval: z.number().optional(),
      })
    )
    .optional(),
  maxConcurrent: z.number().optional(),
  onStartup: z.boolean().optional(),
});

export type CacheWarmingConfig = z.infer<typeof CacheWarmingConfigSchema>;

// ============================================================================
// BENCHMARK TYPES
// ============================================================================

/**
 * Benchmark configuration
 */
export const BenchmarkConfigSchema = z.object({
  name: z.string(),
  description: z.string().optional(),
  iterations: z.number(),
  warmupIterations: z.number(),
  timeout: z.number(),
  minSamples: z.number().optional(),
  maxSamples: z.number().optional(),
  targetRelativeError: z.number().optional(),
  setup: z.function().optional(),
  teardown: z.function().optional(),
  beforeEach: z.function().optional(),
  afterEach: z.function().optional(),
});

export type BenchmarkConfig = z.infer<typeof BenchmarkConfigSchema>;

/**
 * Benchmark result
 */
export const BenchmarkResultSchema = z.object({
  name: z.string(),
  timestamp: z.number(),
  iterations: z.number(),
  samples: z.array(z.number()),
  stats: z.object({
    mean: z.number(),
    median: z.number(),
    stdDev: z.number(),
    min: z.number(),
    max: z.number(),
    p75: z.number(),
    p90: z.number(),
    p95: z.number(),
    p99: z.number(),
    marginOfError: z.number(),
    relativeMarginOfError: z.number(),
    opsPerSecond: z.number(),
  }),
  memory: z
    .object({
      avgHeapUsed: z.number(),
      peakHeapUsed: z.number(),
      avgHeapTotal: z.number(),
    })
    .optional(),
  comparison: z
    .object({
      baseline: z.string(),
      difference: z.number(),
      percentChange: z.number(),
      significant: z.boolean(),
    })
    .optional(),
});

export type BenchmarkResult = z.infer<typeof BenchmarkResultSchema>;

/**
 * Benchmark suite result
 */
export const BenchmarkSuiteResultSchema = z.object({
  name: z.string(),
  timestamp: z.number(),
  duration: z.number(),
  benchmarks: z.array(BenchmarkResultSchema),
  fastest: z.string(),
  slowest: z.string(),
  ranking: z.array(
    z.object({
      name: z.string(),
      opsPerSecond: z.number(),
      relativeSpeed: z.number(),
    })
  ),
  environment: z.object({
    platform: z.string(),
    arch: z.string(),
    nodeVersion: z.string(),
    v8Version: z.string(),
    cpuModel: z.string(),
    cpuCores: z.number(),
    totalMemory: z.number(),
  }),
});

export type BenchmarkSuiteResult = z.infer<typeof BenchmarkSuiteResultSchema>;

// ============================================================================
// ALERT & THRESHOLD TYPES
// ============================================================================

/**
 * Performance threshold
 */
export const PerformanceThresholdSchema = z.object({
  metricName: z.string(),
  warningThreshold: z.number(),
  criticalThreshold: z.number(),
  comparisonOperator: z.enum(["gt", "gte", "lt", "lte"]),
  windowSize: z.number(),
  minDataPoints: z.number().optional(),
  enabled: z.boolean(),
});

export type PerformanceThreshold = z.infer<typeof PerformanceThresholdSchema>;

/**
 * Performance alert
 */
export const PerformanceAlertSchema = z.object({
  id: z.string(),
  timestamp: z.number(),
  severity: SeverityLevelSchema,
  threshold: PerformanceThresholdSchema,
  currentValue: z.number(),
  metricHistory: z.array(MetricDataPointSchema),
  context: z.record(z.unknown()),
  acknowledged: z.boolean(),
  acknowledgedBy: z.string().optional(),
  acknowledgedAt: z.number().optional(),
  resolved: z.boolean(),
  resolvedAt: z.number().optional(),
  resolution: z.string().optional(),
});

export type PerformanceAlert = z.infer<typeof PerformanceAlertSchema>;

// ============================================================================
// PERFORMANCE REPORT TYPES
// ============================================================================

/**
 * Performance report section
 */
export const PerformanceReportSectionSchema = z.object({
  title: z.string(),
  summary: z.string(),
  metrics: z.array(
    z.object({
      name: z.string(),
      value: z.number(),
      unit: z.string(),
      trend: z.enum(["improving", "stable", "degrading"]).optional(),
      changePercent: z.number().optional(),
    })
  ),
  charts: z
    .array(
      z.object({
        type: z.enum(["line", "bar", "pie", "heatmap", "flamegraph"]),
        title: z.string(),
        data: z.unknown(),
      })
    )
    .optional(),
  recommendations: z.array(z.string()).optional(),
});

export type PerformanceReportSection = z.infer<
  typeof PerformanceReportSectionSchema
>;

/**
 * Full performance report
 */
export const PerformanceReportSchema = z.object({
  id: z.string(),
  title: z.string(),
  generatedAt: z.number(),
  periodStart: z.number(),
  periodEnd: z.number(),
  executiveSummary: z.string(),
  overallScore: z.number(),
  scoreBreakdown: z.object({
    cpu: z.number(),
    memory: z.number(),
    latency: z.number(),
    throughput: z.number(),
    reliability: z.number(),
  }),
  sections: z.array(PerformanceReportSectionSchema),
  alerts: z.array(PerformanceAlertSchema),
  optimizations: z.array(OptimizationOpportunitySchema),
  comparisonWithBaseline: z
    .object({
      baselineDate: z.number(),
      improvements: z.array(z.string()),
      regressions: z.array(z.string()),
      unchanged: z.array(z.string()),
    })
    .optional(),
});

export type PerformanceReport = z.infer<typeof PerformanceReportSchema>;

// ============================================================================
// SERVICE INTERFACES
// ============================================================================

/**
 * Profiler service interface
 */
export interface IProfilerService {
  startCPUProfile(name: string): Promise<void>;
  stopCPUProfile(): Promise<CPUProfileResult>;
  takeHeapSnapshot(): Promise<HeapSnapshotResult>;
  startTracing(categories: string[]): Promise<void>;
  stopTracing(): Promise<ExecutionTrace>;
  getMemoryAllocations(since: number): Promise<MemoryAllocation[]>;
}

/**
 * Optimizer service interface
 */
export interface IOptimizerService {
  analyzeCode(
    code: string,
    language: string
  ): Promise<OptimizationOpportunity[]>;
  optimizeQuery(
    query: string,
    type: QueryOptimization["queryType"]
  ): Promise<QueryOptimization>;
  analyzeBundle(bundleStats: unknown): Promise<BundleOptimization>;
  getAutoOptimizationRules(): AutoOptimizationRule[];
  addAutoOptimizationRule(
    rule: Omit<AutoOptimizationRule, "id">
  ): Promise<AutoOptimizationRule>;
  removeAutoOptimizationRule(ruleId: string): Promise<void>;
}

/**
 * Memory manager service interface
 */
export interface IMemoryManagerService {
  getStats(): MemoryStats;
  createPool<T>(config: MemoryPoolConfig): string;
  acquire<T>(poolName: string): T;
  release<T>(poolName: string, obj: T): void;
  triggerGC(): void;
  detectLeaks(): Promise<MemoryLeakDetection>;
  getGCHistory(limit: number): GCEvent[];
}

/**
 * Cache service interface
 */
export interface ICacheService<T = unknown> {
  get(key: string): Promise<T | undefined>;
  set(
    key: string,
    value: T,
    options?: { ttl?: number; tags?: string[] }
  ): Promise<void>;
  delete(key: string): Promise<boolean>;
  has(key: string): Promise<boolean>;
  clear(): Promise<void>;
  getStats(): CacheStats;
  getMetadata(key: string): Promise<CacheEntryMetadata | undefined>;
  invalidateByTags(tags: string[]): Promise<number>;
  warmup(): Promise<void>;
}

/**
 * Benchmark service interface
 */
export interface IBenchmarkService {
  run(
    name: string,
    fn: () => void | Promise<void>,
    config?: Partial<BenchmarkConfig>
  ): Promise<BenchmarkResult>;
  runSuite(
    name: string,
    benchmarks: Array<{ name: string; fn: () => void | Promise<void> }>,
    config?: Partial<BenchmarkConfig>
  ): Promise<BenchmarkSuiteResult>;
  compare(
    baseline: BenchmarkResult,
    current: BenchmarkResult
  ): BenchmarkResult["comparison"];
  saveBaseline(name: string, result: BenchmarkResult): Promise<void>;
  getBaseline(name: string): Promise<BenchmarkResult | undefined>;
}

/**
 * Performance monitoring service interface
 */
export interface IPerformanceMonitorService {
  registerMetric(definition: MetricDefinition): void;
  recordMetric(
    name: string,
    value: number,
    labels?: Record<string, string>
  ): void;
  getMetric(
    name: string,
    options: {
      startTime: number;
      endTime: number;
      resolution?: TimeResolution;
      aggregation?: AggregationMethod;
    }
  ): Promise<MetricTimeSeries>;
  setThreshold(threshold: PerformanceThreshold): void;
  getAlerts(options?: {
    severity?: SeverityLevel;
    acknowledged?: boolean;
  }): PerformanceAlert[];
  acknowledgeAlert(alertId: string, acknowledgedBy: string): Promise<void>;
  generateReport(
    periodStart: number,
    periodEnd: number
  ): Promise<PerformanceReport>;
}

// ============================================================================
// EXPORT ALL SCHEMAS FOR RUNTIME VALIDATION
// ============================================================================

export const PerformanceSchemas = {
  MetricCategory: MetricCategorySchema,
  TimeResolution: TimeResolutionSchema,
  AggregationMethod: AggregationMethodSchema,
  SeverityLevel: SeverityLevelSchema,
  MetricDataPoint: MetricDataPointSchema,
  MetricDefinition: MetricDefinitionSchema,
  MetricTimeSeries: MetricTimeSeriesSchema,
  CPUProfileSample: CPUProfileSampleSchema,
  CPUProfileResult: CPUProfileResultSchema,
  MemoryAllocation: MemoryAllocationSchema,
  HeapSnapshotNode: HeapSnapshotNodeSchema,
  HeapSnapshotResult: HeapSnapshotResultSchema,
  TraceEvent: TraceEventSchema,
  ExecutionTrace: ExecutionTraceSchema,
  OptimizationOpportunity: OptimizationOpportunitySchema,
  AutoOptimizationRule: AutoOptimizationRuleSchema,
  QueryOptimization: QueryOptimizationSchema,
  BundleOptimization: BundleOptimizationSchema,
  MemoryPoolConfig: MemoryPoolConfigSchema,
  MemoryPoolStats: MemoryPoolStatsSchema,
  GCEvent: GCEventSchema,
  MemoryPressureLevel: MemoryPressureLevelSchema,
  MemoryStats: MemoryStatsSchema,
  MemoryLeakDetection: MemoryLeakDetectionSchema,
  CacheEvictionPolicy: CacheEvictionPolicySchema,
  CacheTier: CacheTierSchema,
  CacheConfig: CacheConfigSchema,
  CacheEntryMetadata: CacheEntryMetadataSchema,
  CacheStats: CacheStatsSchema,
  MultiTierCacheConfig: MultiTierCacheConfigSchema,
  CacheWarmingConfig: CacheWarmingConfigSchema,
  BenchmarkConfig: BenchmarkConfigSchema,
  BenchmarkResult: BenchmarkResultSchema,
  BenchmarkSuiteResult: BenchmarkSuiteResultSchema,
  PerformanceThreshold: PerformanceThresholdSchema,
  PerformanceAlert: PerformanceAlertSchema,
  PerformanceReportSection: PerformanceReportSectionSchema,
  PerformanceReport: PerformanceReportSchema,
};
