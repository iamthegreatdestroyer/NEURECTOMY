/**
 * @neurectomy/performance-engine - Profiler Module
 *
 * @elite-agent-collective @VELOCITY @CORE
 *
 * High-precision profiling for CPU, memory, and execution tracing.
 */

export {
  CPUProfiler,
  HotPathAnalyzer,
  type ProfilerOptions,
  type StackFrame,
  type ProfileNode,
  type ProfileSession,
  type PathFrame,
  type ExecutionPath,
  type DetectedPattern,
  type OptimizationRecommendation,
  type HotPathAnalysis,
} from "./cpu-profiler.js";

export {
  MemoryProfiler,
  type MemoryProfilerOptions,
  type AllocationSite,
  type HeapDiff,
} from "./memory-profiler.js";
