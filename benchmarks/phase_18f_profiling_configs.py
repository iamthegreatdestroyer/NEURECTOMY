"""
Phase 18F: Advanced Profiling Benchmark Configurations

Comprehensive benchmark suite with profiling, sub-linear algorithms,
and performance optimization strategies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import json


class ProfilingMode(Enum):
    """Profiling modes for benchmark execution."""
    NONE = "none"  # No profiling, just timing
    PY_SPY = "py_spy"  # Sampling profiler
    CPROFILE = "cprofile"  # Deterministic profiler
    LINE_PROFILE = "line_profile"  # Line-level profiling
    MEMORY = "memory"  # Memory profiling
    COMBINED = "combined"  # All profilers


@dataclass
class ProfilingConfig:
    """Configuration for profiling behavior."""
    
    mode: ProfilingMode = ProfilingMode.PY_SPY
    enabled: bool = True
    
    # py-spy settings
    sample_rate: int = 100  # Hz (samples per second)
    duration: int = 60  # seconds
    
    # cProfile settings
    sort_by: str = "cumulative"  # Sort profile output by
    
    # line_profiler settings
    profile_functions: List[str] = field(default_factory=list)
    
    # Memory profiler settings
    peak_memory: bool = True
    memory_interval: float = 0.01  # seconds
    
    # Output
    output_format: str = "html"  # html, text, json, flamegraph
    save_artifacts: bool = True
    artifacts_dir: str = "profiling_results"


@dataclass
class RegressionConfig:
    """Regression detection configuration."""
    
    enabled: bool = True
    threshold_percent: float = 10.0  # 10% regression tolerance
    
    # Comparison
    compare_against: str = "baseline"  # baseline, previous, threshold
    baseline_file: Optional[str] = None
    
    # Alert
    alert_on_regression: bool = True
    alert_threshold_percent: float = 20.0  # Critical threshold
    
    # Statistical significance
    min_iterations: int = 3
    min_samples: int = 30


@dataclass
class BenchmarkConfig:
    """Master benchmark configuration."""
    
    # Identification
    name: str
    description: str = ""
    category: str = ""  # inference, compression, storage, agent
    tags: List[str] = field(default_factory=list)
    
    # Execution parameters
    warmup_iterations: int = 3
    benchmark_iterations: int = 10
    timeout_seconds: float = 300.0
    
    # Test parameters (parameterization)
    test_parameters: Dict[str, List[Any]] = field(default_factory=dict)
    
    # Success criteria
    targets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Profiling
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    
    # Regression detection
    regression: RegressionConfig = field(default_factory=RegressionConfig)
    
    # Output
    save_results: bool = True
    results_dir: str = "results"
    include_traces: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "tags": self.tags,
            "warmup_iterations": self.warmup_iterations,
            "benchmark_iterations": self.benchmark_iterations,
            "timeout_seconds": self.timeout_seconds,
            "test_parameters": self.test_parameters,
            "targets": self.targets,
            "profiling": {
                "mode": self.profiling.mode.value,
                "enabled": self.profiling.enabled,
                "sample_rate": self.profiling.sample_rate,
                "duration": self.profiling.duration,
            },
            "regression": {
                "enabled": self.regression.enabled,
                "threshold_percent": self.regression.threshold_percent,
            },
        }


# ============================================================================
# TIER 1: MICROBENCHMARKS (Fast, Precise)
# ============================================================================

MICROBENCHMARK_INFERENCE_TTFT = BenchmarkConfig(
    name="inference_ttft_micro",
    description="Time to first token (TTFT) - microbenchmark",
    category="inference",
    tags=["critical", "latency", "ttft"],
    
    warmup_iterations=2,
    benchmark_iterations=5,
    
    test_parameters={
        "prompt_length": [10],  # Single prompt size for micro
        "max_tokens": [1],  # Just one token for TTFT
    },
    
    targets={
        "ttft_ms": {
            "metric": "Time to first token (milliseconds)",
            "target": 100,
            "acceptable_range": [50, 200],
            "unit": "ms",
        },
        "p99_ttft_ms": {
            "metric": "P99 TTFT",
            "target": 120,
            "acceptable_range": [80, 250],
            "unit": "ms",
        },
    },
    
    profiling=ProfilingConfig(
        mode=ProfilingMode.PY_SPY,
        enabled=True,
        sample_rate=100,
        duration=10,
    ),
    
    regression=RegressionConfig(
        enabled=True,
        threshold_percent=15.0,  # 15% tolerance for micro
        alert_threshold_percent=25.0,
    ),
)

MICROBENCHMARK_INFERENCE_TOKENS_PER_SEC = BenchmarkConfig(
    name="inference_tokens_per_sec_micro",
    description="Token generation throughput - microbenchmark",
    category="inference",
    tags=["critical", "throughput"],
    
    warmup_iterations=2,
    benchmark_iterations=5,
    
    test_parameters={
        "prompt_length": [50],
        "max_tokens": [10],  # Small batch for micro
    },
    
    targets={
        "tokens_per_second": {
            "metric": "Tokens per second",
            "target": 50,
            "acceptable_range": [30, 100],
            "unit": "tokens/sec",
        },
        "p99_tokens_per_sec": {
            "metric": "P99 throughput",
            "target": 45,
            "acceptable_range": [25, 90],
            "unit": "tokens/sec",
        },
    },
    
    profiling=ProfilingConfig(
        mode=ProfilingMode.PY_SPY,
        enabled=True,
        sample_rate=100,
        duration=10,
    ),
)

MICROBENCHMARK_COMPRESSION_RATIO = BenchmarkConfig(
    name="compression_ratio_micro",
    description="Compression ratio for ΣLANG",
    category="compression",
    tags=["critical", "efficiency"],
    
    warmup_iterations=2,
    benchmark_iterations=5,
    
    test_parameters={
        "text_size": [1000],  # 1KB
    },
    
    targets={
        "compression_ratio": {
            "metric": "Compression ratio (original/compressed)",
            "target": 3.0,
            "acceptable_range": [2.5, 4.0],
            "unit": "ratio",
        },
        "p99_ratio": {
            "metric": "P99 compression ratio",
            "target": 2.8,
            "acceptable_range": [2.4, 3.8],
            "unit": "ratio",
        },
    },
    
    profiling=ProfilingConfig(
        mode=ProfilingMode.COMBINED,
        enabled=True,
        sample_rate=100,
    ),
)

MICROBENCHMARK_COMPRESSION_THROUGHPUT = BenchmarkConfig(
    name="compression_throughput_micro",
    description="Compression throughput for ΣLANG",
    category="compression",
    tags=["critical", "throughput"],
    
    warmup_iterations=2,
    benchmark_iterations=5,
    
    test_parameters={
        "text_size": [10000],  # 10KB
    },
    
    targets={
        "mb_per_second": {
            "metric": "Compression throughput MB/s",
            "target": 100,
            "acceptable_range": [50, 200],
            "unit": "MB/s",
        },
        "p99_mb_per_sec": {
            "metric": "P99 throughput",
            "target": 80,
            "acceptable_range": [40, 180],
            "unit": "MB/s",
        },
    },
    
    profiling=ProfilingConfig(
        mode=ProfilingMode.PY_SPY,
        enabled=True,
        sample_rate=100,
    ),
)

MICROBENCHMARK_STORAGE_WRITE = BenchmarkConfig(
    name="storage_write_micro",
    description="Storage write latency for ΣVAULT",
    category="storage",
    tags=["critical", "latency"],
    
    warmup_iterations=2,
    benchmark_iterations=5,
    
    test_parameters={
        "data_size": [1000],  # 1KB
    },
    
    targets={
        "write_latency_ms": {
            "metric": "Write latency (p50)",
            "target": 5,
            "acceptable_range": [2, 10],
            "unit": "ms",
        },
        "p99_write_latency_ms": {
            "metric": "Write latency (p99)",
            "target": 10,
            "acceptable_range": [5, 20],
            "unit": "ms",
        },
    },
    
    profiling=ProfilingConfig(
        mode=ProfilingMode.MEMORY,
        enabled=True,
        peak_memory=True,
    ),
)

MICROBENCHMARK_STORAGE_READ = BenchmarkConfig(
    name="storage_read_micro",
    description="Storage read latency for ΣVAULT",
    category="storage",
    tags=["critical", "latency"],
    
    warmup_iterations=2,
    benchmark_iterations=5,
    
    test_parameters={
        "data_size": [1000],  # 1KB
    },
    
    targets={
        "read_latency_ms": {
            "metric": "Read latency (p50)",
            "target": 3,
            "acceptable_range": [1, 5],
            "unit": "ms",
        },
        "p99_read_latency_ms": {
            "metric": "Read latency (p99)",
            "target": 5,
            "acceptable_range": [2, 10],
            "unit": "ms",
        },
    },
    
    profiling=ProfilingConfig(
        mode=ProfilingMode.PY_SPY,
        enabled=True,
        sample_rate=100,
    ),
)

MICROBENCHMARK_AGENT_TASK_LATENCY = BenchmarkConfig(
    name="agent_task_latency_micro",
    description="Agent collective task execution latency",
    category="agent",
    tags=["critical", "latency"],
    
    warmup_iterations=2,
    benchmark_iterations=5,
    
    test_parameters={
        "task_complexity": ["simple"],
        "num_agents": [1],
    },
    
    targets={
        "task_latency_ms": {
            "metric": "Task latency (p50)",
            "target": 20,
            "acceptable_range": [10, 50],
            "unit": "ms",
        },
        "p99_task_latency_ms": {
            "metric": "Task latency (p99)",
            "target": 50,
            "acceptable_range": [30, 100],
            "unit": "ms",
        },
    },
    
    profiling=ProfilingConfig(
        mode=ProfilingMode.PY_SPY,
        enabled=True,
        sample_rate=100,
    ),
)

# ============================================================================
# TIER 2: MACROBENCHMARKS (Realistic, Medium Duration)
# ============================================================================

MACROBENCHMARK_INFERENCE_FULL = BenchmarkConfig(
    name="inference_full",
    description="Full inference pipeline with realistic parameters",
    category="inference",
    tags=["realistic", "throughput"],
    
    warmup_iterations=3,
    benchmark_iterations=10,
    timeout_seconds=600,
    
    test_parameters={
        "prompt_length": [100, 500, 1000],
        "max_tokens": [100, 500],
        "batch_size": [1, 4, 8],
    },
    
    targets={
        "ttft_ms": {
            "metric": "TTFT across scenarios",
            "target": 100,
            "acceptable_range": [50, 200],
            "unit": "ms",
        },
        "throughput_tokens_per_sec": {
            "metric": "Average throughput",
            "target": 50,
            "acceptable_range": [30, 100],
            "unit": "tokens/sec",
        },
        "memory_mb": {
            "metric": "Peak memory usage",
            "target": 2048,
            "acceptable_range": [1024, 4096],
            "unit": "MB",
        },
    },
    
    profiling=ProfilingConfig(
        mode=ProfilingMode.COMBINED,
        enabled=True,
        sample_rate=100,
        duration=60,
    ),
    
    regression=RegressionConfig(
        enabled=True,
        threshold_percent=10.0,
        alert_threshold_percent=20.0,
    ),
)

MACROBENCHMARK_COMPRESSION_SCALING = BenchmarkConfig(
    name="compression_scaling",
    description="Compression performance at various data sizes",
    category="compression",
    tags=["throughput", "scaling"],
    
    warmup_iterations=3,
    benchmark_iterations=10,
    timeout_seconds=300,
    
    test_parameters={
        "text_size": [1000, 10000, 100000, 1000000],  # 1KB to 1MB
    },
    
    targets={
        "compression_ratio": {
            "metric": "Ratio (all sizes)",
            "target": 3.0,
            "acceptable_range": [2.5, 4.0],
            "unit": "ratio",
        },
        "throughput_mb_per_sec": {
            "metric": "Throughput (all sizes)",
            "target": 100,
            "acceptable_range": [50, 200],
            "unit": "MB/s",
        },
        "consistency_stddev": {
            "metric": "Throughput consistency (stddev)",
            "target": 10,
            "acceptable_range": [5, 25],
            "unit": "%",
        },
    },
    
    profiling=ProfilingConfig(
        mode=ProfilingMode.LINE_PROFILE,
        enabled=True,
        profile_functions=["compress", "decompress"],
    ),
)

MACROBENCHMARK_STORAGE_ENDURANCE = BenchmarkConfig(
    name="storage_endurance",
    description="Storage performance over sustained load",
    category="storage",
    tags=["endurance", "throughput"],
    
    warmup_iterations=2,
    benchmark_iterations=100,  # Many iterations for endurance
    timeout_seconds=600,
    
    test_parameters={
        "data_size": [1000, 10000, 100000],
        "operation_mix": ["write_heavy", "read_heavy", "balanced"],
    },
    
    targets={
        "write_throughput_ops_per_sec": {
            "metric": "Write ops/sec",
            "target": 1000,
            "acceptable_range": [500, 2000],
            "unit": "ops/sec",
        },
        "read_throughput_ops_per_sec": {
            "metric": "Read ops/sec",
            "target": 10000,
            "acceptable_range": [5000, 50000],
            "unit": "ops/sec",
        },
        "p99_latency_ms": {
            "metric": "P99 latency",
            "target": 10,
            "acceptable_range": [5, 50],
            "unit": "ms",
        },
    },
    
    profiling=ProfilingConfig(
        mode=ProfilingMode.MEMORY,
        enabled=True,
        peak_memory=True,
        memory_interval=0.1,
    ),
)

MACROBENCHMARK_AGENT_COLLECTIVE = BenchmarkConfig(
    name="agent_collective_workflow",
    description="Multi-agent workflow with realistic load",
    category="agent",
    tags=["multi-agent", "coordination"],
    
    warmup_iterations=3,
    benchmark_iterations=10,
    timeout_seconds=600,
    
    test_parameters={
        "num_agents": [4, 8, 16],
        "task_complexity": ["simple", "medium", "complex"],
        "concurrency": [1, 4, 8],
    },
    
    targets={
        "task_latency_p99_ms": {
            "metric": "Task latency P99",
            "target": 50,
            "acceptable_range": [30, 100],
            "unit": "ms",
        },
        "queue_depth_avg": {
            "metric": "Average queue depth",
            "target": 20,
            "acceptable_range": [10, 50],
            "unit": "tasks",
        },
        "throughput_tasks_per_sec": {
            "metric": "Task throughput",
            "target": 100,
            "acceptable_range": [50, 200],
            "unit": "tasks/sec",
        },
    },
    
    profiling=ProfilingConfig(
        mode=ProfilingMode.COMBINED,
        enabled=True,
        sample_rate=100,
        duration=60,
    ),
)

# ============================================================================
# TIER 3: PROFILING BENCHMARKS (Deep Analysis, Slow)
# ============================================================================

PROFILING_BENCHMARK_INFERENCE_DETAILED = BenchmarkConfig(
    name="inference_profiling_detailed",
    description="Detailed profiling of inference pipeline with full traces",
    category="inference",
    tags=["profiling", "analysis"],
    
    warmup_iterations=1,
    benchmark_iterations=3,
    timeout_seconds=900,
    
    test_parameters={
        "prompt_length": [100, 1000],
        "max_tokens": [100],
    },
    
    targets={
        "ttft_ms": {"target": 100},
        "tokens_per_sec": {"target": 50},
    },
    
    profiling=ProfilingConfig(
        mode=ProfilingMode.COMBINED,
        enabled=True,
        sample_rate=1000,  # High frequency sampling
        duration=300,
        output_format="flamegraph",
        save_artifacts=True,
    ),
    
    regression=RegressionConfig(
        enabled=False,  # Profiling benchmarks don't compare
    ),
)

PROFILING_BENCHMARK_STORAGE_MEMORY = BenchmarkConfig(
    name="storage_memory_profiling",
    description="Memory profiling of storage operations",
    category="storage",
    tags=["profiling", "memory"],
    
    warmup_iterations=1,
    benchmark_iterations=5,
    timeout_seconds=600,
    
    test_parameters={
        "data_size": [100000, 1000000],
        "num_operations": [1000],
    },
    
    targets={
        "peak_memory_mb": {"target": 512},
        "memory_per_op_kb": {"target": 50},
    },
    
    profiling=ProfilingConfig(
        mode=ProfilingMode.MEMORY,
        enabled=True,
        peak_memory=True,
        memory_interval=0.01,
        output_format="json",
        save_artifacts=True,
    ),
    
    regression=RegressionConfig(
        enabled=False,
    ),
)

PROFILING_BENCHMARK_AGENT_COMMUNICATION = BenchmarkConfig(
    name="agent_communication_profiling",
    description="Profile inter-agent communication and synchronization",
    category="agent",
    tags=["profiling", "communication"],
    
    warmup_iterations=1,
    benchmark_iterations=3,
    timeout_seconds=600,
    
    test_parameters={
        "num_agents": [8],
        "message_size": [100, 1000, 10000],
        "num_messages": [1000],
    },
    
    targets={
        "communication_latency_ms": {"target": 5},
        "queue_operations_per_sec": {"target": 100000},
    },
    
    profiling=ProfilingConfig(
        mode=ProfilingMode.COMBINED,
        enabled=True,
        sample_rate=1000,
        duration=120,
        output_format="json",
        save_artifacts=True,
    ),
    
    regression=RegressionConfig(
        enabled=False,
    ),
)

# ============================================================================
# BENCHMARK SUITES
# ============================================================================

class BenchmarkSuite:
    """Container for related benchmarks."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.benchmarks: List[BenchmarkConfig] = []
    
    def add(self, config: BenchmarkConfig):
        """Add benchmark to suite."""
        self.benchmarks.append(config)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert suite to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "benchmarks": [b.to_dict() for b in self.benchmarks],
        }


# Pre-built suites

SUITE_MICROBENCHMARKS = BenchmarkSuite(
    name="microbenchmarks",
    description="Fast, precise benchmarks for CI/CD (runtime: <1min)"
)
SUITE_MICROBENCHMARKS.add(MICROBENCHMARK_INFERENCE_TTFT)
SUITE_MICROBENCHMARKS.add(MICROBENCHMARK_INFERENCE_TOKENS_PER_SEC)
SUITE_MICROBENCHMARKS.add(MICROBENCHMARK_COMPRESSION_RATIO)
SUITE_MICROBENCHMARKS.add(MICROBENCHMARK_COMPRESSION_THROUGHPUT)
SUITE_MICROBENCHMARKS.add(MICROBENCHMARK_STORAGE_WRITE)
SUITE_MICROBENCHMARKS.add(MICROBENCHMARK_STORAGE_READ)
SUITE_MICROBENCHMARKS.add(MICROBENCHMARK_AGENT_TASK_LATENCY)

SUITE_MACROBENCHMARKS = BenchmarkSuite(
    name="macrobenchmarks",
    description="Realistic benchmarks for daily/PR runs (runtime: 5-20min)"
)
SUITE_MACROBENCHMARKS.add(MACROBENCHMARK_INFERENCE_FULL)
SUITE_MACROBENCHMARKS.add(MACROBENCHMARK_COMPRESSION_SCALING)
SUITE_MACROBENCHMARKS.add(MACROBENCHMARK_STORAGE_ENDURANCE)
SUITE_MACROBENCHMARKS.add(MACROBENCHMARK_AGENT_COLLECTIVE)

SUITE_PROFILING = BenchmarkSuite(
    name="profiling",
    description="Deep profiling for optimization sprints (runtime: 15-45min)"
)
SUITE_PROFILING.add(PROFILING_BENCHMARK_INFERENCE_DETAILED)
SUITE_PROFILING.add(PROFILING_BENCHMARK_STORAGE_MEMORY)
SUITE_PROFILING.add(PROFILING_BENCHMARK_AGENT_COMMUNICATION)


# ============================================================================
# BOTTLENECK DETECTION CONFIGURATION
# ============================================================================

@dataclass
class BottleneckDetectionConfig:
    """Configuration for automatic bottleneck detection."""
    
    # CPU bottleneck detection
    cpu_threshold_percent: float = 60.0  # >60% CPU with <20% I/O
    io_threshold_percent: float = 20.0
    
    # Memory bottleneck detection
    memory_threshold_percent: float = 50.0  # >50% memory used
    cache_miss_threshold_percent: float = 50.0  # >50% cache misses
    
    # Synchronization bottleneck
    lock_contention_threshold: float = 0.1  # >10% time in locks
    
    # Hotspot ranking
    self_time_weight: float = 0.4
    total_time_weight: float = 0.4
    call_count_weight: float = 0.2
    
    # ROI calculation
    min_complexity_estimate: float = 0.5  # Complexity score 0-1
    roi_threshold: float = 10.0  # ROI > 10 for optimization
    
    # Output
    top_n_bottlenecks: int = 10  # Report top 10 bottlenecks


BOTTLENECK_DETECTION_CONFIG = BottleneckDetectionConfig()


# ============================================================================
# OPTIMIZATION OPPORTUNITIES CONFIGURATION
# ============================================================================

@dataclass
class OptimizationOpportunity:
    """Describes an optimization opportunity."""
    
    name: str
    description: str
    category: str  # "algorithm", "datastructure", "cache", "parallelism", "sublinear"
    
    # Impact estimation
    estimated_speedup: float  # e.g., 2.0 for 2× speedup
    estimated_complexity: str  # "low", "medium", "high"
    
    # Applicability
    applicable_components: List[str]  # components where this could apply
    current_complexity: str  # e.g., "O(n²)"
    target_complexity: str  # e.g., "O(n log n)"
    
    # Implementation notes
    notes: str = ""
    references: List[str] = field(default_factory=list)


# Pre-configured opportunities for discovery

OPPORTUNITIES = {
    # Inference optimizations
    "kv_cache_pruning": OptimizationOpportunity(
        name="KV-Cache Pruning",
        description="Remove low-attention tokens from KV cache during generation",
        category="algorithm",
        estimated_speedup=1.3,
        estimated_complexity="medium",
        applicable_components=["ryot_inference"],
        current_complexity="O(n)",
        target_complexity="O(n * pruning_ratio)",
        notes="Reduces memory and computation",
    ),
    
    # Compression optimizations
    "stream_compression": OptimizationOpportunity(
        name="Streaming Compression",
        description="Process data in chunks instead of loading entire buffer",
        category="algorithm",
        estimated_speedup=1.2,
        estimated_complexity="low",
        applicable_components=["sigma_lang_compress"],
        current_complexity="O(n) memory",
        target_complexity="O(chunk_size) memory",
        notes="Reduces memory footprint significantly",
    ),
    
    # Storage optimizations
    "lru_cache_layer": OptimizationOpportunity(
        name="LRU Cache Layer",
        description="Add in-memory LRU cache for frequently accessed RSUs",
        category="cache",
        estimated_speedup=100.0,  # 100-1000× for cache hits
        estimated_complexity="low",
        applicable_components=["sigma_vault_read"],
        current_complexity="O(log n) disk access",
        target_complexity="O(1) memory access",
        notes="Huge improvement for read-heavy workloads",
    ),
    
    # Sub-linear algorithm opportunities
    "bloom_filter_negative_queries": OptimizationOpportunity(
        name="Bloom Filter for Negative Lookups",
        description="Use Bloom filter to quickly reject non-existent RSUs",
        category="sublinear",
        estimated_speedup=10.0,
        estimated_complexity="low",
        applicable_components=["sigma_vault_read"],
        current_complexity="O(log n) lookups",
        target_complexity="O(k) Bloom checks",
        notes="Great for 'not found' cases",
    ),
    
    # Parallelism opportunities
    "token_batch_generation": OptimizationOpportunity(
        name="Token Batch Generation",
        description="Generate multiple tokens in parallel during decode phase",
        category="parallelism",
        estimated_speedup=4.0,
        estimated_complexity="medium",
        applicable_components=["ryot_inference"],
        current_complexity="O(n) sequential",
        target_complexity="O(n/batch_size) parallel",
        notes="Linear speedup up to 8-16× on modern CPUs",
    ),
}


if __name__ == "__main__":
    # Example: Print benchmark configs as JSON
    import json
    
    config = MICROBENCHMARK_INFERENCE_TTFT.to_dict()
    print(json.dumps(config, indent=2, default=str))
    
    # Example: Print available opportunities
    print("\n\nOptimization Opportunities:")
    for name, opp in OPPORTUNITIES.items():
        print(f"  - {opp.name}: {opp.estimated_speedup}× speedup")
