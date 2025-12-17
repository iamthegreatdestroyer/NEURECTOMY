"""
Phase 18F-3: Profiling Configuration Framework

This module provides all configuration for profiling execution across
Ryot, ΣLANG, ΣVAULT, and Agent services.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime
import json


class ProfilingMode(str, Enum):
    """Profiling tool selection"""
    PY_SPY = "py-spy"  # Sampling profiler, minimal overhead
    CPROFILE = "cprofile"  # Deterministic profiler, precise timing
    LINE_PROFILER = "line_profiler"  # Line-by-line profiling
    MEMORY_PROFILER = "memory_profiler"  # Memory usage tracking
    SCALENE = "scalene"  # Combined profiler (CPU + memory + GPU)


class BenchmarkSuite(str, Enum):
    """Benchmark categorization"""
    MICROBENCHMARKS = "microbenchmarks"  # <1 second each
    MACROBENCHMARKS = "macrobenchmarks"  # 5-20 minutes
    PROFILING = "profiling"  # 15-45 minutes


@dataclass
class ProfilingConfig:
    """Master configuration for profiling session"""
    
    # Session metadata
    session_id: str  # UUID for this profiling run
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    phase: str = "18F-3"
    day: int = 1  # Day of profiling (1-5)
    
    # Component under test
    component: str = "ryot"  # ryot, sigmalang, sigmavault, agents
    
    # Profiling settings
    profiler: ProfilingMode = ProfilingMode.PY_SPY
    duration_seconds: int = 300  # Total profiling duration
    iterations: int = 100  # Number of benchmark iterations
    warmup_iterations: int = 10  # Warmup runs before profiling
    
    # Output configuration
    output_dir: str = "results/phase_18f/"
    save_raw_profile: bool = True
    save_flame_graph: bool = True
    save_json_metrics: bool = True
    
    # Benchmark configuration
    suite: BenchmarkSuite = BenchmarkSuite.MICROBENCHMARKS
    benchmark_name: Optional[str] = None
    
    # Performance thresholds for alerts
    thresholds: Dict[str, Any] = field(default_factory=dict)
    
    # Environment
    isolation: bool = True  # Run benchmark in isolated process
    gc_enabled: bool = False  # Disable GC during profiling
    
    def to_json(self) -> str:
        """Serialize configuration to JSON"""
        return json.dumps({
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "phase": self.phase,
            "day": self.day,
            "component": self.component,
            "profiler": self.profiler.value,
            "duration_seconds": self.duration_seconds,
            "iterations": self.iterations,
            "suite": self.suite.value,
            "benchmark": self.benchmark_name,
        }, indent=2)


@dataclass
class BenchmarkConfig:
    """Configuration for individual benchmark"""
    
    name: str  # Benchmark identifier
    component: str  # Target component
    suite: BenchmarkSuite  # Category
    duration_seconds: int  # How long benchmark runs
    iterations: int = 100  # Iterations to run
    profiler: ProfilingMode = ProfilingMode.PY_SPY
    
    # Expected metrics
    expected_ttft_ms: Optional[float] = None  # For Ryot
    expected_throughput: Optional[float] = None  # Tokens/sec, MB/s, ops/sec
    expected_latency_p99_ms: Optional[float] = None  # p99 latency
    
    # Regression thresholds (% deviation from baseline is warning)
    regression_threshold: float = 10.0  # 10% slower is regression
    improvement_threshold: float = 5.0  # 5% faster is improvement


@dataclass
class BottleneckConfig:
    """Configuration for bottleneck detection"""
    
    min_self_time_pct: float = 1.0  # Only report functions >1% self time
    min_total_time_pct: float = 2.0  # Only report functions >2% total time
    top_n: int = 20  # Report top N bottlenecks
    
    # Weighting for bottleneck scoring
    self_time_weight: float = 0.4  # 40% weight to self time
    total_time_weight: float = 0.4  # 40% weight to total time
    call_count_weight: float = 0.2  # 20% weight to call count
    
    # Statistical significance
    min_samples: int = 100  # Minimum samples for statistical validity
    confidence_level: float = 0.95  # 95% confidence interval


# Component-specific configurations

RYOT_PROFILING_TARGETS = {
    "FirstTokenLatencyBenchmark": BenchmarkConfig(
        name="FirstTokenLatencyBenchmark",
        component="ryot",
        suite=BenchmarkSuite.MICROBENCHMARKS,
        duration_seconds=30,
        iterations=100,
        expected_ttft_ms=100.0,  # Target: <100ms
    ),
    "TokenGenerationBenchmark": BenchmarkConfig(
        name="TokenGenerationBenchmark",
        component="ryot",
        suite=BenchmarkSuite.MICROBENCHMARKS,
        duration_seconds=45,
        iterations=50,
        expected_throughput=50.0,  # Target: >50 tokens/sec
    ),
}

SIGMALANG_PROFILING_TARGETS = {
    "CompressionRatioBenchmark": BenchmarkConfig(
        name="CompressionRatioBenchmark",
        component="sigmalang",
        suite=BenchmarkSuite.MICROBENCHMARKS,
        duration_seconds=20,
        iterations=100,
        expected_throughput=3.0,  # Target: >3:1 ratio
    ),
    "CompressionThroughputBenchmark": BenchmarkConfig(
        name="CompressionThroughputBenchmark",
        component="sigmalang",
        suite=BenchmarkSuite.MICROBENCHMARKS,
        duration_seconds=30,
        iterations=50,
        expected_throughput=100.0,  # Target: >100 MB/s
    ),
}

SIGMAVAULT_PROFILING_TARGETS = {
    "RSUReadBenchmark": BenchmarkConfig(
        name="RSUReadBenchmark",
        component="sigmavault",
        suite=BenchmarkSuite.MICROBENCHMARKS,
        duration_seconds=20,
        iterations=1000,
        expected_latency_p99_ms=10.0,  # Target: <10ms p99
    ),
    "RSUWriteBenchmark": BenchmarkConfig(
        name="RSUWriteBenchmark",
        component="sigmavault",
        suite=BenchmarkSuite.MICROBENCHMARKS,
        duration_seconds=20,
        iterations=1000,
        expected_latency_p99_ms=20.0,  # Target: <20ms p99
    ),
}

AGENTS_PROFILING_TARGETS = {
    "AgentTaskLatencyBenchmark": BenchmarkConfig(
        name="AgentTaskLatencyBenchmark",
        component="agents",
        suite=BenchmarkSuite.MICROBENCHMARKS,
        duration_seconds=15,
        iterations=100,
        expected_latency_p99_ms=50.0,  # Target: <50ms p99
    ),
}

# Macrobenchmarks (longer running)
MACROBENCHMARKS = {
    "FullLLMInference": BenchmarkConfig(
        name="FullLLMInference",
        component="ryot",
        suite=BenchmarkSuite.MACROBENCHMARKS,
        duration_seconds=600,  # 10 minutes
        iterations=5,
        profiler=ProfilingMode.PY_SPY,
    ),
    "ScalingTest": BenchmarkConfig(
        name="ScalingTest",
        component="all",
        suite=BenchmarkSuite.MACROBENCHMARKS,
        duration_seconds=900,  # 15 minutes
        iterations=3,
    ),
    "EnduranceTest": BenchmarkConfig(
        name="EnduranceTest",
        component="all",
        suite=BenchmarkSuite.MACROBENCHMARKS,
        duration_seconds=3600,  # 1 hour
        iterations=1,
        profiler=ProfilingMode.MEMORY_PROFILER,
    ),
    "CollectiveWorkflow": BenchmarkConfig(
        name="CollectiveWorkflow",
        component="all",
        suite=BenchmarkSuite.MACROBENCHMARKS,
        duration_seconds=600,  # 10 minutes
        iterations=5,
    ),
}

# Profiling suites (detailed profiling)
PROFILING_SUITES = {
    "DetailedRyotInferenceProfiling": BenchmarkConfig(
        name="DetailedRyotInferenceProfiling",
        component="ryot",
        suite=BenchmarkSuite.PROFILING,
        duration_seconds=1800,  # 30 minutes
        iterations=10,
        profiler=ProfilingMode.CPROFILE,
    ),
    "SigmaVaultMemoryProfiling": BenchmarkConfig(
        name="SigmaVaultMemoryProfiling",
        component="sigmavault",
        suite=BenchmarkSuite.PROFILING,
        duration_seconds=1800,  # 30 minutes
        iterations=10,
        profiler=ProfilingMode.MEMORY_PROFILER,
    ),
    "AgentCommunicationProfiling": BenchmarkConfig(
        name="AgentCommunicationProfiling",
        component="agents",
        suite=BenchmarkSuite.PROFILING,
        duration_seconds=900,  # 15 minutes
        iterations=5,
        profiler=ProfilingMode.PY_SPY,
    ),
}


def get_benchmarks_for_day(day: int) -> List[BenchmarkConfig]:
    """Get benchmarks to run for specific day"""
    if day == 1:
        # Day 1: Ryot microbenchmarks
        return list(RYOT_PROFILING_TARGETS.values())
    elif day == 2:
        # Day 2: ΣLANG and ΣVAULT microbenchmarks
        return (
            list(SIGMALANG_PROFILING_TARGETS.values()) +
            list(SIGMAVAULT_PROFILING_TARGETS.values()) +
            list(AGENTS_PROFILING_TARGETS.values())
        )
    elif day == 3:
        # Day 3: Macrobenchmarks
        return list(MACROBENCHMARKS.values())
    elif day == 4:
        # Day 4: Detailed profiling
        return list(PROFILING_SUITES.values())
    else:
        return []


# Bottleneck detection configuration
BOTTLENECK_CONFIG = BottleneckConfig(
    min_self_time_pct=1.0,
    min_total_time_pct=2.0,
    top_n=20,
    self_time_weight=0.4,
    total_time_weight=0.4,
    call_count_weight=0.2,
)

# Optimization opportunities registry
OPTIMIZATION_OPPORTUNITIES = {
    "flash_attention": {
        "name": "Flash Attention",
        "description": "Implement Flash Attention for faster attention computation",
        "target_component": "ryot",
        "estimated_speedup": 1.4,  # 40% faster
        "implementation_days": 2,
        "complexity": 2,  # 1-5 scale
        "risk": 1,  # 1-5 scale
    },
    "binary_search_dict": {
        "name": "Binary Search Dictionary Lookup",
        "description": "Replace linear dictionary lookup with binary search",
        "target_component": "sigmalang",
        "estimated_speedup": 1.5,  # 50% faster
        "implementation_days": 1,
        "complexity": 1,
        "risk": 1,
    },
    "lru_cache_optimization": {
        "name": "Skip-List LRU Cache",
        "description": "Replace standard LRU with skip-list for O(log n) access",
        "target_component": "sigmavault",
        "estimated_speedup": 3.0,  # 3× faster
        "implementation_days": 3,
        "complexity": 3,
        "risk": 2,
    },
    "lock_free_queue": {
        "name": "Lock-Free Task Queue",
        "description": "Replace mutex-protected queue with lock-free implementation",
        "target_component": "agents",
        "estimated_speedup": 2.0,  # 2× faster
        "implementation_days": 4,
        "complexity": 4,
        "risk": 3,
    },
}


if __name__ == "__main__":
    # Print configuration for Day 1
    print("Day 1 Benchmarks:")
    for bench in get_benchmarks_for_day(1):
        print(f"  - {bench.name}: {bench.duration_seconds}s, {bench.iterations} iterations")
