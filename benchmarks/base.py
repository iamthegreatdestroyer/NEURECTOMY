"""
Benchmark Base Classes
======================
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import time
import statistics


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    
    # Run settings
    warmup_iterations: int = 3
    benchmark_iterations: int = 10
    timeout_seconds: float = 300.0
    
    # Output
    verbose: bool = True
    save_results: bool = True
    results_dir: str = "results"
    
    # Comparison
    baseline_file: Optional[str] = None
    regression_threshold: float = 0.10  # 10% regression triggers warning


@dataclass
class BenchmarkResult:
    """Result from a single benchmark."""
    
    name: str
    iterations: int
    
    # Timing (milliseconds)
    total_time_ms: float
    min_time_ms: float
    max_time_ms: float
    mean_time_ms: float
    median_time_ms: float
    stddev_ms: float
    
    # Percentiles
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    
    # Throughput
    operations_per_second: float
    
    # Custom metrics
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "timing": {
                "total_ms": self.total_time_ms,
                "min_ms": self.min_time_ms,
                "max_ms": self.max_time_ms,
                "mean_ms": self.mean_time_ms,
                "median_ms": self.median_time_ms,
                "stddev_ms": self.stddev_ms,
            },
            "percentiles": {
                "p50_ms": self.p50_ms,
                "p90_ms": self.p90_ms,
                "p95_ms": self.p95_ms,
                "p99_ms": self.p99_ms,
            },
            "throughput": {
                "ops_per_second": self.operations_per_second,
            },
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
        }


class Benchmark(ABC):
    """Base class for all benchmarks."""
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self._timings: List[float] = []
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Benchmark name."""
        pass
    
    @abstractmethod
    def setup(self) -> None:
        """Setup before benchmark runs."""
        pass
    
    @abstractmethod
    def run_iteration(self) -> Dict[str, Any]:
        """Run a single benchmark iteration. Returns custom metrics."""
        pass
    
    @abstractmethod
    def teardown(self) -> None:
        """Cleanup after benchmark runs."""
        pass
    
    def execute(self) -> BenchmarkResult:
        """Execute the full benchmark."""
        self._timings = []
        all_metrics: List[Dict[str, Any]] = []
        
        # Setup
        self.setup()
        
        # Warmup
        if self.config.verbose:
            print(f"  Warming up ({self.config.warmup_iterations} iterations)...")
        
        for _ in range(self.config.warmup_iterations):
            self.run_iteration()
        
        # Benchmark
        if self.config.verbose:
            print(f"  Running benchmark ({self.config.benchmark_iterations} iterations)...")
        
        for i in range(self.config.benchmark_iterations):
            start = time.perf_counter()
            metrics = self.run_iteration()
            elapsed = (time.perf_counter() - start) * 1000  # ms
            
            self._timings.append(elapsed)
            all_metrics.append(metrics)
            
            if self.config.verbose:
                print(f"    Iteration {i+1}: {elapsed:.2f}ms")
        
        # Teardown
        self.teardown()
        
        # Calculate results
        return self._calculate_results(all_metrics)
    
    def _calculate_results(self, all_metrics: List[Dict[str, Any]]) -> BenchmarkResult:
        """Calculate benchmark results from timings."""
        sorted_timings = sorted(self._timings)
        n = len(sorted_timings)
        
        # Aggregate custom metrics
        aggregated_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m.get(key, 0) for m in all_metrics if isinstance(m.get(key), (int, float))]
                if values:
                    aggregated_metrics[f"{key}_mean"] = statistics.mean(values)
                    aggregated_metrics[f"{key}_total"] = sum(values)
        
        return BenchmarkResult(
            name=self.name,
            iterations=n,
            total_time_ms=sum(self._timings),
            min_time_ms=min(self._timings),
            max_time_ms=max(self._timings),
            mean_time_ms=statistics.mean(self._timings),
            median_time_ms=statistics.median(self._timings),
            stddev_ms=statistics.stdev(self._timings) if n > 1 else 0.0,
            p50_ms=sorted_timings[int(n * 0.50)],
            p90_ms=sorted_timings[int(n * 0.90)] if n >= 10 else sorted_timings[-1],
            p95_ms=sorted_timings[int(n * 0.95)] if n >= 20 else sorted_timings[-1],
            p99_ms=sorted_timings[int(n * 0.99)] if n >= 100 else sorted_timings[-1],
            operations_per_second=1000.0 / statistics.mean(self._timings) if self._timings else 0,
            metrics=aggregated_metrics,
        )
