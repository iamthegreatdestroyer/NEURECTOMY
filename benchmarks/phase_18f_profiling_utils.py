"""
Phase 18F: Profiling Utilities and Analysis Tools

Tools for running benchmarks with profiling, analyzing bottlenecks,
and detecting regressions.
"""

import time
import psutil
import json
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import statistics


@dataclass
class ProfileResult:
    """Result from profiling a benchmark run."""
    
    benchmark_name: str
    
    # Timing statistics
    mean_time_ms: float
    median_time_ms: float
    stddev_ms: float
    min_time_ms: float
    max_time_ms: float
    
    # Percentiles
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    
    # Throughput
    operations_per_second: float
    
    # Memory
    peak_memory_mb: float
    avg_memory_mb: float
    
    # Profiling artifacts
    profile_file: Optional[str] = None
    flame_graph_file: Optional[str] = None
    
    # Metadata
    num_iterations: int = 0
    profiling_mode: str = "none"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class BenchmarkProfiler:
    """Profiles benchmark execution with multiple profiling backends."""
    
    def __init__(self, mode: str = "py_spy", enabled: bool = True):
        self.mode = mode
        self.enabled = enabled
        self.start_time = None
        self.measurements = []
        self.memory_samples = []
    
    def start(self):
        """Start profiling."""
        if not self.enabled:
            return
        
        self.start_time = time.perf_counter()
        self.measurements = []
        self.memory_samples = []
    
    def record_measurement(self, elapsed_ms: float):
        """Record a single measurement."""
        self.measurements.append(elapsed_ms)
        
        # Sample memory usage
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_samples.append(memory_mb)
        except:
            pass
    
    def stop(self) -> ProfileResult:
        """Stop profiling and return results."""
        if not self.measurements:
            return None
        
        # Calculate statistics
        mean = statistics.mean(self.measurements)
        median = statistics.median(self.measurements)
        stdev = statistics.stdev(self.measurements) if len(self.measurements) > 1 else 0
        
        sorted_times = sorted(self.measurements)
        p50 = sorted_times[int(len(sorted_times) * 0.50)]
        p90 = sorted_times[int(len(sorted_times) * 0.90)]
        p95 = sorted_times[int(len(sorted_times) * 0.95)]
        p99 = sorted_times[int(len(sorted_times) * 0.99)]
        
        # Memory statistics
        peak_memory = max(self.memory_samples) if self.memory_samples else 0
        avg_memory = statistics.mean(self.memory_samples) if self.memory_samples else 0
        
        # Throughput (ops/sec)
        total_time_sec = sum(self.measurements) / 1000
        throughput = len(self.measurements) / total_time_sec if total_time_sec > 0 else 0
        
        return ProfileResult(
            benchmark_name="",
            mean_time_ms=mean,
            median_time_ms=median,
            stddev_ms=stdev,
            min_time_ms=min(self.measurements),
            max_time_ms=max(self.measurements),
            p50_ms=p50,
            p90_ms=p90,
            p95_ms=p95,
            p99_ms=p99,
            operations_per_second=throughput,
            peak_memory_mb=peak_memory,
            avg_memory_mb=avg_memory,
            num_iterations=len(self.measurements),
            profiling_mode=self.mode,
        )


class BottleneckAnalyzer:
    """Analyzes profiling data to identify bottlenecks."""
    
    @staticmethod
    def analyze_function_profile(profile_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze function profile to rank bottlenecks.
        
        Expected profile_data format:
        {
            "functions": [
                {
                    "name": "function_name",
                    "self_time_ms": 100,
                    "total_time_ms": 500,
                    "call_count": 1000,
                }
            ]
        }
        """
        
        functions = profile_data.get("functions", [])
        if not functions:
            return []
        
        # Calculate metrics
        total_time = sum(f.get("total_time_ms", 0) for f in functions)
        
        ranked = []
        for func in functions:
            self_time_ms = func.get("self_time_ms", 0)
            total_time_ms = func.get("total_time_ms", 0)
            call_count = func.get("call_count", 1)
            
            self_time_pct = (self_time_ms / total_time * 100) if total_time > 0 else 0
            total_time_pct = (total_time_ms / total_time * 100) if total_time > 0 else 0
            call_count_pct = call_count / sum(f.get("call_count", 1) for f in functions) * 100
            
            # ROI score: weighted combination of metrics
            score = (
                self_time_pct * 0.4 +
                total_time_pct * 0.4 +
                call_count_pct * 0.2
            )
            
            # Time per call
            time_per_call_us = (self_time_ms / call_count * 1000) if call_count > 0 else 0
            
            ranked.append({
                "name": func.get("name"),
                "self_time_ms": self_time_ms,
                "self_time_pct": self_time_pct,
                "total_time_ms": total_time_ms,
                "total_time_pct": total_time_pct,
                "call_count": call_count,
                "time_per_call_us": time_per_call_us,
                "score": score,
            })
        
        # Sort by score
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked
    
    @staticmethod
    def identify_bottleneck_category(profile_data: Dict[str, Any]) -> str:
        """
        Identify bottleneck category: cpu, memory, io, or synchronization.
        """
        
        cpu_pct = profile_data.get("cpu_percent", 0)
        memory_pct = profile_data.get("memory_percent", 0)
        io_wait_pct = profile_data.get("io_wait_percent", 0)
        lock_time_pct = profile_data.get("lock_time_percent", 0)
        
        if cpu_pct > 60 and io_wait_pct < 20:
            return "CPU-bound"
        elif memory_pct > 50 or profile_data.get("cache_miss_rate", 0) > 0.5:
            return "Memory-bound"
        elif io_wait_pct > 30:
            return "I/O-bound"
        elif lock_time_pct > 0.1:
            return "Synchronization-bound"
        else:
            return "Balanced"
    
    @staticmethod
    def estimate_speedup_potential(ranked_functions: List[Dict]) -> List[Dict]:
        """
        Estimate speedup potential for top functions.
        """
        
        results = []
        for i, func in enumerate(ranked_functions[:10]):
            self_time_pct = func["self_time_pct"]
            
            # Amdahl's law speedup estimation
            # If we optimize function to 50% of current time
            current_fraction = self_time_pct / 100
            optimized_fraction = current_fraction * 0.5  # 2× optimization
            remaining = 1 - current_fraction
            
            speedup_2x = 1 / (optimized_fraction + remaining)
            
            # If we 10× optimize
            optimized_fraction_10x = current_fraction / 10
            speedup_10x = 1 / (optimized_fraction_10x + remaining)
            
            results.append({
                "function": func["name"],
                "self_time_pct": self_time_pct,
                "speedup_if_2x_optimized": speedup_2x,
                "speedup_if_10x_optimized": speedup_10x,
                "priority": "HIGH" if self_time_pct > 10 else "MEDIUM" if self_time_pct > 5 else "LOW",
            })
        
        return results


class RegressionDetector:
    """Detects performance regressions."""
    
    @staticmethod
    def compare_to_baseline(
        current: ProfileResult,
        baseline: ProfileResult,
        threshold_percent: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Compare current results to baseline.
        
        Returns regression status and impact assessment.
        """
        
        if not baseline:
            return {
                "status": "no_baseline",
                "current_value": current.mean_time_ms,
            }
        
        # Calculate regression
        baseline_mean = baseline.mean_time_ms
        current_mean = current.mean_time_ms
        
        regression_pct = ((current_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean > 0 else 0
        
        # Determine status
        if regression_pct > threshold_percent:
            status = "regression"
        elif regression_pct < -threshold_percent:
            status = "improvement"
        else:
            status = "stable"
        
        # Statistical significance (if we have std dev)
        # Use simple t-test approximation
        is_significant = False
        if baseline.stddev_ms > 0 and current.stddev_ms > 0:
            # If regression > 2 stddevs, it's likely significant
            combined_stddev = (baseline.stddev_ms + current.stddev_ms) / 2
            if abs(current_mean - baseline_mean) > 2 * combined_stddev:
                is_significant = True
        
        return {
            "status": status,
            "regression_percent": regression_pct,
            "baseline_mean_ms": baseline_mean,
            "current_mean_ms": current_mean,
            "difference_ms": current_mean - baseline_mean,
            "threshold_percent": threshold_percent,
            "is_significant": is_significant,
            "p99_baseline_ms": baseline.p99_ms,
            "p99_current_ms": current.p99_ms,
            "p99_regression_percent": ((current.p99_ms - baseline.p99_ms) / baseline.p99_ms * 100) if baseline.p99_ms > 0 else 0,
        }
    
    @staticmethod
    def detect_memory_regression(
        current: ProfileResult,
        baseline: ProfileResult,
        threshold_percent: float = 20.0,
    ) -> Dict[str, Any]:
        """
        Detect memory usage regressions.
        """
        
        if not baseline:
            return {"status": "no_baseline"}
        
        baseline_memory = baseline.peak_memory_mb
        current_memory = current.peak_memory_mb
        
        regression_pct = ((current_memory - baseline_memory) / baseline_memory) * 100 if baseline_memory > 0 else 0
        
        status = "regression" if regression_pct > threshold_percent else "stable"
        
        return {
            "status": status,
            "regression_percent": regression_pct,
            "baseline_memory_mb": baseline_memory,
            "current_memory_mb": current_memory,
            "difference_mb": current_memory - baseline_memory,
            "threshold_percent": threshold_percent,
        }


class SubLinearAlgorithmAnalyzer:
    """Analyzes opportunities for sub-linear algorithms."""
    
    @staticmethod
    def analyze_opportunities(
        component: str,
        current_complexity: str,
    ) -> List[Dict[str, Any]]:
        """
        Suggest sub-linear algorithm opportunities.
        """
        
        opportunities = []
        
        if component == "storage_read" and current_complexity == "O(log n)":
            # Bloom filter opportunity
            opportunities.append({
                "algorithm": "Bloom Filter",
                "current_complexity": "O(log n)",
                "target_complexity": "O(k)",
                "space_complexity": "O(n) bits",
                "estimated_speedup": 10,
                "trade_off": "1-2% false positive rate",
                "applicability": "Fast 'not found' detection",
            })
        
        if component == "compression" and current_complexity == "O(n)":
            # Streaming sketches opportunity
            opportunities.append({
                "algorithm": "Count-Min Sketch",
                "current_complexity": "O(n) space full scan",
                "target_complexity": "O(log 1/δ) space",
                "space_complexity": "O(log 1/δ)",
                "estimated_speedup": 1000,  # Space reduction
                "trade_off": "Approximate frequency estimates",
                "applicability": "Frequency tracking in streams",
            })
        
        if component == "agent_collective" and current_complexity == "O(n)":
            # HyperLogLog for cardinality
            opportunities.append({
                "algorithm": "HyperLogLog",
                "current_complexity": "O(n) full enumeration",
                "target_complexity": "O(log n) space",
                "space_complexity": "O(log n) registers",
                "estimated_speedup": 1000,  # Space reduction
                "trade_off": "~2% error",
                "applicability": "Counting unique agents/tasks",
            })
        
        if component == "storage_read" and "O(n²)" in current_complexity:
            # LSH for similarity
            opportunities.append({
                "algorithm": "Locality-Sensitive Hashing (LSH)",
                "current_complexity": "O(n²) pairwise comparison",
                "target_complexity": "O(n) expected",
                "space_complexity": "O(n)",
                "estimated_speedup": 100,
                "trade_off": "Approximate results",
                "applicability": "Finding similar stored items",
            })
        
        return opportunities


# Example usage functions

def profile_benchmark_function(
    func: Callable,
    iterations: int = 10,
    warmup: int = 3,
    profiler: Optional[BenchmarkProfiler] = None,
) -> ProfileResult:
    """
    Profile a benchmark function.
    """
    
    if profiler is None:
        profiler = BenchmarkProfiler(enabled=True)
    
    # Warmup
    for _ in range(warmup):
        func()
    
    # Actual measurements
    profiler.start()
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed_ms = (time.perf_counter() - start) * 1000
        profiler.record_measurement(elapsed_ms)
    
    result = profiler.stop()
    return result


def compare_optimization_impact(
    baseline_result: ProfileResult,
    optimized_result: ProfileResult,
) -> Dict[str, Any]:
    """
    Compare baseline vs optimized results.
    """
    
    detector = RegressionDetector()
    
    latency_comparison = detector.compare_to_baseline(
        optimized_result,
        baseline_result,
        threshold_percent=5.0,
    )
    
    memory_comparison = detector.detect_memory_regression(
        optimized_result,
        baseline_result,
        threshold_percent=10.0,
    )
    
    # Calculate actual speedup
    speedup = baseline_result.mean_time_ms / optimized_result.mean_time_ms
    
    return {
        "speedup": speedup,
        "latency_comparison": latency_comparison,
        "memory_comparison": memory_comparison,
        "improvement_percent": ((baseline_result.mean_time_ms - optimized_result.mean_time_ms) / baseline_result.mean_time_ms) * 100,
    }


if __name__ == "__main__":
    # Example: Create sample profile results
    baseline = ProfileResult(
        benchmark_name="test",
        mean_time_ms=100,
        median_time_ms=95,
        stddev_ms=10,
        min_time_ms=85,
        max_time_ms=120,
        p50_ms=95,
        p90_ms=112,
        p95_ms=118,
        p99_ms=120,
        operations_per_second=10,
        peak_memory_mb=512,
        avg_memory_mb=256,
    )
    
    optimized = ProfileResult(
        benchmark_name="test",
        mean_time_ms=80,
        median_time_ms=78,
        stddev_ms=8,
        min_time_ms=70,
        max_time_ms=95,
        p50_ms=78,
        p90_ms=88,
        p95_ms=92,
        p99_ms=95,
        operations_per_second=12.5,
        peak_memory_mb=384,
        avg_memory_mb=200,
    )
    
    # Compare
    detector = RegressionDetector()
    result = detector.compare_to_baseline(optimized, baseline, threshold_percent=10)
    print("Comparison result:", json.dumps(result, indent=2))
