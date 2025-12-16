"""
Benchmark Runner
================

Unified benchmark execution and reporting.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

from .base import Benchmark, BenchmarkConfig, BenchmarkResult
from .inference_bench import get_inference_benchmarks
from .compression_bench import get_compression_benchmarks
from .storage_bench import get_storage_benchmarks
from .agent_bench import get_agent_benchmarks
from .e2e_bench import get_e2e_benchmarks


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    
    results: List[BenchmarkResult]
    config: BenchmarkConfig
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    system_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "config": {
                "warmup_iterations": self.config.warmup_iterations,
                "benchmark_iterations": self.config.benchmark_iterations,
            },
            "system_info": self.system_info,
            "results": [r.to_dict() for r in self.results],
            "summary": self._generate_summary(),
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.results:
            return {}
        
        return {
            "total_benchmarks": len(self.results),
            "total_time_ms": sum(r.total_time_ms for r in self.results),
            "categories": self._categorize_results(),
        }
    
    def _categorize_results(self) -> Dict[str, Dict[str, float]]:
        """Categorize results by type."""
        categories = {
            "inference": [],
            "compression": [],
            "storage": [],
            "agent": [],
            "e2e": [],
        }
        
        for result in self.results:
            name = result.name.lower()
            if "token" in name or "inference" in name or "context" in name or "batch" in name:
                categories["inference"].append(result.mean_time_ms)
            elif "compress" in name or "decompress" in name:
                categories["compression"].append(result.mean_time_ms)
            elif "rsu" in name or "storage" in name or "search" in name or "manifold" in name:
                categories["storage"].append(result.mean_time_ms)
            elif "agent" in name or "team" in name or "routing" in name:
                categories["agent"].append(result.mean_time_ms)
            else:
                categories["e2e"].append(result.mean_time_ms)
        
        summary = {}
        for cat, times in categories.items():
            if times:
                summary[cat] = {
                    "count": len(times),
                    "avg_ms": sum(times) / len(times),
                    "min_ms": min(times),
                    "max_ms": max(times),
                }
        
        return summary
    
    def save(self, path: str) -> None:
        """Save report to JSON file."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'BenchmarkReport':
        """Load report from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        # Reconstruct results
        results = []
        for r in data.get("results", []):
            results.append(BenchmarkResult(
                name=r["name"],
                iterations=r["iterations"],
                total_time_ms=r["timing"]["total_ms"],
                min_time_ms=r["timing"]["min_ms"],
                max_time_ms=r["timing"]["max_ms"],
                mean_time_ms=r["timing"]["mean_ms"],
                median_time_ms=r["timing"]["median_ms"],
                stddev_ms=r["timing"]["stddev_ms"],
                p50_ms=r["percentiles"]["p50_ms"],
                p90_ms=r["percentiles"]["p90_ms"],
                p95_ms=r["percentiles"]["p95_ms"],
                p99_ms=r["percentiles"]["p99_ms"],
                operations_per_second=r["throughput"]["ops_per_second"],
                metrics=r.get("metrics", {}),
            ))
        
        config = BenchmarkConfig()
        return cls(results=results, config=config, system_info=data.get("system_info", {}))


class BenchmarkRunner:
    """
    Unified benchmark runner.
    
    Executes all benchmarks and generates reports.
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self._benchmarks: List[Benchmark] = []
    
    def add_benchmark(self, benchmark: Benchmark) -> None:
        """Add a benchmark to run."""
        self._benchmarks.append(benchmark)
    
    def add_benchmarks(self, benchmarks: List[Benchmark]) -> None:
        """Add multiple benchmarks."""
        self._benchmarks.extend(benchmarks)
    
    def add_all_benchmarks(self) -> None:
        """Add all available benchmarks."""
        self._benchmarks.extend(get_inference_benchmarks(self.config))
        self._benchmarks.extend(get_compression_benchmarks(self.config))
        self._benchmarks.extend(get_storage_benchmarks(self.config))
        self._benchmarks.extend(get_agent_benchmarks(self.config))
        self._benchmarks.extend(get_e2e_benchmarks(self.config))
    
    def add_category(self, category: str) -> None:
        """Add benchmarks from a specific category."""
        categories = {
            "inference": get_inference_benchmarks,
            "compression": get_compression_benchmarks,
            "storage": get_storage_benchmarks,
            "agent": get_agent_benchmarks,
            "e2e": get_e2e_benchmarks,
        }
        
        if category in categories:
            self._benchmarks.extend(categories[category](self.config))
    
    def run(self) -> BenchmarkReport:
        """Run all benchmarks and generate report."""
        results = []
        
        print("=" * 60)
        print("  NEURECTOMY BENCHMARK SUITE")
        print("=" * 60)
        print()
        
        for i, benchmark in enumerate(self._benchmarks, 1):
            print(f"[{i}/{len(self._benchmarks)}] Running: {benchmark.name}")
            
            try:
                result = benchmark.execute()
                results.append(result)
                print(f"  ✓ Completed: {result.mean_time_ms:.2f}ms avg")
            except Exception as e:
                print(f"  ✗ Failed: {e}")
            
            print()
        
        # Get system info
        system_info = self._get_system_info()
        
        report = BenchmarkReport(
            results=results,
            config=self.config,
            system_info=system_info,
        )
        
        # Print summary
        self._print_summary(report)
        
        # Save if configured
        if self.config.save_results:
            output_path = Path(self.config.results_dir) / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report.save(str(output_path))
            print(f"\nResults saved to: {output_path}")
        
        return report
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Collect system information."""
        import platform
        import sys
        
        return {
            "platform": platform.platform(),
            "python_version": sys.version,
            "processor": platform.processor(),
        }
    
    def _print_summary(self, report: BenchmarkReport) -> None:
        """Print benchmark summary."""
        print("=" * 60)
        print("  BENCHMARK SUMMARY")
        print("=" * 60)
        print()
        
        summary = report._generate_summary()
        
        print(f"Total benchmarks: {summary.get('total_benchmarks', 0)}")
        print(f"Total time: {summary.get('total_time_ms', 0):.2f}ms")
        print()
        
        for category, stats in summary.get("categories", {}).items():
            print(f"{category.upper()}:")
            print(f"  Count: {stats['count']}")
            print(f"  Avg: {stats['avg_ms']:.2f}ms")
            print(f"  Min: {stats['min_ms']:.2f}ms")
            print(f"  Max: {stats['max_ms']:.2f}ms")
            print()


def compare_reports(baseline_path: str, current_path: str) -> Dict[str, Any]:
    """Compare two benchmark reports."""
    baseline = BenchmarkReport.load(baseline_path)
    current = BenchmarkReport.load(current_path)
    
    baseline_by_name = {r.name: r for r in baseline.results}
    current_by_name = {r.name: r for r in current.results}
    
    comparisons = []
    
    for name in current_by_name:
        if name in baseline_by_name:
            base = baseline_by_name[name]
            curr = current_by_name[name]
            
            change_pct = ((curr.mean_time_ms - base.mean_time_ms) / base.mean_time_ms) * 100
            
            comparisons.append({
                "name": name,
                "baseline_ms": base.mean_time_ms,
                "current_ms": curr.mean_time_ms,
                "change_pct": change_pct,
                "regression": change_pct > 10,  # >10% slower is regression
                "improvement": change_pct < -10,  # >10% faster is improvement
            })
    
    return {
        "comparisons": comparisons,
        "regressions": [c for c in comparisons if c["regression"]],
        "improvements": [c for c in comparisons if c["improvement"]],
    }
