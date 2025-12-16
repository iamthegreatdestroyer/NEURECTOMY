#!/usr/bin/env python3
"""Phase 9A Verification - Benchmark Suite"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def verify_benchmark_base():
    """Verify benchmark base classes."""
    try:
        from benchmarks.base import Benchmark, BenchmarkConfig, BenchmarkResult
        
        config = BenchmarkConfig(warmup_iterations=1, benchmark_iterations=2)
        assert config.warmup_iterations == 1
        assert config.benchmark_iterations == 2
        assert config.timeout_seconds == 300.0
        
        print("✓ Benchmark base classes verified")
        return True
    except Exception as e:
        print(f"❌ Benchmark base failed: {e}")
        return False


def verify_benchmark_categories():
    """Verify all benchmark categories are loadable."""
    try:
        from benchmarks.inference_bench import get_inference_benchmarks
        from benchmarks.compression_bench import get_compression_benchmarks
        from benchmarks.storage_bench import get_storage_benchmarks
        from benchmarks.agent_bench import get_agent_benchmarks
        from benchmarks.e2e_bench import get_e2e_benchmarks
        
        counts = {
            "inference": len(get_inference_benchmarks()),
            "compression": len(get_compression_benchmarks()),
            "storage": len(get_storage_benchmarks()),
            "agent": len(get_agent_benchmarks()),
            "e2e": len(get_e2e_benchmarks()),
        }
        
        total = sum(counts.values())
        
        print(f"✓ Benchmark categories verified ({total} total benchmarks)")
        for cat, count in counts.items():
            print(f"   - {cat}: {count} benchmarks")
        
        return True
    except Exception as e:
        print(f"❌ Benchmark categories failed: {e}")
        return False


def verify_runner():
    """Verify benchmark runner functionality."""
    try:
        from benchmarks import BenchmarkRunner, BenchmarkConfig
        
        config = BenchmarkConfig(
            warmup_iterations=1,
            benchmark_iterations=1,
            save_results=False,
        )
        
        runner = BenchmarkRunner(config)
        runner.add_category("inference")
        
        # Verify benchmarks were added
        assert len(runner._benchmarks) > 0, "No benchmarks added"
        
        print(f"✓ Benchmark runner verified ({len(runner._benchmarks)} benchmarks queued)")
        return True
    except Exception as e:
        print(f"❌ Benchmark runner failed: {e}")
        return False


def verify_report():
    """Verify benchmark report functionality."""
    try:
        from benchmarks.runner import BenchmarkReport
        from benchmarks.base import BenchmarkConfig, BenchmarkResult
        from datetime import datetime
        
        # Create a mock result
        result = BenchmarkResult(
            name="test_benchmark",
            iterations=10,
            total_time_ms=100.0,
            min_time_ms=8.0,
            max_time_ms=12.0,
            mean_time_ms=10.0,
            median_time_ms=10.0,
            stddev_ms=1.0,
            p50_ms=10.0,
            p90_ms=11.0,
            p95_ms=11.5,
            p99_ms=12.0,
            operations_per_second=100.0,
            metrics={"custom_metric": 42},
        )
        
        config = BenchmarkConfig()
        report = BenchmarkReport(
            results=[result],
            config=config,
        )
        
        # Verify to_dict
        report_dict = report.to_dict()
        assert "results" in report_dict
        assert "summary" in report_dict
        assert len(report_dict["results"]) == 1
        
        print("✓ Benchmark report verified")
        return True
    except Exception as e:
        print(f"❌ Benchmark report failed: {e}")
        return False


def verify_comparison():
    """Verify comparison functionality."""
    try:
        from benchmarks.runner import compare_reports, BenchmarkReport
        from benchmarks.base import BenchmarkConfig, BenchmarkResult
        import tempfile
        import os
        
        # Create two mock reports
        result1 = BenchmarkResult(
            name="test_benchmark",
            iterations=10,
            total_time_ms=100.0,
            min_time_ms=8.0,
            max_time_ms=12.0,
            mean_time_ms=10.0,
            median_time_ms=10.0,
            stddev_ms=1.0,
            p50_ms=10.0,
            p90_ms=11.0,
            p95_ms=11.5,
            p99_ms=12.0,
            operations_per_second=100.0,
        )
        
        result2 = BenchmarkResult(
            name="test_benchmark",
            iterations=10,
            total_time_ms=110.0,
            min_time_ms=9.0,
            max_time_ms=13.0,
            mean_time_ms=11.0,
            median_time_ms=11.0,
            stddev_ms=1.0,
            p50_ms=11.0,
            p90_ms=12.0,
            p95_ms=12.5,
            p99_ms=13.0,
            operations_per_second=90.9,
        )
        
        config = BenchmarkConfig()
        report1 = BenchmarkReport(results=[result1], config=config)
        report2 = BenchmarkReport(results=[result2], config=config)
        
        # Save to temp files
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = os.path.join(tmpdir, "baseline.json")
            path2 = os.path.join(tmpdir, "current.json")
            
            report1.save(path1)
            report2.save(path2)
            
            # Compare
            comparison = compare_reports(path1, path2)
            
            assert "comparisons" in comparison
            assert len(comparison["comparisons"]) == 1
            
        print("✓ Comparison functionality verified")
        return True
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        return False


def verify_imports():
    """Verify main package imports work."""
    try:
        from benchmarks import BenchmarkRunner, BenchmarkConfig, BenchmarkResult, BenchmarkReport
        
        print("✓ Package imports verified")
        return True
    except Exception as e:
        print(f"❌ Package imports failed: {e}")
        return False


def main():
    print("=" * 60)
    print("  PHASE 9A: Benchmark Suite - Verification")
    print("=" * 60)
    print()
    
    results = [
        verify_imports(),
        verify_benchmark_base(),
        verify_benchmark_categories(),
        verify_runner(),
        verify_report(),
        verify_comparison(),
    ]
    
    print()
    passed = sum(results)
    total = len(results)
    
    if all(results):
        print("=" * 60)
        print(f"  ✅ PHASE 9A VERIFICATION COMPLETE ({passed}/{total} checks passed)")
        print("=" * 60)
        print()
        print("  Benchmark Suite Files Created:")
        print("    - benchmarks/__init__.py")
        print("    - benchmarks/base.py")
        print("    - benchmarks/results.py")
        print("    - benchmarks/runner.py")
        print("    - benchmarks/inference_bench.py")
        print("    - benchmarks/compression_bench.py")
        print("    - benchmarks/storage_bench.py")
        print("    - benchmarks/agent_bench.py")
        print("    - benchmarks/e2e_bench.py")
        print()
        print("  Run benchmarks with:")
        print("    python scripts/run_benchmarks.py")
        print("    python scripts/run_benchmarks.py --category inference")
        print("    python scripts/run_benchmarks.py --output results/baseline.json")
        print()
        return 0
    else:
        print("=" * 60)
        print(f"  ❌ PHASE 9A VERIFICATION FAILED ({passed}/{total} checks passed)")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
