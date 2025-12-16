#!/usr/bin/env python3
"""
Run Neurectomy Benchmarks
=========================

Usage:
    python scripts/run_benchmarks.py                    # Run all benchmarks
    python scripts/run_benchmarks.py --category inference  # Run only inference
    python scripts/run_benchmarks.py --output results/baseline.json
    python scripts/run_benchmarks.py --compare results/baseline.json results/current.json
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks import BenchmarkRunner, BenchmarkConfig
from benchmarks.runner import compare_reports


def main():
    parser = argparse.ArgumentParser(description="Neurectomy Benchmark Suite")
    parser.add_argument(
        "--category",
        choices=["inference", "compression", "storage", "agent", "e2e", "all"],
        default="all",
        help="Benchmark category to run",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for results",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE", "CURRENT"),
        help="Compare two benchmark results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    
    args = parser.parse_args()
    
    # Handle comparison mode
    if args.compare:
        baseline, current = args.compare
        print("Comparing benchmark results...")
        print(f"  Baseline: {baseline}")
        print(f"  Current:  {current}")
        print()
        
        comparison = compare_reports(baseline, current)
        
        print("=" * 60)
        print("  BENCHMARK COMPARISON")
        print("=" * 60)
        print()
        
        for c in comparison["comparisons"]:
            status = "🔴" if c["regression"] else ("🟢" if c["improvement"] else "⚪")
            print(f"{status} {c['name']}")
            print(f"   Baseline: {c['baseline_ms']:.2f}ms")
            print(f"   Current:  {c['current_ms']:.2f}ms")
            print(f"   Change:   {c['change_pct']:+.1f}%")
            print()
        
        if comparison["regressions"]:
            print(f"⚠️  {len(comparison['regressions'])} regressions detected!")
            return 1
        
        return 0
    
    # Configure runner
    config = BenchmarkConfig(
        warmup_iterations=args.warmup,
        benchmark_iterations=args.iterations,
        verbose=args.verbose,
        save_results=args.output is not None,
        results_dir=str(Path(args.output).parent) if args.output else "results",
    )
    
    runner = BenchmarkRunner(config)
    
    # Add benchmarks
    if args.category == "all":
        runner.add_all_benchmarks()
    else:
        runner.add_category(args.category)
    
    # Run
    report = runner.run()
    
    # Save with custom path if specified
    if args.output:
        report.save(args.output)
        print(f"\nResults saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
