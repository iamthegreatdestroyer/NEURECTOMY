#!/usr/bin/env python3
"""
Phase 18F-3 Profiling Initialization Script

Sets up the profiling environment and runs Day 1 baseline collection.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime


def setup_profiling_environment():
    """Initialize profiling environment and directories"""
    
    print("="*80)
    print("PHASE 18F-3: PROFILING ENVIRONMENT SETUP")
    print("="*80)
    print()
    
    # Create result directories
    dirs = [
        "results/phase_18f/raw_profiles",
        "results/phase_18f/flame_graphs",
        "results/phase_18f/metrics_json",
        "results/phase_18f/daily_reports",
    ]
    
    print("Creating result directories...")
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {dir_path}")
    
    print()


def check_profiling_tools():
    """Verify all profiling tools are installed"""
    
    print("Checking profiling tools...")
    
    tools = [
        ("py-spy", "py_spy"),
        ("memory_profiler", "memory_profiler"),
        ("line_profiler", "line_profiler"),
    ]
    
    missing = []
    for package_name, import_name in tools:
        try:
            __import__(import_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} - MISSING")
            missing.append(package_name)
    
    if missing:
        print()
        print("Installing missing tools...")
        subprocess.run(
            ["pip", "install"] + missing,
            check=False
        )
        print("  Installation complete")
    
    print()


def create_session_metadata():
    """Create metadata for this profiling session"""
    
    metadata = {
        "session_id": f"phase_18f_3_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "start_time": datetime.now().isoformat(),
        "phase": "18F-3",
        "component": "all",
        "duration_days": 5,
        "status": "initialized",
        "schedule": {
            "day_1": "Ryot LLM baselines",
            "day_2": "ΣLANG, ΣVAULT, Agents baselines",
            "day_3": "Macrobenchmarks",
            "day_4": "Detailed profiling",
            "day_5": "Analysis and roadmap",
        },
    }
    
    metadata_file = Path("results/phase_18f/session_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def display_execution_guide():
    """Display how to execute profiling"""
    
    print("="*80)
    print("EXECUTION GUIDE")
    print("="*80)
    print()
    print("To begin Day 1 profiling, run:")
    print()
    print("  python benchmarks/runner_18f.py 1")
    print()
    print("Daily Execution Schedule:")
    print("  Day 1 (Today):    python benchmarks/runner_18f.py 1    # Ryot")
    print("  Day 2 (Dec 18):   python benchmarks/runner_18f.py 2    # Multi-component")
    print("  Day 3 (Dec 19):   python benchmarks/runner_18f.py 3    # Macrobenchmarks")
    print("  Day 4 (Dec 20):   python benchmarks/runner_18f.py 4    # Profiling")
    print("  Day 5 (Dec 21):   python benchmarks/runner_18f.py 5    # Analysis")
    print()
    print("Expected Time:")
    print("  Day 1: 30-45 minutes")
    print("  Day 2: 1.5-2 hours")
    print("  Day 3: 1-2 hours per benchmark (endurance test longest)")
    print("  Day 4: 2-3 hours")
    print("  Day 5: Analysis (variable)")
    print()


def create_quick_reference():
    """Create quick reference for operators"""
    
    reference = """# Phase 18F-3 Quick Reference

## Status Check
cat results/phase_18f/daily_reports/day_*_report_*.json | jq '.total_benchmarks'

## View Metrics
python -c "
import json, glob
reports = sorted(glob.glob('results/phase_18f/daily_reports/*.json'))
if reports:
    with open(reports[-1]) as f:
        data = json.load(f)
        for r in data['results']:
            print(f\"{r['benchmark_name']:40} | Success: {r['success']}\")
"

## Generate Bottleneck Report
python -c "
import json, glob
from benchmarks.runner_18f import ProfileAnalyzer
analyzer = ProfileAnalyzer()
bottlenecks = analyzer.identify_bottlenecks(1)
print(json.dumps(bottlenecks, indent=2))
"

## View Results Directory
ls -lah results/phase_18f/
tree results/phase_18f/

## Clean Results (Start Over)
rm -rf results/phase_18f/*
mkdir -p results/phase_18f/{raw_profiles,flame_graphs,metrics_json,daily_reports}
"""
    
    with open("PHASE-18F-3-QUICK-REFERENCE.md", 'w') as f:
        f.write(reference)
    
    print("✓ Created PHASE-18F-3-QUICK-REFERENCE.md")


def main():
    """Main initialization"""
    
    os.chdir(Path(__file__).parent.parent)  # Go to project root
    
    # Setup
    setup_profiling_environment()
    check_profiling_tools()
    
    # Create session metadata
    metadata = create_session_metadata()
    print("Session Metadata:")
    print(f"  Session ID: {metadata['session_id']}")
    print(f"  Start Time: {metadata['start_time']}")
    print()
    
    # Create quick reference
    create_quick_reference()
    print()
    
    # Display guide
    display_execution_guide()
    
    print("="*80)
    print("✓ PROFILING ENVIRONMENT READY")
    print("="*80)
    print()
    print("Next Step: Run Day 1 benchmarks")
    print("  python benchmarks/runner_18f.py 1")
    print()


if __name__ == "__main__":
    main()
