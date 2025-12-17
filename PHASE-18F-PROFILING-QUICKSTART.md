# Phase 18F: Performance Profiling Quick Start

## Installation

### Prerequisites

```bash
# Install profiling tools
pip install py-spy memory-profiler line-profiler scalene psutil

# Install benchmarking tools
pip install pytest-benchmark google-benchmark hypothesis

# Install visualization tools
pip install flamegraph py-spy-viewer
```

### Verify Installation

```bash
py-spy --help
memory_profiler --help
```

---

## Quick Start: Run Benchmark Suite

### 1. Run Microbenchmarks (Fast, <1 minute)

```bash
# Run all microbenchmarks
python -m benchmarks.runner microbenchmarks

# Run specific component
python -m benchmarks.runner microbenchmarks --component=inference

# Run with output
python -m benchmarks.runner microbenchmarks --output=json --save-results
```

Expected output:

```
Benchmark: inference_ttft_micro
  Mean: 95.2 ms
  P99:  118.5 ms
  Status: ✓ PASS (target: 100ms)

Benchmark: compression_ratio_micro
  Mean: 3.2x
  Status: ✓ PASS (target: 3.0x)

...
```

### 2. Run Macrobenchmarks (Realistic, 5-20 minutes)

```bash
# Full macro suite
python -m benchmarks.runner macrobenchmarks

# Specific benchmark
python -m benchmarks.runner --suite=macro --name=inference_full

# With profiling enabled
python -m benchmarks.runner macrobenchmarks --profiling=enabled
```

### 3. Deep Profiling (Optimization Sprint, 15-45 minutes)

```bash
# Profiling suite with full traces
python -m benchmarks.runner profiling

# With specific profiler
python -m benchmarks.runner profiling --profiler=py-spy --sample-rate=1000

# Generate flame graphs
python -m benchmarks.runner profiling --output=flamegraph
```

---

## Component-Specific Profiling

### A. Profile LLM Inference (Ryot)

```bash
# 1. Quick TTFT measurement
python -c "
from benchmarks.inference_bench import FirstTokenLatencyBenchmark
b = FirstTokenLatencyBenchmark()
result = b.run()
print(f'TTFT: {result.mean_time_ms:.1f}ms')
"

# 2. Profile with py-spy
py-spy record -o inference_profile.svg -- \
  python -m benchmarks.runner --component=inference --duration=60

# 3. Detailed profiling with line_profiler
kernprof -l -v neurectomy/ryot/inference.py

# 4. Measure tokens/sec throughput
python -c "
from benchmarks.inference_bench import TokenGenerationBenchmark
for max_tokens in [100, 500, 1000]:
    b = TokenGenerationBenchmark(max_tokens=max_tokens)
    result = b.run()
    print(f'{max_tokens} tokens: {result.metrics[\"tokens_per_second\"]:.1f} tok/sec')
"
```

### B. Profile Compression (ΣLANG)

```bash
# 1. Quick compression ratio test
python -c "
from benchmarks.compression_bench import CompressionRatioBenchmark
for size in [1000, 10000, 100000, 1000000]:
    b = CompressionRatioBenchmark(text_size=size)
    result = b.run()
    print(f'{size:7d} bytes: {result.metrics[\"compression_ratio\"]:.2f}x')
"

# 2. Measure throughput at different sizes
python -m benchmarks.runner --component=compression \
  --test-params=text_size:[1K,10K,100K,1M] \
  --output=json

# 3. Profile hot path with line_profiler
kernprof -l -v neurectomy/sigma_lang/compressor.py

# 4. Memory profiling
python -m memory_profiler \
  neurectomy/sigma_lang/compressor.py --profile-function=compress

# 5. Generate performance report
python -c "
import json
from benchmarks.compression_bench import CompressionThroughputBenchmark
results = []
for size in [1000, 10000, 100000, 1000000]:
    b = CompressionThroughputBenchmark(text_size=size)
    r = b.run()
    results.append(r.metrics)
print(json.dumps(results, indent=2))
" > compression_results.json
```

### C. Profile Storage (ΣVAULT)

```bash
# 1. Quick latency test
python -c "
from benchmarks.storage_bench import RSUReadBenchmark, RSUWriteBenchmark
write_b = RSUWriteBenchmark(data_size=1000)
read_b = RSUReadBenchmark(data_size=1000)
write_result = write_b.run()
read_result = read_b.run()
print(f'Write: {write_result.mean_time_ms:.1f}ms')
print(f'Read:  {read_result.mean_time_ms:.1f}ms')
"

# 2. Throughput test (1M operations)
python -m benchmarks.runner --component=storage \
  --benchmark=storage_endurance \
  --test-params=num_operations:[1000000]

# 3. Memory profiling
python -m memory_profiler --profile-function=store_rsu \
  neurectomy/sigma_vault/storage.py

# 4. Cache hit rate analysis
python -c "
from neurectomy.sigma_vault.storage import CachedStorage
storage = CachedStorage()
for i in range(1000):
    storage.read(f'key_{i % 100}')  # 100 unique keys, 1000 accesses
cache_stats = storage.get_cache_stats()
print(f'Hit rate: {cache_stats[\"hit_rate\"]:.1%}')
"
```

### D. Profile Agent Collective

```bash
# 1. Quick task latency measurement
python -c "
from benchmarks.agent_bench import AgentTaskLatencyBenchmark
b = AgentTaskLatencyBenchmark(num_agents=4)
result = b.run()
print(f'Task latency: {result.mean_time_ms:.1f}ms (p99: {result.metrics[\"p99_ms\"]:.1f}ms)')
"

# 2. Queue depth monitoring
python -m benchmarks.runner --component=agent \
  --profile-mode=combined \
  --collect-metrics=queue_depth

# 3. Inter-agent communication profiling
python -c "
import os
os.environ['OTEL_ENABLED'] = 'true'
from benchmarks.agent_bench import AgentCommunicationBenchmark
b = AgentCommunicationBenchmark(num_agents=8)
b.run()
"

# 4. Resource utilization per agent
python -m benchmarks.runner --component=agent \
  --profile-mode=memory \
  --output=agent_resources.json
```

---

## Analyzing Results

### 1. View Profiling Results

```bash
# HTML flame graph (from py-spy)
open inference_profile.html

# JSON results
cat results/benchmark_results.json | jq '.[] | select(.category=="inference")'

# Compare multiple runs
python -c "
from benchmarks.phase_18f_profiling_utils import RegressionDetector
from benchmarks.phase_18f_profiling_configs import *

# Load results
import json
baseline = json.load(open('baseline.json'))
current = json.load(open('current.json'))

# Detect regressions
for bench_name in baseline:
    result = RegressionDetector.compare_to_baseline(
        current[bench_name],
        baseline[bench_name],
        threshold_percent=10
    )
    print(f'{bench_name}: {result[\"status\"]} ({result[\"regression_percent\"]:.1f}%)')
"
```

### 2. Identify Bottlenecks

```python
from benchmarks.phase_18f_profiling_utils import BottleneckAnalyzer
import json

# Load profiling data
profile_data = json.load(open('inference_profile.json'))

# Rank bottlenecks
ranked = BottleneckAnalyzer.analyze_function_profile(profile_data)

# Print top 10
for i, func in enumerate(ranked[:10], 1):
    print(f"{i}. {func['name']}")
    print(f"   Self: {func['self_time_pct']:.1f}%")
    print(f"   Per call: {func['time_per_call_us']:.1f}µs")
    print()

# Estimate speedup potential
speedup_estimates = BottleneckAnalyzer.estimate_speedup_potential(ranked)
for est in speedup_estimates:
    print(f"{est['function']}: {est['speedup_if_10x_optimized']:.1f}× potential speedup")
```

### 3. Detect Regressions

```bash
# Automatic regression detection after benchmark run
python -m benchmarks.regression_checker \
  --baseline=baseline_v1.json \
  --current=results.json \
  --threshold=10 \
  --alert-on-critical

# Output
# Benchmark,Regression%,Status,Impact
# inference_ttft,-2.3%,✓ Improvement,Positive
# compression_throughput,8.1%,✓ Stable,OK
# storage_latency,15.2%,⚠️ WARNING,High (exceeds 10% threshold)
# agent_queue_depth,35.0%,🔴 CRITICAL,Critical (exceeds 20% threshold)
```

---

## Optimization Workflow

### Phase 1: Establish Baseline (30 min)

```bash
# 1. Run full benchmark suite
python -m benchmarks.runner all --output=json --save-results

# 2. Save as baseline
cp results/benchmark_results.json baseline_before_optimization.json

# 3. Generate profiling report
python -c "
from benchmarks.phase_18f_profiling_utils import BottleneckAnalyzer
import json

results = json.load(open('baseline_before_optimization.json'))
for component in ['inference', 'compression', 'storage', 'agent']:
    print(f'\n=== {component.upper()} ===')
    analyzer = BottleneckAnalyzer()
    ranked = analyzer.analyze_function_profile(results[component])
    for func in ranked[:5]:
        print(f'{func[\"name\"]}: {func[\"self_time_pct\"]:.1f}% of time')
"
```

### Phase 2: Optimize (varies)

```bash
# For each optimization:

# 1. Implement change
# (edit code...)

# 2. Re-run benchmarks
python -m benchmarks.runner component_name --output=json --save-results

# 3. Compare to baseline
python -m benchmarks.regression_checker \
  --baseline=baseline_before_optimization.json \
  --current=results/benchmark_results.json \
  --threshold=5 \
  --show-speedup

# 4. If improvement <5%, revert. If >5%, commit.
```

### Phase 3: Validate (15 min)

```bash
# 1. Run full suite 3 times
for i in {1..3}; do
  python -m benchmarks.runner all --output=json \
    --save-results --name="validation_run_$i"
done

# 2. Check consistency
python -c "
import json
results = []
for i in range(1, 4):
    with open(f'results/validation_run_{i}.json') as f:
        results.append(json.load(f))

# Calculate consistency
import statistics
for benchmark in results[0]:
    values = [r[benchmark]['mean_time_ms'] for r in results]
    stddev = statistics.stdev(values)
    mean = statistics.mean(values)
    cv = (stddev / mean) * 100
    print(f'{benchmark}: CV={cv:.1f}% (acceptable if <5%)')
"

# 3. Measure improvement
python -m benchmarks.regression_checker \
  --baseline=baseline_before_optimization.json \
  --current=results/validation_run_1.json \
  --show-summary
```

---

## Performance SLOs & Alerting

### Set SLOs

```yaml
# performance_slos.yaml
slos:
  inference_ttft:
    p50: 100ms
    p99: 120ms
    alert_threshold: 150ms

  compression_throughput:
    target: 100 MB/s
    alert_threshold: 50 MB/s

  storage_latency_p99:
    target: 10ms
    alert_threshold: 20ms

  agent_queue_depth:
    target: 20
    alert_threshold: 100
```

### Monitor SLOs

```bash
# Continuous monitoring (runs benchmarks every hour)
python -m benchmarks.slo_monitor \
  --config=performance_slos.yaml \
  --interval=3600 \
  --alert-webhook=https://alerting-service/webhook
```

---

## Troubleshooting

### py-spy Issues

```bash
# If py-spy fails to record
# Try with sudo
sudo py-spy record -o profile.svg -- python script.py

# Or use cProfile instead
python -m cProfile -o profile.stats script.py
python -m pstats profile.stats

# Sort by cumulative time
(pstats) sort cumulative
(pstats) stats 20
```

### Memory Profiler Issues

```bash
# If memory_profiler is slow
# Profile specific function only
@profile
def my_function():
    ...

python -m memory_profiler script.py

# Or use scalene (faster)
scalene --profile-interval=0.01 script.py
```

### High-Overhead Profiling

```bash
# If profiling overhead >10%, use sampling
py-spy record --sample-rate=50 ...  # Sample at 50Hz instead of 100Hz

# Or skip profiling for specific benchmarks
python -m benchmarks.runner --profiling=disabled --fast
```

---

## Further Reading

- **py-spy:** https://github.com/benfred/py-spy
- **memory_profiler:** https://github.com/pythonprofilers/memory_profiler
- **line_profiler:** https://github.com/pyflame/line_profiler
- **Benchmarking Best Practices:** https://easyperf.net/blog/
- **Profiling Optimization:** https://github.com/brendangregg/FlameGraph
