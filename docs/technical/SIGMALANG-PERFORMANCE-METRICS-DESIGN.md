# ΣLANG Performance Metrics Design

## Comprehensive Metrics Strategy for Compression Service Optimization

**Author**: @VELOCITY Performance Optimization Agent  
**Date**: December 16, 2025  
**Purpose**: Design metrics that reveal sub-linear opportunities and performance bottlenecks  
**Target**: Sub-linear algorithm optimization, 10-50x compression ratio detection, real-time performance analysis

---

## Executive Summary

This document provides a complete metrics strategy for ΣLANG compression service, designed from a **performance optimization perspective**. Rather than just measuring what happens, these metrics are designed to:

1. **Identify bottlenecks** through latency distribution and resource utilization patterns
2. **Detect sub-optimal scenarios** where compression ratio falls below target ranges
3. **Guide algorithm selection** by tracking effectiveness of different compression strategies
4. **Enable predictive optimization** through trend analysis and anomaly detection
5. **Quantify optimization impact** with before/after comparisons

**Key Principle**: Every metric should answer the question: _"Where should we optimize next?"_

---

## Part 1: Compression Operation Metrics

### 1.1 Compression Request Tracking

#### `sigmalang_compression_requests_total` (Counter)

```yaml
Type: Counter
Name: sigmalang_compression_requests_total
Labels:
  - status: (success|failure|cache_hit|cache_miss)
  - algorithm: (semantic_primitive|learned_pattern|delta_encode|hybrid)
  - data_type: (token_sequence|text|semantic_tree)
  - encoding_mode: (balanced|aggressive|streaming)
Help: Total compression requests with outcome categorization
```

**Why This Matters**:

- `cache_hit` / `cache_miss` ratio reveals RSU (Recyclable Semantic Unit) effectiveness
- Algorithm distribution shows real-world compression strategy usage
- Failure tracking identifies problematic data patterns
- Streaming mode vs batch comparison

**Optimization Insights**:

- High cache miss rate → Need to expand RSU vocabulary or improve pattern recognition
- Frequent algorithm failures → Algorithm selection heuristic needs adjustment
- Low cache hit rate on popular inputs → RSU eviction policy too aggressive

---

#### `sigmalang_compression_outcome_counts` (Counter - Granular)

```yaml
Type: Counter
Name: sigmalang_compression_outcome_counts
Labels:
  - outcome: (encode_success|decode_success|encode_failure|decode_failure|
      ratio_excellent|ratio_good|ratio_poor|ratio_uncompressible)
  - algorithm_family: (semantic|pattern|delta)
  - compression_quality: (high|balanced|fast)
Help: Granular compression outcome tracking for bottleneck identification
```

**Detection Strategy**:

- `ratio_poor` events: Input data resistant to compression → Algorithm switching opportunity
- `decode_failure`: Data corruption or encoding edge case → Validation improvement needed
- Algorithm family distribution: Shows which technique dominates workload

---

### 1.2 Compression Ratio Metrics

#### `sigmalang_compression_ratio` (Histogram)

```yaml
Type: Histogram
Name: sigmalang_compression_ratio
Labels:
  - algorithm: (semantic_primitive|learned_pattern|delta_encode|hybrid)
  - data_category: (code|prose|data_structure|mixed)
  - size_range: (tiny_<1kb|small_1-10kb|medium_10-100kb|large_>100kb)
Buckets: [1.5, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100]
Help: Compression ratio distribution (original_size / compressed_size)
```

**Bucket Strategy** (Sub-Linear Optimization Focused):

- **1.5-2x**: Minimal compression, data likely incompressible or redundant structure
- **2-5x**: Sub-optimal scenarios, investigate algorithm effectiveness
- **5-10x**: Good baseline (target range starts here)
- **10-30x**: Excellent compression, high semantic redundancy detected
- **30-50x**: Outstanding, likely code or highly repetitive text
- **50-100x**: Extreme compression, pattern reuse or data structure compression

**Optimization Opportunities by Bucket**:
| Bucket | Interpretation | Action |
|--------|---|---|
| 1.5x | Incompressible data | Use bypass/streaming mode |
| 2x | Underperforming | Algorithm selection issue |
| 5x | Baseline | Add algorithm-specific tuning |
| 10x | Target achieved | Monitor consistency |
| 20x+ | Exceptional | Analyze pattern for learning |
| 50x+ | Breakthrough | Promote to learning engine |

---

#### `sigmalang_compression_ratio_trend` (Gauge - Time Series)

```yaml
Type: Gauge
Name: sigmalang_compression_ratio_trend
Labels:
  - algorithm: (semantic_primitive|learned_pattern|delta_encode|hybrid)
  - time_window: (1h|1d|1w)
Help: Moving average compression ratio to detect degradation or improvement trends
```

**Optimization Use Case**:

- Downtrend in ratio → Check if new data patterns emerged or RSU cache degraded
- Uptrend → Learning engine improving, breakthrough detected
- Sudden drops → Possible algorithm regression or data distribution change

---

### 1.3 Compression Speed & Duration Metrics

#### `sigmalang_compression_duration_seconds` (Histogram)

```yaml
Type: Histogram
Name: sigmalang_compression_duration_seconds
Labels:
  - algorithm: (semantic_primitive|learned_pattern|delta_encode|hybrid)
  - compression_level: (1_fast|5_balanced|9_maximum)
  - input_size_range: (tiny|small|medium|large|huge)
Buckets: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
Help: Compression operation latency distribution
```

**Sub-Linear Optimization Focus**:

- **0.1-1ms**: Cache hits or minimal work (goal)
- **1-10ms**: Normal compression, tunable with optimizations
- **10-100ms**: Slow path, investigate memory access patterns
- **100ms+**: Pathological cases, profile for optimization opportunities

**Latency Breakdown Strategy**:
Use composite metrics to identify which phase is slow:

---

#### `sigmalang_compression_phase_duration_seconds` (Histogram)

```yaml
Type: Histogram
Name: sigmalang_compression_phase_duration_seconds
Labels:
  - phase: (parse|encode|optimize|serialize)
  - algorithm: (semantic_primitive|learned_pattern|delta_encode)
Buckets: [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
Help: Break down compression into phases for bottleneck identification
```

**Phases**:

1. **parse** (0.1-1ms): Semantic tree construction from tokens
2. **encode** (0.5-5ms): Tree → glyph conversion (main compression work)
3. **optimize** (0.01-0.5ms): Pattern matching and delta encoding
4. **serialize** (0.01-0.1ms): Glyph → binary output

**Optimization Logic**:

- If `parse` > 10% total: Slow tokenizer, optimize parsing
- If `encode` > 70% total: Compression algorithm is bottleneck
- If `optimize` > 20% total: Pattern matching too aggressive, tune threshold
- If `serialize` > 5% total: Buffer management issue

---

#### `sigmalang_compression_throughput_bytes_per_second` (Gauge)

```yaml
Type: Gauge
Name: sigmalang_compression_throughput_bytes_per_second
Labels:
  - algorithm: (semantic_primitive|learned_pattern|delta_encode|hybrid)
  - mode: (streaming|batch|adaptive)
Help: Real-time throughput in bytes/second during compression
```

**Calculation**:

```
throughput = input_bytes / compression_duration
```

**Performance Targets**:

- Semantic primitive: 1-10 MB/s
- Learned pattern: 5-50 MB/s
- Delta encoding: 50-500 MB/s
- Hybrid (adaptive): 10-100 MB/s

**Optimization Trigger**:

- Throughput < 1 MB/s: Investigate memory bottleneck or algorithm issue
- High variance (>2x): Unstable performance, enable profiling

---

### 1.4 Decompression Operation Metrics

#### `sigmalang_decompression_requests_total` (Counter)

```yaml
Type: Counter
Name: sigmalang_decompression_requests_total
Labels:
  - status: (success|failure|cache_hit)
  - decode_algorithm: (direct|pattern_lookup|delta_reversal)
Help: Total decompression operations and outcomes
```

**Optimization Insight**:

- High `cache_hit` rate: Streaming decode working well
- Failures indicate corruption or encoding errors

---

#### `sigmalang_decompression_duration_seconds` (Histogram)

```yaml
Type: Histogram
Name: sigmalang_decompression_duration_seconds
Labels:
  - decode_algorithm: (direct|pattern_lookup|delta_reversal)
  - output_size_range: (tiny|small|medium|large|huge)
Buckets: [0.0001, 0.0005, 0.001, 0.01, 0.1, 1.0, 10.0]
Help: Decompression latency (should be faster than compression)
```

**Performance Asymmetry**:

- Decompression should be 2-10x faster than compression
- If not: Encoding format suboptimal, optimize serialization

---

#### `sigmalang_encode_decode_consistency` (Gauge)

```yaml
Type: Gauge
Name: sigmalang_encode_decode_consistency
Labels:
  - metric: (fidelity_percentage|token_match_rate)
Help: Percentage of encode→decode round-trips that perfectly recover original
```

**Target**: 100% (zero information loss)  
**Acceptable**: 99.9%+  
**Red Flag**: <99%

---

### 1.5 Effectiveness Metrics

#### `sigmalang_compression_effectiveness_score` (Gauge)

```yaml
Type: Gauge
Name: sigmalang_compression_effectiveness_score
Labels:
  - algorithm: (semantic_primitive|learned_pattern|delta_encode|hybrid)
  - data_category: (code|prose|data_structure)
Help: Composite score: (ratio - 1) / (expected_ratio - 1), normalized [0-1]
```

**Formula**:

```
effectiveness = (actual_ratio - 1.0) / (expected_ratio - 1.0)

Where:
- actual_ratio: Real compression achieved
- expected_ratio: Target for this algorithm/data type
- Result: 1.0 = meeting target, 0.5 = 50% of target, >1.0 = exceeding target
```

**Example**:

- Target: 10x, Actual: 12x → Score: (12-1)/(10-1) = 1.22 ✅
- Target: 10x, Actual: 5x → Score: (5-1)/(10-1) = 0.44 ⚠️

---

#### `sigmalang_semantic_redundancy_detected` (Counter)

```yaml
Type: Counter
Name: sigmalang_semantic_redundancy_detected
Labels:
  - redundancy_type: (repeated_pattern|common_substructure|learned_primitive|rsu_reuse)
  - reduction_potential_percent: (10|25|50|75)
Help: Count of optimization opportunities discovered
```

**Optimization Intelligence**:

- High `learned_primitive` reuse: Training is working, continue
- High `rsu_reuse` with low `cache_hit`: RSU index needs optimization
- Detected but not used: Increase compression level or improve selection heuristic

---

## Part 2: Size Tracking Metrics

### 2.1 Original Data Size Distribution

#### `sigmalang_input_size_bytes` (Histogram)

```yaml
Type: Histogram
Name: sigmalang_input_size_bytes
Labels:
  - data_type: (token_sequence|text|semantic_tree)
  - source: (user_input|context_window|conversation_history)
Buckets: [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 10000000]
Help: Distribution of input data sizes before compression
```

**Size Optimization Analysis**:

- Tiny (<1KB): Overhead dominates, consider bypass logic
- Small (1-10KB): Good compression candidates
- Medium (10-100KB): Main workload, optimize aggressively
- Large (>100KB): Likely code or data structures, highest compression potential

---

#### `sigmalang_input_size_percentiles` (Gauge)

```yaml
Type: Gauge
Name: sigmalang_input_size_percentiles
Labels:
  - percentile: (p50|p75|p90|p95|p99)
Help: Percentile distribution of input sizes
```

**Workload Characterization**:

```
If p50 = 2KB and p99 = 500KB:
  → 50% of requests are tiny (optimize for latency)
  → 1% are huge (optimize for ratio)
  → Design two-tier strategy
```

---

### 2.2 Compressed Data Size Distribution

#### `sigmalang_output_size_bytes` (Histogram)

```yaml
Type: Histogram
Name: sigmalang_output_size_bytes
Labels:
  - algorithm: (semantic_primitive|learned_pattern|delta_encode|hybrid)
  - compression_level: (1_fast|5_balanced|9_maximum)
Buckets: [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
Help: Distribution of compressed data sizes
```

**Storage Planning**:

- Median compressed size: Baseline for cache allocation
- p99: Worst-case for buffer sizing
- Distribution shape: Identifies data type prevalence

---

### 2.3 Ratio Calculations

#### `sigmalang_space_savings_bytes` (Gauge)

```yaml
Type: Gauge
Name: sigmalang_space_savings_bytes
Labels:
  - time_window: (last_hour|last_day|since_start)
Help: Cumulative bytes saved: sum(original_size - compressed_size)
```

**Business Impact Metric**:

- Storage reduction quantified in bytes
- Compare against target (e.g., "save 90% of token storage")
- Example: If processing 1TB of tokens, save 900GB → 100GB stored

---

#### `sigmalang_cumulative_compression_ratio` (Gauge)

```yaml
Type: Gauge
Name: sigmalang_cumulative_compression_ratio
Labels:
  - time_window: (1h|1d|1w|all_time)
Help: Overall compression ratio across entire time window
```

**Calculation**:

```
ratio = sum(original_sizes) / sum(compressed_sizes)
```

**Trend Detection**:

- Increasing ratio over time: Learning engine working
- Stable ratio: Consistent performance
- Decreasing ratio: Data distribution changing or model degrading

---

## Part 3: Performance Characteristics Metrics

### 3.1 Algorithm Efficiency Tracking

#### `sigmalang_algorithm_selection_accuracy` (Gauge)

```yaml
Type: Gauge
Name: sigmalang_algorithm_selection_accuracy
Labels:
  - selection_strategy: (heuristic|learned|hybrid)
Help: Percentage of algorithm selections that achieved target ratio
```

**Metric Definition**:

```
For each operation:
  if actual_ratio >= expected_ratio[algorithm]:
    count as correct

accuracy = correct_selections / total_selections
```

**Optimization Insight**:

- <80%: Selection heuristic needs improvement
- 80-95%: Good, acceptable misses
- > 95%: Excellent, trust the heuristic

---

#### `sigmalang_algorithm_efficiency_comparison` (Histogram)

```yaml
Type: Histogram
Name: sigmalang_algorithm_efficiency_comparison
Labels:
  - algorithm_pair: (semantic_vs_pattern|pattern_vs_delta|semantic_vs_delta)
  - metric: (ratio_delta|speed_delta|efficiency_ratio)
Buckets: [0.5, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 2.0]
Help: Relative efficiency between algorithm pairs
```

**Example Interpretation**:

```
semantic_vs_pattern ratio_delta = 1.2
  → Semantic primitive is 20% more efficient (better)

semantic_vs_pattern speed_delta = 0.8
  → Semantic primitive is 20% slower (tradeoff analysis)
```

---

### 3.2 Throughput Analysis

#### `sigmalang_throughput_categories` (Counter)

```yaml
Type: Counter
Name: sigmalang_throughput_categories
Labels:
  - category: (excellent_>100mb|good_10-100mb|acceptable_1-10mb|slow_<1mb)
Help: Categorize compression throughput into buckets
```

**Distribution Analysis**:

- > 90% in "excellent" or "good": Optimization successful
- > 10% in "slow": Investigate why (algorithm, data type, size)

---

### 3.3 Cache Performance Metrics

#### `sigmalang_cache_statistics` (Gauge)

```yaml
Type: Gauge
Name: sigmalang_cache_statistics
Labels:
  - cache_level: (l1_rsu|l2_pattern|l3_lru)
  - metric: (hit_rate|miss_rate|eviction_rate|size_percent)
Help: Cache performance at each tier
```

**L1: RSU Cache** (Recyclable Semantic Units)

```yaml
hit_rate:
  - ? >70
    : Excellent, conversation coherence working
  - 50-70%: Good, baseline
  - <50%: Improve RSU selection or increase cache

eviction_rate:
  - <10%: Healthy cache management
  - ? >30
    : Cache too small or policy too aggressive
```

**L2: Pattern Cache** (Learned patterns)

```yaml
hit_rate:
  - ? >80
    : Training effective
  - 60-80%: Decent, room for improvement
  - <60%: Increase training or pattern vocabulary
```

**L3: LRU Cache** (Recent decompressed data)

```yaml
hit_rate:
  - ? >60
    : Temporal locality working
  - 40-60%: Moderate
  - <40%: Poor temporal locality, adjust window size
```

---

#### `sigmalang_cache_eviction_reason_distribution` (Counter)

```yaml
Type: Counter
Name: sigmalang_cache_eviction_reason_distribution
Labels:
  - reason: (lru|size_pressure|age_threshold|manual_clear)
Help: Why items were evicted from cache
```

**Optimization Insight**:

- High `size_pressure`: Increase cache size
- High `age_threshold`: Patterns aging out, lower threshold or retrain
- High `manual_clear`: Check for correctness issues requiring cache reset

---

### 3.4 Compression Level vs Speed Tradeoffs

#### `sigmalang_compression_level_tradeoff` (Histogram)

```yaml
Type: Histogram
Name: sigmalang_compression_level_tradeoff
Labels:
  - level: (1_fast|5_balanced|9_maximum)
  - metric: (ratio|speed_ms|efficiency)
Buckets: [Varies by metric - see below]
Help: Compression level impact on ratio and speed
```

**Tradeoff Matrix**:

```
Level 1 (Fast):
  Ratio: 5-8x
  Speed: <1ms
  Use: Streaming, real-time

Level 5 (Balanced):
  Ratio: 10-15x
  Speed: 5-10ms
  Use: Default, batch processing

Level 9 (Maximum):
  Ratio: 20-30x
  Speed: 50-100ms
  Use: Offline, storage optimization
```

**Dynamic Selection Logic**:

- Real-time: Use Level 1 (speed > compression)
- Batch: Use Level 5 (balanced)
- Storage: Use Level 9 (compression > speed)
- Adaptive: Start Level 5, adjust based on latency tail

---

## Part 4: Resource Utilization Metrics

### 4.1 CPU Usage During Compression

#### `sigmalang_cpu_usage_percent` (Gauge)

```yaml
Type: Gauge
Name: sigmalang_cpu_usage_percent
Labels:
  - algorithm: (semantic_primitive|learned_pattern|delta_encode|hybrid)
  - cpu_core: (total|core_0|core_1|...)
Help: CPU utilization percentage during compression
```

**Baseline Expectations**:

- Single-threaded operation: 25-100% on one core
- Multi-threaded batch: Can scale to multiple cores
- Low utilization (<20%): Likely I/O-bound or under-loaded

---

#### `sigmalang_cpu_cycles_per_byte` (Gauge)

```yaml
Type: Gauge
Name: sigmalang_cpu_cycles_per_byte
Labels:
  - algorithm: (semantic_primitive|learned_pattern|delta_encode|hybrid)
Help: CPU cycles required to compress one byte
```

**Efficiency Benchmark**:

```
cycles_per_byte = (cpu_cycles_used) / (bytes_compressed)

Target ranges (lower is better):
- Semantic: 100-500 cycles/byte
- Pattern: 200-1000 cycles/byte
- Delta: 50-300 cycles/byte
```

**Optimization Opportunity**:

- If increasing over time: Degradation detected, profile with perf/VTune
- If baseline high: Algorithm needs optimization (SIMD, cache-aware coding)

---

#### `sigmalang_branch_misprediction_rate` (Gauge)

```yaml
Type: Gauge
Name: sigmalang_branch_misprediction_rate
Labels:
  - algorithm: (semantic_primitive|learned_pattern|delta_encode|hybrid)
Help: CPU branch misprediction rate (from perf)
```

**Optimization Signal**:

- > 10%: Hot path has unpredictable branches
- Improvement: Use branch prediction hinting, restructure conditionals
- Action: Profile with `perf record -b` for hot branches

---

### 4.2 Memory Consumption Patterns

#### `sigmalang_memory_usage_bytes` (Gauge)

```yaml
Type: Gauge
Name: sigmalang_memory_usage_bytes
Labels:
  - component: (rsu_cache|pattern_cache|lru_cache|working_set)
  - algorithm: (semantic_primitive|learned_pattern|delta_encode)
Help: Memory consumption by component
```

**Memory Budgets** (per algorithm):

```
Semantic Primitive:
  RSU Cache: 10-50MB (1K-10K entries)
  Working: 5-20MB (in-flight compressions)
  Total: 15-70MB

Learned Pattern:
  Pattern Cache: 50-200MB (codebook)
  RSU Cache: 20-100MB
  Working: 10-30MB
  Total: 80-330MB

Delta Encode:
  History Buffer: 5-20MB
  Working: 5-15MB
  Total: 10-35MB
```

**Red Flags**:

- Memory > budget: Leak or misconfiguration
- Growing linearly: Memory leak in encoding
- Sudden spike: Pathological input pattern

---

#### `sigmalang_cache_memory_efficiency` (Gauge)

```yaml
Type: Gauge
Name: sigmalang_cache_memory_efficiency
Labels:
  - cache_level: (l1_rsu|l2_pattern|l3_lru)
Help: Bytes stored per cache hit (lower = more efficient)
```

**Calculation**:

```
efficiency = cache_size_bytes / cache_hit_count
```

**Target**:

- L1 (RSU): 1-10KB per hit (compression working)
- L2 (Pattern): 50-500 bytes per hit (learned patterns efficient)
- L3 (LRU): 100-1000 bytes per hit (decompressed data)

**Interpretation**:

- Increasing efficiency: More compression happening in smaller space
- Decreasing efficiency: Cache pollution or size inefficiency

---

#### `sigmalang_memory_allocation_pattern` (Counter)

```yaml
Type: Counter
Name: sigmalang_memory_allocation_pattern
Labels:
  - pattern: (small_frequent|large_burst|steady_growth)
Help: Classification of memory allocation behavior
```

**Optimization Insights**:

- Small frequent: Object pool opportunities
- Large burst: Batch sizing issue, tune batch size
- Steady growth: Leak suspected

---

### 4.3 I/O Patterns for Data Processing

#### `sigmalang_io_operations_total` (Counter)

```yaml
Type: Counter
Name: sigmalang_io_operations_total
Labels:
  - operation: (read_codebook|write_cache|read_cache|checkpoint_save)
  - io_type: (memory|disk|network)
Help: I/O operations for compression service
```

**Monitoring**:

- `read_codebook`: Should happen once on startup
- `write_cache`: Should be rare (only on eviction)
- `checkpoint_save`: Periodic, quantify frequency

---

#### `sigmalang_io_latency_milliseconds` (Histogram)

```yaml
Type: Histogram
Name: sigmalang_io_latency_milliseconds
Labels:
  - operation: (read_codebook|write_cache|read_cache)
Buckets: [0.1, 1, 10, 100, 1000, 10000]
Help: I/O operation latency
```

**Baselines**:

- Memory operations: <1ms
- Disk operations: 1-100ms
- Network operations: 10-1000ms

**Bottleneck Detection**:

- If disk reads > 10ms: I/O contention, check disk saturation
- If network latency > 100ms: Network issue, check bandwidth/latency

---

#### `sigmalang_io_throughput_mb_per_second` (Gauge)

```yaml
Type: Gauge
Name: sigmalang_io_throughput_mb_per_second
Labels:
  - operation: (read_codebook|write_cache|read_cache)
Help: I/O throughput for streaming operations
```

**Performance Targets**:

- Disk read: 100-500 MB/s (SSD) or 50-200 MB/s (HDD)
- Disk write: 50-300 MB/s
- Network: Depends on link speed

---

### 4.4 Parallelization Effectiveness

#### `sigmalang_parallel_efficiency` (Gauge)

```yaml
Type: Gauge
Name: sigmalang_parallel_efficiency
Labels:
  - algorithm: (semantic_primitive|learned_pattern|delta_encode|hybrid)
  - worker_count: (2|4|8|16)
Help: Parallel speedup as fraction of theoretical (1.0 = linear scaling)
```

**Formula**:

```
efficiency = actual_speedup / theoretical_speedup

Where:
- actual_speedup = baseline_time / parallel_time
- theoretical_speedup = worker_count

Example:
  Baseline (1 worker): 100ms
  2 workers: 60ms
  actual_speedup = 100/60 = 1.67
  efficiency = 1.67/2 = 0.83 (83% of theoretical - good!)
```

**Optimization Targets**:

- > 80%: Excellent, minimal synchronization overhead
- 60-80%: Good, acceptable overhead
- <60%: Poor, needs optimization (reduce locks, better workload balance)

---

#### `sigmalang_worker_utilization_percent` (Gauge)

```yaml
Type: Gauge
Name: sigmalang_worker_utilization_percent
Labels:
  - worker_id: (0|1|2|3|...)
  - metric: (cpu_percent|idle_time_percent)
Help: Per-worker utilization to detect load imbalance
```

**Load Balancing Analysis**:

- All workers ~80-90%: Perfect load distribution
- Some workers <50%: Imbalance detected
  - Action: Check work-stealing algorithm or batch size
- Some workers 100%: Bottleneck on that worker
  - Action: Profile to find hot spot

---

#### `sigmalang_thread_contention_events` (Counter)

```yaml
Type: Counter
Name: sigmalang_thread_contention_events
Labels:
  - lock_name: (rsu_cache_lock|pattern_cache_lock|codebook_lock)
Help: Number of lock contention events detected
```

**Red Flags**:

- > 1000 per second: Lock contention limiting parallelization
- Increasing over time: More parallelism but worse contention
- Action: Switch to lock-free or RwLock for read-heavy locks

---

## Part 5: Optimization Opportunity Metrics

### 5.1 Bottleneck Identification

#### `sigmalang_bottleneck_detection` (Gauge)

```yaml
Type: Gauge
Name: sigmalang_bottleneck_detection
Labels:
  - bottleneck_type: (memory_bandwidth|cpu_compute|cache_misses|io_wait|lock_contention)
  - severity: (critical_>50pct|high_25-50pct|medium_10-25pct|low_<10pct)
Help: Identified bottleneck and its contribution to total latency
```

**Calculation**:

```
For each potential bottleneck:
  severity_percent = bottleneck_time / total_latency * 100

severity classification:
  >50%: Critical (fix immediately)
  25-50%: High (important optimization target)
  10-25%: Medium (optimize after critical/high)
  <10%: Low (Amdahl's law suggests not worth optimizing)
```

**Example Analysis**:

```
Operation: Compress 1MB input

Timeline:
- Parse: 1ms (2%)
- Encode: 35ms (70%) ← CRITICAL
- Optimize: 8ms (16%)
- Serialize: 6ms (12%)
Total: 50ms

Action: Focus optimization on Encode phase
Target: 15% reduction = 5.2ms saved = 10% overall improvement
```

---

#### `sigmalang_optimization_opportunity_score` (Gauge)

```yaml
Type: Gauge
Name: sigmalang_optimization_opportunity_score
Labels:
  - opportunity: (cache_miss_reduction|algorithm_selection|compression_level|
      memory_optimization|parallelization|io_batching)
Help: Estimated impact if optimization completed (0-1 scale)
```

**Scoring Framework**:

```
opportunity_score = (frequency * impact) / max_impact

Where:
- frequency: How often this issue occurs (0-1)
- impact: Performance gain if fixed (0-1)

Example:
- Cache miss in hot path, 20% frequency, 50% impact
- score = 0.2 * 0.5 = 0.1 (10% potential improvement)

Priority = sort by score descending
```

---

### 5.2 Compression Ratio Trends

#### `sigmalang_compression_ratio_anomaly` (Gauge)

```yaml
Type: Gauge
Name: sigmalang_compression_ratio_anomaly
Labels:
  - anomaly_type: (below_expected_ratio|above_expected_ratio|high_variance)
  - algorithm: (semantic_primitive|learned_pattern|delta_encode|hybrid)
Help: Deviation from expected compression ratio in standard deviations
```

**Anomaly Detection**:

```
z_score = (actual_ratio - expected_ratio) / std_dev

Alert conditions:
- z_score < -2: Below expected (underperforming)
- z_score > 2: Above expected (overperforming - analyze why)
- High variance: Inconsistent algorithm
```

**Triggering Action**:

- Below expected: Check algorithm selection or profile for regression
- Above expected: Check if learning engine made breakthrough, promote strategy
- High variance: Unstable input data or algorithm tuning needed

---

#### `sigmalang_compression_ratio_forecast` (Gauge)

```yaml
Type: Gauge
Name: sigmalang_compression_ratio_forecast
Labels:
  - algorithm: (semantic_primitive|learned_pattern|delta_encode|hybrid)
  - forecast_horizon: (next_hour|next_day)
Help: Predicted compression ratio using linear trend + seasonality
```

**Predictive Analytics**:

- If forecast shows decline: Early warning of data distribution change
- If forecast shows improvement: Learning engine making progress
- Large prediction interval: High uncertainty, needs investigation

---

### 5.3 Algorithm Selection Monitoring

#### `sigmalang_algorithm_selection_efficiency` (Gauge)

```yaml
Type: Gauge
Name: sigmalang_algorithm_selection_efficiency
Labels:
  - scenario: (token_sequences|code_snippets|natural_language|mixed_content)
Help: How well algorithm selection matches optimal algorithm for scenario
```

**Measurement**:

```
For each scenario:
  best_ratio = max(all_algorithms[ratio])
  selected_ratio = selected_algorithm[ratio]
  efficiency = selected_ratio / best_ratio

Target: >0.95 (selecting within 5% of optimal)
```

---

#### `sigmalang_suboptimal_scenario_detection` (Counter)

```yaml
Type: Counter
Name: sigmalang_suboptimal_scenario_detection
Labels:
  - scenario: (token_sequences|code_snippets|natural_language|mixed_content)
  - suboptimal_reason: (algorithm_mismatch|compression_level_too_low|
      cache_miss|pattern_not_learned)
Help: Detected scenarios where compression could be improved
```

**Optimization Workflow**:

```
1. Detect suboptimal scenario
2. Analyze reason from labels
3. Apply corrective action:
   - algorithm_mismatch: Improve selection heuristic
   - compression_level_too_low: Use higher level
   - cache_miss: Expand cache or improve eviction policy
   - pattern_not_learned: Trigger training on pattern
4. Measure improvement
5. Update decision tree
```

---

## Part 6: Prometheus Queries for Performance Analysis

### 6.1 Compression Effectiveness Queries

#### Overall Compression Ratio

```promql
avg(sigmalang_compression_ratio)
```

#### Algorithm Comparison

```promql
avg by (algorithm) (sigmalang_compression_ratio)
```

#### Compression Ratio Over Time (1-hour moving average)

```promql
avg_over_time(sigmalang_compression_ratio[1h])
```

#### Percentile Distribution (p50, p95, p99)

```promql
histogram_quantile(0.50, sigmalang_compression_ratio)
histogram_quantile(0.95, sigmalang_compression_ratio)
histogram_quantile(0.99, sigmalang_compression_ratio)
```

---

### 6.2 Performance Analysis Queries

#### Compression Latency Percentiles

```promql
histogram_quantile(0.50, rate(sigmalang_compression_duration_seconds_bucket[5m]))
histogram_quantile(0.95, rate(sigmalang_compression_duration_seconds_bucket[5m]))
histogram_quantile(0.99, rate(sigmalang_compression_duration_seconds_bucket[5m]))
```

#### Throughput Analysis

```promql
avg by (algorithm) (sigmalang_compression_throughput_bytes_per_second)
```

#### Throughput Trend (increasing/decreasing)

```promql
rate(sigmalang_compression_throughput_bytes_per_second[1h])
```

---

### 6.3 Resource Utilization Queries

#### Cache Hit Rate

```promql
(
  increase(sigmalang_compression_requests_total{status="cache_hit"}[5m])
  /
  increase(sigmalang_compression_requests_total[5m])
) * 100
```

#### Memory Usage Trend

```promql
avg by (component) (sigmalang_memory_usage_bytes)
```

#### CPU Efficiency

```promql
avg by (algorithm) (sigmalang_cpu_cycles_per_byte)
```

---

### 6.4 Anomaly Detection Queries

#### Compression Ratio Below Threshold

```promql
(sigmalang_compression_ratio < 5) and
(sigmalang_compression_requests_total > 0)
```

#### High Latency Detection

```promql
histogram_quantile(0.99, sigmalang_compression_duration_seconds_bucket) > 0.1
```

#### Memory Leak Detection

```promql
increase(sigmalang_memory_usage_bytes[1h]) > 0
```

---

### 6.5 Optimization Opportunity Queries

#### Bottleneck Analysis

```promql
max by (bottleneck_type) (sigmalang_bottleneck_detection)
```

#### Algorithm Selection Accuracy

```promql
avg(sigmalang_algorithm_selection_accuracy)
```

#### Optimization Opportunity Ranking

```promql
sort_desc(sigmalang_optimization_opportunity_score)
```

---

## Part 7: Performance Impact Estimation

### 7.1 Optimization Impact Framework

#### Estimated Impact Categories

| Optimization                      | Effort | Potential Gain             | Complexity | Priority |
| --------------------------------- | ------ | -------------------------- | ---------- | -------- |
| **Cache hit rate improvement**    | Low    | 5-10% throughput           | Medium     | HIGH     |
| **Algorithm selection heuristic** | Medium | 15-20% compression         | High       | MEDIUM   |
| **SIMD vectorization**            | High   | 30-40% CPU                 | Very High  | MEDIUM   |
| **Lock-free data structures**     | High   | 20-50% parallel efficiency | Very High  | LOW      |
| **Memory pool allocation**        | Low    | 10-15% CPU                 | Low        | HIGH     |
| **Compression level tuning**      | Low    | 5-15% latency              | Low        | MEDIUM   |
| **I/O batching**                  | Medium | 20-40% I/O throughput      | Medium     | MEDIUM   |

---

### 7.2 Amdahl's Law Application

For any optimization, calculate realistic impact:

```
Speedup = 1 / ((1 - f) + f/S)

Where:
- f = fraction of time spent in optimizable part
- S = speedup achieved in that part

Example:
- Optimizing Encode phase (70% of time)
- Achieve 2x speedup in Encode
- Speedup = 1 / ((1 - 0.70) + 0.70/2)
         = 1 / (0.30 + 0.35)
         = 1 / 0.65
         = 1.54x (54% overall improvement)
```

---

### 7.3 Measurement Before/After Template

When implementing optimizations:

```yaml
Optimization: [Name]
Expected Impact: [% improvement]

Before:
  Metric1: [value]
  Metric2: [value]
  Metric3: [value]

After (5 runs):
  Run1: [values]
  Run2: [values]
  Run3: [values]
  Run4: [values]
  Run5: [values]

Analysis:
  Mean improvement: [%]
  Confidence interval: [±%]
  Statistical significance: [p-value]
  Amdahl's law validation: [matches prediction? Y/N]

Commit if:
  - Improvement >= 90% of expected
  - Confidence interval is tight (±5%)
  - No regression in other metrics
```

---

## Part 8: Sub-Linear Algorithm Optimization Opportunities

### 8.1 Bloom Filter Applications

#### Fast Set Membership Check (O(1))

```python
# Use case: Check if input pattern seen before
# Current: Linear search through pattern cache
# Optimization: Bloom filter for fast negative confirmation

sigmalang_bloom_filter_lookups_total (Counter)
  - status: (confirmed_present|confirmed_absent|false_positive_rate)

Expected improvement:
  - 10x faster pattern cache misses
  - False positive rate: 1% (acceptable)
```

---

### 8.2 HyperLogLog Applications

#### Cardinality Estimation (O(1) space)

```python
# Use case: Estimate unique semantic patterns
# Current: Store all patterns in memory
# Optimization: HyperLogLog cardinality tracking

sigmalang_unique_patterns_cardinality (Gauge)
  - algorithm: (semantic_primitive|learned_pattern)

Expected improvement:
  - 100x memory reduction
  - 1-2% error acceptable
  - Early warning when cardinality growing too fast
```

---

### 8.3 Count-Min Sketch Applications

#### Frequency Estimation (O(1) space)

```python
# Use case: Track most frequent compression patterns
# Current: HashMap with all frequencies
# Optimization: Count-Min Sketch

sigmalang_pattern_frequency_estimate (Gauge)
  - pattern_id: [0-256]
  - estimated_frequency: [count]
  - error_bound_percent: [±%]

Expected improvement:
  - 50x memory reduction vs HashMap
  - Error bound: ±10% (configurable)
```

---

### 8.4 LSH (Locality Sensitive Hashing) Applications

#### Fast Semantic Similarity (O(1) expected)

```python
# Use case: Find similar semantic trees for reuse
# Current: Compare against all learned patterns
# Optimization: LSH index for approximate matching

sigmalang_lsh_index_query_time_ms (Histogram)

Improvement:
  - 1000x faster approximate semantic search
  - Miss rate acceptable: 5-10%
  - Fall back to exact match on demand
```

---

### 8.5 t-Digest Applications

#### Quantile Tracking (O(δ) space)

```python
# Use case: Track compression ratio distribution
# Current: Store all ratios for percentile calculation
# Optimization: t-Digest for memory-efficient percentiles

sigmalang_tdigest_compression_ratio_percentiles (Gauge)
  - percentile: (p50|p75|p90|p95|p99)

Improvement:
  - 1000x memory reduction
  - Bounded error: ±1% at high percentiles
```

---

### 8.6 HNSW (Hierarchical Navigable Small World) Graph

#### Fast Approximate Nearest Neighbor (O(log n))

```python
# Use case: Find semantically similar cached compressions
# Current: Semantic search through all RSUs
# Optimization: HNSW graph for O(log n) search

sigmalang_hnsw_semantic_search_ms (Histogram)
  - search_depth: (shallow|deep)

Improvement:
  - 100-1000x faster semantic search
  - Accuracy: 95%+ of exact nearest neighbor
```

---

## Part 9: Implementation Roadmap

### Phase 1: Core Metrics (Week 1-2)

- [ ] Compression operation metrics
- [ ] Size tracking metrics
- [ ] Basic duration histograms
- [ ] Prometheus endpoint

**Target**: Full observability into compression operations

### Phase 2: Performance Analysis (Week 3-4)

- [ ] Cache statistics
- [ ] Memory profiling
- [ ] Throughput tracking
- [ ] Phase breakdown timing

**Target**: Identify top 3 optimization opportunities

### Phase 3: Sub-Linear Algorithms (Week 5-6)

- [ ] Bloom filter for pattern lookup
- [ ] Count-Min sketch for frequency
- [ ] HyperLogLog for cardinality
- [ ] Measure improvements

**Target**: 10-50% performance improvement

### Phase 4: Predictive Optimization (Week 7-8)

- [ ] Anomaly detection
- [ ] Trend forecasting
- [ ] Auto-alerting on degradation
- [ ] Dashboard alerts

**Target**: Proactive problem detection

---

## Part 10: Grafana Dashboard Specifications

### Dashboard 1: Compression Overview

```
Row 1:
  - Compression Request Rate (graph)
  - Avg Compression Ratio (gauge)
  - Space Saved (stat)
  - Cache Hit Rate (gauge)

Row 2:
  - Compression Ratio by Algorithm (heatmap)
  - Compression Duration p50/p95/p99 (graph)
  - Throughput by Algorithm (graph)
  - Error Rate Trend (graph)
```

### Dashboard 2: Performance Bottlenecks

```
Row 1:
  - Phase Duration Breakdown (pie chart)
  - CPU Cycles per Byte (graph)
  - Memory Usage by Component (stacked bar)
  - Cache Miss Rate (graph)

Row 2:
  - Latency Heat Map (heatmap)
  - Bottleneck Detection (table)
  - Optimization Opportunity Score (sorted list)
```

### Dashboard 3: Optimization Tracking

```
Row 1:
  - Algorithm Selection Accuracy (gauge)
  - Parallel Efficiency (graph)
  - Worker Utilization (sparklines)
  - Lock Contention Events (graph)

Row 2:
  - Compression Ratio Anomalies (graph)
  - Compression Ratio Forecast (graph)
  - Suboptimal Scenario Detection (table)
```

---

## Appendix A: Metric Collection Code Template

```python
from prometheus_client import Counter, Histogram, Gauge
import time
from functools import wraps

# Compression metrics
compression_requests = Counter(
    'sigmalang_compression_requests_total',
    'Compression requests',
    ['status', 'algorithm', 'data_type']
)

compression_duration = Histogram(
    'sigmalang_compression_duration_seconds',
    'Compression duration',
    ['algorithm', 'compression_level'],
    buckets=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
)

compression_ratio = Histogram(
    'sigmalang_compression_ratio',
    'Compression ratio',
    ['algorithm', 'data_category'],
    buckets=[1.5, 2, 5, 10, 20, 50, 100]
)

# Decorator for automatic metric collection
def track_compression(algorithm, compression_level='5_balanced'):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'
            try:
                result = func(*args, **kwargs)
                input_size, output_size = result.get('size_info', (0, 0))
                if output_size > 0:
                    ratio = input_size / output_size
                    compression_ratio.labels(
                        algorithm=algorithm,
                        data_category='general'
                    ).observe(ratio)
                return result
            except Exception as e:
                status = 'failure'
                raise
            finally:
                duration = time.time() - start_time
                compression_duration.labels(
                    algorithm=algorithm,
                    compression_level=compression_level
                ).observe(duration)
                compression_requests.labels(
                    status=status,
                    algorithm=algorithm,
                    data_type='general'
                ).inc()
        return wrapper
    return decorator
```

---

## Conclusion

This metrics design strategy transforms ΣLANG compression monitoring from passive observation to **active performance optimization**. Each metric answers the critical question: _"Where should we optimize next?"_

**Key Principles**:

1. **Measure ruthlessly** - Every optimization needs baseline metrics
2. **Analyze bottlenecks** - Amdahl's law guides prioritization
3. **Optimize intentionally** - Sub-linear algorithms for 10-100x improvements
4. **Validate impact** - Before/after measurement discipline
5. **Automate detection** - Anomalies trigger alerts and action

By implementing this comprehensive metrics framework, the ΣLANG compression service gains the visibility needed to achieve and sustain **10-50x compression ratios** while maintaining sub-10ms latency targets.
