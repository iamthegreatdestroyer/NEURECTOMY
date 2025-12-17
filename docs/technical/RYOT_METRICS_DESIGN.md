# Ryot LLM Inference Service: Comprehensive Metrics Design

**Author**: ML/Inference Team (TENSOR Mode)  
**Status**: Production Design Specification  
**Version**: 1.0  
**Date**: December 2025

---

## Executive Summary

This document defines a complete metrics strategy for the Ryot LLM inference service, covering request handling, token generation, GPU utilization, and performance characteristics. The metrics are designed to be lightweight, production-ready, and provide comprehensive observability for model inference operations.

**Key Design Principles:**

- **Sub-linear overhead**: Metrics collection < 1% latency impact
- **Model-agnostic labels**: Support for multiple model architectures
- **Granular tracking**: Token-level, batch-level, and request-level metrics
- **Cache-aware monitoring**: Track hot-loading and cold-loading patterns
- **GPU-optimized histograms**: Bucket ranges tuned for LLM inference

---

## Part 1: Core Metrics Definitions

### 1.1 Request Counting Metrics

#### **ryot_inference_requests_total** (Counter)

```yaml
Type: Counter
Purpose: Track total inference requests with success/error/timeout status
Labels:
  - status: [success, error, timeout, partial]
  - model: Model name (e.g., "bitnet-7b", "llama-13b", "gpt2-small")
  - endpoint: API endpoint (e.g., "generate", "stream", "batch")
  - error_type: [timeout, oom, invalid_request, model_error] (if status=error)

Example Usage:
  ryot_inference_requests_total{status="success",model="bitnet-7b",endpoint="generate"} 15234
  ryot_inference_requests_total{status="error",model="bitnet-7b",error_type="timeout"} 42
```

**Rationale**: Enables SLO tracking and error rate monitoring. Error_type label allows debugging specific failure modes.

---

#### **ryot_inference_errors_total** (Counter)

```yaml
Type: Counter
Purpose: Detailed error tracking by error category
Labels:
  - error_type:
      [
        oom,
        timeout,
        cuda_error,
        invalid_prompt,
        model_not_found,
        inference_error,
        tokenization_error,
      ]
  - model: Model identifier
  - severity: [critical, warning, info]

Subcategories:
  - Out of Memory (OOM): GPU memory exhaustion
  - Timeout: Exceeds max inference time
  - CUDA Errors: GPU kernel failures
  - Invalid Input: Malformed prompts
  - Model Issues: Missing/corrupted models
```

**Rationale**: Separates error tracking from success/failure ratio. Provides root cause analysis data.

---

### 1.2 Latency Metrics

#### **ryot_inference_latency_seconds** (Histogram)

```yaml
Type: Histogram
Purpose: End-to-end inference latency (prompt → completion)
Labels:
  - model: Model name
  - batch_size: Bucketed size [1, 2-4, 5-8, 9-16, 17-32, 33+]
  - generation_mode: [standard, streaming, batch]

Buckets: ┌─────────────────────────────────────────────────┐
  │ LATENCY BUCKET DESIGN (LLM-Optimized)           │
  ├─────────────────────────────────────────────────┤
  │ 0.01s   │ <10ms    (cache hit, very short)       │
  │ 0.025s  │ 25ms     (cache hit, short)            │
  │ 0.05s   │ 50ms     (TTFT target)                 │
  │ 0.1s    │ 100ms    (warm start)                  │
  │ 0.25s   │ 250ms    (medium requests)             │
  │ 0.5s    │ 500ms    (substantial generation)      │
  │ 1.0s    │ 1s       (long generation)             │
  │ 2.5s    │ 2.5s     (very long generation)        │
  │ 5.0s    │ 5s       (batch operations)            │
  │ 10.0s   │ 10s      (cold start or complex)       │
  │ 30.0s   │ 30s      (outliers, edge cases)        │
  │ +Inf    │ >30s     (SLA violations)              │
  └─────────────────────────────────────────────────┘

Rationale:
  - Sub-50ms: Cache hits and KV cache reuse
  - 50-500ms: Typical inference window
  - 500ms-5s: Multi-token generations
  - ? >5
    : Edge cases, batch operations, cold starts
```

**Formula**: `end_time - start_time`

---

#### **ryot_ttft_latency_seconds** (Histogram)

```yaml
Type: Histogram (Time-To-First-Token)
Purpose: Measure prompt processing + first token generation time
Labels:
  - model: Model identifier
  - prompt_size: [tiny: <50, small: 50-500, medium: 500-2k, large: 2k-10k, xlarge: >10k]
  - device: [gpu, cpu] (if supported)

Buckets (TTFT optimized):
  - 0.005s  (5ms)    → Exceptional (KV cache hit)
  - 0.01s   (10ms)   → Excellent
  - 0.025s  (25ms)   → Good
  - 0.05s   (50ms)   → Target for production
  - 0.1s    (100ms)  → Acceptable
  - 0.25s   (250ms)  → Slow
  - 0.5s    (500ms)  → Very slow
  - 1.0s    (1s)     → Cold start likely
  - 2.5s    (2.5s)   → SLA concern
  - +Inf

Rationale: TTFT is critical UX metric. Separate from total latency for
independent analysis. Warm vs cold cache effects are visible here.
```

---

#### **ryot_model_loading_time_seconds** (Histogram)

```yaml
Type: Histogram
Purpose: Model initialization time (model → ready for inference)
Labels:
  - model: Model name
  - load_type: [cold_start, warm_start, cache_hit]
  - device_type: [gpu, cpu]

Buckets:
  - 0.5s
  - 1.0s
  - 2.5s
  - 5.0s
  - 10.0s
  - 30.0s
  - 60.0s
  - 120.0s
  - +Inf

Example Distribution:
  Cold start (first load): ~5-30s (GPU memory transfer)
  Warm start (model cached): ~500ms-2s (memory move to GPU)
  Cache hit (ready to infer): ~10-50ms (no-op, immediate)
```

**Rationale**: Separate metric for model loading critical for understanding infrastructure performance. Identifies cache eviction patterns.

---

### 1.3 Token Metrics

#### **ryot_tokens_generated_total** (Counter)

```yaml
Type: Counter
Purpose: Total tokens generated across all inferences
Labels:
  - model: Model identifier
  - token_type: [completion, prompt, total]
  - generation_mode: [standard, streaming, batch]

Relationship: total_tokens = prompt_tokens + completion_tokens
  tracked separately for independent analysis

Example:
  ryot_tokens_generated_total{token_type="completion",model="bitnet-7b"} 1847392
  ryot_tokens_generated_total{token_type="prompt",model="bitnet-7b"} 524128
```

**Rationale**: Foundation for token rate calculations and cost tracking.

---

#### **ryot_tokens_per_second** (Gauge)

```yaml
Type: Gauge (computed from histogram buckets)
Purpose: Real-time token generation throughput rate
Labels:
  - model: Model identifier
  - time_window: [1m, 5m, 15m] (computed from metrics)
  - batch_size: Bucketed size

Calculation:
  TPS(t) = Δ(tokens_generated) / Δ(time_seconds)

Production Targets:
  BitNet 1.58b (CPU):     100-150 tps
  Llama 7b (GPU):         150-250 tps
  Llama 13b (GPU):        80-150 tps
  GPT2-small (GPU):       200-400 tps

Monitoring Strategy:
  - Track 1m rolling window for responsiveness
  - Alert if TPS < 80% of model baseline
  - Identify batch size impact on TPS
```

**Rationale**: TPS is critical for SLA compliance and capacity planning.

---

#### **ryot_token_efficiency_ratio** (Gauge)

```yaml
Type: Gauge
Purpose: Model-specific efficiency metrics (tokens generated per second per GPU)
Labels:
  - model: Model identifier
  - batch_size: [1, 2-4, 5-8, 9-16, 17-32, 33+]
  - dtype: [float32, float16, int8, bfloat16]

Formula:
  efficiency = tokens_per_second / gpu_memory_used_percent

  Measures: "How efficiently are we using GPU resources?"

  High efficiency (>500):   Well-optimized inference
  Normal (100-500):         Good baseline
  Low (<100):               Investigate optimization opportunities
```

**Rationale**: Captures relationship between throughput and resource consumption.

---

#### **ryot_token_cost_estimate_usd** (Counter)

```yaml
Type: Counter
Purpose: Estimated cost of token generation (for billing/optimization)
Labels:
  - model: Model identifier
  - token_type: [prompt, completion]
  - cost_model: [fixed_rate, tiered, dynamic]

Cost Mapping (example):
  Prompt tokens: $0.0015 per 1000 tokens
  Completion tokens: $0.006 per 1000 tokens

  Calculation: (prompt_tokens * rate_prompt + completion_tokens * rate_completion) / 1000

Example:
  ryot_token_cost_estimate_usd{model="bitnet-7b",token_type="prompt"} 142.67
  ryot_token_cost_estimate_usd{model="bitnet-7b",token_type="completion"} 1105.40
```

**Rationale**: Essential for cost-aware scaling and resource allocation.

---

### 1.4 GPU Memory and Hardware Metrics

#### **ryot_gpu_memory_usage_bytes** (Gauge)

```yaml
Type: Gauge
Purpose: GPU memory consumption by model
Labels:
  - model: Model identifier
  - memory_type: [model_weights, kv_cache, activations, peak]
  - device_id: GPU device ID (if multi-GPU)

Example Distribution:
  Model Weights (BitNet 1.58b):  ~1.6 GB (1-bit quantization)
  KV Cache (2k context):         ~1.2 GB
  Activations (batch=1):         ~0.8 GB
  Total Peak (batch=32):         ~8-12 GB

Memory Breakdown:
  ┌────────────────────────────────────┐
  │ GPU Memory Components              │
  ├────────────────────────────────────┤
  │ Model Weights:     50-60%          │
  │ KV Cache:          20-30%          │
  │ Activations:       10-20%          │
  │ Buffers/Overhead:  5-10%           │
  └────────────────────────────────────┘
```

**Rationale**: Critical for OOM prevention and batch size optimization.

---

#### **ryot_gpu_memory_percentage** (Gauge)

```yaml
Type: Gauge
Purpose: GPU memory utilization as percentage of available
Labels:
  - model: Model identifier
  - device_id: GPU device ID
  - batch_size: Current batch size

Alert Thresholds:
  <20%: Underutilized (opportunity to increase batch)
  20-70%: Optimal zone
  70-85%: Getting tight
  85-95%: Risky zone (risk of OOM)
  ? >95
  : Critical (near OOM)

Example: ryot_gpu_memory_percentage{model="bitnet-7b",device_id="0",batch_size="8"} 62.4
```

---

#### **ryot_gpu_utilization_percent** (Gauge)

```yaml
Type: Gauge
Purpose: GPU compute utilization (not just memory)
Labels:
  - model: Model identifier
  - device_id: GPU device ID
  - kernel_type: [matmul, attention, other]

Healthy Indicators:
  >80%: Good utilization
  50-80%: Acceptable
  <50%: Underutilized, check for:
    - I/O bottlenecks
    - Small batch sizes
    - Inefficient kernels
```

---

#### **ryot_kv_cache_size_bytes** (Gauge)

```yaml
Type: Gauge
Purpose: Key-Value cache memory usage (critical for transformer models)
Labels:
  - model: Model identifier
  - context_length: [512, 1024, 2048, 4096, 8192, 16384]
  - batch_size: Current batch size

KV Cache Growth:
  Formula: 2 * num_layers * batch_size * context_length * (d_model/8) bytes

  Example (Llama 7B):
    Layers: 32
    d_model: 4096
    Context: 2048
    Batch: 8

    KV Cache = 2 * 32 * 8 * 2048 * 512 = ~1 GB

Cache Optimization Opportunities:
  - Attention window reduction (sliding window)
  - KV cache quantization
  - Sparse attention patterns
```

**Rationale**: KV cache is often the largest memory consumer. Tracking enables optimization.

---

### 1.5 Batch and Concurrency Metrics

#### **ryot_active_inference_requests** (Gauge)

```yaml
Type: Gauge
Purpose: Current number of in-flight inference requests
Labels:
  - model: Model identifier
  - request_type: [standard, streaming, batch]
  - endpoint: API endpoint

Example:
  ryot_active_inference_requests{model="bitnet-7b",request_type="standard"} 42
  ryot_active_inference_requests{model="bitnet-7b",request_type="streaming"} 15
```

**Rationale**: Indicates load and helps identify queuing bottlenecks.

---

#### **ryot_batch_size_distribution** (Histogram)

```yaml
Type: Histogram
Purpose: Distribution of batch sizes used in inference
Labels:
  - model: Model identifier
  - request_type: [standard, streaming, batch]

Buckets:
  - 1     (single request)
  - 2-4   (small batch)
  - 5-8   (medium batch)
  - 9-16  (large batch)
  - 17-32 (xlarge batch)
  - 33+   (huge batch)

Expected Distribution:
  Single requests (batch=1): ~70%
  Small batches (2-4): ~15%
  Medium batches (5-8): ~10%
  Large batches (9-16): ~4%
  XL batches (17-32): ~1%

Example Query: Histogram_quantile(0.95, rate(ryot_batch_size_distribution_bucket[5m]))
```

**Rationale**: Shows actual request patterns and optimization opportunities.

---

#### **ryot_queue_depth** (Gauge)

```yaml
Type: Gauge
Purpose: Number of requests waiting for GPU availability
Labels:
  - model: Model identifier
  - priority_level: [high, normal, low]

Thresholds:
  Queue depth > 10: Possible bottleneck
  Queue depth > 50: Capacity issue
  Queue depth > 100: Scale-out required
```

---

### 1.6 Cache Effectiveness Metrics

#### **ryot_kv_cache_hit_ratio** (Gauge)

```yaml
Type: Gauge
Purpose: How often KV cache is reused vs. recomputed
Labels:
  - model: Model identifier
  - context_type: [conversation, batch, streaming]

Formula: hit_ratio = cache_hits / (cache_hits + cache_misses)

Healthy Targets:
  Conversation context: 75-95% (high reuse)
  Batch processing: 10-30% (low reuse)
  Streaming: 30-70% (medium reuse)

Example: ryot_kv_cache_hit_ratio{model="bitnet-7b",context_type="conversation"} 0.87
```

---

#### **ryot_model_cache_evictions_total** (Counter)

```yaml
Type: Counter
Purpose: Track when models are evicted from GPU memory cache
Labels:
  - model: Model identifier
  - eviction_reason: [memory_pressure, timeout, manual]

High eviction rate indicates:
  - Insufficient GPU memory
  - Too many models
  - Need for model sharding/quantization
```

---

#### **ryot_prompt_cache_effectiveness** (Histogram)

```yaml
Type: Histogram
Purpose: Measure reuse of prompt processing results
Labels:
  - model: Model identifier
  - context_reuse_window: [1h, 24h, 7d]

Example: Within 1 hour, how many duplicate prompts were processed?
  Track to identify high-value caching opportunities
```

---

### 1.7 Startup and Initialization Metrics

#### **ryot_model_load_attempts_total** (Counter)

```yaml
Type: Counter
Purpose: Track model loading attempts and outcomes
Labels:
  - model: Model identifier
  - result: [success, failed, partial]
  - load_type: [cold, warm, from_cache]

Example: ryot_model_load_attempts_total{model="bitnet-7b",result="success",load_type="cold"} 123
```

---

#### **ryot_initialization_startup_time_seconds** (Histogram)

```yaml
Type: Histogram
Purpose: Time from service start to ready for inference
Labels:
  - startup_phase: [model_load, cache_init, warmup, ready]

Buckets:
  - 1.0s
  - 5.0s
  - 10.0s
  - 30.0s
  - 60.0s
  - +Inf
```

---

## Part 2: Histogram Bucket Strategy

### 2.1 Rationale for Bucket Ranges

#### **Key Principles**

```
1. Exponential Growth: Buckets should scale by ~2.5x
   Reason: Captures resolution at low values, efficiency at high values

2. Model-Specific Baselines: Calibrate to model characteristics
   - CPU inference (BitNet): Expect 50-200ms latency
   - GPU inference (Llama): Expect 20-100ms latency

3. Business Targets: Align with SLAs
   - P50 < 100ms (target TTFT)
   - P95 < 500ms
   - P99 < 2.5s

4. Alert Thresholds: Buckets enable clear alerting
```

### 2.2 Token Rate Buckets

```yaml
ryot_tokens_per_second Buckets:
  ┌──────────────────────────────────────────┐
  │ Token Rate Histogram                     │
  ├──────────────────────────────────────────┤
  │ 10 tps      │ Very slow inference        │
  │ 25 tps      │ Slow                       │
  │ 50 tps      │ Below expected             │
  │ 100 tps     │ Good for CPU models        │
  │ 150 tps     │ Good for GPU models        │
  │ 250 tps     │ Excellent GPU throughput   │
  │ 400 tps     │ Exceptional               │
  │ 600 tps     │ Peak performance          │
  │ +Inf        │ Edge cases                │
  └──────────────────────────────────────────┘

Justification:
  - Sub-50 tps: Early warning of degradation
  - 50-150 tps: Normal operational range
  - 150-600 tps: Optimization opportunities
  - >600 tps: Batch optimization success
```

### 2.3 Batch Size Buckets

```yaml
ryot_batch_size_distribution Buckets:
  Discrete buckets (not continuous):
    - 1
    - 2, 3, 4
    - 5, 6, 7, 8
    - 9, 10, ..., 16
    - 17, 18, ..., 32
    - 33+

Reasoning:
  - Batch=1 has different characteristics (no batching overhead)
  - Small batches (2-8): Sweet spot for latency
  - Medium (9-16): Balanced throughput
  - Large (17-32): Maximum throughput
  - XL (33+): Rare, edge cases
```

---

## Part 3: Label Strategy

### 3.1 Dimension Hierarchy

```yaml
Primary Dimensions:
  1. Model: Direct identity (bitnet-7b, llama-13b, etc.)
  2. Status/Result: Success/error/timeout
  3. Resource Context: Batch size, device type

Secondary Dimensions:
  4. Request Type: Mode of operation (standard, streaming, batch)
  5. Error Category: Specific failure reason
  6. Performance Markers: Context length, model size

Avoid Over-Labeling: ✗ Too many label combinations → Cardinality explosion
  ✗ Real-time computed labels → Overhead
  ✓ Pre-computed labels at request time
  ✓ Max 5-6 labels per metric
```

### 3.2 Label Examples

```yaml
Good Label Design:
  Counter metric with labels:
    - model: "bitnet-7b"
    - status: "success"
    - endpoint: "generate"
  Cardinality: 3 models × 3 statuses × 3 endpoints = 27 combinations

Problematic Design: Histogram with 10+ labels from request context
  - Cardinality explosion
  - Prometheus storage overhead
  - Query performance degradation

Solution:
  - Use counters for low-cardinality dimensions
  - Use separate metrics for high-cardinality data
  - Pre-aggregate expensive computations
```

---

## Part 4: Integration with Prometheus

### 4.1 Scrape Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: "ryot-inference"
    static_configs:
      - targets: ["localhost:9090"]

    # Scrape metrics endpoint
    metrics_path: "/metrics"

    # Reduce cardinality from high-variance labels
    metric_relabel_configs:
      - source_labels: [__name__]
        regex: "ryot_.*"
        action: "keep"
```

### 4.2 Recording Rules

```yaml
# ryot_recording_rules.yml
groups:
  - name: ryot_inference_rules
    interval: 30s
    rules:
      # Compute token rate
      - record: ryot:tokens_per_second:rate1m
        expr: rate(ryot_tokens_generated_total[1m])

      # Compute error rate
      - record: ryot:error_rate:rate5m
        expr: rate(ryot_inference_errors_total[5m]) / rate(ryot_inference_requests_total[5m])

      # Compute latency percentile
      - record: ryot:inference_latency:p95:5m
        expr: histogram_quantile(0.95, rate(ryot_inference_latency_seconds_bucket[5m]))

      # Batch size average
      - record: ryot:batch_size:avg:5m
        expr: rate(ryot_batch_size_distribution_sum[5m]) / rate(ryot_batch_size_distribution_count[5m])

      # GPU memory efficiency
      - record: ryot:gpu_efficiency:ratio
        expr: ryot_tokens_per_second / (ryot_gpu_memory_percentage / 100)
```

### 4.3 Alert Rules

```yaml
# ryot_alerts.yml
groups:
  - name: ryot_inference_alerts
    interval: 30s
    rules:
      - alert: RyotHighErrorRate
        expr: rate(ryot_inference_errors_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
          component: ryot
        annotations:
          summary: "Ryot error rate > 5%"
          description: "Error rate: {{ $value | humanizePercentage }}"

      - alert: RyotHighLatency
        expr: histogram_quantile(0.95, rate(ryot_inference_latency_seconds_bucket[5m])) > 2.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Ryot P95 latency {{ $value }}s (target: <2.5s)"

      - alert: RyotGPUMemoryPressure
        expr: ryot_gpu_memory_percentage > 0.85
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "GPU memory utilization > 85%"

      - alert: RyotLowTokenThroughput
        expr: rate(ryot_tokens_generated_total[1m]) < 50
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Token generation rate dropped below 50 tps"

      - alert: RyotQueueBacklog
        expr: ryot_queue_depth > 100
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Request queue depth: {{ $value }} (SLA at risk)"
```

---

## Part 5: Example Prometheus Queries

### 5.1 SLO Monitoring

```promql
# P95 latency SLO (target: <500ms)
histogram_quantile(0.95, rate(ryot_inference_latency_seconds_bucket[5m]))

# P99 latency SLO (target: <2.5s)
histogram_quantile(0.99, rate(ryot_inference_latency_seconds_bucket[5m]))

# Success rate SLO (target: >99.9%)
(1 - (rate(ryot_inference_requests_total{status="error"}[5m]) /
      rate(ryot_inference_requests_total[5m]))) * 100

# Error rate per model
rate(ryot_inference_errors_total[5m]) by (model, error_type)
```

### 5.2 Capacity & Resource Monitoring

```promql
# Average token rate per model (last 5 minutes)
sum(rate(ryot_tokens_generated_total[5m])) by (model)

# GPU memory utilization across all models
avg(ryot_gpu_memory_percentage) by (model, device_id)

# Batch size distribution (what sizes are actually used?)
rate(ryot_batch_size_distribution_bucket[5m]) by (le)

# Queue depth trend (is backlog growing?)
rate(ryot_queue_depth[5m]) by (model)
```

### 5.3 Performance Analysis

```promql
# TTFT performance (first token latency)
histogram_quantile(0.95, rate(ryot_ttft_latency_seconds_bucket{model="bitnet-7b"}[5m]))

# Time to next token (generation speed after first token)
(histogram_quantile(0.95, rate(ryot_inference_latency_seconds_bucket[5m])) -
 histogram_quantile(0.95, rate(ryot_ttft_latency_seconds_bucket[5m])))

# Model loading time trends (detecting cache evictions)
rate(ryot_model_loading_time_seconds_bucket{load_type="cold_start"}[1h])

# KV cache hit rate (reuse efficiency)
ryot_kv_cache_hit_ratio by (context_type)
```

### 5.4 Cost & Billing Queries

```promql
# Total token cost estimate (last 24h)
sum(increase(ryot_token_cost_estimate_usd[24h])) by (token_type)

# Cost per model
sum(increase(ryot_token_cost_estimate_usd[24h])) by (model)

# Cost per request (for chargeback)
increase(ryot_token_cost_estimate_usd[24h]) / increase(ryot_inference_requests_total[24h])
```

### 5.5 Debugging Queries

```promql
# Which models have high error rates?
(rate(ryot_inference_errors_total[5m]) /
 rate(ryot_inference_requests_total[5m])) > 0.01

# Models exceeding latency SLA
histogram_quantile(0.95, rate(ryot_inference_latency_seconds_bucket[5m])) > 2.5

# GPU memory hot spots
topk(5, ryot_gpu_memory_percentage)

# Batch size analysis (are we using small batches inefficiently?)
avg(rate(ryot_batch_size_distribution_sum[5m]) /
    rate(ryot_batch_size_distribution_count[5m])) by (model)
```

---

## Part 6: Performance Overhead Analysis

### 6.1 Overhead Breakdown

```yaml
Metric Collection Overhead (per request):

Counter Increments:
  - ryot_inference_requests_total: ~50ns (atomic increment)
  - ryot_tokens_generated_total: ~50ns per token
  - ryot_inference_errors_total: ~50ns (if error)
  Total counter overhead: ~150ns = 0.00015ms

Histogram Observations:
  - ryot_inference_latency_seconds: ~1-2μs per observation
  - ryot_ttft_latency_seconds: ~1-2μs
  - ryot_batch_size_distribution: ~1-2μs
  Total histogram overhead: ~5μs = 0.005ms

Gauge Updates:
  - ryot_gpu_memory_usage_bytes: ~1-2μs (GPU query required)
  - ryot_active_inference_requests: ~50ns (atomic)
  Total gauge overhead: ~2-3μs = 0.002-0.003ms

Total Overhead per Request:
  ~8-10μs = 0.008-0.01ms

As Percentage of 100ms Inference:
  0.01ms / 100ms = 0.01% overhead ✓ (Well below 1% target)

As Percentage of 10ms TTFT:
  0.01ms / 10ms = 0.1% overhead ✓ (Acceptable)

Scrape Overhead:
  - /metrics endpoint exposed via prometheus_client
  - Scrape interval: 15s
  - Per-request impact: negligible
  - Prometheus memory: ~50-100MB for 40-50 metrics
```

### 6.2 Optimization Techniques

```yaml
1. Label Cardinality Control:
  - Pre-compute and cache label values
  - Avoid dynamic labels with unbounded values
  - Use categorization (batch_size_bucket vs raw batch_size)
  - Estimated savings: 80% reduction in cardinality

2. Batched Metric Updates:
  - Update histograms in batches rather than per-token
  - Trade: Accuracy vs CPU overhead
  - Example: Update per 100 tokens instead of per-token
  - Estimated savings: 50% reduction in histogram overhead

3. Conditional Metric Collection:
  - Skip low-importance metrics during high load
  - Use sampling for detailed metrics (e.g., 1:100 sampling)
  - Example: GPU memory only sampled every 10 requests
  - Estimated savings: 30% reduction in gauge overhead

4. Metric Aggregation:
  - Pre-compute aggregates instead of querying raw metrics
  - Use recording rules to cache computed metrics
  - Example: Cache TPS per model at 30s intervals
  - Estimated savings: 40% reduction in query load

Recommended Configuration:
  - Always collect: Request counters, error counters, TTFT histogram
  - Always collect: Token generation counters, GPU memory gauge
  - Sample (10%): Detailed batch size distribution
  - Sample (1/min): Model loading times
  - Aggregate: TPS, efficiency ratios (via recording rules)
```

---

## Part 7: Integration Points

### 7.1 Application Code Integration

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Initialize metrics
inference_requests = Counter(
    'ryot_inference_requests_total',
    'Total inference requests',
    ['model', 'status', 'endpoint']
)

inference_latency = Histogram(
    'ryot_inference_latency_seconds',
    'Inference latency',
    ['model', 'batch_size'],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
)

tokens_generated = Counter(
    'ryot_tokens_generated_total',
    'Tokens generated',
    ['model', 'token_type']
)

gpu_memory = Gauge(
    'ryot_gpu_memory_bytes',
    'GPU memory usage',
    ['model', 'memory_type']
)

# Usage in inference function
def inference_handler(prompt: str, model: str, batch_size: int) -> str:
    start_time = time.time()

    try:
        # Inference logic
        result = run_inference(prompt, model, batch_size)

        # Record metrics
        latency = time.time() - start_time
        inference_latency.labels(
            model=model,
            batch_size=f"batch_{batch_size}"
        ).observe(latency)

        inference_requests.labels(
            model=model,
            status="success",
            endpoint="generate"
        ).inc()

        # Token metrics
        tokens_generated.labels(
            model=model,
            token_type="completion"
        ).inc(len(result.tokens))

        return result

    except Exception as e:
        inference_requests.labels(
            model=model,
            status="error",
            endpoint="generate"
        ).inc()
        raise
```

### 7.2 Middleware Integration

```python
from fastapi import FastAPI, Request
from functools import wraps
import time

app = FastAPI()

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Automatic metrics collection for all endpoints"""
    start_time = time.time()

    response = await call_next(request)

    latency = time.time() - start_time

    # Record HTTP-level metrics
    if request.url.path.startswith("/v1/inference"):
        # Extract model from request body or params
        model = request.query_params.get("model", "unknown")
        batch_size = request.query_params.get("batch_size", "1")

        inference_latency.labels(
            model=model,
            batch_size=batch_size
        ).observe(latency)

        inference_requests.labels(
            model=model,
            status="success" if response.status_code < 400 else "error",
            endpoint="generate"
        ).inc()

    return response
```

### 7.3 GPU Monitoring Integration

```python
import pynvml

def update_gpu_metrics():
    """Update GPU-specific metrics (called periodically)"""
    pynvml.nvmlInit()

    for device_id in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

        # Memory stats
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory.labels(
            model="current",
            memory_type="used"
        ).set(mem_info.used)

        # Utilization
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_utilization.labels(
            device_id=str(device_id)
        ).set(util.gpu)

# Schedule this to run every 5 seconds
scheduler.add_job(update_gpu_metrics, 'interval', seconds=5)
```

---

## Part 8: Summary Reference Tables

### 8.1 Metrics Checklist

| Metric                          | Type      | Priority | Scrape |
| ------------------------------- | --------- | -------- | ------ |
| ryot_inference_requests_total   | Counter   | Critical | 15s    |
| ryot_inference_errors_total     | Counter   | Critical | 15s    |
| ryot_inference_latency_seconds  | Histogram | Critical | 15s    |
| ryot_ttft_latency_seconds       | Histogram | High     | 15s    |
| ryot_tokens_generated_total     | Counter   | High     | 15s    |
| ryot_gpu_memory_usage_bytes     | Gauge     | High     | 30s    |
| ryot_gpu_memory_percentage      | Gauge     | High     | 30s    |
| ryot_tokens_per_second          | Gauge     | High     | 15s    |
| ryot_batch_size_distribution    | Histogram | Medium   | 30s    |
| ryot_model_loading_time_seconds | Histogram | Medium   | 60s    |
| ryot_kv_cache_hit_ratio         | Gauge     | Medium   | 30s    |
| ryot_active_inference_requests  | Gauge     | Medium   | 15s    |
| ryot_queue_depth                | Gauge     | Medium   | 15s    |
| ryot_gpu_utilization_percent    | Gauge     | Medium   | 30s    |
| ryot_token_cost_estimate_usd    | Counter   | Low      | 60s    |

### 8.2 Latency Bucket Reference

```yaml
Histogram Buckets (in seconds):
  TTFT (First Token): [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
  Total Latency: [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
  Token Rate (tps): [10, 25, 50, 100, 150, 250, 400, 600]
  Batch Size (discrete): [1, 2-4, 5-8, 9-16, 17-32, 33+]
  Model Loading: [0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0]
```

### 8.3 SLO Targets

| Metric                  | Target    | P-ile | Window |
| ----------------------- | --------- | ----- | ------ |
| Inference Latency       | < 500ms   | P95   | 5m     |
| Inference Latency (P99) | < 2.5s    | P99   | 5m     |
| TTFT                    | < 50ms    | P95   | 5m     |
| Token Throughput        | > 100 tps | avg   | 1m     |
| Success Rate            | > 99.9%   | -     | 1h     |
| Error Rate              | < 0.1%    | -     | 1h     |
| GPU Memory Headroom     | > 15%     | avg   | 5m     |
| Queue Depth             | < 50      | max   | 5m     |

---

## Part 9: Implementation Roadmap

### Phase 1: Core Metrics (Week 1)

- [ ] Implement request counters
- [ ] Add latency histograms
- [ ] Token generation tracking
- [ ] Deploy to staging

### Phase 2: GPU Monitoring (Week 2)

- [ ] GPU memory gauges
- [ ] GPU utilization metrics
- [ ] KV cache tracking
- [ ] Performance baseline

### Phase 3: Advanced Metrics (Week 3)

- [ ] Cache effectiveness metrics
- [ ] Batch size distribution
- [ ] Cost estimation
- [ ] Query optimization

### Phase 4: Dashboards & Alerts (Week 4)

- [ ] Grafana dashboards
- [ ] Alert rules
- [ ] SLO definition
- [ ] Production rollout

---

## Part 10: Conclusion

This metrics strategy provides comprehensive observability for the Ryot LLM inference service with:

✓ **Minimal overhead** (<0.01% latency impact)
✓ **Production-grade buckets** tuned for LLM inference
✓ **Comprehensive coverage** from requests to GPU utilization
✓ **Clear integration points** for application code
✓ **SLO-aligned alerting** for operational reliability
✓ **Cost tracking** for business optimization

**Next Steps:**

1. Implement Phase 1 metrics
2. Establish baseline performance
3. Define SLOs with stakeholders
4. Deploy to production with gradual rollout
