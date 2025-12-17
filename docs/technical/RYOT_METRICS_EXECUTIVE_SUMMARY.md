# Ryot LLM Metrics: Executive Summary & Architecture

**Audience**: ML Leaders, Platform Engineers, Operations  
**Status**: Production Design  
**Date**: December 2025

---

## 📋 One-Page Summary

### The Metrics Strategy

The Ryot LLM inference service requires **15 core metrics** across 3 categories to provide complete observability:

| Category       | Metric Count | Purpose                 | Update Rate     |
| -------------- | ------------ | ----------------------- | --------------- |
| **Counters**   | 5            | Track cumulative events | Per request     |
| **Histograms** | 6            | Measure distributions   | Per observation |
| **Gauges**     | 8            | Monitor current state   | Every 15-30s    |

**Total Overhead**: ~0.01ms per inference (<0.01% impact)

### Key Design Principles

```
✓ Production-grade bucket ranges tuned for LLM inference
✓ Token-level tracking for cost & efficiency analysis
✓ GPU-aware metrics for resource optimization
✓ Cache-effectiveness monitoring for performance gains
✓ Sub-linear overhead via efficient label cardinality
```

---

## 🎯 Metrics Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RYOT METRICS HIERARCHY                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────┐ │
│  │   REQUEST LAYER    │  │   TOKEN LAYER      │  │ RESOURCE   │ │
│  ├────────────────────┤  ├────────────────────┤  │ LAYER      │ │
│  │ • Requests count   │  │ • Tokens generated │  ├────────────┤ │
│  │ • Error tracking   │  │ • Token rate (TPS) │  │ • GPU mem  │ │
│  │ • Latency dist.    │  │ • Cost estimate    │  │ • GPU util │ │
│  │ • TTFT tracking    │  │ • Efficiency ratio │  │ • Queue    │ │
│  └────────────────────┘  └────────────────────┘  └────────────┘ │
│         ↓                        ↓                      ↓         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │          PROMETHEUS STORAGE & AGGREGATION               │   │
│  │  • Scrape interval: 15s                                 │   │
│  │  • Retention: 2 weeks (raw), 1 year (aggregated)        │   │
│  │  • Recording rules: Pre-compute common queries          │   │
│  └──────────────────────────────────────────────────────────┘   │
│         ↓                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │      ALERTING & DASHBOARDS                              │   │
│  │  • Real-time alerts on SLO violations                   │   │
│  │  • Grafana dashboards for all user roles                │   │
│  │  • Custom queries for debugging & tuning                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔢 Metric Definitions Quick Reference

### Counters (Always Increasing)

```
ryot_inference_requests_total
├── Labels: model, status, endpoint
├── Example: model="bitnet-7b", status="success"
└── Use: Track request volume and error rates

ryot_tokens_generated_total
├── Labels: model, token_type, generation_mode
├── Example: token_type="completion"
└── Use: Billing, capacity planning, cost analysis

ryot_inference_errors_total
├── Labels: error_type, model, severity
├── Example: error_type="oom"
└── Use: Debugging, SRE alerting, trend analysis
```

### Histograms (Distributions with Buckets)

```
ryot_inference_latency_seconds
├── Buckets: [10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s, 30s]
├── Rationale: Exponential growth, LLM-optimized
└── Use: P50/P95/P99 latency analysis, SLO tracking

ryot_ttft_latency_seconds (Time-To-First-Token)
├── Buckets: [5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s]
├── Rationale: Prompt processing + first token generation
└── Use: UX performance, model loading optimization

ryot_model_loading_time_seconds
├── Buckets: [0.5s, 1s, 2.5s, 5s, 10s, 30s, 60s, 120s]
├── Rationale: Cold/warm/cache-hit characterization
└── Use: Cache strategy optimization, infrastructure sizing

ryot_batch_size_distribution
├── Buckets: [1, 2-4, 5-8, 9-16, 17-32, 33+]
├── Rationale: Discrete distribution of actual batch sizes
└── Use: Batching strategy effectiveness, GPU utilization
```

### Gauges (Current State)

```
ryot_active_inference_requests
├── Labels: model, request_type
├── Range: 0 to current capacity
└── Use: Load monitoring, autoscaling triggers

ryot_gpu_memory_percentage
├── Labels: model, device_id, batch_size
├── Range: 0-100%
├── Alert: > 85% = memory pressure
└── Use: Capacity planning, OOM prevention

ryot_queue_depth
├── Labels: model, priority_level
├── Range: 0 to queue max
├── Alert: > 100 = backlog risk
└── Use: Load detection, scaling decisions
```

---

## 📊 Latency Bucket Design Rationale

### Why These Buckets?

```
┌──────────────┬──────────────┬──────────────────────────────────────┐
│ Bucket (ms)  │ Representative │ What it Captures                  │
├──────────────┼──────────────┼──────────────────────────────────────┤
│ 10           │ KV cache hit │ Best case: prompt cached            │
│ 25           │ Warm cache   │ Model hot in memory                  │
│ 50           │ TTFT target  │ First-token latency SLO             │
│ 100          │ Warm start   │ Model loaded, prompt processing     │
│ 250          │ Normal gen   │ Mid-range generation latency        │
│ 500          │ Medium gen   │ Longer sequences                    │
│ 1000         │ Long gen     │ Multi-token generations             │
│ 2500         │ Very long    │ Batch operations                    │
│ 5000         │ Edge case    │ Cold loads or complex ops           │
│ 10000        │ Degraded     │ System under stress                 │
│ 30000        │ Critical     │ SLA violations likely               │
│ ∞            │ Timeout      │ Requests exceeding limits           │
└──────────────┴──────────────┴──────────────────────────────────────┘
```

### Token Rate Buckets

```
┌──────────┬────────────────────────────────────────────┐
│ Tps      │ Operational Interpretation                 │
├──────────┼────────────────────────────────────────────┤
│ 10       │ ⚠️  Very slow - investigate immediately   │
│ 25       │ ⚠️  Below baseline - check GPU            │
│ 50       │ ⚠️  Below target - scaling needed         │
│ 100      │ ✓ Good for CPU models                     │
│ 150      │ ✓ Good for single GPU                     │
│ 250      │ ✓ Excellent GPU throughput                │
│ 400      │ ✓ Very good multi-GPU                     │
│ 600      │ ✓ Peak performance achieved               │
│ >600     │ 🚀 Exceptional (rare, edge cases)         │
└──────────┴────────────────────────────────────────────┘
```

---

## 📈 Data Flow Architecture

```
┌─────────────────┐
│ Inference       │
│ Request Arrives │
└────────┬────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│ Metrics Collection                       │
│                                          │
│  ┌─────────────────────────────────────┐ │
│  │ 1. Record request start time        │ │
│  │ 2. Increment active_requests gauge  │ │
│  └─────────────────────────────────────┘ │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│ Model Inference Execution                │
│                                          │
│  ┌─────────────────────────────────────┐ │
│  │ 1. First token generated → record   │ │
│  │    TTFT                             │ │
│  │ 2. Each token generated → counter   │ │
│  │    increment                        │ │
│  └─────────────────────────────────────┘ │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│ Request Completion                       │
│                                          │
│  ┌─────────────────────────────────────┐ │
│  │ 1. Record total latency histogram   │ │
│  │ 2. Record token count               │ │
│  │ 3. Record status (success/error)    │ │
│  │ 4. Decrement active_requests        │ │
│  │ 5. Update error metrics (if needed) │ │
│  └─────────────────────────────────────┘ │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│ Prometheus Scrape (every 15s)            │
│                                          │
│  ┌─────────────────────────────────────┐ │
│  │ /metrics endpoint exposes all       │ │
│  │ counters, histograms, gauges        │ │
│  └─────────────────────────────────────┘ │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│ Time Series Storage                      │
│                                          │
│  ┌─────────────────────────────────────┐ │
│  │ Raw data: 2 weeks retention         │ │
│  │ Aggregates: 1 year retention        │ │
│  └─────────────────────────────────────┘ │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│ Recording Rules (30s intervals)          │
│                                          │
│  ┌─────────────────────────────────────┐ │
│  │ Pre-compute common queries:         │ │
│  │ • P95/P99 latencies                 │ │
│  │ • Error rates                       │ │
│  │ • Token rates                       │ │
│  │ • GPU efficiency                    │ │
│  └─────────────────────────────────────┘ │
└────────┬─────────────────────────────────┘
         │
         ├─────────────────────────┐
         │                         │
         ▼                         ▼
    ┌─────────┐           ┌──────────────┐
    │ Alerts  │           │ Dashboards   │
    │ (Real-  │           │ (Grafana)    │
    │ time)   │           │              │
    └─────────┘           └──────────────┘
```

---

## 🎬 Lifecycle: Request → Metrics

### Single Request Example

```
REQUEST: Generate 50 tokens from "bitnet-7b" with batch_size=8

Timeline:
┌─────────────────────────────────────────────────────────────┐
│ T=0ms      Record metrics entry point                       │
│            • active_requests.inc()                          │
│            • start_time = now()                             │
│                                                             │
│ T=5ms      First token generated                           │
│            • ttft_latency = 5ms → histogram.observe()       │
│            • tokens_count = 1                              │
│                                                             │
│ T=15ms     Tokens 2-50 generated (streamed)                │
│            • tokens_count += 49                            │
│            • Update GPU metrics                            │
│                                                             │
│ T=25ms     Request completed                               │
│            • total_latency = 25ms → latency_histogram      │
│            • batch_size=8 → batch_dist_histogram           │
│            • tokens_generated_total.inc(50)                │
│            • inference_requests_total.inc(status=success)  │
│            • active_requests.dec()                         │
│                                                             │
│ T=30s      Prometheus scrape (next cycle)                  │
│            • Fetch all metrics from /metrics               │
│            • Store in time series DB                       │
│                                                             │
│ T=60s      Recording rules execute                         │
│            • Compute P95 latency                           │
│            • Compute token rate                            │
│            • Update dashboards                             │
└─────────────────────────────────────────────────────────────┘

METRICS RECORDED FOR THIS REQUEST:

Counter increments:
  ✓ ryot_inference_requests_total[model=bitnet-7b, status=success] += 1
  ✓ ryot_tokens_generated_total[model=bitnet-7b, token_type=completion] += 50

Histogram observations:
  ✓ ryot_ttft_latency_seconds[model=bitnet-7b] → 5ms
  ✓ ryot_inference_latency_seconds[model=bitnet-7b, batch_size=5-8] → 25ms
  ✓ ryot_batch_size_distribution[model=bitnet-7b] → 8

Gauge updates:
  ✓ ryot_active_inference_requests[model=bitnet-7b] → decreased by 1
  ✓ GPU memory metrics updated (from GPU query)
  ✓ Queue depth potentially decreased
```

---

## 🎯 SLO Definition Example

### Latency SLO

```
Name: Ryot P95 Inference Latency
Definition: p95(ryot_inference_latency_seconds) < 500ms
Time Window: 5-minute rolling
Calculation Frequency: Every 30 seconds
Alert Threshold: 2+ violations in 10-minute window

Alert Rule:
  histogram_quantile(0.95, rate(ryot_inference_latency_seconds_bucket[5m])) > 0.5s
  for: 10m
  → Send alert to on-call team
```

### Success Rate SLO

```
Name: Ryot Inference Success Rate
Definition: success_rate > 99.9%
Formula: (requests_success / requests_total) > 0.999
Time Window: 1-hour rolling
Calculation Frequency: Every 5 minutes

Alert Rule:
  (rate(ryot_inference_requests_total{status="error"}[1h]) /
   rate(ryot_inference_requests_total[1h])) < 0.001
  → Healthy
```

### Throughput SLO

```
Name: Ryot Minimum Token Throughput
Definition: token_rate > 100 tps (per model)
Formula: rate(ryot_tokens_generated_total[1m]) > 100
Time Window: 1-minute rolling
Calculation Frequency: Every 15 seconds

Alert Rule:
  sum(rate(ryot_tokens_generated_total[1m])) by (model) < 100
  for: 10m
  → May indicate degradation or underload
```

---

## 📱 Label Cardinality Analysis

### Preventing Cardinality Explosion

```
DO: Pre-computed, bounded labels
  ✓ model: {bitnet-7b, llama-7b, llama-13b, gpt2-small}
           Cardinality: 4 (bounded)

  ✓ batch_size_bucket: {1, 2-4, 5-8, 9-16, 17-32, 33+}
           Cardinality: 6 (discrete)

  ✓ error_type: {oom, timeout, cuda_error, invalid_request}
           Cardinality: 4 (bounded)

DON'T: Unbounded labels
  ✗ request_id: {uuid-1, uuid-2, ...}
           Cardinality: Unbounded (explosion!)

  ✗ user_id: {user-1, user-2, ...}
           Cardinality: Unbounded (storage disaster!)

MAXIMUM CARDINALITY:
  ryot_inference_requests_total:
    = |models| × |statuses| × |endpoints|
    = 4 × 3 × 2
    = 24 combinations (safe)

  ryot_batch_size_distribution:
    = |models| × |request_types| × |batch_buckets|
    = 4 × 2 × 6
    = 48 combinations (safe)
```

---

## 🔐 Performance Overhead Accounting

### Where Does the 0.01ms Come From?

```
Per Inference Request:

Counter Updates (5 total):
  • ryot_inference_requests_total.inc()     → ~50ns
  • ryot_tokens_generated_total.inc(count)  → ~50ns per inc()
  • ryot_inference_errors_total.inc()       → ~50ns (if error)
  Subtotal: ~150ns

Histogram Updates (3 total):
  • ryot_inference_latency_seconds.observe()    → ~1-2μs
  • ryot_ttft_latency_seconds.observe()         → ~1-2μs
  • ryot_batch_size_distribution.observe()      → ~1-2μs
  Subtotal: ~5μs

Gauge Operations (2 total):
  • ryot_active_inference_requests.inc/dec()   → ~100ns
  • ryot_queue_depth operations                 → ~100ns
  Subtotal: ~200ns

Total per request: ~5.35μs = 0.00535ms

Amortized per token (50 tokens): 0.00011ms
As % of 100ms inference latency: 0.005% ✓✓✓
As % of 10ms TTFT: 0.05% ✓✓

GPU Memory Overhead:
  • Metrics storage: ~50-100MB in Prometheus
  • Application memory: <1MB per service instance
```

---

## 🚀 Implementation Phases

```
PHASE 1: Core Metrics (Week 1)
  Duration: 3-5 days
  Metrics: Request counters, latency histograms, token counts
  Deliverable: Basic observability in staging
  Risk: Low

  ├─ Day 1: Implement counter metrics
  ├─ Day 2: Implement latency histograms
  ├─ Day 3: Integrate with FastAPI middleware
  ├─ Day 4: Deploy to staging
  └─ Day 5: Basic alerting validation

PHASE 2: GPU Monitoring (Week 2)
  Duration: 3-5 days
  Metrics: GPU memory, GPU utilization, KV cache metrics
  Deliverable: Hardware-level observability
  Risk: Medium (GPU query overhead)

  ├─ Day 1: Implement GPU memory gauges
  ├─ Day 2: Integrate GPU metrics collection
  ├─ Day 3: Performance baseline testing
  ├─ Day 4: Deploy to staging
  └─ Day 5: Alert tuning

PHASE 3: Advanced Metrics (Week 3)
  Duration: 2-3 days
  Metrics: Cache effectiveness, cost estimation, batch analysis
  Deliverable: Optimization insights
  Risk: Low

  ├─ Day 1: Implement cache metrics
  ├─ Day 2: Cost tracking integration
  └─ Day 3: Deploy + validate

PHASE 4: Production Rollout (Week 4)
  Duration: 5 days
  Deliverable: Production observability
  Risk: Medium (gradual rollout mitigates)

  ├─ Day 1: Create Grafana dashboards
  ├─ Day 2: Define alert rules & SLOs
  ├─ Day 3: Runbook creation
  ├─ Day 4: Gradual rollout (10% → 50% → 100%)
  └─ Day 5: Monitoring & optimization
```

---

## ✅ Success Criteria

### Technical

- [ ] All 15 metrics implemented and validated
- [ ] Metrics overhead < 0.01% per request
- [ ] Prometheus retention working as designed
- [ ] Recording rules pre-computing correctly
- [ ] All alerts triggering appropriately

### Operational

- [ ] Dashboards displaying real-time data
- [ ] Alert notifications reaching on-call
- [ ] Runbooks available and tested
- [ ] Team trained on metric interpretation
- [ ] SLOs defined and monitored

### Business

- [ ] Cost tracking enabled and accurate
- [ ] Capacity planning insights generated
- [ ] Performance baselines established
- [ ] Optimization opportunities identified
- [ ] Billing/chargeback data available

---

## 📞 Questions & Support

**For Design Questions**: See [RYOT_METRICS_DESIGN.md](RYOT_METRICS_DESIGN.md)  
**For Implementation Details**: See [RYOT_METRICS_IMPLEMENTATION.md](RYOT_METRICS_IMPLEMENTATION.md)  
**For Operational Use**: See [RYOT_METRICS_QUICK_REFERENCE.md](RYOT_METRICS_QUICK_REFERENCE.md)
