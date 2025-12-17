# Phase 18D: Adaptive Sampling Strategy

**Document Version:** 1.0  
**Effective Date:** 2024-2025  
**Last Updated:** Phase 18D

---

## 🎯 Sampling Philosophy

Sampling balances observability with cost. We use **adaptive probabilistic sampling** that:

1. **Baseline**: Low probability (10%) for normal operations
2. **Adaptive**: 100% for errors, high latency, and business-critical operations
3. **Per-Service**: Tuned based on volume and criticality
4. **Per-Operation**: Special handling for specific operations

### Cost Model

```
Annual Cost = (Traces/Day × 365 × Cost/Trace)

Baseline (10% sampling):
  • 1M spans/day → 100k sampled spans/day
  • Cost: $100k/year @ $0.27/span

Adaptive (with intelligent filtering):
  • Same 1M spans/day but 98% accuracy maintained
  • Cost: $50k-70k/year (30-50% savings)
```

---

## 📊 Service-Level Sampling

### RYOT (Python LLM Service)

**Volume Estimate:** 50,000 requests/day  
**Critical Operations:** LLM inference, prompt caching  
**Cost Priority:** Medium

```yaml
service: ryot
default_strategy:
  type: probabilistic
  param: 0.2 # 20% baseline

operation_strategies:
  - operation: llm.inference
    type: probabilistic
    param: 0.5 # 50% for inference
    reason: "High latency variability, user-visible impact"

  - operation: llm.inference.error
    type: probabilistic
    param: 1.0 # 100% for errors
    reason: "Every error is critical"

  - operation: prompt.process
    type: probabilistic
    param: 0.1 # 10% for processing
    reason: "High volume, low variability"

  - operation: cache.hit
    type: probabilistic
    param: 0.05 # 5% for cache hits
    reason: "High volume, predictable latency"

  - operation: cache.miss
    type: probabilistic
    param: 1.0 # 100% for cache misses
    reason: "Indicates cache efficiency issues"

  - operation: vector.search
    type: probabilistic
    param: 0.3 # 30% for vector operations
    reason: "Variable latency, important for performance"

latency_sampling:
  # Increase sampling for slow traces
  - latency_threshold: 1000ms # >1 second
    param: 1.0 # 100%
  - latency_threshold: 500ms # >500ms
    param: 0.8 # 80%
  - latency_threshold: 100ms # >100ms
    param: 0.5 # 50%

tag_sampling:
  # Always sample certain conditions
  - tag: error=true
    param: 1.0
  - tag: cache.hit=false
    param: 0.5
  - tag: user.premium=true
    param: 0.3
```

**Expected Daily Traces:**

- Normal operations: 50,000 × 0.2 = 10,000 traces
- Errors: 500 × 1.0 = 500 traces
- **Total: ~10,500 traces/day**

---

### ΣLANG (Rust Language Service)

**Volume Estimate:** 20,000 compilations/day  
**Critical Operations:** Parse, type check, code generation  
**Cost Priority:** High

```yaml
service: sigmalang
default_strategy:
  type: probabilistic
  param: 0.15 # 15% baseline

operation_strategies:
  - operation: sigma.parse
    type: probabilistic
    param: 0.1 # 10% for parsing
    reason: "High volume, deterministic latency"

  - operation: sigma.parse.error
    type: probabilistic
    param: 1.0 # 100% for parse errors
    reason: "Syntax errors are critical for debugging"

  - operation: sigma.typecheck
    type: probabilistic
    param: 0.2 # 20% for type checking
    reason: "Medium volume, variable latency"

  - operation: sigma.typecheck.error
    type: probabilistic
    param: 1.0 # 100% for type errors
    reason: "Type errors indicate logical issues"

  - operation: sigma.codegen
    type: probabilistic
    param: 0.3 # 30% for code generation
    reason: "Smaller subset, important for performance"

  - operation: sigma.wasm_compile
    type: probabilistic
    param: 0.5 # 50% for WASM compilation
    reason: "Often slow, important to trace"

  - operation: sigma.vm_exec
    type: probabilistic
    param: 0.2 # 20% for VM execution
    reason: "High volume, usually fast"

latency_sampling:
  - latency_threshold: 5000ms
    param: 1.0 # >5 seconds always
  - latency_threshold: 2000ms
    param: 0.9 # >2 seconds, 90%
  - latency_threshold: 500ms
    param: 0.5 # >500ms, 50%

error_sampling:
  - error.type: "SyntaxError"
    param: 1.0
  - error.type: "TypeError"
    param: 1.0
  - error.type: "RuntimeError"
    param: 0.8
```

**Expected Daily Traces:**

- Normal compilations: 20,000 × 0.15 = 3,000 traces
- Errors: 200 × 1.0 = 200 traces
- High-latency: 100 × 0.9 = 90 traces
- **Total: ~3,290 traces/day**

---

### ΣVAULT (Rust Storage Service)

**Volume Estimate:** 100,000 operations/day  
**Critical Operations:** Store, retrieve, encryption  
**Cost Priority:** Medium-High

```yaml
service: sigmavault
default_strategy:
  type: probabilistic
  param: 0.25 # 25% baseline

operation_strategies:
  - operation: storage.store
    type: probabilistic
    param: 0.5 # 50% for store operations
    reason: "User-visible, variable latency"

  - operation: storage.retrieve
    type: probabilistic
    param: 0.3 # 30% for retrieve operations
    reason: "Common operation, but high volume"

  - operation: storage.delete
    type: probabilistic
    param: 0.1 # 10% for delete operations
    reason: "Low volume, predictable"

  - operation: encryption.encrypt
    type: probabilistic
    param: 1.0 # 100% for encryption
    reason: "Critical for security, variable latency"

  - operation: encryption.decrypt
    type: probabilistic
    param: 0.5 # 50% for decryption
    reason: "Important for performance analysis"

  - operation: storage.migrate
    type: probabilistic
    param: 1.0 # 100% for migrations
    reason: "Long-running, must be monitored"

  - operation: storage.snapshot
    type: probabilistic
    param: 1.0 # 100% for snapshots
    reason: "Critical backup operation"

latency_sampling:
  - latency_threshold: 10000ms
    param: 1.0 # >10 seconds always
  - latency_threshold: 1000ms
    param: 0.9 # >1 second, 90%
  - latency_threshold: 500ms
    param: 0.7 # >500ms, 70%
  - latency_threshold: 100ms
    param: 0.4 # >100ms, 40%

tier_sampling:
  # Higher sampling for more important tiers
  - storage.tier: hot
    param: 0.4
  - storage.tier: warm
    param: 0.3
  - storage.tier: cold
    param: 0.2
  - storage.tier: archive
    param: 0.1

failure_sampling:
  - status: error
    param: 1.0
  - status: timeout
    param: 1.0
```

**Expected Daily Traces:**

- Normal operations: 100,000 × 0.25 = 25,000 traces
- Encryption: 50,000 × 1.0 = 50,000 traces
- Errors: 500 × 1.0 = 500 traces
- **Total: ~75,500 traces/day** (higher due to encryption criticality)

---

### Agents (Multi-Service)

**Volume Estimate:** 10,000 tasks/day  
**Critical Operations:** Memory retrieval, task execution, collaboration  
**Cost Priority:** Low

```yaml
service: agents
default_strategy:
  type: probabilistic
  param: 0.1 # 10% baseline

operation_strategies:
  - operation: agent.execute
    type: probabilistic
    param: 0.2 # 20% for execution
    reason: "Important for understanding agent behavior"

  - operation: memory.retrieve
    type: probabilistic
    param: 0.5 # 50% for memory ops
    reason: "Critical path for agent performance"

  - operation: memory.store
    type: probabilistic
    param: 0.3 # 30% for storage
    reason: "Less critical than retrieval"

  - operation: agent.collaborate
    type: probabilistic
    param: 1.0 # 100% for collaboration
    reason: "Rare operations, must trace all"

  - operation: agent.error
    type: probabilistic
    param: 1.0 # 100% for errors
    reason: "Every failure is important"

latency_sampling:
  - latency_threshold: 60000ms
    param: 1.0 # >1 minute always
  - latency_threshold: 10000ms
    param: 0.8 # >10 seconds, 80%

complexity_sampling:
  # Higher sampling for complex tasks
  - task.complexity: high
    param: 0.5
  - task.complexity: medium
    param: 0.2
  - task.complexity: low
    param: 0.05
```

**Expected Daily Traces:**

- Normal operations: 10,000 × 0.1 = 1,000 traces
- Memory operations: 5,000 × 0.35 = 1,750 traces
- Collaboration: 50 × 1.0 = 50 traces
- **Total: ~2,800 traces/day**

---

## 🔄 Dynamic Sampling Rules

### Rule 1: Error Amplification

When error rate exceeds threshold, increase sampling for that operation:

```
IF error_rate[operation] > 5% THEN
  sampling_rate[operation] = min(1.0, sampling_rate[operation] × 2)
END
```

**Triggers:**

- RYOT LLM inference error rate > 2%
- ΣLANG compilation error rate > 5%
- ΣVAULT storage error rate > 1%
- Agent execution error rate > 3%

---

### Rule 2: Latency Amplification

When latency exceeds SLO, increase sampling:

```
IF p95_latency[operation] > SLO[operation] THEN
  sampling_rate[operation] = min(1.0, sampling_rate[operation] × 1.5)
END
```

**SLO Thresholds:**

- RYOT LLM inference: 2000ms
- ΣLANG compilation: 5000ms
- ΣVAULT storage: 1000ms
- Agent execution: 60000ms

---

### Rule 3: Load-Based Sampling

Scale sampling inversely with load to maintain constant trace volume:

```
target_traces_per_sec = 100  # Budget
current_requests_per_sec = get_request_rate()
dynamic_sampling_rate = target_traces_per_sec / current_requests_per_sec
```

**Benefits:**

- Constant infrastructure costs regardless of load
- More traces when system is underutilized
- Fewer traces when system is at capacity

---

### Rule 4: User-Tier Sampling

Sample more for premium/enterprise users:

```yaml
user_tier_sampling:
  - tier: enterprise
    param: 0.8
  - tier: premium
    param: 0.5
  - tier: standard
    param: 0.15
  - tier: free
    param: 0.05
```

---

## 📈 Sampling Configuration File

**File:** `deploy/monitoring/sampling-config.yaml`

```yaml
# Global sampling settings
global:
  initial_sampling_rate: 0.001
  max_sampling_rate: 1.0
  min_sampling_rate: 0.001

  # Dynamic adjustment
  enable_dynamic_sampling: true
  adjustment_interval: 300s # Every 5 minutes

  # Cost budgeting
  max_traces_per_day: 200000
  cost_per_span: 0.27
  max_daily_cost: 54000 # 200k spans

# Service configurations
services:
  ryot:
    enabled: true
    base_sampling: 0.2
    rules:
      - name: "error_amplification"
        condition: "error_rate > 0.02"
        action: "increase_sampling"
        factor: 2.0
      - name: "latency_amplification"
        condition: "p95_latency > 2000"
        action: "increase_sampling"
        factor: 1.5
      - name: "high_volume_reduction"
        condition: "request_rate > 1000/sec"
        action: "decrease_sampling"
        factor: 0.5

  sigmalang:
    enabled: true
    base_sampling: 0.15
    rules:
      - name: "compilation_error_trace_all"
        condition: "error_type == 'CompilationError'"
        action: "set_sampling"
        factor: 1.0

  sigmavault:
    enabled: true
    base_sampling: 0.25
    rules:
      - name: "encryption_always"
        condition: "operation == 'encrypt'"
        action: "set_sampling"
        factor: 1.0

  agents:
    enabled: true
    base_sampling: 0.1
    rules:
      - name: "collaboration_all"
        condition: "operation == 'collaborate'"
        action: "set_sampling"
        factor: 1.0

# Trace retention policies
retention:
  hot_storage_days: 3 # In Jaeger
  warm_storage_days: 30 # In S3/cold storage
  archive_storage_days: 365 # In long-term archive

# Exporter configuration
exporters:
  jaeger:
    endpoint: "http://jaeger-collector:14250"
    max_batch_size: 512
    batch_timeout: 5s
    queue_size: 2048

  otlp:
    endpoint: "http://otlp-collector:4317"
    enabled: false
```

---

## 🎛️ Implementation Checklist

- [ ] Deploy sampling strategies.json in Jaeger ConfigMap
- [ ] Configure per-service sampling rates
- [ ] Set up dynamic sampling adjustment rules
- [ ] Implement cost tracking and budgeting
- [ ] Create sampling dashboard in Grafana
- [ ] Set up alerting for sampling anomalies
- [ ] Document operation-level SLOs
- [ ] Test sampling under high load
- [ ] Validate trace coverage for errors
- [ ] Monitor sampling effectiveness

---

## 📊 Monitoring the Sampling System

### Key Metrics

```promql
# Sampling rate per service
jaeger_sampling_rate{service="ryot"}
jaeger_sampling_rate{service="sigmalang"}
jaeger_sampling_rate{service="sigmavault"}
jaeger_sampling_rate{service="agents"}

# Effective sampling efficiency
traces_sampled_total / spans_total

# Cost projection
spans_per_day * cost_per_span

# Error capture rate
error_traces_captured / total_errors
```

### Alerting Rules

```yaml
- alert: LowSamplingRate
  expr: jaeger_sampling_rate < 0.05
  annotations:
    summary: "Sampling rate too low: {{ $value }}"

- alert: HighErrorRate
  expr: error_rate > 0.05
  annotations:
    summary: "Error rate spike detected"

- alert: SamplingCostOverrun
  expr: projected_daily_cost > max_daily_cost
  annotations:
    summary: "Sampling cost projection exceeds budget"
```

---

## 🎯 Success Metrics

| Metric                    | Target | Rationale                |
| ------------------------- | ------ | ------------------------ |
| Error Trace Capture       | 100%   | Never miss errors        |
| P95 Latency Trace Capture | 95%+   | Catch performance issues |
| Daily Cost                | < $200 | Budget constraint        |
| Trace Availability        | 99.9%  | Reliability SLO          |
| Query Latency             | < 1s   | User experience          |
| Storage Utilization       | 80-90% | Efficiency               |
