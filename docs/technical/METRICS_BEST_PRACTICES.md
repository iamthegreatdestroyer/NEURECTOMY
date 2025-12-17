# Metrics Best Practices Guide

**Location:** `docs/technical/METRICS_BEST_PRACTICES.md`

Complete operational playbook for managing metrics systems in ΣVAULT and Elite Agents environments.

---

## Table of Contents

1. [Operational Fundamentals](#operational-fundamentals)
2. [Common Pitfalls](#common-pitfalls)
3. [Performance Optimization](#performance-optimization)
4. [Cost Management](#cost-management)
5. [Alerting Strategy](#alerting-strategy)
6. [Case Studies](#case-studies)
7. [Team Training](#team-training)
8. [Disaster Recovery](#disaster-recovery)

---

## Operational Fundamentals

### 24/7 Monitoring Framework

**Tier 1: Critical Metrics** (Checked every 30 seconds)

```yaml
- Agent collective health score
- Storage available capacity
- Error rate (threshold: > 5%)
- Task processing latency (p95 > 1s)
```

**Tier 2: Important Metrics** (Checked every 5 minutes)

```yaml
- Individual agent success rates
- Tier-level utilization
- Storage tier growth rate
- Financial cost trends
```

**Tier 3: Contextual Metrics** (Checked daily)

```yaml
- Capacity projection accuracy
- Cost per operation
- Agent specialization distribution
- Knowledge sharing effectiveness
```

### On-Call Runbook Integration

**Directory Structure:**

```
docs/runbooks/
├── alerts/
│   ├── agent-health-degraded.md
│   ├── storage-capacity-critical.md
│   ├── cost-spike-detected.md
│   └── latency-increased.md
├── procedures/
│   ├── scale-up-agents.md
│   ├── migrate-storage-tier.md
│   └── incident-response.md
└── quick-reference.md
```

**Runbook Template:**

```markdown
# Alert: High Error Rate

## Alert Fired When

- Error rate > 10% for 5 minutes
- Severity: critical
- Prometheus rule: `error_rate_high`

## Detection in Grafana

- Dashboard: "Elite Agents - Performance"
- Panel: "Error Rate Trend"
- Threshold: Red > 10%

## Diagnosis Steps

1. Check which agents are erroring
2. Review recent deployments
3. Check system resources
4. Review error logs

## Remediation

- Quick: Restart affected agent (5 min)
- Medium: Scale tier up (15 min)
- Long: Update agent code (1 hour)

## Escalation

- 5 min no improvement: Page backend team
- 15 min no improvement: Page platform team
```

### Alert Routing Strategy

```yaml
Low Severity (Slack):
  - Metrics not updated (5 min)
  - High memory usage (< 80%)
  - Unusual patterns detected

Medium Severity (PagerDuty):
  - Error rate > 5% for 10 min
  - Task latency p95 > 5s for 10 min
  - Storage utilization > 80% for 30 min

Critical Severity (Page + Call Bridge):
  - Error rate > 20%
  - Collective health < 50%
  - Storage capacity < 10%
  - All agents in tier down
```

---

## Common Pitfalls

### Pitfall 1: Unbounded Label Cardinality

**Problem:** Using high-cardinality labels (user_id, customer_id, transaction_id)

**Bad Example:**

```python
# DON'T DO THIS
@track_storage_operation(
    operation_type='store',
    user_id=request.user_id,  # ❌ HIGH CARDINALITY
    transaction_id=uuid.uuid4()  # ❌ INFINITE CARDINALITY
)
def store_data(data):
    pass
```

**Impact:**

- 1,000 users × 100 transactions = 100,000 series
- Memory: 100,000 × 1.5KB ≈ 150MB
- Query performance: Minutes to hours
- Storage growth: 500MB+/day

**Solution:**

```python
# DO THIS INSTEAD
@track_storage_operation(
    operation_type='store',
    storage_class='hot',  # ✅ LOW CARDINALITY
    cost_center='engineering'  # ✅ 5-10 values max
)
def store_data(data, user_id):  # user_id as parameter only
    pass

# Use recording rules for high-cardinality queries
# Rule: high_volume_users:sum
# Query: sum(requests_total{user_id="important_user"})
```

**Cardinality Budget:**

```yaml
Service A: 50 metrics × 4 labels = 200 series ✅
Service B: 100 metrics × 3 labels = 300 series ✅
Service C: 30 metrics × 8 labels = 240 series ⚠️

Total: 740 series (limit: 10,000) ✅
```

### Pitfall 2: Over-Instrumentation

**Problem:** Too many metrics collected

**Bad Example:**

```python
# ❌ DON'T: 50+ metrics per function
for metric in all_metrics:
    metric.observe(value)
    metric.track(timestamp)
    metric.log(data)
    metric.export(backend)
```

**Impact:**

- Query performance degrades
- Dashboard becomes unusable
- Alert fatigue increases
- Storage costs multiply

**Solution:** Follow 80/20 rule

```python
# ✅ DO: Core metrics only
# Measure: Latency, Throughput, Errors, Utilization
core_metrics = {
    'latency_seconds': 'Histogram',
    'throughput_ops': 'Counter',
    'errors_total': 'Counter',
    'utilization_ratio': 'Gauge'
}

# Optional: Add context with traces or logs, not metrics
```

**Metric Selection Criteria:**

1. **Is it actionable?** (Can we respond to this alert?)
2. **Is it dimensional?** (Can we break it down?)
3. **Is it frequently checked?** (Used in dashboards?)

### Pitfall 3: Wrong Histogram Buckets

**Problem:** Buckets don't match SLA

**Bad Example:**

```python
# ❌ DON'T: Buckets don't match SLA (target: p95 < 100ms)
latency_histogram = Histogram(
    'request_latency_seconds',
    buckets=[10, 20, 50, 100, 500, 1000]  # Too coarse
)
```

**Impact:**

- Can't calculate p95 accurately
- Histogram precision lost
- SLA calculations meaningless

**Solution:** Design buckets around SLA

```python
# ✅ DO: Buckets match SLA (target: p95 < 100ms)
latency_histogram = Histogram(
    'request_latency_seconds',
    buckets=[
        0.005,    # 5ms    - Fast baseline
        0.010,    # 10ms   - Good performance
        0.025,    # 25ms   - Acceptable
        0.050,    # 50ms   - Approaching warning
        0.075,    # 75ms   - Warning
        0.100,    # 100ms  - SLA threshold
        0.250,    # 250ms  - Degraded
        1.0       # 1s     - Poor
    ]
)
```

**Bucket Selection Algorithm:**

1. Identify SLA: "p95 < 100ms"
2. Add buckets around SLA: [80ms, 100ms, 120ms]
3. Add buckets below SLA: [5ms, 10ms, 25ms, 50ms]
4. Add buckets for degradation: [250ms, 500ms, 1s]
5. Result: 10-15 buckets optimal

### Pitfall 4: Noisy Alerts

**Problem:** Too many false positives (alert fatigue)

**Example:**

```yaml
# ❌ DON'T: Too sensitive
alert: HighErrorRate
  expr: error_rate > 1%
  for: 1m  # Alert immediately on any error

# ❌ DON'T: Not specific enough
alert: Performance
  expr: latency > 50ms
  for: 5m  # What does "performance" mean?
```

**Impact:**

- Team ignores 90% of alerts
- Real incidents missed
- On-call burnout
- Mean time to response (MTTR) increases

**Solution:** Context-aware, tuned thresholds

```yaml
# ✅ DO: Meaningful, specific alerts
alert: HighErrorRateCritical
  expr: error_rate > 10%  # 10x above acceptable
  for: 5m  # Allow brief spikes
  action: page_oncall  # Immediate action required

alert: HighErrorRateWarning
  expr: error_rate > 5%   # 5x above baseline
  for: 15m  # Give time to trend down
  action: slack_notify  # Informational

# AVOID: One generic "Performance" alert
# INSTEAD: Specific alerts per metric
```

**Alert Tuning Formula:**

```
Threshold = Baseline + (2 × Standard_Deviation)
Duration = Max(time_to_investigate, time_to_fix) / 2

Example:
- Error rate baseline: 0.5%
- Std dev: 0.3%
- Threshold = 0.5% + (2 × 0.3%) = 1.1% ≈ 1%
- Investigation time: 10 min
- Fix time: 20 min
- Duration: max(10, 20) / 2 = 10 min
- Alert: error_rate > 1% for 10m
```

### Pitfall 5: Missing Baseline Context

**Problem:** Alerts without historical context

**Bad Alert:**

```yaml
alert: StorageSpike
  expr: storage_bytes > 1e12  # 1TB
  # Problem: Is 1TB high?
```

**Solution:** Comparison with baseline

```yaml
# ✅ DO: Compare against history
alert: StorageGrowthAnomaly
  expr: storage_bytes > avg_over_time(storage_bytes[30d]) * 1.5
  # Triggers if 50% above 30-day average

# ✅ DO: Day-over-day comparison
alert: CostAnomaly
  expr: rate(cost[1d]) > rate(cost offset 7d[1d]) * 1.2
  # Triggers if today's cost 20% above week-ago cost
```

---

## Performance Optimization

### 1. Query Optimization

**Slow Query Profile:**

```promql
# ❌ SLOW: 5+ seconds
sum(
  {__name__=~".*_total|.*_seconds",
   job=~"api|storage|agents"}
)
```

**Optimized Query:**

```promql
# ✅ FAST: < 100ms
sum(sigmavault_storage_operations_total{job="storage"})
```

**Optimization Techniques:**

1. **Filter Early:**

   ```promql
   # ❌ Slow: aggregate first
   sum(rate(metric_total[5m])) and keep_common

   # ✅ Fast: filter first
   sum(rate(metric_total{job="api"}[5m]))
   ```

2. **Use Recording Rules:**

   ```yaml
   # Precompute expensive aggregations
   groups:
     - name: storage
       interval: 1m
       rules:
         - record: storage:utilization:per_class
           expr: |
             sum(sigmavault_storage_capacity_used_bytes) by (storage_class) /
             sum(sigmavault_storage_capacity_total_bytes) by (storage_class)
   ```

3. **Reduce Cardinality:**

   ```promql
   # ❌ Too many labels
   rate(requests_total[5m])  # potentially 100k series

   # ✅ Filter by label
   rate(requests_total{endpoint="/api/storage"}[5m])  # 100 series
   ```

### 2. Metric Downsampling

**Strategy:** Store high-resolution data short-term, low-resolution long-term

```yaml
# Retention Policy
1 minute resolution: 1 month
5 minute resolution: 1 year
1 hour resolution: 5 years

# Prometheus config
global:
  scrape_interval: 15s # High resolution

# Remote storage (optional)
remote_write:
  - url: "http://thanos-receive:19291/api/v1/receive"
    write_relabel_configs:
      # Keep fine resolution short-term
      - source_labels: [__name__]
        regex: .* # All metrics
        target_label: __tmp_cardinality
        replacement: high

      # Downsample for long-term
      - source_labels: [__name__]
        regex: .*_total|.*_seconds
        target_label: __tmp_downsample
        replacement: "5m"
```

### 3. Scrape Interval Tuning

**Analysis: When to adjust scrape interval**

```yaml
# ❌ 5 second interval: Over-collection
scrape_interval: 5s
# Impact: 12,960 samples/day per metric, huge storage

# ✅ 30 second interval: Standard production
scrape_interval: 30s
# 2,880 samples/day per metric, balanced

# ✅ 2 minute interval: Long-running services
scrape_interval: 2m
# 720 samples/day per metric, minimal storage
```

**Decision Tree:**

```
Does service change < 30 seconds?
  → Yes (real-time systems)  : 15s interval
  → No (stable services)     : 30s interval

Do you need sub-minute trends?
  → Yes (markets, trading)   : 10s interval
  → No (most cases)          : 30s+ interval
```

### 4. Alert Rule Optimization

**Avoid expensive operations:**

```yaml
# ❌ SLOW: Complex join
alert: Alert1
  expr: |
    (rate(requests_total[5m]) > 100) and
    (on(job) group_left rate(errors_total[5m]) > 10)

# ✅ FAST: Precomputed via recording rule
alert: Alert1
  expr: error_ratio > 0.1
  # Computed by recording rule every 1m
```

---

## Cost Management

### 1. Storage Cost Optimization

**Driver Analysis:**

```
Prometheus Storage Cost =
  (Metrics × Series × Samples/day × Cost/sample) +
  (Storage/GB × Cost/GB)

Example:
  100 metrics × 500 series × 6,000 samples/day
  × $0.000001/sample × 365 days
  = ~$1,100/year per service
```

**Cost Reduction Strategies:**

**Strategy 1: Reduce Retention**

```yaml
# Default: 15 days
# Optimized: 7 days
storage:
  tsdb:
    retention: 7d

# Savings: ~50% storage cost
```

**Strategy 2: Increase Scrape Interval**

```yaml
# Before: 15s interval
scrape_interval: 15s  # 5,760 samples/metric/day

# After: 30s interval
scrape_interval: 30s  # 2,880 samples/metric/day

# Savings: ~50% storage cost
```

**Strategy 3: Reduce Cardinality**

```python
# Before: 1,000 series per metric
metric = Counter('requests', labels=['user_id', 'endpoint'])
# Generates: 1M users × 1K endpoints = 1B series 😱

# After: 50 series per metric
metric = Counter('requests', labels=['endpoint'])
# Generates: 1K endpoints = 1K series ✅

# Savings: 99.9% cardinality reduction
```

**Strategy 4: Use Remote Storage**

```yaml
# Prometheus + Remote Storage (e.g., S3)
# Local: 1-month hot storage (expensive)
# Remote: Long-term cold storage (cheap)

remote_write:
  - url: "s3://metrics-archive/prometheus"
    queue_config:
      capacity: 100000
      # Older data compressed and moved to S3
```

### 2. Query Cost Analysis

**Track query performance:**

```prometheus
# Alert on expensive queries
alert: SlowPrometheusQuery
  expr: prometheus_rule_evaluation_duration_seconds > 5
  for: 10m
  annotations:
    message: "Query {{ $labels.rule_name }} taking {{ $value }}s"
```

**Expensive Query Patterns:**

```promql
# ❌ EXPENSIVE: No job filter
rate(requests_total[5m])  # Scans all services

# ✅ CHEAP: Filtered by job
rate(requests_total{job="api"}[5m])  # Scans one service

# Cost ratio: 50-100x difference
```

---

## Alerting Strategy

### Alert Threshold Formula

**SLA-Based Thresholds:**

```
For SLA Target: p95 < 100ms

Calculate baseline from production:
  - Measure p95 latency over 7 days
  - Example: Baseline = 80ms

Threshold = Baseline × 1.5 (50% headroom)
  - Threshold = 80ms × 1.5 = 120ms

Alert Rule:
  histogram_quantile(0.95, latency) > 120ms for 5m
```

**Error Rate Thresholds:**

```
For acceptable error rate: < 1%

Calculate from production:
  - Measure error rate over 7 days
  - Example: Average = 0.2%
  - Std dev = 0.1%

Threshold = Average + (3 × Std dev)
  - Threshold = 0.2% + (3 × 0.1%) = 0.5%

Alert Rule:
  error_rate > 0.5% for 5m
```

### Alert Routing Configuration

```yaml
# AlertManager routing tree
route:
  receiver: default
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h

  routes:
    # Critical - immediate page
    - match:
        severity: critical
      receiver: pagerduty
      group_wait: 0s
      repeat_interval: 1h

    # Warning - daily digest
    - match:
        severity: warning
      receiver: slack-warnings
      group_wait: 5m
      repeat_interval: 24h

    # Info - weekly digest
    - match:
        severity: info
      receiver: slack-info
      group_wait: 15m
      repeat_interval: 7d

receivers:
  - name: pagerduty
    pagerduty_configs:
      - routing_key: "xxxxx"

  - name: slack-warnings
    slack_configs:
      - channel: "#alerts-warnings"

  - name: slack-info
    slack_configs:
      - channel: "#alerts-info"
```

---

## Case Studies

### Case Study 1: Cardinality Explosion

**Scenario:** Metrics storage grew 100× unexpectedly

**Root Cause Analysis:**

```promql
# Find highest cardinality metrics
count by (__name__) ({__name__=~".+"}) > 1000

# Result:
# user_requests_total: 10,000 series (HIGH!)
# api_requests_total: 100 series (normal)
```

**Investigation:**

```python
# Found problematic code:
@app.route('/api/<path:path>')
def api_handler(path):
    # ❌ BUG: user_id label has 1M+ values
    metric = Counter('user_requests_total',
                     labels=['method', 'user_id', 'path'])
    metric.labels(method=request.method,
                  user_id=request.user_id,
                  path=path).inc()
```

**Solution Implemented:**

```python
# Before: 1M users × 1K paths = 1B series
# After: Remove user_id label

@app.route('/api/<path:path>')
def api_handler(path):
    # ✅ FIX: Only low-cardinality labels
    metric = Counter('api_requests_total',
                     labels=['method', 'endpoint'])
    metric.labels(method=request.method,
                  endpoint=path).inc()

# User-specific metrics use separate system:
# - Log-based analytics for user trends
# - Traces for individual requests
```

**Results:**

- Cardinality reduced: 1B → 100K series (99.99%)
- Storage reduced: 500GB → 50MB
- Query time reduced: 5min → 100ms
- Memory: 200GB → 20MB
- Cost savings: 90%

**Lessons Learned:**

1. Never use unbounded labels
2. Validate cardinality in code review
3. Set up cardinality alerts early
4. Use separate systems for high-cardinality data

---

### Case Study 2: Alert Fatigue

**Scenario:** Team ignoring 95% of alerts

**Problem Analysis:**

```yaml
# Existing alerts (50 total):
- HighMemory: Fires 200×/day (false positives)
- LowDiskSpace: Fires 500×/day (slow degradation)
- SlowQuery: Fires 1000×/day (normal variation)
- HighLatency: Fires 50×/day (real issues)
# Alert fatigue: Team disabled monitoring 😱
```

**Alert Audit Results:**

```
Alert name              | Daily Fires | Real Issues | Accuracy
HighMemory              | 200         | 2           | 1%
LowDiskSpace            | 500         | 1           | 0.2%
SlowQuery               | 1000        | 5           | 0.5%
HighLatency             | 50          | 45          | 90%
HighErrorRate           | 30          | 30          | 100%
```

**Solution Implemented:**

1. **Tuned thresholds using statistics:**

```yaml
alert: HighMemory
  # Before: > 80% (fires constantly on stable systems)
  # After: > 95% for 10m (real problem threshold)
  expr: memory_usage_ratio > 0.95 for 10m

alert: SlowQuery
  # Before: > 100ms (normal for some queries)
  # After: > 2s for 5m (significant degradation)
  expr: query_duration > 2s for 5m
```

2. **Removed non-actionable alerts:**

```yaml
# Deleted (no action possible):
- LowDiskSpace (ops team manages)
- HighMemory (often false positive)

# Kept (actionable):
- DiskQuotaExceeded (add storage now)
- MemoryLeak (restart service)
```

3. **Added context to remaining alerts:**

```yaml
alert: HighLatency
  annotations:
    summary: "Latency {{ $value }}s (SLA: 100ms)"
    runbook: "docs/runbooks/latency-investigation.md"
    dashboard: "http://grafana/d/performance"
```

**Results:**

- Alert count: 50 → 15 alerts
- Daily fires: 1,780 → 120
- Accuracy: 18% → 85%
- Team re-enabled monitoring ✅
- MTTR (Mean Time To Response): 1h → 15min

---

### Case Study 3: Cost Runaway

**Scenario:** Monthly storage cost increased 10× without warning

**Cost Analysis:**

```
Month 1: $500/month (baseline)
Month 2: $1,200/month (+140%)
Month 3: $5,100/month (+325%)
```

**Root Cause:**

```promql
# Query revealed:
# agents_task_metrics increasing 100×
# after new feature deployment

# Investigation found:
# - New feature creates 100 metrics per task
# - 100,000 tasks/day × 100 metrics = 10M time series
# - Old system: 1M series → new system: 10M series
```

**Solution:**

```python
# Before (100 metrics per task):
def record_task_completion(task):
    for i in range(100):
        metrics[i].observe(task.duration)

# After (5 core metrics only):
def record_task_completion(task):
    task_duration_seconds.observe(task.duration)
    task_queue_depth.set(queue.size())
    task_status.labels(status=task.status).inc()
    task_cost_usd.observe(task.cost)
    task_memory_mb.observe(task.memory)
```

**Cost Optimization Implemented:**

1. Reduced metrics: 100 → 5
2. Increased scrape interval: 10s → 30s
3. Reduced retention: 30d → 7d
4. Removed DEBUG level metrics

**Results:**

- Storage reduced: 10M → 200K series (98%)
- Monthly cost: $5,100 → $200 (96% reduction)
- Query performance: 10s → 100ms
- Team happiness: Restored ✅

---

### Case Study 4: Failed Collective Intelligence

**Scenario:** Elite Agents collective suddenly reported intelligence_score = 0

**Detection:**

```prometheus
alert: CollectiveIntelligenceFailure
  expr: agents_collective_intelligence_score < 0.1 for 5m
```

**Investigation Process:**

Step 1: Check individual agent metrics

```promql
# Are individual agents healthy?
agents_success_rate  # Result: All > 99% ✅

agents_error_rate    # Result: All < 1% ✅
```

Step 2: Check collaboration

```promql
# Are agents communicating?
rate(agents_collaboration_events[1m])  # Result: 0 events ❌
```

Step 3: Check knowledge sharing

```promql
# Is knowledge being shared?
agents_knowledge_sharing_events  # Result: Dropped to 0 ❌
```

Step 4: System analysis

```
Hypothesis: Metrics collection broken, not agents
Evidence:
- All agent metrics present and reasonable
- Collaboration metric = 0 (seems possible)
- Intelligence score = 0 (impossible if any collaboration)
```

**Root Cause Found:**

```python
# Bug in collective_intelligence calculation:
def calculate_intelligence():
    # ❌ BUG: Division by zero when no recent events
    recent_events = collaboration_events[-1h]
    if recent_events == 0:
        return 0  # ❌ Should return previous_score

    intelligence = sum_breakthroughs / recent_events
    return intelligence
```

**Solution Deployed:**

```python
# ✅ FIX: Handle zero events gracefully
def calculate_intelligence():
    recent_events = collaboration_events[-1h]
    if recent_events == 0:
        return previous_score  # ✅ Maintain previous value

    intelligence = sum_breakthroughs / recent_events
    return max(intelligence, previous_score * 0.99)  # ✅ Graceful degradation
```

**Prevention Measures:**

1. Add unit test for edge cases
2. Add alert for zero collaboration events
3. Add dashboard showing intelligence score components
4. Add runbook for this specific failure

---

## Team Training

### Metrics Literacy Program

**Module 1: Metrics Fundamentals (1 hour)**

Objectives:

- Understand metric types (Counter, Gauge, Histogram)
- Read basic Prometheus queries
- Interpret SLA targets

Activities:

1. Interactive: "Guess the metric type"
2. Lab: Write first Prometheus query
3. Case study: How metrics prevented outage

**Module 2: Dashboard Interpretation (1 hour)**

Objectives:

- Understand visualization types
- Identify anomalies in data
- Use drill-down navigation

Activities:

1. Live demo: "Read a dashboard like a detective"
2. Practice: Find problem in sample data
3. Lab: Create first dashboard panel

**Module 3: Alert Tuning (1 hour)**

Objectives:

- Understand alert lifecycle
- Calculate meaningful thresholds
- Write actionable alerts

Activities:

1. Theory: Threshold calculation formula
2. Practice: Tune alert from false positives
3. Lab: Create alert for your service

**Certification Tracks:**

**Track A: Operator (Platform Team)**

```
Module 1: Fundamentals ✓
Module 2: Dashboard Interpretation ✓
Module 3: Alert Tuning ✓
Module 4: Incident Response (extra)
Module 5: Capacity Planning (extra)
Certification: "Metrics Operator"
```

**Track B: Developer**

```
Module 1: Fundamentals ✓
Module 4: Instrumentation Best Practices
Module 5: Query Optimization
Certification: "Metrics Developer"
```

**Track C: Architect**

```
All modules +
- Metrics Strategy
- Cost Optimization
- Monitoring Architecture
Certification: "Metrics Architect"
```

### Onboarding Checklist

**New Team Member Onboarding:**

Week 1:

- [ ] Access to Grafana (read-only)
- [ ] Access to Prometheus
- [ ] Read: Quick-start guide
- [ ] Watch: 10-minute intro video
- [ ] Quiz: Basic metrics concepts

Week 2:

- [ ] Complete Module 1: Fundamentals
- [ ] Read: Integration guide
- [ ] Lab: Add metrics to test service
- [ ] Pair: Review metrics code

Week 3:

- [ ] Complete Module 2: Dashboard Interpretation
- [ ] Lab: Create first dashboard
- [ ] Read: Alert tuning best practices
- [ ] Practice: Analyze real dashboards

Week 4:

- [ ] Complete Module 3: Alert Tuning
- [ ] Lab: Create first alert
- [ ] On-call rotation (with mentor)
- [ ] Review: Ask final questions

---

## Disaster Recovery

### Data Loss Scenarios

**Scenario 1: Prometheus Disk Failure**

**Detection:**

```
Symptom: "error opening the tsdb: missing blocks directory"
```

**Recovery Steps:**

```bash
# Step 1: Check backup status
ls -la /backups/prometheus/

# Step 2: If recent backup available (< 24h old)
systemctl stop prometheus
rm -rf /var/lib/prometheus/tsdb
cp -r /backups/prometheus/tsdb-2025-12-16 /var/lib/prometheus/
systemctl start prometheus

# Step 3: If no backup available
# Start with empty database (will resume collection)
systemctl start prometheus

# Step 4: Restore from remote storage if configured
# (e.g., Thanos, VictoriaMetrics)
```

**Prevention:**

```yaml
# Setup automated backups
backup_config:
  schedule: "0 2 * * *" # 2 AM daily
  retention: 30d
  destination: "s3://prometheus-backups/"
```

### Metric Corruption

**Scenario: Metrics have impossible values**

**Detection:**

```promql
# Alert on anomalies
alert: ImpossibleMetricValue
  expr: |
    (request_latency_seconds > 1000000) or
    (storage_utilization_ratio > 1.0) or
    (success_rate > 1.0)
```

**Recovery:**

```bash
# Option 1: Query validation
# Add validation layer before metrics export
if not 0 <= value <= expected_max:
    log("Invalid metric", value)
    return  # Don't export

# Option 2: Prometheus querying
# Clean corrupted range from remote storage
# (requires Thanos or similar)
```

### Historical Data Reconstruction

**Scenario: Need data older than retention**

**If archived to long-term storage:**

```bash
# Query archived data from S3/GCS
thanos query --store=s3://archive/ --from=2025-01-01
```

**If no archive available:**

```bash
# Reconstruct from application logs
# Example: Generate metrics from access logs
cat /var/log/access.log | \
  awk '{print $10}' | \  # Extract response time
  awk '{sum+=$1; count++} END {print sum/count}' | \
  # Calculate average latency historically
```

---

## Key Takeaways

### The Metrics Manifesto

1. **Measure what matters**
   - Focus on business outcomes
   - Align with SLA targets
   - Remove noise

2. **Keep it simple**
   - Low cardinality only
   - Few metrics, well chosen
   - Clear naming

3. **Act on insights**
   - Alerts must be actionable
   - Dashboards must enable decisions
   - Queries must answer business questions

4. **Optimize continuously**
   - Review alert accuracy weekly
   - Audit cardinality monthly
   - Forecast costs quarterly

5. **Invest in team**
   - Training program for all levels
   - Clear runbooks for incidents
   - Blameless retrospectives

---

## Additional Resources

- [Prometheus Best Practices](https://prometheus.io/docs/practices/naming/)
- [Google SRE Book - Monitoring](https://sre.google/sre-book/monitoring-distributed-systems/)
- [Monitoring Distributed Systems (Brendan Gregg)](http://www.brendangregg.com/usemethod.html)

---

**Version:** 1.0  
**Last Updated:** December 16, 2025  
**Maintained By:** @SCRIBE | Platform Team
