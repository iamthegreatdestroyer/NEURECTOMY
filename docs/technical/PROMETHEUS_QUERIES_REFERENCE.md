# Prometheus Queries Reference

**Location:** `docs/technical/PROMETHEUS_QUERIES_REFERENCE.md`

## Overview

Complete reference guide for 50+ Prometheus queries across ΣVAULT storage and Elite Agents monitoring systems. Each query includes explanation, use case, expected values, and interpretation.

---

## ΣVAULT Storage Service Queries (25 Queries)

### Operation Metrics

#### Q1: Total Storage Operations Per Second

```promql
rate(sigmavault_storage_operations_total[1m])
```

**Use Case:** Monitor throughput of storage service  
**Expected Range:** 10-1000 ops/sec depending on load  
**Alert Threshold:** < 1 ops/sec (possible failure)

**Interpretation:**

- Baseline: Your normal throughput
- Spike: Sudden traffic increase
- Drop: Potential service issue

---

#### Q2: Storage Operations Success Rate

```promql
sum(rate(sigmavault_storage_operations_total{status="success"}[5m])) /
sum(rate(sigmavault_storage_operations_total[5m]))
```

**Use Case:** Monitor operational health  
**Expected Range:** 0.95-1.0 (95-100%)  
**Alert Threshold:** < 0.95 (more than 5% failures)

**Prometheus Formula:**

```promql
# Success rate percentage
(rate(sigmavault_storage_operations_total{status="success"}[5m]) /
 rate(sigmavault_storage_operations_total[5m])) * 100
```

---

#### Q3: Operation Rate by Type

```promql
rate(sigmavault_storage_operations_total[1m]) by (operation_type)
```

**Use Case:** Identify which operations dominate  
**Operations Types:**

- `store`: Data storage operations
- `retrieve`: Data retrieval operations
- `delete`: Data deletion operations
- `snapshot`: Snapshot creation operations

**Example Output:**

```
store:      50 ops/sec
retrieve:   100 ops/sec
delete:     5 ops/sec
snapshot:   0.5 ops/sec
```

---

#### Q4: Operation Error Rate

```promql
rate(sigmavault_storage_operations_total{status="error"}[5m]) /
rate(sigmavault_storage_operations_total[5m])
```

**Use Case:** Monitor failure rate  
**Expected Range:** 0.0-0.05 (0-5%)

---

#### Q5: Failed Operations by Error Type

```promql
rate(sigmavault_storage_operation_errors_total[5m]) by (error_type)
```

**Use Case:** Diagnose failure causes  
**Common Error Types:**

- `timeout`: Operation exceeded deadline
- `permission_denied`: Access control violation
- `insufficient_space`: Storage tier full
- `corruption`: Data integrity failure
- `network`: Network communication failure

---

### Performance Metrics

#### Q6: Operation Latency - P50 (Median)

```promql
histogram_quantile(0.5, rate(sigmavault_storage_operation_duration_seconds_bucket[5m]))
```

**Use Case:** Monitor median performance  
**Expected Range:** 10-100ms depending on operation  
**Interpretation:** Half of operations complete faster

---

#### Q7: Operation Latency - P95

```promql
histogram_quantile(0.95, rate(sigmavault_storage_operation_duration_seconds_bucket[5m]))
```

**Use Case:** Monitor tail latency (impactful to user experience)  
**Expected Range:** 50-500ms depending on operation  
**Alert Threshold:** > 1000ms for store operations

**Rule of Thumb:**

- P95 < 2 × P50: Normal distribution
- P95 > 10 × P50: High variability (investigate)

---

#### Q8: Operation Latency - P99

```promql
histogram_quantile(0.99, rate(sigmavault_storage_operation_duration_seconds_bucket[5m]))
```

**Use Case:** Monitor extreme tail latency  
**Expected Range:** 100-2000ms

---

#### Q9: Average Latency by Operation Type

```promql
rate(sigmavault_storage_operation_duration_seconds_sum[5m]) by (operation_type) /
rate(sigmavault_storage_operation_duration_seconds_count[5m]) by (operation_type)
```

**Use Case:** Compare performance across operation types  
**Typical Values:**

- Store (cold tier): 500-1000ms
- Retrieve (hot tier): 10-50ms
- Delete: 5-20ms
- Snapshot: 100-500ms

---

#### Q10: Throughput by Operation Type

```promql
rate(sigmavault_storage_operation_duration_seconds_count[1m]) by (operation_type)
```

**Use Case:** Monitor operation rate by type  
**Expected Range:** Varies by workload

---

### Capacity and Storage Metrics

#### Q11: Total Storage Usage

```promql
sum(sigmavault_storage_capacity_used_bytes) by (storage_class)
```

**Use Case:** Monitor storage utilization  
**Labels:** hot, warm, cold, archive  
**Expected Range:** < 80% of tier capacity

---

#### Q12: Storage Capacity - Free Space

```promql
sum(sigmavault_storage_capacity_total_bytes) by (storage_class) -
sum(sigmavault_storage_capacity_used_bytes) by (storage_class)
```

**Use Case:** Monitor available capacity  
**Alert Threshold:** < 20% free (immediate attention)

---

#### Q13: Storage Utilization Percentage

```promql
(sum(sigmavault_storage_capacity_used_bytes) by (storage_class) /
 sum(sigmavault_storage_capacity_total_bytes) by (storage_class)) * 100
```

**Use Case:** Monitor tier utilization  
**Expected Range:**

- Hot: 30-70%
- Warm: 50-80%
- Cold: 70-90%
- Archive: 80-95%

**Interpretation:**

- < 50%: Under-utilized (cost opportunity)
- 50-80%: Healthy utilization
- > 80%: High utilization (monitor closely)
- > 95%: Critical (immediate action)

---

#### Q14: Growth Rate - 24 Hour Projection

```promql
sum(sigmavault_storage_capacity_used_bytes) +
(increase(sigmavault_storage_capacity_used_bytes[1h]) * 24)
```

**Use Case:** Forecast when storage will be full  
**Alert Threshold:** > 90% in 24h projection

---

#### Q15: Object Count by Tier

```promql
sigmavault_storage_object_count by (storage_class)
```

**Use Case:** Monitor object distribution  
**Typical Distribution:**

- Hot: 10-20% of objects
- Warm: 20-30%
- Cold: 30-40%
- Archive: 20-40%

---

### Financial Metrics

#### Q16: Storage Cost by Tier

```promql
sum(sigmavault_storage_cost_usd{cost_type="storage"}) by (storage_class)
```

**Use Case:** Identify expensive storage tiers  
**Pricing Reference:**

- Hot: $0.023/GB/month
- Warm: $0.010/GB/month
- Cold: $0.004/GB/month
- Archive: $0.0013/GB/month

---

#### Q17: Total Daily Cost

```promql
sum(increase(sigmavault_storage_cost_usd[24h]))
```

**Use Case:** Monitor daily operational costs  
**Expected Range:** $100-10000/day depending on scale

---

#### Q18: Cost by Category

```promql
sum(increase(sigmavault_storage_cost_usd[24h])) by (cost_type)
```

**Cost Categories:**

- `storage`: Tier storage costs
- `operation`: Operation costs (store, retrieve, delete, snapshot)
- `transfer`: Data transfer costs (ingress, egress, cross-region)
- `encryption`: Encryption operation costs

---

#### Q19: Cost Trend - 7 Day

```promql
sum(increase(sigmavault_storage_cost_usd[7d])) by (cost_center)
```

**Use Case:** Identify cost trends by department  
**Interpretation:**

- Stable: Normal operations
- Increasing: Growing usage or inefficiency
- Decreasing: Lifecycle transitions (data aging)

---

#### Q20: Cost Per GB Stored

```promql
sum(sigmavault_storage_cost_usd{cost_type="storage"}) /
sum(sigmavault_storage_capacity_used_bytes) * 1e9
```

**Use Case:** Monitor unit cost efficiency  
**Expected Range:** $0.01-0.05/GB/month  
**Benchmark:** Compare against cloud provider rates

---

### Reliability Metrics

#### Q21: Data Integrity Checks - Pass Rate

```promql
sum(rate(sigmavault_data_integrity_checks_total{result="pass"}[24h])) /
sum(rate(sigmavault_data_integrity_checks_total[24h]))
```

**Use Case:** Monitor data health  
**Expected Range:** 0.999+ (99.9%+)  
**Alert Threshold:** < 0.99 (less than 99%)

---

#### Q22: Average Replication Lag

```promql
avg(sigmavault_replication_lag_seconds)
```

**Use Case:** Monitor cross-region replication health  
**Expected Range:**

- Same region: < 100ms
- Cross-region: < 1s
- Extreme geo: < 5s

---

#### Q23: Backup Success Rate

```promql
sum(rate(sigmavault_backup_operations_total{status="success"}[24h])) /
sum(rate(sigmavault_backup_operations_total[24h]))
```

**Use Case:** Monitor backup reliability  
**Expected Range:** 0.999+ (99.9%+)  
**Alert Threshold:** < 0.99 for 2 consecutive days

---

#### Q24: SLA Compliance - Availability

```promql
(86400 - sum(increase(sigmavault_downtime_seconds[24h]))) / 86400 * 100
```

**Use Case:** Monitor SLA achievement  
**Expected Range:**

- 99.9%: Excellent
- 99.0%: Good
- 95.0%: Fair
- < 95%: Poor

---

#### Q25: Encryption Key Rotation Status

```promql
increase(sigmavault_encryption_key_rotation_total[7d]) by (result)
```

**Use Case:** Monitor encryption hygiene  
**Expected:** At least 1 successful rotation per week  
**Alert:** 0 rotations in 7 days

---

## Elite Agents Collective Queries (25+ Queries)

### Individual Agent Queries

#### Q26: Agent Status Summary

```promql
agents_status by (agent_name, tier)
```

**Values:**

- 0: Healthy
- 1: Degraded
- 2: Failed

**Visualization:** Status color by agent

---

#### Q27: Agent Availability

```promql
avg(agents_availability_ratio) by (agent_name)
```

**Expected Range:** 0.95-1.0 (95-100%)  
**Alert Threshold:** < 0.99 (less than 99%)

---

#### Q28: Agent Success Rate

```promql
agents_success_rate by (agent_name)
```

**Expected Range:** 0.95-1.0  
**Tiers:**

- ≥ 0.99: Excellent
- 0.95-0.99: Good
- 0.90-0.95: Fair
- < 0.90: Poor

---

#### Q29: Agent Error Rate

```promql
agents_error_rate by (agent_name)
```

**Expected Range:** 0.0-0.05 (0-5%)

---

#### Q30: Agent Timeout Rate

```promql
agents_timeout_rate by (agent_name)
```

**Expected Range:** < 0.01 (< 1%)  
**Alert Threshold:** > 0.05 (> 5%)

---

#### Q31: Average Task Duration

```promql
rate(agents_task_duration_seconds_sum[5m]) by (agent_name) /
rate(agents_task_duration_seconds_count[5m]) by (agent_name)
```

**Baseline by Type:**

- Fast (Analysis): 0.5s
- Medium (Coding): 5s
- Slow (Research): 30s

---

#### Q32: P95 Task Latency

```promql
histogram_quantile(0.95, rate(agents_task_duration_seconds_bucket[5m])) by (agent_name)
```

**Use Case:** SLA setting (set timeout to P95 × 1.5)

---

#### Q33: Agent Utilization

```promql
agents_utilization_ratio by (agent_name)
```

**Interpretation:**

- 0.0-0.3: Under-utilized
- 0.3-0.7: Healthy
- 0.7-0.9: High utilization
- 0.9-1.0: At capacity

---

#### Q34: Task Queue Depth

```promql
agents_queue_length / agents_max_capacity by (agent_name)
```

**Interpretation:**

- < 1.0: No backlog
- 1.0-3.0: Moderate queue
- > 3.0: Severe bottleneck

---

#### Q35: Agent Task Rate

```promql
rate(agents_tasks_total[1m]) by (agent_name)
```

**Use Case:** Monitor per-agent throughput

---

### Collective Queries

#### Q36: Collective Health Percentage

```promql
(agents_collective_healthy / agents_collective_total) * 100
```

**Expected Range:** > 95%  
**Alert Threshold:** < 80%

---

#### Q37: Failed Agents Count

```promql
agents_collective_failed
```

**Alert Threshold:** > 2 agents

---

#### Q38: Total Active Tasks

```promql
agents_collective_active_tasks
```

**Use Case:** Monitor collective workload

---

#### Q39: Collective Throughput

```promql
rate(agents_tasks_completed[1m])
```

**Unit:** Tasks per second  
**Baseline:** 50-500 tasks/sec in production

---

#### Q40: Collective Success Rate

```promql
agents_collective_success_rate
```

**Expected:** > 0.95

---

#### Q41: Collective Error Rate

```promql
agents_collective_error_rate
```

**Expected:** < 0.05

---

#### Q42: Collective Intelligence Score

```promql
agents_collective_intelligence_score
```

**Interpretation:**

- < 0.5: Under-performing
- 0.5-0.7: Normal
- 0.7-0.9: High performance
- 0.9+: Exceptional

---

#### Q43: Breakthrough Discovery Rate

```promql
rate(agents_collective_breakthrough_count[24h])
```

**Unit:** Breakthroughs per day  
**Expected:** > 0 (at least weekly)

---

### Tier-Level Queries

#### Q44: Tier Health Scores

```promql
agents_tier_health_score by (tier)
```

**Tiers:**

- 1: Foundational (5 agents)
- 2: Specialists (12 agents)
- 3-4: Innovators (2 agents)
- 5: Domain (5 agents)
- 6: Emerging (5 agents)
- 7: Human-Centric (5 agents)
- 8: Enterprise (2 agents)

---

#### Q45: Tier Utilization

```promql
agents_tier_utilization_ratio by (tier)
```

**Use Case:** Load distribution across tiers

---

#### Q46: Tier Task Throughput

```promql
sum by(tier) (rate(agents_tier_task_count[1m]))
```

**Use Case:** Identify bottleneck tiers

---

#### Q47: Tier Error Rates

```promql
agents_tier_error_rate by (tier)
```

**Use Case:** Identify problematic tiers

---

### Collaboration Queries

#### Q48: Collaboration Event Rate

```promql
rate(agents_collaboration_events_total[1m])
```

**Use Case:** Monitor inter-agent interaction frequency

---

#### Q49: Most Collaborative Agent Pairs

```promql
topk(20, increase(agents_collaboration_events_total[24h]))
```

**Use Case:** Identify strong agent partnerships

---

#### Q50: Average Communication Latency

```promql
rate(agents_communication_latency_seconds_sum[5m]) /
rate(agents_communication_latency_seconds_count[5m])
```

**Expected Range:**

- Same tier: 1-10ms
- Adjacent: 10-50ms
- Distant: 50-500ms

---

#### Q51: Handoff Frequency by Source

```promql
sum by(source_agent) (rate(agents_handoff_total[1h]))
```

**Use Case:** Identify agents frequently delegating work

---

#### Q52: Knowledge Sharing Events

```promql
sum(rate(agents_knowledge_sharing_events[24h])) by (source_tier, target_tier)
```

**Use Case:** Monitor cross-tier knowledge flow

---

## Advanced Queries

### Rate of Change Analysis

#### Q53: Storage Growth Rate (1 day)

```promql
rate(increase(sigmavault_storage_capacity_used_bytes[1d])[7d:1d])
```

**Use Case:** Forecast storage depletion

---

#### Q54: Agent Reliability Trending

```promql
(
  agents_success_rate{agent_name="APEX"} -
  agents_success_rate{agent_name="APEX"} offset 24h
) / agents_success_rate{agent_name="APEX"} offset 24h
```

**Unit:** Percentage change in success rate

---

### Aggregation Queries

#### Q55: Top 5 Most Utilized Agents

```promql
topk(5, agents_utilization_ratio)
```

---

#### Q56: Top 5 Most Collaborative Agents

```promql
topk(5, sum by(initiator_agent) (rate(agents_collaboration_events_total[24h])))
```

---

#### Q57: Least Reliable Agents

```promql
bottomk(5, agents_success_rate)
```

---

### Cross-System Queries

#### Q58: Combined System Health

```promql
(
  (agents_collective_healthy / agents_collective_total) +
  (1 - (sum(sigmavault_storage_operation_errors_total) / sum(sigmavault_storage_operations_total)))
) / 2
```

**Use Case:** Single metric for overall health

---

#### Q59: Cost per Agent Task

```promql
sum(increase(sigmavault_storage_cost_usd[24h])) /
sum(increase(agents_tasks_completed[24h]))
```

**Use Case:** Operational cost per task

---

#### Q60: System Efficiency Score

```promql
(
  agents_collective_success_rate * 0.5 +
  (1 - agents_collective_error_rate) * 0.3 +
  agents_collective_intelligence_score * 0.2
)
```

**Use Case:** Composite efficiency metric

---

## Query Tips and Tricks

### Debugging Range Issues

```promql
# Check if data exists for time range
increase(metric_name[24h]) > 0

# Find when data started
rate(metric_name[1h]) > 0

# Get latest value
metric_name{}
```

### Comparing Baselines

```promql
# Current vs 24h ago
metric_name / metric_name offset 24h

# Week-over-week
metric_name - metric_name offset 7d

# Month-over-month
metric_name - metric_name offset 30d
```

### Anomaly Detection

```promql
# Deviation from rolling average
abs(metric_name - avg_over_time(metric_name[7d])) / avg_over_time(metric_name[7d]) > 0.5

# Current vs quantile
metric_name > histogram_quantile(0.95, metric_name)
```

### Creating Ratios

```promql
# Safely divide (avoid division by zero)
metric_numerator / metric_denominator

# Better:
(metric_numerator / metric_denominator) or vector(0)

# With label filtering:
sum without(label) (metric_numerator) /
sum without(label) (metric_denominator)
```

---

## Dashboard Panel Examples

### Query for Heatmap

```promql
sum(rate(agents_task_duration_seconds_bucket[5m])) by (le, agent_name)
```

### Query for Bar Chart

```promql
topk(10, sum by(tier) (rate(agents_tier_task_count[1h])))
```

### Query for Table

```promql
{__name__=~"agents_.*_ratio"}
```

### Query for Gauge

```promql
agents_collective_intelligence_score
```

---

## Common Query Patterns

### Pattern: Success vs Error Rate

```promql
# Success rate
sum(rate(metric_success[5m])) / sum(rate(metric_total[5m]))

# Error rate
sum(rate(metric_error[5m])) / sum(rate(metric_total[5m]))

# Inverse check
success_rate + error_rate = 1.0
```

### Pattern: Utilization

```promql
# Current usage / Capacity
current_usage / max_capacity

# Over time trend
current_usage / max_capacity offset 24h
```

### Pattern: Latency Percentiles

```promql
P50: histogram_quantile(0.50, ...)
P95: histogram_quantile(0.95, ...)
P99: histogram_quantile(0.99, ...)
```

### Pattern: Rate of Change

```promql
# Per second
rate(counter_metric[1m])

# Per hour
rate(counter_metric[1h])

# Over day
increase(counter_metric[24h])
```

---

## Performance Tips

1. **Use ranges carefully:**
   - 1m: Real-time (frequently updated)
   - 5m: Default (balance precision/performance)
   - 1h: Trends (historical analysis)

2. **Filter early:**
   - Filter by label before aggregation
   - Use `{label="value"}` not `label_re=".*pattern.*"`

3. **Avoid excessive aggregation:**
   - Don't aggregate before filtering
   - Use `without()` not `by()` when aggregating most labels

4. **Use appropriate functions:**
   - `rate()` for counters
   - `increase()` for total growth
   - `avg()`, `sum()` for gauges

---

**Version:** 2.0  
**Last Updated:** December 16, 2025  
**Maintained By:** @SCRIBE
