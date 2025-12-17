# Elite Agent Collective - Comprehensive Metrics Design

**Document**: ELITE-AGENTS-METRICS-DESIGN.md  
**Status**: Production Architecture  
**Version**: 1.0  
**Date**: December 2025  
**Orchestrator**: @OMNISCIENT (Meta-Learning Coordinator)

---

## Executive Summary

This document defines production-grade observability architecture for the **Elite Agent Collective** — a sophisticated system of 40 specialized AI agents organized across 8 tiers with distinct capabilities and collaboration patterns.

**Key Metrics Dimensions:**

1. **Agent Health** — Status, availability, uptime, recovery events
2. **Task Management** — Assignment, completion, failures, duration
3. **Utilization** — Capacity, active tasks, queue length, idle time
4. **Performance** — Success rate, error rate, timeouts, retries
5. **Collective** — Aggregate metrics, throughput, coordination
6. **Collaboration** — Handoffs, specialization overlap, load balance
7. **Tier-Based** — Per-tier aggregations and performance
8. **Meta-Intelligence** — Learning, breakthroughs, knowledge sharing, memory fitness

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│              Elite Agent Collective (40 Agents)               │
│                         8 Tiers                               │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│              Metrics Collection Layer                          │
│  (neurectomy/agents/monitoring/metrics.py)                    │
│  ├─ 65+ Prometheus metrics                                    │
│  ├─ Label strategies for coordination                         │
│  ├─ Sub-linear aggregation patterns                          │
│  └─ Per-tier and per-specialization tracking                 │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│              Prometheus Time-Series Database                   │
│  (Port 9090, scrape interval: 15s)                            │
│  ├─ Raw metric storage                                        │
│  ├─ 15-day retention (configurable)                          │
│  ├─ 2-week cardinality management                            │
│  └─ Query language (PromQL)                                   │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│              Grafana Dashboards                                │
│  ├─ Individual agent dashboards                               │
│  ├─ Collective health dashboard                               │
│  ├─ Tier performance views                                    │
│  ├─ Specialization analysis                                   │
│  ├─ Collaboration network                                     │
│  └─ Meta-intelligence tracking                                │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│              AlertManager                                      │
│  (Port 9093)                                                   │
│  ├─ Agent failure alerts (CRITICAL)                           │
│  ├─ Degradation alerts (WARNING)                              │
│  ├─ Performance alerts (WARNING/CRITICAL)                     │
│  ├─ Utilization alerts (WARNING)                              │
│  └─ Meta-intelligence alerts (INFO/WARNING)                   │
└────────────────────────────────────────────────────────────────┘
```

---

## Part 1: Complete Metrics Definitions

### 1.1 Agent Health & Status Metrics (5 metrics)

#### elite_agent_status (Gauge)

- **Type**: Gauge (0, 1, or 2)
- **Purpose**: Operational status of each agent
- **Status Codes**:
  - `0` = HEALTHY (normal operation)
  - `1` = DEGRADED (reduced performance, partial availability)
  - `2` = FAILED (not responding)
- **Labels**:
  - `agent_id` — Unique agent identifier (e.g., "apex", "cipher")
  - `agent_name` — Display name (e.g., "APEX", "CIPHER")
  - `tier` — Tier number (1-8)
  - `specialization` — Agent specialization (e.g., "cs_engineering")
- **Update Frequency**: Real-time (heartbeat-driven)
- **Scrape Source**: Agent supervisor

**Example Query**:

```promql
# Status distribution
count by (status) (elite_agent_status)

# Failed agents
count(elite_agent_status == 2)

# Show status timeline
elite_agent_status{agent_id="apex"}
```

---

#### elite_agent_availability_percent (Gauge)

- **Type**: Gauge (0-100)
- **Purpose**: Availability percentage for agent
- **Calculation**: (uptime / total_time) × 100
- **Time Window**: Typically 24-hour rolling window
- **Labels**:
  - `agent_id`, `agent_name`, `tier`
- **Update Frequency**: Every 5 minutes

**SLO Integration**:

- Target: ≥ 99.5% availability
- Alert Threshold: < 80%

**Example Query**:

```promql
# Agents below SLO
elite_agent_availability_percent < 99.5

# Availability by tier
avg by (tier) (elite_agent_availability_percent)
```

---

#### elite_agent_recovery_events (Counter)

- **Type**: Counter (monotonically increasing)
- **Purpose**: Track recovery/restart events
- **Increments**: +1 when agent recovers from failed state
- **Labels**:
  - `agent_id`, `agent_name`, `tier`
- **Retention**: Lifetime of agent

**Example Query**:

```promql
# Recovery events in past 24h
increase(elite_agent_recovery_events[24h])

# Frequent restarts (more than 3 in 24h)
increase(elite_agent_recovery_events[24h]) > 3
```

---

#### elite_agent_uptime_seconds (Gauge)

- **Type**: Gauge (seconds)
- **Purpose**: Total uptime since last failure
- **Update**: Continuous or per-heartbeat
- **Labels**:
  - `agent_id`, `agent_name`, `tier`

**Example Query**:

```promql
# Average uptime by tier
avg by (tier) (elite_agent_uptime_seconds)

# Agents with low uptime
elite_agent_uptime_seconds < 3600  # Less than 1 hour
```

---

### 1.2 Task Management Metrics (4 metrics)

#### elite_tasks_assigned_total (Counter)

- **Type**: Counter
- **Purpose**: Total tasks assigned to agent (lifetime)
- **Labels**:
  - `agent_id`, `agent_name`, `tier`
  - `task_type` — Category of task (e.g., "design", "security", "optimization")
- **Update**: +1 for each task assignment

**Example Query**:

```promql
# Task volume by agent
rate(elite_tasks_assigned_total[5m])

# Total tasks per tier
sum by (tier) (increase(elite_tasks_assigned_total[24h]))

# Task distribution across agent
sum by (agent_id, task_type) (increase(elite_tasks_assigned_total[24h]))
```

---

#### elite_tasks_completed_total (Counter)

- **Type**: Counter
- **Purpose**: Successfully completed tasks
- **Labels**:
  - `agent_id`, `agent_name`, `tier`, `task_type`
- **Update**: +1 for each successful completion

**Success Rate Calculation**:

```
success_rate = tasks_completed_total / tasks_assigned_total × 100
```

**Example Query**:

```promql
# Completion rate (tasks/sec)
rate(elite_tasks_completed_total[5m])

# Success rate per agent
(rate(elite_tasks_completed_total[5m]) /
 rate(elite_tasks_assigned_total[5m])) × 100
```

---

#### elite_tasks_failed_total (Counter)

- **Type**: Counter
- **Purpose**: Failed tasks with root cause analysis
- **Labels**:
  - `agent_id`, `agent_name`, `tier`, `task_type`
  - `error_type` — Failure reason (e.g., "timeout", "oom", "validation_error", "dependency_failure")
- **Update**: +1 for each failure

**Error Rate Calculation**:

```
error_rate = tasks_failed_total / tasks_assigned_total × 100
```

**Example Query**:

```promql
# Error rate per agent
(rate(elite_tasks_failed_total[5m]) /
 rate(elite_tasks_assigned_total[5m])) × 100

# Failure distribution by error type
sum by (error_type) (rate(elite_tasks_failed_total[5m]))

# Agents with high error rate
(rate(elite_tasks_failed_total[5m]) /
 rate(elite_tasks_assigned_total[5m])) > 0.2
```

---

#### elite_task_duration_seconds (Histogram)

- **Type**: Histogram with buckets
- **Purpose**: Task execution duration distribution
- **Buckets**: 0.01s, 0.05s, 0.1s, 0.5s, 1s, 5s, 10s, 30s, 60s
- **Labels**:
  - `agent_id`, `agent_name`, `tier`, `task_type`
- **Update**: Record completion time for each task

**Quantile Tracking**:

```
p50  (50th percentile) — Typical task duration
p95  (95th percentile) — Most tasks complete within this time
p99  (99th percentile) — Tail latency
p100 (maximum) — Worst-case duration
```

**Example Query**:

```promql
# Average duration
rate(elite_task_duration_seconds_sum[5m]) /
rate(elite_task_duration_seconds_count[5m])

# Percentiles
histogram_quantile(0.5, rate(elite_task_duration_seconds_bucket[5m]))   # p50
histogram_quantile(0.95, rate(elite_task_duration_seconds_bucket[5m]))  # p95
histogram_quantile(0.99, rate(elite_task_duration_seconds_bucket[5m]))  # p99

# Agents with long tasks
histogram_quantile(0.95, rate(elite_task_duration_seconds_bucket[5m])) > 10
```

---

### 1.3 Utilization & Capacity Metrics (5 metrics)

#### elite_agent_utilization_ratio (Gauge)

- **Type**: Gauge (0.0 to 1.0)
- **Purpose**: Current resource utilization
- **Calculation**: active_tasks / max_concurrent_capacity
- **Labels**:
  - `agent_id`, `agent_name`, `tier`
- **Update Frequency**: Every 10 seconds

**Health Zones**:

- `0.0 - 0.3` — Underutilized (inefficient)
- `0.3 - 0.7` — Optimal (balanced load)
- `0.7 - 0.95` — Heavily loaded (saturated)
- `0.95 - 1.0` — Over capacity (queuing)

**Example Query**:

```promql
# Utilization by tier
avg by (tier) (elite_agent_utilization_ratio)

# Over-utilized agents
elite_agent_utilization_ratio > 0.95

# Underutilized agents
elite_agent_utilization_ratio < 0.3

# Utilization trend
avg_over_time(elite_agent_utilization_ratio[1h])
```

---

#### elite_agent_active_tasks (Gauge)

- **Type**: Gauge (integer)
- **Purpose**: Currently executing tasks
- **Labels**:
  - `agent_id`, `agent_name`, `tier`
- **Update**: Real-time

**Example Query**:

```promql
# Total active tasks across collective
sum(elite_agent_active_tasks)

# Agents with active tasks
elite_agent_active_tasks > 0

# Active tasks by tier
sum by (tier) (elite_agent_active_tasks)
```

---

#### elite_agent_max_concurrent_capacity (Gauge)

- **Type**: Gauge (integer)
- **Purpose**: Maximum concurrent tasks supported
- **Labels**:
  - `agent_id`, `agent_name`, `tier`
- **Update**: On config change

**Static per Agent** (typical):

- Tier 1: 10-20 concurrent tasks
- Tier 2: 8-15 concurrent tasks
- Tiers 3-8: 5-12 concurrent tasks

**Example Query**:

```promql
# Total collective capacity
sum(elite_agent_max_concurrent_capacity)

# Capacity by tier
sum by (tier) (elite_agent_max_concurrent_capacity)

# Spare capacity
sum(elite_agent_max_concurrent_capacity) -
sum(elite_agent_active_tasks)
```

---

#### elite_agent_wait_queue_length (Gauge)

- **Type**: Gauge (integer)
- **Purpose**: Tasks waiting for execution
- **Labels**:
  - `agent_id`, `agent_name`, `tier`
- **Update**: Real-time

**Queue Health**:

- `0` — No queue (ideal)
- `1-5` — Normal (slight backlog)
- `5-20` — Elevated (needs attention)
- `>20` — Critical (capacity issue)

**Example Query**:

```promql
# Total queue length
sum(elite_agent_wait_queue_length)

# Agents with large queues
elite_agent_wait_queue_length > 20

# Average queue depth by tier
avg by (tier) (elite_agent_wait_queue_length)
```

---

#### elite_agent_idle_time_percent (Gauge)

- **Type**: Gauge (0-100)
- **Purpose**: Percentage of time agent is idle
- **Calculation**: (idle_seconds / total_seconds) × 100
- **Window**: 5-minute rolling
- **Labels**:
  - `agent_id`, `agent_name`, `tier`

**Ideal Range**: 30-50% (balanced utilization)

**Example Query**:

```promql
# Idle time by agent
elite_agent_idle_time_percent

# Highly idle agents (inefficient)
elite_agent_idle_time_percent > 70

# Busy agents (low idle time)
elite_agent_idle_time_percent < 20
```

---

### 1.4 Performance & Quality Metrics (4 metrics)

#### elite_agent_success_rate (Gauge)

- **Type**: Gauge (0-100)
- **Purpose**: Percentage of successful tasks
- **Calculation**: (completed / assigned) × 100
- **Window**: 5-minute rolling average
- **Labels**:
  - `agent_id`, `agent_name`, `tier`

**Quality Tiers**:

- `95-100%` — Excellent
- `85-95%` — Good
- `70-85%` — Acceptable
- `<70%` — Poor (needs intervention)

**Example Query**:

```promql
# Success rate by agent
elite_agent_success_rate

# Low success rate alert
elite_agent_success_rate < 70

# Success rate trend
avg_over_time(elite_agent_success_rate[1h])
```

---

#### elite_agent_error_rate (Gauge)

- **Type**: Gauge (0-100)
- **Purpose**: Percentage of failed tasks
- **Calculation**: (failed / assigned) × 100
- **Window**: 5-minute rolling average
- **Labels**:
  - `agent_id`, `agent_name`, `tier`

**Error Severity**:

- `0-5%` — Acceptable
- `5-15%` — Warning
- `15-25%` — Critical
- `>25%` — Severe (intervention required)

**Example Query**:

```promql
# Error rate by agent
elite_agent_error_rate

# High error rate alert
elite_agent_error_rate > 15

# Error rate by tier
avg by (tier) (elite_agent_error_rate)
```

---

#### elite_agent_timeout_rate (Gauge)

- **Type**: Gauge (0-100)
- **Purpose**: Percentage of tasks that timeout
- **Calculation**: (timeouts / assigned) × 100
- **Labels**:
  - `agent_id`, `agent_name`, `tier`

**Warning Thresholds**:

- `>10%` — Elevated (check performance)
- `>20%` — High (capacity or config issue)

**Example Query**:

```promql
# Timeout rate analysis
elite_agent_timeout_rate

# Agents with high timeout rate
elite_agent_timeout_rate > 10
```

---

#### elite_agent_retry_rate (Gauge)

- **Type**: Gauge (0-100)
- **Purpose**: Percentage of tasks requiring retry
- **Calculation**: (retries / assigned) × 100
- **Labels**:
  - `agent_id`, `agent_name`, `tier`

**Retry Analysis**:

- `0-5%` — Normal
- `5-15%` — Elevated instability
- `>15%` — Systemic issue

**Example Query**:

```promql
# Agents with high retry rate
elite_agent_retry_rate > 10

# Retry trend
avg_over_time(elite_agent_retry_rate[1h])
```

---

### 1.5 Collective-Level Metrics (5 metrics)

#### elite_collective_utilization_ratio (Gauge)

- **Type**: Gauge (0.0 to 1.0)
- **Purpose**: Average utilization across tier
- **Calculation**: avg(utilization) by tier
- **Labels**:
  - `tier` — Tier number (1-8)

**Example Query**:

```promql
# Collective utilization by tier
elite_collective_utilization_ratio

# Overall collective utilization
avg(elite_collective_utilization_ratio)
```

---

#### elite_collective_active_tasks (Gauge)

- **Type**: Gauge (integer)
- **Purpose**: Total active tasks in tier
- **Labels**:
  - `tier`

**Example Query**:

```promql
# Total active tasks in collective
sum(elite_collective_active_tasks)

# Active tasks by tier
elite_collective_active_tasks
```

---

#### elite_collective_throughput_tasks_per_sec (Gauge)

- **Type**: Gauge (tasks/second)
- **Purpose**: Collective completion rate
- **Calculation**: rate(tasks_completed[5m])
- **Labels**:
  - `tier`

**Example Query**:

```promql
# Throughput by tier
elite_collective_throughput_tasks_per_sec

# Total collective throughput
sum(elite_collective_throughput_tasks_per_sec)

# Peak throughput
max(elite_collective_throughput_tasks_per_sec)
```

---

#### elite_collective_error_rate (Gauge)

- **Type**: Gauge (0-100)
- **Purpose**: Collective error rate by tier
- **Labels**:
  - `tier`

**Example Query**:

```promql
# Error rate by tier
elite_collective_error_rate

# Alert on high collective error
elite_collective_error_rate > 15
```

---

#### elite_collective_coordination_effectiveness (Gauge)

- **Type**: Gauge (0-100)
- **Purpose**: How well agents coordinate
- **Calculation**: Blend of handoff success, minimal retry, smooth load balance
- **Labels**:
  - `tier`

**Example Query**:

```promql
# Coordination effectiveness
elite_collective_coordination_effectiveness

# Poor coordination alert
elite_collective_coordination_effectiveness < 60
```

---

### 1.6 Cross-Agent Collaboration Metrics (4 metrics)

#### elite_agent_handoff_events (Counter)

- **Type**: Counter
- **Purpose**: Track task handoffs between agents
- **Labels**:
  - `from_agent`, `to_agent` — Agent IDs
  - `from_tier`, `to_tier` — Tier numbers
- **Update**: +1 for each handoff

**Handoff Types**:

- Same-tier handoffs (intra-tier collaboration)
- Cross-tier handoffs (inter-tier delegation)
- Fallback handoffs (degradation recovery)

**Example Query**:

```promql
# Total handoff rate
rate(elite_agent_handoff_events_total[5m])

# Handoffs from specific agent
rate(elite_agent_handoff_events_total{from_agent="apex"}[5m])

# Inter-tier handoffs
sum by (from_tier, to_tier) (
  rate(elite_agent_handoff_events_total[5m]))

# Most common handoff paths
topk(10, rate(elite_agent_handoff_events_total[5m]))
```

---

#### elite_agent_collaboration_score (Gauge)

- **Type**: Gauge (0-1)
- **Purpose**: Collaboration strength between specific agents
- **Labels**:
  - `agent_a`, `agent_b` — Agent IDs
  - `tier` — Shared or primary tier
- **Update Frequency**: Hourly

**Calculation Factors**:

- Frequency of handoffs
- Success rate of collaborative tasks
- Specialization complementarity
- Communication latency

**Example Query**:

```promql
# Strongest collaborations
topk(10, elite_agent_collaboration_score)

# Weak collaborations
elite_agent_collaboration_score < 0.3
```

---

#### elite_specialization_overlap (Gauge)

- **Type**: Gauge (0-1)
- **Purpose**: How much specializations overlap
- **Labels**:
  - `agent_id`, `agent_name`
  - `specialization`
- **Update Frequency**: Static (configuration-driven)

**Interpretation**:

- `0.0-0.2` — Highly specialized (complementary)
- `0.2-0.5` — Some overlap (reasonable)
- `0.5-0.8` — Significant overlap (potential redundancy)
- `0.8-1.0` — Nearly identical (redundant)

**Example Query**:

```promql
# Overlapping agents
elite_specialization_overlap > 0.5

# Complementary pairs
elite_specialization_overlap < 0.2
```

---

#### elite_load_balance_efficiency (Gauge)

- **Type**: Gauge (0-100)
- **Purpose**: How efficiently tasks are distributed
- **Calculation**: (ideal_distribution_score / actual_distribution) × 100
- **Labels**:
  - `tier`

**Efficiency Metric**:

```
Score = 100 × (1 - std_dev(utilization) / mean(utilization))
```

**Interpretation**:

- `>90%` — Excellent balance
- `70-90%` — Good balance
- `50-70%` — Fair balance
- `<50%` — Poor balance (rebalance needed)

**Example Query**:

```promql
# Load balance by tier
elite_load_balance_efficiency

# Poor load balance alert
elite_load_balance_efficiency < 50
```

---

### 1.7 Tier-Based Performance Metrics (5 metrics)

#### elite_tier_utilization_ratio (Gauge)

- **Type**: Gauge (0.0 to 1.0)
- **Purpose**: Average utilization for entire tier
- **Labels**:
  - `tier`, `tier_name` (e.g., "1", "Tier 1 - Foundational")

**Example Query**:

```promql
# Utilization by tier
elite_tier_utilization_ratio

# Tier rankings by utilization
sort_desc(elite_tier_utilization_ratio)
```

---

#### elite_tier_agent_count (Gauge)

- **Type**: Gauge (integer)
- **Purpose**: How many agents in each tier
- **Labels**:
  - `tier`, `tier_name`

**Agent Distribution**:

- Tier 1: 5 agents (foundational)
- Tier 2: 8 agents (specialists)
- Tier 3: 4 agents (innovators)
- Tier 4: 1 agent (meta - @OMNISCIENT)
- Tier 5: 5 agents (domain)
- Tier 6: 5 agents (emerging)
- Tier 7: 5 agents (human-centric)
- Tier 8: 5 agents (enterprise)
- **Total: 40 agents**

**Example Query**:

```promql
# Agent count by tier
elite_tier_agent_count

# Verify expected counts
elite_tier_agent_count == on(tier) group_left() EXPECTED_COUNTS
```

---

#### elite_tier_total_tasks (Counter)

- **Type**: Counter
- **Purpose**: Cumulative task count per tier
- **Labels**:
  - `tier`, `tier_name`

**Example Query**:

```promql
# 24-hour task volume
increase(elite_tier_total_tasks[24h])

# Task rate by tier
rate(elite_tier_total_tasks[5m])
```

---

#### elite_tier_success_rate (Gauge)

- **Type**: Gauge (0-100)
- **Purpose**: Average success rate for tier
- **Labels**:
  - `tier`, `tier_name`

**Example Query**:

```promql
# Success rate ranking
sort_desc(elite_tier_success_rate)

# Tiers below SLO
elite_tier_success_rate < 95
```

---

#### elite_tier_throughput (Gauge)

- **Type**: Gauge (tasks/second)
- **Purpose**: Throughput for each tier
- **Labels**:
  - `tier`, `tier_name`

**Example Query**:

```promql
# Throughput by tier
elite_tier_throughput

# Total collective throughput
sum(elite_tier_throughput)

# Tier ranking by throughput
sort_desc(elite_tier_throughput)
```

---

### 1.8 Meta-Intelligence Metrics (5 metrics)

#### elite_agent_learning_rate (Gauge)

- **Type**: Gauge (0-1)
- **Purpose**: How fast agent is improving
- **Calculation**: (current_success - previous_success) / time_window
- **Window**: 24-hour rolling
- **Labels**:
  - `agent_id`, `agent_name`, `tier`

**Interpretation**:

- `>0.05` — Rapid improvement
- `0.01 to 0.05` — Steady improvement
- `-0.01 to 0.01` — Stable
- `<-0.01` — Degradation

**Example Query**:

```promql
# Agents improving fastest
topk(10, elite_agent_learning_rate)

# Agents with negative learning rate
elite_agent_learning_rate < 0
```

---

#### elite_breakthrough_discoveries (Counter)

- **Type**: Counter
- **Purpose**: Count of breakthrough solutions
- **Labels**:
  - `agent_id`, `agent_name`, `tier`, `specialization`
- **Update**: +1 when solution fitness > 0.9

**Breakthrough Criteria**:

- Fitness score exceeding 0.9
- Novel approach (not seen before)
- Cross-tier applicable

**Example Query**:

```promql
# Breakthroughs in past 24h
increase(elite_breakthrough_discoveries_total[24h])

# Breakthroughs by specialization
sum by (specialization) (
  increase(elite_breakthrough_discoveries_total[24h]))

# Breakthroughs by tier
sum by (tier) (
  increase(elite_breakthrough_discoveries_total[24h]))
```

---

#### elite_knowledge_sharing_events (Counter)

- **Type**: Counter
- **Purpose**: Knowledge transfers between agents
- **Labels**:
  - `from_agent`, `to_agent`
  - `from_tier`, `to_tier`
- **Update**: +1 for each knowledge share

**Example Query**:

```promql
# Knowledge sharing rate
rate(elite_knowledge_sharing_events_total[5m])

# Most shared knowledge
topk(10, sum by (from_agent) (
  rate(elite_knowledge_sharing_events_total[5m])))

# Inter-tier knowledge transfer
sum by (from_tier, to_tier) (
  rate(elite_knowledge_sharing_events_total[5m]))
```

---

#### elite_collective_intelligence_score (Gauge)

- **Type**: Gauge (0-100)
- **Purpose**: Overall collective capability
- **Calculation**: Weighted blend of:
  - Success rates (40%)
  - Breakthrough discoveries (30%)
  - Knowledge sharing (20%)
  - Coordination effectiveness (10%)
- **Labels**:
  - `tier`

**Example Query**:

```promql
# Intelligence by tier
elite_collective_intelligence_score

# Overall collective intelligence
avg(elite_collective_intelligence_score)

# Intelligence trend
avg_over_time(elite_collective_intelligence_score[7d])
```

---

#### elite_mnemonic_memory_fitness (Gauge)

- **Type**: Gauge (0-1)
- **Purpose**: Quality of learned experiences
- **Calculation**: Average fitness of all stored experiences for agent
- **Labels**:
  - `agent_id`, `agent_name`, `tier`
- **Update Frequency**: Hourly

**Fitness Score Breakdown**:

- Quality (0.0-1.0): Does solution work well?
- Relevance (0.0-1.0): How applicable is it?
- Novelty (0.0-2.0): Is it new/innovative?

```
Overall Fitness = sqrt(Quality × Relevance) × (1 + Novelty × 0.5)
```

**Example Query**:

```promql
# Memory fitness by agent
elite_mnemonic_memory_fitness

# Agents with low memory fitness
elite_mnemonic_memory_fitness < 0.5

# Average fitness by tier
avg by (tier) (elite_mnemonic_memory_fitness)
```

---

#### elite_retrieval_efficiency_percent (Gauge)

- **Type**: Gauge (0-100)
- **Purpose**: Effectiveness of memory retrieval
- **Labels**:
  - `retrieval_type` — One of:
    - `bloom_filter` (O(1) exact match)
    - `lsh_index` (O(1) approximate NN)
    - `hnsw_graph` (O(log n) semantic search)
  - `tier`

**Efficiency Metric**:

```
Efficiency = (Relevant Results / Total Results) × 100
```

**Targets**:

- Bloom Filter: >99% (should be near-perfect)
- LSH Index: >85% (approximate, some false positives)
- HNSW Graph: >90% (high precision)

**Example Query**:

```promql
# Retrieval efficiency by type
elite_retrieval_efficiency_percent

# Under-performing retrieval
elite_retrieval_efficiency_percent < 80
```

---

## Part 2: Label Strategy for Agent Coordination

### Label Naming Conventions

```
{
  agent_id: string,           # Unique identifier (lowercase)
  agent_name: string,         # Display name (uppercase)
  tier: int,                  # Tier number 1-8
  tier_name: string,          # Tier display name
  specialization: string,     # Agent specialization
  task_type: string,          # Category of task
  error_type: string,         # Error classification
  status: string,             # Operational state
  retrieval_type: string,     # Memory retrieval technique
}
```

### High-Cardinality Reduction

**Problem**: Too many label combinations create metric explosion

**Solutions**:

1. **Metric Aggregation**: Pre-compute common aggregations

   ```promql
   # Instead of: elite_task_duration_seconds{agent_id="...", task_type="..."}
   # Use: elite_task_duration_seconds_aggregate{task_type="..."}
   ```

2. **Downsampling**: Reduce resolution over time

   ```yaml
   # Keep raw data for 7 days
   # Keep 5-min samples for 30 days
   # Keep 1-hour samples for 1 year
   ```

3. **Label Dropping**: Remove unnecessary labels
   ```promql
   # Only keep tier, not agent_id for collective metrics
   ```

---

## Part 3: Aggregation Patterns

### By-Agent Aggregation

```promql
# Query individual agent metrics
elite_agent_status{agent_id="apex"}
elite_agent_utilization_ratio{agent_id="apex"}
elite_tasks_completed_total{agent_id="apex"}
```

### By-Tier Aggregation

```promql
# Sum across all agents in tier
sum by (tier) (elite_agent_active_tasks)
avg by (tier) (elite_agent_utilization_ratio)
sum by (tier) (rate(elite_tasks_completed_total[5m]))
```

### By-Specialization Aggregation

```promql
# Group by specialization
sum by (specialization) (increase(
  elite_breakthrough_discoveries_total[24h]))

avg by (specialization) (elite_agent_success_rate)
```

### Collective Aggregation (All Tiers)

```promql
# Aggregate across all agents
sum(elite_agent_active_tasks)
avg(elite_agent_success_rate)
sum(rate(elite_tasks_completed_total[5m]))
```

### Cross-Tier Aggregation

```promql
# From Tier 1 to Tier 2 handoffs
sum by (from_tier, to_tier) (
  rate(elite_agent_handoff_events_total[5m]))
```

---

## Part 4: Alert Rules

### Critical Alerts

**AgentFailed**: Agent not responding

```yaml
expr: elite_agent_status == 2
for: 2m
severity: critical
```

**HighErrorRate**: Agent error rate > 20%

```yaml
expr: elite_agent_error_rate > 20
for: 5m
severity: critical
```

**LowSuccessRate**: Agent success rate < 50%

```yaml
expr: elite_agent_success_rate < 50
for: 10m
severity: critical
```

### Warning Alerts

**AgentDegraded**: Agent performance reduced

```yaml
expr: elite_agent_status == 1
for: 5m
severity: warning
```

**HighTimeoutRate**: Task timeout rate > 10%

```yaml
expr: elite_agent_timeout_rate > 10
for: 5m
severity: warning
```

**AgentOverUtilized**: Utilization > 95%

```yaml
expr: elite_agent_utilization_ratio > 0.95
for: 5m
severity: warning
```

### Info Alerts

**BreakthroughDiscovered**: New high-fitness solution

```yaml
expr: increase(elite_breakthrough_discoveries_total[1h]) > 0
for: 1m
severity: info
```

---

## Part 5: Grafana Dashboard Configuration

### Dashboard 1: Collective Health Overview

**Layout**:

- Top row:
  - Collective Utilization (gauge)
  - Total Active Tasks (stat)
  - Collective Success Rate (gauge)
  - Failed Agents (stat)

- Middle row:
  - Active Tasks by Tier (bar chart)
  - Success Rate by Tier (line chart)
  - Error Rate Distribution (pie chart)

- Bottom row:
  - Agent Status Table (with live status)
  - Recent Recovery Events (timeline)
  - Throughput Trend (line chart)

---

### Dashboard 2: Individual Agent Details

**Variables**: agent_id selector

**Layout**:

- Top: Agent Status, Availability, Uptime
- Row 1: Utilization Trend, Active Tasks, Queue Length
- Row 2: Success Rate, Error Rate, Timeout Rate
- Row 3: Task Duration Distribution, Task Type Breakdown
- Row 4: Collaboration Network (this agent's handoffs)

---

### Dashboard 3: Tier Performance Analysis

**Variables**: tier selector

**Layout**:

- Top: Tier Utilization, Tier Throughput, Tier Success Rate
- Row 1: Per-Agent Utilization (table), Load Balance Efficiency
- Row 2: Task Distribution, Error Type Breakdown
- Row 3: Inter-tier Handoffs (from/to this tier)

---

### Dashboard 4: Meta-Intelligence Tracking

**Layout**:

- Row 1:
  - Breakthrough Discoveries (24h count)
  - Knowledge Sharing Events (24h count)
  - Learning Rate Distribution (heatmap)
  - Collective Intelligence Score

- Row 2:
  - Memory Fitness by Agent (bar chart)
  - Retrieval Efficiency by Type (multi-gauge)
  - Memory Fitness Trend (line chart)

- Row 3:
  - Agent Learning Rates (heatmap)
  - Breakthrough Timeline (events over time)

---

## Part 6: Optimization Opportunities

### Utilization Imbalance

**Detection**: Max utilization - min utilization > 0.4

**Recommendation**: Implement dynamic task routing, consider task specialization boundaries

### High Error Rates

**Detection**: Error rate > 20% for agent

**Recommendation**: Review error logs, implement circuit breaker, add retry logic

### Collaboration Gaps

**Detection**: Low handoff count between complementary tiers

**Recommendation**: Review specialization boundaries, implement knowledge sharing mechanisms

### Memory Fitness Degradation

**Detection**: Average memory fitness < 0.5

**Recommendation**: Audit stored experiences, refresh low-fitness items, retrain learning model

---

## Conclusion

This comprehensive metrics design provides complete observability for the 40-agent Elite Agent Collective. The architecture supports:

✅ Individual agent monitoring
✅ Per-tier aggregation and analysis  
✅ Cross-agent collaboration tracking
✅ Collective health and coordination
✅ Meta-intelligence and learning metrics
✅ Memory system observability
✅ Automated alerting
✅ Optimization opportunity detection

**Next Steps**:

1. Deploy metrics.py to production
2. Configure Prometheus scraper
3. Create Grafana dashboards
4. Define SLOs and alerting thresholds
5. Monitor and continuously optimize
