# Elite Agent Collective Metrics - Complete Reference

## Overview

The Elite Agent Collective is a sophisticated multi-agent system consisting of 40 specialized AI agents organized into 8 tiers, with comprehensive monitoring across individual agents, collectives, and inter-agent collaboration. This document provides complete reference documentation for all metrics.

**File Location:** `agents/monitoring/metrics.py` (487 lines)

### Architecture Overview

```
TIER 1: FOUNDATIONAL (5 agents)
├── @APEX (01) - Computer Science Engineering
├── @CIPHER (02) - Cryptography & Security
├── @ARCHITECT (03) - Systems Architecture
├── @AXIOM (04) - Pure Mathematics
└── @VELOCITY (05) - Performance Optimization

TIER 2: SPECIALISTS (12 agents)
├── @QUANTUM, @TENSOR, @FORTRESS, @NEURAL, @CRYPTO, @FLUX
├── @PRISM, @SYNAPSE, @CORE, @HELIX, @VANGUARD, @ECLIPSE

TIER 3-4: INNOVATORS (2 agents)
├── @NEXUS (18) - Paradigm Synthesis
└── @GENESIS (19) - Zero-to-One Innovation

TIER 5: DOMAIN SPECIALISTS (5 agents)
├── @ATLAS, @FORGE, @SENTRY, @VERTEX, @STREAM

TIER 6: EMERGING TECH (5 agents)
├── @PHOTON, @LATTICE, @MORPH, @PHANTOM, @ORBIT

TIER 7: HUMAN-CENTRIC (5 agents)
├── @CANVAS, @LINGUA, @SCRIBE, @MENTOR, @BRIDGE

TIER 8: ENTERPRISE (2 agents)
├── @AEGIS, @LEDGER, @PULSE, @ARBITER, @ORACLE
```

---

## Individual Agent Metrics

### Agent Status and Health

#### `agents_status` (Gauge)

**Type:** Gauge  
**Labels:** `agent_id`, `agent_name`, `tier`  
**Range:** 0 (healthy), 1 (degraded), 2 (failed)  
**Help:** Current agent operational status

**Agent ID Map:**

```
Tier 1: 01-05
Tier 2: 06-17
Tier 3: 18-19
Tier 5: 21-25
Tier 6: 26-30
Tier 7: 31-35
Tier 8: 36-40
```

**Example Query:**

```promql
# Health status by tier
agents_status by (tier)

# Failed agents
agents_status == 2

# Degraded agents
agents_status == 1
```

**Alert Rules:**

```yaml
- alert: AgentDown
  expr: agents_status == 2
  for: 2m
  annotations:
    summary: "Agent {{ $labels.agent_name }} ({{ $labels.agent_id }}) is down"

- alert: AgentDegraded
  expr: agents_status == 1
  for: 5m
  annotations:
    summary: "Agent {{ $labels.agent_name }} degraded"
```

---

#### `agents_active_tasks` (Gauge)

**Type:** Gauge  
**Labels:** `agent_id`, `agent_name`  
**Help:** Number of currently active tasks for agent

**Usage:**

```python
agents_active_tasks.labels(agent_id='01', agent_name='APEX').set(5)
```

**Example Queries:**

```promql
# Most loaded agents
topk(5, agents_active_tasks)

# Idle agents
agents_active_tasks == 0

# Total active tasks
sum(agents_active_tasks)
```

---

#### `agents_availability_ratio` (Gauge)

**Type:** Gauge  
**Labels:** `agent_id`, `agent_name`  
**Range:** 0.0 to 1.0  
**Help:** Agent availability (uptime ratio)

**Calculation:**

```python
# Track uptime in seconds
uptime = time_since_last_failure
downtime = sum(failure_durations)
availability = uptime / (uptime + downtime)

agents_availability_ratio.labels(agent_id='01', agent_name='APEX').set(availability)
```

**SLA Targets:**

- 99.9% = Agent available 99.9% of time
- 99.95% = ~4 hours downtime/month
- 99.99% = ~52 minutes downtime/month

---

### Agent Task Metrics

#### `agents_tasks_total` (Counter)

**Type:** Counter  
**Labels:** `agent_id`, `agent_name`, `task_type`  
**Help:** Cumulative task count for agent

**Task Types by Agent:**

```
@APEX:       coding, debugging, optimization, design
@CIPHER:     cryptanalysis, key_gen, security_audit, threat_model
@ARCHITECT:  design, pattern_selection, decomposition
@AXIOM:      proof, analysis, complexity_analysis, optimization
@VELOCITY:   profiling, optimization, algorithm_selection
@TENSOR:     model_design, training, inference, tuning
@FORTRESS:   pentest, vuln_analysis, threat_hunt, malware_analysis
@NEURAL:     research, reasoning, alignment, meta_learning
```

**Example Query:**

```promql
# Task rate (tasks/second) by agent
rate(agents_tasks_total[1m]) by (agent_name)

# Total tasks assigned
increase(agents_tasks_total[24h]) by (agent_id)
```

---

#### `agents_tasks_completed` (Counter)

**Type:** Counter  
**Labels:** `agent_id`, `agent_name`, `task_type`  
**Help:** Tasks completed successfully

**Example Queries:**

```promql
# Task completion rate
rate(agents_tasks_completed[1m]) by (agent_name)

# Success percentage
(rate(agents_tasks_completed[5m]) / rate(agents_tasks_total[5m])) * 100
```

---

#### `agents_tasks_failed` (Counter)

**Type:** Counter  
**Labels:** `agent_id`, `agent_name`, `error_type`  
**Help:** Tasks that failed

**Error Types:**

- `timeout` - Task exceeded time limit
- `resource_exhausted` - Memory/CPU insufficient
- `invalid_input` - Bad input data
- `dependency_failure` - Required resource unavailable
- `unknown` - Unknown failure

**Example Query:**

```promql
# Failure rate by error type
rate(agents_tasks_failed[5m]) by (error_type)

# Agent with highest failure rate
topk(5, (rate(agents_tasks_failed[5m]) / rate(agents_tasks_total[5m])))
```

---

#### `agents_task_duration_seconds` (Histogram)

**Type:** Histogram  
**Labels:** `agent_id`, `agent_name`, `task_type`  
**Buckets:** 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0 (seconds)  
**Help:** Task execution duration

**Bucket Selection:**

- 100ms-500ms: Quick analysis tasks
- 1-5s: Medium complexity tasks
- 10-30s: Complex problem solving
- 60s+: Research/deep analysis

**Example Queries:**

```promql
# P50 (median) task duration
histogram_quantile(0.5, rate(agents_task_duration_seconds_bucket[5m])) by (agent_name)

# P95 task duration
histogram_quantile(0.95, rate(agents_task_duration_seconds_bucket[5m])) by (agent_name)

# P99 task duration (long tail)
histogram_quantile(0.99, rate(agents_task_duration_seconds_bucket[5m])) by (agent_name)

# Average task time
rate(agents_task_duration_seconds_sum[5m]) / rate(agents_task_duration_seconds_count[5m]) by (agent_name)
```

---

### Agent Performance Metrics

#### `agents_success_rate` (Gauge)

**Type:** Gauge  
**Labels:** `agent_id`, `agent_name`  
**Range:** 0.0 to 1.0  
**Help:** Agent success rate (fraction of successful tasks)

**Calculation:**

```python
success_rate = completed_tasks / (completed_tasks + failed_tasks)
agents_success_rate.labels(agent_id='01', agent_name='APEX').set(success_rate)
```

**Performance Tiers:**

- ≥ 0.99: Excellent (99%+ success)
- 0.95-0.99: Good (95-99% success)
- 0.90-0.95: Fair (90-95% success)
- < 0.90: Poor (needs attention)

**Example Query:**

```promql
# Agents with < 95% success rate
agents_success_rate < 0.95
```

**Alert Rules:**

```yaml
- alert: LowAgentSuccessRate
  expr: agents_success_rate < 0.95
  for: 30m
  annotations:
    summary: "Agent {{ $labels.agent_name }} success rate: {{ $value | humanizePercentage }}"
```

---

#### `agents_error_rate` (Gauge)

**Type:** Gauge  
**Labels:** `agent_id`, `agent_name`  
**Range:** 0.0 to 1.0  
**Help:** Agent error rate (fraction of failed tasks)

**Inverse Relationship:**

```
error_rate = 1 - success_rate
```

---

#### `agents_timeout_rate` (Gauge)

**Type:** Gauge  
**Labels:** `agent_id`, `agent_name`  
**Range:** 0.0 to 1.0  
**Help:** Fraction of tasks that timed out

**Threshold Analysis:**

- < 1%: Normal
- 1-5%: Elevated (monitor)
- > 5%: High (investigate)

---

#### `agents_average_task_duration` (Gauge)

**Type:** Gauge  
**Labels:** `agent_id`, `agent_name`  
**Unit:** Seconds  
**Help:** Average task execution time

**Baseline by Agent Category:**

```
Fast (Analysis):       0.5s  - @PRISM, @AXIOM
Medium (Coding):       5s    - @APEX, @ARCHITECT
Slow (Research):       30s   - @NEURAL, @VANGUARD
Very Slow (Training):  300s+ - @TENSOR
```

---

#### `agents_p95_task_duration` (Gauge)

**Type:** Gauge  
**Labels:** `agent_id`, `agent_name`  
**Unit:** Seconds  
**Help:** 95th percentile task execution time

**Use Case:**

- SLA setting: Set timeout to P95 \* 1.5
- Capacity planning: Plan for P95, not average
- Bottleneck detection: Large P95/P50 ratio indicates variability

---

### Agent Utilization Metrics

#### `agents_utilization_ratio` (Gauge)

**Type:** Gauge  
**Labels:** `agent_id`, `agent_name`  
**Range:** 0.0 to 1.0  
**Help:** Agent utilization (active_tasks / max_capacity)

**Interpretation:**

```
0.0-0.3:  Under-utilized (consider consolidation)
0.3-0.7:  Healthy utilization
0.7-0.9:  High utilization (monitor)
0.9-1.0:  Fully utilized (consider scaling)
```

**Example Queries:**

```promql
# Over-utilized agents
agents_utilization_ratio > 0.9

# Under-utilized agents
agents_utilization_ratio < 0.2

# Average collective utilization
avg(agents_utilization_ratio)
```

---

#### `agents_max_capacity` (Gauge)

**Type:** Gauge  
**Labels:** `agent_id`, `agent_name`  
**Help:** Maximum concurrent tasks for agent

**Capacity by Agent Type:**

```
Foundational:     100 concurrent tasks
Specialists:      50 concurrent tasks
Innovators:       10 concurrent tasks (high complexity)
Domain:           75 concurrent tasks
Emerging:         60 concurrent tasks
Human-Centric:    80 concurrent tasks
Enterprise:       40 concurrent tasks
```

---

#### `agents_queue_length` (Gauge)

**Type:** Gauge  
**Labels:** `agent_id`, `agent_name`  
**Help:** Number of tasks waiting for agent

**Queue Health:**

- Queue = 0: No backlog
- Queue < max_capacity: Normal
- Queue > max_capacity: Backlog forming
- Queue > 3 \* max_capacity: Severe bottleneck

**Example Query:**

```promql
# Tasks waiting per agent
agents_queue_length / agents_max_capacity by (agent_name)
```

---

#### `agents_idle_time_seconds` (Counter)

**Type:** Counter  
**Labels:** `agent_id`, `agent_name`  
**Unit:** Seconds  
**Help:** Cumulative idle time for agent

**Utilization Pattern:**

```
idle_ratio = idle_time_seconds / total_time_seconds

if idle_ratio > 0.7:  # > 70% idle
    consider_consolidation()
```

---

### Agent Specialization Metrics

#### `agents_specialization` (Gauge)

**Type:** Gauge  
**Labels:** `agent_id`, `agent_name`, `specialization`  
**Range:** 0.0 to 1.0  
**Help:** Agent proficiency in specialization

**Specializations by Agent:**

```
@APEX:          coding, architecture, algorithms
@CIPHER:        cryptography, security, key_management
@TENSOR:        deep_learning, training, inference
@FORTRESS:      security_testing, threat_modeling, exploitation
@NEURAL:        reasoning, meta_learning, alignment
@PRISM:         statistics, data_analysis, forecasting
```

**Proficiency Scale:**

- 0.0-0.3: Novice
- 0.3-0.6: Intermediate
- 0.6-0.8: Advanced
- 0.8-1.0: Expert

---

#### `agents_task_type_distribution` (Histogram)

**Type:** Histogram  
**Labels:** `agent_id`, `agent_name`, `task_type`  
**Buckets:** 0.01, 0.05, 0.1, 0.2, 0.3, 0.5  
**Help:** Distribution of task types for agent

**Interpretation:**

- Measure of agent task diversity
- High distribution = multi-skilled agent
- Low distribution = specialist (single task focus)

---

### Agent Recovery and Resilience

#### `agents_recovery_events_total` (Counter)

**Type:** Counter  
**Labels:** `agent_id`, `agent_name`, `recovery_type`  
**Help:** Recovery event count

**Recovery Types:**

- `restart` - Agent restart
- `failover` - Failover to replica
- `circuit_breaker` - Automatic circuit breaker trip
- `manual_intervention` - Operator intervention
- `auto_healing` - Automatic recovery mechanism

**Example Query:**

```promql
# Recovery event rate
rate(agents_recovery_events_total[1h]) by (recovery_type)

# Agents with high recovery frequency
topk(5, rate(agents_recovery_events_total[24h]))
```

---

#### `agents_restart_count` (Counter)

**Type:** Counter  
**Labels:** `agent_id`, `agent_name`  
**Help:** Number of agent restarts

**Alert Rules:**

```yaml
- alert: FrequentRestarts
  expr: increase(agents_restart_count[1h]) > 5
  annotations:
    summary: "Agent {{ $labels.agent_name }} restarted 5+ times in 1h"
```

---

#### `agents_mttr_seconds` (Gauge)

**Type:** Gauge  
**Labels:** `agent_id`, `agent_name`  
**Unit:** Seconds  
**Help:** Mean time to recovery

**Calculation:**

```python
mttr = total_recovery_time / number_of_failures
agents_mttr_seconds.labels(agent_id='01', agent_name='APEX').set(mttr)
```

**Baseline Targets:**

- < 30 seconds: Excellent
- 30-60 seconds: Good
- 60-300 seconds: Fair
- > 300 seconds: Needs improvement

---

## Collective (Aggregate) Metrics

### Collective Health

#### `agents_collective_total` (Gauge)

**Type:** Gauge  
**Help:** Total number of agents in collective

**Example Query:**

```promql
agents_collective_total  # Typically 40
```

---

#### `agents_collective_healthy` (Gauge)

**Type:** Gauge  
**Help:** Number of healthy agents

**Example Query:**

```promql
# Collective health percentage
(agents_collective_healthy / agents_collective_total) * 100
```

---

#### `agents_collective_degraded` (Gauge)

**Type:** Gauge  
**Help:** Number of degraded agents

---

#### `agents_collective_failed` (Gauge)

**Type:** Gauge  
**Help:** Number of failed agents

**Alert Rules:**

```yaml
- alert: CollectiveHealthDegraded
  expr: agents_collective_failed > 2
  for: 5m
  annotations:
    summary: "{{ $value }} agents in collective have failed"

- alert: CriticalCollectiveFailure
  expr: (agents_collective_healthy / agents_collective_total) < 0.8
  for: 2m
  annotations:
    summary: "Collective health below 80% ({{ $value | humanizePercentage }})"
```

---

### Collective Workload

#### `agents_collective_active_tasks` (Gauge)

**Type:** Gauge  
**Help:** Total active tasks across all agents

**Example Queries:**

```promql
# Current collective workload
agents_collective_active_tasks

# Workload distribution
topk(10, agents_active_tasks)

# Average load per agent
agents_collective_active_tasks / agents_collective_healthy
```

---

#### `agents_collective_utilization_ratio` (Gauge)

**Type:** Gauge  
**Range:** 0.0 to 1.0  
**Help:** Overall resource utilization

**Interpretation:**

- Low (< 0.3): Resources underutilized, consider consolidation
- Normal (0.3-0.7): Healthy utilization
- High (0.7-0.9): Operating near capacity
- Critical (> 0.9): Immediate scaling needed

---

#### `agents_collective_throughput_tasks_per_second` (Gauge)

**Type:** Gauge  
**Unit:** Tasks/second  
**Help:** Collective throughput

**Example Query:**

```promql
# Task rate
rate(agents_tasks_completed[1m])

# Peak throughput (5-minute window)
max_over_time(agents_collective_throughput_tasks_per_second[5m])
```

**Baseline Throughput:**

- Development: 1-10 tasks/sec
- Production: 50-500 tasks/sec
- Enterprise: 1000+ tasks/sec

---

### Collective Performance

#### `agents_collective_error_rate` (Gauge)

**Type:** Gauge  
**Range:** 0.0 to 1.0  
**Help:** Collective error rate

**Alert Rules:**

```yaml
- alert: CollectiveErrorRateHigh
  expr: agents_collective_error_rate > 0.05
  for: 10m
  annotations:
    summary: "Collective error rate: {{ $value | humanizePercentage }}"
```

---

#### `agents_collective_success_rate` (Gauge)

**Type:** Gauge  
**Range:** 0.0 to 1.0  
**Help:** Collective success rate

**Inverse Relationship:**

```
collective_success_rate + collective_error_rate = 1.0
```

---

## Tier-Level Metrics

### Tier Health and Performance

#### `agents_tier_health_score` (Gauge)

**Type:** Gauge  
**Labels:** `tier`  
**Range:** 0.0 to 1.0  
**Help:** Health score for tier

**Calculation:**

```python
# Tier health = avg success rate + availability factor
agent_success_rates = [agents_success_rate for agent in tier]
agent_availability = [agents_availability_ratio for agent in tier]

tier_health = (
    (mean(agent_success_rates) * 0.7) +  # 70% weight on success
    (mean(agent_availability) * 0.3)     # 30% weight on availability
)
```

**Tier Organization:**

```
Tier 1 (Foundational):     @APEX, @CIPHER, @ARCHITECT, @AXIOM, @VELOCITY
Tier 2 (Specialists):      @QUANTUM through @ECLIPSE (12 agents)
Tier 3-4 (Innovators):     @NEXUS, @GENESIS
Tier 5 (Domain):           @ATLAS, @FORGE, @SENTRY, @VERTEX, @STREAM
Tier 6 (Emerging):         @PHOTON through @ORBIT
Tier 7 (Human-Centric):    @CANVAS through @BRIDGE
Tier 8 (Enterprise):       @AEGIS through @ORACLE
```

**Example Queries:**

```promql
# Health by tier
agents_tier_health_score by (tier)

# Ranking tiers by health
sort(agents_tier_health_score)
```

---

#### `agents_tier_utilization_ratio` (Gauge)

**Type:** Gauge  
**Labels:** `tier`  
**Range:** 0.0 to 1.0  
**Help:** Average utilization in tier

**Example Query:**

```promql
# Most loaded tiers
sort_desc(agents_tier_utilization_ratio)
```

---

#### `agents_tier_task_count` (Counter)

**Type:** Counter  
**Labels:** `tier`  
**Help:** Cumulative tasks handled by tier

**Example Query:**

```promql
# Task distribution across tiers
sum by(tier) (rate(agents_tier_task_count[24h]))
```

---

#### `agents_tier_error_rate` (Gauge)

**Type:** Gauge  
**Labels:** `tier`  
**Range:** 0.0 to 1.0  
**Help:** Error rate for tier

---

## Inter-Agent Collaboration Metrics

### Agent Communication

#### `agents_collaboration_events_total` (Counter)

**Type:** Counter  
**Labels:** `initiator_agent`, `target_agent`  
**Help:** Inter-agent collaboration count

**Use Cases:**

- Agent seeking specialized knowledge from another
- Joint problem solving
- Resource sharing
- Knowledge transfer

**Example Queries:**

```promql
# Collaboration edges (directed)
agents_collaboration_events_total

# Most collaborative agents
topk(10, sum by(initiator_agent) (rate(agents_collaboration_events_total[24h])))

# Collaboration matrix heatmap
sum by(initiator_agent, target_agent) (rate(agents_collaboration_events_total[1h]))
```

---

#### `agents_handoff_total` (Counter)

**Type:** Counter  
**Labels:** `source_agent`, `target_agent`  
**Help:** Task handoff count between agents

**Scenarios:**

- Source agent reaches capacity, hands off to target
- Source agent specialization insufficient, escalates to expert
- Load balancing handoff
- Failure recovery handoff

**Example Query:**

```promql
# Handoff frequency
rate(agents_handoff_total[1h]) by (source_agent, target_agent)

# Most common handoff paths
topk(20, increase(agents_handoff_total[24h]))
```

---

#### `agents_communication_latency_seconds` (Histogram)

**Type:** Histogram  
**Labels:** None  
**Buckets:** 0.001, 0.01, 0.1, 1.0, 10.0 (seconds)  
**Help:** Communication latency between agents

**Latency Baseline:**

- Same tier: 1-10ms
- Adjacent tier: 10-50ms
- Distant tier: 50-500ms

**Example Query:**

```promql
# Average inter-agent latency
rate(agents_communication_latency_seconds_sum[5m]) / rate(agents_communication_latency_seconds_count[5m])

# P95 latency (critical for real-time tasks)
histogram_quantile(0.95, rate(agents_communication_latency_seconds_bucket[5m]))
```

---

## Collective Intelligence Metrics

### Emergent Properties

#### `agents_collective_intelligence_score` (Gauge)

**Type:** Gauge  
**Range:** 0.0 to 1.0  
**Help:** Collective intelligence metric

**Calculation:**

```python
# Multi-factor intelligence score
collaboration_factor = collective_collaboration_events / time_period
diversity_factor = len(unique_agent_specializations) / 40
efficiency_factor = collective_success_rate * (1 - collective_error_rate)
emergence_factor = breakthrough_count / time_period

collective_intelligence_score = (
    (collaboration_factor * 0.3) +
    (diversity_factor * 0.2) +
    (efficiency_factor * 0.4) +
    (emergence_factor * 0.1)
)
```

**Interpretation:**

- < 0.5: Collective under-performing
- 0.5-0.7: Normal operation
- 0.7-0.9: High performance
- 0.9+: Exceptional collective intelligence

---

#### `agents_collective_breakthrough_count` (Counter)

**Type:** Counter  
**Help:** Novel insights and breakthroughs

**Definition:**

- Solution with fitness score > 0.9
- Novel algorithm or pattern
- Cross-tier knowledge synthesis
- Emergent behavior not pre-programmed

**Example Query:**

```promql
# Breakthrough rate
rate(agents_collective_breakthrough_count[24h])

# Breakthroughs this month
increase(agents_collective_breakthrough_count[30d])
```

**Alert Rules:**

```yaml
- alert: BreakthroughDiscovered
  expr: increase(agents_collective_breakthrough_count[1h]) > 0
  annotations:
    summary: "Collective achieved {{ $value }} breakthrough(s)"
```

---

### Knowledge Sharing

#### `agents_knowledge_sharing_events` (Counter)

**Type:** Counter  
**Labels:** `source_tier`, `target_tier`  
**Help:** Cross-tier knowledge sharing

**Knowledge Flow Patterns:**

```
Tier 1 → Tier 2:  Foundational patterns shared with specialists
Tier 2 → Tier 1:  Specialized insights feed back to foundations
Tier 1 → Tier 8:  Enterprise governance patterns
```

**Example Queries:**

```promql
# Knowledge flow matrix
sum by(source_tier, target_tier) (rate(agents_knowledge_sharing_events[24h]))

# Tier 1 → others
sum by(target_tier) (rate(agents_knowledge_sharing_events{source_tier="1"}[24h]))

# Bidirectional knowledge flow
agents_knowledge_sharing_events{source_tier="1", target_tier="2"} +
agents_knowledge_sharing_events{source_tier="2", target_tier="1"}
```

---

## System Information

#### `agents_system` (Info)

**Type:** Info  
**Labels:** `version`, `total_agents`, `total_tiers`  
**Help:** Elite Agent Collective system information

**Usage:**

```python
system_info.info({
    'version': '2.0',
    'total_agents': '40',
    'total_tiers': '8',
    'architecture': 'multi-tier-collective',
    'memory_system': 'MNEMONIC'
})
```

---

#### `agents_info` (Info)

**Type:** Info  
**Labels:** `agent_id`, `agent_name`, `tier`, `specialization`  
**Help:** Individual agent information

**Usage:**

```python
agent_info.info({
    'agent_id': '01',
    'agent_name': 'APEX',
    'tier': '1',
    'specialization': 'computer_science_engineering',
    'description': 'Elite computer science engineering'
})
```

---

## Usage Examples

### 1. Individual Agent Monitoring

```python
from agents.monitoring.metrics import (
    agents_tasks_total,
    agents_tasks_completed,
    agents_task_duration_seconds,
    agents_success_rate,
    agent_active_tasks
)
import time

class AgentExecutor:
    async def execute_task(self, agent_id: str, agent_name: str, task_type: str, task_func):
        start_time = time.time()

        # Track start
        agents_tasks_total.labels(agent_id=agent_id, agent_name=agent_name, task_type=task_type).inc()
        agent_active_tasks.labels(agent_id=agent_id, agent_name=agent_name).inc()

        try:
            result = await task_func()

            # Track success
            agents_tasks_completed.labels(agent_id=agent_id, agent_name=agent_name, task_type=task_type).inc()

            return result
        except Exception as e:
            raise
        finally:
            # Record duration
            duration = time.time() - start_time
            agents_task_duration_seconds.labels(
                agent_id=agent_id,
                agent_name=agent_name,
                task_type=task_type
            ).observe(duration)

            # Update active tasks
            agent_active_tasks.labels(agent_id=agent_id, agent_name=agent_name).dec()
```

### 2. Collective Health Dashboard

```python
from prometheus_client import CollectorRegistry
import json

def get_collective_health():
    """Return collective health summary"""
    return {
        'total_agents': 40,
        'healthy': agents_collective_healthy.get(),
        'degraded': agents_collective_degraded.get(),
        'failed': agents_collective_failed.get(),
        'active_tasks': agents_collective_active_tasks.get(),
        'utilization': agents_collective_utilization_ratio.get(),
        'success_rate': agents_collective_success_rate.get(),
        'intelligence_score': agents_collective_intelligence_score.get(),
        'breakthroughs_today': increase(agents_collective_breakthrough_count[24h])
    }
```

### 3. Tier-Level Monitoring

```python
def monitor_tier(tier: str):
    """Monitor specific tier"""
    return {
        'tier': tier,
        'health_score': agents_tier_health_score.labels(tier=tier).get(),
        'utilization': agents_tier_utilization_ratio.labels(tier=tier).get(),
        'error_rate': agents_tier_error_rate.labels(tier=tier).get(),
        'task_throughput': rate(agents_tier_task_count[1m]).labels(tier=tier)
    }
```

### 4. Collaboration Analysis

```python
def analyze_collaboration():
    """Analyze inter-agent collaboration patterns"""
    return {
        'total_collaboration_events': increase(agents_collaboration_events_total[24h]),
        'total_handoffs': increase(agents_handoff_total[24h]),
        'avg_communication_latency': avg(agents_communication_latency_seconds),
        'collaboration_efficiency': (
            increase(agents_collaboration_events_total[24h]) /
            increase(agents_tasks_total[24h])  # Collaboration per task
        ),
        'knowledge_sharing_events': increase(agents_knowledge_sharing_events[24h])
    }
```

---

## Prometheus Queries Reference

### Individual Agent Queries

```promql
# 1. Agent status summary
agents_status by (agent_name, tier)

# 2. Agent success rate
agents_success_rate by (agent_name)

# 3. Average task duration by agent
rate(agents_task_duration_seconds_sum[5m]) / rate(agents_task_duration_seconds_count[5m]) by (agent_name)

# 4. P95 latency per agent
histogram_quantile(0.95, rate(agents_task_duration_seconds_bucket[5m])) by (agent_name)

# 5. Agent utilization
agents_utilization_ratio by (agent_name)

# 6. Task queue depth
agents_queue_length by (agent_name)

# 7. Agent error rate
agents_error_rate by (agent_name)

# 8. Agent timeout rate
agents_timeout_rate by (agent_name)

# 9. Mean time to recovery
agents_mttr_seconds by (agent_name)

# 10. Availability ratio
agents_availability_ratio by (agent_name)
```

### Collective Queries

```promql
# 11. Collective health percentage
(agents_collective_healthy / agents_collective_total) * 100

# 12. Failed agents
agents_collective_failed

# 13. Total active tasks
agents_collective_active_tasks

# 14. Collective throughput (tasks/sec)
rate(agents_tasks_completed[1m])

# 15. Collective success rate
agents_collective_success_rate

# 16. Collective error rate
agents_collective_error_rate

# 17. Average utilization
avg(agents_utilization_ratio)

# 18. Peak utilization
max(agents_utilization_ratio)

# 19. Collective intelligence score
agents_collective_intelligence_score

# 20. Breakthrough discovery rate
rate(agents_collective_breakthrough_count[24h])
```

### Tier Queries

```promql
# 21. Tier health scores
agents_tier_health_score by (tier)

# 22. Tier utilization
agents_tier_utilization_ratio by (tier)

# 23. Tier error rates
agents_tier_error_rate by (tier)

# 24. Task distribution across tiers
sum by(tier) (rate(agents_tier_task_count[1h]))

# 25. Tier 1 health
agents_tier_health_score{tier="1"}
```

### Collaboration Queries

```promql
# 26. Collaboration event rate
rate(agents_collaboration_events_total[1m])

# 27. Handoff frequency
rate(agents_handoff_total[1h])

# 28. Most collaborative pairs
topk(20, increase(agents_collaboration_events_total[24h]))

# 29. Average communication latency
rate(agents_communication_latency_seconds_sum[5m]) / rate(agents_communication_latency_seconds_count[5m])

# 30. P95 communication latency
histogram_quantile(0.95, rate(agents_communication_latency_seconds_bucket[5m]))

# 31. Knowledge sharing events
rate(agents_knowledge_sharing_events[24h])

# 32. Tier-to-tier knowledge flow
sum by(source_tier, target_tier) (increase(agents_knowledge_sharing_events[24h]))

# 33. Bidirectional communication between tiers
(agents_collaboration_events_total{source_tier="1", target_tier="2"} +
 agents_collaboration_events_total{source_tier="2", target_tier="1"})

# 34. Agent specialization distribution
agents_specialization by (specialization)

# 35. Most specialized agents
max by(agent_name) (agents_specialization)
```

---

## Alert Rules

### Agent Health Alerts

```yaml
groups:
  - name: agents.health
    interval: 30s
    rules:
      - alert: AgentDown
        expr: agents_status == 2
        for: 2m
        annotations:
          summary: "Agent {{ $labels.agent_name }} ({{ $labels.agent_id }}) is down"
          description: "Agent status: {{ $value }}"

      - alert: AgentDegraded
        expr: agents_status == 1
        for: 5m
        annotations:
          summary: "Agent {{ $labels.agent_name }} degraded"

      - alert: LowAgentSuccessRate
        expr: agents_success_rate < 0.95
        for: 30m
        annotations:
          summary: "Agent {{ $labels.agent_name }} success rate: {{ $value | humanizePercentage }}"

      - alert: HighAgentErrorRate
        expr: agents_error_rate > 0.1
        for: 10m
        annotations:
          summary: "Agent {{ $labels.agent_name }} error rate: {{ $value | humanizePercentage }}"

      - alert: HighTimeoutRate
        expr: agents_timeout_rate > 0.05
        for: 5m
        annotations:
          summary: "Agent {{ $labels.agent_name }} timeout rate: {{ $value | humanizePercentage }}"
```

### Performance Alerts

```yaml
- name: agents.performance
  interval: 1m
  rules:
    - alert: SlowTaskExecution
      expr: histogram_quantile(0.95, rate(agents_task_duration_seconds_bucket[5m])) > 60
      for: 10m
      annotations:
        summary: "Agent {{ $labels.agent_name }} P95 latency > 60s: {{ $value }}s"

    - alert: TaskQueueBacklog
      expr: agents_queue_length / agents_max_capacity > 3
      for: 5m
      annotations:
        summary: "Agent {{ $labels.agent_name }} has large queue backlog"

    - alert: LowUtilization
      expr: avg(agents_utilization_ratio) < 0.2
      for: 1h
      annotations:
        summary: "Collective utilization < 20%: {{ $value | humanizePercentage }}"

    - alert: HighUtilization
      expr: max(agents_utilization_ratio) > 0.9
      for: 10m
      annotations:
        summary: "Agent {{ $labels.agent_name }} utilization > 90%: {{ $value | humanizePercentage }}"
```

### Collective Alerts

```yaml
- name: agents.collective
  interval: 2m
  rules:
    - alert: CollectiveHealthDegraded
      expr: agents_collective_failed > 2
      for: 5m
      annotations:
        summary: "{{ $value }} agents in collective have failed"

    - alert: CriticalCollectiveFailure
      expr: (agents_collective_healthy / agents_collective_total) < 0.8
      for: 2m
      annotations:
        summary: "Collective health below 80%: {{ $value | humanizePercentage }}"

    - alert: CollectiveErrorRateHigh
      expr: agents_collective_error_rate > 0.05
      for: 10m
      annotations:
        summary: "Collective error rate: {{ $value | humanizePercentage }}"

    - alert: LowCollectiveIntelligence
      expr: agents_collective_intelligence_score < 0.5
      for: 30m
      annotations:
        summary: "Collective intelligence score low: {{ $value }}"
```

### Resilience Alerts

```yaml
- name: agents.resilience
  interval: 5m
  rules:
    - alert: FrequentRestarts
      expr: increase(agents_restart_count[1h]) > 5
      for: 5m
      annotations:
        summary: "Agent {{ $labels.agent_name }} restarted 5+ times in 1h"

    - alert: HighMeanTimeToRecovery
      expr: agents_mttr_seconds > 300
      for: 30m
      annotations:
        summary: "Agent {{ $labels.agent_name }} MTTR > 5min: {{ $value }}s"

    - alert: LowAvailability
      expr: agents_availability_ratio < 0.99
      for: 1h
      annotations:
        summary: "Agent {{ $labels.agent_name }} availability < 99%: {{ $value | humanizePercentage }}"
```

---

## Multi-Agent Monitoring Patterns

### 1. Tier-Level Monitoring

```python
def monitor_all_tiers():
    """Monitor all 8 tiers comprehensively"""
    tiers = ['1', '2', '3', '4', '5', '6', '7', '8']
    results = {}

    for tier in tiers:
        results[tier] = {
            'health': agents_tier_health_score.labels(tier=tier).get(),
            'utilization': agents_tier_utilization_ratio.labels(tier=tier).get(),
            'error_rate': agents_tier_error_rate.labels(tier=tier).get(),
            'agent_count': len(get_tier_agents(tier))
        }

    return results
```

### 2. Cross-Tier Collaboration Pattern

```python
def track_cross_tier_collaboration():
    """Monitor knowledge flow between tiers"""
    collaboration_matrix = {}

    for source_tier in range(1, 9):
        for target_tier in range(1, 9):
            key = f"Tier{source_tier}→Tier{target_tier}"
            collaboration_matrix[key] = increase(
                agents_knowledge_sharing_events{
                    source_tier=source_tier,
                    target_tier=target_tier
                }[24h]
            )

    return collaboration_matrix
```

### 3. Agent Collaboration Network

```python
def build_collaboration_graph():
    """Build collaboration network for analysis"""
    edges = []

    for event in agents_collaboration_events_total:
        initiator = event.labels['initiator_agent']
        target = event.labels['target_agent']
        weight = event.value

        edges.append({
            'from': initiator,
            'to': target,
            'weight': weight
        })

    return {
        'nodes': list(ALL_AGENTS),
        'edges': edges
    }
```

### 4. Capacity Planning

```python
def plan_collective_capacity():
    """Plan for future growth"""
    current_throughput = rate(agents_tasks_completed[1m])
    current_utilization = avg(agents_utilization_ratio)

    # Forecast 6 months
    growth_factor = 1.2  # 20% growth
    future_throughput = current_throughput * growth_factor
    future_utilization = current_utilization * growth_factor

    if future_utilization > 0.8:
        recommend_scaling()

    return {
        'current_throughput': current_throughput,
        'future_throughput': future_throughput,
        'scaling_needed': future_utilization > 0.8
    }
```

---

## Troubleshooting Guide

### Agent Consistently Failing

**Symptoms:** Agent success rate < 90%, error rate > 10%

**Diagnostic:**

```promql
# Check error distribution
rate(agents_tasks_failed[5m]) by (error_type, agent_name)

# Check task types causing failures
increase(agents_tasks_failed[1h]) by (task_type)

# Check agent resource usage
agents_utilization_ratio{agent_name="TARGET_AGENT"}
```

**Solutions:**

1. Check agent logs for error details
2. Verify dependencies are available
3. Increase agent capacity if over-utilized
4. Review recent task types assigned

### Collective Throughput Degradation

**Symptoms:** Throughput < baseline by 30%

**Diagnostic:**

```promql
# Throughput by tier
sum by(tier) (rate(agents_tier_task_count[1m]))

# Queue depth across collective
sum(agents_queue_length)

# Agent availability
count(agents_status == 0) / count(agents_status)
```

**Solutions:**

1. Identify failed/degraded agents
2. Restart unhealthy agents
3. Rebalance tasks to healthy agents
4. Scale up if structurally under-capacity

### High Latency

**Symptoms:** P95 latency > 60s

**Diagnostic:**

```promql
# Task duration distribution
histogram_quantile(0.99, rate(agents_task_duration_seconds_bucket[5m])) by (agent_name)

# Queue wait time
avg(agents_queue_length) by (agent_name)

# Agent specialization fit
agents_specialization by (agent_id) - histogram_quantile(0.95, rate(agents_task_duration_seconds_bucket[5m]))
```

**Solutions:**

1. Ensure tasks assigned to appropriate specializations
2. Increase agent parallelism
3. Optimize task routing algorithm
4. Consider task decomposition

### Collective Intelligence Decline

**Symptoms:** Intelligence score declining, fewer breakthroughs

**Diagnostic:**

```promql
# Collaboration events
rate(agents_collaboration_events_total[1h])

# Knowledge sharing
rate(agents_knowledge_sharing_events[1h])

# Breakthrough rate
rate(agents_collective_breakthrough_count[24h])

# Tier interaction
increase(agents_knowledge_sharing_events[24h]) by (source_tier, target_tier)
```

**Solutions:**

1. Review tier collaboration patterns
2. Promote successful collaboration patterns
3. Ensure diverse task allocation
4. Verify memory system (MNEMONIC) is functioning

---

## Dashboard Guide

### Recommended Dashboard Sections

1. **Collective Overview**
   - Health status pie chart
   - Total active tasks gauge
   - Throughput (tasks/sec)
   - Collective intelligence score

2. **Agent Status Matrix**
   - Table: All 40 agents with status, utilization, success rate
   - Sort/filter by tier, status, utilization

3. **Performance Trends**
   - Throughput over time
   - Success rate over time
   - Error rate trend
   - Latency (P50, P95, P99)

4. **Tier Analysis**
   - Bar chart: Tier health scores
   - Line chart: Tier utilization
   - Heatmap: Tier error rates

5. **Collaboration Network**
   - Force-directed graph of agent interactions
   - Edge width = collaboration frequency

6. **Capacity Planning**
   - Utilization trend (24h, 7d, 30d)
   - Queue depth trend
   - Throughput trend with growth projection

---

## Integration Checklist

- [ ] Deploy Prometheus scrape config for agents endpoint (`:9092/metrics`)
- [ ] Configure retention policy (minimum 30 days for trend analysis)
- [ ] Create Grafana dashboards (see GRAFANA_DASHBOARD_GUIDE.md)
- [ ] Set up alert rules (copy from Alert Rules section)
- [ ] Configure notification channels (Slack/PagerDuty)
- [ ] Test metrics collection in staging
- [ ] Document agent ID to agent name mapping
- [ ] Train team on interpreting collective intelligence metrics
- [ ] Set up weekly collective health reports
- [ ] Establish baseline performance metrics

---

## References

- [Prometheus Query Language](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [Agent Collective Documentation](../../../agents/README.md)
- [MNEMONIC Memory System](../../../docs/MNEMONIC_ARCHITECTURE.md)
- [Grafana Dashboard Design](https://grafana.com/docs/grafana/latest/dashboards/)

---

**Version:** 2.0  
**Last Updated:** December 16, 2025  
**Maintained By:** @SCRIBE
