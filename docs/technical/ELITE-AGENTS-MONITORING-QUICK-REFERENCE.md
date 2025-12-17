# Elite Agent Collective Monitoring - Quick Reference

## 🚀 TL;DR - Get Started in 5 Minutes

### 1. Import the Client Library

```python
from neurectomy.agents.monitoring import MetricsClient, get_metrics
```

### 2. Initialize in Your Agent

```python
class MyAgent(BaseAgent):
    def __init__(self, agent_id, agent_name, tier, specialization):
        # ... agent initialization ...
        self.metrics = MetricsClient(agent_id, agent_name, tier, specialization)
```

### 3. Record Events

```python
# Task events
self.metrics.record_task_assigned(task_type="analysis")
self.metrics.record_task_completed(task_type="analysis", duration=2.5, success=True)

# Status updates
self.metrics.update_status(AgentStatus.ACTIVE)
self.metrics.update_utilization(active_count=5, max_capacity=10)

# Collaboration
self.metrics.record_handoff_event(to_agent_id="apex", specialization_match_score=0.95)
```

### 4. Let Batch Processing Handle the Rest

```python
# No explicit flushing needed - batch flush happens:
# - When batch reaches 100 items (default)
# - Every 60 seconds (default)
# - Supervisor heartbeat triggers flush
```

### 5. Query Your Metrics

```python
from neurectomy.agents.monitoring import query_agent_status, query_tier_performance

# Single agent status
status = query_agent_status("apex")
print(f"APEX health: {status['health']}, utilization: {status['utilization']}")

# Tier performance
tier_stats = query_tier_performance("tier_1")
print(f"Tier 1 success rate: {tier_stats['success_rate']}")
```

---

## 📋 Metric Recording Methods

### Task Events

```python
# When task is assigned
metrics.record_task_assigned(task_type, priority=0, estimated_duration=None)

# When task completes successfully
metrics.record_task_completed(task_type, duration, success=True, result_quality=1.0)

# When task fails
metrics.record_task_failed(task_type, error_type, duration, retry_count=0)

# When task times out
metrics.record_task_timeout(task_type, duration)
```

### Health Status

```python
# Update status (ACTIVE, IDLE, RECOVERING, ERROR)
metrics.update_status(status_code)

# Record recovery from error
metrics.record_recovery_event(recovery_type, recovery_duration=0)

# Update utilization
metrics.update_utilization(active_task_count, max_concurrent_capacity)

# Update queue
metrics.update_queue_length(queue_size)

# Update idle percentage
metrics.update_idle_percentage(idle_percent)
```

### Collaboration

```python
# Record task handoff to another agent
metrics.record_handoff_event(to_agent_id, specialization_match_score=0.0)

# Record knowledge sharing
metrics.record_knowledge_share(to_agent_id, knowledge_type, quality=1.0)

# Update collaboration score
metrics.update_collaboration_score(partner_agent_id, score)
```

### Meta-Intelligence

```python
# Record breakthrough discovery
metrics.record_breakthrough_discovery(discovery_type, fitness_score=0.9)

# Update learning rate
metrics.update_learning_rate(improvement_percent)

# Update MNEMONIC memory fitness
metrics.update_memory_fitness(fitness_score, retrieval_type)

# Record memory insight
metrics.record_memory_insight(insight_type, quality, relevance)
```

---

## 🔍 Query Templates

### Individual Agent Metrics

```python
from neurectomy.agents.monitoring import PrometheusQueries

# Get preconfigured queries
queries = PrometheusQueries()

# Agent health
query = queries.agent_health("apex")
# Result: 'agent_status{agent_id="apex"} == 1'

# Agent success rate
query = queries.agent_success_rate("apex")
# Result: 'agent_task_success_rate{agent_id="apex"}'

# Agent utilization
query = queries.agent_utilization("apex")
# Result: 'agent_utilization_ratio{agent_id="apex"}'
```

### Tier Metrics

```python
# Tier utilization
query = queries.tier_utilization("tier_1")
# Result: 'avg(agent_utilization_ratio{tier="tier_1"})'

# Tier success rate
query = queries.tier_success_rate("tier_1")
# Result: 'avg(agent_task_success_rate{tier="tier_1"})'

# Tier health
query = queries.tier_health("tier_1")
# Result: 'tier_health_score{tier="tier_1"}'
```

### Collective Metrics

```python
# Collective health
query = queries.collective_health()
# Result: 'collective_intelligence_score'

# Collective utilization
query = queries.collective_utilization()
# Result: 'avg(agent_utilization_ratio)'

# Collective throughput
query = queries.collective_throughput()
# Result: 'sum(rate(agent_tasks_completed_total[5m]))'
```

---

## ⚠️ Common Issues & Fixes

### Issue 1: Metrics Not Appearing in Prometheus

**Symptoms**: Prometheus shows no data, or "No data"

**Diagnosis**:

```bash
# Check metrics endpoint
curl http://localhost:8000/metrics | grep agent_

# Check Prometheus targets
http://localhost:9090/targets
```

**Fix**:

1. Ensure agent is calling `metrics.record_*()` methods
2. Verify batch flushing is enabled (supervisor heartbeat)
3. Check Prometheus scrape interval (default 15s)
4. Review logs for errors: `grep -i metric /path/to/logs`

### Issue 2: High Memory Usage from Metrics

**Symptoms**: Metrics process memory growing rapidly

**Diagnosis**:

```python
# Check pending updates
client = get_client("apex")
pending = client._pending_updates
print(f"Pending updates: {sum(len(v) for v in pending.values())}")
```

**Fix**:

1. Reduce batch size: `metrics.set_batch_size(50)`
2. Increase flush frequency: `metrics.set_flush_interval(30)`
3. Ensure supervisor is triggering flushes
4. Check for agents not being cleaned up

### Issue 3: Cardinality Explosion

**Symptoms**: Prometheus disk usage growing fast, queries slow

**Cause**: Unbounded label values (task types, error types)

**Fix**:

1. Standardize task types to known set
2. Normalize error types
3. Use label dropping for high-cardinality metrics:

```yaml
# In prometheus.yml
scrape_configs:
  - job_name: "neurectomy-metrics"
    metric_relabel_configs:
      - source_labels: [__name__]
        regex: "(task_type|error_type)_.*"
        action: drop
```

### Issue 4: Alert Not Firing

**Symptoms**: Alert condition met but no alert

**Diagnosis**:

1. Check alert rule syntax: `http://localhost:9090/rules`
2. Verify metrics exist: `http://localhost:9090/graph`
3. Check AlertManager status: `http://localhost:9093`

**Fix**:

1. Verify metric names match exactly
2. Check label names (case-sensitive)
3. Verify AlertManager notification config
4. Test alert rule: `promtool check rules alert_rules.yml`

### Issue 5: Agent Metrics Spike/Abnormal Values

**Symptoms**: Metric suddenly very high or very low

**Diagnosis**:

```python
# Check recent metric history
snap = client.get_metrics_snapshot()
print(json.dumps(snap, indent=2))

# Review batch pending
print(f"Pending: {client._pending_updates}")
```

**Fix**:

1. Check if many tasks failed simultaneously
2. Verify agent status didn't change unexpectedly
3. Review agent logs for errors
4. Check if batch flushing working (should see metrics every 60s)

---

## 🎯 Integration Patterns

### Pattern 1: Task Tracking

```python
async def execute_task(self, task):
    start_time = time.time()
    self.metrics.record_task_assigned(task.type)

    try:
        result = await task.execute()
        duration = time.time() - start_time
        self.metrics.record_task_completed(
            task_type=task.type,
            duration=duration,
            success=True,
            result_quality=self.evaluate_quality(result)
        )
        return result
    except TimeoutError:
        duration = time.time() - start_time
        self.metrics.record_task_timeout(task.type, duration)
        raise
    except Exception as e:
        duration = time.time() - start_time
        self.metrics.record_task_failed(
            task_type=task.type,
            error_type=type(e).__name__,
            duration=duration,
            retry_count=task.retry_count
        )
        raise
```

### Pattern 2: Status Updates

```python
async def heartbeat(self):
    # Update current status
    self.metrics.update_status(self.current_status)

    # Update utilization
    active_tasks = len(self.task_queue.active)
    max_capacity = self.config.max_concurrent_tasks
    self.metrics.update_utilization(active_tasks, max_capacity)

    # Update queue
    pending_tasks = len(self.task_queue.pending)
    self.metrics.update_queue_length(pending_tasks)

    # Idle percentage
    if max_capacity > 0:
        idle_pct = ((max_capacity - active_tasks) / max_capacity) * 100
        self.metrics.update_idle_percentage(idle_pct)

    # Explicit flush
    self.metrics.flush_pending_updates()
```

### Pattern 3: Collaboration Tracking

```python
async def handoff_task(self, task, target_agent_id):
    # Find target agent to calculate match
    target = self.registry.get(target_agent_id)
    match_score = self.calculate_specialization_match(task, target)

    # Record handoff
    self.metrics.record_handoff_event(target_agent_id, match_score)

    # Hand off task
    await target.receive_task(task)

    # Track if knowledge needs sharing
    if task.complexity > 0.7:
        self.metrics.record_knowledge_share(
            target_agent_id,
            knowledge_type="complex_task_guidance",
            quality=0.9
        )
```

### Pattern 4: Learning & Breakthroughs

```python
def on_task_success(self, task, result):
    # Update learning
    improvement = self.calculate_improvement(task, result)
    self.metrics.update_learning_rate(improvement * 100)

    # Check if breakthrough
    if self.is_breakthrough(result):
        fitness_score = self.calculate_fitness(result)
        self.metrics.record_breakthrough_discovery(
            discovery_type="algorithm_optimization",
            fitness_score=fitness_score
        )
        self.logger.info(f"Breakthrough! Fitness: {fitness_score}")

def on_memory_retrieval(self, retrieval):
    # Update memory metrics
    self.metrics.update_memory_fitness(
        fitness_score=retrieval.fitness,
        retrieval_type="experience_recall"
    )

    # Record insight if valuable
    if retrieval.quality > 0.8:
        self.metrics.record_memory_insight(
            insight_type=retrieval.type,
            quality=retrieval.quality,
            relevance=retrieval.relevance
        )
```

---

## 🔧 Configuration Options

### Client-Side (MetricsClient)

```python
client = MetricsClient(agent_id, agent_name, tier, specialization)

# Configure batch processing
client.set_batch_size(100)        # Default: 100 items
client.set_flush_interval(60)     # Default: 60 seconds

# Manual operations
updates = client.flush_pending_updates()  # Force flush
snapshot = client.get_metrics_snapshot()  # Get current state
```

### Prometheus (docker/prometheus/prometheus.yml)

```yaml
global:
  scrape_interval: 15s # How often to scrape
  evaluation_interval: 30s # How often to evaluate rules

scrape_configs:
  - job_name: "neurectomy-metrics"
    static_configs:
      - targets: ["localhost:8000"]
```

### Prometheus Retention

```yaml
# Keep raw data for 15 days
--storage.tsdb.retention.time=15d

# Keep 50GB of data
--storage.tsdb.retention.size=50GB
```

### Grafana Refresh

```
Refresh interval: 30 seconds (edit dashboard)
```

---

## 📊 Metric Glossary

| Metric                                 | Type    | Description                                              |
| -------------------------------------- | ------- | -------------------------------------------------------- |
| `agent_status`                         | Gauge   | Current status (0=ERROR, 1=ACTIVE, 2=IDLE, 3=RECOVERING) |
| `agent_tasks_assigned_total`           | Counter | Total tasks assigned                                     |
| `agent_tasks_completed_total`          | Counter | Total tasks completed                                    |
| `agent_utilization_ratio`              | Gauge   | Active tasks / max capacity (0-1)                        |
| `agent_task_success_rate`              | Gauge   | Success rate (0-1)                                       |
| `agent_task_failure_rate`              | Gauge   | Failure rate (0-1)                                       |
| `collective_intelligence_score`        | Gauge   | Overall system health (0-100)                            |
| `agent_breakthrough_discoveries_total` | Counter | Breakthrough count                                       |
| `agent_learning_rate`                  | Gauge   | Learning improvement percentage                          |
| `tier_health_score`                    | Gauge   | Per-tier health (0-100)                                  |

---

## 💾 Exporting & Archiving

### Export to Prometheus Format

```python
from neurectomy.agents.monitoring import get_metrics

metrics = get_metrics()
prometheus_text = metrics.export_metrics()
print(prometheus_text)

# Save to file
with open("metrics_export.txt", "w") as f:
    f.write(prometheus_text)
```

### Query Prometheus API

```bash
# Get metric value
curl 'http://localhost:9090/api/v1/query?query=agent_status{agent_id="apex"}'

# Get metric range
curl 'http://localhost:9090/api/v1/query_range?query=agent_utilization_ratio&start=2024-01-01T00:00:00Z&end=2024-01-02T00:00:00Z&step=60s'
```

---

## 🚨 Production Checklist

- [ ] All agents initialized with MetricsClient
- [ ] Metrics recorded for all task events
- [ ] Status updates at regular intervals
- [ ] Collaboration events being tracked
- [ ] Batch flushing enabled (supervisor heartbeat)
- [ ] Prometheus targets configured correctly
- [ ] Alert rules imported
- [ ] Grafana dashboards imported
- [ ] Notification channels configured
- [ ] Retention policies set appropriately
- [ ] Disk space sufficient for metrics (~1% of main data)
- [ ] Performance overhead verified (<1%)
- [ ] Monitoring dashboards accessible
- [ ] On-call procedures documented

---

## 📞 Getting Help

**For integration help**: See [ELITE-AGENTS-INTEGRATION-GUIDE.md](ELITE-AGENTS-INTEGRATION-GUIDE.md)

**For troubleshooting**: See troubleshooting section in integration guide or this document

**For metric definitions**: See [ELITE-AGENTS-METRICS-DESIGN.md](ELITE-AGENTS-METRICS-DESIGN.md)

**For API reference**: See `neurectomy/agents/monitoring/client.py` docstrings

**For implementation**: See `neurectomy/agents/monitoring/metrics.py` source code
