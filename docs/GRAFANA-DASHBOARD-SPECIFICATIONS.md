# GRAFANA DASHBOARD SPECIFICATIONS

# Neurectomy Phase 18A - 5 Production-Ready Dashboards

## DASHBOARD 1: Ryot LLM Metrics

## Location: /d/ryot-llm-metrics

## Refresh: 30s | Time Range: 1h

### Panel 1: Inference Latency P50

Type: Gauge
Query: `ryot:inference_latency:p50{model="default"}`
Threshold: [0, 100ms]
Unit: ms

### Panel 2: Inference Latency P95

Type: Gauge
Query: `ryot:inference_latency:p95{model="default"}`
Threshold: [0, 500ms]
Unit: ms

### Panel 3: Inference Latency P99

Type: Gauge
Query: `ryot:inference_latency:p99{model="default"}`
Threshold: [0, 2000ms]
Unit: ms

### Panel 4: Latency Distribution Over Time

Type: Graph with Stacked Areas
Query1: `ryot:inference_latency:p50`
Query2: `ryot:inference_latency:p95`
Query3: `ryot:inference_latency:p99`
Legend: ["P50", "P95", "P99"]
Y-Axis: Latency (ms)

### Panel 5: Time To First Token (TTFT) Distribution

Type: Histogram
Query: `histogram_quantile(0.95, sum(rate(neurectomy_ryot_ttft_seconds_bucket[5m])) by (le))`
Title: TTFT P95
Unit: s

### Panel 6: Token Generation Rate

Type: Graph
Query: `ryot:tokens_per_second`
Legend: ["Tokens/sec"]
Y-Axis: Tokens per second

### Panel 7: Tokens Generated (Total)

Type: Stat
Query: `neurectomy_ryot_tokens_generated_total`
Unit: short

### Panel 8: GPU Memory Usage

Type: Gauge
Query: `neurectomy_ryot_gpu_memory_usage_bytes / neurectomy_ryot_gpu_memory_total_bytes * 100`
Threshold: [0, 50, 80, 100]
Unit: percent

### Panel 9: Batch Size Distribution

Type: Histogram
Query: `histogram_quantile(0.95, sum(rate(neurectomy_ryot_batch_size_bucket[5m])) by (le))`
Title: Batch Size (P95)

### Panel 10: Request Throughput

Type: Graph
Query: `ryot:throughput:requests_per_minute`
Legend: ["Success", "Error", "Timeout"]
Y-Axis: Requests/min

### Panel 11: Error Rate

Type: Stat
Query: `tier2:ryot_error_rate * 100`
Unit: percent
Alert Threshold: 5%

### Panel 12: GPU OOM Risk Indicator

Type: Alert Status
Query: `neurectomy_ryot_gpu_memory_usage_bytes / neurectomy_ryot_gpu_memory_total_bytes > 0.9`

---

## DASHBOARD 2: ΣLANG Compression Metrics

## Location: /d/sigmalang-compression

## Refresh: 30s | Time Range: 1h

### Panel 1: Compression Ratio (Gauge)

Type: Gauge
Query: `tier2:sigmalang_compression_ratio:avg`
Threshold: [1, 5, 10, 30]
Unit: x

### Panel 2: Compression Ratio Over Time

Type: Graph
Query: `avg(neurectomy_sigmalang_compression_ratio)`
Y-Axis: Compression Ratio (x)

### Panel 3: Compression Ratio by Method

Type: Bar Chart
Query: `avg(neurectomy_sigmalang_compression_ratio) by (method)`
Legend: ["DEFLATE", "LZMA", "ZSTD", "Hybrid"]

### Panel 4: Incompressible Data Percentage

Type: Gauge
Query: `tier2:sigmalang_incompressible_bytes_ratio`
Unit: percent
Threshold: [0, 25, 50, 100]

### Panel 5: Cache Hit Ratio

Type: Gauge
Query: `sigmalang:cache_hit_ratio * 100`
Unit: percent
Threshold: [0, 50, 75, 100]

### Panel 6: Cache Performance Over Time

Type: Graph
Query1: `sum(rate(neurectomy_sigmalang_cache_hits_total[5m]))`
Query2: `sum(rate(neurectomy_sigmalang_cache_misses_total[5m]))`
Legend: ["Hits", "Misses"]
Y-Axis: Events per second

### Panel 7: Throughput (Bytes/sec)

Type: Stat
Query: `sigmalang:throughput:bytes_per_second`
Unit: Bps

### Panel 8: Throughput Trend

Type: Graph
Query: `sigmalang:throughput:bytes_per_second`
Y-Axis: Throughput (Bps)

### Panel 9: Sub-Linear Speedup

Type: Gauge
Query: `sigmalang:sub_linear_speedup`
Unit: x
Threshold: [1, 2, 5, 10]

### Panel 10: Compression Request Rate

Type: Stat
Query: `sum(rate(neurectomy_sigmalang_compression_requests_total[5m])) by (status)`

### Panel 11: Success vs Failure Rate

Type: Pie Chart
Query: `sum(rate(neurectomy_sigmalang_compression_requests_total[5m])) by (status)`
Legend: ["Success", "Error"]

### Panel 12: Optimization Opportunities

Type: Table
Query: `GROUP BY (method) HAVING avg(neurectomy_sigmalang_compression_ratio) < 5`
Columns: ["Method", "Avg Ratio", "Action Required"]

---

## DASHBOARD 3: ΣVAULT Storage Metrics

## Location: /d/sigmavault-storage

## Refresh: 1m | Time Range: 30d

### Panel 1: Capacity Utilization

Type: Gauge
Query: `tier2:sigmavault_capacity_utilization:percent`
Unit: percent
Threshold: [0, 50, 80, 95, 100]

### Panel 2: Available Capacity

Type: Gauge
Query: `tier2:sigmavault_available_capacity:percent`
Unit: percent

### Panel 3: Capacity Over Time

Type: Area Graph
Query1: `sum(neurectomy_sigmavault_used_bytes)`
Query2: `sum(neurectomy_sigmavault_total_bytes)`
Legend: ["Used", "Total"]
Unit: bytes

### Panel 4: Monthly Cost Trend

Type: Graph
Query: `neurectomy_sigmavault_cost_usd`
Y-Axis: Cost (USD)
Series: ["Current Month"]

### Panel 5: Cost Projection

Type: Stat
Query: `costs:monthly_projection`
Unit: percent
Suffix: "% of Budget"

### Panel 6: Cost by Storage Class

Type: Pie Chart
Query: `sum(neurectomy_sigmavault_cost_usd) by (storage_class)`
Legend: ["Standard", "Infrequent", "Archive"]

### Panel 7: SLA Compliance

Type: Gauge
Query: `tier2:sigmavault_sla_compliance:percent`
Unit: percent
Threshold: [99.0, 99.5, 99.9, 99.99]

### Panel 8: Replication Lag

Type: Stat
Query: `tier2:sigmavault_replication_lag:seconds`
Unit: s

### Panel 9: Read Latency P95

Type: Gauge
Query: `tier2:sigmavault_read_latency:p95`
Unit: ms
Threshold: [0, 50, 100]

### Panel 10: Write Latency P95

Type: Gauge
Query: `tier2:sigmavault_write_latency:p95`
Unit: ms
Threshold: [0, 50, 100]

### Panel 11: Operations Per Minute by Type

Type: Bar Chart
Query: `sum(rate(neurectomy_sigmavault_operations_total[1m])) by (operation)`
Legend: ["Read", "Write", "Delete", "List"]

### Panel 12: Cost Per GB (Monthly)

Type: Stat
Query: `tier2:sigmavault_cost_per_gb_month`
Unit: "$"

---

## DASHBOARD 4: Elite Agent Collective

## Location: /d/agent-collective

## Refresh: 30s | Time Range: 1h

### Panel 1: Collective Health Heatmap (8 Tiers)

Type: Heatmap
Query: `neurectomy_agent_health_score`
Legend: Tier 1-8 on Y-axis
Colors: Green (>0.9) → Yellow (0.7-0.9) → Red (<0.7)

### Panel 2: Agent Status Grid (40 Agents)

Type: Table
Query: `neurectomy_agent_health_score`
Columns: ["Agent", "Tier", "Status", "Last Heartbeat", "Tasks/5m"]
Sorts: by Tier, then by Name

### Panel 3: Task Throughput

Type: Graph
Query: `tier3:agent_task_throughput:rate5m by (agent_tier)`
Legend: ["Tier 1", "Tier 2", ... "Tier 8"]
Y-Axis: Tasks per 5 minutes

### Panel 4: Overall Error Rate

Type: Gauge
Query: `(1 - tier3:agent_success_rate) * 100`
Unit: percent
Threshold: [0, 2, 5, 10]

### Panel 5: Success Rate by Tier

Type: Bar Chart
Query: `tier3:agent_success_rate by (agent_tier) * 100`
Unit: percent

### Panel 6: Collaboration Count

Type: Stat
Query: `sum(rate(neurectomy_agent_collaboration_total[5m]))`
Unit: "collaborations/5m"

### Panel 7: Collaboration Heatmap (Tier to Tier)

Type: Heatmap
Query: `tier3:agent_collaboration_count:rate5m by (initiator_tier, responder_tier)`
Legend: Tiers on both axes

### Panel 8: MNEMONIC Retrieval Latency

Type: Gauge
Query: `tier3:mnemonic_retrieval_latency:p95`
Unit: ms
Threshold: [0, 100, 500]

### Panel 9: Breakthrough Discoveries (1h)

Type: Stat
Query: `increase(neurectomy_mnemonic_breakthrough_promoted_total[1h])`
Unit: "discoveries/hour"

### Panel 10: Collective Fitness Score

Type: Gauge
Query: `tier3:collective_fitness_score * 100`
Unit: percent
Threshold: [0, 50, 70, 90, 100]

### Panel 11: Memory System Performance

Type: Table
Query: `TOPK(10, neurectomy_mnemonic_retrieval_duration_seconds by (agent_tier))`
Columns: ["Tier", "P50 (ms)", "P95 (ms)", "P99 (ms)"]

### Panel 12: Agent Failure Alerts

Type: Alert Status List
Query: Alert Rules for Agent Collective
Colors: Red for OPEN, Green for RESOLVED

---

## DASHBOARD 5: SLO Dashboard (Error Budget)

## Location: /d/slo-error-budget

## Refresh: 5m | Time Range: 30d

### Panel 1: 99.9% SLO - Neurectomy API

Type: Stat
Query: `(neurectomy_uptime_seconds / 2592000) * 100`
Suffix: "% Availability"
Target: 99.9%
Color: Green if >99.9%, Yellow if 99.0-99.9%, Red if <99.0%

### Panel 2: Error Budget Remaining (99.9%)

Type: Gauge
Query: `(1 - (1 - neurectomy_uptime_seconds / 2592000)) * 2592000 / 60`
Unit: minutes
Threshold: [0, 4.3, 43] # 30 days: 0-4.3 mins = critical, 4.3-43 mins = warning

### Panel 3: 99.0% SLO - All Services

Type: Stat
Query: `sum(neurectomy_uptime_seconds) / count(neurectomy_service_count) / 2592000 * 100`
Suffix: "%"

### Panel 4: 99.99% SLO - Agent Collective

Type: Stat
Query: `(neurectomy_agent_collective_uptime_seconds / 2592000) * 100`
Suffix: "%"

### Panel 5: 99.5% SLO - Storage (ΣVAULT)

Type: Stat
Query: `(neurectomy_sigmavault_uptime_seconds / 2592000) * 100`
Suffix: "%"

### Panel 6: Budget Burn-Down Chart (30 days)

Type: Area Graph
Query1: `error_budget_total{slo="99.9"}`
Query2: `error_budget_consumed{slo="99.9"}`
Query3: `error_budget_available{slo="99.9"}`
Legend: ["Total", "Consumed", "Available"]
Y-Axis: Time (minutes)

### Panel 7: Monthly SLO Trend

Type: Graph
Query: `neurectomy_uptime_seconds / 2592000 * 100`
Y-Axis: Availability %
Target Line: 99.9%

### Panel 8: Incidents This Month

Type: Table
Query: Alert logs for incidents
Columns: ["Start Time", "Duration", "Service", "Impact", "Cause"]

### Panel 9: Downtime Timeline (Daily)

Type: Bar Chart
Query: `sum by (day) (neurectomy_downtime_seconds)`
X-Axis: Day of Month
Y-Axis: Downtime (seconds)

### Panel 10: Error Budget Allocation

Type: Pie Chart
Query: `error_budget_consumed by (service)`
Legend: Services with portion of error budget

### Panel 11: SLO Status Matrix

Type: Table
Query: All SLOs
Columns: ["Service", "SLO %", "Current %", "Budget Remaining", "Status"]
Colors: Green if compliant, Red if violated

### Panel 12: Budget Run-Rate (7d forecast)

Type: Stat
Query: `error_budget_burn_rate * 7`
Unit: "days until exhausted"
Suffix: " (7-day forecast)"

---

## CROSS-DASHBOARD FEATURES

### Global Variables

- `$environment`: production, staging, development
- `$service_tier`: tier-1, tier-2, tier-3
- `$time_window`: 1h, 24h, 7d, 30d
- `$threshold_percentile`: 50, 95, 99, 99.9

### Navigation Links

- Ryot Dashboard → Agent Collective Dashboard (trace task origin)
- Compression Dashboard → Storage Dashboard (shows impact)
- Storage Dashboard → Cost Dashboard (shows cost)
- Any Dashboard → Alert Rules Dashboard (drill into alerting)

### Annotations

- Enable Prometheus alerts as annotations (overlay incidents)
- Tag: Deployments, Configuration Changes, Maintenance Windows

### Data Sources

- Primary: Prometheus (neurectomy-prod)
- Secondary: Loki (logs, for Phase 18C)
- Trace Source: Jaeger (for Phase 18B)

### Alerting

- All dashboards integrated with Alertmanager
- Alert boxes show current critical/warning alerts
- "Acknowledge" button for alert triage
- 1-click escalation to PagerDuty
