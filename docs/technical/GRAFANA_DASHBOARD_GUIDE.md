# Grafana Dashboard Creation Guide

**Location:** `docs/technical/GRAFANA_DASHBOARD_GUIDE.md`

## Overview

Comprehensive guide for creating and managing Grafana dashboards for ΣVAULT storage and Elite Agents monitoring systems. Includes step-by-step instructions, dashboard templates, and best practices.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Dashboard Components](#dashboard-components)
3. [Panel Types](#panel-types)
4. [ΣVAULT Dashboard](#σvault-dashboard)
5. [Elite Agents Dashboard](#elite-agents-dashboard)
6. [Best Practices](#best-practices)
7. [Advanced Features](#advanced-features)

---

## Quick Start

### Creating Your First Dashboard

1. **Login to Grafana** (default: `http://localhost:3000`)
   - Username: `admin`
   - Password: `admin`

2. **Create Dashboard**
   - Click "+" in left sidebar
   - Select "New dashboard"
   - Click "Add panel"

3. **Configure Data Source**
   - Panel Settings → Data Source
   - Select "Prometheus"
   - Verify connection: "Save & Test"

4. **Add Query**
   - In Metrics field, enter query:

   ```promql
   agents_collective_active_tasks
   ```

5. **Configure Display**
   - Set Title: "Active Tasks"
   - Set Unit: "tasks"
   - Set Thresholds (optional)

6. **Save Panel**
   - Click "Apply"
   - Save Dashboard: Ctrl+S

---

## Dashboard Components

### 1. Dashboard Settings

**Access:** Dashboard Settings (gear icon)

```yaml
Name: "ΣVAULT Storage Monitoring"
Description: "Real-time monitoring for encrypted storage service"
Tags:
  - sigma
  - storage
  - production
Timezone: UTC
Refresh: 30s (auto-refresh)
```

### 2. Variables (Templating)

Variables allow dynamic dashboard filtering.

**Example: Agent Selection Variable**

```yaml
Name: agent_name
Type: Query
Data source: Prometheus
Query: label_values(agents_status, agent_name)
Refresh: On dashboard load
Multi-select: true
Include all: true
```

**Usage in Panels:**

```promql
# Before
agents_success_rate

# After with variable
agents_success_rate{agent_name="$agent_name"}
```

### 3. Annotations

Annotations mark important events on graphs.

```yaml
Name: "Deployments"
Type: Query
Data source: Prometheus
Query: ALERTS{alertname="DeploymentStarted"}
```

### 4. Panel Links

Link panels to drill-down dashboards.

```yaml
Type: Dashboard
Title: "View Agent Details"
Dashboard: "Elite Agents - Agent Detail"
Variables:
  - agent_name: $agent_name
```

---

## Panel Types

### Time Series Panel (Recommended)

**Best for:** Metrics over time (lines, areas, bars)

**Configuration:**

```yaml
Graph style: Line (default)
Legend:
  - Show legend: true
  - Placement: Bottom
  - Values: Min, Mean, Max, Last
Tooltip: Shared (all series)
Thresholds:
  - 0.95: Green
  - 0.90: Yellow
  - 0: Red
```

**Example Query:**

```promql
rate(sigmavault_storage_operations_total[1m]) by (operation_type)
```

### Stat Panel

**Best for:** Single values, status, counts

**Configuration:**

```yaml
Value:
  Field: Last (non-null)
Orientation: Auto
Text mode: Auto
Color mode:
  - Background (for status)
  - Value (for metrics)
Unit: Short/Bytes/Percent/Custom
Thresholds:
  - Green if > 0.95
  - Yellow if > 0.90
  - Red if <= 0.90
```

**Example Query:**

```promql
agents_collective_intelligence_score
```

### Gauge Panel

**Best for:** Utilization, percentages, ratios

**Configuration:**

```yaml
Unit: Percent
Orientation: Gauge (circular)
Show threshold labels: true
Thresholds:
  - 0: Red
  - 50: Yellow
  - 80: Green
Max value: 100
```

**Example Query:**

```promql
avg(agents_utilization_ratio) * 100
```

### Bar Gauge

**Best for:** Category comparisons, rankings

**Configuration:**

```yaml
Orientation: Horizontal
Display mode: Gradient gauge
Value sizing: Auto
Thresholds:
  - Percentage
Color scheme: From thresholds
```

**Example Query:**

```promql
topk(10, agents_utilization_ratio)
```

### Table Panel

**Best for:** Detailed data, lists, matrices

**Configuration:**

```yaml
Columns:
  - agent_name: String
  - success_rate: Percent
  - error_rate: Percent
  - utilization: Percent
Sort by: success_rate (ascending)
Row options: Show pagination
```

**Example Query:**

```promql
{__name__=~"agents_success_rate|agents_error_rate|agents_utilization_ratio"}
```

### Heatmap Panel

**Best for:** Distribution histograms, latency

**Configuration:**

```yaml
Bucket size: Auto
Hide zero: false
Color scheme: Scheme
Show legend: true
```

**Example Query:**

```promql
sum(rate(agents_task_duration_seconds_bucket[5m])) by (le, agent_name)
```

### Pie Chart

**Best for:** Proportions, distribution

**Configuration:**

```yaml
Pie type: Pie
Display: Single series
Unit: Short
Legend: Show
```

**Example Query:**

```promql
sum(increase(agents_tier_task_count[24h])) by (tier)
```

### State Timeline

**Best for:** Status changes over time

**Configuration:**

```yaml
Data format: Time series (default)
Show thresholds: true
Connect nulls: false
```

**Example Query:**

```promql
agents_status by (agent_name)
```

### Logs Panel

**Best for:** Log aggregation

**Configuration:**

```yaml
Data source: Loki
Show logs context: true
Log level: From field
```

---

## ΣVAULT Dashboard

### Dashboard Structure

```
ΣVAULT Storage Monitoring
├── Row 1: System Overview
│   ├── Total Storage Used (Stat)
│   ├── Available Capacity (Gauge)
│   ├── Success Rate (Stat)
│   └── Daily Cost (Stat)
├── Row 2: Performance
│   ├── Operation Latency (Time Series)
│   ├── Throughput (Time Series)
│   ├── Error Rate (Time Series)
│   └── Operation Distribution (Pie)
├── Row 3: Capacity Planning
│   ├── Storage by Tier (Bar)
│   ├── Growth Trend (Time Series)
│   ├── Capacity Projection (Time Series)
│   └── Utilization by Tier (Bar Gauge)
└── Row 4: Financial
    ├── Cost Trend (Time Series)
    ├── Cost by Category (Pie)
    ├── Cost by Tier (Bar)
    └── Unit Cost (Stat)
```

### Panel 1: Total Storage Used

**Panel Type:** Stat  
**Title:** "Total Storage Used"  
**Unit:** Bytes

```promql
sum(sigmavault_storage_capacity_used_bytes)
```

**Thresholds:**

- 0: Green
- 1e12: Yellow (1TB)
- 5e12: Red (5TB)

---

### Panel 2: Storage by Tier

**Panel Type:** Bar Chart  
**Title:** "Storage Capacity by Tier"  
**Unit:** Bytes  
**Legend:** Placement: Right

```promql
sum(sigmavault_storage_capacity_used_bytes) by (storage_class)
```

---

### Panel 3: Operation Latency

**Panel Type:** Time Series  
**Title:** "Operation Latency (P50, P95, P99)"  
**Unit:** Seconds  
**Legend:** Show

```promql
histogram_quantile(0.50, rate(sigmavault_storage_operation_duration_seconds_bucket[5m])) as "P50"
histogram_quantile(0.95, rate(sigmavault_storage_operation_duration_seconds_bucket[5m])) as "P95"
histogram_quantile(0.99, rate(sigmavault_storage_operation_duration_seconds_bucket[5m])) as "P99"
```

---

### Panel 4: Success Rate

**Panel Type:** Stat  
**Title:** "Success Rate"  
**Unit:** Percent

```promql
(sum(rate(sigmavault_storage_operations_total{status="success"}[5m])) /
 sum(rate(sigmavault_storage_operations_total[5m]))) * 100
```

**Thresholds:**

- 0: Red
- 95: Yellow
- 99: Green

---

### Panel 5: Storage Growth

**Panel Type:** Time Series  
**Title:** "Storage Growth Trend"  
**Unit:** Bytes

```promql
sum(sigmavault_storage_capacity_used_bytes) by (storage_class)
```

---

### Panel 6: Daily Cost

**Panel Type:** Stat  
**Title:** "Daily Cost"  
**Unit:** Dollars

```promql
sum(increase(sigmavault_storage_cost_usd[24h]))
```

---

### Panel 7: Cost Trend

**Panel Type:** Time Series  
**Title:** "Cost Trend (7 days)"  
**Unit:** Dollars

```promql
sum(increase(sigmavault_storage_cost_usd[1d])) by (cost_type)
```

---

### Panel 8: Operation Throughput

**Panel Type:** Time Series  
**Title:** "Operation Throughput"  
**Unit:** ops/sec

```promql
sum(rate(sigmavault_storage_operations_total[1m])) by (operation_type)
```

---

## Elite Agents Dashboard

### Dashboard Structure

```
Elite Agent Collective Monitoring
├── Row 1: Collective Overview
│   ├── Collective Health (Gauge)
│   ├── Failed Agents (Stat)
│   ├── Active Tasks (Stat)
│   ├── Intelligence Score (Stat)
│   └── Breakthrough Count (Stat)
├── Row 2: Performance
│   ├── Success Rate (Stat)
│   ├── Error Rate (Stat)
│   ├── Throughput (Time Series)
│   └── Latency (Time Series)
├── Row 3: Agent Status Matrix
│   ├── All Agents Table (status, utilization, success rate)
│   ├── Top Utilized Agents (Bar)
│   ├── Least Reliable Agents (Bar)
│   └── Queue Depth (Table)
└── Row 4: Tier and Collaboration
    ├── Tier Health Scores (Bar)
    ├── Tier Utilization (Bar Gauge)
    ├── Collaboration Events (Time Series)
    └── Knowledge Sharing (Heat map)
```

### Panel 1: Collective Health

**Panel Type:** Gauge  
**Title:** "Collective Health"  
**Unit:** Percent

```promql
(agents_collective_healthy / agents_collective_total) * 100
```

**Thresholds:**

- 0: Red
- 80: Yellow
- 95: Green

---

### Panel 2: Collective Intelligence Score

**Panel Type:** Gauge  
**Title:** "Intelligence Score"  
**Unit:** Custom (0-1)

```promql
agents_collective_intelligence_score
```

**Thresholds:**

- 0: Red
- 0.5: Yellow
- 0.7: Green

---

### Panel 3: Agent Status Table

**Panel Type:** Table  
**Title:** "Agent Status Matrix"

```promql
{__name__=~"agents_success_rate|agents_utilization_ratio|agents_status"}
```

**Columns:**
| Agent Name | Tier | Status | Success Rate | Utilization | Active Tasks |
|-----------|------|--------|-------------|------------|--------------|
| @APEX | 1 | Healthy | 99.5% | 45% | 8 |
| @CIPHER | 1 | Healthy | 98.2% | 52% | 5 |
| ... | ... | ... | ... | ... | ... |

---

### Panel 4: Top Utilized Agents

**Panel Type:** Bar Chart  
**Title:** "Most Utilized Agents"  
**Unit:** Percent

```promql
topk(10, agents_utilization_ratio * 100)
```

---

### Panel 5: Tier Health Scores

**Panel Type:** Bar Gauge  
**Title:** "Tier Health Scores"  
**Unit:** Custom (0-1)

```promql
agents_tier_health_score by (tier)
```

---

### Panel 6: Collaboration Events

**Panel Type:** Time Series  
**Title:** "Inter-Agent Collaboration"  
**Unit:** events/sec

```promql
rate(agents_collaboration_events_total[1m])
```

---

### Panel 7: Task Latency Distribution

**Panel Type:** Heatmap  
**Title:** "Task Duration Distribution"  
**Unit:** Seconds

```promql
sum(rate(agents_task_duration_seconds_bucket[5m])) by (le, agent_name)
```

---

### Panel 8: Collective Throughput

**Panel Type:** Time Series  
**Title:** "Collective Throughput"  
**Unit:** tasks/sec

```promql
rate(agents_tasks_completed[1m])
```

---

## Best Practices

### 1. Dashboard Organization

- **One concern per dashboard**
  - ✅ "ΣVAULT Storage Monitoring"
  - ❌ "All Systems Dashboard"

- **Logical row grouping**
  - Overview row at top
  - Detail rows below
  - Drill-down links to other dashboards

- **Consistent naming**
  - Title: "[System] [Category] [Metric]"
  - Example: "ΣVAULT Performance Latency"

### 2. Color Scheme

**Consistent thresholds across dashboards:**

```yaml
Green: Healthy state (>95% success, <50% utilization)
Yellow: Warning state (>80% success, >70% utilization)
Red: Critical state (<80% success, >90% utilization)
```

### 3. Time Range Selection

**Default values:**

```yaml
Last 1h: Real-time monitoring
Last 24h: Daily trends
Last 7d: Weekly patterns
Last 30d: Monthly trends
```

### 4. Legend Configuration

```yaml
Show legend: true
Placement: Right (for time series)
Values:
  - Min (minimum value)
  - Mean (average)
  - Max (maximum value)
  - Last (current value)
```

### 5. Unit Selection

**Correct units for common metrics:**

```yaml
Latency: Seconds, Milliseconds
Throughput: Short (ops/sec)
Storage: Bytes
Cost: Dollars, Currencyish
Percentage: Percent
Ratio: Percentunit (0-1 display as 0-100)
Count: Short
```

### 6. Query Optimization

**Fast queries:**

- Use low-cardinality labels
- Filter before aggregation
- Use appropriate time ranges

**Example (Good):**

```promql
rate(metric_total{job="api"}[1m])
```

**Example (Bad):**

```promql
rate(metric_total[1m])
```

### 7. Alert Integration

**Show alert status on dashboard:**

```yaml
Panel type: Alert state
Query: ALERTS{job="storage"}
Show: Current state
```

---

## Advanced Features

### 1. Dashboard Variables

**Variable Types:**

**Query Variable (Dynamic):**

```yaml
Name: agent_id
Type: Query
Query: label_values(agents_status, agent_id)
Refresh: On dashboard load
Multi-select: true
```

**Custom Variable (Static):**

```yaml
Name: environment
Type: Custom
Values: prod,staging,dev
```

**Using Variables in Panels:**

```promql
agents_success_rate{agent_id="$agent_id", environment="$environment"}
```

### 2. Transformations

Transform data before visualization.

**Example: Calculate percentage:**

```
Transformation: Organize fields
→ Mode: Name by regex
→ Regex: (?<first>.*success.*)|(?<second>.*total.*)
→ Output name: $first vs $second
```

### 3. Links and Drill-Downs

**Panel Link to Dashboard:**

```yaml
Title: "View Agent Details"
URL: /d/agent-detail?agent=$agent_name
Open in: New tab
```

**Data Link (table cell click):**

```yaml
Field: agent_name
URL: /d/agent-detail?agent=${agent_name:text}
```

### 4. Annotations and Markers

**Add deployment markers:**

```yaml
Name: Deployments
Type: Alert
Query: ALERTS{alertname="DeploymentStarted"}
Time field: timestamp
```

### 5. Repeating Panels

Generate panels dynamically from variable.

```yaml
Repeat: agent_name
Max per row: 3
Direction: h (horizontal)
```

This creates one panel per agent in the variable.

### 6. Value Mapping

Map numeric values to text.

```yaml
Type: Value mapping
Mappings:
  0: Healthy
  1: Degraded
  2: Failed
```

---

## Dashboard JSON Examples

### Minimal Storage Panel

```json
{
  "id": 1,
  "type": "timeseries",
  "title": "Storage Used",
  "targets": [
    {
      "expr": "sum(sigmavault_storage_capacity_used_bytes)",
      "legendFormat": "Total Storage"
    }
  ],
  "fieldConfig": {
    "unit": "bytes",
    "custom": {
      "lineWidth": 2,
      "fillOpacity": 10
    },
    "thresholds": {
      "steps": [
        { "color": "green", "value": 0 },
        { "color": "yellow", "value": 1e12 },
        { "color": "red", "value": 5e12 }
      ]
    }
  }
}
```

### Agent Health Panel

```json
{
  "id": 2,
  "type": "gauge",
  "title": "Collective Health",
  "targets": [
    {
      "expr": "(agents_collective_healthy / agents_collective_total) * 100"
    }
  ],
  "fieldConfig": {
    "unit": "percent",
    "min": 0,
    "max": 100,
    "thresholds": {
      "steps": [
        { "color": "red", "value": 0 },
        { "color": "yellow", "value": 80 },
        { "color": "green", "value": 95 }
      ]
    }
  }
}
```

---

## Troubleshooting

### Issue: Panels Show "No Data"

**Causes:**

1. Query syntax error
2. Metrics not being scraped
3. Time range too narrow

**Solutions:**

1. Check query in Prometheus directly
2. Verify scrape config in prometheus.yml
3. Expand time range (e.g., Last 7d)

### Issue: Dashboard Loads Slowly

**Causes:**

1. Too many panels (30+)
2. Inefficient queries
3. High-cardinality metrics

**Solutions:**

1. Split into multiple dashboards
2. Optimize queries (filter early)
3. Reduce cardinality (add label filters)

### Issue: Legends Show Unreadable Names

**Causes:**

1. Long metric names
2. Too many series

**Solutions:**

1. Use `legendFormat` to rename
   ```promql
   rate(metric_total[1m]) as "{{operation_type}}"
   ```
2. Filter to reduce series count

---

## Example Dashboard JSON Template

```json
{
  "dashboard": {
    "title": "ΣVAULT Storage Monitoring",
    "tags": ["sigma", "storage"],
    "timezone": "UTC",
    "panels": [
      {
        "id": 1,
        "title": "Total Storage Used",
        "type": "stat",
        "targets": [{ "expr": "sum(sigmavault_storage_capacity_used_bytes)" }],
        "fieldConfig": {
          "unit": "bytes",
          "thresholds": {
            "steps": [
              { "color": "green", "value": 0 },
              { "color": "yellow", "value": 1e12 },
              { "color": "red", "value": 5e12 }
            ]
          }
        }
      }
    ],
    "refresh": "30s",
    "time": {
      "from": "now-24h",
      "to": "now"
    },
    "templating": {
      "list": [
        {
          "name": "storage_class",
          "type": "query",
          "datasource": "Prometheus",
          "query": "label_values(sigmavault_storage_capacity_used_bytes, storage_class)",
          "multi": true,
          "includeAll": true
        }
      ]
    }
  }
}
```

---

## Grafana CLI Commands

### Export Dashboard

```bash
grafana-cli admin export-dashboard storage-monitoring storage-dashboard.json
```

### Import Dashboard

```bash
grafana-cli admin import-dashboard storage-dashboard.json
```

### Create Organization

```bash
grafana-cli admin create-org --name "Storage Team"
```

---

## Dashboard Export/Import

### Export to JSON

1. Click "Share" (top right)
2. Select "Export" tab
3. Choose "Export for sharing externally"
4. Save JSON file

### Import from JSON

1. Click "+" in left sidebar
2. Select "Import dashboard"
3. Upload JSON file or paste content
4. Select datasource (Prometheus)
5. Click "Import"

---

## Integration with Alert Manager

Link dashboard panels to alert rules.

**Alert Panel Query:**

```promql
ALERTS{severity="critical"}
```

**Alert Details Tooltip:**

- Shows alert name
- Shows current value
- Shows threshold
- Shows when alert fired

---

## References

- [Grafana Dashboard Documentation](https://grafana.com/docs/grafana/latest/dashboards/)
- [Grafana Panel Types](https://grafana.com/docs/grafana/latest/panels/)
- [Prometheus Query Language](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [Grafana Variables](https://grafana.com/docs/grafana/latest/dashboards/variables/)

---

**Version:** 1.0  
**Last Updated:** December 16, 2025  
**Maintained By:** @SCRIBE
