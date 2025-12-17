# Phase 18D: Grafana Integration with Jaeger Traces

**Document Version:** 1.0  
**Component:** Grafana Trace Linking & Visualization  
**Target:** Neurectomy Phase 18D

---

## 📊 Overview

This guide enables seamless integration between Grafana dashboards and Jaeger distributed traces, allowing analysts to:

1. Click from metrics to correlated traces
2. View trace topology and dependencies
3. Correlate latency spikes with trace analysis
4. Debug performance issues end-to-end

---

## 🔗 Grafana Data Source Configuration

### Add Jaeger as Trace Data Source

**Step 1: Create Jaeger Data Source in Grafana**

1. Navigate to Configuration → Data Sources
2. Click "Add data source"
3. Select "Jaeger"
4. Configure:

```
Name: Jaeger - Traces
URL: http://jaeger-query.monitoring.svc.cluster.local:16686
Access: Server
Auth: None
Trace to Logs: Disabled (for now)
Trace to Metrics: Enabled (see below)
Service Graph: Enabled
```

**Step 2: Configure Trace-to-Metrics Link**

Enables clicking from traces to metrics:

```yaml
Data Source: Jaeger - Traces
Trace to Metrics:
  Enabled: true
  Data Source: Prometheus
  Tags:
    - service
    - operation
    - span.kind
    - http.status_code
```

**Step 3: Configure Trace-to-Logs Link**

Links traces to Loki logs:

```yaml
Data Source: Jaeger - Traces
Trace to Logs:
  Enabled: true
  Data Source: Loki
  Tags:
    - service
    - trace_id
    - span_id
```

### Prometheus Data Source Configuration

Ensure Prometheus is configured to scrape Jaeger metrics:

```yaml
# In prometheus ConfigMap
scrape_configs:
  - job_name: "jaeger"
    static_configs:
      - targets: ["jaeger-query:16687", "jaeger-collector:14269"]
    scrape_interval: 30s
    scrape_timeout: 10s
```

---

## 📈 Dashboard: Trace-Driven Performance Analysis

**File:** `deploy/k8s/grafana-dashboards/trace-analysis.json`

### Dashboard 1: Service Health with Trace Correlation

```json
{
  "dashboard": {
    "title": "Phase 18D: Distributed Trace Analysis",
    "tags": ["traces", "jaeger", "neurectomy"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Service Request Rate (Last 6h)",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(request_total[5m])) by (service)",
            "legendFormat": "{{service}}"
          }
        ],
        "fieldConfig": {
          "custom": {
            "links": [
              {
                "title": "Show Traces",
                "url": "/d/jaeger-traces?var-service=${service}",
                "targetBlank": true
              }
            ]
          }
        }
      },
      {
        "id": 2,
        "title": "P95 Latency by Service",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(request_duration_seconds_bucket[5m])) by (service, le))",
            "legendFormat": "{{service}} p95"
          }
        ],
        "thresholds": [
          { "value": 1000, "color": "yellow" },
          { "value": 5000, "color": "red" }
        ]
      },
      {
        "id": 3,
        "title": "Error Rate by Service",
        "type": "stat",
        "targets": [
          {
            "expr": "(sum(rate(errors_total[5m])) / sum(rate(requests_total[5m]))) * 100",
            "legendFormat": "Error Rate %"
          }
        ],
        "thresholds": {
          "mode": "absolute",
          "steps": [
            { "color": "green", "value": null },
            { "color": "yellow", "value": 1 },
            { "color": "red", "value": 5 }
          ]
        }
      },
      {
        "id": 4,
        "title": "Trace Sampling Rate",
        "type": "gauge",
        "targets": [
          {
            "expr": "avg(jaeger_sampling_rate)",
            "legendFormat": "Sampling %"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percentunit",
            "max": 1,
            "min": 0
          }
        }
      },
      {
        "id": 5,
        "title": "Service Dependency Graph",
        "type": "nodeGraph",
        "targets": [
          {
            "expr": "jaeger_service_graph_nodes",
            "format": "graph"
          }
        ]
      },
      {
        "id": 6,
        "title": "Operation Latency Breakdown",
        "type": "heatmap",
        "targets": [
          {
            "expr": "sum(rate(span_duration_seconds_bucket[5m])) by (operation, le)",
            "format": "heatmap"
          }
        ]
      },
      {
        "id": 7,
        "title": "Trace Search",
        "type": "alertlist",
        "targets": [
          {
            "expr": "topk(20, traces_sampled_total)"
          }
        ],
        "options": {
          "links": [
            {
              "title": "Open in Jaeger",
              "url": "http://jaeger-query.monitoring.svc.cluster.local:16686/search?service=${service}&operation=${operation}"
            }
          ]
        }
      }
    ]
  }
}
```

---

## 🔍 Dashboard: Per-Service Trace Analysis

### Dashboard 2: RYOT LLM Service Traces

```json
{
  "dashboard": {
    "title": "RYOT LLM: Trace Analysis",
    "tags": ["traces", "ryot"],
    "panels": [
      {
        "id": 1,
        "title": "LLM Inference Trace Timeline",
        "type": "trace",
        "targets": [
          {
            "datasource": "Jaeger - Traces",
            "query": {
              "service": "ryot",
              "operation": "llm.inference",
              "limit": 20
            }
          }
        ]
      },
      {
        "id": 2,
        "title": "Inference Duration Distribution",
        "type": "heatmap",
        "targets": [
          {
            "expr": "sum(rate(llm_inference_duration_seconds_bucket[5m])) by (le)",
            "format": "heatmap",
            "legendFormat": "{{le}}"
          }
        ]
      },
      {
        "id": 3,
        "title": "Cache Hit Traces vs Miss Traces",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(cache_hits_total) / sum(cache_total)",
            "legendFormat": "Hit Ratio"
          }
        ]
      },
      {
        "id": 4,
        "title": "Vector Search Latency Traces",
        "type": "scatterplot",
        "targets": [
          {
            "expr": "vector_search_duration_seconds",
            "legendFormat": "{{dimensions}}d search"
          }
        ]
      },
      {
        "id": 5,
        "title": "Error Traces - Last 24h",
        "type": "table",
        "targets": [
          {
            "datasource": "Jaeger - Traces",
            "query": {
              "service": "ryot",
              "tags": { "error": "true" },
              "limit": 50
            }
          }
        ]
      }
    ]
  }
}
```

### Dashboard 3: ΣVAULT Storage Traces

```json
{
  "dashboard": {
    "title": "ΣVAULT: Storage Operation Traces",
    "tags": ["traces", "sigmavault"],
    "panels": [
      {
        "id": 1,
        "title": "Storage Operations Timeline",
        "type": "trace",
        "targets": [
          {
            "datasource": "Jaeger - Traces",
            "query": {
              "service": "sigmavault",
              "operation": "storage.*"
            }
          }
        ]
      },
      {
        "id": 2,
        "title": "Encryption Duration by Operation Size",
        "type": "scatterplot",
        "targets": [
          {
            "expr": "encryption_duration_seconds",
            "legendFormat": "{{algorithm}}"
          }
        ]
      },
      {
        "id": 3,
        "title": "Tier Migration Traces",
        "type": "trace",
        "targets": [
          {
            "datasource": "Jaeger - Traces",
            "query": {
              "service": "sigmavault",
              "operation": "storage.migrate"
            }
          }
        ]
      },
      {
        "id": 4,
        "title": "Storage Latency Heatmap",
        "type": "heatmap",
        "targets": [
          {
            "expr": "sum(rate(storage_operation_duration_seconds_bucket[5m])) by (storage_tier, le)"
          }
        ]
      }
    ]
  }
}
```

### Dashboard 4: ΣLANG Compilation Traces

```json
{
  "dashboard": {
    "title": "ΣLANG: Compilation Trace Analysis",
    "tags": ["traces", "sigmalang"],
    "panels": [
      {
        "id": 1,
        "title": "Compilation Phases Timeline",
        "type": "trace",
        "targets": [
          {
            "datasource": "Jaeger - Traces",
            "query": {
              "service": "sigmalang",
              "operation": "sigma.*"
            }
          }
        ]
      },
      {
        "id": 2,
        "title": "Parse/TypeCheck/Codegen Duration",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(sigma_compilation_duration_seconds_bucket[5m])) by (phase, le))"
          }
        ]
      },
      {
        "id": 3,
        "title": "Error Traces by Compilation Phase",
        "type": "table",
        "targets": [
          {
            "datasource": "Jaeger - Traces",
            "query": {
              "service": "sigmalang",
              "tags": { "error": "true" }
            }
          }
        ]
      }
    ]
  }
}
```

### Dashboard 5: Agent Execution Traces

```json
{
  "dashboard": {
    "title": "Agent Collective: Execution Traces",
    "tags": ["traces", "agents"],
    "panels": [
      {
        "id": 1,
        "title": "Agent Task Execution Timeline",
        "type": "trace",
        "targets": [
          {
            "datasource": "Jaeger - Traces",
            "query": {
              "service": "agents",
              "operation": "agent.execute"
            }
          }
        ]
      },
      {
        "id": 2,
        "title": "Memory Retrieval Latency Traces",
        "type": "trace",
        "targets": [
          {
            "datasource": "Jaeger - Traces",
            "query": {
              "service": "agents",
              "operation": "memory.retrieve"
            }
          }
        ]
      },
      {
        "id": 3,
        "title": "Agent Collaboration Traces",
        "type": "nodeGraph",
        "targets": [
          {
            "datasource": "Jaeger - Traces",
            "query": {
              "service": "agents",
              "operation": "agent.collaborate"
            }
          }
        ]
      },
      {
        "id": 4,
        "title": "Task Complexity vs Duration",
        "type": "scatterplot",
        "targets": [
          {
            "expr": "agent_execution_duration_seconds"
          }
        ]
      }
    ]
  }
}
```

---

## 🔗 Trace Link Panel Configuration

### Panel: Link to Related Traces

```json
{
  "type": "stat",
  "title": "Click to See Traces",
  "targets": [
    {
      "expr": "rate(request_duration_seconds_sum[5m]) / rate(request_duration_seconds_count[5m])",
      "legendFormat": "{{service}} avg latency"
    }
  ],
  "options": {
    "links": [
      {
        "title": "View Traces",
        "url": "/explore?datasource=Jaeger&query=service:${service} AND operation:${operation}",
        "targetBlank": true
      },
      {
        "title": "Export as PDF",
        "url": "/api/v1/traces/export?service=${service}&format=pdf"
      }
    ]
  }
}
```

---

## 🎯 Template Variables

Enable dashboard filtering by service, operation, and time range:

```json
{
  "templating": {
    "list": [
      {
        "name": "service",
        "type": "query",
        "datasource": "Prometheus",
        "query": "label_values(request_total, service)",
        "current": { "value": "ryot", "text": "ryot" },
        "multi": false
      },
      {
        "name": "operation",
        "type": "query",
        "datasource": "Jaeger - Traces",
        "query": "operations(service:${service})",
        "current": { "value": "llm.inference", "text": "llm.inference" }
      },
      {
        "name": "error_status",
        "type": "custom",
        "options": [
          { "value": "all", "text": "All" },
          { "value": "true", "text": "Errors Only" },
          { "value": "false", "text": "Success Only" }
        ],
        "current": { "value": "all", "text": "All" }
      },
      {
        "name": "min_latency",
        "type": "custom",
        "options": [
          { "value": "0", "text": "All" },
          { "value": "100", "text": "100ms+" },
          { "value": "500", "text": "500ms+" },
          { "value": "1000", "text": "1s+" }
        ],
        "current": { "value": "0", "text": "All" }
      }
    ]
  }
}
```

---

## 📊 PromQL + Jaeger Query Examples

### Find Slow Traces with Metrics

```promql
# Query: P95 latency for slow operations
histogram_quantile(0.95,
  sum(rate(span_duration_seconds_bucket{service="ryot"}[5m])) by (le, operation)
)

# Then link to Jaeger:
# /explore?datasource=Jaeger&query=service:ryot AND operation:llm.inference AND latency_ms>=[p95 value]
```

### Find Error Traces

```promql
# Query: Error rate spike
increase(errors_total{service="sigmavault"}[5m]) > 10

# Link to traces:
# /explore?datasource=Jaeger&query=service:sigmavault AND status:ERROR
```

### Find Traces by Duration

```promql
# Query: Traces exceeding SLO
sum(rate(span_duration_seconds_bucket{job="jaeger",le="+Inf"}[5m])) -
sum(rate(span_duration_seconds_bucket{job="jaeger",le="5.0"}[5m]))

# Link to traces:
# /explore?datasource=Jaeger&query=minDuration:5000ms&maxDuration:60000ms
```

---

## 🔍 Jaeger Query Integration in Grafana

### Query Syntax for Tracing

**Service + Operation:**

```
service:ryot AND operation:llm.inference
```

**With Latency Filter:**

```
service:ryot AND minDuration:100ms AND maxDuration:5000ms
```

**With Error Filter:**

```
service:sigmavault AND error:true
```

**With Tag Filter:**

```
service:agents AND tag:task.complexity:high
```

**Complex Query:**

```
(service:ryot OR service:sigmalang) AND
(operation:llm.inference OR operation:sigma.parse) AND
minDuration:500ms AND
(tag:error:true OR tag:status:timeout)
```

---

## 🚀 Deployment Instructions

### 1. Create Grafana Dashboards

```bash
# Apply all dashboard ConfigMaps
kubectl apply -f deploy/k8s/grafana-dashboards/

# Verify dashboards are loaded
kubectl get cm -n monitoring | grep grafana-dashboard
```

### 2. Configure Data Sources

```bash
# Create datasource provisioning
cat > deploy/k8s/grafana-datasources.yaml << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-datasources
  namespace: monitoring
data:
  jaeger-traces.yaml: |
    apiVersion: 1
    datasources:
    - name: Jaeger - Traces
      type: jaeger
      access: proxy
      url: http://jaeger-query:16686
      isDefault: false
      editable: true
      jsonData:
        tracesToMetricsUI:
          enabled: true
          datasourceUid: prometheus
        tracesToMetricsURL: /api/datasources/proxy/uid:prometheus/api/v1/series
        serviceMap:
          datasourceUid: prometheus
        spanBar:
          tag: http.status_code
EOF

kubectl apply -f deploy/k8s/grafana-datasources.yaml
```

### 3. Link Dashboards

```bash
# Restart Grafana to load new dashboards
kubectl rollout restart deployment/grafana -n monitoring
```

---

## ✅ Verification Checklist

- [ ] Jaeger data source configured in Grafana
- [ ] Trace-to-metrics linking enabled
- [ ] Service dependency graph visible
- [ ] Dashboard panels show trace data
- [ ] Template variables work correctly
- [ ] Links to Jaeger UI functional
- [ ] Query examples return results
- [ ] Error traces captured and visible
- [ ] Sampling rates reflected in dashboards
