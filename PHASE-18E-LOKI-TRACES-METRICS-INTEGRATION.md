# NEURECTOMY Phase 18E - Loki + Metrics + Traces Integration

## Observability Triangle: Logs, Metrics, Traces

```
                    OBSERVABILITY
                    /    |    \
                   /     |     \
                  /      |      \
                 /       |       \
           LOGS         METRICS    TRACES
           /               |         \
          /                |          \
    Promtail          Prometheus     Jaeger
       |                   |           |
       |                   |           |
    Loki                   |           |
       |                   |           |
       └───────────┬───────┴───────────┘
                   |
               GRAFANA
           (Unified View)
```

---

## Part 1: Logs + Metrics Integration

### Use Case 1: Derive Metrics from Logs

Promtail can extract metrics directly from logs using the metrics pipeline stage:

```yaml
# promtail-configmap.yaml
pipeline_stages:
  - json:
      expressions:
        timestamp: timestamp
        level: level
        message: message
        duration_ms: duration
        status: status

  # OPTION 1: Counter for error events
  - metrics:
      log_errors_total:
        type: Counter
        description: "Total error logs"
        prefix: promtail_
        value: "1"
        match: 'level="error"'

  # OPTION 2: Histogram for request latency
  - metrics:
      request_duration_seconds:
        type: Histogram
        description: "Request duration from logs"
        prefix: promtail_
        buckets: [0.01, 0.1, 0.5, 1, 2, 5, 10]
        value_name: duration_ms
        value_type: milliseconds

  # OPTION 3: Gauge for current memory
  - metrics:
      memory_usage_bytes:
        type: Gauge
        description: "Memory usage"
        prefix: promtail_
        value_name: memory_bytes
```

### Use Case 2: Correlate Logs and Metrics in Grafana

#### Dashboard Panel: Error Rate + Error Logs

```json
{
  "title": "Error Analysis: Metrics + Logs",
  "type": "row",
  "panels": [
    {
      "title": "Error Rate (Metric)",
      "type": "timeseries",
      "targets": [
        {
          "expr": "rate(promtail_log_errors_total[5m])",
          "legendFormat": "{{ service }}"
        }
      ]
    },
    {
      "title": "Error Logs (Log)",
      "type": "logs",
      "targets": [
        {
          "expr": "{level=\"error\"}",
          "refId": "A"
        }
      ],
      "fieldConfig": {
        "defaults": {
          "custom": {
            "links": [
              {
                "title": "View Trace",
                "url": "/explore?left={\"datasource\":\"Jaeger\",\"queries\":[{\"queryType\":\"search\",\"query\":{\"traceID\":\"${trace_id}\"}}]}"
              }
            ]
          }
        }
      }
    }
  ]
}
```

### Use Case 3: Alert on Logs (via Metrics)

```yaml
# Prometheus alert rules
groups:
  - name: loki_metrics_alerts
    rules:
      # Alert when error rate spikes
      - alert: ErrorRateSpike
        expr: |
          rate(promtail_log_errors_total[5m]) > 
          avg_over_time(promtail_log_errors_total[5m] offset 1h)
        for: 5m
        annotations:
          summary: "Error rate spike in {{ $labels.service }}"
          logs_link: |
            /explore?left={"datasource":"Loki","queries":[{"expr":"{service=\"{{ $labels.service }}\",level=\"error\"}"}]}

      # Alert when request latency is high
      - alert: HighLatency
        expr: |
          histogram_quantile(0.99, 
            rate(promtail_request_duration_seconds_bucket[5m])
          ) > 1
        for: 5m
        annotations:
          summary: "P99 latency > 1s in {{ $labels.service }}"
```

---

## Part 2: Logs + Traces Integration

### Use Case 1: Trace ID Injection

All services must inject trace IDs into logs:

#### **Python/FastAPI Example**

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
import logging
import json

# Configure tracing
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)
trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

# Configure logging with trace context
class TraceContextFilter(logging.Filter):
    def filter(self, record):
        span = trace.get_current_span()
        ctx = span.get_span_context()
        record.trace_id = format(ctx.trace_id, "032x")
        record.span_id = format(ctx.span_id, "016x")
        return True

logger = logging.getLogger(__name__)
logger.addFilter(TraceContextFilter())

# Use JSON formatter to include trace IDs
handler = logging.StreamHandler()
formatter = logging.Formatter(json.dumps({
    "timestamp": "%(asctime)s",
    "level": "%(levelname)s",
    "message": "%(message)s",
    "trace_id": "%(trace_id)s",
    "span_id": "%(span_id)s",
}))
handler.setFormatter(formatter)
logger.addHandler(handler)

# Your application code
tracer = trace.get_tracer(__name__)

@app.get("/process")
def process_request():
    with tracer.start_as_current_span("process"):
        logger.info("Processing request")
        # Your code here
```

#### **Go Example**

```go
import (
    "log"
    "encoding.json"
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/trace"
)

func ProcessRequest(ctx context.Context) {
    span := trace.SpanFromContext(ctx)
    traceID := span.SpanContext().TraceID().String()
    spanID := span.SpanContext().SpanID().String()

    log.Printf(
        `{"level":"info","message":"Processing request","trace_id":"%s","span_id":"%s"}`,
        traceID, spanID,
    )
}
```

### Use Case 2: Query Logs by Trace ID

```logql
# Query all logs for a trace
{trace_id="4bf92f3577b34da6a3ce929d0e0e4736"}
| json
| line_format "[{{ .service }}:{{ .span_id }}] {{ .level }}: {{ .message }}"
| order by timestamp asc
```

### Use Case 3: Grafana Dashboard - Trace Logs View

```json
{
  "type": "logs",
  "title": "Trace Logs",
  "targets": [
    {
      "expr": "{trace_id=\"$trace_id\"}",
      "refId": "A"
    }
  ],
  "options": {
    "showTime": true,
    "showLabels": ["service", "span_id", "level"],
    "dedupStrategy": "none",
    "maxLines": 1000,
    "sortOrder": "Ascending"
  },
  "links": [
    {
      "title": "View in Jaeger",
      "url": "http://jaeger:16686/trace/${trace_id}",
      "targetBlank": true
    }
  ]
}
```

### Use Case 4: Trace-Logs Correlation in Jaeger

Configure Jaeger to link to logs:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: jaeger-config
  namespace: monitoring
data:
  sampling.json: |
    {
      "default_strategy": {
        "type": "probabilistic",
        "param": 0.1
      }
    }
  # Add Loki link configuration
  ui-config.json: |
    {
      "dependencies": {
        "menuEnabled": true
      },
      "tracking": {
        "gaID": ""
      },
      "menu": [
        {
          "label": "Logs",
          "items": [
            {
              "label": "View Logs in Loki",
              "url": "http://grafana:3000/explore?orgId=1&left={\"datasource\":\"Loki\",\"queries\":[{\"expr\":\"{trace_id=\\\"$traceID\\\"}\"}]}"
            }
          ]
        }
      ]
    }
```

---

## Part 3: Metrics + Traces Integration

### Use Case 1: Service Metrics with Trace Sampling

Prometheus metrics can track which spans are sampled:

```yaml
# Prometheus remote write to Tempo
remote_write:
  - url: http://tempo:3200/api/prom/push
    resource_to_telemetry_conversion:
      enabled: true
```

### Use Case 2: Metrics-Based Trace Sampling

Use RED method (Rate, Errors, Duration) to drive trace sampling:

```python
# In OpenTelemetry SDK configuration
from opentelemetry.sdk.trace.sampler import TraceIdRatioBased
from opentelemetry.sdk.trace.samplers import ParentBasedSampler

# Sample 100% of error traces, 10% of success traces
class SmartSampler:
    def should_sample(self, sampling_context):
        # Check if this is an error path
        if 'error' in sampling_context:
            return True  # Sample all errors

        # Check latency
        if sampling_context.get('duration_ms', 0) > 1000:
            return True  # Sample slow requests

        # Otherwise sample 10%
        import random
        return random.random() < 0.1

tracer_provider.sampler = SmartSampler()
```

### Use Case 3: Grafana - Unified Dashboard

```json
{
  "title": "Service Health - Metrics + Traces + Logs",
  "panels": [
    {
      "title": "Request Rate",
      "type": "timeseries",
      "targets": [
        {
          "datasource": "Prometheus",
          "expr": "rate(http_requests_total[5m])",
          "legendFormat": "{{ service }}"
        }
      ]
    },
    {
      "title": "Error Rate",
      "type": "timeseries",
      "targets": [
        {
          "datasource": "Prometheus",
          "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
          "legendFormat": "{{ service }}"
        }
      ]
    },
    {
      "title": "P99 Latency",
      "type": "timeseries",
      "targets": [
        {
          "datasource": "Prometheus",
          "expr": "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))",
          "legendFormat": "{{ service }}"
        }
      ],
      "links": [
        {
          "title": "View Traces",
          "url": "/explore?left={\"datasource\":\"Jaeger\",\"queries\":[{\"queryType\":\"search\",\"query\":{\"service\":\"{{ service }}\",\"operation\":\"\",\"minDuration\":\"1s\"}}]}"
        }
      ]
    },
    {
      "title": "Error Logs",
      "type": "logs",
      "targets": [
        {
          "datasource": "Loki",
          "expr": "{level=\"error\"}"
        }
      ],
      "links": [
        {
          "title": "View Trace",
          "url": "/explore?left={\"datasource\":\"Jaeger\",\"queries\":[{\"queryType\":\"search\",\"query\":{\"traceID\":\"${trace_id}\"}}]}"
        }
      ]
    }
  ]
}
```

---

## Part 4: Three-Way Correlation Pattern

### Complete Flow: Metrics → Logs → Traces

```
1. DETECT ANOMALY (Metrics)
   ├─ Prometheus alert: error_rate > 10%
   ├─ Alert annotation includes error_log_query
   └─ Link to Loki dashboard

2. INVESTIGATE LOGS
   ├─ Query Loki: {level="error", service="ryot"}
   ├─ Extract trace_id from error logs
   ├─ Note timestamp and pattern
   └─ Link to Jaeger trace view

3. ANALYZE TRACE
   ├─ Query Jaeger: traceID from log
   ├─ View complete request flow
   ├─ Identify service causing error
   ├─ Check latency in each span
   └─ Link back to Loki for that span's logs

4. CORRELATE ALL
   ├─ Metrics show impact scale
   ├─ Logs show error details
   ├─ Traces show root cause
   └─ Create response/post-mortem
```

### Example Alert Definition

```yaml
groups:
  - name: service_health_correlation
    rules:
      - alert: ServiceDegradation
        expr: |
          (
            rate(http_requests_total{status=~"5.."}[5m]) > 0.1
          ) or (
            histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m])) > 1
          )
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "{{ $labels.service }} degradation detected"

          # Link to investigate via metrics
          metrics_dashboard: |
            /d/service-health?var-service={{ $labels.service }}

          # Link to investigate via logs
          error_logs: |
            /explore?left={"datasource":"Loki","queries":[{"expr":"{service=\"{{ $labels.service }}\",level=\"error\"}"}]}

          # Link to investigate via traces
          slow_traces: |
            /explore?left={"datasource":"Jaeger","queries":[{"queryType":"search","query":{"service":"{{ $labels.service }}","minDuration":"1s"}}]}
```

---

## Part 5: Storage Backend Integration

### Unified Data Lake Architecture

```yaml
# storage_config in loki-configmap.yaml
storage_config:
  # Loki stores chunks in S3
  aws:
    s3: s3://neurectomy-data/loki-chunks/

  # Jaeger stores traces in another S3 bucket
  # (configured separately in Jaeger)

  # Prometheus stores metrics in TimescaleDB
  # (configured separately in Prometheus)
```

### Cross-Service Data Retention

| Component            | Storage     | Retention | Access Pattern            |
| -------------------- | ----------- | --------- | ------------------------- |
| Logs (Loki)          | S3 Standard | 30 days   | Sequential (recent first) |
| Metrics (Prometheus) | TimescaleDB | 1 year    | Random, indexed           |
| Traces (Jaeger)      | S3 Glacier  | 7 days    | Random by trace ID        |

### Cost Optimization

```
Daily Data Generation:
├─ Logs: 30GB uncompressed → 3GB (10:1) → $0.23/day
├─ Metrics: 5GB uncompressed → 1GB (5:1) → $0.08/day
└─ Traces: 10GB uncompressed → 2GB (5:1) → $0.15/day

Total: $0.46/day = $14/month

With Tiering:
├─ Hot (7d): $5/day
├─ Warm (7-30d): $1/day
├─ Archive (30+ days): $0.50/day
└─ Total: ~$150/month
```

---

## Part 6: Implementation Checklist

### Phase 18E Loki Deployment

- [ ] **Loki Infrastructure**
  - [ ] StatefulSet with 3 replicas deployed
  - [ ] PersistentVolumes provisioned (100GB each)
  - [ ] BoltDB Shipper configured
  - [ ] S3 bucket created and credentials set
  - [ ] Memcached deployed for caching

- [ ] **Promtail Collectors**
  - [ ] DaemonSet deployed on all nodes
  - [ ] Kubernetes SD configured
  - [ ] Scrape configs for 4 services complete
  - [ ] JSON parsing working
  - [ ] Metrics extraction functional

- [ ] **Service Integration**
  - [ ] Ryot logs flowing to Loki
  - [ ] ΣLANG logs flowing to Loki
  - [ ] ΣVAULT logs flowing to Loki
  - [ ] Agents logs flowing to Loki
  - [ ] All services include trace_id in logs

- [ ] **Grafana Integration**
  - [ ] Loki datasource added
  - [ ] Log panels created
  - [ ] Trace links configured
  - [ ] Dashboards published

- [ ] **Alert Configuration**
  - [ ] Error rate alerts active
  - [ ] Latency alerts active
  - [ ] Log pattern alerts active
  - [ ] Alert annotations link to dashboards

- [ ] **Trace Correlation**
  - [ ] OpenTelemetry instrumentation complete
  - [ ] Jaeger exporter configured in all services
  - [ ] Trace IDs in log output verified
  - [ ] Grafana trace-logs links working

- [ ] **Monitoring & Observability**
  - [ ] Loki health metrics scraped by Prometheus
  - [ ] Promtail metrics available
  - [ ] Storage utilization tracked
  - [ ] Query performance baseline established

- [ ] **Documentation & Training**
  - [ ] LogQL query templates documented
  - [ ] Troubleshooting guide published
  - [ ] Team trained on log-based debugging
  - [ ] On-call runbooks updated

---

## Part 7: Advanced Patterns

### Pattern 1: Log-Based Canary Analysis

```logql
# Analyze error rate before/after deployment
let before = count_over_time({service="ryot", level="error", version="v1"} [1h]);
let after = count_over_time({service="ryot", level="error", version="v2"} [1h]);
(after - before) / before > 0.1
```

### Pattern 2: Dependency Discovery from Logs

```logql
# Extract service dependencies
{component="http_client"}
| json
| line_format "{{.service}} → {{.called_service}}"
| stats count() as count by line
| sort by count desc
```

### Pattern 3: Anomaly Detection

```logql
# Detect unusual error messages
{level="error"}
| json
| stats count() as count by message
| count > 10 and count < 1000  # Neither too common nor too rare
```

### Pattern 4: SLO Compliance Tracking

```logql
# Calculate request success rate for SLO
sum by (service) (count_over_time({status="200"} [1h])) /
sum by (service) (count_over_time({} [1h])) > 0.99
```

---

## Troubleshooting Integration Issues

### Issue: Trace IDs not in logs

```bash
# Check application logs
kubectl logs deployment/ryot -n neurectomy | grep trace_id

# Should see: "trace_id":"4bf92f3577b34da6a3ce929d0e0e4736"
```

### Issue: Logs not correlating with metrics

```logql
# Verify log volume matches metric count
{service="ryot"} | stats count() as log_count

# In Prometheus:
count(rate(promtail_log_errors_total[5m]))  # Should match
```

### Issue: Jaeger traces not linking to logs

1. Verify trace ID format (32-char hex)
2. Check Grafana datasource links configuration
3. Verify Loki datasource is accessible from Jaeger

---

## References

- [Grafana Loki Documentation](https://grafana.com/docs/loki/latest/)
- [OpenTelemetry Integration](https://opentelemetry.io/docs/)
- [Jaeger Integration](https://www.jaegertracing.io/docs/1.35/integration/)
- [Prometheus + Loki](https://grafana.com/blog/2020/10/26/loki-prometheus-like-experience-for-logs/)
