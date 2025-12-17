# NEURECTOMY Phase 18E - Loki Centralized Logging Integration

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Log Pipeline](#log-pipeline)
3. [Storage Strategy](#storage-strategy)
4. [Retention Policies](#retention-policies)
5. [LogQL Query Reference](#logql-query-reference)
6. [Integration Guide](#integration-guide)
7. [Deployment Guide](#deployment-guide)
8. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    LOG AGGREGATION PIPELINE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Applications (4 Services)                                      │
│  ├── Ryot (Main App)                                           │
│  ├── ΣLANG (Sigma Language)                                    │
│  ├── ΣVAULT (Secure Vault)                                     │
│  └── Agents (Elite Agent Collective)                           │
│                                                                 │
│                           ↓                                     │
│                                                                 │
│  Promtail DaemonSet (on every node)                            │
│  ├── Kubernetes SD (pod discovery)                             │
│  ├── File tail (container logs)                                │
│  ├── JSON parsing & relabeling                                 │
│  └── Metrics generation (from logs)                            │
│                                                                 │
│                           ↓                                     │
│                                                                 │
│  Loki Distributor (stateless)                                  │
│  ├── Rate limiting (100K lines/sec)                            │
│  ├── Stream validation                                         │
│  └── Consistent hashing                                        │
│                                                                 │
│                           ↓                                     │
│                                                                 │
│  Loki Ingester (stateful, 3 replicas)                          │
│  ├── In-memory buffer (chunk assembly)                         │
│  ├── Index generation                                          │
│  └── Persistent storage (WAL)                                  │
│                                                                 │
│                           ↓                                     │
│                                                                 │
│  Storage Backend                                               │
│  ├── BoltDB Shipper (index, 24h periods)                       │
│  ├── S3/MinIO (chunk storage, compressed)                      │
│  └── Memcached (query cache, 1h TTL)                           │
│                                                                 │
│                           ↓                                     │
│                                                                 │
│  Grafana Loki Datasource                                       │
│  ├── LogQL queries                                             │
│  ├── Log visualization                                         │
│  └── Alerts from logs                                          │
│                                                                 │
│                           ↓                                     │
│                                                                 │
│  Integration Points                                            │
│  ├── Jaeger (trace correlation)                                │
│  ├── Prometheus (metrics correlation)                          │
│  └── AlertManager (log-based alerts)                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Service Scrape Targets

| Service    | Port | Labels                                 | Log Levels                      | Pipeline                   |
| ---------- | ---- | -------------------------------------- | ------------------------------- | -------------------------- |
| **Ryot**   | 8000 | service=ryot, tier=main                | DEBUG, INFO, WARN, ERROR        | JSON + metrics extraction  |
| **ΣLANG**  | 9001 | service=sigmalang, tier=compiler       | INFO, WARN, ERROR               | Multiline + timestamp      |
| **ΣVAULT** | 9002 | service=sigmavault, tier=security      | DEBUG, INFO, WARN, ERROR, AUDIT | JSON + operation labels    |
| **Agents** | 9003 | service=agents, agent_tier, agent_name | DEBUG, INFO, WARN, ERROR        | JSON + agent_id extraction |

---

## Log Pipeline

### Promtail Scrape Configuration Details

#### 1. **Kubernetes Service Discovery**

```yaml
kubernetes_sd_configs:
  - role: pod
    namespaces:
      names:
        - neurectomy
        - default
relabel_configs:
  # Filter by app label
  - source_labels: [__meta_kubernetes_pod_label_app]
    regex: ryot
    action: keep

  # Extract metadata
  - source_labels: [__meta_kubernetes_pod_name]
    target_label: pod

  - source_labels: [__meta_kubernetes_namespace]
    target_label: namespace

  - source_labels: [__meta_kubernetes_pod_label_version]
    target_label: version

  - replacement: ryot
    target_label: service
```

#### 2. **Log Parsing Pipeline Stages**

```yaml
pipeline_stages:
  # JSON parsing
  - json:
      expressions:
        timestamp: timestamp
        level: level
        message: message
        module: module
        trace_id: trace_id
        span_id: span_id

  # Create labels for indexing
  - labels:
      level:
      module:
      service:
      trace_id:
      span_id:

  # Extract metrics from logs
  - metrics:
      log_lines_total:
        type: Counter
        description: "Total number of log lines"
        prefix: promtail_
        value: "1"
```

#### 3. **Multiline Log Handling (for ΣVAULT)**

```yaml
pipeline_stages:
  - multiline:
      line_start_pattern: '^\d{4}-\d{2}-\d{2}'

  - json:
      expressions:
        timestamp: timestamp
        level: level
        message: message
```

#### 4. **Timestamp Parsing**

```yaml
pipeline_stages:
  - timestamp:
      source: timestamp
      format: "2006-01-02T15:04:05.000Z07:00"
      # formats: [RFC3339Nano, UnixMs, UnixUs, Unix]
```

### Log Volume Estimation

- **Ryot**: ~1,000 lines/min (INFO level)
- **ΣLANG**: ~500 lines/min (compilation logs)
- **ΣVAULT**: ~800 lines/min (security audit)
- **Agents**: ~2,000 lines/min (task execution)
- **Total**: ~4,300 lines/min = **71 lines/sec**
- **Peak**: **300 lines/sec** (with DEBUG enabled)

**Storage Estimate**:

- Average log size: 500 bytes
- Daily volume: 30GB uncompressed → 3GB compressed (10:1 ratio)
- 30-day retention: 90GB storage required

---

## Storage Strategy

### Multi-Tier Storage Architecture

```
┌────────────────────────────────────────────────┐
│         STORAGE TIER STRATEGY                  │
├────────────────────────────────────────────────┤
│                                                │
│  TIER 1: HOT (0-7 days)                        │
│  └─ Local SSD (fast, expensive)                │
│     ├─ Full chunk retention                    │
│     ├─ All indices available                   │
│     ├─ Real-time queries                       │
│     └─ Cost: ~$10/GB/month                     │
│                                                │
│  TIER 2: WARM (7-30 days)                      │
│  └─ S3 Standard                                │
│     ├─ Full chunks                             │
│     ├─ Searchable via S3 Select                │
│     ├─ 100ms query latency                     │
│     └─ Cost: ~$2.3/GB/month                    │
│                                                │
│  TIER 3: COLD (30+ days)                       │
│  └─ S3 Glacier                                 │
│     ├─ Compressed archive                      │
│     ├─ Restore on demand                       │
│     ├─ 1hr query latency                       │
│     └─ Cost: ~$0.5/GB/month                    │
│                                                │
└────────────────────────────────────────────────┘
```

### Storage Backend Configuration

#### **BoltDB Shipper (Index Storage)**

```yaml
storage_config:
  boltdb_shipper:
    active_index_directory: /loki/boltdb-shipper-active
    cache_location: /loki/boltdb-shipper-cache
    shared_store: s3
    index_gateway_client:
      server_address: dns:///index-gateway.monitoring.svc.cluster.local:9096
```

**Characteristics**:

- Index created every 24 hours
- Indices are immutable after creation
- Shipped to S3 for durability
- Queried from local cache for performance
- Index size: ~100MB/day (for 4,300 lines/min)

#### **S3 Object Storage (Chunks)**

```yaml
storage_config:
  aws:
    s3: s3://loki-data/chunks/
    endpoint: minio:9000
    access_key_id: ${AWS_ACCESS_KEY_ID}
    secret_access_key: ${AWS_SECRET_ACCESS_KEY}
    bucketnames: loki-chunks
    region: us-east-1
```

**Characteristics**:

- Chunks compressed with Snappy (10:1 ratio)
- Object key format: `s3://loki-chunks/<tenant>/<period>/<stream>/<chunk_id>`
- Multi-part upload for reliability
- Automatic retry on failure
- Lifecycle policies for cost optimization

#### **Memcached Query Cache**

```yaml
cache_config:
  memcached_client:
    addresses: memcached:11211
    batch_size: 1024
    parallelism: 100
    max_idle_conns: 100
    update_interval: 1m
```

**Performance**:

- Query cache: 1-hour TTL (default)
- Cache hit rate: 70-80% typical
- Compression: Off (network bandwidth favorable)
- Max cache size: 512MB per instance

---

## Retention Policies

### Default Retention by Tier

```yaml
limits_config:
  retention_period: 720h # 30 days default
  split_queries_by_interval: 24h
  retention_overrides:
    # Debug logs: 72 hours
    - selector: '{tier="debug"}'
      retention: 72h

    # Production logs: 30 days
    - selector: '{tier="prod"}'
      retention: 720h

    # Agent logs: 60 days
    - selector: '{service="agents"}'
      retention: 1440h
```

### Retention by Service

| Service | Retention | Reason             | Storage   |
| ------- | --------- | ------------------ | --------- |
| Ryot    | 30 days   | Production logs    | 30GB      |
| ΣLANG   | 14 days   | Build artifacts    | 15GB      |
| ΣVAULT  | 60 days   | Audit trail        | 60GB      |
| Agents  | 60 days   | Long-term analysis | 60GB      |
| Debug   | 3 days    | Development only   | On-demand |

### Automatic Cleanup

Loki automatically manages retention via:

```yaml
table_manager:
  poll_interval: 10m
  retention_deletes_enabled: true
  retention_period: 0h # Handled by retention_overrides
```

**Process**:

1. Every 10 minutes, check for expired indices
2. Compare creation time + retention period vs current time
3. Delete expired tables from BoltDB Shipper
4. Delete associated chunks from S3
5. Update table manager metrics

---

## LogQL Query Reference

### Query Language Overview

LogQL is Loki's query language, similar to PromQL but for logs.

#### **Syntax Basics**

```logql
# Log stream selector (labels)
{service="ryot"}

# With multiple labels
{service="ryot", level="error"}

# Label matching operators
{service!="ryot"}           # Not equal
{service=~"ryot|sigmalang"} # Regex match
{service!~"debug.*"}        # Regex not match

# Line content filtering
{service="ryot"} |= "error"      # Contains
{service="ryot"} != "debug"      # Does not contain
{service="ryot"} |~ "auth.*fail" # Regex match
```

### Essential Queries

#### **1. Count Errors by Service (Last Hour)**

```logql
sum by (service) (
  count_over_time({level="error"} [1h])
)
```

**Use Case**: Dashboard widget showing error distribution

#### **2. Error Rate Over Time (Ryot)**

```logql
rate({service="ryot", level="error"} [5m])
```

**Use Case**: Alert when error rate exceeds threshold

#### **3. Latency Analysis (Agent Tasks)**

```logql
histogram_quantile(0.95,
  sum by (service) (
    rate(request_duration_ms_bucket[5m])
  )
)
```

**Use Case**: P95 latency monitoring

#### **4. Log Volume by Level (ΣVAULT)**

```logql
sum by (level) (
  count_over_time({service="sigmavault"} [1h])
)
```

**Use Case**: Audit log analysis

#### **5. Failed Authentication Attempts**

```logql
{service="sigmavault", operation="auth", status!="success"}
| json
| line_format "{{.user}} failed at {{.timestamp}}"
```

**Use Case**: Security monitoring

#### **6. Trace ID Correlation**

```logql
{trace_id="abc123def456"}
| json
| line_format "[{{.service}}] {{.level}}: {{.message}}"
```

**Use Case**: Distributed trace debugging

#### **7. Long-Running Queries (Agents)**

```logql
{service="agents", type="task"}
| json
| duration_ms > 5000
| line_format "{{.agent}}: {{.task_id}} took {{.duration_ms}}ms"
```

**Use Case**: Performance analysis

#### **8. Memory Operations Analysis**

```logql
sum by (agent) (
  count_over_time(
    {service="agents", type="memory_op"} [1h]
  )
)
```

**Use Case**: Agent collective memory system

#### **9. Error Pattern Detection**

```logql
{level="error"}
| json
| line_format "{{.message}}"
| stats count() by line
| sort by count desc
```

**Use Case**: Identify common error patterns

#### **10. Service Dependencies from Logs**

```logql
{component="caller"}
| json
| line_format "{{.service}} -> {{.called_service}}"
| stats count() by line
```

**Use Case**: Dependency graph extraction

### Advanced Query Patterns

#### **Pattern Matching (Multi-field Extraction)**

```logql
{service="ryot"}
| pattern "<_> - <method> <path> <_> <status> <duration>"
| status >= 500
```

#### **Aggregation Functions**

```logql
# Count total logs
count(count_over_time({service="ryot"}[1h]))

# Sum values from logs
sum(sum_over_time({service="agents"} | json | memory_ops [1h]))

# Average
avg(avg_over_time({service="agents"} | json | duration [1h]))

# Min/Max
min(min_over_time({service="agents"} | json | memory_ops [1h]))
max(max_over_time({service="agents"} | json | memory_ops [1h]))
```

#### **Binning / Bucketing**

```logql
# Logs per minute
sum by (service) (
  count_over_time({service="ryot"} [1m])
) > 0
```

#### **With Instant Query (single point)**

```logql
count({service="ryot"})  # Current count
```

#### **Range Query (time series)**

```logql
count_over_time({service="ryot"} [1h])  # Count per 1m interval
```

### Debugging Queries

#### **Check what's being scraped**

```logql
{service="ryot"}
| line_format "pod={{.pod}} container={{.container}}"
| stats count() by pod, container
```

#### **Find missing labels**

```logql
{service="ryot", trace_id=""}
| stats count()
```

#### **Analyze label cardinality**

```logql
count(count_over_time({service="ryot"} [1d])) by level
```

#### **High cardinality warning**

```logql
count(count_over_time({service="ryot"} [1h])) by request_id > 1000
```

---

## Integration Guide

### Grafana Integration

#### **Step 1: Add Loki Datasource**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-datasources
  namespace: monitoring
data:
  loki.yaml: |
    apiVersion: 1
    datasources:
      - name: Loki
        type: loki
        access: proxy
        url: http://loki.monitoring.svc.cluster.local:3100
        jsonData:
          maxLines: 1000
          timeInterval: "5s"
        editable: true
```

#### **Step 2: Create Log Panels**

```json
{
  "type": "logs",
  "title": "Ryot Errors (Last Hour)",
  "targets": [
    {
      "expr": "{service=\"ryot\", level=\"error\"}",
      "refId": "A"
    }
  ],
  "options": {
    "showTime": true,
    "showLabels": ["service", "level", "module"],
    "maxLines": 100,
    "dedupStrategy": "none"
  }
}
```

#### **Step 3: Trace-Logs Integration**

```json
{
  "type": "logs",
  "title": "Distributed Trace Logs",
  "targets": [
    {
      "expr": "{trace_id=\"$trace_id\"}",
      "refId": "A"
    }
  ],
  "links": [
    {
      "title": "Jaeger",
      "url": "http://jaeger:16686/trace/${trace_id}",
      "targetBlank": true
    }
  ]
}
```

### Prometheus Integration

#### **Query Loki for Metrics Extraction**

```yaml
# Extract metrics from logs in Prometheus
- job_name: loki-metrics-extraction
  scrape_interval: 30s
  scrape_configs:
    - targets: ["localhost:9080"] # Promtail metrics
```

#### **Alert on Log Patterns**

```yaml
groups:
  - name: loki_alerts
    rules:
      - alert: HighErrorRate
        expr: |
          rate({level="error"}[5m]) > 10
        for: 5m
        annotations:
          summary: "High error rate in {{ $labels.service }}"
          dashboard: "http://grafana:3000/d/logs-dashboard"
```

### Jaeger Trace-Log Correlation

#### **Add Trace IDs to Logs**

```python
# Application code (Python example)
import logging
import opentelemetry.api

logger = logging.getLogger(__name__)
tracer = opentelemetry.api.trace.get_tracer(__name__)

def process_request(request):
    with tracer.start_as_current_span("process_request") as span:
        trace_id = span.get_span_context().trace_id
        logger.info(
            "Processing request",
            extra={
                "trace_id": trace_id,
                "span_id": span.get_span_context().span_id,
                "user_id": request.user_id
            }
        )
        # Process request...
```

#### **Query by Trace ID**

```logql
# Find all logs for a trace
{trace_id="4bf92f3577b34da6a3ce929d0e0e4736"}
```

#### **Link Jaeger→Logs**

In Grafana, configure Data Source Links:

```
Jaeger → Logs:
  - Name: View Logs
  - URL: /explore?orgId=1&left={"datasource":"Loki","queries":[{"expr":"{trace_id=\\"$trace_id\\"}"}]}
```

---

## Deployment Guide

### Prerequisites

- Kubernetes 1.21+
- 3 nodes minimum (for HA)
- 50GB storage per node (for local cache)
- S3 or MinIO bucket configured
- Prometheus (for metrics scraping)
- Grafana (for visualization)

### Deployment Steps

#### **1. Create Namespace**

```bash
kubectl create namespace monitoring
```

#### **2. Apply Storage and Cache**

```bash
kubectl apply -f deploy/k8s/18-storage-cache-networking.yaml
```

Wait for storage provisioning (5-10 minutes):

```bash
kubectl get pvc -n monitoring
kubectl get pods -n monitoring | grep memcached
```

#### **3. Deploy Loki**

```bash
kubectl apply -f deploy/k8s/18-loki-configmap.yaml
kubectl apply -f deploy/k8s/18-loki-secrets.yaml
kubectl apply -f deploy/k8s/18-loki-deployment.yaml
```

Wait for rollout:

```bash
kubectl rollout status statefulset/loki -n monitoring
kubectl logs -f statefulset/loki -n monitoring -c loki
```

#### **4. Deploy Promtail**

```bash
kubectl apply -f deploy/k8s/18-promtail-configmap.yaml
kubectl apply -f deploy/k8s/18-promtail-daemonset.yaml
```

Verify on all nodes:

```bash
kubectl get daemonset promtail -n monitoring
kubectl get pods -n monitoring | grep promtail
```

#### **5. Verify Integration**

```bash
# Port-forward Loki
kubectl port-forward -n monitoring svc/loki 3100:3100

# Test Loki API
curl http://localhost:3100/ready

# Query for logs
curl 'http://localhost:3100/loki/api/v1/query' \
  --data-urlencode 'query={service="ryot"}'
```

#### **6. Configure Grafana**

```bash
# Port-forward Grafana
kubectl port-forward -n monitoring svc/grafana 3000:3000

# Access Grafana: http://localhost:3000
# Add Loki datasource: http://loki:3100
```

### Configuration Tuning

#### **For High Volume (> 1000 lines/sec)**

```yaml
# Increase ingester settings
ingester:
  chunk_idle_period: 1m # Reduce from 3m
  max_chunk_age: 30m # Increase from 1h
  max_streams_per_user: 50000 # Increase

# Increase distributor rate limit
distributor:
  rate_limit: 500000 # Increase from 100k
  rate_limit_burst: 1000000
```

#### **For Cost Optimization**

```yaml
# Reduce retention
limits_config:
  retention_period: 168h # 7 days instead of 30

# Compress more
ingester:
  chunk_encoding: gzip # Instead of snappy
```

#### **For Query Performance**

```yaml
# Increase cache size
query_range:
  results_cache:
    cache:
      embedded_cache:
        enabled: true
        max_size_mb: 500 # Increase from 100

# Increase concurrent queries
querier:
  max_concurrent: 50 # Increase from 20
```

---

## Troubleshooting

### Common Issues

#### **Issue: "No such host" (Loki unreachable)**

```bash
# Check Loki service
kubectl get svc -n monitoring | grep loki

# Check DNS resolution
kubectl run -it --rm debug --image=busybox --restart=Never \
  -- nslookup loki.monitoring.svc.cluster.local

# Check Loki readiness
kubectl get pods -n monitoring -l app=loki
```

#### **Issue: High memory usage in Loki**

```bash
# Check resource usage
kubectl top pod -n monitoring -l app=loki

# Increase limits
kubectl set resources statefulset/loki \
  -n monitoring \
  -c loki \
  --limits=cpu=4000m,memory=8Gi
```

#### **Issue: Slow queries (>5s)**

```logql
# Find cardinality issues
count(count_over_time({service="ryot"} [1h])) by pod > 1000

# Check query cache hit rate
# In Grafana: Loki metrics → cache_hits / (cache_hits + cache_misses)
```

#### **Issue: S3 permission denied**

```bash
# Check S3 credentials
kubectl get secret loki-s3-creds -n monitoring -o yaml

# Test S3 connectivity from Loki pod
kubectl exec -it statefulset/loki -n monitoring -c loki \
  -- aws s3 ls s3://loki-chunks/
```

#### **Issue: Promtail not shipping logs**

```bash
# Check Promtail status
kubectl logs -f daemonset/promtail -n monitoring

# Verify config
kubectl describe daemonset promtail -n monitoring

# Check Promtail metrics
kubectl port-forward daemonset/promtail 9080:9080 -n monitoring
curl http://localhost:9080/metrics | grep promtail_entries
```

### Performance Tuning Checklist

- [ ] Loki ingester memory utilization < 80%
- [ ] Promtail lag < 10s (promtail_entries vs logs received)
- [ ] Cache hit rate > 70%
- [ ] P99 query latency < 5s
- [ ] Error rate < 0.1%
- [ ] Storage growth ~3GB/day
- [ ] Retention cleanup successful

### Monitoring Loki Health

```yaml
# Prometheus recording rules for Loki monitoring
groups:
  - name: loki_health
    interval: 30s
    rules:
      # Ingestion rate
      - record: loki:ingestion_rate
        expr: rate(loki_distributor_lines_received_total[5m])

      # Chunk size
      - record: loki:chunk_size_bytes
        expr: avg(loki_ingester_chunk_size_bytes)

      # Cache hit rate
      - record: loki:cache_hit_rate
        expr: |
          rate(cortex_memcache_request_cache_hits_total[5m]) /
          (rate(cortex_memcache_request_cache_hits_total[5m]) +
           rate(cortex_memcache_request_cache_misses_total[5m]))
```

---

## Best Practices

### Log Labeling Strategy

**Recommended Labels** (low cardinality):

- `service` (Ryot, ΣLANG, ΣVAULT, Agents)
- `level` (DEBUG, INFO, WARN, ERROR)
- `tier` (debug, prod, audit)
- `environment` (dev, staging, prod)

**Avoid High-Cardinality Labels**:

- ❌ `user_id` (millions of values)
- ❌ `request_id` (unique per request)
- ❌ `timestamp` (use log timestamp instead)
- ✅ `user_id` (extract via pipeline, filter in queries)

### Query Optimization

1. **Use Labels First**: Filter by labels before parsing JSON

   ```logql
   {service="ryot", level="error"} | json  # ✅ Good
   {service="ryot"} | json | level="error"  # ❌ Slow
   ```

2. **Limit Query Range**: Don't query entire 30 days at once

   ```logql
   {service="ryot"} [1h]  # ✅ Good
   {service="ryot"} [30d]  # ❌ Slow, use sampling
   ```

3. **Use Recording Rules**: Pre-compute common queries
   ```yaml
   - record: loki:error_rate:5m
     expr: rate({level="error"}[5m])
   ```

### Cost Optimization

| Strategy                      | Savings | Trade-off                       |
| ----------------------------- | ------- | ------------------------------- |
| Reduce retention (30d→7d)     | 75%     | Can't investigate old incidents |
| Compress harder (Snappy→Gzip) | 20%     | Slightly higher CPU             |
| S3 tiering (hot/warm/cold)    | 60%     | Query latency increases         |
| Sample logs (50%)             | 50%     | Miss rare events                |

---

## References

- [Loki Documentation](https://grafana.com/docs/loki/latest/)
- [LogQL Query Language](https://grafana.com/docs/loki/latest/logql/)
- [Promtail Configuration](https://grafana.com/docs/loki/latest/clients/promtail/configuration/)
- [Best Practices](https://grafana.com/docs/loki/latest/best-practices/)
