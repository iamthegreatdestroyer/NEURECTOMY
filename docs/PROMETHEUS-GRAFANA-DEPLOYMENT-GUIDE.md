# Prometheus & Grafana Deployment & Validation Guide

# Neurectomy Phase 18A Integration

## DEPLOYMENT CHECKLIST

### Pre-Deployment Verification

#### 1. Configuration Validation

```bash
# Validate prometheus.yml syntax
$ docker run --rm -v $(pwd)/docker/prometheus:/etc/prometheus \
  prom/prometheus:latest \
  promtool check config /etc/prometheus/prometheus.yml

# Validate recording rules
$ docker run --rm -v $(pwd)/docker/prometheus:/etc/prometheus \
  prom/prometheus:latest \
  promtool check rules /etc/prometheus/recording.rules.yml

# Validate alert rules
$ docker run --rm -v $(pwd)/docker/prometheus:/etc/prometheus \
  prom/prometheus:latest \
  promtool check rules /etc/prometheus/alert.rules.yml
```

#### 2. Service Availability Checks

```bash
# Check all target services are running
$ docker-compose ps
  neurectomy-api    UP
  ryot-llm           UP
  sigmalang          UP
  sigmavault         UP
  agent-collective   UP
  postgres           UP
  redis              UP
  prometheus         UP
  grafana            UP
  alertmanager       UP
```

#### 3. Port Verification

```bash
# Verify all ports are accessible
$ curl http://localhost:9090       # Prometheus
$ curl http://localhost:3000       # Grafana
$ curl http://localhost:8000       # Neurectomy API (metrics)
$ curl http://localhost:9000       # Ryot (metrics)
$ curl http://localhost:9001       # ΣLANG (metrics)
$ curl http://localhost:9002       # ΣVAULT (metrics)
$ curl http://localhost:9003       # Agent Collective (metrics)
```

### Docker Compose Deployment

#### 1. Start Monitoring Stack

```bash
# Update docker-compose.yml to include all services
$ docker-compose up -d prometheus grafana alertmanager

# Verify containers started
$ docker-compose logs prometheus
$ docker-compose logs grafana
$ docker-compose logs alertmanager
```

#### 2. Verify Prometheus Targets

```
Navigate to: http://localhost:9090/targets
Expected Status: All Green (UP)

Targets:
  ✓ prometheus (self)
  ✓ neurectomy-api
  ✓ ryot-llm
  ✓ sigmalang
  ✓ sigmavault
  ✓ agent-collective
  ✓ postgres-exporter
  ✓ redis-exporter
  ✓ alertmanager
  ✓ grafana
```

#### 3. Verify Prometheus Data Collection

```bash
# Query Prometheus API for active metrics
$ curl -s "http://localhost:9090/api/v1/targets?state=active" | jq '.data.activeTargets | length'
# Should return: 10 or greater

# Check specific metric availability
$ curl -s "http://localhost:9090/api/v1/query?query=up" | jq '.data.result | length'

# Verify recording rules are active
$ curl -s "http://localhost:9090/api/v1/query?query=tier1:error_rate" | jq '.data.result'
```

### Kubernetes Deployment

#### 1. Create Namespace

```bash
$ kubectl create namespace neurectomy
$ kubectl label namespace neurectomy monitoring=enabled
```

#### 2. Create Secrets

```bash
# Grafana admin password
$ kubectl create secret generic grafana-admin \
  --from-literal=password=YourSecurePassword123 \
  -n neurectomy

# Basic auth for Grafana Ingress
$ htpasswd -c auth admin
$ kubectl create secret generic grafana-basic-auth \
  --from-file=auth \
  -n neurectomy
```

#### 3. Deploy Prometheus

```bash
$ kubectl apply -f deploy/k8s/prometheus-statefulset.yaml

# Verify deployment
$ kubectl get statefulset -n neurectomy
$ kubectl get pod -n neurectomy -l app=prometheus
$ kubectl logs -n neurectomy -l app=prometheus -f
```

#### 4. Deploy Grafana

```bash
$ kubectl apply -f deploy/k8s/grafana-deployment.yaml

# Verify deployment
$ kubectl get deployment -n neurectomy
$ kubectl get svc -n neurectomy grafana
```

#### 5. Verify Data Flow

```bash
# Port-forward for testing
$ kubectl port-forward -n neurectomy svc/prometheus 9090:9090 &
$ kubectl port-forward -n neurectomy svc/grafana 3000:3000 &

# Test Prometheus
$ curl http://localhost:9090/api/v1/query?query=up

# Access Grafana
# Open: http://localhost:3000
# Login: admin / YourSecurePassword123
```

### Validation Tests

#### Test 1: Scrape Job Target Verification

```bash
# Check each scrape job target is discoverable
$ curl -s "http://localhost:9090/api/v1/targets?state=active" | jq '
  .data.activeTargets[] |
  {job: .labels.job, instance: .labels.instance, state: .health}'

# Expected Output:
# {
#   "job": "neurectomy-api",
#   "instance": "neurectomy-api:8000",
#   "state": "up"
# }
# ... (repeated for each service)
```

#### Test 2: Recording Rule Validation

```bash
# Query each recorded metric
$ curl -s 'http://localhost:9090/api/v1/query?query=tier1:http_requests_total:rate5m' | jq '.data.result[0]'
$ curl -s 'http://localhost:9090/api/v1/query?query=tier2:ryot_latency:p95' | jq '.data.result[0]'
$ curl -s 'http://localhost:9090/api/v1/query?query=tier2:sigmavault_capacity_utilization' | jq '.data.result[0]'
$ curl -s 'http://localhost:9090/api/v1/query?query=tier3:agent_success_rate' | jq '.data.result[0]'

# All should return data within last 5 minutes
```

#### Test 3: Alert Rule Validation

```bash
# Check alert state
$ curl -s 'http://localhost:9090/api/v1/alerts' | jq '
  .data.alerts[] |
  {alert: .labels.alertname, state: .state, severity: .labels.severity}'

# Check specific alert can fire
$ curl -s 'http://localhost:9090/api/v1/query?query=HighHTTPErrorRate' | jq '.data.result'
```

#### Test 4: Dashboard Load Performance

```bash
# Measure dashboard query performance
$ time curl -s 'http://localhost:9090/api/v1/query_range?query=tier1:error_rate&start=1700000000&end=1700086400&step=60' > /dev/null

# Should complete in < 1 second for 1-day range
# If > 5 seconds, check:
#   - Prometheus storage optimization
#   - Add more recording rules for pre-aggregation
#   - Increase Prometheus memory allocation
```

#### Test 5: Grafana Data Source Connectivity

```bash
# Login to Grafana (http://localhost:3000)
# Admin Menu → Data Sources → Prometheus
# Click "Test" button
# Expected: "Data source is working"

# Test query in Grafana
# Explore → Enter: up
# Should return multiple series for each scrape job
```

#### Test 6: End-to-End Metric Flow

```bash
# 1. Generate load on API
$ for i in {1..100}; do curl -s http://localhost:8000/health > /dev/null; done

# 2. Query for HTTP requests in Prometheus (wait 30-60s)
$ curl -s 'http://localhost:9090/api/v1/query?query=rate(neurectomy_http_requests_total[1m])' | jq '.data.result[0].value[1]'
# Should return value > 0

# 3. Check dashboard shows request rate
# Navigate to Grafana Dashboard 1 (Ryot LLM Metrics)
# Panel: "Request Throughput" should show activity

# 4. Verify Prometheus UI shows recent data
# Visit http://localhost:9090/graph
# Query: neurectomy_http_requests_total{job="neurectomy-api"}
# Execution Time should be < 100ms
```

### Performance & Cost Analysis

#### Prometheus Storage Estimation

```
# Metrics per service:
#   - Neurectomy API:    ~50 metrics
#   - Ryot LLM:          ~30 metrics
#   - ΣLANG:             ~25 metrics
#   - ΣVAULT:            ~40 metrics
#   - Agent Collective:  ~45 metrics
#   - Infrastructure:    ~60 metrics
# Total: ~250 metrics

# With 8 label combinations per metric:
#   250 metrics × 8 labels = 2,000 time series

# Storage consumption:
#   2,000 series × 86,400 points/day (15s scrape) × 1.2 KB/point = 207 GB/30 days
#
#   With 30-day retention:
#   207 GB / 30 × 30 = 207 GB (steady state)

# Recommended storage: 100-150 GB persistent volume
# Query performance: <500ms for dashboard queries
```

#### Resource Allocation

```yaml
# Prometheus StatefulSet
resources:
  requests:
    cpu: 500m
    memory: 2Gi
  limits:
    cpu: 2000m
    memory: 8Gi

# Grafana Deployment
resources:
  requests:
    cpu: 100m
    memory: 128Mi
  limits:
    cpu: 500m
    memory: 512Mi
```

### PromQL Query Performance Analysis

#### Optimize Dashboard Queries

```bash
# Measure query execution time
$ time curl -s 'http://localhost:9090/api/v1/query?query=QUERY' | jq '.data.result'

# Query complexity levels:
# Level 1 (< 100ms): tier1:error_rate
# Level 2 (100-500ms): histogram_quantile(0.95, ...)
# Level 3 (500ms-2s): Complex aggregations with multiple joins
# Level 4 (> 2s): Requires optimization or caching

# Optimization strategies:
#  1. Use recorded metrics instead of raw metrics
#  2. Increase evaluation_interval for less-critical metrics
#  3. Pre-aggregate in recording rules
#  4. Use offset modifiers for baseline comparisons
#  5. Implement query result caching in Grafana
```

### Operational Runbooks

#### Alert: HighHTTPErrorRate

```
1. Check /targets endpoint
   curl http://prometheus:9090/targets
   Look for DOWN targets

2. Query recent errors
   Query: rate(neurectomy_http_requests_total{status=~"5.."}[5m])

3. Check Neurectomy API logs
   docker logs neurectomy-api | tail -100

4. If dependency service is down:
   a. Check service status: docker ps | grep <service>
   b. Restart: docker-compose restart <service>
   c. Monitor error rate decrease

5. If persistent:
   a. Check disk space: df -h
   b. Check memory: docker stats
   c. Check network connectivity: ping <service>
```

#### Alert: HighStorageUtilization

```
1. Check current utilization
   Query: tier2:sigmavault_capacity_utilization

2. Identify large datasets
   - Check ΣVAULT dashboard
   - Query: sum(neurectomy_sigmavault_used_bytes) by (bucket)

3. Implement retention policies
   - Archive old data to cheaper storage class
   - Delete temporary/test data

4. Plan capacity expansion
   - Add new storage volumes
   - Implement tiered storage
```

### Disaster Recovery

#### Backup Prometheus Data

```bash
# Docker Compose
$ docker-compose exec prometheus tar -czf prometheus-backup.tar.gz /prometheus
$ docker cp neurectomy-prometheus:/prometheus-backup.tar.gz ./backups/

# Kubernetes
$ kubectl exec -n neurectomy prometheus-0 -- \
  tar -czf prometheus-backup.tar.gz /prometheus
$ kubectl cp neurectomy/prometheus-0:/prometheus-backup.tar.gz \
  ./backups/prometheus-backup.tar.gz
```

#### Restore Prometheus Data

```bash
# Docker Compose
$ docker-compose down prometheus
$ docker-compose exec prometheus tar -xzf prometheus-backup.tar.gz -C /
$ docker-compose up -d prometheus

# Kubernetes
$ kubectl delete sts prometheus -n neurectomy
$ kubectl cp ./backups/prometheus-backup.tar.gz neurectomy/prometheus-0:/
$ kubectl exec -n neurectomy prometheus-0 -- \
  tar -xzf prometheus-backup.tar.gz -C /
$ kubectl rollout restart sts/prometheus -n neurectomy
```

### Monitoring the Monitors

#### Prometheus Self-Monitoring

```
Metrics to watch:
- prometheus_tsdb_symbol_table_size_bytes (< 9.2GB = healthy)
- prometheus_tsdb_compaction_duration_seconds (watch spikes)
- prometheus_sd_discovered_targets (should = expected targets)
- prometheus_rule_evaluation_duration_seconds (< 5s = healthy)
- scrape_duration_seconds (all < 10s)
```

#### Grafana Health

```
Metrics to watch:
- grafana_requests_total (should be increasing)
- grafana_db_conn_active (should be < max_connections)
- grafana_datasource_request_duration_seconds (< 500ms)
```

### Success Criteria

✓ All 10+ scrape targets showing UP
✓ Recording rules evaluated every 30s
✓ Alert rules checked every 30s
✓ Dashboard queries execute in < 1 second
✓ Grafana loads dashboards in < 2 seconds
✓ No scrape errors in Prometheus logs
✓ Storage growing at expected rate (~7GB/day)
✓ All 5 dashboards accessible and populated
✓ Alert firing/resolving working as expected
✓ Cost metrics properly tracked by cost_center labels
