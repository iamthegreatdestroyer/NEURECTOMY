# NEURECTOMY Phase 18E - Loki Implementation Validation Checklist

## Pre-Deployment Validation

### Infrastructure Prerequisites

- [ ] Kubernetes cluster 1.24+ running
- [ ] kubectl configured and authenticated
- [ ] StorageClass "fast-ssd" available (`kubectl get storageclasses`)
- [ ] S3 or MinIO backend accessible
- [ ] DNS resolution working in cluster
- [ ] Network policies enabled in cluster (`kubectl api-resources | grep networkpolicies`)

**Verification Command:**

```bash
#!/bin/bash
set -e

echo "✓ Checking Kubernetes version..."
K8S_VERSION=$(kubectl version -o json | jq -r '.serverVersion.minor')
[[ $K8S_VERSION -ge 24 ]] && echo "  Version OK: 1.$K8S_VERSION" || exit 1

echo "✓ Checking StorageClass..."
kubectl get storageclass fast-ssd || (echo "  Creating StorageClass..."; exit 1)

echo "✓ Checking DNS..."
kubectl run -it --rm dns-test --image=busybox --restart=Never \
  -- nslookup kube-dns.kube-system.svc.cluster.local && echo "  DNS OK"

echo "✓ All prerequisites met!"
```

---

## Phase 1: Configuration Validation

### Kubernetes Manifests Review

**18-storage-cache-networking.yaml**

- [ ] StorageClass name: `fast-ssd`
- [ ] StorageClass provisioner: `ebs.csi.aws.com` or local
- [ ] Memcached replicas: 3
- [ ] Memcached resource limit: 512Mi
- [ ] NetworkPolicy ingress rules defined
- [ ] NetworkPolicy egress rules defined

**18-loki-configmap.yaml**

- [ ] Distributor rate limit: 100000 lines/sec
- [ ] Distributor rate limit burst: 200000
- [ ] Ingester chunk_idle_period: 3m
- [ ] Ingester max_chunk_age: 1h
- [ ] Storage backend: S3 or MinIO configured
- [ ] Index backend: boltdb-shipper
- [ ] Retention period: 720h (30 days)
- [ ] Compression: snappy
- [ ] Alertmanager URL: http://alertmanager.monitoring.svc.cluster.local:9093

**18-promtail-configmap.yaml**

- [ ] Server port: 9080
- [ ] Loki client URL: http://loki.monitoring.svc.cluster.local:3100
- [ ] Job config for ryot: present with JSON parsing
- [ ] Job config for sigmalang: present with multiline support
- [ ] Job config for sigmavault: present with audit labels
- [ ] Job config for agents: present with agent metrics
- [ ] Kubernetes job config: present
- [ ] Syslog job config: present on UDP 1514

**18-loki-deployment.yaml**

- [ ] StatefulSet replicas: 3
- [ ] Container image: grafana/loki:2.9.4 or 3.0.0+
- [ ] CPU requests: 500m
- [ ] Memory requests: 1Gi
- [ ] Storage size per replica: 100Gi
- [ ] Liveness probe: /ready endpoint
- [ ] Readiness probe: /ready endpoint
- [ ] Startup probe: /ready endpoint
- [ ] Pod anti-affinity: configured
- [ ] ServiceAccount: loki
- [ ] ClusterRole: includes configmaps, pods access

**18-promtail-daemonset.yaml**

- [ ] DaemonSet replicas: N/A (one per node)
- [ ] Container image: promtail:2.9.4 or 3.0.0+
- [ ] Node tolerations: all nodes
- [ ] PriorityClass: system-node-critical
- [ ] Volume mounts: /var/log, /var/lib/docker/containers
- [ ] ServiceAccount: promtail
- [ ] ClusterRole: includes pods/log access

**18-loki-secrets.yaml**

- [ ] Secret name: loki-s3-creds
- [ ] Secret keys: access_key, secret_key
- [ ] Base64 encoded (not plaintext)
- [ ] Alternative: loki-api-token secret

**Validation Command:**

```bash
#!/bin/bash
echo "✓ Validating YAML manifests..."

for file in 18-*.yaml; do
  echo "  Checking $file..."
  kubectl apply -f "$file" --dry-run=client -o yaml > /dev/null || \
    (echo "    ❌ YAML syntax error"; exit 1)
done

echo "✓ All manifests are valid!"
```

---

## Phase 2: Deployment Validation

### Step 1: Storage Deployment

```bash
# Expected output:
# - 1 StorageClass: fast-ssd
# - 3 Memcached pods running
# - 1 Memcached service
# - 2 NetworkPolicies: loki, promtail

echo "✓ Deploying storage resources..."
kubectl apply -f 18-storage-cache-networking.yaml

echo "✓ Waiting for Memcached..."
kubectl wait --for=condition=Ready pod \
  -l app=memcached -n monitoring --timeout=5m

echo "✓ Verifying..."
kubectl get all -n monitoring -l app=memcached
kubectl get storageclass fast-ssd
kubectl get networkpolicies -n monitoring
```

**Acceptance Criteria:**

- [ ] Memcached pods: 3 running
- [ ] Memcached service: active
- [ ] StorageClass: fast-ssd exists
- [ ] NetworkPolicies: 2 present

### Step 2: Loki Deployment

```bash
echo "✓ Creating S3 credentials secret..."
kubectl create secret generic loki-s3-creds -n monitoring \
  --from-literal=access_key=minioadmin \
  --from-literal=secret_key=minioadmin \
  --dry-run=client -o yaml | kubectl apply -f -

echo "✓ Deploying Loki ConfigMap..."
kubectl apply -f 18-loki-configmap.yaml

echo "✓ Deploying Loki StatefulSet..."
kubectl apply -f 18-loki-deployment.yaml

echo "✓ Waiting for Loki..."
kubectl rollout status statefulset/loki -n monitoring --timeout=10m

echo "✓ Verifying..."
kubectl get all -n monitoring -l app=loki
kubectl logs statefulset/loki -n monitoring -c loki --tail=20
```

**Acceptance Criteria:**

- [ ] Loki pods: 3 running
- [ ] All Loki pods: Ready (1/1)
- [ ] Loki service: active
- [ ] No error logs in startup

### Step 3: Promtail Deployment

```bash
echo "✓ Deploying Promtail ConfigMap..."
kubectl apply -f 18-promtail-configmap.yaml

echo "✓ Deploying Promtail DaemonSet..."
kubectl apply -f 18-promtail-daemonset.yaml

echo "✓ Waiting for Promtail..."
kubectl rollout status daemonset/promtail -n monitoring --timeout=5m

echo "✓ Verifying..."
kubectl get all -n monitoring -l app=promtail
kubectl get daemonset promtail -n monitoring -o wide

# Should show 1 Promtail pod per node
echo "✓ Promtail pod count (should equal node count):"
echo "  Expected: $(kubectl get nodes --no-headers | wc -l)"
echo "  Actual: $(kubectl get pods -n monitoring -l app=promtail --no-headers | wc -l)"
```

**Acceptance Criteria:**

- [ ] Promtail pods: 1 per node
- [ ] All Promtail pods: Ready (1/1)
- [ ] No error logs in startup

---

## Phase 3: API Validation

### Loki Readiness

```bash
echo "✓ Testing Loki API..."

# Port-forward
kubectl port-forward svc/loki -n monitoring 3100:3100 &
PF_PID=$!
sleep 2

# Test ready endpoint
echo "Testing /ready..."
curl -s http://localhost:3100/ready && echo "✓ Ready endpoint works"

# Test labels endpoint
echo "Testing /labels..."
curl -s 'http://localhost:3100/loki/api/v1/labels' | jq . && echo "✓ Labels endpoint works"

# Test query endpoint
echo "Testing /query..."
curl -s 'http://localhost:3100/loki/api/v1/query' \
  --data-urlencode 'query={job="kubernetes"}' | jq . && echo "✓ Query endpoint works"

kill $PF_PID
```

**Acceptance Criteria:**

- [ ] /ready returns 200 OK
- [ ] /labels returns valid JSON
- [ ] /query returns valid JSON (may be empty initially)

### Promtail Health

```bash
echo "✓ Testing Promtail health..."

# Get first Promtail pod
POD=$(kubectl get pod -n monitoring -l app=promtail -o name | head -1 | cut -d/ -f2)

# Check metrics endpoint
echo "Testing metrics..."
kubectl exec -it pod/$POD -n monitoring -c promtail \
  -- curl -s http://localhost:9080/metrics | grep promtail_entries_total

# Check config endpoint
echo "Testing scrape config..."
kubectl exec -it pod/$POD -n monitoring -c promtail \
  -- curl -s http://localhost:9080/scrape_configs | jq '.scrape_configs[0].job_name'
```

**Acceptance Criteria:**

- [ ] Metrics endpoint returns valid metrics
- [ ] Scrape configs endpoint returns job names
- [ ] No connection errors in logs

---

## Phase 4: Data Flow Validation

### Verify Log Ingestion

```bash
#!/bin/bash
echo "✓ Validating data flow..."

# Wait for initial logs
sleep 10

# Port-forward to Loki
kubectl port-forward svc/loki -n monitoring 3100:3100 &
PF_PID=$!
sleep 2

# Query for any logs
echo "Querying for any logs..."
LOGS=$(curl -s 'http://localhost:3100/loki/api/v1/query' \
  --data-urlencode 'query={job=~".+"}' | jq '.data.result | length')

if [[ $LOGS -gt 0 ]]; then
  echo "✓ Found $LOGS log streams"

  # Query specific service
  for SERVICE in ryot sigmalang sigmavault agents; do
    COUNT=$(curl -s 'http://localhost:3100/loki/api/v1/query' \
      --data-urlencode "query={service=\"$SERVICE\"}" | jq '.data.result | length')
    if [[ $COUNT -gt 0 ]]; then
      echo "  ✓ Service $SERVICE: $COUNT log streams"
    fi
  done
else
  echo "✗ No logs found - check Promtail scraping"
fi

kill $PF_PID
```

**Acceptance Criteria:**

- [ ] Total log streams: > 0
- [ ] Ryot logs present
- [ ] ΣLANG logs present
- [ ] ΣVAULT logs present
- [ ] Agents logs present

### Verify Metrics Extraction

```bash
echo "✓ Checking metrics extraction..."

# Port-forward Prometheus
kubectl port-forward svc/prometheus -n monitoring 9090:9090 &
PF_PID=$!
sleep 2

# Check for Promtail extracted metrics
echo "Promtail metrics:"
curl -s 'http://localhost:9090/api/v1/query?query=promtail_entries_total' | jq '.data.result[0]'

echo "Application metrics:"
curl -s 'http://localhost:9090/api/v1/query?query=log_lines_total' | jq '.data.result[0]'

kill $PF_PID
```

**Acceptance Criteria:**

- [ ] promtail_entries_total metric present
- [ ] Service-specific metrics present (log_lines_total, etc.)

---

## Phase 5: Integration Validation

### Grafana Data Source

```bash
echo "✓ Configuring Grafana datasource..."

# Port-forward Grafana
kubectl port-forward svc/grafana -n monitoring 3000:3000 &
PF_PID=$!
sleep 2

# Add Loki datasource via API
curl -X POST http://localhost:3000/api/datasources \
  -H "Content-Type: application/json" \
  -u admin:admin \
  -d '{
    "name": "Loki",
    "type": "loki",
    "url": "http://loki.monitoring.svc.cluster.local:3100",
    "access": "proxy",
    "isDefault": false
  }'

echo "✓ Datasource added"

kill $PF_PID
```

**Acceptance Criteria:**

- [ ] Loki datasource appears in Grafana
- [ ] Datasource health check passes

### Trace ID Injection (Optional - Phase 18E+)

```bash
echo "✓ Checking trace ID injection..."

# Query for logs with trace_id field
kubectl port-forward svc/loki -n monitoring 3100:3100 &
PF_PID=$!
sleep 2

TRACE_COUNT=$(curl -s 'http://localhost:3100/loki/api/v1/query' \
  --data-urlencode 'query={trace_id!=""}' | jq '.data.result | length')

if [[ $TRACE_COUNT -gt 0 ]]; then
  echo "✓ Trace ID injection working - $TRACE_COUNT streams have trace_id"
else
  echo "⚠ Trace ID injection not detected - configure in applications"
fi

kill $PF_PID
```

**Acceptance Criteria:**

- [ ] Trace ID field present in logs (if configured)
- [ ] Or acknowledge as future work

---

## Phase 6: Performance Validation

### Baseline Metrics

```bash
echo "✓ Collecting baseline metrics..."

# Port-forward Prometheus
kubectl port-forward svc/prometheus -n monitoring 9090:9090 &
PF_PID=$!
sleep 2

# Ingestion rate
echo "Log ingestion rate (lines/sec):"
curl -s 'http://localhost:9090/api/v1/query?query=rate(loki_distributor_lines_received_total[5m])' \
  | jq '.data.result[0].value[1]' | tr -d '"'

# Query latency
echo "Query latency (P99, seconds):"
curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.99,rate(loki_request_duration_seconds_bucket[5m]))' \
  | jq '.data.result[0].value[1]' | tr -d '"'

# Cache hit rate
echo "Cache hit rate:"
curl -s 'http://localhost:9090/api/v1/query?query=rate(cortex_memcache_request_cache_hits_total[5m])/(rate(cortex_memcache_request_cache_hits_total[5m])+rate(cortex_memcache_request_cache_misses_total[5m]))' \
  | jq '.data.result[0].value[1]' | tr -d '"'

kill $PF_PID
```

**Expected Values:**

- [ ] Ingestion rate: > 100 lines/sec
- [ ] Query latency P99: < 1 second
- [ ] Cache hit rate: > 50%

### Resource Usage

```bash
echo "✓ Checking resource usage..."

echo "Loki resource usage:"
kubectl top pod -l app=loki -n monitoring

echo "Promtail resource usage:"
kubectl top pod -l app=promtail -n monitoring

echo "Memcached resource usage:"
kubectl top pod -l app=memcached -n monitoring
```

**Expected Values:**

- [ ] Loki CPU: < 2 cores per pod
- [ ] Loki Memory: < 2Gi per pod
- [ ] Promtail CPU: < 500m per pod
- [ ] Promtail Memory: < 512Mi per pod
- [ ] Memcached Memory: < 512Mi per pod

---

## Phase 7: Test Query Validation

### Essential Queries

Test each query returns results:

```bash
# Port-forward
kubectl port-forward svc/loki -n monitoring 3100:3100 &
PF_PID=$!
sleep 2

QUERIES=(
  '{service="ryot"}'
  '{level="error"}'
  '{service="ryot"} | json'
  'count_over_time({service="ryot"}[1m])'
  '{service=~"ryot|sigmalang"}'
)

for query in "${QUERIES[@]}"; do
  echo "Testing: $query"
  curl -s 'http://localhost:3100/loki/api/v1/query' \
    --data-urlencode "query=$query" | jq '.data.result | length' && echo "  ✓ OK"
done

kill $PF_PID
```

**Acceptance Criteria:**

- [ ] All 5 queries return valid JSON
- [ ] At least 3 queries return results (not empty)

---

## Deployment Success Criteria (Go/No-Go)

### ✅ GO - Phase 18E Complete

- [ ] All 6 YAML manifests deployed successfully
- [ ] Loki: 3/3 pods running, Ready
- [ ] Promtail: N/N pods running (1 per node), Ready
- [ ] Memcached: 3/3 pods running, Ready
- [ ] All API endpoints responding (200 OK)
- [ ] Log data flowing from all 4 services
- [ ] Grafana datasource configured and healthy
- [ ] Query latency < 1s for standard queries
- [ ] Resource usage within limits

### ❌ NO-GO - Debug Required

If any criteria not met:

1. **Check component logs:**

   ```bash
   kubectl logs statefulset/loki -n monitoring -c loki --tail=50
   kubectl logs daemonset/promtail -n monitoring --tail=50
   ```

2. **Check resource availability:**

   ```bash
   kubectl top nodes
   kubectl describe pvc -n monitoring
   ```

3. **Run diagnostics:**

   ```bash
   # Full diagnostic report
   kubectl cluster-info
   kubectl get events -n monitoring --sort-by='.lastTimestamp' | tail -20
   kubectl debug node/<node-name>
   ```

4. **Consult troubleshooting guide:**
   - See PHASE-18E-LOKI-INTEGRATION.md - Troubleshooting section
   - See PHASE-18E-LOKI-QUICKSTART.md - Debugging Common Issues section

---

## Post-Deployment Checklist

- [ ] Document actual log volume vs. estimated
- [ ] Adjust retention if needed
- [ ] Set up monitoring alerts
- [ ] Train team on LogQL syntax
- [ ] Create on-call runbooks
- [ ] Schedule backup procedures
- [ ] Document S3 cost monthly
- [ ] Plan upgrade path

---

## Sign-Off

**Deployment Date:** ******\_\_\_******
**Deployed By:** ******\_\_\_******
**Verified By:** ******\_\_\_******
**Status:** ✅ PASS / ❌ FAIL

**Notes:**

```
[Space for deployment notes, issues resolved, deviations from plan]
```

---

**Next Phase:** [PHASE-18F-TBD]

See reference documentation:

- [PHASE-18E-LOKI-INTEGRATION.md](PHASE-18E-LOKI-INTEGRATION.md)
- [PHASE-18E-LOKI-TRACES-METRICS-INTEGRATION.md](PHASE-18E-LOKI-TRACES-METRICS-INTEGRATION.md)
- [PHASE-18E-LOKI-QUICKSTART.md](PHASE-18E-LOKI-QUICKSTART.md)
