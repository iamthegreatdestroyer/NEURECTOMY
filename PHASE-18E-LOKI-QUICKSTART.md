# NEURECTOMY Phase 18E - Loki Deployment Quick Start

## 5-Minute Quick Start

### Prerequisites Check

```bash
# Verify Kubernetes cluster
kubectl cluster-info
kubectl get nodes

# Check available storage
kubectl get storageclasses

# Verify namespaces
kubectl get namespace monitoring
```

### Step 1: Create Storage Resources (2 min)

```bash
cd deploy/k8s

# Apply storage, networking, memcached
kubectl apply -f 18-storage-cache-networking.yaml

# Wait for storage provisioning
kubectl wait --for=condition=Bound pvc/memcached-cache -n monitoring --timeout=5m

# Verify memcached is running
kubectl get pods -n monitoring | grep memcached
kubectl wait --for=condition=Ready pod -l app=memcached -n monitoring --timeout=5m
```

**Expected Output**:

```
persistentvolumeclaim/memcached-cache bound pvc-xxxxx 50Gi ...
pod/memcached-0 1/1 Running ...
pod/memcached-1 1/1 Running ...
pod/memcached-2 1/1 Running ...
```

### Step 2: Deploy Loki (2 min)

```bash
# Configure S3 credentials
kubectl create secret generic loki-s3-creds \
  --from-literal=access_key=minioadmin \
  --from-literal=secret_key=minioadmin \
  -n monitoring

# Apply Loki configs and deployment
kubectl apply -f 18-loki-configmap.yaml
kubectl apply -f 18-loki-secrets.yaml
kubectl apply -f 18-loki-deployment.yaml

# Wait for Loki StatefulSet
kubectl rollout status statefulset/loki -n monitoring --timeout=10m
```

**Expected Output**:

```
statefulset.apps/loki rolled out successfully
pod/loki-0 1/1 Running ...
pod/loki-1 1/1 Running ...
pod/loki-2 1/1 Running ...
```

### Step 3: Deploy Promtail (1 min)

```bash
# Apply Promtail configs and DaemonSet
kubectl apply -f 18-promtail-configmap.yaml
kubectl apply -f 18-promtail-daemonset.yaml

# Verify on all nodes
kubectl get daemonset promtail -n monitoring
kubectl get pods -n monitoring | grep promtail

# Should show 1 promtail pod per node
```

### Step 4: Verify Integration

```bash
# Port-forward to Loki
kubectl port-forward svc/loki -n monitoring 3100:3100 &

# Test Loki API
curl -s http://localhost:3100/ready
# Expected: 200 OK

# Query for logs
curl -s 'http://localhost:3100/loki/api/v1/query' \
  --data-urlencode 'query={service="ryot"}' | jq .

# Should show logs if applications are logging
```

---

## Detailed Deployment Instructions

### Configuration Setup

#### Configure S3/MinIO Backend

```bash
# Using MinIO (local S3 compatible)
MINIO_HOST="minio.monitoring.svc.cluster.local"
MINIO_PORT="9000"
MINIO_ACCESS_KEY="minioadmin"
MINIO_SECRET_KEY="minioadmin"

# Create buckets
kubectl exec -it svc/minio -n monitoring \
  -- mc mb minio/loki-chunks

kubectl exec -it svc/minio -n monitoring \
  -- mc mb minio/loki-index

# Or using AWS S3
AWS_REGION="us-east-1"
AWS_ACCESS_KEY_ID="your-access-key"
AWS_SECRET_ACCESS_KEY="your-secret-key"
S3_BUCKET="neurectomy-loki"

# Create secret
kubectl create secret generic loki-s3-creds \
  -n monitoring \
  --from-literal=access_key=${AWS_ACCESS_KEY_ID} \
  --from-literal=secret_key=${AWS_SECRET_ACCESS_KEY}
```

#### Enable DNS for Service Discovery

```yaml
# Ensure CoreDNS is running
kubectl get pods -n kube-system | grep coredns

# Test DNS resolution
kubectl run -it --rm debug --image=busybox --restart=Never \
  -- nslookup loki.monitoring.svc.cluster.local
```

### Health Checks

```bash
# 1. Check Loki is ready
kubectl exec -it statefulset/loki -n monitoring -c loki \
  -- curl localhost:3100/ready

# 2. Check Promtail is scraping
kubectl logs -f daemonset/promtail -n monitoring | grep "sent"

# 3. Verify logs are flowing
curl -s 'http://localhost:3100/loki/api/v1/query' \
  --data-urlencode 'query=count(count_over_time({service="ryot"}[1m]))'

# 4. Check storage usage
kubectl exec -it statefulset/loki-0 -n monitoring -c loki \
  -- du -sh /loki/*
```

### Application Integration

#### Add Loki Datasource in Grafana

```bash
# Port-forward Grafana
kubectl port-forward svc/grafana -n monitoring 3000:3000 &

# Access: http://localhost:3000
# Default: admin/admin

# Menu → Configuration → Data Sources → Add
# Name: Loki
# URL: http://loki.monitoring.svc.cluster.local:3100
# Access: Server (default)
# Skip TLS Verify: false (if cert valid)
# Save & Test

# Should show: "Data source is working"
```

#### Create Test Dashboard

```json
{
  "dashboard": {
    "title": "Loki - Service Logs",
    "panels": [
      {
        "title": "All Services",
        "type": "logs",
        "targets": [
          {
            "expr": "{job='neuectomy'}",
            "refId": "A"
          }
        ],
        "options": {
          "showTime": true,
          "showLabels": ["service", "level"]
        }
      }
    ]
  }
}
```

---

## Debugging Common Issues

### Issue 1: No logs appearing

#### Diagnose

```bash
# Check Promtail logs
kubectl logs -f daemonset/promtail -n monitoring | grep -i error

# Check Promtail config
kubectl get configmap promtail-config -n monitoring -o yaml

# Verify scrape targets
kubectl port-forward daemonset/promtail 9080:9080 -n monitoring
curl http://localhost:9080/scrape_configs
```

#### Solution

```bash
# Ensure applications have correct labels
kubectl get pods -n neurectomy -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.labels.app}{"\n"}{end}'

# Must show app labels like: ryot, sigmalang, sigmavault, agents

# If missing, add labels to deployment
kubectl patch deployment ryot -n neurectomy -p \
  '{"spec":{"template":{"metadata":{"labels":{"app":"ryot"}}}}}'
```

### Issue 2: Loki out of memory

#### Diagnose

```bash
kubectl top pod -l app=loki -n monitoring

# Should show: Loki < 2Gi usage
```

#### Solution

```bash
# Increase memory limit
kubectl set resources statefulset/loki \
  -n monitoring \
  -c loki \
  --limits=memory=8Gi

# Or reduce chunk_idle_period
kubectl edit configmap loki-config -n monitoring
# Change: chunk_idle_period: 1m (from 3m)

# Restart Loki
kubectl rollout restart statefulset/loki -n monitoring
```

### Issue 3: S3 connection errors

#### Diagnose

```bash
# Check credentials
kubectl get secret loki-s3-creds -n monitoring -o yaml

# Test S3 access from Loki pod
kubectl exec -it statefulset/loki-0 -n monitoring -c loki \
  -- aws s3 ls s3://loki-chunks/
```

#### Solution

```bash
# Update credentials
kubectl delete secret loki-s3-creds -n monitoring
kubectl create secret generic loki-s3-creds \
  -n monitoring \
  --from-literal=access_key=correct-key \
  --from-literal=secret_key=correct-secret

# Restart Loki to pick up new credentials
kubectl rollout restart statefulset/loki -n monitoring
```

### Issue 4: Slow queries

#### Diagnose

```bash
# Check query performance
kubectl port-forward svc/loki 3100:3100 -n monitoring

# Slow query
time curl -s 'http://localhost:3100/loki/api/v1/query_range' \
  --data-urlencode 'query={service="ryot"}' \
  --data-urlencode 'start=<timestamp_7days_ago>' \
  --data-urlencode 'end=<now>'
```

#### Solution

```bash
# Increase cache size
kubectl edit configmap loki-config -n monitoring

# In results_cache section:
# max_size_mb: 500  (from 100)

# Reduce query range (use filters)
# Good:  {service="ryot", level="error"} [1h]
# Bad:   {service="ryot"} [7d]

# Add label matchers before JSON parsing
# Good:  {service="ryot", level="error"} | json
# Bad:   {service="ryot"} | json | level="error"
```

---

## Monitoring Dashboard Setup

### Create Loki Health Dashboard

```bash
# Grafana Dashboard: Loki Operational
cat > /tmp/loki-dashboard.json <<'EOF'
{
  "dashboard": {
    "title": "Loki System Health",
    "panels": [
      {
        "title": "Log Ingestion Rate",
        "targets": [
          {
            "datasource": "Prometheus",
            "expr": "rate(loki_distributor_lines_received_total[5m])"
          }
        ]
      },
      {
        "title": "Chunk Size",
        "targets": [
          {
            "datasource": "Prometheus",
            "expr": "avg(loki_ingester_chunk_size_bytes)"
          }
        ]
      },
      {
        "title": "Query Latency (P99)",
        "targets": [
          {
            "datasource": "Prometheus",
            "expr": "histogram_quantile(0.99, rate(loki_request_duration_seconds_bucket{route=~\".*query.*\"}[5m]))"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "targets": [
          {
            "datasource": "Prometheus",
            "expr": "rate(cortex_memcache_request_cache_hits_total[5m]) / (rate(cortex_memcache_request_cache_hits_total[5m]) + rate(cortex_memcache_request_cache_misses_total[5m]))"
          }
        ]
      }
    ]
  }
}
EOF

# Import to Grafana
curl -X POST http://localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @/tmp/loki-dashboard.json
```

---

## Performance Tuning

### For Development (Small Cluster)

```yaml
# 1 Loki replica, 512Mi memory
replicas: 1
resources:
  requests:
    memory: 256Mi
    cpu: 100m
  limits:
    memory: 512Mi
    cpu: 500m

# Reduce retention
retention_period: 72h # 3 days

# Reduce chunk idle period
chunk_idle_period: 1m
```

### For Production (Large Cluster)

```yaml
# 3 Loki replicas, 8Gi memory each
replicas: 3
resources:
  requests:
    memory: 4Gi
    cpu: 2000m
  limits:
    memory: 8Gi
    cpu: 4000m

# Full retention
retention_period: 720h # 30 days

# Longer chunk idle
chunk_idle_period: 3m

# Increase distributor rate limits
rate_limit: 500000
rate_limit_burst: 1000000

# Larger cache
max_size_mb: 1000
```

---

## Scaling Guide

### Horizontal Scaling (Add More Loki Replicas)

```bash
# Current state
kubectl get statefulset loki -n monitoring
# replicas: 3

# Scale up to 5
kubectl scale statefulset loki --replicas=5 -n monitoring

# Monitor rollout
kubectl rollout status statefulset/loki -n monitoring
```

### Vertical Scaling (Increase Resources)

```bash
# Check current resource usage
kubectl top pod -l app=loki -n monitoring

# If > 80% memory, increase limits
kubectl set resources statefulset/loki \
  -c loki \
  --limits=memory=12Gi \
  --requests=memory=8Gi \
  -n monitoring

# Rolling restart (one pod at a time)
kubectl rollout restart statefulset/loki -n monitoring
```

---

## Backup & Recovery

### Backup Loki Data

```bash
# Backup S3 bucket
aws s3 sync s3://loki-chunks s3://loki-chunks-backup-$(date +%Y%m%d)

# Backup BoltDB indices
kubectl exec -it statefulset/loki-0 -n monitoring -c loki \
  -- tar -czf /tmp/loki-backup.tar.gz /loki/boltdb-shipper-active

kubectl cp monitoring/loki-0:/tmp/loki-backup.tar.gz /tmp/loki-backup.tar.gz -c loki
```

### Restore Loki Data

```bash
# Delete and recreate StatefulSet
kubectl delete statefulset loki -n monitoring
kubectl delete pvc -l app=loki -n monitoring

# Reapply manifests (will restore from backup)
kubectl apply -f 18-storage-cache-networking.yaml
kubectl apply -f 18-loki-configmap.yaml
kubectl apply -f 18-loki-deployment.yaml

# Restore S3 data
aws s3 sync s3://loki-chunks-backup-<date> s3://loki-chunks
```

---

## Upgrade Path (Phase 18E → Next Phase)

### Pre-Upgrade Checklist

```bash
# 1. Backup current data
kubectl exec -it statefulset/loki-0 -n monitoring \
  -c loki -- tar -czf /loki/pre-upgrade-backup.tar.gz /loki/

# 2. Document current state
kubectl describe statefulset loki -n monitoring > /tmp/loki-state.txt

# 3. Check Loki version
kubectl exec -it statefulset/loki-0 -n monitoring \
  -c loki -- loki -version
```

### Upgrade Process

```bash
# 1. Update image version in manifest
kubectl edit statefulset loki -n monitoring
# Change: image: grafana/loki:2.9.4 → 3.0.0

# 2. Rolling update (automatic for StatefulSet)
kubectl rollout status statefulset/loki -n monitoring

# 3. Verify after upgrade
curl http://localhost:3100/ready
curl http://localhost:3100/api/v1/labels
```

---

## Next Steps

1. **Configure Retention Policies** - Adjust retention per service
2. **Set Up Alerts** - Create error rate and latency alerts
3. **Integration Testing** - Verify logs from all 4 services
4. **Team Training** - Teach team LogQL query syntax
5. **On-Call Runbooks** - Create debugging playbooks

See [PHASE-18E-LOKI-INTEGRATION.md](PHASE-18E-LOKI-INTEGRATION.md) for comprehensive reference.
