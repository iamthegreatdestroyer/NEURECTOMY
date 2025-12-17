# Kubernetes Deployment Guide for Neurectomy Phase 18A Monitoring Stack

## Overview

This comprehensive Kubernetes deployment provides a production-grade monitoring stack with:

- **High Availability**: 2 Prometheus replicas, 2 Grafana replicas, 3 AlertManager replicas
- **Security**: NetworkPolicies, RBAC, Pod Security Policies, TLS encryption
- **Reliability**: Pod Disruption Budgets, cross-node affinity, graceful shutdowns
- **Observability**: Comprehensive metric collection and alerting

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Neurectomy Services (4 Tiers)              │
│  ├─ Tier 1: API (port 8000)                             │
│  ├─ Tier 2: Ryot LLM, ΣLANG, ΣVAULT (8001-9000)        │
│  └─ Tier 3: Elite Agent Collective                      │
└────────────────┬────────────────────────────────────────┘
                 │ (Expose metrics)
┌────────────────▼────────────────────────────────────────┐
│         Prometheus (2 replicas, 200GB SSD)              │
│  ├─ Scrape interval: 15s                                │
│  ├─ Recording rules: 20+ pre-computed aggregations      │
│  └─ Alert rules: 40+ severity-based alerts              │
└────────────────┬────────────────────────────────────────┘
                 │ (Query metrics)
    ┌────────────┼────────────┐
    │            │            │
┌───▼──┐    ┌───▼──┐    ┌────▼────┐
│Grafana   │AlertMgr│   │Config
│(2x)      │ (3x)   │    │Management
└──────┘    └────────┘   └─────────┘
```

## Prerequisites

### Kubernetes Cluster

- Version: 1.24+
- Nodes: 3+ with 4GB+ RAM each
- Storage: 200GB+ available for Prometheus TSDB

### Tools

- `kubectl` (1.24+)
- `helm` (optional, for package management)
- `kustomize` (optional, for configuration management)

### Cloud Provider Support

- **AWS**: EBS volumes, LoadBalancer services
- **GCP**: GCE persistent disks, Ingress
- **Azure**: Azure Disks, Application Gateway

## Installation Steps

### 1. Prerequisites Setup

```bash
# Configure kubectl access
export KUBECONFIG=/path/to/config.yaml
kubectl cluster-info  # Verify connection

# Create logging directory
mkdir -p logs

# Install deployment script
chmod +x deploy/scripts/deploy-monitoring.sh
```

### 2. Configure Secrets

Edit `deploy/k8s/06-secrets.yaml` and update:

- Grafana admin password
- AlertManager webhook URLs (Slack, PagerDuty)
- Basic auth credentials for Prometheus/AlertManager

```bash
# Update secrets
kubectl create secret generic grafana-admin \
  --from-literal=admin-user=admin \
  --from-literal=admin-password='your-secure-password' \
  -n monitoring --dry-run=client -o yaml | kubectl apply -f -
```

### 3. Deploy Monitoring Stack

```bash
# Validate manifests
./deploy/scripts/deploy-monitoring.sh validate staging

# Deploy to staging
./deploy/scripts/deploy-monitoring.sh deploy staging

# Deploy to production
./deploy/scripts/deploy-monitoring.sh deploy production
```

### 4. Verify Deployment

```bash
# Check pod status
./deploy/scripts/health-check.sh

# Verbose health check
./deploy/scripts/health-check.sh verbose

# Check logs
kubectl logs -n monitoring -l app=prometheus -f
kubectl logs -n monitoring -l app=grafana -f
kubectl logs -n monitoring -l app=alertmanager -f
```

## Service Access

### Internal (ClusterIP)

- **Prometheus**: `http://prometheus:9090`
- **Grafana**: `http://grafana:3000`
- **AlertManager**: `http://alertmanager:9093`

### External via Ingress (with TLS)

- **Prometheus**: `https://prometheus.neurectomy.local` (requires basic auth)
- **Grafana**: `https://grafana.neurectomy.local`
- **AlertManager**: `https://alertmanager.neurectomy.local` (requires basic auth)

### Port Forwarding (for development)

```bash
# Prometheus
kubectl port-forward -n monitoring svc/prometheus 9090:9090

# Grafana
kubectl port-forward -n monitoring svc/grafana 3000:3000

# AlertManager
kubectl port-forward -n monitoring svc/alertmanager 9093:9093
```

## Configuration Management

### Update Prometheus Configuration

```yaml
# Edit ConfigMap
kubectl edit configmap prometheus-config -n monitoring

# Verify changes
kubectl rollout restart statefulset/prometheus -n monitoring
kubectl rollout status statefulset/prometheus -n monitoring --timeout=300s
```

### Add Alert Rules

```yaml
# Edit alert rules ConfigMap
kubectl edit configmap prometheus-alert-rules -n monitoring

# Verify Prometheus reloads config
kubectl exec -n monitoring prometheus-0 -- \
  curl -X POST http://localhost:9090/-/reload
```

### Update AlertManager Configuration

```bash
# Edit secret
kubectl edit secret alertmanager-config -n monitoring

# Verify configuration
kubectl exec -n monitoring alertmanager-0 -- \
  amtool check-config /etc/alertmanager/alertmanager.yml

# Reload AlertManager
kubectl exec -n monitoring alertmanager-0 -- \
  curl -X POST http://localhost:9093/-/reload
```

## High Availability Setup

### Cross-Node Affinity

Pods are configured with `podAntiAffinity` to spread across nodes:

```yaml
affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        podAffinityTerm:
          labelSelector:
            matchExpressions:
              - key: app
                operator: In
                values:
                  - prometheus
          topologyKey: kubernetes.io/hostname
```

### Pod Disruption Budgets

Ensure minimum replicas remain during node maintenance:

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: prometheus-pdb
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: prometheus
```

### Scaling Considerations

```bash
# Scale Prometheus (StatefulSet)
kubectl scale statefulset prometheus -n monitoring --replicas=3

# Scale Grafana (Deployment)
kubectl scale deployment grafana -n monitoring --replicas=3

# Scale AlertManager (StatefulSet with clustering)
kubectl scale statefulset alertmanager -n monitoring --replicas=5
```

## Backup & Recovery

### Create Backup

```bash
# Full backup including TSDB
./deploy/scripts/backup-restore.sh backup

# Backup location
ls -la backups/full_*/

# List available backups
./deploy/scripts/backup-restore.sh list

# Verify backup integrity
./deploy/scripts/backup-restore.sh verify backups/full_YYYYMMDDhhmmss.tar.gz
```

### Restore from Backup

```bash
# Restore full stack
./deploy/scripts/backup-restore.sh restore backups/full_YYYYMMDDhhmmss.tar.gz

# Monitor restoration
kubectl rollout status -n monitoring --all

# Verify restoration
./deploy/scripts/health-check.sh verbose
```

### Automated S3 Backup

```bash
# Set up S3 credentials
export AWS_ACCESS_KEY_ID=xxx
export AWS_SECRET_ACCESS_KEY=xxx
export S3_BACKUP_BUCKET=neurectomy-backups

# Sync to S3
./deploy/scripts/backup-restore.sh s3-sync

# Set up daily backup cronjob
0 2 * * * /path/to/backup-restore.sh backup && /path/to/backup-restore.sh s3-sync
```

## Disaster Recovery

### RTO: 15 minutes | RPO: 1 hour

#### Scenario 1: Pod Failure

- **Automatic**: Kubernetes reschedules pod
- **Recovery Time**: < 30 seconds
- **Data Loss**: None (PVC persists)

```bash
# Monitor recovery
kubectl get pods -n monitoring -w
```

#### Scenario 2: Node Failure

- **Automatic**: Pod rescheduled to healthy node (pod anti-affinity)
- **Recovery Time**: 2-5 minutes
- **Data Loss**: None (EBS volumes replicate)

```bash
# Force rescheduling if needed
kubectl delete pod <pod-name> -n monitoring
```

#### Scenario 3: Storage Failure

- **Action**: Restore from backup
- **Recovery Time**: 10-15 minutes
- **Backup Location**: S3 with hourly sync

```bash
# Restore specific component
./deploy/scripts/backup-restore.sh restore <backup-file>
```

#### Scenario 4: Complete Cluster Loss

- **Action**: Redeploy from manifests + restore backups
- **Recovery Time**: 15 minutes
- **Procedure**:
  1. Deploy new cluster
  2. Run `deploy-monitoring.sh deploy production`
  3. Restore from backup

```bash
# Full redeploy and restore
./deploy/scripts/deploy-monitoring.sh deploy production
./deploy/scripts/backup-restore.sh restore <latest-backup>
```

## Security Hardening

### Network Policies

Restrict ingress/egress traffic:

```bash
# Verify network policies
kubectl get networkpolicies -n monitoring

# Test connectivity
kubectl exec -n monitoring prometheus-0 -- \
  curl http://grafana:3000/api/health
```

### RBAC

Minimal permissions per component:

```bash
# Verify RBAC
kubectl get roles,rolebindings,clusterroles,clusterrolebindings -n monitoring

# Test permissions
kubectl auth can-i get pods --as=system:serviceaccount:monitoring:prometheus
```

### Pod Security Policies

Enforce security standards:

```bash
# Verify PSP
kubectl get psp monitoring-restricted -o yaml

# Check pod compliance
kubectl get pods -n monitoring -o jsonpath='{.items[*].spec.securityContext}'
```

### TLS/HTTPS

All external connections encrypted:

```bash
# Verify TLS certificates
kubectl get ingress -n monitoring -o yaml

# Check certificate expiry
kubectl get certificate -n monitoring
```

## Resource Utilization

### Default Resource Requests/Limits

| Component    | CPU Request | CPU Limit | Memory Request | Memory Limit |
| ------------ | ----------- | --------- | -------------- | ------------ |
| Prometheus   | 2           | 4         | 8Gi            | 16Gi         |
| Grafana      | 1           | 2         | 2Gi            | 4Gi          |
| AlertManager | 500m        | 1         | 1Gi            | 2Gi          |

### Scaling Guidelines

#### High Load (100k+ metrics/sec)

```yaml
- Prometheus: 8 CPU, 32GB RAM
- Grafana: 4 CPU, 8GB RAM
- AlertManager: 2 CPU, 4GB RAM
- Storage: 500GB+ SSD
```

#### Ultra High Load (1M+ metrics/sec)

```yaml
- Prometheus: 16+ CPU, 64GB+ RAM (consider sharding)
- Grafana: 8 CPU, 16GB RAM
- AlertManager: 4 CPU, 8GB RAM
- Storage: 1TB+ SSD (distributed)
```

## Performance Tuning

### Prometheus Query Optimization

```yaml
# Reduce query timeout
--query.timeout=5m

# Increase max samples per query
--query.max-samples=100000000

# Enable query log
--log.format=json
```

### Grafana Caching

```yaml
- name: GF_USERS_AUTO_ASSIGN_ORG_ROLE
  value: Admin
- name: GF_CACHE_DEFAULT_MAX_AGE_SECONDS
  value: "600"
```

### AlertManager Batching

```yaml
group_wait: 10s # Wait to batch alerts
group_interval: 10s # Interval between groups
repeat_interval: 12h # How often to send existing alerts
```

## Troubleshooting

### Prometheus Not Scraping Targets

```bash
# Check service discovery
kubectl exec -n monitoring prometheus-0 -- \
  curl -s http://localhost:9090/service-discovery

# Check targets status
kubectl exec -n monitoring prometheus-0 -- \
  curl -s http://localhost:9090/api/v1/targets | jq
```

### Grafana Not Connecting to Prometheus

```bash
# Test connectivity from Grafana pod
kubectl exec -n monitoring -it <grafana-pod> -- \
  curl -s http://prometheus:9090/-/ready

# Check datasource configuration
kubectl get secret grafana-datasources -n monitoring -o yaml
```

### AlertManager Not Sending Alerts

```bash
# Check AlertManager config
kubectl exec -n monitoring alertmanager-0 -- \
  amtool check-config /etc/alertmanager/alertmanager.yml

# Test webhook
kubectl exec -n monitoring alertmanager-0 -- \
  curl -X POST http://webhook-receiver:5001/alert
```

### Out of Disk Space

```bash
# Check PVC utilization
kubectl exec -n monitoring prometheus-0 -- \
  df -h /prometheus

# Reduce retention
kubectl set env statefulset/prometheus -n monitoring \
  PROMETHEUS_RETENTION_TIME=15d

# Or increase PVC size
kubectl patch pvc prometheus-storage -n monitoring \
  -p '{"spec":{"resources":{"requests":{"storage":"300Gi"}}}}'
```

## Upgrades

### In-Place Upgrade

```bash
# Update image in StatefulSet/Deployment
kubectl set image statefulset/prometheus \
  prometheus=prom/prometheus:v2.49.0 \
  -n monitoring

# Monitor rollout
kubectl rollout status statefulset/prometheus -n monitoring --timeout=600s

# Verify functionality
./deploy/scripts/health-check.sh
```

### Blue-Green Upgrade

```bash
# Deploy new version alongside old
kubectl apply -f deploy/k8s/08-prometheus-statefulset-v2.yaml

# Verify new version
kubectl logs -n monitoring prometheus-new-0

# Switch traffic
kubectl patch service prometheus -n monitoring \
  -p '{"spec":{"selector":{"version":"v2"}}}'

# Remove old version
kubectl delete statefulset prometheus -n monitoring
```

## Cleanup

### Remove Monitoring Stack

```bash
# Destroy with backup
./deploy/scripts/deploy-monitoring.sh destroy

# Manual cleanup if needed
kubectl delete namespace monitoring
```

## Support & Documentation

- **Prometheus Docs**: https://prometheus.io/docs/
- **Grafana Docs**: https://grafana.com/docs/grafana/
- **AlertManager Docs**: https://prometheus.io/docs/alerting/latest/overview/
- **Kubernetes Docs**: https://kubernetes.io/docs/

## Best Practices

1. **Monitor the Monitors**: Set up alerts on Prometheus/AlertManager health
2. **Regular Backups**: Automated daily backups to S3
3. **Change Control**: All changes via Helm/Kustomize with git tracking
4. **Testing**: Validate all config changes in staging first
5. **Documentation**: Keep runbooks for common issues
6. **Capacity Planning**: Monitor growth and plan for scaling
7. **Security**: Regular security audits and patch management
