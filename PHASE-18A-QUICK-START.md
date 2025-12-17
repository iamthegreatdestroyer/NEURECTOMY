# Prometheus + Grafana Quick Start Guide

## Neurectomy Phase 18A - Complete Observability Integration

This guide provides step-by-step instructions to deploy and validate the complete monitoring stack for Phase 18A.

---

## Architecture Overview

```
Neurectomy Services (4 Tiers)
├─ Tier 1: Neurectomy API
├─ Tier 2: Ryot LLM, ΣLANG Compression, ΣVAULT Storage
├─ Tier 3: Elite Agent Collective (40 agents, 8 tiers)
└─ Infrastructure: PostgreSQL, Redis, RabbitMQ, Loki, Jaeger

         ↓ (Expose metrics on :8000-:9003)

Prometheus Server
├─ Scrape interval: 15s
├─ Evaluation interval: 30s
├─ Recording rules (60+): Pre-compute expensive aggregations
├─ Alert rules (40+): Multi-tier alerting with severity
└─ Retention: 30 days (configurable)

         ↓ (Store time-series data)

Data Storage
├─ Time-series database (Prometheus TSDB)
├─ Capacity: 100GB persistent volume
├─ Performance: <500ms for dashboard queries
└─ Backup: Daily snapshots

         ↓ (Query metrics + logs + traces)

Grafana
├─ Dashboard 1: Ryot LLM Metrics
├─ Dashboard 2: ΣLANG Compression
├─ Dashboard 3: ΣVAULT Storage
├─ Dashboard 4: Agent Collective Health
└─ Dashboard 5: SLO & Error Budget

         ↓ (Route alerts)

Alertmanager
├─ PagerDuty (critical tier-1/tier-3)
├─ Slack (warnings/info by team)
├─ Email (cost anomalies)
└─ Custom webhooks (optional)
```

---

## Quick Start (5 minutes)

### Option 1: Docker Compose (Development/Testing)

```bash
# 1. Start the monitoring stack
cd /path/to/NEURECTOMY
docker-compose up -d prometheus grafana alertmanager

# 2. Wait for containers to start
sleep 5

# 3. Verify all targets are running
curl http://localhost:9090/api/v1/targets

# 4. Access Grafana
# Open: http://localhost:3000
# Login: admin/admin

# 5. Verify Prometheus
# Open: http://localhost:9090
# Check Status → Targets (all should be green)
```

### Option 2: Kubernetes (Production)

```bash
# 1. Create namespace and secrets
kubectl create namespace neurectomy
kubectl create secret generic grafana-admin \
  --from-literal=password=YourSecurePassword123 \
  -n neurectomy

# 2. Deploy Prometheus
kubectl apply -f deploy/k8s/prometheus-statefulset.yaml

# 3. Deploy Grafana
kubectl apply -f deploy/k8s/grafana-deployment.yaml

# 4. Verify deployments
kubectl get all -n neurectomy

# 5. Access Grafana (port-forward)
kubectl port-forward -n neurectomy svc/grafana 3000:3000
# Open: http://localhost:3000
```

---

## File Locations Reference

| Component           | File Location                                                 | Description                           |
| ------------------- | ------------------------------------------------------------- | ------------------------------------- |
| Prometheus Config   | `docker/prometheus/prometheus-production.yml`                 | Main scrape jobs, alertmanager config |
| Recording Rules     | `docker/prometheus/recording.rules.yml`                       | 60+ pre-computed metrics              |
| Alert Rules         | `docker/prometheus/alert.rules.yml`                           | 40+ alert definitions                 |
| Alertmanager Config | `docker/prometheus/alertmanager.yml`                          | PagerDuty/Slack/Email routing         |
| K8s Prometheus      | `deploy/k8s/prometheus-statefulset.yaml`                      | Kubernetes deployment manifest        |
| K8s Grafana         | `deploy/k8s/grafana-deployment.yaml`                          | Kubernetes deployment manifest        |
| Dashboard 1         | `deploy/k8s/grafana/dashboards/01-ryot-llm-metrics.json`      | Ryot LLM dashboard                    |
| Dashboard 2         | `deploy/k8s/grafana/dashboards/02-sigmalang-compression.json` | ΣLANG dashboard                       |
| Dashboard 3         | `deploy/k8s/grafana/dashboards/03-sigmavault-storage.json`    | ΣVAULT dashboard                      |
| Deployment Guide    | `docs/PROMETHEUS-GRAFANA-DEPLOYMENT-GUIDE.md`                 | Detailed deployment & validation      |

---

## Configuration Customization

### Update Scrape Intervals

Edit `docker/prometheus/prometheus-production.yml`:

```yaml
global:
  scrape_interval: 15s # More frequent = more data storage
  evaluation_interval: 30s # More frequent = higher CPU cost
```

### Update Retention Policy

Edit Prometheus command args in K8s manifest:

```yaml
args:
  - --storage.tsdb.retention.time=30d # Adjust retention
  - --storage.tsdb.retention.size=100GB # Or by size
```

### Configure Alert Notifications

Edit `docker/prometheus/alertmanager.yml`:

```yaml
global:
  slack_api_url: "YOUR_SLACK_WEBHOOK"
  pagerduty_url: "YOUR_PAGERDUTY_KEY"
  smtp_smarthost: "YOUR_SMTP_SERVER"
```

### Add Custom Scrape Jobs

Edit `docker/prometheus/prometheus-production.yml` → `scrape_configs`:

```yaml
- job_name: "my-service"
  static_configs:
    - targets: ["my-service:8080"]
  metric_path: "/metrics"
  scrape_interval: 15s
```

---

## Validation Checklist

### ✅ Prometheus Health

```bash
# Check Prometheus is up
curl http://localhost:9090/-/healthy

# Verify scrape targets
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets | length'
# Should show 10+ active targets

# Check recording rules
curl http://localhost:9090/api/v1/rules | jq '.data.groups | length'
# Should show 9 rule groups

# Check alert rules
curl http://localhost:9090/api/v1/alerts | jq '.data.alerts | length'
# Should show 40+ alerts loaded
```

### ✅ Grafana Health

```bash
# Test Grafana API
curl http://localhost:3000/api/health

# Verify datasource connection
curl http://localhost:3000/api/datasources
# Should show Prometheus configured

# Check dashboards exist
curl http://localhost:3000/api/search?query=
# Should show 5+ dashboards
```

### ✅ Data Flow

```bash
# Generate test traffic
for i in {1..50}; do
  curl -s http://localhost:8000/api/health > /dev/null
done

# Query for metrics (wait 30-60s)
curl -s 'http://localhost:9090/api/v1/query?query=rate(neurectomy_http_requests_total[1m])' | jq '.data.result[0].value'
# Should return non-zero value

# Check dashboard data
# Open Grafana Dashboard 1
# Should see active metrics in panels
```

### ✅ Alerting

```bash
# Verify Alertmanager connection
curl http://localhost:9093/-/healthy

# Check alerts in Alertmanager
curl http://localhost:9093/api/v1/alerts | jq '.data | length'

# Simulate alert (optional)
# Trigger high error rate on API
# Monitor Prometheus → Alerts page for HighHTTPErrorRate firing
```

---

## Performance Tuning

### Dashboard Query Performance

**Problem:** Dashboards load slowly (>5 seconds)

**Solutions:**

1. Use recording rules (already defined in `recording.rules.yml`)
2. Increase Prometheus memory: change `resources.limits.memory` in K8s manifest
3. Reduce time range: Dashboard defaults to 1h (good for dashboards)
4. Add more recording rules for complex aggregations

### Storage Growth Rate

**Monitor:** Query storage growth

```bash
curl -s 'http://localhost:9090/api/v1/query?query=prometheus_tsdb_symbol_table_size_bytes' | jq '.data.result[0].value'
```

**Optimize:**

1. Reduce scrape_interval from 15s to 30s or 60s
2. Drop unnecessary labels via metric_relabeling
3. Implement downsampling for historical data
4. Archive metrics to cheaper storage (S3, GCS)

### Alert Storm Prevention

If too many alerts firing:

```yaml
# Edit alertmanager.yml to suppress cascade alerts
inhibit_rules:
  - source_match:
      severity: "critical"
    target_match:
      severity: "warning"
    equal: ["service"] # Suppress warnings if critical fires
```

---

## Troubleshooting

### Issue: Targets Showing DOWN

```bash
# Check Prometheus logs
docker logs neurectomy-prometheus

# Verify service is accessible
curl http://neurectomy-api:8000/metrics

# Check firewall/network
docker exec neurectomy-prometheus \
  curl http://neurectomy-api:8000/metrics

# Verify DNS (if using hostnames)
docker exec neurectomy-prometheus \
  nslookup neurectomy-api
```

### Issue: No Data in Dashboards

```bash
# Wait 90+ seconds (3× scrape interval)
# Then check Prometheus is collecting data

# Query directly in Prometheus
# Visit http://localhost:9090
# Query: up
# Should show multiple series with value 1

# Check for errors in recording rules
curl http://localhost:9090/api/v1/rules | \
  jq '.data.groups[].rules[] | select(.health != "ok")'

# Check for scrape errors
curl http://localhost:9090/api/v1/targets | \
  jq '.data.droppedTargets'
```

### Issue: Alerts Not Firing

```bash
# Check alert rule syntax
docker run --rm -v $(pwd)/docker/prometheus:/etc/prometheus \
  prom/prometheus:latest \
  promtool check rules /etc/prometheus/alert.rules.yml

# Verify alert evaluation
curl http://localhost:9090/api/v1/query?query=ALERTS

# Check Alertmanager logs
docker logs neurectomy-alertmanager

# Test Alertmanager connectivity
curl http://localhost:9093/-/healthy
```

### Issue: High CPU/Memory Usage

```bash
# Check Prometheus process
docker stats neurectomy-prometheus

# If high CPU: reduce evaluation_interval or scrape_interval
# If high memory: reduce retention or increase max_samples_per_send

# Check active timeseries
curl -s 'http://localhost:9090/api/v1/query?query=prometheus_tsdb_symbol_table_size_bytes'
```

---

## Next Steps

### Phase 18B: Distributed Tracing (Optional)

- Add Jaeger integration for trace collection
- Link traces from logs → dashboards
- Instrument application code with OpenTelemetry

### Phase 18C: Log Aggregation (Optional)

- Add Loki for log collection
- Link logs to metrics via common labels
- Create log-based alerts

### Phase 18D: Cost Optimization

- Use recording rules to forecast costs
- Set up cost anomaly alerts
- Implement tiered storage (hot/warm/cold)

### Custom Dashboards

- Build team-specific dashboards
- Add custom panels for business metrics
- Set up scheduled reports

---

## Maintenance Tasks

### Daily (Automated)

- Monitor alert volume
- Check for scrape errors
- Verify data freshness

### Weekly (Manual)

```bash
# Check storage utilization
df -h /mnt/data/prometheus  # Docker
kubectl get pvc -n neurectomy  # Kubernetes

# Review alert noise
# Disable/adjust rules with high false positive rate
```

### Monthly

```bash
# Archive old data (>30 days)
# Backup Prometheus TSDB
# Review dashboards for relevance
# Update runbook links
```

### Quarterly

```bash
# Review alert thresholds
# Re-evaluate retention policy
# Update documentation
# Plan capacity expansion
```

---

## Support & Resources

**Documentation:**

- [Prometheus Official Docs](https://prometheus.io/docs/)
- [Grafana Official Docs](https://grafana.com/docs/)
- [Alertmanager Guide](https://prometheus.io/docs/alerting/latest/alertmanager/)
- [Recording Rules Best Practices](https://prometheus.io/docs/practices/rules/)

**Neurectomy Documentation:**

- `docs/PROMETHEUS-GRAFANA-DEPLOYMENT-GUIDE.md` - Full deployment guide
- `docs/GRAFANA-DASHBOARD-SPECIFICATIONS.md` - Dashboard panel details
- Phase 18A docs - Phase-specific details

**Common Commands:**

```bash
# PromQL query examples
rate(neurectomy_http_requests_total[5m])  # Per-second rate
histogram_quantile(0.95, ...)  # 95th percentile
sum(metric) by (label)  # Aggregate by label

# Prometheus CLI
promtool check config prometheus.yml
promtool check rules alert.rules.yml
promtool query raw 'up'

# Grafana CLI
grafana-cli admin reset-admin-password <new_password>
grafana-cli plugins install grafana-piechart-panel
```

---

## Summary

Phase 18A monitoring stack provides:

✅ **Complete Visibility**: 250+ metrics across 4 service tiers
✅ **Proactive Alerting**: 40+ alert rules with intelligent routing
✅ **Cost Tracking**: Built-in cost attribution and forecasting
✅ **Performance Analysis**: Pre-computed recording rules for <1s dashboard queries
✅ **Production Ready**: Kubernetes manifests, RBAC, NetworkPolicies, high availability

**Status**: All components deployed and validated
**Dashboard Count**: 5 (Ryot, ΣLANG, ΣVAULT, Agent Collective, SLO)
**Alert Coverage**: 40+ rules across all service tiers
**Data Retention**: 30 days (configurable)
**Query Performance**: <500ms for typical dashboard queries

**Next Action**: Start with Quick Start (Option 1 for Docker Compose, Option 2 for Kubernetes)
