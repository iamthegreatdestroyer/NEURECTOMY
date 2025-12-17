# Phase 18: Quick Start Guide

## What is Phase 18?

**Production Monitoring & Observability** - Comprehensive visibility into your running Neurectomy system.

After deploying (Phase 14), hardening (Phase 15), building ecosystem (Phase 16), and integrating (Phase 17), Phase 18 gives you eyes on everything running in production.

## The 3 Pillars of Observability

### 1. Metrics (What's happening?)

- Request rates, latency, errors
- Resource usage (CPU, memory, disk)
- Business metrics (users, tokens, cost)
- **Tool**: Prometheus + Grafana

### 2. Traces (Why is it slow?)

- Request flow across services
- Bottleneck identification
- Latency breakdown
- **Tool**: OpenTelemetry + Jaeger

### 3. Logs (What went wrong?)

- Centralized log aggregation
- Search and analysis
- Error tracking
- **Tool**: Loki/ELK Stack

## Phase 18 in 3 Steps

### Step 1: Deploy Monitoring (Day 1-2) ✅

```bash
# Deploy Prometheus + Grafana
bash infrastructure/monitoring/install.sh

Result: Monitoring stack running
```

### Step 2: Add Metrics (Day 3-5) ✅ NEURECTOMY COMPLETE

```bash
# Metrics already implemented:
# - neurectomy/monitoring/metrics.py (Phase 18A-2) ✅
#
# Next phases:
# - Ryot LLM metrics (Phase 18A-3)
# - ΣLANG metrics (Phase 18A-4)
# - ΣVAULT metrics (Phase 18A-5)
# - Agent metrics (Phase 18A-6)

Result: All services emitting metrics
```

### Step 3: View Dashboards (Day 5+)

```bash
# Access Grafana
kubectl port-forward -n monitoring svc/grafana 3000:80

# Open: http://localhost:3000
# Username: admin
# Password: (as configured during install)

Result: Real-time visibility into your system
```

## Key Metrics to Watch

### Health Indicators

- ✅ **Request Rate**: Steady = healthy, spiky = issues
- ✅ **Error Rate**: < 1% = good, > 5% = alert
- ✅ **Latency (p95)**: < 1s = fast, > 5s = slow
- ✅ **CPU Usage**: < 70% = headroom, > 90% = scale

### Performance Metrics

- **Tokens/second** (Ryot LLM): Target 10+
- **Compression ratio** (ΣLANG): Target 15x+
- **Storage latency** (ΣVAULT): Target < 100ms
- **Agent utilization**: Target 60-80%

### Business Metrics

- **Active users**: Growth trend
- **API calls/day**: Usage patterns
- **Cost per request**: Optimization target
- **Token consumption**: Billing driver

## What You'll See in Grafana

### Dashboard 1: System Overview

- Total requests/sec across all services
- Overall error rate
- p95 latency
- Resource usage (CPU, memory, disk)

### Dashboard 2: Service Health

- Individual service metrics
- Circuit breaker states
- Retry attempts
- Fallback triggers

### Dashboard 3: Business Metrics

- Active users
- Tokens generated
- Compression savings
- Cost tracking

### Dashboard 4: Agent Collective

- 40 agent health status
- Task distribution
- Agent utilization
- Recovery events

## Alerting Rules (Added in Phase 18D)

```yaml
# High error rate
ALERT HighErrorRate
  IF rate(http_requests_total{status=~"5.."}[5m]) > 0.05
  FOR 5m
  LABELS { severity = "critical" }

# High latency
ALERT HighLatency
  IF histogram_quantile(0.95, http_request_duration_seconds) > 5
  FOR 10m
  LABELS { severity = "warning" }

# Service down
ALERT ServiceDown
  IF up == 0
  FOR 1m
  LABELS { severity = "critical" }
```

## Files Created in Phase 18

```
infrastructure/
├── monitoring/
│   ├── prometheus-config.yaml
│   ├── install.sh
│   └── dashboards/
│       ├── system-overview.json
│       ├── service-health.json
│       └── business-metrics.json
└── kubernetes/
    └── monitoring/
        ├── prometheus-deployment.yaml
        └── grafana-deployment.yaml

neurectomy/
└── monitoring/
    └── metrics.py

ryot/
└── monitoring/
    └── metrics.py

sigmalang/
└── monitoring/
    └── metrics.py

sigmavault/
└── monitoring/
    └── metrics.py

agents/
└── monitoring/
    └── metrics.py
```

## Common Issues & Solutions

### Issue: Prometheus not scraping metrics

**Solution**: Check pod annotations:

```yaml
annotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "9090"
  prometheus.io/path: "/metrics"
```

### Issue: Grafana dashboard empty

**Solution**: Verify Prometheus datasource connected

### Issue: Metrics endpoint returns 404

**Solution**: Ensure `app.mount("/metrics", metrics_app)` in main.py

## Time Investment

- **Phase 18A (Metrics)**: 3-4 days
- **Phase 18B (Tracing)**: 2-3 days
- **Phase 18C (Logging)**: 2-3 days
- **Phase 18D (Alerting)**: 2-3 days
- **Phase 18E (Optimization)**: 2-3 days
- **Total**: 11-16 days (2-3 weeks)

## Success Checklist

After Phase 18, you should have:

- [ ] Prometheus collecting metrics from all services
- [ ] Grafana dashboards showing real-time data
- [ ] Alerts configured for critical conditions
- [ ] Distributed tracing capturing request flows
- [ ] Centralized logging operational
- [ ] Performance profiling tools available
- [ ] Cost tracking dashboards active
- [ ] Incident response playbooks ready

## What Comes Next?

### Phase 19: Advanced Features

- A/B testing
- Feature flags
- Canary deployments

### Phase 20: AI/ML Ops

- Model monitoring
- Training pipelines
- Drift detection

### Phase 21: Multi-Region

- Geographic distribution
- Disaster recovery
- Global load balancing

Ready to start? Begin with **PHASE-18A-1-PROMETHEUS-SETUP.md**!
