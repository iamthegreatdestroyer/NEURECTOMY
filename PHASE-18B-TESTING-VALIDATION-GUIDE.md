# SLO Monitoring Stack - Testing & Validation Guide

## Phase 18B: Complete Alert Routing & SLO Validation

---

## 1. ALERT ROUTING VALIDATION

### 1.1 Critical Alert Routing Test

**Objective:** Verify critical alerts route to PagerDuty + Slack + Email

#### Test Procedure:

```bash
# SSH into Prometheus container
kubectl exec -it prometheus-0 -n neurectomy -- /bin/sh

# Create a test alert rule (temporary):
cat > /etc/prometheus/alert_test_critical.yml << 'EOF'
groups:
  - name: test_critical
    interval: 10s
    rules:
      - alert: TestCriticalAlert
        expr: 1
        labels:
          severity: critical
          alertname: TestCriticalAlert
          component: test
        annotations:
          summary: "Test Critical Alert for Routing Verification"
          description: "This alert should route to PagerDuty + Slack + Email"
EOF

# Reload Prometheus config
curl -X POST http://localhost:9090/-/reload
```

**Expected Results:**

1. ✅ PagerDuty incident created within 10 seconds
2. ✅ Slack message in critical channel within 10s (group_wait=10s)
3. ✅ Email received at admin email within 2 minutes
4. ✅ AlertManager shows alert in UI at http://alertmanager.neurectomy.local:9093

**Slack Message Format:**

```
🚨 CRITICAL: Test Critical Alert
Component: test
Summary: Test Critical Alert for Routing Verification
Description: This alert should route to PagerDuty + Slack + Email
```

**PagerDuty Event:**

- Title: `TestCriticalAlert - test`
- Severity: `critical`
- Timestamp: [current time]
- Additional Details: All labels and annotations

---

### 1.2 Warning Alert Routing Test

**Objective:** Verify warning alerts route to component-specific Slack channels

#### Test Procedure:

```bash
# Create test warning alert for each component
cat > /etc/prometheus/alert_test_warning.yml << 'EOF'
groups:
  - name: test_warning
    interval: 10s
    rules:
      - alert: TestWarningRyot
        expr: 1
        labels:
          severity: warning
          component: ryot
        annotations:
          summary: "Test Warning Alert for Ryot"

      - alert: TestWarningSigmaLang
        expr: 1
        labels:
          severity: warning
          component: sigmalang
        annotations:
          summary: "Test Warning Alert for ΣLANG"

      - alert: TestWarningSigmaVault
        expr: 1
        labels:
          severity: warning
          component: sigmavault
        annotations:
          summary: "Test Warning Alert for ΣVAULT"
EOF

curl -X POST http://localhost:9090/-/reload
```

**Expected Results:**

1. ✅ Ryot warning → `#neurectomy-alerts` Slack channel
2. ✅ ΣLANG warning → `#neurectomy-alerts` Slack channel
3. ✅ ΣVAULT warning → `#neurectomy-alerts` Slack channel
4. ⏱️ Alert grouped with similar warnings (group_wait=1m)
5. ⏱️ Repeat every 4 hours (group_interval=5m, repeat_interval=4h)

---

### 1.3 SLO Burn Rate Alert Routing Test

**Objective:** Verify SLO burn rate alerts route correctly (fast/slow burn separation)

#### Test Procedure:

```bash
# Create test SLO burn alerts
cat > /etc/prometheus/alert_test_slo.yml << 'EOF'
groups:
  - name: test_slo
    interval: 10s
    rules:
      # Fast burn (should route to slo-burn-fast receiver)
      - alert: TestSLOFastBurn
        expr: 1
        labels:
          severity: critical
          slo_burn_type: "fast"
          component: "ryot"
        annotations:
          summary: "Test SLO Fast Burn Alert"

      # Slow burn (should route to slo-burn-slow receiver)
      - alert: TestSLOSlowBurn
        expr: 1
        labels:
          severity: warning
          slo_burn_type: "slow"
          component: "ryot"
        annotations:
          summary: "Test SLO Slow Burn Alert"
EOF

curl -X POST http://localhost:9090/-/reload
```

**Expected Results - Fast Burn:**

1. ✅ PagerDuty page created immediately (group_wait=10s)
2. ✅ Slack critical channel notification
3. ✅ Severity: critical
4. ✅ Repeat every 15 minutes

**Expected Results - Slow Burn:**

1. ✅ Slack warning channel notification
2. ✅ No PagerDuty (goes to slo-burn-slow receiver)
3. ✅ Severity: warning
4. ✅ Group wait: 2 minutes (slower escalation)
5. ✅ Repeat every 1 hour

---

### 1.4 Inhibition Rules Test

**Objective:** Verify lower-severity alerts suppressed when higher-severity exists

#### Test Procedure:

```bash
# Create service with both critical and warning alerts
cat > /etc/prometheus/alert_test_inhibit.yml << 'EOF'
groups:
  - name: test_inhibit
    interval: 10s
    rules:
      # Critical alert (parent)
      - alert: TestServiceDown
        expr: 1
        labels:
          severity: critical
          service: test-service
        annotations:
          summary: "Test Service Down (Critical)"

      # Warning alert (should be inhibited)
      - alert: TestHighLatency
        expr: 1
        labels:
          severity: warning
          service: test-service
        annotations:
          summary: "Test High Latency (Should be Inhibited)"

      # Info alert (should be inhibited)
      - alert: TestHighMemory
        expr: 1
        labels:
          severity: info
          service: test-service
        annotations:
          summary: "Test High Memory (Should be Inhibited)"
EOF

curl -X POST http://localhost:9090/-/reload
```

**Expected Results:**

1. ✅ ServiceDown alert routes normally (PagerDuty + Slack)
2. ✅ HighLatency alert SUPPRESSED (not visible in AlertManager)
3. ✅ HighMemory alert SUPPRESSED (not visible in AlertManager)
4. ⏱️ After service recovers, suppression lifted (alerts appear immediately)

**Verification:**

```bash
# Check suppressed alerts in AlertManager UI
curl http://alertmanager.neurectomy.local:9093/api/v1/alerts | jq '.data[] | select(.status.inhibited == true)'
```

---

## 2. NOTIFICATION CHANNEL VALIDATION

### 2.1 Slack Channel Validation

**Prerequisites:**

- `SLACK_WEBHOOK_URL` environment variable set
- Slack workspace configured with Neurectomy app

#### Test Procedure:

```bash
# Send test message to Slack webhook
curl -X POST $SLACK_WEBHOOK_URL \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "✅ Neurectomy Slack Webhook Test - Monitoring Stack Ready",
    "attachments": [
      {
        "color": "good",
        "fields": [
          {
            "title": "Test Type",
            "value": "Webhook Connectivity",
            "short": true
          },
          {
            "title": "Timestamp",
            "value": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'",
            "short": true
          },
          {
            "title": "Status",
            "value": "SUCCESS",
            "short": false
          }
        ]
      }
    ]
  }'
```

**Expected Result:**
✅ Message appears in configured Slack channel within 5 seconds

**Required Channels:**

- `#neurectomy-critical` - Critical/security alerts
- `#neurectomy-alerts` - Warning/info alerts
- `#neurectomy-slo` - SLO burn rate alerts
- `#neurectomy-ops` - Resource utilization alerts

---

### 2.2 PagerDuty Integration Validation

**Prerequisites:**

- `PAGERDUTY_SERVICE_KEY_CRITICAL` set
- `PAGERDUTY_SERVICE_KEY_SLO` set
- `PAGERDUTY_SERVICE_KEY_SECURITY` set

#### Test Procedure:

```bash
# Send test event to PagerDuty
curl -X POST https://events.pagerduty.com/v2/enqueue \
  -H 'Content-Type: application/json' \
  -d '{
    "routing_key": "'$PAGERDUTY_SERVICE_KEY_CRITICAL'",
    "event_action": "trigger",
    "dedup_key": "test-critical-connectivity",
    "payload": {
      "summary": "Neurectomy Monitoring - Test Critical Alert",
      "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'",
      "severity": "critical",
      "source": "Neurectomy AlertManager Test"
    }
  }'
```

**Expected Result:**
✅ Incident created in PagerDuty within 30 seconds

**Verification:**

```bash
# Check incidents via PagerDuty API
curl https://api.pagerduty.com/incidents \
  -H "Authorization: Token token=$PAGERDUTY_API_TOKEN" \
  -H "Content-Type: application/json" | jq '.incidents[0]'
```

---

### 2.3 Email Notification Validation

**Prerequisites:**

- `SMTP_HOST`, `SMTP_USERNAME`, `SMTP_PASSWORD` set
- SMTP credentials valid for email provider

#### Test Procedure:

```bash
# Check AlertManager email configuration
kubectl exec -it alertmanager-0 -n neurectomy -- \
  grep -A5 "smtp_smarthost" /etc/alertmanager/alertmanager.yml

# Send test email via AlertManager webhook
curl -X POST http://alertmanager.neurectomy.local:9093/api/v1/alerts \
  -H 'Content-Type: application/json' \
  -d '[{
    "labels": {
      "alertname": "TestEmailAlert",
      "severity": "critical",
      "component": "test"
    },
    "annotations": {
      "summary": "Neurectomy Email Test",
      "description": "This is a test email notification"
    }
  }]'
```

**Expected Result:**
✅ Email received at configured SMTP recipient within 2 minutes

**Required SMTP Configuration:**

```yaml
smtp_smarthost: ${SMTP_HOST}:587
smtp_auth_username: ${SMTP_USERNAME}
smtp_auth_password: ${SMTP_PASSWORD}
smtp_from: alerts@neurectomy.local
smtp_require_tls: true
```

---

### 2.4 Webhook Integration Test

**Objective:** Verify custom webhook receivers working

#### Test Procedure:

```bash
# Deploy simple test webhook receiver
kubectl apply -f - << 'EOF'
apiVersion: v1
kind: Service
metadata:
  name: webhook-test
  namespace: neurectomy
spec:
  ports:
  - port: 8080
    targetPort: 8080
  selector:
    app: webhook-test
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: webhook-test
  namespace: neurectomy
spec:
  replicas: 1
  selector:
    matchLabels:
      app: webhook-test
  template:
    metadata:
      labels:
        app: webhook-test
    spec:
      containers:
      - name: receiver
        image: kennethreitz/httpbin
        ports:
        - containerPort: 8080
EOF

# Test webhook delivery
curl -X POST http://alertmanager.neurectomy.local:9093/api/v1/alerts \
  -H 'Content-Type: application/json' \
  -d '[{
    "labels": {
      "alertname": "TestWebhookAlert",
      "severity": "critical",
      "component": "test"
    },
    "annotations": {
      "summary": "Webhook Test Alert"
    }
  }]'

# Verify webhook received data
kubectl logs -n neurectomy deploy/webhook-test | grep "POST"
```

**Expected Result:**
✅ Webhook endpoint receives POST request with alert payload

---

## 3. SLO CALCULATION VALIDATION

### 3.1 Recording Rules Verification

**Objective:** Verify 30-day SLO window calculations are correct

#### Test Procedure:

```bash
# Query recording rules from Prometheus
kubectl port-forward -n neurectomy svc/prometheus 9090:9090 &

# Verify Ryot TTFT SLO recording
curl 'http://localhost:9090/api/v1/query?query=slo:ryot:ttft:window_ratio' | jq '.data.result[0].value'

# Expected: Value between 0 and 1 (representing 0-100% compliance)
# For 99.9% target: expect ~0.999 or higher

# Verify Ryot Error Rate SLO recording
curl 'http://localhost:9090/api/v1/query?query=slo:ryot:error_rate:window_ratio' | jq '.data.result[0].value'

# Verify ΣLANG Compression Ratio
curl 'http://localhost:9090/api/v1/query?query=slo:sigmalang:ratio:window_ratio' | jq '.data.result[0].value'

# Expected: Value >= 5.0 (5x compression ratio)

# Verify ΣLANG Success Rate
curl 'http://localhost:9090/api/v1/query?query=slo:sigmalang:success:window_ratio' | jq '.data.result[0].value'

# Verify ΣVAULT Availability
curl 'http://localhost:9090/api/v1/query?query=slo:sigmavault:availability:window_ratio' | jq '.data.result[0].value'

# Expected: Value >= 0.9999 (99.99%)

# Verify Agent Collective Health
curl 'http://localhost:9090/api/v1/query?query=slo:agent:health:ratio' | jq '.data.result[0].value'

# Expected: Value >= 0.95 (95%)
```

**Validation Criteria:**
| SLO | Metric | Expected | Actual | Status |
|-----|--------|----------|--------|--------|
| Ryot TTFT | slo:ryot:ttft:window_ratio | ≥ 0.999 | ? | ? |
| Ryot Error | slo:ryot:error_rate:window_ratio | ≥ 0.999 | ? | ? |
| ΣLANG Ratio | slo:sigmalang:ratio:window_ratio | ≥ 5.0 | ? | ? |
| ΣLANG Success | slo:sigmalang:success:window_ratio | ≥ 0.99 | ? | ? |
| ΣVAULT Availability | slo:sigmavault:availability:window_ratio | ≥ 0.9999 | ? | ? |
| ΣVAULT Latency | slo:sigmavault:latency:window_ratio | ≥ 0.999 | ? | ? |
| Agent Health | slo:agent:health:ratio | ≥ 0.95 | ? | ? |
| Agent Throughput | slo:agent:throughput:variance | ≤ 0.1 | ? | ? |

---

### 3.2 Burn Rate Alert Verification

**Objective:** Verify burn rate alerts trigger at correct thresholds

#### Test Procedure:

```bash
# Check alert evaluation status
curl 'http://localhost:9090/api/v1/rules?type=alert' | jq '.data.groups[] | select(.name | contains("slo"))'

# Sample queries for each burn type:

# Fast Burn: 1 hour of budget consumed in 15 minutes
# For 99.9% target: error_budget = 0.001 per 30 days
# Fast burn = 0.001 / 30 per day = 3.3e-5 per day
# In 15 min window: should see 0.1% error rate (0.999 -> 0.989)

curl 'http://localhost:9090/api/v1/query?query=ALERTS{slo_burn_type="fast"}' | jq '.data.result'

# Slow Burn: 1 hour of budget consumed in 72 minutes
# In 6 hour window: should see 0.016% error rate (0.999 -> 0.998)

curl 'http://localhost:9090/api/v1/query?query=ALERTS{slo_burn_type="slow"}' | jq '.data.result'

# Critical Breach: SLO target violated immediately
curl 'http://localhost:9090/api/v1/query?query=CriticalSLOViolation' | jq '.data.result'
```

**Expected Triggers:**

1. ✅ Fast burn alerts when error rate exceeds SLO for 15 minutes
2. ✅ Slow burn alerts when sustained low compliance for 1 hour
3. ✅ Critical breach immediately when SLO target violated

---

## 4. DASHBOARD VALIDATION

### 4.1 Dashboard Import Steps

```bash
# Import SLO dashboards into Grafana
kubectl exec -it grafana-0 -n neurectomy -- /bin/bash

# Use Grafana API to import dashboards
for dashboard in slo-ryot slo-sigmalang slo-sigmavault slo-agent-collective; do
  curl -X POST http://localhost:3000/api/dashboards/db \
    -H "Authorization: Bearer $GRAFANA_API_TOKEN" \
    -H "Content-Type: application/json" \
    -d @"/tmp/${dashboard}.json"
done
```

### 4.2 Dashboard Panel Validation

#### Ryot Dashboard Panels:

1. ✅ TTFT SLO Compliance (Target: 99.9%)
   - Query: `slo:ryot:ttft:window_ratio * 100`
   - Type: Timeseries
   - Threshold: 99.9% green, <99.8% yellow, <99% red

2. ✅ Error Rate SLO Compliance (Target: 99.9%)
   - Query: `(1 - ...) * 100`
   - Type: Timeseries
   - Threshold: 99.9% green

3. ✅ Error Budget Usage (30-day)
   - Query: Error percentage calculation
   - Type: Gauge
   - Threshold: 0% green → 50% yellow → 100% red

4. ✅ Request Throughput
   - Query: `rate(ryot_requests_total[5m])`
   - Type: Timeseries
   - Shows requests per second trend

5. ✅ TTFT Latency Distribution
   - Query: P50, P95, P99 histogram quantiles
   - Type: Timeseries
   - SLO target: P95 < 50ms

#### Expected Dashboard Behavior:

- Load time: < 2 seconds
- Refresh interval: 30s
- Timezone: UTC
- Time range: 7 days

---

### 4.3 Query Performance Testing

```bash
# Test dashboard query performance
time curl 'http://prometheus:9090/api/v1/query?query=slo:ryot:ttft:window_ratio[30d]'

# Expected: < 500ms response time
# Verify indexes are used for 30-day queries

# Check Prometheus storage efficiency
curl http://prometheus:9090/api/v1/label/__name__/values | jq 'length'
# Should return ~50-100 metric names (not thousands)
```

---

## 5. PRODUCTION DEPLOYMENT CHECKLIST

### Pre-Deployment Validation

- [ ] All environment variables set (see Section 6 below)
- [ ] AlertManager config valid: `amtool config routes`
- [ ] Prometheus alert rules valid: `promtool check rules alert_rules.yml`
- [ ] Slack webhooks respond (< 200ms)
- [ ] PagerDuty API accessible
- [ ] SMTP credentials valid
- [ ] All 4 SLO dashboards imported in Grafana
- [ ] Test alert firing and routing works end-to-end

### Deployment Steps

```bash
# 1. Update Kubernetes secrets with environment variables
kubectl create secret generic alertmanager-env \
  --from-literal=SLACK_WEBHOOK_URL="..." \
  --from-literal=PAGERDUTY_SERVICE_KEY_CRITICAL="..." \
  --from-literal=PAGERDUTY_SERVICE_KEY_SLO="..." \
  --from-literal=PAGERDUTY_SERVICE_KEY_SECURITY="..." \
  --from-literal=SMTP_HOST="..." \
  --from-literal=SMTP_USERNAME="..." \
  --from-literal=SMTP_PASSWORD="..." \
  --from-literal=OPSGENIE_API_KEY="..." \
  -n neurectomy

# 2. Update ConfigMap with alert rules
kubectl create configmap prometheus-alerts \
  --from-file=/docker/prometheus/alert_rules.yml \
  -n neurectomy

# 3. Update AlertManager StatefulSet
kubectl apply -f deploy/k8s/11-alertmanager-statefulset.yaml

# 4. Verify pods started successfully
kubectl get pods -n neurectomy | grep alertmanager
kubectl logs -n neurectomy alertmanager-0

# 5. Port-forward and test
kubectl port-forward -n neurectomy svc/alertmanager 9093:9093 &
curl http://localhost:9093/api/v1/status | jq '.data'
```

---

## 6. ENVIRONMENT VARIABLES SETUP

### Required Variables

```bash
# Slack Integration
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
export SLACK_CRITICAL_CHANNEL="#neurectomy-critical"
export SLACK_ALERTS_CHANNEL="#neurectomy-alerts"
export SLACK_SLO_CHANNEL="#neurectomy-slo"
export SLACK_OPS_CHANNEL="#neurectomy-ops"

# PagerDuty Integration
export PAGERDUTY_SERVICE_KEY_CRITICAL="<integration-key-critical>"
export PAGERDUTY_SERVICE_KEY_SLO="<integration-key-slo>"
export PAGERDUTY_SERVICE_KEY_SECURITY="<integration-key-security>"
export PAGERDUTY_ESCALATION_POLICY="<policy-id>"

# SMTP Configuration (Email)
export SMTP_HOST="smtp.example.com"
export SMTP_PORT="587"
export SMTP_USERNAME="alerts@example.com"
export SMTP_PASSWORD="<password>"
export SMTP_FROM="alerts@neurectomy.local"
export SMTP_TO="admin@neurectomy.local,oncall@neurectomy.local"

# OpsGenie Integration
export OPSGENIE_API_KEY="<api-key>"
export OPSGENIE_REGION="us"  # or "eu"

# Grafana
export GRAFANA_API_TOKEN="<token>"
export GRAFANA_URL="http://grafana.neurectomy.local"
```

### Kubernetes Secret Creation

```bash
kubectl create secret generic alertmanager-secrets \
  --from-literal=slack-webhook-url="$SLACK_WEBHOOK_URL" \
  --from-literal=pagerduty-critical-key="$PAGERDUTY_SERVICE_KEY_CRITICAL" \
  --from-literal=pagerduty-slo-key="$PAGERDUTY_SERVICE_KEY_SLO" \
  --from-literal=pagerduty-security-key="$PAGERDUTY_SERVICE_KEY_SECURITY" \
  --from-literal=smtp-host="$SMTP_HOST" \
  --from-literal=smtp-username="$SMTP_USERNAME" \
  --from-literal=smtp-password="$SMTP_PASSWORD" \
  --from-literal=opsgenie-api-key="$OPSGENIE_API_KEY" \
  -n neurectomy
```

---

## 7. MONITORING THE MONITORING STACK

### AlertManager Health

```bash
# Check AlertManager status
curl http://alertmanager.neurectomy.local:9093/api/v1/status | jq '.data'

# View current alerts
curl http://alertmanager.neurectomy.local:9093/api/v1/alerts | jq '.data | length'

# View alert groups
curl http://alertmanager.neurectomy.local:9093/api/v1/alerts/groups | jq '.data'
```

### Prometheus Health

```bash
# Check Prometheus targets
curl http://prometheus.neurectomy.local:9090/api/v1/targets | jq '.data.activeTargets | length'

# Check rules evaluation
curl http://prometheus.neurectomy.local:9090/api/v1/rules | jq '.data.groups | length'

# Check alert evaluation count
curl http://prometheus.neurectomy.local:9090/api/v1/query?query='count(ALERTS)' | jq '.data'
```

### Grafana Health

```bash
# Check Grafana datasource connectivity
curl -H "Authorization: Bearer $GRAFANA_API_TOKEN" \
  http://grafana.neurectomy.local/api/datasources | jq '.[] | {name, type}'

# Test Prometheus datasource
curl -H "Authorization: Bearer $GRAFANA_API_TOKEN" \
  http://grafana.neurectomy.local/api/datasources/1/health
```

---

## 8. TROUBLESHOOTING GUIDE

### Alert Not Firing

**Problem:** Alert defined but never fires
**Solution:**

```bash
# 1. Check alert rule syntax
promtool check rules alert_rules.yml

# 2. Verify metric exists
curl 'http://prometheus:9090/api/v1/query?query=<metric_name>'

# 3. Check metric cardinality
curl 'http://prometheus:9090/api/v1/query?query=count(<metric_name>)'

# 4. Simulate alert expression
curl 'http://prometheus:9090/api/v1/query?query=<alert_expr>'
```

### Alert Not Routing

**Problem:** Alert fires but doesn't reach expected channel
**Solution:**

```bash
# 1. Check AlertManager config
kubectl exec -it alertmanager-0 -n neurectomy -- \
  cat /etc/alertmanager/alertmanager.yml

# 2. Verify routing rules match
amtool config routes

# 3. Check receiver configuration
curl http://alertmanager:9093/api/v1/alerts | jq '.data[] | select(.labels.alertname=="YourAlert")'

# 4. Test webhook manually
curl -X POST http://webhook-endpoint \
  -H 'Content-Type: application/json' \
  -d '{alert payload}'
```

### Dashboard Queries Slow

**Problem:** SLO dashboard panels take >5s to load
**Solution:**

```bash
# 1. Check Prometheus query performance
curl 'http://prometheus:9090/api/v1/query?query=slo:ryot:ttft:window_ratio[30d]' \
  -H 'X-Prometheus-Inspect: true'

# 2. Verify index optimization
curl 'http://prometheus:9090/api/v1/labels'

# 3. Reduce time range in dashboard (from 30d to 7d)
# 4. Add rate limiting: interval=5m (not 1m)
```

### Notification Channel Not Working

**Problem:** Slack/PagerDuty/Email not receiving alerts
**Solution:**

**Slack:**

```bash
# Test webhook directly
curl -X POST $SLACK_WEBHOOK_URL \
  -H 'Content-Type: application/json' \
  -d '{"text":"Test"}'

# Check logs
kubectl logs -n neurectomy alertmanager-0 | grep -i slack
```

**PagerDuty:**

```bash
# Verify API key
curl -H "Authorization: Token token=$PAGERDUTY_API_TOKEN" \
  https://api.pagerduty.com/users | jq '.users[0]'

# Check recent events
curl "https://events.pagerduty.com/v2/enqueue" \
  -X POST \
  -H 'Content-Type: application/json' \
  --data-raw '{
    "routing_key": "'$PAGERDUTY_SERVICE_KEY_CRITICAL'",
    "event_action": "trigger",
    "payload": {
      "summary": "Test",
      "severity": "critical",
      "source": "test"
    }
  }'
```

**Email:**

```bash
# Test SMTP connection
telnet $SMTP_HOST $SMTP_PORT

# Check logs
kubectl logs -n neurectomy alertmanager-0 | grep -i smtp
```

---

## 9. SUCCESS CRITERIA

All items below must be verified ✅ before considering SLO monitoring complete:

### Configuration ✅

- [ ] AlertManager YAML valid and deployment successful
- [ ] Prometheus alert rules loaded and evaluating
- [ ] All 4 SLO dashboards imported in Grafana
- [ ] Environment variables set in Kubernetes secrets

### Routing ✅

- [ ] Critical alerts route to PagerDuty + Slack + Email (< 15s)
- [ ] Warning alerts route to Slack channel (< 2m)
- [ ] Info alerts log to webhook only (< 5m)
- [ ] SLO fast burn alerts trigger and notify (< 20s)
- [ ] SLO slow burn alerts trigger on sustained burn (< 2m 30s)
- [ ] Inhibition rules suppress lower severity (verified in UI)

### Channels ✅

- [ ] Slack messages arrive with proper formatting
- [ ] PagerDuty incidents created with full context
- [ ] Email notifications received (< 2m delivery)
- [ ] Webhooks receive alert payloads

### SLOs ✅

- [ ] Recording rules calculate 30-day windows correctly
- [ ] Burn rate calculations accurate (fast vs slow)
- [ ] Dashboard queries return in < 2s
- [ ] All panels visualize data correctly
- [ ] Error budget tracking visible

### Dashboards ✅

- [ ] Ryot dashboard shows TTFT and error rate
- [ ] ΣLANG dashboard shows compression ratio
- [ ] ΣVAULT dashboard shows availability/latency
- [ ] Agent Collective dashboard shows health
- [ ] All time series render without errors

---

## 10. MONITORING STACK ARCHITECTURE DIAGRAM

```
┌────────────────────────────────────────────────────────────────┐
│                    Prometheus Services                         │
│  (Scrape metrics every 15s)                                    │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│                   Alert Rule Evaluation                        │
│  (Every 15s: Check 19 alert rules + 4 SLO groups)              │
│                                                                │
│  • ML Service Alerts (3)                                       │
│  • Training Job Alerts (3)                                     │
│  • Resource Alerts (5)                                         │
│  • Security Alerts (4)                                         │
│  • Analytics Alerts (4)                                        │
│  → SLO Rules (4 groups × 6 rules = 24)                         │
└────────────────────────────────────────────────────────────────┘
                              ↓
              ┌───────────────┴───────────────┐
              ↓                               ↓
    ┌──────────────────┐        ┌──────────────────┐
    │  AlertManager    │        │    Grafana       │
    │  (Alert Routing) │        │  (Visualization) │
    └──────────────────┘        └──────────────────┘
              ↓                         ↓
    ┌─────────┴─────────┐          4 SLO Dashboards
    ↓                   ↓          - Ryot (TTFT)
 Evaluation       Routing         - ΣLANG (Compression)
 (10s wait)      (Match rules)    - ΣVAULT (Storage)
    ↓                   ↓          - Agent Collective
 Grouping         Receivers       (Every 30s refresh)
    ↓                   ↓
Deduplication  ┌───────┴───────┬────────┬────────┐
    ↓          ↓               ↓        ↓        ↓
 Inhibition   Slack        PagerDuty  Email   Webhook
    ↓         ├─critical  ├─critical ├─critical ├─logging
 Silencing    ├─alerts    ├─slo      ├─slo
              ├─slo       ├─security
              ├─ops
              └─debug
```

---

**Document Version:** 1.0
**Last Updated:** Phase 18B
**Status:** Production Ready for Testing
