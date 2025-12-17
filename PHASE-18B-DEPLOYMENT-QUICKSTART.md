# Phase 18B - Monitoring Stack Deployment Quick Start

## 🚀 TL;DR - 5 Minute Setup

```bash
# 1. Set environment variables
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
export PAGERDUTY_SERVICE_KEY_CRITICAL="ba1234567890..."
export PAGERDUTY_SERVICE_KEY_SLO="bc1234567890..."
export SMTP_HOST="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USERNAME="your-email@gmail.com"
export SMTP_PASSWORD="xxxx xxxx xxxx xxxx"
export SMTP_FROM="alerts@neurectomy.local"

# 2. Run deployment script
chmod +x deploy-monitoring-stack.sh
./deploy-monitoring-stack.sh --namespace neurectomy

# 3. Port forward for testing
kubectl port-forward -n neurectomy svc/prometheus 9090:9090 &
kubectl port-forward -n neurectomy svc/alertmanager 9093:9093 &
kubectl port-forward -n neurectomy svc/grafana 3000:3000 &

# 4. Test alert routing
# See PHASE-18B-TESTING-VALIDATION-GUIDE.md

# 5. View dashboards
# Visit http://localhost:3000 → SLO Dashboards
```

---

## 📋 Full Deployment Checklist

### Phase 1: Environment Setup (20 min)

- [ ] **Get Slack Webhook URL** (5 min)
  - Go to https://api.slack.com/apps/
  - Create app or select existing
  - Incoming Webhooks → Add New Webhook
  - Select channel → Allow
  - Copy webhook URL to `SLACK_WEBHOOK_URL`
- [ ] **Get PagerDuty Service Keys** (10 min)
  - Go to https://app.pagerduty.com/
  - Services → Create/select "Neurectomy Critical"
  - Integrations → Add Events API v2
  - Copy integration key to `PAGERDUTY_SERVICE_KEY_CRITICAL`
  - Repeat for SLO and Security services
- [ ] **Configure SMTP** (5 min)
  - Gmail: Generate app password at https://myaccount.google.com/apppasswords
  - Or use corporate SMTP server
  - Set: `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_FROM`

### Phase 2: Kubernetes Deployment (10 min)

- [ ] **Verify Prerequisites**
  - Kubernetes cluster running
  - kubectl configured
  - Prometheus already deployed
  - Namespace exists: `kubectl get namespace neurectomy`

- [ ] **Set Environment Variables**

  ```bash
  source ~/.env.alertmanager  # or set manually
  ```

- [ ] **Run Deployment Script**

  ```bash
  cd /path/to/neurectomy
  chmod +x deploy-monitoring-stack.sh
  ./deploy-monitoring-stack.sh --namespace neurectomy
  ```

- [ ] **Verify Deployment**
  ```bash
  kubectl get pods -n neurectomy | grep -E "(prometheus|alertmanager|grafana)"
  # All pods should be RUNNING
  ```

### Phase 3: Testing & Validation (30 min)

- [ ] **Test Slack Webhook**

  ```bash
  curl -X POST $SLACK_WEBHOOK_URL \
    -H 'Content-Type: application/json' \
    -d '{"text":"✅ Webhook Test"}'
  # Verify message appears in Slack
  ```

- [ ] **Test PagerDuty Integration**
  - Follow "Test PagerDuty Event" in PHASE-18B-TESTING-VALIDATION-GUIDE.md
  - Verify incident created in console

- [ ] **Test SMTP**

  ```bash
  python3 -c "
  import smtplib
  from email.mime.text import MIMEText
  msg = MIMEText('SMTP Test')
  msg['Subject'] = 'Neurectomy SMTP'
  msg['From'] = '$SMTP_FROM'
  msg['To'] = 'admin@neurectomy.local'
  server = smtplib.SMTP('$SMTP_HOST', $SMTP_PORT)
  server.starttls()
  server.login('$SMTP_USERNAME', '$SMTP_PASSWORD')
  server.send_message(msg)
  server.quit()
  print('✓ SMTP Test Success')
  "
  ```

- [ ] **Test Alert Routing**
  - See "Alert Routing Validation" section in PHASE-18B-TESTING-VALIDATION-GUIDE.md
  - Create test alert rule
  - Verify routing to correct channel

- [ ] **Verify SLO Dashboards**
  - Port-forward to Grafana: `kubectl port-forward -n neurectomy svc/grafana 3000:3000`
  - Visit http://localhost:3000
  - Navigate to Dashboards
  - Verify 4 SLO dashboards loaded:
    - Ryot SLO Dashboard
    - Sigma-Lang SLO Dashboard
    - Sigma-Vault SLO Dashboard
    - Agent Collective SLO Dashboard
  - Check that panels display data (no "No data" errors)

### Phase 4: Production Validation (Optional)

- [ ] **Generate Sample Alerts**
  - Trigger test alert in production
  - Verify all 3 notification channels receive alert

- [ ] **Monitor for 24 Hours**
  - Watch dashboard for stability
  - Check alert evaluation times in Prometheus
  - Monitor AlertManager queue depth

- [ ] **Document Runbook**
  - Who to contact if monitoring fails
  - How to troubleshoot each component
  - Escalation procedures

---

## 🔍 Verification Commands

### Check Deployment Status

```bash
# All pods running
kubectl get pods -n neurectomy

# Prometheus health
curl http://prometheus.neurectomy.local:9090/-/healthy

# AlertManager health
curl http://alertmanager.neurectomy.local:9093/-/healthy

# Grafana health
curl http://grafana.neurectomy.local:3000/api/health
```

### Check Alert Rules Loaded

```bash
# List all alert rules
curl http://prometheus.neurectomy.local:9090/api/v1/rules | jq '.data.groups[] | .name'

# Count SLO alert rules
curl http://prometheus.neurectomy.local:9090/api/v1/rules | \
  jq '[.data.groups[] | select(.name | contains("slo"))] | length'

# Expected: 4 SLO groups (ryot, sigmalang, sigmavault, agent-collective)
```

### Check AlertManager Routes

```bash
# Show routing configuration
kubectl exec -n neurectomy alertmanager-0 -- amtool config routes

# Show current alerts
curl http://alertmanager.neurectomy.local:9093/api/v1/alerts

# Show alert groups
curl http://alertmanager.neurectomy.local:9093/api/v1/alerts/groups
```

### Check Grafana Datasources

```bash
# Port-forward to Grafana
kubectl port-forward -n neurectomy svc/grafana 3000:3000 &

# Get Grafana API token (if available)
GRAFANA_TOKEN=$(kubectl get secret grafana-secrets -n neurectomy \
  -o jsonpath='{.data.admin_password}' | base64 -d)

# List datasources
curl -H "Authorization: Bearer $GRAFANA_TOKEN" \
  http://localhost:3000/api/datasources
```

---

## 📊 Dashboard Access

Once deployed, access SLO dashboards:

1. **Ryot (LLM Inference)**
   - URL: http://localhost:3000/d/slo-ryot
   - SLO: TTFT <50ms (99.9%), Error <1% (99.9%)
   - Budget: 43.2 min/month

2. **ΣLANG (Compression)**
   - URL: http://localhost:3000/d/slo-sigmalang
   - SLO: Ratio >5x (99.5%), Success >99% (99.5%)
   - Budget: 216 min/month

3. **ΣVAULT (Storage)**
   - URL: http://localhost:3000/d/slo-sigmavault
   - SLO: Availability >99.99%, Latency <100ms p95 (99.9%)
   - Budget: 4.32 min/month (MOST CONSTRAINED!)

4. **Agent Collective**
   - URL: http://localhost:3000/d/slo-agent-collective
   - SLO: Health >95%, Throughput ±10%
   - Budget: Depends on defined SLO

---

## 🔧 Troubleshooting

### "Pod is not starting"

```bash
# Check logs
kubectl logs -n neurectomy alertmanager-0

# Check resource constraints
kubectl describe pod alertmanager-0 -n neurectomy

# Check if secrets mounted
kubectl get secret alertmanager-secrets -n neurectomy
```

### "Prometheus not scraping metrics"

```bash
# Check targets
curl http://prometheus.neurectomy.local:9090/api/v1/targets

# Check alerts not firing
curl http://prometheus.neurectomy.local:9090/api/v1/rules | jq '.data.groups[].rules[] | select(.state=="inactive")'
```

### "Slack/PagerDuty/Email not receiving alerts"

```bash
# Check AlertManager logs
kubectl logs -n neurectomy alertmanager-0 | grep -i slack/pagerduty/smtp

# Test webhook directly
curl -X POST $SLACK_WEBHOOK_URL -d '{"text":"test"}'

# Check AlertManager config
kubectl exec -n neurectomy alertmanager-0 -- cat /etc/alertmanager/alertmanager.yml
```

### "Grafana dashboards not showing data"

```bash
# Verify Prometheus datasource works
# In Grafana: Configuration → Datasources → Test

# Check if metrics exist
curl 'http://prometheus.neurectomy.local:9090/api/v1/query?query=slo:ryot:ttft:window_ratio'

# Should return value between 0 and 1
```

---

## 📝 Files Reference

| File                                       | Purpose                      |
| ------------------------------------------ | ---------------------------- |
| `deploy-monitoring-stack.sh`               | Main deployment script       |
| `PHASE-18B-ENVIRONMENT-VARIABLES-SETUP.md` | How to configure credentials |
| `PHASE-18B-TESTING-VALIDATION-GUIDE.md`    | Complete testing procedures  |
| `/docker/alertmanager/alertmanager.yml`    | AlertManager configuration   |
| `/docker/prometheus/alert_rules.yml`       | SLO alert rules              |
| `/deploy/k8s/slo-dashboard-*.json`         | SLO dashboard definitions    |

---

## ⏱️ Deployment Timeline

| Phase | Task                                     | Duration | Status  |
| ----- | ---------------------------------------- | -------- | ------- |
| 1     | Environment Setup (Slack/PagerDuty/SMTP) | 20 min   | ⏳ TODO |
| 2     | Kubernetes Deployment                    | 10 min   | ⏳ TODO |
| 3     | Testing & Validation                     | 30 min   | ⏳ TODO |
| 4     | Production Monitoring (24h)              | Ongoing  | ⏳ TODO |

**Total Time: ~1 hour for initial deployment**

---

## ✅ Success Criteria

All items below must be ✅ before considering deployment complete:

- [x] AlertManager configuration complete with 13 receivers
- [x] Prometheus alert rules loaded (4 SLO groups)
- [x] SLO dashboards created in JSON format
- [ ] Environment variables set in Kubernetes secrets
- [ ] AlertManager pod running and healthy
- [ ] Prometheus scraping alert rules successfully
- [ ] Slack webhook delivering messages
- [ ] PagerDuty incidents being created
- [ ] Email notifications received
- [ ] SLO dashboards displaying data in Grafana
- [ ] Test alert routes to correct channel
- [ ] No errors in AlertManager/Prometheus logs

---

## 🚨 Emergency Contacts

When alerts are triggered:

- **Critical/Security**: [PagerDuty Escalation]
- **Warning**: Slack #neurectomy-alerts
- **SLO Burn**: Slack #neurectomy-slo + PagerDuty
- **System Down**: All channels simultaneously

---

## 📞 Support

For issues:

1. Check logs: `kubectl logs -n neurectomy alertmanager-0`
2. Consult troubleshooting guide above
3. Review PHASE-18B-TESTING-VALIDATION-GUIDE.md
4. Contact SRE team

---

**Last Updated:** Phase 18B
**Status:** Ready for Deployment
