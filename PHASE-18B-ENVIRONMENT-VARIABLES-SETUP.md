# Neurectomy SLO Monitoring - Environment Variables Setup

## 1. SLACK WEBHOOK SETUP

### Step 1: Create Slack Webhook URL

```bash
# Visit Slack API dashboard (as workspace admin):
# https://api.slack.com/apps/

# 1. Create New App
#    Name: "Neurectomy Monitoring"
#    Workspace: Select your workspace

# 2. Navigate to "Incoming Webhooks"
#    Enable Incoming Webhooks: ON

# 3. Click "Add New Webhook to Workspace"
#    Select channel: #neurectomy-critical
#    Click: "Allow"

# This generates: https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXX
```

### Step 2: Configure Multiple Channels (Optional)

```bash
# For multiple channels, create separate webhooks:

# Critical alerts
SLACK_WEBHOOK_URL_CRITICAL="https://hooks.slack.com/services/T.../B.../CRITICAL_WEBHOOK"

# General alerts
SLACK_WEBHOOK_URL_ALERTS="https://hooks.slack.com/services/T.../B.../ALERTS_WEBHOOK"

# SLO alerts
SLACK_WEBHOOK_URL_SLO="https://hooks.slack.com/services/T.../B.../SLO_WEBHOOK"

# Ops alerts
SLACK_WEBHOOK_URL_OPS="https://hooks.slack.com/services/T.../B.../OPS_WEBHOOK"

# In alertmanager.yml, use templating for multi-channel:
# slack_api_url: '{{ .alerts[0].Labels.slack_webhook | default $defaultWebhook }}'
```

### Step 3: Test Webhook

```bash
# Direct webhook test:
curl -X POST \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "✅ Neurectomy Slack Integration Test",
    "attachments": [{
      "color": "good",
      "fields": [
        {"title": "Status", "value": "CONNECTED", "short": true},
        {"title": "Timestamp", "value": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'", "short": true}
      ]
    }]
  }' \
  "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

# Expected: Response code 200 with message "ok"
```

---

## 2. PAGERDUTY SERVICE KEY SETUP

### Step 1: Create PagerDuty Service

```bash
# Visit PagerDuty console (as admin):
# https://app.pagerduty.com/

# 1. Navigate to "Services" → "New Service"
#    Name: "Neurectomy Critical Alerts"
#    Type: "Generic Events API"
#    Description: "Production SLO and critical alert routing"

# 2. Save service (generates SERVICE_ID)

# 3. Navigate to "Integrations" → "Add Integration"
#    Type: "Events API V2"
#    Click: "Add"

# This generates INTEGRATION_KEY (service integration key for Events API v2)
```

### Step 2: Create Escalation Policy

```bash
# Navigate to "Escalation Policies" → "New Escalation Policy"

# Define escalation path:
# Level 1 (0 min): Primary on-call engineer
# Level 2 (15 min): Secondary on-call engineer
# Level 3 (30 min): Engineering manager

# Assign to service created above
```

### Step 3: Generate Integration Keys

```bash
# For each severity level, create separate service:

# CRITICAL alerts service
# Service: "Neurectomy Critical"
# Integration: Events API v2
PAGERDUTY_SERVICE_KEY_CRITICAL="<integration-key-from-console>"

# SLO burn rate alerts service
# Service: "Neurectomy SLO Alerts"
# Integration: Events API v2
PAGERDUTY_SERVICE_KEY_SLO="<integration-key-from-console>"

# Security alerts service
# Service: "Neurectomy Security"
# Integration: Events API v2
PAGERDUTY_SERVICE_KEY_SECURITY="<integration-key-from-console>"

# WARNING: These are sensitive - store in secure secret manager
```

### Step 4: Test Event API

```bash
# Send test event to verify integration key works:

curl -X POST https://events.pagerduty.com/v2/enqueue \
  -H 'Content-Type: application/json' \
  -d '{
    "routing_key": "'"$PAGERDUTY_SERVICE_KEY_CRITICAL"'",
    "event_action": "trigger",
    "dedup_key": "neurectomy-test-'$(date +%s)'",
    "payload": {
      "summary": "Neurectomy Monitoring Stack Test - CRITICAL",
      "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'",
      "severity": "critical",
      "source": "Neurectomy AlertManager Test Suite",
      "component": "monitoring-stack",
      "class": "integration-test"
    },
    "client": "Neurectomy Monitoring",
    "client_url": "https://monitoring.neurectomy.local"
  }'

# Expected: HTTP 202 Accepted with:
# {
#   "status": "success",
#   "message": "Event processed",
#   "dedup_key": "..."
# }

# Verify incident created:
curl -H "Authorization: Token token=$PAGERDUTY_API_TOKEN" \
  https://api.pagerduty.com/incidents \
  | jq '.incidents[0] | {title, status, urgency}'

# Expected: Recent incident with status="triggered", urgency="high"
```

---

## 3. SMTP EMAIL CONFIGURATION

### Step 1: Choose Email Provider

**Option A: Gmail / Google Workspace**

```bash
# 1. Enable 2FA on Google Account
# 2. Generate App Password (not regular password):
#    https://myaccount.google.com/apppasswords

# Configuration:
export SMTP_HOST="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USERNAME="your-email@gmail.com"
export SMTP_PASSWORD="xxxx xxxx xxxx xxxx"  # 16-character app password
export SMTP_FROM="neurectomy-alerts@gmail.com"
export SMTP_TLS="true"

# Test connection:
telnet smtp.gmail.com 587
# Expected: "220 smtp.gmail.com ESMTP" response
```

**Option B: Microsoft 365 / Outlook**

```bash
# Configuration:
export SMTP_HOST="smtp.office365.com"
export SMTP_PORT="587"
export SMTP_USERNAME="your-email@company.onmicrosoft.com"
export SMTP_PASSWORD="your-password"
export SMTP_FROM="neurectomy-alerts@company.onmicrosoft.com"
export SMTP_TLS="true"
```

**Option C: Self-Hosted (Postfix/Sendmail)**

```bash
# Configuration:
export SMTP_HOST="mail.neurectomy.local"
export SMTP_PORT="25"  # or 587 for TLS
export SMTP_USERNAME="neurectomy"
export SMTP_PASSWORD="password"
export SMTP_FROM="alerts@neurectomy.local"
export SMTP_TLS="false"  # or "true" if using port 587
```

### Step 2: Test SMTP Configuration

```bash
# Method 1: Direct telnet test
(
  sleep 1
  echo "EHLO neurectomy.local"
  sleep 1
  echo "STARTTLS"
  sleep 1
  echo "AUTH LOGIN"
  sleep 1
  # Enter base64-encoded username, then password
  echo "QUIT"
) | telnet $SMTP_HOST $SMTP_PORT

# Method 2: Using swaks tool (install: apt-get install swaks)
swaks --to recipient@example.com \
  --from "$SMTP_FROM" \
  --server "$SMTP_HOST:$SMTP_PORT" \
  --auth-user "$SMTP_USERNAME" \
  --auth-password "$SMTP_PASSWORD" \
  --header "Subject: Neurectomy SMTP Test" \
  --body "SMTP configuration test successful"

# Method 3: Using Python
python3 << 'EOF'
import smtplib
from email.mime.text import MIMEText

host = "$SMTP_HOST"
port = int("$SMTP_PORT")
username = "$SMTP_USERNAME"
password = "$SMTP_PASSWORD"

msg = MIMEText("Test message from Neurectomy")
msg['Subject'] = "Neurectomy SMTP Test"
msg['From'] = "$SMTP_FROM"
msg['To'] = "admin@neurectomy.local"

try:
    server = smtplib.SMTP(host, port)
    server.starttls()
    server.login(username, password)
    server.send_message(msg)
    server.quit()
    print("✅ SMTP test successful")
except Exception as e:
    print(f"❌ SMTP test failed: {e}")
EOF
```

### Step 3: Configure AlertManager Recipients

```bash
# List all recipients who should receive critical alerts:
export SMTP_TO_CRITICAL="oncall@neurectomy.local,admin@neurectomy.local"

# Separate recipients for different alert types:
export SMTP_TO_WARNINGS="team@neurectomy.local"
export SMTP_TO_SECURITY="security@neurectomy.local"

# In alertmanager.yml receivers:
receivers:
  - name: critical-receiver
    email_configs:
      - to: "{{ $externalURL }}"
        headers:
          Subject: "🚨 CRITICAL: {{ .GroupLabels.alertname }}"

  - name: security-critical
    email_configs:
      - to: "security@neurectomy.local"
        headers:
          Subject: "🔒 SECURITY ALERT"
```

---

## 4. OPSGENIE INTEGRATION (Optional)

### Step 1: Create OpsGenie Integration

```bash
# Visit OpsGenie dashboard:
# https://app.opsgenie.com/

# 1. Navigate to Settings → Integrations
# 2. Click "Create Integration"
#    Type: AlertManager
#    Name: "Neurectomy Monitoring"

# 3. Copy API Key
export OPSGENIE_API_KEY="your-api-key-from-opsgenie"
export OPSGENIE_REGION="us"  # or "eu"

# Test connectivity:
curl https://api.$OPSGENIE_REGION.opsgenie.com/v1/integrations/validate \
  -H "Authorization: GenieKey $OPSGENIE_API_KEY" \
  -H "Content-Type: application/json"
```

---

## 5. KUBERNETES SECRET CREATION

### Step 1: Create .env file (Local Development)

```bash
# Create file: .env.alertmanager
cat > .env.alertmanager << 'EOF'
# Slack
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# PagerDuty
PAGERDUTY_SERVICE_KEY_CRITICAL=<integration-key>
PAGERDUTY_SERVICE_KEY_SLO=<integration-key>
PAGERDUTY_SERVICE_KEY_SECURITY=<integration-key>

# SMTP
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=xxxx xxxx xxxx xxxx
SMTP_FROM=neurectomy-alerts@gmail.com
SMTP_TO_CRITICAL=oncall@neurectomy.local

# OpsGenie (optional)
OPSGENIE_API_KEY=<api-key>
OPSGENIE_REGION=us
EOF

# Source environment:
source .env.alertmanager
```

### Step 2: Create Kubernetes Secrets

```bash
# Delete existing secret (if present):
kubectl delete secret alertmanager-secrets -n neurectomy 2>/dev/null || true

# Create from environment variables:
kubectl create secret generic alertmanager-secrets \
  --from-literal=slack_webhook_url="$SLACK_WEBHOOK_URL" \
  --from-literal=pagerduty_service_key_critical="$PAGERDUTY_SERVICE_KEY_CRITICAL" \
  --from-literal=pagerduty_service_key_slo="$PAGERDUTY_SERVICE_KEY_SLO" \
  --from-literal=pagerduty_service_key_security="$PAGERDUTY_SERVICE_KEY_SECURITY" \
  --from-literal=smtp_host="$SMTP_HOST" \
  --from-literal=smtp_port="$SMTP_PORT" \
  --from-literal=smtp_username="$SMTP_USERNAME" \
  --from-literal=smtp_password="$SMTP_PASSWORD" \
  --from-literal=smtp_from="$SMTP_FROM" \
  --from-literal=smtp_to_critical="$SMTP_TO_CRITICAL" \
  --from-literal=opsgenie_api_key="$OPSGENIE_API_KEY" \
  -n neurectomy

# Verify secret created:
kubectl get secret alertmanager-secrets -n neurectomy -o yaml

# Verify values are encoded:
kubectl get secret alertmanager-secrets -n neurectomy \
  -o jsonpath='{.data.slack_webhook_url}' | base64 -d
# Expected: Your Slack webhook URL
```

### Step 3: Update AlertManager StatefulSet to Use Secrets

```yaml
# In deploy/k8s/11-alertmanager-statefulset.yaml:

spec:
  template:
    spec:
      containers:
        - name: alertmanager
          env:
            - name: SLACK_WEBHOOK_URL
              valueFrom:
                secretKeyRef:
                  name: alertmanager-secrets
                  key: slack_webhook_url
            - name: PAGERDUTY_SERVICE_KEY_CRITICAL
              valueFrom:
                secretKeyRef:
                  name: alertmanager-secrets
                  key: pagerduty_service_key_critical
            - name: SMTP_HOST
              valueFrom:
                secretKeyRef:
                  name: alertmanager-secrets
                  key: smtp_host
          # ... repeat for all secrets ...
```

---

## 6. ALERTMANAGER.YML SECRET TEMPLATING

### Update Configuration to Use Environment Variables

```yaml
# /docker/alertmanager/alertmanager.yml

global:
  resolve_timeout: 5m
  slack_api_url: "${SLACK_WEBHOOK_URL}"

  pagerduty_url: "https://events.pagerduty.com/v2/enqueue"

  smtp_smarthost: "${SMTP_HOST}:${SMTP_PORT}"
  smtp_from: "${SMTP_FROM}"
  smtp_auth_username: "${SMTP_USERNAME}"
  smtp_auth_password: "${SMTP_PASSWORD}"

  opsgenie_api_url: "https://api.opsgenie.com/"
  opsgenie_api_key: "${OPSGENIE_API_KEY}"

route:
  # ... routing rules ...

receivers:
  - name: critical-pagerduty
    pagerduty_configs:
      - service_key: "${PAGERDUTY_SERVICE_KEY_CRITICAL}"

  - name: critical-slo
    pagerduty_configs:
      - service_key: "${PAGERDUTY_SERVICE_KEY_SLO}"

  - name: security-critical
    pagerduty_configs:
      - service_key: "${PAGERDUTY_SERVICE_KEY_SECURITY}"
```

### Note: AlertManager Does NOT Support Direct Env Variable Substitution

**Solution: Use Wrapper Script**

```bash
#!/bin/bash
# /docker/alertmanager/start.sh

# Substitute environment variables into config file
envsubst < /etc/alertmanager/alertmanager.yml.template > /etc/alertmanager/alertmanager.yml

# Start AlertManager
/bin/alertmanager \
  --config.file=/etc/alertmanager/alertmanager.yml \
  --storage.path=/alertmanager \
  --web.external-url=http://alertmanager.neurectomy.local:9093
```

Update Dockerfile:

```dockerfile
# In Dockerfile
FROM prom/alertmanager:v0.x

# Install gettext (for envsubst)
RUN apk add --no-cache gettext

# Copy config template and start script
COPY alertmanager.yml.template /etc/alertmanager/
COPY start.sh /

ENTRYPOINT ["/start.sh"]
```

---

## 7. VERIFICATION CHECKLIST

### Pre-Deployment

- [ ] All environment variables sourced: `env | grep -E "(SLACK|PAGERDUTY|SMTP|OPSGENIE)"`
- [ ] Slack webhook responds: `curl -X POST $SLACK_WEBHOOK_URL -d '{"text":"test"}' | grep -q ok`
- [ ] PagerDuty event sent successfully (check console)
- [ ] SMTP connection successful: `telnet $SMTP_HOST $SMTP_PORT`
- [ ] OpsGenie API accessible: `curl https://api.opsgenie.com/v1/integrations/validate -H "Authorization: GenieKey $OPSGENIE_API_KEY"`

### Post-Deployment

```bash
# Verify Kubernetes secrets created:
kubectl get secrets -n neurectomy | grep alertmanager

# Verify AlertManager pod has access to secrets:
kubectl exec -it alertmanager-0 -n neurectomy -- env | grep -E "(SLACK|PAGERDUTY|SMTP)"

# Verify AlertManager config valid:
kubectl exec -it alertmanager-0 -n neurectomy -- amtool config routes

# Test alert routing (see PHASE-18B-TESTING-VALIDATION-GUIDE.md):
kubectl apply -f - << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: test-alert-rule
  namespace: neurectomy
data:
  test.yml: |
    groups:
      - name: test
        interval: 10s
        rules:
          - alert: TestAlert
            expr: 1
            labels:
              severity: critical
            annotations:
              summary: "Testing Alert Routing"
EOF
```

---

## 8. ENVIRONMENT VARIABLE REFERENCE TABLE

| Variable                         | Source          | Sensitivity | Example                       |
| -------------------------------- | --------------- | ----------- | ----------------------------- |
| `SLACK_WEBHOOK_URL`              | Slack API       | HIGH        | `https://hooks.slack.com/...` |
| `PAGERDUTY_SERVICE_KEY_CRITICAL` | PagerDuty API   | HIGH        | `ba1234567890...`             |
| `PAGERDUTY_SERVICE_KEY_SLO`      | PagerDuty API   | HIGH        | `bc1234567890...`             |
| `PAGERDUTY_SERVICE_KEY_SECURITY` | PagerDuty API   | HIGH        | `bd1234567890...`             |
| `SMTP_HOST`                      | Mail server     | LOW         | `smtp.gmail.com`              |
| `SMTP_PORT`                      | Mail server     | LOW         | `587`                         |
| `SMTP_USERNAME`                  | Mail account    | MEDIUM      | `your-email@gmail.com`        |
| `SMTP_PASSWORD`                  | Mail account    | HIGH        | `xxxx xxxx xxxx xxxx`         |
| `SMTP_FROM`                      | Mail account    | LOW         | `alerts@neurectomy.local`     |
| `SMTP_TO_CRITICAL`               | Recipient list  | LOW         | `oncall@neurectomy.local`     |
| `OPSGENIE_API_KEY`               | OpsGenie API    | HIGH        | `12345678-...`                |
| `OPSGENIE_REGION`                | OpsGenie config | LOW         | `us` or `eu`                  |

**Sensitivity Levels:**

- HIGH: Store in sealed secret, rotate quarterly, audit access
- MEDIUM: Store in secret, standard security practices
- LOW: Can be in ConfigMap, public knowledge

---

## 9. TROUBLESHOOTING ENVIRONMENT SETUP

### Problem: Slack webhook returns 403

```bash
# Solution:
# 1. Verify webhook URL hasn't expired (expire after 30 days of no use)
# 2. Create new webhook: https://api.slack.com/apps/YOUR-APP/incoming-webhooks
# 3. Verify channel still exists
# 4. Verify app not deactivated

curl -X POST $SLACK_WEBHOOK_URL -H 'Content-Type: application/json' \
  -d '{"text":"Test"}' -v
# Expected: 200 ok
```

### Problem: PagerDuty authentication fails

```bash
# Solution:
# 1. Verify service key not expired
# 2. Create new integration: https://app.pagerduty.com/services/

# Test with curl:
curl -X POST https://events.pagerduty.com/v2/enqueue \
  -H 'Content-Type: application/json' \
  -d '{
    "routing_key": "'$PAGERDUTY_SERVICE_KEY_CRITICAL'",
    "event_action": "trigger",
    "payload": {"summary": "Test", "severity": "critical", "source": "test"}
  }' -v

# Expected: 202 Accepted
```

### Problem: SMTP connection timeout

```bash
# Solution:
# 1. Verify firewall allows port 587 (TLS) or 25 (plain)
# 2. Verify SMTP server accessible from container/pod

# Test connection:
timeout 5 bash -c "exec 3<>/dev/tcp/$SMTP_HOST/$SMTP_PORT; cat <&3"

# Expected: "220 smtp.gmail.com ESMTP"

# If fails: Check firewall, VPN, proxy settings
```

---

## 10. PRODUCTION ENVIRONMENT CHECKLIST

- [ ] All 3 PagerDuty service keys obtained and validated
- [ ] Slack webhook URL created and tested
- [ ] SMTP credentials configured and tested
- [ ] OpsGenie integration completed (if using)
- [ ] All secrets created in Kubernetes
- [ ] AlertManager config templated with env variables
- [ ] AlertManager deployment updated to use secrets
- [ ] Webhook start.sh script created and Dockerfile updated
- [ ] All environment variables documented in runbook
- [ ] Rotation schedule established for API keys/passwords
- [ ] Access controls verified (who can view secrets)
- [ ] Audit logging enabled for secret access

---

**Document Version:** 1.0
**Last Updated:** Phase 18B  
**Status:** Ready for Configuration
