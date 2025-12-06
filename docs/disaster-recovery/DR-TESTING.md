# Disaster Recovery Testing Procedures

## Overview

Regular DR testing is essential to ensure recovery procedures work as expected. This document outlines testing schedules, procedures, and success criteria.

## Testing Schedule

### Frequency Matrix

| Test Type           | Frequency     | Duration  | Stakeholders     |
| ------------------- | ------------- | --------- | ---------------- |
| Backup Verification | Daily         | 15 min    | Automated        |
| Component Restore   | Weekly        | 1 hour    | Platform Team    |
| Partial Failover    | Monthly       | 2-4 hours | Platform + SRE   |
| Full DR Test        | Quarterly     | 4-8 hours | All Engineering  |
| Multi-Cloud Test    | Semi-annually | Full day  | All + Executives |
| Chaos Engineering   | Continuous    | Varies    | Platform Team    |

### Annual DR Test Calendar

```
Q1: January
  - Week 2: Quarterly Full DR Test
  - Week 4: Database recovery test

Q2: April
  - Week 2: Quarterly Full DR Test + Multi-Cloud
  - Week 4: Network isolation test

Q3: July
  - Week 2: Quarterly Full DR Test
  - Week 4: Data corruption recovery test

Q4: October
  - Week 2: Quarterly Full DR Test + Multi-Cloud
  - Week 4: Complete chaos day
```

## Test Types

### Type 1: Backup Verification (Daily)

**Objective:** Verify backup completeness and integrity

**Automated Process:**

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: daily-backup-verification
  namespace: velero
spec:
  schedule: "0 8 * * *" # 08:00 UTC daily
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: velero-test
          containers:
            - name: verify
              image: bitnami/kubectl:latest
              command:
                - /bin/bash
                - -c
                - |
                  #!/bin/bash
                  set -e

                  # Get latest backup
                  LATEST=$(velero backup get -o json | jq -r '.items | sort_by(.metadata.creationTimestamp) | last | .metadata.name')

                  echo "Verifying backup: ${LATEST}"

                  # Check status
                  STATUS=$(velero backup describe "${LATEST}" -o json | jq -r '.status.phase')
                  if [ "$STATUS" != "Completed" ]; then
                    echo "FAIL: Backup status is ${STATUS}"
                    exit 1
                  fi

                  # Check items count
                  ITEMS=$(velero backup describe "${LATEST}" -o json | jq '.status.progress.totalItems')
                  if [ "$ITEMS" -lt 50 ]; then
                    echo "WARN: Low item count: ${ITEMS}"
                  fi

                  # Check volume snapshots
                  SNAPSHOTS=$(velero backup describe "${LATEST}" -o json | jq '.status.volumeSnapshotsAttempted')
                  COMPLETED=$(velero backup describe "${LATEST}" -o json | jq '.status.volumeSnapshotsCompleted')
                  if [ "$SNAPSHOTS" != "$COMPLETED" ]; then
                    echo "FAIL: Volume snapshots incomplete: ${COMPLETED}/${SNAPSHOTS}"
                    exit 1
                  fi

                  echo "SUCCESS: Backup verification passed"
          restartPolicy: OnFailure
```

**Success Criteria:**

- [ ] Latest backup status is "Completed"
- [ ] Item count is within expected range
- [ ] All volume snapshots completed
- [ ] Backup age < 2 hours

### Type 2: Component Restore Test (Weekly)

**Objective:** Verify ability to restore individual components

**Procedure:**

```bash
#!/bin/bash
# scripts/weekly-restore-test.sh

set -e

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
TEST_NS="restore-test-${TIMESTAMP}"

echo "=== Weekly Component Restore Test ==="
echo "Timestamp: ${TIMESTAMP}"
echo "Test Namespace: ${TEST_NS}"

# Create test namespace
kubectl create namespace "${TEST_NS}"

# Get latest backup
BACKUP=$(velero backup get -o json | jq -r '.items | sort_by(.metadata.creationTimestamp) | last | .metadata.name')
echo "Using backup: ${BACKUP}"

# Test 1: Restore ConfigMaps
echo "Test 1: Restoring ConfigMaps..."
velero restore create "test-cm-${TIMESTAMP}" \
  --from-backup "${BACKUP}" \
  --include-resources configmaps \
  --namespace-mappings neurectomy:${TEST_NS} \
  --wait

CM_COUNT=$(kubectl get configmaps -n "${TEST_NS}" --no-headers | wc -l)
echo "ConfigMaps restored: ${CM_COUNT}"

# Test 2: Restore Secrets
echo "Test 2: Restoring Secrets..."
velero restore create "test-secrets-${TIMESTAMP}" \
  --from-backup "${BACKUP}" \
  --include-resources secrets \
  --namespace-mappings neurectomy:${TEST_NS} \
  --wait

SECRET_COUNT=$(kubectl get secrets -n "${TEST_NS}" --no-headers | wc -l)
echo "Secrets restored: ${SECRET_COUNT}"

# Test 3: Restore Deployments (without running)
echo "Test 3: Restoring Deployments..."
velero restore create "test-deploy-${TIMESTAMP}" \
  --from-backup "${BACKUP}" \
  --include-resources deployments \
  --namespace-mappings neurectomy:${TEST_NS} \
  --wait

# Scale down immediately (don't actually run workloads)
kubectl scale deployment --all --replicas=0 -n "${TEST_NS}"

DEPLOY_COUNT=$(kubectl get deployments -n "${TEST_NS}" --no-headers | wc -l)
echo "Deployments restored: ${DEPLOY_COUNT}"

# Cleanup
echo "Cleaning up test namespace..."
kubectl delete namespace "${TEST_NS}"

# Generate report
echo ""
echo "=== Weekly Restore Test Report ==="
echo "Date: $(date)"
echo "Backup Used: ${BACKUP}"
echo "ConfigMaps: ${CM_COUNT}"
echo "Secrets: ${SECRET_COUNT}"
echo "Deployments: ${DEPLOY_COUNT}"
echo "Status: SUCCESS"
echo "=================================="
```

**Success Criteria:**

- [ ] All ConfigMaps restored correctly
- [ ] All Secrets restored (encrypted)
- [ ] All Deployments restored
- [ ] Resources match expected counts

### Type 3: Partial Failover Test (Monthly)

**Objective:** Test failover of non-critical services to DR region

**Scope:** Monitoring stack (Prometheus, Grafana)

**Procedure:**

```bash
#!/bin/bash
# scripts/monthly-partial-failover.sh

set -e

echo "=== Monthly Partial Failover Test ==="
echo "Target: Monitoring stack (Prometheus, Grafana)"
echo "Start Time: $(date)"

# Phase 1: Pre-test validation
echo ""
echo "Phase 1: Pre-test Validation"
echo "----------------------------"

# Check primary monitoring
kubectl --context primary get pods -n monitoring
PRIMARY_PROM_STATUS=$(kubectl --context primary exec -n monitoring deploy/prometheus -- curl -s localhost:9090/-/healthy)
echo "Primary Prometheus: ${PRIMARY_PROM_STATUS}"

# Check DR monitoring
kubectl --context dr get pods -n monitoring
DR_PROM_STATUS=$(kubectl --context dr exec -n monitoring deploy/prometheus -- curl -s localhost:9090/-/healthy)
echo "DR Prometheus: ${DR_PROM_STATUS}"

# Phase 2: Simulate primary monitoring failure
echo ""
echo "Phase 2: Simulating Primary Failure"
echo "------------------------------------"

# Scale down primary monitoring
kubectl --context primary scale deployment prometheus --replicas=0 -n monitoring
kubectl --context primary scale deployment grafana --replicas=0 -n monitoring

echo "Primary monitoring scaled down"

# Phase 3: Verify DR monitoring takes over
echo ""
echo "Phase 3: Verifying DR Takeover"
echo "------------------------------"

sleep 30  # Wait for alerting thresholds

# Check Grafana datasource failover
DR_GRAFANA_URL=$(kubectl --context dr get svc grafana -n monitoring -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
curl -s "http://${DR_GRAFANA_URL}/api/health" | jq .

# Run test queries
kubectl --context dr exec -n monitoring deploy/prometheus -- \
  curl -s 'localhost:9090/api/v1/query?query=up' | jq '.data.result | length'

# Phase 4: Recovery
echo ""
echo "Phase 4: Recovery"
echo "-----------------"

# Restore primary monitoring
kubectl --context primary scale deployment prometheus --replicas=2 -n monitoring
kubectl --context primary scale deployment grafana --replicas=1 -n monitoring

# Wait for recovery
kubectl --context primary rollout status deployment/prometheus -n monitoring --timeout=300s
kubectl --context primary rollout status deployment/grafana -n monitoring --timeout=300s

echo "Primary monitoring restored"

# Phase 5: Post-test validation
echo ""
echo "Phase 5: Post-test Validation"
echo "-----------------------------"

kubectl --context primary get pods -n monitoring
kubectl --context dr get pods -n monitoring

echo ""
echo "=== Partial Failover Test Complete ==="
echo "End Time: $(date)"
```

**Success Criteria:**

- [ ] Primary monitoring scaled down successfully
- [ ] DR monitoring continued collecting metrics
- [ ] Grafana dashboards accessible via DR
- [ ] Primary monitoring restored successfully
- [ ] No metric data gaps > 2 minutes

### Type 4: Full DR Test (Quarterly)

**Objective:** Complete failover to DR region and back

**Duration:** 4-8 hours

**Pre-Test Checklist:**

- [ ] Stakeholder notification sent
- [ ] Support team on standby
- [ ] Customer notification (if production)
- [ ] Rollback plan confirmed
- [ ] All participants confirmed

**Test Plan:**

```markdown
# Quarterly Full DR Test Plan

## Schedule

- Start: [DATE] 06:00 UTC (low traffic window)
- Duration: 4-6 hours
- Rollback deadline: [DATE] 12:00 UTC

## Participants

- Test Lead: [Name]
- Platform Engineers: [Names]
- SRE On-Call: [Name]
- Stakeholder Observer: [Name]

## Phases

### Phase 1: Pre-Test (30 min)

1. Verify all systems healthy
2. Create pre-test backup
3. Document baseline metrics
4. Confirm all participants ready

### Phase 2: Failover (1 hour)

1. Initiate region failover script
2. Monitor failover progress
3. Validate DR region active
4. Update DNS records
5. Verify customer-facing endpoints

### Phase 3: Validation (1 hour)

1. Run smoke tests
2. Run integration tests
3. Verify data integrity
4. Check monitoring/alerting
5. Validate backup in DR region

### Phase 4: Soak Period (1-2 hours)

1. Monitor system under DR
2. Execute sample workloads
3. Monitor performance metrics
4. Check for anomalies

### Phase 5: Failback (1 hour)

1. Initiate failback procedure
2. Sync data back to primary
3. Restore primary region
4. Update DNS records
5. Verify primary active

### Phase 6: Post-Test (30 min)

1. Verify all systems healthy
2. Document any issues
3. Update runbooks if needed
4. Send completion report
```

**Success Criteria:**

- [ ] RTO met (< 30 minutes for failover)
- [ ] RPO met (< 15 minutes data loss)
- [ ] All critical services operational in DR
- [ ] Successful failback to primary
- [ ] No data loss or corruption
- [ ] All tests passed

### Type 5: Chaos Engineering (Continuous)

**Objective:** Continuously test system resilience

**Tools:** Chaos Mesh, Litmus

**Chaos Experiments:**

```yaml
# chaos-mesh/experiments/pod-failure.yaml
apiVersion: chaos-mesh.org/v1alpha1
kind: PodChaos
metadata:
  name: ml-service-pod-kill
  namespace: chaos-testing
spec:
  action: pod-kill
  mode: one
  selector:
    namespaces:
      - neurectomy
    labelSelectors:
      app: ml-service
  scheduler:
    cron: "@every 4h"
---
# chaos-mesh/experiments/network-delay.yaml
apiVersion: chaos-mesh.org/v1alpha1
kind: NetworkChaos
metadata:
  name: database-network-delay
  namespace: chaos-testing
spec:
  action: delay
  mode: all
  selector:
    namespaces:
      - neurectomy
    labelSelectors:
      app: postgresql
  delay:
    latency: "100ms"
    correlation: "25"
    jitter: "25ms"
  duration: "5m"
  scheduler:
    cron: "0 */6 * * *"
---
# chaos-mesh/experiments/stress-test.yaml
apiVersion: chaos-mesh.org/v1alpha1
kind: StressChaos
metadata:
  name: ml-service-cpu-stress
  namespace: chaos-testing
spec:
  mode: one
  selector:
    namespaces:
      - neurectomy
    labelSelectors:
      app: ml-service
  stressors:
    cpu:
      workers: 2
      load: 80
  duration: "10m"
  scheduler:
    cron: "0 2 * * 1" # Every Monday at 2 AM
```

**Monitoring Chaos Tests:**

```yaml
# Prometheus rules for chaos test monitoring
groups:
  - name: chaos-monitoring
    rules:
      - alert: ChaosExperimentActive
        expr: chaos_mesh_experiments_active > 0
        for: 1m
        labels:
          severity: info
        annotations:
          summary: "Chaos experiment in progress"

      - alert: ServiceDegradedDuringChaos
        expr: |
          (
            sum(rate(http_requests_total{status=~"5.."}[5m])) /
            sum(rate(http_requests_total[5m]))
          ) > 0.01
          and
          chaos_mesh_experiments_active > 0
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Service degraded during chaos test"
```

## Test Reporting

### Test Report Template

```markdown
# DR Test Report

## Test Information

- **Test Type:** [Full DR / Partial Failover / Component Restore]
- **Date:** YYYY-MM-DD
- **Duration:** X hours Y minutes
- **Test Lead:** [Name]

## Executive Summary

[Brief summary of test results]

## Objectives

- [ ] Objective 1
- [ ] Objective 2
- [ ] Objective 3

## Results

### Metrics

| Metric         | Target | Actual | Status |
| -------------- | ------ | ------ | ------ |
| RTO            | 30 min | XX min | ✅/❌  |
| RPO            | 15 min | XX min | ✅/❌  |
| Failover Time  | XX min | XX min | ✅/❌  |
| Data Integrity | 100%   | XX%    | ✅/❌  |

### Phase Results

| Phase      | Duration | Status | Notes |
| ---------- | -------- | ------ | ----- |
| Pre-Test   | XX min   | ✅/❌  |       |
| Failover   | XX min   | ✅/❌  |       |
| Validation | XX min   | ✅/❌  |       |
| Failback   | XX min   | ✅/❌  |       |

## Issues Encountered

1. **Issue:** [Description]
   - **Impact:** [Impact]
   - **Resolution:** [Resolution]
   - **Action Item:** [Follow-up action]

## Recommendations

1. [Recommendation 1]
2. [Recommendation 2]

## Action Items

| Item | Owner | Due Date | Priority |
| ---- | ----- | -------- | -------- |
|      |       |          |          |

## Appendix

- Test logs: [Link]
- Metrics dashboard: [Link]
- Recording: [Link]
```

### Automated Test Reporting

```python
#!/usr/bin/env python3
# scripts/generate_dr_report.py

import json
import datetime
from jinja2 import Template

def generate_dr_report(test_results):
    """Generate DR test report from test results."""

    template = """
# Automated DR Test Report

Generated: {{ timestamp }}

## Test Summary
- **Test Type:** {{ test_type }}
- **Status:** {{ "✅ PASSED" if passed else "❌ FAILED" }}
- **Duration:** {{ duration_minutes }} minutes

## Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
{% for metric in metrics %}
| {{ metric.name }} | {{ metric.target }} | {{ metric.actual }} | {{ "✅" if metric.passed else "❌" }} |
{% endfor %}

## Component Status

{% for component in components %}
### {{ component.name }}
- Backup Age: {{ component.backup_age }}
- Restore Time: {{ component.restore_time }}
- Data Integrity: {{ component.data_integrity }}%
{% endfor %}

## Alerts Generated During Test
{% for alert in alerts %}
- {{ alert.timestamp }}: {{ alert.message }} ({{ alert.severity }})
{% endfor %}

## Next Scheduled Test
{{ next_test_date }}
"""

    tmpl = Template(template)
    return tmpl.render(**test_results)

if __name__ == "__main__":
    # Example usage
    results = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "test_type": "Weekly Component Restore",
        "passed": True,
        "duration_minutes": 45,
        "metrics": [
            {"name": "Backup Age", "target": "< 2h", "actual": "1h 23m", "passed": True},
            {"name": "Restore Time", "target": "< 30m", "actual": "15m", "passed": True},
        ],
        "components": [
            {"name": "PostgreSQL", "backup_age": "1h 23m", "restore_time": "8m", "data_integrity": 100},
            {"name": "MLflow", "backup_age": "1h 23m", "restore_time": "4m", "data_integrity": 100},
        ],
        "alerts": [],
        "next_test_date": "2025-02-01"
    }

    print(generate_dr_report(results))
```

## Related Documents

- [DR Overview](./DR-OVERVIEW.md)
- [Recovery Runbook](./RECOVERY-RUNBOOK.md)
- [RTO/RPO Policies](./RTO-RPO-POLICIES.md)
