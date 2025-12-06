# Backup Procedures

## Overview

This document details the backup procedures for NEURECTOMY, including Velero configuration, backup schedules, and verification processes.

## Backup Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        NEURECTOMY Backup Architecture                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    EKS CLUSTER                                                               │
│    ┌────────────────────────────────────────────────────────────────┐       │
│    │                                                                  │       │
│    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │       │
│    │  │ ML Service   │  │ PostgreSQL   │  │ MLflow       │          │       │
│    │  │ (Stateless)  │  │ (Stateful)   │  │ (Stateful)   │          │       │
│    │  └──────────────┘  └──────┬───────┘  └──────┬───────┘          │       │
│    │                           │                  │                   │       │
│    │                    ┌──────▼──────────────────▼──────┐           │       │
│    │                    │      EBS Volumes               │           │       │
│    │                    │      (PersistentVolumeClaims)  │           │       │
│    │                    └──────────────┬─────────────────┘           │       │
│    │                                   │                              │       │
│    └───────────────────────────────────┼──────────────────────────────┘       │
│                                        │                                      │
│    VELERO                              │                                      │
│    ┌───────────────────────────────────▼──────────────────────────────┐      │
│    │                                                                   │      │
│    │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │      │
│    │   │ K8s         │    │ EBS         │    │ S3          │         │      │
│    │   │ Resources   │    │ Snapshots   │    │ Artifacts   │         │      │
│    │   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘         │      │
│    │          │                  │                  │                 │      │
│    └──────────┼──────────────────┼──────────────────┼─────────────────┘      │
│               │                  │                  │                        │
│               ▼                  ▼                  ▼                        │
│    ┌──────────────────────────────────────────────────────────────────┐     │
│    │                    S3: neurectomy-backups-us-east-1              │     │
│    │                                                                   │     │
│    │   /velero/backups/hourly-*/                                      │     │
│    │   /velero/backups/daily-*/                                       │     │
│    │   /velero/backups/weekly-*/                                      │     │
│    │   /restic/                                                       │     │
│    └────────────────────────────────┬─────────────────────────────────┘     │
│                                     │                                        │
│                         Cross-Region Replication                             │
│                                     │                                        │
│                                     ▼                                        │
│    ┌──────────────────────────────────────────────────────────────────┐     │
│    │                    S3: neurectomy-backups-us-west-2              │     │
│    │                    (DR Region - Secondary)                       │     │
│    └──────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Velero Backup Schedules

### Production Schedules

| Schedule          | Frequency           | Retention | Scope          |
| ----------------- | ------------------- | --------- | -------------- |
| neurectomy-hourly | Every hour          | 24 hours  | Critical data  |
| neurectomy-daily  | Daily at 02:00 UTC  | 30 days   | Full namespace |
| neurectomy-weekly | Sunday at 03:00 UTC | 12 weeks  | Full namespace |
| dr-replication    | Every 15 min        | 48 hours  | DR sync        |

### Schedule Definitions

```yaml
# Hourly Backup - Critical Data
apiVersion: velero.io/v1
kind: Schedule
metadata:
  name: neurectomy-hourly
  namespace: velero
spec:
  schedule: "0 * * * *" # Every hour
  template:
    ttl: 24h
    includedNamespaces:
      - neurectomy
    includedResources:
      - persistentvolumeclaims
      - secrets
      - configmaps
    labelSelector:
      matchExpressions:
        - key: backup-tier
          operator: In
          values: ["critical", "high"]
    snapshotVolumes: true
    storageLocation: aws-primary
    volumeSnapshotLocations:
      - aws-ebs-us-east-1

---
# Daily Backup - Full Namespace
apiVersion: velero.io/v1
kind: Schedule
metadata:
  name: neurectomy-daily
  namespace: velero
spec:
  schedule: "0 2 * * *" # 02:00 UTC daily
  template:
    ttl: 720h # 30 days
    includedNamespaces:
      - neurectomy
      - monitoring
    snapshotVolumes: true
    storageLocation: aws-primary
    volumeSnapshotLocations:
      - aws-ebs-us-east-1
    hooks:
      resources:
        - name: postgresql-backup-hook
          includedNamespaces:
            - neurectomy
          labelSelector:
            matchLabels:
              app: postgresql
          pre:
            - exec:
                container: postgresql
                command:
                  - /bin/bash
                  - -c
                  - "pg_dump -U postgres neurectomy > /var/lib/postgresql/data/backup.sql"
                onError: Fail
                timeout: 5m
```

## Backup Procedures

### Manual Backup (Ad-hoc)

```bash
# Create manual backup before major changes
velero backup create manual-$(date +%Y%m%d-%H%M%S) \
  --include-namespaces neurectomy \
  --snapshot-volumes \
  --wait

# Create backup with specific labels
velero backup create pre-upgrade-$(date +%Y%m%d) \
  --include-namespaces neurectomy \
  --selector app=postgresql \
  --snapshot-volumes

# Create backup excluding specific resources
velero backup create app-backup-$(date +%Y%m%d) \
  --include-namespaces neurectomy \
  --exclude-resources events,pods \
  --snapshot-volumes
```

### Pre-Deployment Backup

```bash
#!/bin/bash
# scripts/pre-deployment-backup.sh

set -e

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BACKUP_NAME="pre-deploy-${TIMESTAMP}"

echo "Creating pre-deployment backup: ${BACKUP_NAME}"

# Create backup
velero backup create "${BACKUP_NAME}" \
  --include-namespaces neurectomy \
  --snapshot-volumes \
  --wait

# Verify backup completed
STATUS=$(velero backup describe "${BACKUP_NAME}" -o json | jq -r '.status.phase')

if [ "$STATUS" != "Completed" ]; then
  echo "ERROR: Backup failed with status: ${STATUS}"
  exit 1
fi

echo "Backup ${BACKUP_NAME} completed successfully"
echo "Proceed with deployment..."
```

### PostgreSQL-Specific Backup

```bash
#!/bin/bash
# scripts/postgresql-backup.sh

set -e

POD_NAME=$(kubectl get pods -n neurectomy -l app=postgresql -o jsonpath='{.items[0].metadata.name}')
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BACKUP_FILE="neurectomy-db-${TIMESTAMP}.sql.gz"

echo "Creating PostgreSQL backup: ${BACKUP_FILE}"

# Create SQL dump
kubectl exec -n neurectomy "${POD_NAME}" -- \
  pg_dump -U postgres neurectomy | gzip > "/tmp/${BACKUP_FILE}"

# Upload to S3
aws s3 cp "/tmp/${BACKUP_FILE}" "s3://neurectomy-backups-us-east-1/database-backups/${BACKUP_FILE}"

# Cleanup
rm "/tmp/${BACKUP_FILE}"

echo "PostgreSQL backup completed: s3://neurectomy-backups-us-east-1/database-backups/${BACKUP_FILE}"
```

### MLflow Artifacts Backup

```bash
#!/bin/bash
# scripts/mlflow-artifacts-backup.sh

set -e

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
SOURCE_BUCKET="neurectomy-mlflow-artifacts"
BACKUP_BUCKET="neurectomy-backups-us-east-1"

echo "Syncing MLflow artifacts to backup bucket..."

aws s3 sync \
  "s3://${SOURCE_BUCKET}/" \
  "s3://${BACKUP_BUCKET}/mlflow-artifacts-backup/${TIMESTAMP}/" \
  --storage-class STANDARD_IA

echo "MLflow artifacts backup completed"
```

## Backup Verification

### Daily Verification Job

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: backup-verification
  namespace: velero
spec:
  schedule: "0 6 * * *" # 06:00 UTC daily
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: velero
          containers:
            - name: verify
              image: velero/velero:v1.12.2
              command:
                - /bin/bash
                - -c
                - |
                  #!/bin/bash
                  set -e

                  # Get latest backup
                  LATEST_BACKUP=$(velero backup get -o json | jq -r '.items | sort_by(.metadata.creationTimestamp) | last | .metadata.name')

                  echo "Verifying backup: ${LATEST_BACKUP}"

                  # Check backup status
                  STATUS=$(velero backup describe "${LATEST_BACKUP}" -o json | jq -r '.status.phase')

                  if [ "$STATUS" != "Completed" ]; then
                    echo "ALERT: Latest backup status is ${STATUS}"
                    exit 1
                  fi

                  # Check backup age
                  BACKUP_TIME=$(velero backup describe "${LATEST_BACKUP}" -o json | jq -r '.status.completionTimestamp')
                  BACKUP_EPOCH=$(date -d "${BACKUP_TIME}" +%s)
                  NOW_EPOCH=$(date +%s)
                  AGE_HOURS=$(( (NOW_EPOCH - BACKUP_EPOCH) / 3600 ))

                  if [ $AGE_HOURS -gt 25 ]; then
                    echo "ALERT: Latest backup is ${AGE_HOURS} hours old"
                    exit 1
                  fi

                  # Check item counts
                  ITEMS=$(velero backup describe "${LATEST_BACKUP}" -o json | jq '.status.progress.totalItems')

                  if [ "$ITEMS" -lt 50 ]; then
                    echo "WARNING: Backup contains only ${ITEMS} items"
                  fi

                  echo "Backup verification passed"
          restartPolicy: OnFailure
```

### Restore Test Procedure

```bash
#!/bin/bash
# scripts/backup-restore-test.sh

set -e

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
TEST_NAMESPACE="restore-test-${TIMESTAMP}"
LATEST_BACKUP=$(velero backup get -o json | jq -r '.items | sort_by(.metadata.creationTimestamp) | last | .metadata.name')

echo "Testing restore of backup: ${LATEST_BACKUP}"
echo "Target namespace: ${TEST_NAMESPACE}"

# Create test namespace
kubectl create namespace "${TEST_NAMESPACE}"

# Perform restore to test namespace
velero restore create "restore-test-${TIMESTAMP}" \
  --from-backup "${LATEST_BACKUP}" \
  --namespace-mappings "neurectomy:${TEST_NAMESPACE}" \
  --exclude-resources ingresses,services \
  --wait

# Verify critical components restored
echo "Verifying restored components..."

# Check PostgreSQL
kubectl wait --for=condition=ready pod -l app=postgresql -n "${TEST_NAMESPACE}" --timeout=300s
PG_STATUS=$(kubectl exec -n "${TEST_NAMESPACE}" -l app=postgresql -- pg_isready -U postgres)
echo "PostgreSQL status: ${PG_STATUS}"

# Check data integrity
RECORD_COUNT=$(kubectl exec -n "${TEST_NAMESPACE}" -l app=postgresql -- psql -U postgres -t -c "SELECT count(*) FROM ml_models;" 2>/dev/null || echo "0")
echo "Record count in ml_models: ${RECORD_COUNT}"

# Cleanup test namespace
echo "Cleaning up test namespace..."
kubectl delete namespace "${TEST_NAMESPACE}"

echo "Restore test completed successfully"
```

## Backup Monitoring

### Prometheus Metrics

```yaml
# Backup metrics to monitor
groups:
  - name: velero-backup-metrics
    rules:
      # Backup success rate
      - record: velero:backup_success_rate:1d
        expr: |
          sum(velero_backup_success_total) /
          (sum(velero_backup_success_total) + sum(velero_backup_failure_total))

      # Last successful backup age
      - record: velero:backup_last_successful_age_seconds
        expr: |
          time() - max(velero_backup_last_successful_timestamp)

      # Backup duration
      - record: velero:backup_duration_seconds:avg
        expr: |
          avg(velero_backup_duration_seconds)

# Alert rules
groups:
  - name: velero-alerts
    rules:
      - alert: VeleroBackupFailed
        expr: increase(velero_backup_failure_total[1h]) > 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Velero backup failed"
          description: "A Velero backup has failed in the last hour"

      - alert: VeleroBackupMissing
        expr: velero:backup_last_successful_age_seconds > 7200
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "No successful backup in 2 hours"
          description: "The last successful Velero backup was {{ $value | humanizeDuration }} ago"

      - alert: VeleroBackupAgeExceedsRPO
        expr: velero:backup_last_successful_age_seconds > 900
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Backup age exceeds RPO"
          description: "No backup in the last 15 minutes (RPO threshold)"
```

### Grafana Dashboard Queries

```yaml
# Panel: Backup Success Rate (24h)
Query: |
  sum(increase(velero_backup_success_total[24h])) /
  (sum(increase(velero_backup_success_total[24h])) +
   sum(increase(velero_backup_failure_total[24h]))) * 100

# Panel: Last Backup Age
Query: |
  (time() - max(velero_backup_last_successful_timestamp{schedule=~"neurectomy-.*"})) / 60

# Panel: Backup Duration Trend
Query: |
  avg(velero_backup_duration_seconds{schedule=~"neurectomy-.*"}) by (schedule)

# Panel: Backup Size Trend
Query: |
  sum(velero_backup_total_bytes{schedule=~"neurectomy-.*"}) by (schedule)
```

## S3 Lifecycle Policies

### Backup Bucket Lifecycle

```json
{
  "Rules": [
    {
      "ID": "TransitionHourlyToIA",
      "Status": "Enabled",
      "Filter": {
        "Prefix": "velero/backups/hourly-"
      },
      "Transitions": [
        {
          "Days": 1,
          "StorageClass": "STANDARD_IA"
        }
      ],
      "Expiration": {
        "Days": 2
      }
    },
    {
      "ID": "TransitionDailyToIA",
      "Status": "Enabled",
      "Filter": {
        "Prefix": "velero/backups/daily-"
      },
      "Transitions": [
        {
          "Days": 7,
          "StorageClass": "STANDARD_IA"
        },
        {
          "Days": 30,
          "StorageClass": "GLACIER"
        }
      ],
      "Expiration": {
        "Days": 365
      }
    },
    {
      "ID": "TransitionWeeklyToGlacier",
      "Status": "Enabled",
      "Filter": {
        "Prefix": "velero/backups/weekly-"
      },
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "GLACIER"
        },
        {
          "Days": 90,
          "StorageClass": "DEEP_ARCHIVE"
        }
      ],
      "Expiration": {
        "Days": 2555
      }
    }
  ]
}
```

## Disaster Recovery Sync

### Cross-Region Replication

```bash
# Verify cross-region replication status
aws s3api get-bucket-replication --bucket neurectomy-backups-us-east-1

# Check replication metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/S3 \
  --metric-name ReplicationLatency \
  --dimensions Name=SourceBucket,Value=neurectomy-backups-us-east-1 \
               Name=DestinationBucket,Value=neurectomy-backups-us-west-2 \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%SZ) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
  --period 300 \
  --statistics Average
```

## Related Documents

- [DR Overview](./DR-OVERVIEW.md)
- [Recovery Runbook](./RECOVERY-RUNBOOK.md)
- [DR Testing](./DR-TESTING.md)
