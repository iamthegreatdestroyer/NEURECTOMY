# RTO and RPO Policies

## Definitions

### Recovery Time Objective (RTO)

The maximum acceptable time that an application can be offline before business impact becomes unacceptable.

### Recovery Point Objective (RPO)

The maximum acceptable amount of data loss measured in time, representing the point to which data must be restored.

## Service Level Objectives

### Tier Classification

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Service Tier Classification                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    CRITICAL (Tier 1)         HIGH (Tier 2)          MEDIUM (Tier 3)         │
│    RTO: 15 min               RTO: 1 hour            RTO: 4 hours            │
│    RPO: 5 min                RPO: 15 min            RPO: 1 hour             │
│                                                                              │
│    ┌─────────────┐           ┌─────────────┐        ┌─────────────┐         │
│    │ ML Service  │           │ PostgreSQL  │        │ Prometheus  │         │
│    │ API Gateway │           │ MLflow      │        │ Grafana     │         │
│    │ Auth        │           │ Redis       │        │ Jaeger      │         │
│    └─────────────┘           └─────────────┘        └─────────────┘         │
│                                                                              │
│                              LOW (Tier 4)                                    │
│                              RTO: 24 hours                                   │
│                              RPO: 24 hours                                   │
│                                                                              │
│                              ┌─────────────┐                                │
│                              │ Dev Envs    │                                │
│                              │ CI/CD       │                                │
│                              └─────────────┘                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Detailed SLOs by Service

### Tier 1 - Critical Services

| Service                | RTO    | RPO             | Availability | Backup Frequency |
| ---------------------- | ------ | --------------- | ------------ | ---------------- |
| ML Service API         | 15 min | 5 min           | 99.99%       | Continuous       |
| API Gateway            | 15 min | N/A (stateless) | 99.99%       | N/A              |
| Authentication Service | 15 min | 5 min           | 99.99%       | Continuous       |

**Justification:**

- Revenue-impacting services
- Customer-facing endpoints
- Security-critical components

**Recovery Strategy:**

- Active-active multi-region where possible
- Automatic failover with Route53
- Hot standby in DR region
- Sub-second database replication

### Tier 2 - High Priority Services

| Service     | RTO    | RPO    | Availability | Backup Frequency |
| ----------- | ------ | ------ | ------------ | ---------------- |
| PostgreSQL  | 1 hour | 15 min | 99.95%       | Every 15 min     |
| MLflow      | 1 hour | 15 min | 99.95%       | Every 15 min     |
| Redis Cache | 1 hour | 30 min | 99.95%       | Every 30 min     |

**Justification:**

- Data persistence critical
- ML experiment tracking
- Session and cache data

**Recovery Strategy:**

- Streaming replication to DR
- Point-in-time recovery capability
- Automated snapshot scheduling
- Cross-region S3 replication for artifacts

### Tier 3 - Medium Priority Services

| Service    | RTO     | RPO     | Availability | Backup Frequency |
| ---------- | ------- | ------- | ------------ | ---------------- |
| Prometheus | 4 hours | 1 hour  | 99.9%        | Hourly           |
| Grafana    | 4 hours | 1 hour  | 99.9%        | Hourly           |
| Jaeger     | 4 hours | 4 hours | 99.9%        | Every 4 hours    |

**Justification:**

- Observability stack (degraded operation acceptable)
- Historical data recoverable
- Non-critical for core operations

**Recovery Strategy:**

- Velero snapshots for PVCs
- Dashboard exports in Git
- Alert rules in GitOps
- Metric data can be rebuilt

### Tier 4 - Low Priority Services

| Service                  | RTO      | RPO      | Availability | Backup Frequency |
| ------------------------ | -------- | -------- | ------------ | ---------------- |
| Development Environments | 24 hours | 24 hours | 99%          | Daily            |
| CI/CD Runners            | 24 hours | N/A      | 99%          | N/A              |
| Documentation Sites      | 24 hours | 24 hours | 99%          | Daily            |

**Justification:**

- Non-production workloads
- Easily recreatable
- Limited business impact

**Recovery Strategy:**

- Infrastructure as Code rebuild
- GitOps-based recreation
- Daily backup suffices

## Data Protection Policies

### Backup Retention

```yaml
Tier 1 (Critical):
  Continuous Replication: Real-time
  Hourly Backups: 72 hours
  Daily Backups: 30 days
  Weekly Backups: 12 weeks
  Monthly Backups: 12 months
  Annual Backups: 7 years (compliance)

Tier 2 (High):
  Hourly Backups: 48 hours
  Daily Backups: 14 days
  Weekly Backups: 8 weeks
  Monthly Backups: 6 months

Tier 3 (Medium):
  Hourly Backups: 24 hours
  Daily Backups: 7 days
  Weekly Backups: 4 weeks

Tier 4 (Low):
  Daily Backups: 7 days
  Weekly Backups: 4 weeks
```

### Geographic Distribution

```yaml
Primary Region (us-east-1):
  - Hot data
  - Active workloads
  - Real-time backups

DR Region (us-west-2):
  - Replicated data (< 1 min lag)
  - Warm standby infrastructure
  - Secondary backups

Tertiary (Azure West US 2):
  - Cold standby
  - Long-term archive
  - Multi-cloud failover

Archive (S3 Glacier):
  - Compliance archives
  - 7-year retention
  - Quarterly access testing
```

## SLA Commitments

### Internal SLAs

| Metric                   | Target   | Measurement            |
| ------------------------ | -------- | ---------------------- |
| Time to Detect           | < 5 min  | Monitoring alerts      |
| Time to Respond          | < 15 min | On-call acknowledgment |
| Time to Resolve (Tier 1) | < 15 min | Service restoration    |
| Time to Resolve (Tier 2) | < 1 hour | Service restoration    |
| Backup Success Rate      | > 99.9%  | Daily verification     |
| DR Test Success          | 100%     | Quarterly testing      |

### Compliance Requirements

```yaml
SOC 2 Type II:
  - Daily backup verification
  - Annual DR testing
  - Access logging
  - Encryption at rest

GDPR:
  - Data residency compliance
  - Right to erasure support
  - Cross-border transfer controls

HIPAA (if applicable):
  - PHI backup encryption
  - Access audit trails
  - Business Associate Agreements
```

## Escalation Thresholds

### Automatic Escalation

| Condition       | Action                           | Timeline  |
| --------------- | -------------------------------- | --------- |
| RTO 50% elapsed | Escalate to Tech Lead            | Automatic |
| RTO 75% elapsed | Escalate to Engineering Director | Automatic |
| RTO exceeded    | Executive notification           | Immediate |
| RPO at risk     | Activate DR procedures           | Immediate |

### Manual Escalation Triggers

- Data corruption detected
- Security incident in progress
- Multiple region failures
- Vendor/third-party outage
- Compliance violation risk

## Monitoring and Alerting

### Key Metrics

```yaml
Recovery Metrics:
  - Last Successful Backup Age
  - Replication Lag
  - DR Sync Status
  - Backup Size Trends
  - Restore Test Results

Alert Thresholds:
  - Backup Age > RPO: CRITICAL
  - Replication Lag > 1 min: WARNING
  - Replication Lag > 5 min: CRITICAL
  - DR Sync Failed: CRITICAL
  - Backup Size Anomaly (>20%): WARNING
```

### Prometheus Rules

```yaml
groups:
  - name: dr-alerts
    rules:
      - alert: BackupAgeExceedsRPO
        expr: time() - neurectomy_last_backup_timestamp > 300
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Backup age exceeds RPO threshold"

      - alert: ReplicationLagHigh
        expr: neurectomy_replication_lag_seconds > 60
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Database replication lag is high"

      - alert: DRSyncFailed
        expr: neurectomy_dr_sync_status != 1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "DR synchronization has failed"
```

## Review and Updates

### Review Schedule

| Review Type          | Frequency       | Participants           |
| -------------------- | --------------- | ---------------------- |
| RTO/RPO Verification | Monthly         | Platform Team          |
| Policy Review        | Quarterly       | Engineering Leadership |
| Compliance Audit     | Annually        | Security & Compliance  |
| DR Test Review       | After each test | All stakeholders       |

### Change Management

1. All RTO/RPO changes require Engineering Director approval
2. Changes must be documented in ADR format
3. Updated policies require DR test validation
4. Compliance team notification for regulatory services

## Related Documents

- [DR Overview](./DR-OVERVIEW.md)
- [Recovery Runbook](./RECOVERY-RUNBOOK.md)
- [Testing Procedures](./DR-TESTING.md)
