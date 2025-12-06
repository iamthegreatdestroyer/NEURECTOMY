# Disaster Recovery Overview

## Introduction

This document outlines the disaster recovery strategy for the NEURECTOMY platform, designed to ensure business continuity and minimize data loss during any service disruption.

## DR Strategy

### Active-Passive Multi-Region

NEURECTOMY employs an **active-passive** disaster recovery strategy:

- **Primary Region**: us-east-1 (N. Virginia)
- **DR Region**: us-west-2 (Oregon)
- **Tertiary**: Azure West US 2 (multi-cloud failover)

```
                    ┌────────────────────────────┐
                    │       Global DNS           │
                    │      (Route53/CloudFlare)  │
                    └─────────────┬──────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
              ▼                   ▼                   ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │   us-east-1     │ │   us-west-2     │ │   Azure West    │
    │   (PRIMARY)     │ │   (DR)          │ │   (TERTIARY)    │
    │   Active        │ │   Standby       │ │   Cold Standby  │
    └─────────────────┘ └─────────────────┘ └─────────────────┘
```

## Failure Scenarios

### Scenario Matrix

| Scenario              | Severity | Automatic Failover | Manual Intervention | Recovery Time |
| --------------------- | -------- | ------------------ | ------------------- | ------------- |
| Pod failure           | Low      | Yes (Kubernetes)   | No                  | < 1 min       |
| Node failure          | Low      | Yes (Karpenter)    | No                  | < 5 min       |
| AZ failure            | Medium   | Yes (Multi-AZ)     | No                  | < 5 min       |
| Region failure        | High     | Partial            | Yes                 | 15-30 min     |
| Cloud provider outage | Critical | No                 | Yes                 | 1-4 hours     |
| Data corruption       | Critical | No                 | Yes                 | 1-4 hours     |
| Security breach       | Critical | No                 | Yes                 | Variable      |

### Failure Categories

#### Category 1: Infrastructure Failures

1. **Single Pod Failure**
   - Kubernetes automatically reschedules
   - PodDisruptionBudgets ensure minimum availability
   - No manual intervention required

2. **Node Failure**
   - Karpenter provisions replacement nodes
   - Pods automatically rescheduled
   - EBS volumes reattached automatically

3. **Availability Zone Failure**
   - Multi-AZ deployment ensures continuity
   - Automatic pod redistribution
   - Database failover to standby

4. **Region Failure**
   - Route53 health checks detect failure
   - DNS failover to DR region
   - Velero restores from latest backup

#### Category 2: Data Failures

1. **Database Corruption**
   - Point-in-time recovery from RDS
   - Velero snapshots for Kubernetes resources
   - MLflow artifacts restored from S3

2. **Accidental Deletion**
   - S3 versioning enables recovery
   - Velero backups with retention
   - Git history for configuration

3. **Ransomware/Encryption Attack**
   - Air-gapped backup copies
   - Cross-region replication
   - Immutable backup storage

#### Category 3: Application Failures

1. **Bad Deployment**
   - ArgoCD automatic rollback
   - Canary deployment catches issues
   - Instant rollback capability

2. **Configuration Error**
   - GitOps ensures configuration is versioned
   - External Secrets Operator for secrets recovery
   - Terraform state for infrastructure

## DR Components

### Velero Backup System

```yaml
Backup Schedule:
  - Hourly: Last 24 hours retained
  - Daily: Last 30 days retained
  - Weekly: Last 12 weeks retained
  - Monthly: Last 12 months retained

Backup Scope:
  - Kubernetes resources (all namespaces)
  - Persistent volume snapshots
  - Application configurations
  - Secrets (encrypted)
```

### Data Replication

```yaml
S3 Cross-Region Replication:
  Source: neurectomy-backups-us-east-1
  Destination: neurectomy-backups-us-west-2
  Replication: Synchronous (RTC)
  Encryption: SSE-KMS

Database Replication:
  Type: PostgreSQL streaming replication
  Mode: Synchronous to primary, async to DR
  Lag Threshold: < 1 second
```

### Health Monitoring

```yaml
Route53 Health Checks:
  - Endpoint: https://api.neurectomy.io/health
  - Interval: 10 seconds
  - Threshold: 3 failures
  - Regions: us-east-1, eu-west-1, ap-southeast-1

CloudWatch Alarms:
  - ClusterHealth
  - NodeReadiness
  - PodAvailability
  - DatabaseConnectivity
  - BackupSuccess
```

## Recovery Priorities

### Tier 1 - Critical (RTO: 15 min)

| Service        | Dependency | Recovery Order |
| -------------- | ---------- | -------------- |
| API Gateway    | None       | 1              |
| ML Service     | PostgreSQL | 2              |
| Authentication | PostgreSQL | 3              |

### Tier 2 - High (RTO: 1 hour)

| Service    | Dependency     | Recovery Order |
| ---------- | -------------- | -------------- |
| PostgreSQL | EBS Volumes    | 4              |
| MLflow     | PostgreSQL, S3 | 5              |
| Redis      | EBS Volumes    | 6              |

### Tier 3 - Medium (RTO: 4 hours)

| Service    | Dependency | Recovery Order |
| ---------- | ---------- | -------------- |
| Prometheus | PVC        | 7              |
| Grafana    | PostgreSQL | 8              |
| Jaeger     | Storage    | 9              |

### Tier 4 - Low (RTO: 24 hours)

| Service                  | Dependency | Recovery Order |
| ------------------------ | ---------- | -------------- |
| Development Environments | None       | 10             |
| CI/CD Runners            | None       | 11             |

## Communication Plan

### Notification Matrix

| Event             | Internal              | External              | Timeline        |
| ----------------- | --------------------- | --------------------- | --------------- |
| Incident Detected | PagerDuty Alert       | -                     | Immediate       |
| DR Activated      | Slack #incidents      | Status Page           | < 5 min         |
| Recovery Started  | Email All Hands       | Customer Notification | < 15 min        |
| Recovery Progress | Slack Updates         | Status Page Updates   | Every 30 min    |
| Recovery Complete | Post-Mortem Scheduled | All Clear Notice      | Upon completion |

### Status Page Updates

```
Status Levels:
  - Operational: All systems functioning normally
  - Degraded: Partial functionality, workarounds available
  - Partial Outage: Major functionality impacted
  - Major Outage: Critical services unavailable
  - Maintenance: Planned downtime
```

## Related Documents

- [RTO/RPO Policies](./RTO-RPO-POLICIES.md)
- [Recovery Runbook](./RECOVERY-RUNBOOK.md)
- [Testing Procedures](./DR-TESTING.md)
