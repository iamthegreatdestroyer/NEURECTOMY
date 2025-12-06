# Disaster Recovery Documentation

## NEURECTOMY Platform - Disaster Recovery Guide

This documentation provides comprehensive disaster recovery procedures for the NEURECTOMY platform, covering multi-cloud failover, data recovery, and business continuity.

## Table of Contents

1. [DR Overview](./DR-OVERVIEW.md)
2. [RTO/RPO Policies](./RTO-RPO-POLICIES.md)
3. [Backup Procedures](./BACKUP-PROCEDURES.md)
4. [Recovery Runbook](./RECOVERY-RUNBOOK.md)
5. [Multi-Region Failover](./MULTI-REGION-FAILOVER.md)
6. [Testing Procedures](./DR-TESTING.md)

## Quick Reference

### Critical Contacts

| Role              | Contact                     | Escalation Time |
| ----------------- | --------------------------- | --------------- |
| Primary On-Call   | PagerDuty Rotation          | Immediate       |
| Platform Lead     | platform-lead@neurectomy.io | 15 min          |
| Security Team     | security@neurectomy.io      | 30 min          |
| Executive Sponsor | exec-sponsor@neurectomy.io  | 1 hour          |

### Recovery Objectives

| Tier     | RTO      | RPO      | Services                 |
| -------- | -------- | -------- | ------------------------ |
| Critical | 15 min   | 5 min    | ML Service, API Gateway  |
| High     | 1 hour   | 15 min   | MLflow, PostgreSQL       |
| Medium   | 4 hours  | 1 hour   | Grafana, Prometheus      |
| Low      | 24 hours | 24 hours | Development environments |

### Quick Commands

```bash
# Check backup status
velero backup get

# Initiate manual backup
velero backup create emergency-backup --include-namespaces neurectomy

# Check cluster health
kubectl get nodes -o wide
kubectl top nodes

# Failover to DR region
./scripts/dr-failover.sh --target=us-west-2

# Restore from backup
velero restore create --from-backup <backup-name>
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        NEURECTOMY DR Architecture                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    PRIMARY REGION (us-east-1)              DR REGION (us-west-2)            │
│    ─────────────────────────               ────────────────────             │
│                                                                              │
│    ┌─────────────────────┐                ┌─────────────────────┐           │
│    │   EKS Cluster       │                │   EKS Cluster       │           │
│    │   (Active)          │   Replicate    │   (Standby)         │           │
│    │                     │ ───────────►   │                     │           │
│    │ ┌─────────────────┐ │                │ ┌─────────────────┐ │           │
│    │ │ ML Service      │ │                │ │ ML Service      │ │           │
│    │ │ MLflow          │ │                │ │ MLflow          │ │           │
│    │ │ PostgreSQL      │ │                │ │ PostgreSQL      │ │           │
│    │ └─────────────────┘ │                │ └─────────────────┘ │           │
│    └─────────────────────┘                └─────────────────────┘           │
│              │                                      │                        │
│              ▼                                      ▼                        │
│    ┌─────────────────────┐                ┌─────────────────────┐           │
│    │ S3 Primary Bucket   │   Cross-Region │ S3 DR Bucket        │           │
│    │ (Backups, Models)   │ ──Replication─►│ (Backups, Models)   │           │
│    └─────────────────────┘                └─────────────────────┘           │
│                                                                              │
│                           ┌─────────────────┐                               │
│                           │  Route53        │                               │
│                           │  Health Checks  │                               │
│                           │  DNS Failover   │                               │
│                           └─────────────────┘                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Related Documentation

- [Velero Configuration](../../k8s/velero/)
- [Multi-Region Terraform](../../terraform/)
- [Network Policies](../../k8s/base/security/)
- [Monitoring & Alerting](../observability/)
