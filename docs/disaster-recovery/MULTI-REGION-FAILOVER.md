# Multi-Region Failover Guide

## Overview

This document provides detailed procedures for failing over NEURECTOMY services between AWS regions and to multi-cloud environments.

## Failover Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     NEURECTOMY Multi-Region Failover                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                        ┌─────────────────────┐                              │
│                        │     CloudFlare      │                              │
│                        │   Global Load       │                              │
│                        │   Balancer          │                              │
│                        └──────────┬──────────┘                              │
│                                   │                                          │
│                    ┌──────────────┼──────────────┐                          │
│                    │              │              │                          │
│                    ▼              ▼              ▼                          │
│         ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│         │   AWS        │  │   AWS        │  │   Azure      │               │
│         │  us-east-1   │  │  us-west-2   │  │  westus2     │               │
│         │  (PRIMARY)   │  │  (DR)        │  │  (TERTIARY)  │               │
│         │              │  │              │  │              │               │
│         │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │               │
│         │ │   EKS    │ │  │ │   EKS    │ │  │ │   AKS    │ │               │
│         │ │ Cluster  │ │  │ │ Cluster  │ │  │ │ Cluster  │ │               │
│         │ └────┬─────┘ │  │ └────┬─────┘ │  │ └────┬─────┘ │               │
│         │      │       │  │      │       │  │      │       │               │
│         │ ┌────▼─────┐ │  │ ┌────▼─────┐ │  │ ┌────▼─────┐ │               │
│         │ │   RDS    │ │  │ │   RDS    │ │  │ │  Azure   │ │               │
│         │ │ Primary  │─┼──┼─│ Replica  │ │  │ │ Database │ │               │
│         │ └──────────┘ │  │ └──────────┘ │  │ └──────────┘ │               │
│         │              │  │              │  │              │               │
│         │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │               │
│         │ │    S3    │─┼──┼─│    S3    │ │  │ │  Blob    │ │               │
│         │ │ Primary  │ │  │ │ Replica  │ │  │ │ Storage  │ │               │
│         │ └──────────┘ │  │ └──────────┘ │  │ └──────────┘ │               │
│         └──────────────┘  └──────────────┘  └──────────────┘               │
│                                                                              │
│         Priority: 1            Priority: 2       Priority: 3                │
│         Status: Active         Status: Standby   Status: Cold               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Failover Types

### Type 1: Automatic Failover

**Trigger:** Route53 health checks detect endpoint failure

**Scope:** Regional endpoint failure, single service failure

**Action:** Automatic DNS failover to healthy region

```yaml
Automatic Failover Configuration:
  Health Check:
    Endpoint: https://api.neurectomy.io/health
    Protocol: HTTPS
    Port: 443
    Path: /health
    Request Interval: 10 seconds
    Failure Threshold: 3

  DNS Failover:
    Primary: us-east-1 ALB (Priority 1)
    Secondary: us-west-2 ALB (Priority 2)
    TTL: 60 seconds
```

### Type 2: Manual Failover

**Trigger:** Operator decision based on incident assessment

**Scope:** Complete region failure, data corruption, planned maintenance

**Action:** Execute failover runbook manually

## Pre-Failover Checklist

### Before Initiating Failover

- [ ] Incident documented and severity confirmed
- [ ] Impact assessment completed
- [ ] Stakeholders notified
- [ ] DR region health verified
- [ ] Backup status confirmed
- [ ] Communication channels active

### DR Region Verification

```bash
#!/bin/bash
# scripts/verify-dr-region.sh

echo "=== Verifying DR Region Health ==="

# Switch to DR context
export KUBECONFIG=~/.kube/neurectomy-dr.config

# Check cluster connectivity
echo "Checking cluster connectivity..."
kubectl cluster-info

# Check node status
echo "Checking node status..."
kubectl get nodes

# Check critical pods
echo "Checking pod status..."
kubectl get pods -n neurectomy

# Check storage
echo "Checking storage classes..."
kubectl get sc

# Check Velero
echo "Checking Velero status..."
velero backup get | head -5

# Check external connectivity
echo "Checking external connectivity..."
kubectl exec -n neurectomy deploy/ml-service -- curl -s https://api.openai.com/v1/models > /dev/null && echo "External API: OK" || echo "External API: FAILED"

echo "=== DR Region Verification Complete ==="
```

## Failover Procedures

### AWS Region Failover (us-east-1 → us-west-2)

#### Step 1: Assess Situation

```bash
# Check primary region status
aws ec2 describe-availability-zones --region us-east-1

# Check Route53 health check status
aws route53 get-health-check-status --health-check-id <health-check-id>

# Check CloudWatch alarms
aws cloudwatch describe-alarms --alarm-names "neurectomy-cluster-health" --region us-east-1
```

#### Step 2: Activate DR Region

```bash
#!/bin/bash
# scripts/activate-dr-region.sh

set -e

echo "=== Activating DR Region (us-west-2) ==="

# Set DR context
export KUBECONFIG=~/.kube/neurectomy-dr-us-west-2.config
export AWS_DEFAULT_REGION=us-west-2

# Verify DR cluster is healthy
echo "Step 1: Verifying DR cluster health..."
kubectl get nodes
kubectl get pods -n neurectomy --field-selector status.phase!=Running

# Scale up DR workloads (if running reduced capacity)
echo "Step 2: Scaling up DR workloads..."
kubectl scale deployment ml-service --replicas=5 -n neurectomy
kubectl scale deployment mlflow --replicas=3 -n neurectomy

# Promote RDS read replica to primary
echo "Step 3: Promoting RDS replica..."
aws rds promote-read-replica \
  --db-instance-identifier neurectomy-dr-postgres \
  --backup-retention-period 7

# Wait for RDS promotion
echo "Waiting for RDS promotion (this may take several minutes)..."
aws rds wait db-instance-available --db-instance-identifier neurectomy-dr-postgres

# Update database connection strings
echo "Step 4: Updating database configuration..."
kubectl set env deployment/ml-service \
  DATABASE_HOST=neurectomy-dr-postgres.xxxx.us-west-2.rds.amazonaws.com \
  -n neurectomy
kubectl set env deployment/mlflow \
  DATABASE_HOST=neurectomy-dr-postgres.xxxx.us-west-2.rds.amazonaws.com \
  -n neurectomy

# Restart pods to pick up new configuration
echo "Step 5: Rolling restart of deployments..."
kubectl rollout restart deployment -n neurectomy

# Wait for rollout
kubectl rollout status deployment/ml-service -n neurectomy --timeout=300s
kubectl rollout status deployment/mlflow -n neurectomy --timeout=300s

echo "=== DR Region Activated ==="
```

#### Step 3: DNS Failover

```bash
#!/bin/bash
# scripts/dns-failover.sh

set -e

HOSTED_ZONE_ID="Z1234567890ABC"
DR_ALB_DNS="neurectomy-dr-alb-123456.us-west-2.elb.amazonaws.com"
DR_ALB_ZONE_ID="Z1H1FL5HABSF5"

echo "=== Executing DNS Failover ==="

# Failover primary record
aws route53 change-resource-record-sets \
  --hosted-zone-id $HOSTED_ZONE_ID \
  --change-batch '{
    "Changes": [
      {
        "Action": "UPSERT",
        "ResourceRecordSet": {
          "Name": "api.neurectomy.io",
          "Type": "A",
          "SetIdentifier": "primary",
          "Failover": "PRIMARY",
          "AliasTarget": {
            "HostedZoneId": "'$DR_ALB_ZONE_ID'",
            "DNSName": "'$DR_ALB_DNS'",
            "EvaluateTargetHealth": true
          }
        }
      }
    ]
  }'

# Verify DNS propagation
echo "Waiting for DNS propagation..."
sleep 30

# Test new endpoint
echo "Testing failover endpoint..."
curl -s https://api.neurectomy.io/health | jq .

echo "=== DNS Failover Complete ==="
```

#### Step 4: Validate Failover

```bash
#!/bin/bash
# scripts/validate-failover.sh

echo "=== Validating Failover ==="

# DNS resolution check
echo "1. DNS Resolution:"
dig +short api.neurectomy.io

# Health endpoint check
echo "2. Health Check:"
curl -s https://api.neurectomy.io/health | jq .

# API functionality check
echo "3. API Functionality:"
curl -s https://api.neurectomy.io/v1/models | jq '.count'

# Database connectivity
echo "4. Database Connectivity:"
kubectl exec -n neurectomy deploy/ml-service -- \
  python -c "from app.db import engine; print(engine.execute('SELECT 1').scalar())"

# Model inference check
echo "5. Model Inference:"
curl -s -X POST https://api.neurectomy.io/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"input": "test"}' | jq '.status'

# Metrics collection
echo "6. Metrics Collection:"
kubectl exec -n monitoring deploy/prometheus -- \
  curl -s localhost:9090/api/v1/query?query=up | jq '.data.result | length'

echo "=== Failover Validation Complete ==="
```

### Multi-Cloud Failover (AWS → Azure)

#### Prerequisites

```bash
# Ensure Azure credentials are configured
az login
az account set --subscription <subscription-id>

# Get AKS credentials
az aks get-credentials --resource-group neurectomy-rg --name neurectomy-aks

# Verify Azure cluster
kubectl config use-context neurectomy-aks
kubectl get nodes
```

#### Azure Activation Procedure

```bash
#!/bin/bash
# scripts/activate-azure.sh

set -e

echo "=== Activating Azure Environment ==="

# Switch to Azure context
kubectl config use-context neurectomy-aks

# Step 1: Restore from latest backup
echo "Step 1: Restoring from latest backup..."

# Get latest backup from Azure blob
LATEST_BACKUP=$(velero backup get -o json | jq -r '.items | sort_by(.metadata.creationTimestamp) | last | .metadata.name')
echo "Latest backup: ${LATEST_BACKUP}"

velero restore create azure-activation-$(date +%Y%m%d-%H%M%S) \
  --from-backup "${LATEST_BACKUP}" \
  --exclude-namespaces kube-system,velero \
  --wait

# Step 2: Update storage classes
echo "Step 2: Updating storage classes..."
kubectl apply -f k8s/base/storage/azure-storage-classes.yaml

# Step 3: Update secrets to use Azure Key Vault
echo "Step 3: Configuring Azure Key Vault secrets..."
kubectl apply -f k8s/external-secrets/azure-secret-store.yaml

# Refresh all external secrets
kubectl delete externalsecret --all -n neurectomy
kubectl apply -f k8s/external-secrets/

# Wait for secrets to sync
sleep 30
kubectl get externalsecret -n neurectomy

# Step 4: Scale up workloads
echo "Step 4: Scaling workloads..."
kubectl scale deployment ml-service --replicas=3 -n neurectomy
kubectl scale deployment mlflow --replicas=2 -n neurectomy

# Wait for rollout
kubectl rollout status deployment/ml-service -n neurectomy --timeout=600s
kubectl rollout status deployment/mlflow -n neurectomy --timeout=600s

# Step 5: Update Azure Traffic Manager
echo "Step 5: Updating Traffic Manager..."
az network traffic-manager endpoint update \
  --resource-group neurectomy-rg \
  --profile-name neurectomy-tm \
  --name azure-endpoint \
  --type azureEndpoints \
  --endpoint-status Enabled \
  --priority 1

az network traffic-manager endpoint update \
  --resource-group neurectomy-rg \
  --profile-name neurectomy-tm \
  --name aws-endpoint \
  --type externalEndpoints \
  --endpoint-status Disabled

echo "=== Azure Environment Activated ==="
```

## Failback Procedures

### AWS Failback (us-west-2 → us-east-1)

```bash
#!/bin/bash
# scripts/failback-to-primary.sh

set -e

echo "=== Initiating Failback to Primary Region ==="

# Step 1: Verify primary region recovered
echo "Step 1: Verifying primary region..."
export KUBECONFIG=~/.kube/neurectomy-primary-us-east-1.config
kubectl get nodes
kubectl get pods -n neurectomy

# Step 2: Sync data from DR to Primary
echo "Step 2: Syncing data..."

# Create snapshot of DR database
aws rds create-db-snapshot \
  --db-instance-identifier neurectomy-dr-postgres \
  --db-snapshot-identifier neurectomy-failback-$(date +%Y%m%d) \
  --region us-west-2

# Wait for snapshot
aws rds wait db-snapshot-available \
  --db-snapshot-identifier neurectomy-failback-$(date +%Y%m%d) \
  --region us-west-2

# Copy snapshot to primary region
aws rds copy-db-snapshot \
  --source-db-snapshot-identifier arn:aws:rds:us-west-2:123456789:snapshot:neurectomy-failback-$(date +%Y%m%d) \
  --target-db-snapshot-identifier neurectomy-failback-$(date +%Y%m%d) \
  --region us-east-1

# Step 3: Restore primary database
echo "Step 3: Restoring primary database..."
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier neurectomy-postgres-new \
  --db-snapshot-identifier neurectomy-failback-$(date +%Y%m%d) \
  --region us-east-1

# Wait for restore
aws rds wait db-instance-available \
  --db-instance-identifier neurectomy-postgres-new \
  --region us-east-1

# Step 4: Update primary cluster configuration
echo "Step 4: Updating primary cluster..."
kubectl set env deployment/ml-service \
  DATABASE_HOST=neurectomy-postgres-new.xxxx.us-east-1.rds.amazonaws.com \
  -n neurectomy
kubectl set env deployment/mlflow \
  DATABASE_HOST=neurectomy-postgres-new.xxxx.us-east-1.rds.amazonaws.com \
  -n neurectomy

# Restart deployments
kubectl rollout restart deployment -n neurectomy
kubectl rollout status deployment/ml-service -n neurectomy --timeout=300s

# Step 5: DNS failback
echo "Step 5: Executing DNS failback..."
./scripts/dns-failback.sh

# Step 6: Scale down DR region
echo "Step 6: Scaling down DR region..."
export KUBECONFIG=~/.kube/neurectomy-dr-us-west-2.config
kubectl scale deployment ml-service --replicas=1 -n neurectomy
kubectl scale deployment mlflow --replicas=1 -n neurectomy

echo "=== Failback Complete ==="
```

## Communication Templates

### Failover Initiation

```markdown
Subject: [INCIDENT] NEURECTOMY - Initiating Regional Failover

Team,

We are initiating a regional failover for NEURECTOMY from us-east-1 to us-west-2.

**Incident Details:**

- Incident ID: INC-XXXX
- Start Time: YYYY-MM-DD HH:MM UTC
- Affected Services: [List]
- Estimated Recovery: XX minutes

**Current Status:**

- Primary region (us-east-1): Degraded/Unavailable
- DR region (us-west-2): Healthy, activating

**Actions Being Taken:**

1. Activating DR workloads
2. Promoting DR database
3. Executing DNS failover

**Customer Impact:**

- Brief service interruption expected (< 15 minutes)
- All data preserved per RTO/RPO policies

Next update in 15 minutes.

Incident Commander: [Name]
```

### Failover Complete

```markdown
Subject: [RESOLVED] NEURECTOMY - Regional Failover Complete

Team,

The regional failover has been completed successfully.

**Resolution Details:**

- Incident ID: INC-XXXX
- Resolution Time: YYYY-MM-DD HH:MM UTC
- Total Downtime: XX minutes
- Data Loss: None (within RPO)

**Current Status:**

- Active Region: us-west-2
- All services: Operational
- Monitoring: Active

**Post-Incident:**

- Post-mortem scheduled for [Date]
- Failback planning in progress

Status page has been updated.

Incident Commander: [Name]
```

## Related Documents

- [DR Overview](./DR-OVERVIEW.md)
- [Recovery Runbook](./RECOVERY-RUNBOOK.md)
- [DR Testing](./DR-TESTING.md)
