# Recovery Runbook

## Overview

This runbook provides step-by-step procedures for recovering NEURECTOMY services during a disaster event.

## Pre-Recovery Checklist

### Before Starting Recovery

- [ ] Incident declared and documented
- [ ] On-call engineer confirmed
- [ ] Communication channels established (Slack #incident-response)
- [ ] Status page updated
- [ ] Customer communications prepared
- [ ] Recovery target confirmed (latest backup vs point-in-time)

### Access Verification

```bash
# Verify AWS CLI access
aws sts get-caller-identity

# Verify kubectl access to DR cluster
kubectl --context neurectomy-dr-us-west-2 get nodes

# Verify Velero access
velero backup get --kubeconfig ~/.kube/dr-config

# Verify Terraform access
cd terraform && terraform init
```

---

## Scenario 1: Single Service Failure

### Symptoms

- Single service returning 5xx errors
- Pod crashlooping
- Service endpoints unhealthy

### Recovery Steps

```bash
# 1. Identify the failing service
kubectl get pods -n neurectomy -l app=<service-name>
kubectl describe pod <pod-name> -n neurectomy
kubectl logs <pod-name> -n neurectomy --previous

# 2. Check recent deployments
kubectl rollout history deployment/<deployment-name> -n neurectomy

# 3. If recent deployment caused issue - rollback
kubectl rollout undo deployment/<deployment-name> -n neurectomy

# 4. If configuration issue - restore from GitOps
# ArgoCD will automatically sync, or force sync:
argocd app sync neurectomy-<service> --force

# 5. If persistent volume issue - restore from snapshot
velero restore create <restore-name> \
  --from-backup <backup-name> \
  --include-resources persistentvolumeclaims \
  --selector app=<service-name>

# 6. Verify recovery
kubectl get pods -n neurectomy -l app=<service-name>
kubectl logs -f deployment/<deployment-name> -n neurectomy
```

### Estimated Time: 5-15 minutes

---

## Scenario 2: Database Failure (PostgreSQL)

### Symptoms

- Database connection errors
- Data integrity issues
- PostgreSQL pod unresponsive

### Recovery Steps

#### Option A: Pod Recovery

```bash
# 1. Check PostgreSQL status
kubectl get pods -n neurectomy -l app=postgresql
kubectl describe pod postgresql-0 -n neurectomy

# 2. If pod is stuck, delete and let StatefulSet recreate
kubectl delete pod postgresql-0 -n neurectomy

# 3. Wait for pod to be ready
kubectl wait --for=condition=ready pod/postgresql-0 -n neurectomy --timeout=300s

# 4. Verify database connectivity
kubectl exec -it postgresql-0 -n neurectomy -- psql -U postgres -c "SELECT 1;"
```

#### Option B: Point-in-Time Recovery

```bash
# 1. Get available backups
velero backup get | grep postgresql

# 2. Find the backup closest to desired recovery point
velero backup describe <backup-name> --details

# 3. Scale down dependent services
kubectl scale deployment ml-service --replicas=0 -n neurectomy
kubectl scale deployment mlflow --replicas=0 -n neurectomy

# 4. Delete current PostgreSQL StatefulSet (data will be restored)
kubectl delete statefulset postgresql -n neurectomy
kubectl delete pvc data-postgresql-0 -n neurectomy

# 5. Restore from backup
velero restore create postgres-restore-$(date +%Y%m%d-%H%M%S) \
  --from-backup <backup-name> \
  --include-resources statefulsets,persistentvolumeclaims,secrets,configmaps \
  --selector app=postgresql \
  --namespace-mappings neurectomy:neurectomy

# 6. Wait for restore completion
velero restore describe postgres-restore-* --details

# 7. Verify PostgreSQL is running
kubectl wait --for=condition=ready pod/postgresql-0 -n neurectomy --timeout=600s

# 8. Verify data integrity
kubectl exec -it postgresql-0 -n neurectomy -- psql -U postgres -c "SELECT count(*) FROM ml_models;"

# 9. Scale up dependent services
kubectl scale deployment ml-service --replicas=3 -n neurectomy
kubectl scale deployment mlflow --replicas=2 -n neurectomy

# 10. Verify application connectivity
kubectl logs -f deployment/ml-service -n neurectomy | head -50
```

### Estimated Time: 15-60 minutes

---

## Scenario 3: Node Failure

### Symptoms

- Node showing NotReady status
- Multiple pods in Pending/Unknown state
- Resource pressure warnings

### Recovery Steps

```bash
# 1. Identify affected nodes
kubectl get nodes
kubectl describe node <node-name>

# 2. Check node conditions
kubectl get nodes -o custom-columns=NAME:.metadata.name,STATUS:.status.conditions[-1].type,REASON:.status.conditions[-1].reason

# 3. Cordon the problematic node (prevent new scheduling)
kubectl cordon <node-name>

# 4. Drain the node (evacuate pods)
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data --force

# 5. If using Karpenter, it will automatically provision replacement
# Check Karpenter logs:
kubectl logs -n karpenter -l app.kubernetes.io/name=karpenter -f

# 6. If manual intervention needed, terminate the instance
# AWS:
aws ec2 terminate-instances --instance-ids <instance-id>

# 7. Verify new node joined
kubectl get nodes -w

# 8. Verify pods rescheduled
kubectl get pods -n neurectomy -o wide

# 9. If node recovered, uncordon it
kubectl uncordon <node-name>
```

### Estimated Time: 5-15 minutes (Karpenter automatic)

---

## Scenario 4: Availability Zone Failure

### Symptoms

- Multiple nodes in single AZ unavailable
- Regional degradation alerts
- Increased latency from affected AZ

### Recovery Steps

```bash
# 1. Identify affected AZ
kubectl get nodes -o custom-columns=NAME:.metadata.name,ZONE:.metadata.labels.topology\\.kubernetes\\.io/zone

# 2. Check pod distribution
kubectl get pods -n neurectomy -o wide

# 3. Cordon all nodes in affected AZ
kubectl get nodes -l topology.kubernetes.io/zone=<affected-az> -o name | xargs -I {} kubectl cordon {}

# 4. Trigger pod rescheduling
kubectl get pods -n neurectomy -o name --field-selector spec.nodeName=<affected-node> | xargs -I {} kubectl delete {}

# 5. Karpenter will provision nodes in healthy AZs
# Monitor Karpenter:
kubectl logs -n karpenter -l app.kubernetes.io/name=karpenter -f

# 6. Verify pods redistributed to healthy AZs
kubectl get pods -n neurectomy -o wide

# 7. Monitor service health
kubectl get endpoints -n neurectomy

# 8. When AZ recovers, uncordon nodes
kubectl get nodes -l topology.kubernetes.io/zone=<affected-az> -o name | xargs -I {} kubectl uncordon {}
```

### Estimated Time: 5-15 minutes (automatic with multi-AZ)

---

## Scenario 5: Complete Region Failure

### Symptoms

- All cluster endpoints unreachable
- Route53 health checks failing
- AWS status page shows regional issue

### Recovery Steps

```bash
# === PHASE 1: ASSESS AND PREPARE ===

# 1. Confirm regional failure (check AWS status)
open https://status.aws.amazon.com

# 2. Switch to DR cluster context
export KUBECONFIG=~/.kube/neurectomy-dr-us-west-2.config
kubectl cluster-info

# 3. Verify DR cluster health
kubectl get nodes
kubectl get pods -n neurectomy

# === PHASE 2: ACTIVATE DR REGION ===

# 4. Get latest backup
velero backup get --kubeconfig $KUBECONFIG | head -5

# 5. If DR is already synced via continuous replication, skip restore
# Otherwise, restore from latest backup:
velero restore create region-failover-$(date +%Y%m%d-%H%M%S) \
  --from-backup <latest-backup> \
  --exclude-namespaces kube-system,velero

# 6. Wait for restore completion
velero restore describe region-failover-* --details

# 7. Verify all pods running
kubectl get pods -n neurectomy
kubectl wait --for=condition=ready pods --all -n neurectomy --timeout=600s

# === PHASE 3: DNS FAILOVER ===

# 8. If automatic Route53 failover didn't trigger, manual failover:
aws route53 change-resource-record-sets \
  --hosted-zone-id <zone-id> \
  --change-batch '{
    "Changes": [{
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "api.neurectomy.io",
        "Type": "A",
        "AliasTarget": {
          "HostedZoneId": "<dr-alb-zone-id>",
          "DNSName": "<dr-alb-dns>",
          "EvaluateTargetHealth": true
        }
      }
    }]
  }'

# 9. Verify DNS propagation
dig api.neurectomy.io
nslookup api.neurectomy.io

# === PHASE 4: VERIFY AND COMMUNICATE ===

# 10. Run smoke tests
curl https://api.neurectomy.io/health
curl https://api.neurectomy.io/v1/models | jq .

# 11. Update status page
# Mark as "Degraded Performance" or "Operational" based on validation

# 12. Notify stakeholders
# Send all-hands notification about failover completion
```

### Estimated Time: 15-30 minutes

---

## Scenario 6: Data Corruption / Ransomware

### Symptoms

- Data integrity check failures
- Unauthorized encryption detected
- Anomalous data modification patterns

### Recovery Steps

```bash
# === PHASE 1: CONTAIN ===

# 1. IMMEDIATELY isolate the cluster
kubectl cordon --all nodes
kubectl scale deployment --all --replicas=0 -n neurectomy

# 2. Block all ingress
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: emergency-isolation
  namespace: neurectomy
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
EOF

# 3. Revoke all external access
# Disable API keys, rotate secrets

# === PHASE 2: ASSESS ===

# 4. Identify scope of corruption
# Check backup integrity
velero backup describe <backup-name> --details

# 5. Find last known good backup
# Look for backups before corruption timestamp
velero backup get --output json | jq '.items[] | {name: .metadata.name, timestamp: .metadata.creationTimestamp}'

# 6. Test backup integrity
velero restore create test-restore-$(date +%Y%m%d-%H%M%S) \
  --from-backup <pre-corruption-backup> \
  --namespace-mappings neurectomy:neurectomy-test \
  --include-resources persistentvolumeclaims

# === PHASE 3: RECOVER ===

# 7. Create fresh namespace
kubectl create namespace neurectomy-recovered

# 8. Restore from last known good backup
velero restore create full-recovery-$(date +%Y%m%d-%H%M%S) \
  --from-backup <pre-corruption-backup> \
  --namespace-mappings neurectomy:neurectomy-recovered

# 9. Verify data integrity in recovered namespace
kubectl exec -it postgresql-0 -n neurectomy-recovered -- \
  psql -U postgres -c "SELECT count(*), max(updated_at) FROM ml_models;"

# === PHASE 4: SWITCHOVER ===

# 10. Update DNS/Ingress to point to recovered namespace
kubectl patch ingress neurectomy-ingress -n neurectomy-recovered \
  --type='json' -p='[{"op": "replace", "path": "/spec/rules/0/host", "value": "api.neurectomy.io"}]'

# 11. Delete corrupted namespace (after verification)
kubectl delete namespace neurectomy

# 12. Rename recovered namespace (optional)
# Note: Kubernetes doesn't support namespace rename, use ArgoCD to redeploy

# === PHASE 5: POST-INCIDENT ===

# 13. Conduct security review
# 14. Update security policies
# 15. Document incident
# 16. Schedule post-mortem
```

### Estimated Time: 1-4 hours (depending on corruption scope)

---

## Scenario 7: Multi-Cloud Failover (AWS to Azure)

### Prerequisites

- Azure AKS cluster provisioned
- Cross-cloud backup replication configured
- DNS configured for multi-cloud

### Recovery Steps

```bash
# === PHASE 1: PREPARE AZURE ENVIRONMENT ===

# 1. Switch to Azure context
az account set --subscription <subscription-id>
az aks get-credentials --resource-group neurectomy-rg --name neurectomy-aks

# 2. Verify Azure cluster health
kubectl get nodes
kubectl cluster-info

# 3. Install Velero in Azure cluster (if not already)
velero install \
  --provider azure \
  --plugins velero/velero-plugin-for-microsoft-azure:v1.8.0 \
  --bucket neurectomy-backups \
  --secret-file ./azure-credentials \
  --backup-location-config resourceGroup=neurectomy-rg,storageAccount=neurectomybackups

# === PHASE 2: RESTORE FROM CROSS-REGION BACKUP ===

# 4. Sync backup location
velero backup-location get

# 5. Get available backups
velero backup get

# 6. Restore workloads
velero restore create azure-failover-$(date +%Y%m%d-%H%M%S) \
  --from-backup <latest-backup> \
  --exclude-namespaces kube-system,velero,karpenter

# 7. Wait for restore
velero restore describe azure-failover-* --details

# === PHASE 3: AZURE-SPECIFIC CONFIGURATION ===

# 8. Update storage class references
kubectl patch pvc data-postgresql-0 -n neurectomy \
  --type='json' -p='[{"op": "replace", "path": "/spec/storageClassName", "value": "azure-disk-premium"}]'

# 9. Update external secrets to use Azure Key Vault
kubectl apply -f k8s/external-secrets/azure-secret-store.yaml

# 10. Refresh secrets
kubectl delete externalsecret --all -n neurectomy
kubectl apply -f k8s/external-secrets/ml-service-secrets.yaml

# === PHASE 4: DNS FAILOVER ===

# 11. Update DNS to Azure endpoints
# Using Cloudflare or Route53:
aws route53 change-resource-record-sets \
  --hosted-zone-id <zone-id> \
  --change-batch file://azure-dns-failover.json

# 12. Verify DNS propagation
dig api.neurectomy.io
curl https://api.neurectomy.io/health

# === PHASE 5: VALIDATE ===

# 13. Run integration tests
./scripts/integration-tests.sh --environment azure

# 14. Monitor for 30 minutes
kubectl logs -f -n neurectomy -l app=ml-service --since=5m
```

### Estimated Time: 1-4 hours

---

## Post-Recovery Checklist

### Immediate (Within 1 hour)

- [ ] All critical services operational
- [ ] Health checks passing
- [ ] Customer-facing endpoints responding
- [ ] Monitoring and alerting active
- [ ] Status page updated

### Short-term (Within 24 hours)

- [ ] Full data integrity verification
- [ ] Performance baseline comparison
- [ ] Security scan completed
- [ ] Backup schedule resumed
- [ ] Incident timeline documented

### Long-term (Within 1 week)

- [ ] Post-mortem conducted
- [ ] Root cause identified
- [ ] Preventive measures documented
- [ ] Runbook updated if needed
- [ ] DR test scheduled to verify improvements

## Emergency Contacts

| Role            | Name     | Phone     | Email                  |
| --------------- | -------- | --------- | ---------------------- |
| On-Call Primary | Rotation | PagerDuty | oncall@neurectomy.io   |
| Platform Lead   | TBD      | TBD       | platform@neurectomy.io |
| Security Lead   | TBD      | TBD       | security@neurectomy.io |
| AWS TAM         | TBD      | TBD       | aws-support@amazon.com |

## Related Documents

- [DR Overview](./DR-OVERVIEW.md)
- [RTO/RPO Policies](./RTO-RPO-POLICIES.md)
- [Backup Procedures](./BACKUP-PROCEDURES.md)
- [DR Testing](./DR-TESTING.md)
