#!/bin/bash
# scripts/dr/dr-failover.sh
# Primary DR failover script for NEURECTOMY
# 
# Usage: ./dr-failover.sh --target=<region>
# Example: ./dr-failover.sh --target=us-west-2

set -euo pipefail

# Configuration
PRIMARY_REGION="us-east-1"
DR_REGION="us-west-2"
AZURE_REGION="westus2"
HOSTED_ZONE_ID="${HOSTED_ZONE_ID:-}"
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

send_notification() {
    local message="$1"
    local severity="${2:-info}"
    
    if [ -n "$SLACK_WEBHOOK" ]; then
        curl -s -X POST "$SLACK_WEBHOOK" \
            -H 'Content-Type: application/json' \
            -d "{\"text\": \"[${severity^^}] DR Failover: ${message}\"}" \
            > /dev/null 2>&1 || true
    fi
    
    log_info "$message"
}

usage() {
    cat << EOF
NEURECTOMY Disaster Recovery Failover Script

Usage: $0 [OPTIONS]

Options:
    --target=REGION     Target region for failover (us-west-2, westus2)
    --dry-run           Simulate failover without making changes
    --force             Skip confirmation prompts
    --help              Show this help message

Examples:
    $0 --target=us-west-2                 # Failover to AWS DR region
    $0 --target=westus2                   # Failover to Azure
    $0 --target=us-west-2 --dry-run       # Simulate failover

EOF
    exit 0
}

# Parse arguments
TARGET_REGION=""
DRY_RUN=false
FORCE=false

for arg in "$@"; do
    case $arg in
        --target=*)
            TARGET_REGION="${arg#*=}"
            ;;
        --dry-run)
            DRY_RUN=true
            ;;
        --force)
            FORCE=true
            ;;
        --help)
            usage
            ;;
        *)
            log_error "Unknown option: $arg"
            usage
            ;;
    esac
done

if [ -z "$TARGET_REGION" ]; then
    log_error "Target region is required"
    usage
fi

# Validate target region
case $TARGET_REGION in
    us-west-2|us-east-1)
        CLOUD="aws"
        ;;
    westus2|eastus2)
        CLOUD="azure"
        ;;
    *)
        log_error "Invalid target region: $TARGET_REGION"
        exit 1
        ;;
esac

log_info "=========================================="
log_info "NEURECTOMY Disaster Recovery Failover"
log_info "=========================================="
log_info "Target Region: $TARGET_REGION"
log_info "Cloud Provider: $CLOUD"
log_info "Dry Run: $DRY_RUN"
log_info ""

# Confirmation
if [ "$FORCE" = false ] && [ "$DRY_RUN" = false ]; then
    read -p "Are you sure you want to failover to $TARGET_REGION? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        log_info "Failover cancelled"
        exit 0
    fi
fi

send_notification "Initiating failover to $TARGET_REGION" "warning"

# Phase 1: Pre-flight checks
log_info "Phase 1: Pre-flight Checks"
log_info "--------------------------"

if [ "$CLOUD" = "aws" ]; then
    # Check AWS credentials
    if ! aws sts get-caller-identity > /dev/null 2>&1; then
        log_error "AWS credentials not configured"
        exit 1
    fi
    
    # Check target cluster
    KUBECONFIG_DR="${HOME}/.kube/neurectomy-${TARGET_REGION}.config"
    if [ ! -f "$KUBECONFIG_DR" ]; then
        log_error "DR kubeconfig not found: $KUBECONFIG_DR"
        exit 1
    fi
    
    export KUBECONFIG="$KUBECONFIG_DR"
    
    if ! kubectl get nodes > /dev/null 2>&1; then
        log_error "Cannot connect to DR cluster"
        exit 1
    fi
    
    NODE_COUNT=$(kubectl get nodes --no-headers | wc -l)
    log_info "DR cluster nodes: $NODE_COUNT"
    
elif [ "$CLOUD" = "azure" ]; then
    # Check Azure credentials
    if ! az account show > /dev/null 2>&1; then
        log_error "Azure credentials not configured"
        exit 1
    fi
    
    # Get AKS credentials
    az aks get-credentials --resource-group neurectomy-rg --name neurectomy-aks --overwrite-existing
    
    if ! kubectl get nodes > /dev/null 2>&1; then
        log_error "Cannot connect to Azure AKS cluster"
        exit 1
    fi
fi

log_info "Pre-flight checks passed"

# Phase 2: Verify backup status
log_info ""
log_info "Phase 2: Verify Backup Status"
log_info "-----------------------------"

LATEST_BACKUP=$(velero backup get -o json 2>/dev/null | jq -r '.items | sort_by(.metadata.creationTimestamp) | last | .metadata.name' || echo "")

if [ -z "$LATEST_BACKUP" ] || [ "$LATEST_BACKUP" = "null" ]; then
    log_error "No backups found"
    exit 1
fi

BACKUP_STATUS=$(velero backup describe "$LATEST_BACKUP" -o json | jq -r '.status.phase')
BACKUP_TIME=$(velero backup describe "$LATEST_BACKUP" -o json | jq -r '.status.completionTimestamp')

log_info "Latest backup: $LATEST_BACKUP"
log_info "Backup status: $BACKUP_STATUS"
log_info "Backup time: $BACKUP_TIME"

if [ "$BACKUP_STATUS" != "Completed" ]; then
    log_warn "Latest backup is not completed (status: $BACKUP_STATUS)"
    if [ "$FORCE" = false ]; then
        read -p "Continue anyway? (yes/no): " confirm
        if [ "$confirm" != "yes" ]; then
            exit 1
        fi
    fi
fi

# Phase 3: Scale up DR workloads
log_info ""
log_info "Phase 3: Scale Up DR Workloads"
log_info "------------------------------"

if [ "$DRY_RUN" = true ]; then
    log_info "[DRY-RUN] Would scale up deployments in neurectomy namespace"
else
    kubectl scale deployment ml-service --replicas=5 -n neurectomy 2>/dev/null || true
    kubectl scale deployment mlflow --replicas=3 -n neurectomy 2>/dev/null || true
    kubectl scale deployment grafana --replicas=2 -n monitoring 2>/dev/null || true
    
    log_info "Waiting for deployments to be ready..."
    kubectl rollout status deployment/ml-service -n neurectomy --timeout=300s || true
fi

# Phase 4: Database failover (if AWS)
if [ "$CLOUD" = "aws" ]; then
    log_info ""
    log_info "Phase 4: Database Failover"
    log_info "--------------------------"
    
    DB_INSTANCE="neurectomy-dr-postgres"
    
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY-RUN] Would promote RDS read replica: $DB_INSTANCE"
    else
        # Check if DB needs promotion
        DB_STATUS=$(aws rds describe-db-instances --db-instance-identifier "$DB_INSTANCE" --region "$TARGET_REGION" --query 'DBInstances[0].DBInstanceStatus' --output text 2>/dev/null || echo "not-found")
        
        if [ "$DB_STATUS" = "available" ]; then
            DB_ROLE=$(aws rds describe-db-instances --db-instance-identifier "$DB_INSTANCE" --region "$TARGET_REGION" --query 'DBInstances[0].ReadReplicaSourceDBInstanceIdentifier' --output text 2>/dev/null || echo "")
            
            if [ -n "$DB_ROLE" ] && [ "$DB_ROLE" != "None" ]; then
                log_info "Promoting RDS replica to standalone..."
                aws rds promote-read-replica \
                    --db-instance-identifier "$DB_INSTANCE" \
                    --backup-retention-period 7 \
                    --region "$TARGET_REGION"
                
                log_info "Waiting for promotion to complete..."
                aws rds wait db-instance-available \
                    --db-instance-identifier "$DB_INSTANCE" \
                    --region "$TARGET_REGION"
                
                log_info "Database promotion complete"
            else
                log_info "Database is already standalone"
            fi
        else
            log_warn "Database instance status: $DB_STATUS"
        fi
    fi
fi

# Phase 5: DNS Failover
log_info ""
log_info "Phase 5: DNS Failover"
log_info "--------------------"

if [ -z "$HOSTED_ZONE_ID" ]; then
    log_warn "HOSTED_ZONE_ID not set, skipping DNS failover"
else
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY-RUN] Would update DNS to point to $TARGET_REGION"
    else
        if [ "$CLOUD" = "aws" ]; then
            # Get ALB DNS name
            ALB_DNS=$(kubectl get ingress -n neurectomy -o jsonpath='{.items[0].status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
            
            if [ -n "$ALB_DNS" ]; then
                log_info "Updating DNS to: $ALB_DNS"
                
                aws route53 change-resource-record-sets \
                    --hosted-zone-id "$HOSTED_ZONE_ID" \
                    --change-batch "{
                        \"Changes\": [{
                            \"Action\": \"UPSERT\",
                            \"ResourceRecordSet\": {
                                \"Name\": \"api.neurectomy.io\",
                                \"Type\": \"CNAME\",
                                \"TTL\": 60,
                                \"ResourceRecords\": [{\"Value\": \"$ALB_DNS\"}]
                            }
                        }]
                    }"
                
                log_info "DNS update submitted"
            else
                log_warn "Could not determine ALB DNS name"
            fi
        fi
    fi
fi

# Phase 6: Verification
log_info ""
log_info "Phase 6: Verification"
log_info "--------------------"

if [ "$DRY_RUN" = true ]; then
    log_info "[DRY-RUN] Would verify failover"
else
    # Check pods
    READY_PODS=$(kubectl get pods -n neurectomy --no-headers | grep -c "Running" || echo "0")
    TOTAL_PODS=$(kubectl get pods -n neurectomy --no-headers | wc -l)
    log_info "Pods ready: $READY_PODS/$TOTAL_PODS"
    
    # Check services
    ENDPOINTS=$(kubectl get endpoints -n neurectomy --no-headers | wc -l)
    log_info "Service endpoints: $ENDPOINTS"
    
    # Health check
    if command -v curl &> /dev/null; then
        HEALTH_URL=$(kubectl get ingress -n neurectomy -o jsonpath='{.items[0].spec.rules[0].host}' 2>/dev/null || echo "")
        if [ -n "$HEALTH_URL" ]; then
            HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "https://${HEALTH_URL}/health" 2>/dev/null || echo "000")
            log_info "Health check status: $HEALTH_STATUS"
        fi
    fi
fi

# Summary
log_info ""
log_info "=========================================="
log_info "Failover Summary"
log_info "=========================================="
log_info "Target Region: $TARGET_REGION"
log_info "Cloud Provider: $CLOUD"
log_info "Status: $([ "$DRY_RUN" = true ] && echo "DRY-RUN COMPLETE" || echo "COMPLETE")"
log_info "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

send_notification "Failover to $TARGET_REGION completed" "info"

log_info ""
log_info "Next steps:"
log_info "1. Verify all services are healthy"
log_info "2. Run smoke tests: ./scripts/dr/smoke-tests.sh"
log_info "3. Monitor metrics and alerts"
log_info "4. Update status page"
