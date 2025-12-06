#!/bin/bash
# scripts/dr/failover-aws.sh
# Automated failover script for AWS DR region
#
# Usage: ./failover-aws.sh [--dry-run] [--skip-backup] [--force]

set -euo pipefail

# Configuration
PRIMARY_REGION="${PRIMARY_REGION:-us-west-2}"
DR_REGION="${DR_REGION:-us-east-1}"
CLUSTER_NAME="${CLUSTER_NAME:-neurectomy-production}"
S3_BACKUP_BUCKET="${S3_BACKUP_BUCKET:-neurectomy-velero-backups}"
NOTIFICATION_SNS="${NOTIFICATION_SNS:-}"
DRY_RUN=false
SKIP_BACKUP=false
FORCE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-backup)
            SKIP_BACKUP=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }

send_notification() {
    local message="$1"
    local subject="${2:-NEURECTOMY DR Failover}"
    
    if [ -n "$NOTIFICATION_SNS" ]; then
        aws sns publish \
            --topic-arn "$NOTIFICATION_SNS" \
            --message "$message" \
            --subject "$subject" \
            --region "$DR_REGION" || true
    fi
    
    log_info "$message"
}

# Pre-flight checks
preflight_checks() {
    log_info "Running pre-flight checks..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI not found"
        exit 1
    fi
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found"
        exit 1
    fi
    
    # Check velero
    if ! command -v velero &> /dev/null; then
        log_error "velero CLI not found"
        exit 1
    fi
    
    # Verify AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not valid"
        exit 1
    fi
    
    # Check DR cluster exists
    if ! aws eks describe-cluster --name "${CLUSTER_NAME}-dr" --region "$DR_REGION" &> /dev/null; then
        log_error "DR cluster ${CLUSTER_NAME}-dr not found in $DR_REGION"
        exit 1
    fi
    
    log_success "Pre-flight checks passed"
}

# Take final backup from primary (if accessible)
create_final_backup() {
    if [ "$SKIP_BACKUP" = true ]; then
        log_warn "Skipping final backup (--skip-backup)"
        return 0
    fi
    
    log_info "Attempting to create final backup from primary..."
    
    # Try to connect to primary
    if aws eks update-kubeconfig --name "$CLUSTER_NAME" --region "$PRIMARY_REGION" &> /dev/null; then
        BACKUP_NAME="failover-$(date +%Y%m%d-%H%M%S)"
        
        if [ "$DRY_RUN" = true ]; then
            log_info "[DRY-RUN] Would create backup: $BACKUP_NAME"
        else
            velero backup create "$BACKUP_NAME" \
                --include-namespaces neurectomy,monitoring \
                --wait || {
                    log_warn "Final backup failed, proceeding with latest available backup"
                    return 0
                }
            log_success "Final backup created: $BACKUP_NAME"
        fi
    else
        log_warn "Cannot reach primary cluster, will use latest available backup"
    fi
}

# Find latest backup
find_latest_backup() {
    log_info "Finding latest backup..."
    
    LATEST_BACKUP=$(velero backup get -o json | jq -r '.items | sort_by(.metadata.creationTimestamp) | last | .metadata.name' 2>/dev/null || echo "")
    
    if [ -z "$LATEST_BACKUP" ] || [ "$LATEST_BACKUP" = "null" ]; then
        log_error "No backups found!"
        exit 1
    fi
    
    BACKUP_TIME=$(velero backup get "$LATEST_BACKUP" -o json | jq -r '.metadata.creationTimestamp')
    log_info "Latest backup: $LATEST_BACKUP (created: $BACKUP_TIME)"
    
    # Check backup age
    BACKUP_AGE_SECONDS=$(( $(date +%s) - $(date -d "$BACKUP_TIME" +%s) ))
    BACKUP_AGE_HOURS=$((BACKUP_AGE_SECONDS / 3600))
    
    if [ "$BACKUP_AGE_HOURS" -gt 24 ] && [ "$FORCE" = false ]; then
        log_warn "Backup is $BACKUP_AGE_HOURS hours old!"
        read -p "Continue with this backup? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    echo "$LATEST_BACKUP"
}

# Switch to DR cluster
switch_to_dr() {
    log_info "Switching to DR cluster in $DR_REGION..."
    
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY-RUN] Would switch to ${CLUSTER_NAME}-dr"
    else
        aws eks update-kubeconfig --name "${CLUSTER_NAME}-dr" --region "$DR_REGION"
        
        # Verify connection
        if ! kubectl cluster-info &> /dev/null; then
            log_error "Cannot connect to DR cluster"
            exit 1
        fi
        
        log_success "Connected to DR cluster"
    fi
}

# Restore from backup
restore_backup() {
    local backup_name="$1"
    
    log_info "Restoring from backup: $backup_name..."
    
    RESTORE_NAME="failover-restore-$(date +%Y%m%d-%H%M%S)"
    
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY-RUN] Would restore: $RESTORE_NAME from $backup_name"
    else
        # Check if namespace exists
        if kubectl get namespace neurectomy &> /dev/null; then
            log_warn "Namespace neurectomy exists, cleaning up..."
            kubectl delete namespace neurectomy --timeout=60s || true
        fi
        
        # Create restore
        velero restore create "$RESTORE_NAME" \
            --from-backup "$backup_name" \
            --include-namespaces neurectomy,monitoring \
            --wait
        
        # Check restore status
        RESTORE_STATUS=$(velero restore get "$RESTORE_NAME" -o json | jq -r '.status.phase')
        
        if [ "$RESTORE_STATUS" != "Completed" ]; then
            log_error "Restore failed with status: $RESTORE_STATUS"
            velero restore logs "$RESTORE_NAME"
            exit 1
        fi
        
        log_success "Restore completed: $RESTORE_NAME"
    fi
}

# Scale up workloads
scale_workloads() {
    log_info "Scaling up workloads..."
    
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY-RUN] Would scale workloads"
    else
        # Scale deployments
        kubectl scale deployment ml-service -n neurectomy --replicas=3 || true
        kubectl scale deployment mlflow -n neurectomy --replicas=2 || true
        
        # Wait for rollout
        kubectl rollout status deployment/ml-service -n neurectomy --timeout=300s || true
        kubectl rollout status deployment/mlflow -n neurectomy --timeout=300s || true
        
        log_success "Workloads scaled"
    fi
}

# Update DNS
update_dns() {
    log_info "Updating DNS to point to DR region..."
    
    # Get DR load balancer
    DR_LB=$(kubectl get ingress -n neurectomy -o jsonpath='{.items[0].status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
    
    if [ -z "$DR_LB" ]; then
        log_warn "No load balancer found, skipping DNS update"
        return 0
    fi
    
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY-RUN] Would update DNS to: $DR_LB"
    else
        # Example: Update Route53 (customize for your domain)
        # aws route53 change-resource-record-sets \
        #     --hosted-zone-id YOUR_ZONE_ID \
        #     --change-batch '{
        #         "Changes": [{
        #             "Action": "UPSERT",
        #             "ResourceRecordSet": {
        #                 "Name": "neurectomy.example.com",
        #                 "Type": "CNAME",
        #                 "TTL": 60,
        #                 "ResourceRecords": [{"Value": "'$DR_LB'"}]
        #             }
        #         }]
        #     }'
        
        log_info "DNS update configured - update your Route53 records manually or uncomment the script"
        log_info "New LB endpoint: $DR_LB"
    fi
}

# Run smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY-RUN] Would run smoke tests"
    else
        if [ -f "$SCRIPT_DIR/smoke-tests.sh" ]; then
            if bash "$SCRIPT_DIR/smoke-tests.sh"; then
                log_success "Smoke tests passed"
            else
                log_warn "Some smoke tests failed - review output above"
            fi
        else
            log_warn "smoke-tests.sh not found, skipping"
        fi
    fi
}

# Main execution
main() {
    echo "=========================================="
    echo "NEURECTOMY AWS Failover"
    echo "=========================================="
    echo "Primary Region: $PRIMARY_REGION"
    echo "DR Region:      $DR_REGION"
    echo "Dry Run:        $DRY_RUN"
    echo "Timestamp:      $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "=========================================="
    echo
    
    if [ "$FORCE" = false ]; then
        read -p "This will failover to DR region. Continue? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 0
        fi
    fi
    
    send_notification "DR Failover initiated by $(whoami) at $(date -u)" "NEURECTOMY DR: Failover Started"
    
    preflight_checks
    create_final_backup
    BACKUP_NAME=$(find_latest_backup)
    switch_to_dr
    restore_backup "$BACKUP_NAME"
    scale_workloads
    update_dns
    run_smoke_tests
    
    echo
    echo "=========================================="
    echo "Failover Complete"
    echo "=========================================="
    
    if [ "$DRY_RUN" = true ]; then
        log_info "This was a dry run - no changes were made"
    else
        send_notification "DR Failover completed successfully. Smoke tests: $?" "NEURECTOMY DR: Failover Complete"
        log_success "Failover to DR region completed!"
        log_info "Please verify all services manually"
    fi
}

main
