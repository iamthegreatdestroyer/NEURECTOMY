#!/bin/bash
# Backup and Recovery Script for Monitoring Stack
# Usage: ./backup-restore.sh [backup|restore] [backup_file]

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BACKUP_DIR="${PROJECT_DIR}/backups"
NAMESPACE="monitoring"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ACTION="${1:-backup}"
BACKUP_FILE="${2:-}"

mkdir -p "$BACKUP_DIR"

log() {
    local level=$1
    shift
    echo -e "${level}[$(date +'%H:%M:%S')]${NC} $@"
}

info() { log "$BLUE"; }
success() { log "$GREEN"; }
warn() { log "$YELLOW"; }
error() { log "$RED"; }

# Backup Prometheus TSDB
backup_prometheus() {
    info "=== Backing up Prometheus TSDB ==="
    
    local backup_path="${BACKUP_DIR}/prometheus_${TIMESTAMP}"
    mkdir -p "$backup_path"
    
    # Get Prometheus pod
    local prom_pod=$(kubectl get pods -n "$NAMESPACE" -l app=prometheus -o jsonpath='{.items[0].metadata.name}')
    
    # Create tar archive of TSDB
    info "Creating TSDB snapshot..."
    kubectl exec -n "$NAMESPACE" "$prom_pod" -- \
        tar czf /tmp/prometheus-tsdb.tar.gz \
        -C /prometheus . --exclude='*.tmp' --exclude='wal/*' \
        2>/dev/null || true
    
    # Copy from pod
    kubectl cp "$NAMESPACE/$prom_pod:/tmp/prometheus-tsdb.tar.gz" \
        "$backup_path/prometheus-tsdb.tar.gz" 2>/dev/null || true
    
    success "✓ Prometheus TSDB backed up to $backup_path"
    echo "$backup_path"
}

# Backup Grafana state
backup_grafana() {
    info "=== Backing up Grafana State ==="
    
    local backup_path="${BACKUP_DIR}/grafana_${TIMESTAMP}"
    mkdir -p "$backup_path"
    
    # Backup Grafana database
    local grafana_pod=$(kubectl get pods -n "$NAMESPACE" -l app=grafana -o jsonpath='{.items[0].metadata.name}')
    
    kubectl cp "$NAMESPACE/$grafana_pod:/var/lib/grafana/grafana.db" \
        "$backup_path/grafana.db" 2>/dev/null || true
    
    # Backup Grafana provisioning configs
    kubectl get configmap grafana-dashboards -n "$NAMESPACE" -o yaml > "$backup_path/grafana-dashboards-cm.yaml"
    kubectl get secret grafana-datasources -n "$NAMESPACE" -o yaml > "$backup_path/grafana-datasources-secret.yaml"
    
    success "✓ Grafana state backed up to $backup_path"
    echo "$backup_path"
}

# Backup AlertManager state
backup_alertmanager() {
    info "=== Backing up AlertManager State ==="
    
    local backup_path="${BACKUP_DIR}/alertmanager_${TIMESTAMP}"
    mkdir -p "$backup_path"
    
    # Backup AlertManager configuration and state
    local am_pod=$(kubectl get pods -n "$NAMESPACE" -l app=alertmanager -o jsonpath='{.items[0].metadata.name}')
    
    kubectl exec -n "$NAMESPACE" "$am_pod" -- \
        tar czf /tmp/alertmanager-data.tar.gz /alertmanager/data 2>/dev/null || true
    
    kubectl cp "$NAMESPACE/$am_pod:/tmp/alertmanager-data.tar.gz" \
        "$backup_path/alertmanager-data.tar.gz" 2>/dev/null || true
    
    # Backup configuration secret
    kubectl get secret alertmanager-config -n "$NAMESPACE" -o yaml > "$backup_path/alertmanager-config-secret.yaml"
    
    success "✓ AlertManager state backed up to $backup_path"
    echo "$backup_path"
}

# Backup entire monitoring stack
backup_all() {
    info "╔════════════════════════════════════════════════╗"
    info "║   Starting Full Monitoring Stack Backup        ║"
    info "╚════════════════════════════════════════════════╝"
    
    local backup_root="${BACKUP_DIR}/full_${TIMESTAMP}"
    mkdir -p "$backup_root"
    
    # Backup K8s manifests
    info "Backing up Kubernetes manifests..."
    kubectl get all -n "$NAMESPACE" -o yaml > "$backup_root/all-resources.yaml"
    kubectl get pvc -n "$NAMESPACE" -o yaml > "$backup_root/pvcs.yaml"
    kubectl get configmaps -n "$NAMESPACE" -o yaml > "$backup_root/configmaps.yaml"
    kubectl get secrets -n "$NAMESPACE" -o yaml > "$backup_root/secrets.yaml"
    
    # Create individual backups
    backup_prometheus | tail -1 | xargs -I {} cp -r {} "$backup_root/prometheus"
    backup_grafana | tail -1 | xargs -I {} cp -r {} "$backup_root/grafana"
    backup_alertmanager | tail -1 | xargs -I {} cp -r {} "$backup_root/alertmanager"
    
    # Create metadata
    cat > "$backup_root/backup-metadata.txt" << EOF
Backup Timestamp: $TIMESTAMP
Cluster: $(kubectl config current-context)
Namespace: $NAMESPACE
Kubernetes Version: $(kubectl version --short)
Backup Components:
  - Prometheus TSDB
  - Grafana State
  - AlertManager State
  - All K8s Resources
EOF
    
    # Compress entire backup
    info "Compressing backup..."
    tar czf "${backup_root}.tar.gz" -C "$BACKUP_DIR" "full_${TIMESTAMP}"
    rm -rf "$backup_root"
    
    success "✓ Full backup created: ${backup_root}.tar.gz"
    success "✓ Backup size: $(du -h ${backup_root}.tar.gz | cut -f1)"
}

# Restore from backup
restore_backup() {
    if [ -z "$BACKUP_FILE" ]; then
        error "✗ Backup file not specified"
        echo "Usage: $0 restore <backup_file>"
        return 1
    fi
    
    if [ ! -f "$BACKUP_FILE" ]; then
        error "✗ Backup file not found: $BACKUP_FILE"
        return 1
    fi
    
    warn "⚠ Restore will overwrite current monitoring data!"
    read -p "Are you sure you want to continue? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        info "Restore cancelled"
        return 0
    fi
    
    info "═══════════════════════════════════════════════"
    info "Extracting backup..."
    local restore_dir="${BACKUP_DIR}/restore_${TIMESTAMP}"
    mkdir -p "$restore_dir"
    
    if [[ "$BACKUP_FILE" == *.tar.gz ]]; then
        tar xzf "$BACKUP_FILE" -C "$restore_dir"
    else
        error "✗ Unknown backup format. Expected .tar.gz"
        return 1
    fi
    
    # Restore K8s resources
    if [ -f "$restore_dir/all-resources.yaml" ]; then
        info "Restoring Kubernetes resources..."
        kubectl apply -f "$restore_dir/all-resources.yaml" || true
    fi
    
    # Restore Prometheus
    if [ -f "$restore_dir/prometheus/prometheus-tsdb.tar.gz" ]; then
        info "Restoring Prometheus TSDB..."
        local prom_pod=$(kubectl get pods -n "$NAMESPACE" -l app=prometheus -o jsonpath='{.items[0].metadata.name}')
        kubectl cp "$restore_dir/prometheus/prometheus-tsdb.tar.gz" \
            "$NAMESPACE/$prom_pod:/tmp/prometheus-restore.tar.gz"
        kubectl exec -n "$NAMESPACE" "$prom_pod" -- \
            tar xzf /tmp/prometheus-restore.tar.gz -C /prometheus
    fi
    
    # Restore Grafana
    if [ -f "$restore_dir/grafana/grafana.db" ]; then
        info "Restoring Grafana database..."
        local grafana_pod=$(kubectl get pods -n "$NAMESPACE" -l app=grafana -o jsonpath='{.items[0].metadata.name}')
        
        # Scale down Grafana
        kubectl scale deployment grafana -n "$NAMESPACE" --replicas=0
        sleep 5
        
        # Copy database
        kubectl cp "$restore_dir/grafana/grafana.db" \
            "$NAMESPACE/$grafana_pod:/var/lib/grafana/grafana.db" 2>/dev/null || true
        
        # Scale up
        kubectl scale deployment grafana -n "$NAMESPACE" --replicas=2
    fi
    
    success "✓ Restore completed"
    success "✓ Monitor the deployment with: kubectl rollout status -n $NAMESPACE --all"
    
    # Cleanup
    rm -rf "$restore_dir"
}

# List available backups
list_backups() {
    info "=== Available Backups ==="
    
    if [ ! -d "$BACKUP_DIR" ] || [ -z "$(ls -A $BACKUP_DIR)" ]; then
        warn "No backups found in $BACKUP_DIR"
        return 0
    fi
    
    ls -lht "$BACKUP_DIR" | grep "^-" | awk '{print $9, "(" $5 ")"}'
}

# Verify backup integrity
verify_backup() {
    if [ -z "$BACKUP_FILE" ]; then
        error "✗ Backup file not specified"
        return 1
    fi
    
    if [ ! -f "$BACKUP_FILE" ]; then
        error "✗ Backup file not found: $BACKUP_FILE"
        return 1
    fi
    
    info "=== Verifying Backup ==="
    
    if tar tzf "$BACKUP_FILE" > /dev/null 2>&1; then
        success "✓ Backup file integrity verified"
        success "✓ Contents:"
        tar tzf "$BACKUP_FILE" | head -20
    else
        error "✗ Backup file is corrupted"
        return 1
    fi
}

# S3 sync (requires AWS CLI)
sync_to_s3() {
    if ! command -v aws &> /dev/null; then
        error "✗ AWS CLI not found"
        return 1
    fi
    
    info "=== Syncing Backups to S3 ==="
    
    local s3_bucket="${S3_BACKUP_BUCKET:-neurectomy-backups}"
    local s3_path="monitoring/${TIMESTAMP}"
    
    info "Uploading to s3://$s3_bucket/$s3_path..."
    
    if aws s3 sync "$BACKUP_DIR" "s3://$s3_bucket/$s3_path/" --exclude "*" --include "*.tar.gz"; then
        success "✓ Backups synced to S3"
    else
        error "✗ S3 sync failed"
        return 1
    fi
}

# Main execution
main() {
    case "$ACTION" in
        backup)
            backup_all
            list_backups
            ;;
        restore)
            restore_backup
            ;;
        verify)
            verify_backup
            ;;
        list)
            list_backups
            ;;
        s3-sync)
            sync_to_s3
            ;;
        *)
            error "Unknown action: $ACTION"
            echo "Usage: $0 [backup|restore|verify|list|s3-sync] [backup_file]"
            exit 1
            ;;
    esac
}

main "$@"
