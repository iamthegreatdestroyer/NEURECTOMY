#!/bin/bash
# Comprehensive Kubernetes Deployment Script for Neurectomy Phase 18A Monitoring Stack
# Usage: ./deploy-monitoring.sh [action] [environment]
# Actions: deploy, validate, destroy, upgrade, rollback
# Environments: staging, production

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MANIFESTS_DIR="${SCRIPT_DIR}/k8s"
NAMESPACE="monitoring"
ACTION="${1:-deploy}"
ENVIRONMENT="${2:-staging}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${PROJECT_DIR}/logs/deployment_${TIMESTAMP}.log"

# Create logs directory
mkdir -p "$(dirname "$LOG_FILE")"

# Logging function
log() {
    local level=$1
    shift
    local message="$@"
    echo -e "${level}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ${message}" | tee -a "$LOG_FILE"
}

info() { log "$BLUE"; }
success() { log "$GREEN"; }
warn() { log "$YELLOW"; }
error() { log "$RED"; }

# Validate prerequisites
validate_prerequisites() {
    log "$BLUE" "=== Validating Prerequisites ==="
    
    local missing_tools=()
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        missing_tools+=("kubectl")
    else
        log "$GREEN" "✓ kubectl $(kubectl version --client --short)"
    fi
    
    # Check helm
    if ! command -v helm &> /dev/null; then
        missing_tools+=("helm")
    else
        log "$GREEN" "✓ helm $(helm version --short)"
    fi
    
    # Check kustomize
    if ! command -v kustomize &> /dev/null; then
        warn "⚠ kustomize not found (optional)"
    else
        log "$GREEN" "✓ kustomize"
    fi
    
    if [ ${#missing_tools[@]} -gt 0 ]; then
        error "❌ Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    # Verify kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        error "❌ Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log "$GREEN" "✓ Connected to cluster: $(kubectl config current-context)"
    log "$GREEN" "✓ Kubernetes version: $(kubectl version --short)"
}

# Validate manifests
validate_manifests() {
    log "$BLUE" "=== Validating Kubernetes Manifests ==="
    
    local errors=0
    
    # Validate YAML syntax
    for manifest in "${MANIFESTS_DIR}"/*.yaml; do
        if ! kubectl apply -f "$manifest" --dry-run=client > /dev/null 2>&1; then
            error "❌ Validation failed: $(basename $manifest)"
            ((errors++))
        else
            log "$GREEN" "✓ $(basename $manifest)"
        fi
    done
    
    if [ $errors -gt 0 ]; then
        error "❌ Manifest validation failed with $errors errors"
        return 1
    fi
    
    log "$GREEN" "✓ All manifests validated successfully"
}

# Validate Prometheus configuration
validate_prometheus_config() {
    log "$BLUE" "=== Validating Prometheus Configuration ==="
    
    # Extract prometheus.yml from ConfigMap
    local prometheus_config="/tmp/prometheus.yml.${TIMESTAMP}"
    kubectl get configmap prometheus-config -n "$NAMESPACE" -o jsonpath='{.data.prometheus\.yml}' > "$prometheus_config" 2>/dev/null || {
        warn "⚠ Prometheus ConfigMap not yet deployed, skipping validation"
        return 0
    }
    
    # Use promtool for validation if available
    if command -v promtool &> /dev/null; then
        if promtool check config "$prometheus_config" > /dev/null 2>&1; then
            log "$GREEN" "✓ Prometheus configuration valid"
        else
            error "❌ Prometheus configuration invalid"
            promtool check config "$prometheus_config" | tee -a "$LOG_FILE"
            rm -f "$prometheus_config"
            return 1
        fi
    else
        warn "⚠ promtool not found, skipping detailed validation"
    fi
    
    rm -f "$prometheus_config"
}

# Validate AlertManager configuration
validate_alertmanager_config() {
    log "$BLUE" "=== Validating AlertManager Configuration ==="
    
    # Extract alertmanager.yml from Secret
    local alertmanager_config="/tmp/alertmanager.yml.${TIMESTAMP}"
    kubectl get secret alertmanager-config -n "$NAMESPACE" -o jsonpath='{.data.alertmanager\.yml}' | base64 -d > "$alertmanager_config" 2>/dev/null || {
        warn "⚠ AlertManager Secret not yet deployed, skipping validation"
        return 0
    }
    
    # Use amtool for validation if available
    if command -v amtool &> /dev/null; then
        if amtool check-config "$alertmanager_config" > /dev/null 2>&1; then
            log "$GREEN" "✓ AlertManager configuration valid"
        else
            error "❌ AlertManager configuration invalid"
            amtool check-config "$alertmanager_config" | tee -a "$LOG_FILE"
            rm -f "$alertmanager_config"
            return 1
        fi
    else
        warn "⚠ amtool not found, skipping detailed validation"
    fi
    
    rm -f "$alertmanager_config"
}

# Deploy monitoring stack
deploy_stack() {
    log "$BLUE" "=== Deploying Monitoring Stack to ${ENVIRONMENT} ==="
    
    # Create namespace if it doesn't exist
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f - > /dev/null 2>&1
    log "$GREEN" "✓ Namespace ready"
    
    # Apply manifests in correct order
    local manifests=(
        "00-namespace.yaml"
        "01-storageclass.yaml"
        "02-rbac.yaml"
        "03-networkpolicy.yaml"
        "04-podsecuritypolicy.yaml"
        "05-prometheus-configmap.yaml"
        "06-secrets.yaml"
        "07-pvcs.yaml"
        "08-prometheus-statefulset.yaml"
        "09-services.yaml"
        "10-grafana-deployment.yaml"
        "11-alertmanager-statefulset.yaml"
        "12-ingress.yaml"
    )
    
    for manifest in "${manifests[@]}"; do
        log "$BLUE" "Applying $(basename $manifest)..."
        if kubectl apply -f "${MANIFESTS_DIR}/${manifest}" | tee -a "$LOG_FILE"; then
            log "$GREEN" "✓ $(basename $manifest) deployed"
        else
            error "❌ Failed to apply $(basename $manifest)"
            return 1
        fi
    done
    
    log "$GREEN" "✓ All manifests deployed successfully"
}

# Wait for deployments to be ready
wait_for_deployments() {
    log "$BLUE" "=== Waiting for Deployments to be Ready ==="
    
    local timeout=600  # 10 minutes
    local start_time=$(date +%s)
    
    # Wait for Prometheus StatefulSet
    log "$BLUE" "Waiting for Prometheus (max 10 minutes)..."
    if kubectl rollout status statefulset/prometheus -n "$NAMESPACE" --timeout="${timeout}s"; then
        log "$GREEN" "✓ Prometheus ready"
    else
        error "❌ Prometheus deployment timed out"
        return 1
    fi
    
    # Wait for Grafana Deployment
    log "$BLUE" "Waiting for Grafana..."
    if kubectl rollout status deployment/grafana -n "$NAMESPACE" --timeout="${timeout}s"; then
        log "$GREEN" "✓ Grafana ready"
    else
        error "❌ Grafana deployment timed out"
        return 1
    fi
    
    # Wait for AlertManager StatefulSet
    log "$BLUE" "Waiting for AlertManager..."
    if kubectl rollout status statefulset/alertmanager -n "$NAMESPACE" --timeout="${timeout}s"; then
        log "$GREEN" "✓ AlertManager ready"
    else
        error "❌ AlertManager deployment timed out"
        return 1
    fi
}

# Verify deployment health
verify_health() {
    log "$BLUE" "=== Verifying Deployment Health ==="
    
    # Check pod status
    log "$BLUE" "Pod Status:"
    kubectl get pods -n "$NAMESPACE" -o wide | tee -a "$LOG_FILE"
    
    # Check PVC status
    log "$BLUE" "Persistent Volumes:"
    kubectl get pvc -n "$NAMESPACE" | tee -a "$LOG_FILE"
    
    # Check if all pods are running
    local not_ready=$(kubectl get pods -n "$NAMESPACE" -o jsonpath='{.items[?(@.status.phase!="Running")].metadata.name}')
    if [ -z "$not_ready" ]; then
        log "$GREEN" "✓ All pods running"
    else
        warn "⚠ Not ready pods: $not_ready"
    fi
    
    # Test Prometheus endpoint
    log "$BLUE" "Testing Prometheus endpoint..."
    if kubectl run -it --rm debug --image=curlimages/curl --restart=Never -n "$NAMESPACE" -- \
        curl -s http://prometheus:9090/-/ready | grep -q "Prometheus Server"; then
        log "$GREEN" "✓ Prometheus responding"
    else
        warn "⚠ Prometheus health check inconclusive"
    fi
}

# Create backup
create_backup() {
    log "$BLUE" "=== Creating Backup ==="
    
    local backup_dir="${PROJECT_DIR}/backups/${TIMESTAMP}"
    mkdir -p "$backup_dir"
    
    # Backup Prometheus TSDB
    log "$BLUE" "Backing up Prometheus TSDB..."
    kubectl exec -n "$NAMESPACE" prometheus-0 -- tar czf /tmp/prometheus-backup.tar.gz /prometheus/wal /prometheus/chunks_head || {
        warn "⚠ Could not backup Prometheus TSDB"
    }
    
    # Export all resources
    log "$BLUE" "Exporting Kubernetes resources..."
    kubectl get all -n "$NAMESPACE" -o yaml > "${backup_dir}/resources.yaml"
    kubectl get pvc -n "$NAMESPACE" -o yaml > "${backup_dir}/pvcs.yaml"
    kubectl get secrets -n "$NAMESPACE" -o yaml > "${backup_dir}/secrets.yaml"
    
    log "$GREEN" "✓ Backup created at $backup_dir"
    echo "$backup_dir"
}

# Destroy deployment
destroy_stack() {
    log "$YELLOW" "=== Destroying Monitoring Stack ==="
    
    read -p "Are you sure you want to destroy the monitoring stack? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        log "$BLUE" "Destruction cancelled"
        return 0
    fi
    
    # Create backup before destruction
    local backup_dir
    backup_dir=$(create_backup)
    log "$GREEN" "✓ Backup created before destruction at $backup_dir"
    
    # Delete in reverse order
    local manifests=(
        "12-ingress.yaml"
        "11-alertmanager-statefulset.yaml"
        "10-grafana-deployment.yaml"
        "09-services.yaml"
        "08-prometheus-statefulset.yaml"
        "07-pvcs.yaml"
        "06-secrets.yaml"
        "05-prometheus-configmap.yaml"
        "04-podsecuritypolicy.yaml"
        "03-networkpolicy.yaml"
        "02-rbac.yaml"
        "01-storageclass.yaml"
        "00-namespace.yaml"
    )
    
    for manifest in "${manifests[@]}"; do
        log "$BLUE" "Removing $(basename $manifest)..."
        kubectl delete -f "${MANIFESTS_DIR}/${manifest}" --ignore-not-found=true 2>&1 | grep -v "not found" | tee -a "$LOG_FILE" || true
    done
    
    log "$GREEN" "✓ Monitoring stack destroyed"
}

# Upgrade monitoring stack
upgrade_stack() {
    log "$BLUE" "=== Upgrading Monitoring Stack ==="
    
    # Create backup
    create_backup
    
    # Reapply all manifests (kubectl apply is idempotent)
    deploy_stack
    
    # Wait for rollout
    wait_for_deployments
    
    log "$GREEN" "✓ Monitoring stack upgraded successfully"
}

# Rollback deployment
rollback_deployment() {
    log "$YELLOW" "=== Rolling Back Deployment ==="
    
    # Rollback Prometheus
    log "$BLUE" "Rolling back Prometheus..."
    kubectl rollout undo statefulset/prometheus -n "$NAMESPACE"
    kubectl rollout status statefulset/prometheus -n "$NAMESPACE" --timeout=300s
    
    # Rollback Grafana
    log "$BLUE" "Rolling back Grafana..."
    kubectl rollout undo deployment/grafana -n "$NAMESPACE"
    kubectl rollout status deployment/grafana -n "$NAMESPACE" --timeout=300s
    
    # Rollback AlertManager
    log "$BLUE" "Rolling back AlertManager..."
    kubectl rollout undo statefulset/alertmanager -n "$NAMESPACE"
    kubectl rollout status statefulset/alertmanager -n "$NAMESPACE" --timeout=300s
    
    log "$GREEN" "✓ Rollback completed"
}

# Main execution
main() {
    log "$BLUE" "╔═════════════════════════════════════════════════════════╗"
    log "$BLUE" "║  Neurectomy Phase 18A Monitoring Stack Deployment       ║"
    log "$BLUE" "║  Environment: $ENVIRONMENT                              ║"
    log "$BLUE" "║  Action: $ACTION                                        ║"
    log "$BLUE" "╚═════════════════════════════════════════════════════════╝"
    
    validate_prerequisites
    
    case "$ACTION" in
        deploy)
            validate_manifests
            validate_prometheus_config
            validate_alertmanager_config
            deploy_stack
            wait_for_deployments
            verify_health
            log "$GREEN" "✓ Deployment completed successfully!"
            ;;
        validate)
            validate_manifests
            validate_prometheus_config
            validate_alertmanager_config
            log "$GREEN" "✓ All validations passed!"
            ;;
        upgrade)
            upgrade_stack
            ;;
        rollback)
            rollback_deployment
            ;;
        destroy)
            destroy_stack
            ;;
        *)
            error "Unknown action: $ACTION"
            echo "Usage: $0 [deploy|validate|upgrade|rollback|destroy] [staging|production]"
            exit 1
            ;;
    esac
    
    log "$GREEN" "═══════════════════════════════════════════════════════"
    log "$GREEN" "Log file: $LOG_FILE"
}

main "$@"
