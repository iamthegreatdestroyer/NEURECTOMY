#!/bin/bash
# Neurectomy SLO Monitoring Stack Deployment Script
# Phase 18B: Complete AlertManager + SLO Dashboards Deployment
# 
# Usage: ./deploy-monitoring-stack.sh [--dry-run] [--namespace neurectomy]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="${NAMESPACE:-neurectomy}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_VALIDATION="${SKIP_VALIDATION:-false}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ${NC} $*"
}

log_success() {
    echo -e "${GREEN}✓${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}⚠${NC} $*"
}

log_error() {
    echo -e "${RED}✗${NC} $*"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo ""
echo "=========================================="
echo "Neurectomy Monitoring Stack Deployment"
echo "=========================================="
echo "Namespace: $NAMESPACE"
echo "Dry Run: $DRY_RUN"
echo "Validation: $([[ $SKIP_VALIDATION == true ]] && echo 'SKIPPED' || echo 'ENABLED')"
echo ""

# =============================================================================
# SECTION 1: VALIDATION
# =============================================================================

if [[ $SKIP_VALIDATION == false ]]; then
    log_info "Validating environment..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi
    log_success "kubectl found: $(kubectl version --client --short)"
    
    # Check namespace exists
    if ! kubectl get namespace $NAMESPACE &> /dev/null; then
        log_warn "Namespace $NAMESPACE does not exist. Creating..."
        if [[ $DRY_RUN == true ]]; then
            log_info "[DRY-RUN] kubectl create namespace $NAMESPACE"
        else
            kubectl create namespace $NAMESPACE
            log_success "Namespace created"
        fi
    else
        log_success "Namespace $NAMESPACE exists"
    fi
    
    # Check if Prometheus exists
    if ! kubectl get statefulset prometheus -n $NAMESPACE &> /dev/null; then
        log_error "Prometheus StatefulSet not found in namespace $NAMESPACE"
        log_error "Please deploy Prometheus first (deploy/k8s/08-prometheus-statefulset.yaml)"
        exit 1
    fi
    log_success "Prometheus StatefulSet found"
    
    # Check if AlertManager exists
    if ! kubectl get statefulset alertmanager -n $NAMESPACE &> /dev/null; then
        log_warn "AlertManager StatefulSet not found. Will create..."
    else
        log_success "AlertManager StatefulSet found"
    fi
    
    # Check if Grafana exists
    if ! kubectl get deployment grafana -n $NAMESPACE &> /dev/null; then
        log_warn "Grafana Deployment not found. SLO dashboards will not be imported."
    else
        log_success "Grafana Deployment found"
    fi
fi

# =============================================================================
# SECTION 2: ENVIRONMENT VARIABLES VALIDATION
# =============================================================================

log_info "Validating required environment variables..."

required_vars=(
    "SLACK_WEBHOOK_URL"
    "PAGERDUTY_SERVICE_KEY_CRITICAL"
    "PAGERDUTY_SERVICE_KEY_SLO"
    "SMTP_HOST"
    "SMTP_PORT"
    "SMTP_USERNAME"
    "SMTP_PASSWORD"
    "SMTP_FROM"
)

missing_vars=()
for var in "${required_vars[@]}"; do
    if [[ -z "${!var:-}" ]]; then
        missing_vars+=("$var")
    fi
done

if [[ ${#missing_vars[@]} -gt 0 ]]; then
    log_error "Missing required environment variables:"
    for var in "${missing_vars[@]}"; do
        log_error "  - $var"
    done
    log_error ""
    log_error "Please set these variables before deploying. See:"
    log_error "  PHASE-18B-ENVIRONMENT-VARIABLES-SETUP.md"
    exit 1
fi

log_success "All required environment variables set"

# =============================================================================
# SECTION 3: CREATE/UPDATE KUBERNETES SECRETS
# =============================================================================

log_info "Creating/Updating Kubernetes secrets..."

if [[ $DRY_RUN == true ]]; then
    log_info "[DRY-RUN] kubectl create secret generic alertmanager-secrets..."
    log_info "[DRY-RUN]   SLACK_WEBHOOK_URL: (***hidden***)"
    log_info "[DRY-RUN]   PAGERDUTY_SERVICE_KEY_CRITICAL: (***hidden***)"
else
    # Delete existing secret if present
    kubectl delete secret alertmanager-secrets -n $NAMESPACE 2>/dev/null || true
    
    # Create new secret with all environment variables
    kubectl create secret generic alertmanager-secrets \
        --from-literal=slack_webhook_url="$SLACK_WEBHOOK_URL" \
        --from-literal=pagerduty_service_key_critical="$PAGERDUTY_SERVICE_KEY_CRITICAL" \
        --from-literal=pagerduty_service_key_slo="${PAGERDUTY_SERVICE_KEY_SLO:-$PAGERDUTY_SERVICE_KEY_CRITICAL}" \
        --from-literal=pagerduty_service_key_security="${PAGERDUTY_SERVICE_KEY_SECURITY:-$PAGERDUTY_SERVICE_KEY_CRITICAL}" \
        --from-literal=smtp_host="$SMTP_HOST" \
        --from-literal=smtp_port="$SMTP_PORT" \
        --from-literal=smtp_username="$SMTP_USERNAME" \
        --from-literal=smtp_password="$SMTP_PASSWORD" \
        --from-literal=smtp_from="$SMTP_FROM" \
        --from-literal=smtp_to_critical="${SMTP_TO_CRITICAL:-admin@neurectomy.local}" \
        --from-literal=opsgenie_api_key="${OPSGENIE_API_KEY:-}" \
        -n $NAMESPACE
    
    log_success "Secrets created in namespace $NAMESPACE"
fi

# =============================================================================
# SECTION 4: CREATE PROMETHEUS ALERT RULES CONFIGMAP
# =============================================================================

log_info "Updating Prometheus alert rules..."

ALERT_RULES_FILE="$SCRIPT_DIR/docker/prometheus/alert_rules.yml"

if [[ ! -f $ALERT_RULES_FILE ]]; then
    log_error "Alert rules file not found: $ALERT_RULES_FILE"
    exit 1
fi

if [[ $DRY_RUN == true ]]; then
    log_info "[DRY-RUN] kubectl create configmap prometheus-alerts --from-file=$ALERT_RULES_FILE"
else
    # Delete existing ConfigMap if present
    kubectl delete configmap prometheus-alerts -n $NAMESPACE 2>/dev/null || true
    
    # Create new ConfigMap
    kubectl create configmap prometheus-alerts \
        --from-file=alert_rules.yml=$ALERT_RULES_FILE \
        -n $NAMESPACE
    
    log_success "Prometheus alert rules updated"
fi

# =============================================================================
# SECTION 5: CREATE ALERTMANAGER CONFIGMAP
# =============================================================================

log_info "Updating AlertManager configuration..."

ALERTMANAGER_FILE="$SCRIPT_DIR/docker/alertmanager/alertmanager.yml"

if [[ ! -f $ALERTMANAGER_FILE ]]; then
    log_error "AlertManager config file not found: $ALERTMANAGER_FILE"
    exit 1
fi

if [[ $DRY_RUN == true ]]; then
    log_info "[DRY-RUN] kubectl create configmap alertmanager-config --from-file=$ALERTMANAGER_FILE"
else
    # Delete existing ConfigMap if present
    kubectl delete configmap alertmanager-config -n $NAMESPACE 2>/dev/null || true
    
    # Create new ConfigMap
    kubectl create configmap alertmanager-config \
        --from-file=alertmanager.yml=$ALERTMANAGER_FILE \
        -n $NAMESPACE
    
    log_success "AlertManager configuration updated"
fi

# =============================================================================
# SECTION 6: DEPLOY PROMETHEUS STATEFULSET UPDATES
# =============================================================================

log_info "Deploying Prometheus with updated alert rules..."

PROMETHEUS_FILE="$SCRIPT_DIR/deploy/k8s/08-prometheus-statefulset.yaml"

if [[ ! -f $PROMETHEUS_FILE ]]; then
    log_error "Prometheus StatefulSet file not found: $PROMETHEUS_FILE"
    exit 1
fi

if [[ $DRY_RUN == true ]]; then
    log_info "[DRY-RUN] kubectl apply -f $PROMETHEUS_FILE"
else
    kubectl apply -f $PROMETHEUS_FILE -n $NAMESPACE
    log_success "Prometheus StatefulSet deployed"
fi

# =============================================================================
# SECTION 7: DEPLOY ALERTMANAGER STATEFULSET
# =============================================================================

log_info "Deploying AlertManager..."

ALERTMANAGER_STATEFULSET="$SCRIPT_DIR/deploy/k8s/11-alertmanager-statefulset.yaml"

if [[ ! -f $ALERTMANAGER_STATEFULSET ]]; then
    log_error "AlertManager StatefulSet file not found: $ALERTMANAGER_STATEFULSET"
    log_warn "AlertManager will not be deployed. You may need to create it manually."
else
    if [[ $DRY_RUN == true ]]; then
        log_info "[DRY-RUN] kubectl apply -f $ALERTMANAGER_STATEFULSET"
    else
        kubectl apply -f $ALERTMANAGER_STATEFULSET -n $NAMESPACE
        log_success "AlertManager StatefulSet deployed"
    fi
fi

# =============================================================================
# SECTION 8: IMPORT SLO DASHBOARDS
# =============================================================================

log_info "Importing SLO dashboards into Grafana..."

DASHBOARD_DIR="$SCRIPT_DIR/deploy/k8s"
DASHBOARDS=(
    "slo-dashboard-ryot.json"
    "slo-dashboard-sigma-lang.json"
    "slo-dashboard-sigma-vault.json"
    "slo-dashboard-agent-collective.json"
)

# Get Grafana port
GRAFANA_PORT=$(kubectl get svc grafana -n $NAMESPACE -o jsonpath='{.spec.ports[0].port}' 2>/dev/null || echo "3000")
GRAFANA_POD=$(kubectl get pod -n $NAMESPACE -l app=grafana -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

if [[ -z $GRAFANA_POD ]]; then
    log_warn "Grafana pod not found. Skipping dashboard import."
else
    log_info "Found Grafana pod: $GRAFANA_POD"
    
    for dashboard in "${DASHBOARDS[@]}"; do
        dashboard_path="$DASHBOARD_DIR/$dashboard"
        
        if [[ ! -f $dashboard_path ]]; then
            log_warn "Dashboard not found: $dashboard_path"
            continue
        fi
        
        log_info "Importing dashboard: $dashboard"
        
        if [[ $DRY_RUN == true ]]; then
            log_info "[DRY-RUN] Copying $dashboard to Grafana provisioning directory"
        else
            # Copy dashboard to Grafana provisioning directory
            kubectl cp "$dashboard_path" \
                "$NAMESPACE/$GRAFANA_POD:/etc/grafana/provisioning/dashboards/$dashboard" 2>/dev/null || \
            log_warn "Could not copy dashboard $dashboard to Grafana pod. You may need to import manually."
        fi
    done
fi

# =============================================================================
# SECTION 9: WAIT FOR DEPLOYMENTS TO BE READY
# =============================================================================

if [[ $DRY_RUN == false ]]; then
    log_info "Waiting for deployments to be ready (timeout: 5m)..."
    
    # Wait for Prometheus
    if kubectl rollout status statefulset/prometheus -n $NAMESPACE --timeout=5m 2>/dev/null; then
        log_success "Prometheus is ready"
    else
        log_warn "Prometheus rollout timed out. Check pod logs."
    fi
    
    # Wait for AlertManager (if it exists)
    if kubectl get statefulset alertmanager -n $NAMESPACE &>/dev/null; then
        if kubectl rollout status statefulset/alertmanager -n $NAMESPACE --timeout=5m 2>/dev/null; then
            log_success "AlertManager is ready"
        else
            log_warn "AlertManager rollout timed out. Check pod logs."
        fi
    fi
fi

# =============================================================================
# SECTION 10: VALIDATION & POST-DEPLOYMENT CHECKS
# =============================================================================

if [[ $DRY_RUN == false ]]; then
    log_info "Running post-deployment validation..."
    
    # Check Prometheus is scraping alert rules
    sleep 3
    ALERT_RULES_COUNT=$(kubectl exec -n $NAMESPACE prometheus-0 -- \
        curl -s http://localhost:9090/api/v1/rules 2>/dev/null | \
        grep -c "alert" || echo "0")
    
    if [[ $ALERT_RULES_COUNT -gt 0 ]]; then
        log_success "Prometheus loaded $ALERT_RULES_COUNT alert rules"
    else
        log_warn "Prometheus alert rules may not be loaded yet. Check pod logs."
    fi
    
    # Check AlertManager configuration
    if kubectl get statefulset alertmanager -n $NAMESPACE &>/dev/null; then
        sleep 2
        if kubectl exec -n $NAMESPACE alertmanager-0 -- \
            curl -s http://localhost:9093/api/v1/status &>/dev/null; then
            log_success "AlertManager API is responding"
        else
            log_warn "AlertManager API not responding. Check pod logs."
        fi
    fi
fi

# =============================================================================
# SECTION 11: OUTPUT SUMMARY
# =============================================================================

echo ""
echo "=========================================="
echo "Deployment Summary"
echo "=========================================="
echo ""

if [[ $DRY_RUN == true ]]; then
    log_warn "This was a DRY-RUN. No changes were made."
    echo ""
    log_info "To execute the deployment, run:"
    echo "  $0 --namespace $NAMESPACE"
    echo ""
else
    log_success "Deployment completed successfully!"
    echo ""
    log_info "Access monitoring stack:"
    echo "  Prometheus:   http://localhost:9090"
    echo "  AlertManager: http://localhost:9093"
    echo "  Grafana:      http://localhost:3000"
    echo ""
    log_info "Forward ports with:"
    echo "  kubectl port-forward -n $NAMESPACE svc/prometheus 9090:9090 &"
    echo "  kubectl port-forward -n $NAMESPACE svc/alertmanager 9093:9093 &"
    echo "  kubectl port-forward -n $NAMESPACE svc/grafana 3000:3000 &"
    echo ""
    log_info "View logs:"
    echo "  kubectl logs -n $NAMESPACE -f statefulset/prometheus"
    echo "  kubectl logs -n $NAMESPACE -f statefulset/alertmanager"
    echo "  kubectl logs -n $NAMESPACE -f deployment/grafana"
    echo ""
    log_info "Next steps:"
    echo "  1. Test alert routing (see PHASE-18B-TESTING-VALIDATION-GUIDE.md)"
    echo "  2. Verify SLO dashboard displays correctly in Grafana"
    echo "  3. Run test alerts to verify notification channels"
    echo ""
fi

log_success "Script completed!"
echo ""
