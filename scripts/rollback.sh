#!/usr/bin/env bash
# ============================================================================
# NEURECTOMY - Kubernetes Rollback Script
# Manual rollback helper for emergency situations
# ============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_NAME="neurectomy-rollback"
FLUX_NAMESPACE="flux-system"
DEFAULT_TIMEOUT="5m"

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Help text
show_help() {
    cat << EOF
NEURECTOMY Rollback Script

Usage: $0 <command> [options]

Commands:
    status          Show current deployment status
    suspend         Suspend Flux reconciliation for an environment
    resume          Resume Flux reconciliation
    rollback-git    Trigger Git-based rollback (revert last commit)
    rollback-k8s    Rollback Kubernetes deployment to previous revision
    history         Show deployment history
    diff            Show diff between current and previous state

Options:
    -e, --environment   Environment (development|staging|production)
    -d, --deployment    Deployment name (default: ml-service)
    -n, --namespace     Kubernetes namespace
    -r, --revision      Revision to rollback to
    -t, --timeout       Timeout for operations (default: 5m)
    -y, --yes           Skip confirmation prompts
    -h, --help          Show this help message

Examples:
    $0 status -e production
    $0 suspend -e production
    $0 rollback-k8s -e production -d ml-service -r 3
    $0 resume -e production

EOF
}

# Get environment namespace
get_namespace() {
    local env=$1
    case $env in
        development|dev) echo "neurectomy-dev" ;;
        staging|stg) echo "neurectomy-staging" ;;
        production|prod) echo "neurectomy-prod" ;;
        *) log_error "Unknown environment: $env"; exit 1 ;;
    esac
}

# Get Flux Kustomization name
get_kustomization_name() {
    local env=$1
    case $env in
        development|dev) echo "neurectomy-development" ;;
        staging|stg) echo "neurectomy-staging" ;;
        production|prod) echo "neurectomy-production" ;;
        *) log_error "Unknown environment: $env"; exit 1 ;;
    esac
}

# Show deployment status
cmd_status() {
    local env=${ENVIRONMENT:-all}
    
    log_info "Fetching deployment status..."
    
    if [[ "$env" == "all" ]]; then
        echo ""
        echo "=== Flux Kustomizations ==="
        kubectl get kustomizations -n "$FLUX_NAMESPACE" -o wide 2>/dev/null || log_warning "No Flux kustomizations found"
        
        echo ""
        echo "=== Git Repositories ==="
        kubectl get gitrepositories -n "$FLUX_NAMESPACE" -o wide 2>/dev/null || log_warning "No Git repositories found"
        
        echo ""
        echo "=== Deployments ==="
        for ns in neurectomy-dev neurectomy-staging neurectomy-prod; do
            echo "--- $ns ---"
            kubectl get deployments -n "$ns" -o wide 2>/dev/null || echo "Namespace not found or empty"
        done
    else
        local namespace=$(get_namespace "$env")
        local kustomization=$(get_kustomization_name "$env")
        
        echo ""
        echo "=== Environment: $env ==="
        echo ""
        echo "Flux Kustomization:"
        kubectl get kustomization "$kustomization" -n "$FLUX_NAMESPACE" -o yaml 2>/dev/null | grep -A 20 "status:" || log_warning "Kustomization not found"
        
        echo ""
        echo "Deployments:"
        kubectl get deployments -n "$namespace" -o wide 2>/dev/null || log_warning "No deployments found"
        
        echo ""
        echo "Pods:"
        kubectl get pods -n "$namespace" -o wide 2>/dev/null || log_warning "No pods found"
    fi
}

# Suspend Flux reconciliation
cmd_suspend() {
    local env=${ENVIRONMENT:?Environment required}
    local kustomization=$(get_kustomization_name "$env")
    
    log_info "Suspending Flux reconciliation for $env..."
    
    if [[ "$SKIP_CONFIRM" != "true" ]]; then
        read -p "Are you sure you want to suspend $kustomization? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_warning "Aborted"
            exit 0
        fi
    fi
    
    flux suspend kustomization "$kustomization" -n "$FLUX_NAMESPACE"
    log_success "Suspended $kustomization"
}

# Resume Flux reconciliation
cmd_resume() {
    local env=${ENVIRONMENT:?Environment required}
    local kustomization=$(get_kustomization_name "$env")
    
    log_info "Resuming Flux reconciliation for $env..."
    
    flux resume kustomization "$kustomization" -n "$FLUX_NAMESPACE"
    log_success "Resumed $kustomization"
    
    log_info "Triggering immediate reconciliation..."
    flux reconcile kustomization "$kustomization" -n "$FLUX_NAMESPACE" --with-source
}

# Git-based rollback (revert last commit)
cmd_rollback_git() {
    local env=${ENVIRONMENT:?Environment required}
    
    log_warning "Git-based rollback will revert the last commit in the repository"
    
    if [[ "$SKIP_CONFIRM" != "true" ]]; then
        read -p "Are you sure you want to proceed? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_warning "Aborted"
            exit 0
        fi
    fi
    
    log_info "Creating Git revert commit..."
    
    # This assumes you're in the repository root
    git revert HEAD --no-edit
    
    log_info "Pushing revert commit..."
    git push origin HEAD
    
    log_success "Git revert complete. Flux will reconcile automatically."
    
    log_info "Triggering immediate reconciliation..."
    flux reconcile source git neurectomy -n "$FLUX_NAMESPACE"
}

# Kubernetes-native rollback
cmd_rollback_k8s() {
    local env=${ENVIRONMENT:?Environment required}
    local deployment=${DEPLOYMENT:-ml-service}
    local namespace=$(get_namespace "$env")
    local revision=${REVISION:-}
    
    # Prefix deployment name based on environment
    local full_deployment="${env:0:3}-${deployment}"
    if [[ "$env" == "development" ]]; then
        full_deployment="dev-${deployment}"
    elif [[ "$env" == "staging" ]]; then
        full_deployment="staging-${deployment}"
    elif [[ "$env" == "production" ]]; then
        full_deployment="prod-${deployment}"
    fi
    
    log_info "Rolling back deployment $full_deployment in $namespace..."
    
    # Suspend Flux first to prevent it from undoing our rollback
    log_info "Suspending Flux to prevent reconciliation during rollback..."
    flux suspend kustomization "$(get_kustomization_name "$env")" -n "$FLUX_NAMESPACE" 2>/dev/null || true
    
    if [[ -n "$revision" ]]; then
        log_info "Rolling back to revision $revision..."
        kubectl rollout undo deployment/"$full_deployment" -n "$namespace" --to-revision="$revision"
    else
        log_info "Rolling back to previous revision..."
        kubectl rollout undo deployment/"$full_deployment" -n "$namespace"
    fi
    
    log_info "Waiting for rollback to complete..."
    kubectl rollout status deployment/"$full_deployment" -n "$namespace" --timeout="$DEFAULT_TIMEOUT"
    
    log_success "Rollback complete"
    
    log_warning "IMPORTANT: Flux is still suspended. Once you verify the rollback is working:"
    log_warning "  1. Update the Git repository to match the rolled-back state"
    log_warning "  2. Run: $0 resume -e $env"
}

# Show deployment history
cmd_history() {
    local env=${ENVIRONMENT:?Environment required}
    local deployment=${DEPLOYMENT:-ml-service}
    local namespace=$(get_namespace "$env")
    
    local full_deployment="${env:0:3}-${deployment}"
    if [[ "$env" == "development" ]]; then
        full_deployment="dev-${deployment}"
    elif [[ "$env" == "staging" ]]; then
        full_deployment="staging-${deployment}"
    elif [[ "$env" == "production" ]]; then
        full_deployment="prod-${deployment}"
    fi
    
    log_info "Deployment history for $full_deployment in $namespace:"
    echo ""
    kubectl rollout history deployment/"$full_deployment" -n "$namespace"
}

# Show diff between current and previous
cmd_diff() {
    local env=${ENVIRONMENT:?Environment required}
    local namespace=$(get_namespace "$env")
    
    log_info "Flux diff for environment $env:"
    
    flux diff kustomization "$(get_kustomization_name "$env")" -n "$FLUX_NAMESPACE" 2>/dev/null || log_warning "Could not generate diff"
}

# Parse command line arguments
COMMAND=""
ENVIRONMENT=""
DEPLOYMENT=""
NAMESPACE=""
REVISION=""
TIMEOUT="$DEFAULT_TIMEOUT"
SKIP_CONFIRM="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        status|suspend|resume|rollback-git|rollback-k8s|history|diff)
            COMMAND=$1
            shift
            ;;
        -e|--environment)
            ENVIRONMENT=$2
            shift 2
            ;;
        -d|--deployment)
            DEPLOYMENT=$2
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE=$2
            shift 2
            ;;
        -r|--revision)
            REVISION=$2
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT=$2
            shift 2
            ;;
        -y|--yes)
            SKIP_CONFIRM="true"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check required tools
command -v kubectl >/dev/null 2>&1 || { log_error "kubectl is required but not installed."; exit 1; }
command -v flux >/dev/null 2>&1 || { log_error "flux CLI is required but not installed."; exit 1; }

# Execute command
case $COMMAND in
    status) cmd_status ;;
    suspend) cmd_suspend ;;
    resume) cmd_resume ;;
    rollback-git) cmd_rollback_git ;;
    rollback-k8s) cmd_rollback_k8s ;;
    history) cmd_history ;;
    diff) cmd_diff ;;
    "")
        log_error "No command specified"
        show_help
        exit 1
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac
