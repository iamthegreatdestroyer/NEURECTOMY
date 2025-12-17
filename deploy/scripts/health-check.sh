#!/bin/bash
# Health Check and Diagnostics Script for Monitoring Stack
# Usage: ./health-check.sh [verbose]

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

NAMESPACE="monitoring"
VERBOSE="${1:-}"

log() {
    local level=$1
    shift
    echo -e "${level}[$(date +'%H:%M:%S')]${NC} $@"
}

info() { log "$BLUE"; }
success() { log "$GREEN"; }
warn() { log "$YELLOW"; }
error() { log "$RED"; }

# Check cluster connectivity
check_cluster() {
    info "=== Cluster Health ==="
    
    if ! kubectl cluster-info &> /dev/null; then
        error "✗ Cannot connect to cluster"
        return 1
    fi
    
    success "✓ Cluster: $(kubectl config current-context)"
    success "✓ Kubernetes version: $(kubectl version --short | grep Server)"
    
    # Node status
    local not_ready=$(kubectl get nodes -o jsonpath='{.items[?(@.status.conditions[?(@.type=="Ready")].status!="True")].metadata.name}')
    if [ -z "$not_ready" ]; then
        success "✓ All nodes ready"
    else
        warn "⚠ Not ready nodes: $not_ready"
    fi
}

# Check namespace
check_namespace() {
    info "=== Namespace Health ==="
    
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        error "✗ Namespace $NAMESPACE does not exist"
        return 1
    fi
    
    success "✓ Namespace $NAMESPACE exists"
}

# Check pods
check_pods() {
    info "=== Pod Status ==="
    
    kubectl get pods -n "$NAMESPACE" -o wide
    
    # Count pod states
    local running=$(kubectl get pods -n "$NAMESPACE" -o json | jq '[.items[] | select(.status.phase=="Running")] | length')
    local pending=$(kubectl get pods -n "$NAMESPACE" -o json | jq '[.items[] | select(.status.phase=="Pending")] | length')
    local failed=$(kubectl get pods -n "$NAMESPACE" -o json | jq '[.items[] | select(.status.phase=="Failed")] | length')
    
    success "✓ Running: $running, Pending: $pending, Failed: $failed"
    
    if [ "$failed" -gt 0 ]; then
        error "✗ Failed pods detected"
        kubectl get pods -n "$NAMESPACE" -o json | jq '.items[] | select(.status.phase=="Failed")'
        return 1
    fi
}

# Check StatefulSets and Deployments
check_workloads() {
    info "=== Workload Status ==="
    
    # StatefulSets
    info "StatefulSets:"
    kubectl get statefulsets -n "$NAMESPACE" -o wide
    local sts_ready=$(kubectl get statefulsets -n "$NAMESPACE" -o json | jq '[.items[] | select(.status.readyReplicas == .status.replicas)] | length')
    local sts_total=$(kubectl get statefulsets -n "$NAMESPACE" -o json | jq '[.items[]] | length')
    
    if [ "$sts_ready" == "$sts_total" ]; then
        success "✓ All StatefulSets ready ($sts_ready/$sts_total)"
    else
        warn "⚠ Some StatefulSets not ready ($sts_ready/$sts_total)"
    fi
    
    # Deployments
    info "Deployments:"
    kubectl get deployments -n "$NAMESPACE" -o wide
    local dep_ready=$(kubectl get deployments -n "$NAMESPACE" -o json | jq '[.items[] | select(.status.readyReplicas == .status.replicas)] | length')
    local dep_total=$(kubectl get deployments -n "$NAMESPACE" -o json | jq '[.items[]] | length')
    
    if [ "$dep_ready" == "$dep_total" ]; then
        success "✓ All Deployments ready ($dep_ready/$dep_total)"
    else
        warn "⚠ Some Deployments not ready ($dep_ready/$dep_total)"
    fi
}

# Check persistent volumes
check_storage() {
    info "=== Storage Status ==="
    
    kubectl get pvc -n "$NAMESPACE" -o wide
    
    # Check PVC status
    local bound=$(kubectl get pvc -n "$NAMESPACE" -o json | jq '[.items[] | select(.status.phase=="Bound")] | length')
    local total=$(kubectl get pvc -n "$NAMESPACE" -o json | jq '[.items[]] | length')
    
    if [ "$bound" == "$total" ]; then
        success "✓ All PVCs bound ($bound/$total)"
    else
        warn "⚠ Some PVCs not bound ($bound/$total)"
    fi
    
    # Check storage utilization
    info "Storage Utilization:"
    kubectl exec -n "$NAMESPACE" prometheus-0 -- sh -c 'du -h /prometheus 2>/dev/null || echo "N/A"'
}

# Check services and endpoints
check_services() {
    info "=== Services and Endpoints ==="
    
    kubectl get services -n "$NAMESPACE" -o wide
    
    info "Endpoints:"
    kubectl get endpoints -n "$NAMESPACE" -o wide
}

# Check Prometheus metrics
check_prometheus() {
    info "=== Prometheus Health ==="
    
    # Port forward if needed
    local prom_pod=$(kubectl get pods -n "$NAMESPACE" -l app=prometheus -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [ -z "$prom_pod" ]; then
        warn "⚠ No Prometheus pod found"
        return 0
    fi
    
    # Check readiness
    if kubectl exec -n "$NAMESPACE" "$prom_pod" -- wget -q -O - http://localhost:9090/-/ready &> /dev/null; then
        success "✓ Prometheus ready"
    else
        error "✗ Prometheus not ready"
        return 1
    fi
    
    # Check scrape targets
    if [ -n "$VERBOSE" ]; then
        info "Active scrape targets:"
        kubectl exec -n "$NAMESPACE" "$prom_pod" -- \
            wget -q -O - http://localhost:9090/api/v1/targets 2>/dev/null | jq '.data.activeTargets[] | select(.health=="up") | .labels.job' 2>/dev/null || echo "N/A"
    fi
}

# Check Grafana
check_grafana() {
    info "=== Grafana Health ==="
    
    local grafana_pod=$(kubectl get pods -n "$NAMESPACE" -l app=grafana -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [ -z "$grafana_pod" ]; then
        warn "⚠ No Grafana pod found"
        return 0
    fi
    
    # Check health
    if kubectl exec -n "$NAMESPACE" "$grafana_pod" -- \
        wget -q -O - http://localhost:3000/api/health 2>/dev/null | grep -q "ok"; then
        success "✓ Grafana healthy"
    else
        warn "⚠ Grafana health check inconclusive"
    fi
}

# Check AlertManager
check_alertmanager() {
    info "=== AlertManager Health ==="
    
    local am_pod=$(kubectl get pods -n "$NAMESPACE" -l app=alertmanager -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [ -z "$am_pod" ]; then
        warn "⚠ No AlertManager pod found"
        return 0
    fi
    
    # Check readiness
    if kubectl exec -n "$NAMESPACE" "$am_pod" -- wget -q -O - http://localhost:9093/-/ready &> /dev/null; then
        success "✓ AlertManager ready"
    else
        error "✗ AlertManager not ready"
        return 1
    fi
    
    # Check cluster status
    if [ -n "$VERBOSE" ]; then
        info "AlertManager cluster status:"
        kubectl exec -n "$NAMESPACE" "$am_pod" -- \
            wget -q -O - http://localhost:9093/api/v1/status 2>/dev/null | jq '.data.cluster' || echo "N/A"
    fi
}

# Check logs
check_logs() {
    if [ -z "$VERBOSE" ]; then
        return 0
    fi
    
    info "=== Recent Errors in Logs ==="
    
    for pod in $(kubectl get pods -n "$NAMESPACE" -o name); do
        info "Errors in $pod:"
        kubectl logs -n "$NAMESPACE" "$pod" --tail=20 2>/dev/null | grep -i "error\|warn\|fatal" | head -5 || echo "No errors found"
    done
}

# Generate summary
summary() {
    info "=== Summary ==="
    
    local total_pods=$(kubectl get pods -n "$NAMESPACE" -o json | jq '[.items[]] | length')
    local ready_pods=$(kubectl get pods -n "$NAMESPACE" -o json | jq '[.items[] | select(.status.conditions[] | select(.type=="Ready" and .status=="True"))] | length')
    local pvc_bound=$(kubectl get pvc -n "$NAMESPACE" -o json | jq '[.items[] | select(.status.phase=="Bound")] | length')
    
    if [ "$ready_pods" == "$total_pods" ] && [ "$pvc_bound" -gt 0 ]; then
        success "✓ Monitoring stack is HEALTHY"
        success "  - Pods: $ready_pods/$total_pods ready"
        success "  - PVCs: $pvc_bound bound"
    else
        warn "⚠ Monitoring stack needs attention"
        warn "  - Pods: $ready_pods/$total_pods ready"
        warn "  - PVCs: $pvc_bound bound"
    fi
}

# Main execution
main() {
    info "╔════════════════════════════════════════════════╗"
    info "║   Neurectomy Monitoring Stack Health Check     ║"
    info "║   Namespace: $NAMESPACE                             ║"
    if [ -n "$VERBOSE" ]; then
        info "║   Mode: VERBOSE                                ║"
    fi
    info "╚════════════════════════════════════════════════╝"
    
    check_cluster && \
    check_namespace && \
    check_workloads && \
    check_pods && \
    check_storage && \
    check_services && \
    check_prometheus && \
    check_grafana && \
    check_alertmanager && \
    check_logs && \
    summary
}

main
