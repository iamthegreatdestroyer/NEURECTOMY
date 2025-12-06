#!/bin/bash
# scripts/dr/smoke-tests.sh
# Post-failover smoke tests for NEURECTOMY
#
# Usage: ./smoke-tests.sh [--verbose]

set -euo pipefail

VERBOSE=${1:-}
FAILED=0
PASSED=0

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_test() {
    local name="$1"
    local status="$2"
    local details="${3:-}"
    
    if [ "$status" = "PASS" ]; then
        echo -e "${GREEN}✓${NC} $name"
        ((PASSED++))
    else
        echo -e "${RED}✗${NC} $name"
        [ -n "$details" ] && echo -e "  ${RED}→${NC} $details"
        ((FAILED++))
    fi
}

echo "=========================================="
echo "NEURECTOMY DR Smoke Tests"
echo "=========================================="
echo "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

# Test 1: Kubernetes Connectivity
echo "## Kubernetes Cluster"
echo "---------------------"

if kubectl cluster-info > /dev/null 2>&1; then
    log_test "Cluster connectivity" "PASS"
else
    log_test "Cluster connectivity" "FAIL" "Cannot connect to cluster"
fi

NODE_COUNT=$(kubectl get nodes --no-headers 2>/dev/null | grep -c "Ready" || echo "0")
if [ "$NODE_COUNT" -ge 2 ]; then
    log_test "Node availability ($NODE_COUNT nodes)" "PASS"
else
    log_test "Node availability ($NODE_COUNT nodes)" "FAIL" "Expected at least 2 ready nodes"
fi

# Test 2: Core Services
echo ""
echo "## Core Services"
echo "----------------"

# ML Service
ML_READY=$(kubectl get deployment ml-service -n neurectomy -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
ML_DESIRED=$(kubectl get deployment ml-service -n neurectomy -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")
if [ "$ML_READY" = "$ML_DESIRED" ] && [ "$ML_READY" != "0" ]; then
    log_test "ML Service ($ML_READY/$ML_DESIRED replicas)" "PASS"
else
    log_test "ML Service ($ML_READY/$ML_DESIRED replicas)" "FAIL"
fi

# MLflow
MLFLOW_READY=$(kubectl get deployment mlflow -n neurectomy -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
MLFLOW_DESIRED=$(kubectl get deployment mlflow -n neurectomy -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")
if [ "$MLFLOW_READY" = "$MLFLOW_DESIRED" ] && [ "$MLFLOW_READY" != "0" ]; then
    log_test "MLflow ($MLFLOW_READY/$MLFLOW_DESIRED replicas)" "PASS"
else
    log_test "MLflow ($MLFLOW_READY/$MLFLOW_DESIRED replicas)" "FAIL"
fi

# PostgreSQL
PG_STATUS=$(kubectl get pods -n neurectomy -l app=postgresql -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "Unknown")
if [ "$PG_STATUS" = "Running" ]; then
    log_test "PostgreSQL" "PASS"
else
    log_test "PostgreSQL" "FAIL" "Status: $PG_STATUS"
fi

# Test 3: Database Connectivity
echo ""
echo "## Database Connectivity"
echo "------------------------"

DB_CHECK=$(kubectl exec -n neurectomy -l app=postgresql -- psql -U postgres -t -c "SELECT 1;" 2>/dev/null || echo "failed")
if [ "$DB_CHECK" = " 1" ] || [ "$DB_CHECK" = "1" ]; then
    log_test "PostgreSQL connection" "PASS"
else
    log_test "PostgreSQL connection" "FAIL" "Query returned: $DB_CHECK"
fi

# Check data integrity
TABLE_COUNT=$(kubectl exec -n neurectomy -l app=postgresql -- psql -U postgres -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';" 2>/dev/null | tr -d ' ' || echo "0")
if [ "$TABLE_COUNT" -gt 0 ]; then
    log_test "Database tables ($TABLE_COUNT tables)" "PASS"
else
    log_test "Database tables" "FAIL" "No tables found"
fi

# Test 4: Storage
echo ""
echo "## Storage"
echo "---------"

PVC_BOUND=$(kubectl get pvc -n neurectomy --no-headers 2>/dev/null | grep -c "Bound" || echo "0")
PVC_TOTAL=$(kubectl get pvc -n neurectomy --no-headers 2>/dev/null | wc -l || echo "0")
if [ "$PVC_BOUND" = "$PVC_TOTAL" ] && [ "$PVC_TOTAL" != "0" ]; then
    log_test "Persistent Volumes ($PVC_BOUND/$PVC_TOTAL bound)" "PASS"
else
    log_test "Persistent Volumes ($PVC_BOUND/$PVC_TOTAL bound)" "FAIL"
fi

# Test 5: Networking
echo ""
echo "## Networking"
echo "-------------"

# Check services have endpoints
SVC_COUNT=$(kubectl get services -n neurectomy --no-headers 2>/dev/null | wc -l || echo "0")
ENDPOINTS_OK=$(kubectl get endpoints -n neurectomy --no-headers 2>/dev/null | awk '{if ($2 != "<none>") print}' | wc -l || echo "0")
if [ "$ENDPOINTS_OK" -ge 1 ]; then
    log_test "Service endpoints ($ENDPOINTS_OK/$SVC_COUNT with backends)" "PASS"
else
    log_test "Service endpoints" "FAIL" "No endpoints have backends"
fi

# Check ingress
INGRESS_IP=$(kubectl get ingress -n neurectomy -o jsonpath='{.items[0].status.loadBalancer.ingress[0].hostname}' 2>/dev/null || kubectl get ingress -n neurectomy -o jsonpath='{.items[0].status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
if [ -n "$INGRESS_IP" ]; then
    log_test "Ingress load balancer" "PASS"
else
    log_test "Ingress load balancer" "FAIL" "No external IP/hostname"
fi

# Test 6: API Health
echo ""
echo "## API Health"
echo "-------------"

# Internal health check
HEALTH_INTERNAL=$(kubectl exec -n neurectomy deploy/ml-service -- curl -s localhost:8000/health 2>/dev/null | grep -o '"status":"[^"]*"' | head -1 || echo "")
if echo "$HEALTH_INTERNAL" | grep -q "healthy\|ok\|up"; then
    log_test "Internal health endpoint" "PASS"
else
    log_test "Internal health endpoint" "FAIL" "Response: $HEALTH_INTERNAL"
fi

# External health check (if ingress is available)
if [ -n "$INGRESS_IP" ]; then
    HEALTH_EXTERNAL=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 10 "https://${INGRESS_IP}/health" -k 2>/dev/null || echo "000")
    if [ "$HEALTH_EXTERNAL" = "200" ]; then
        log_test "External health endpoint" "PASS"
    else
        log_test "External health endpoint" "FAIL" "HTTP status: $HEALTH_EXTERNAL"
    fi
fi

# Test 7: Monitoring
echo ""
echo "## Monitoring"
echo "-------------"

PROMETHEUS_STATUS=$(kubectl get pods -n monitoring -l app=prometheus -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "Unknown")
if [ "$PROMETHEUS_STATUS" = "Running" ]; then
    log_test "Prometheus" "PASS"
else
    log_test "Prometheus" "FAIL" "Status: $PROMETHEUS_STATUS"
fi

GRAFANA_STATUS=$(kubectl get pods -n monitoring -l app=grafana -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "Unknown")
if [ "$GRAFANA_STATUS" = "Running" ]; then
    log_test "Grafana" "PASS"
else
    log_test "Grafana" "FAIL" "Status: $GRAFANA_STATUS"
fi

# Test 8: Secrets
echo ""
echo "## Secrets & Configuration"
echo "--------------------------"

SECRET_COUNT=$(kubectl get secrets -n neurectomy --no-headers 2>/dev/null | wc -l || echo "0")
if [ "$SECRET_COUNT" -ge 3 ]; then
    log_test "Secrets present ($SECRET_COUNT secrets)" "PASS"
else
    log_test "Secrets present ($SECRET_COUNT secrets)" "FAIL" "Expected at least 3 secrets"
fi

CONFIGMAP_COUNT=$(kubectl get configmaps -n neurectomy --no-headers 2>/dev/null | wc -l || echo "0")
if [ "$CONFIGMAP_COUNT" -ge 1 ]; then
    log_test "ConfigMaps present ($CONFIGMAP_COUNT configmaps)" "PASS"
else
    log_test "ConfigMaps present" "FAIL"
fi

# Test 9: Velero
echo ""
echo "## Backup System"
echo "----------------"

VELERO_STATUS=$(kubectl get pods -n velero -l app.kubernetes.io/name=velero -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "Unknown")
if [ "$VELERO_STATUS" = "Running" ]; then
    log_test "Velero" "PASS"
else
    log_test "Velero" "FAIL" "Status: $VELERO_STATUS"
fi

BACKUP_COUNT=$(velero backup get --no-headers 2>/dev/null | wc -l || echo "0")
if [ "$BACKUP_COUNT" -ge 1 ]; then
    log_test "Backups available ($BACKUP_COUNT backups)" "PASS"
else
    log_test "Backups available" "FAIL" "No backups found"
fi

# Summary
echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo "Total:  $((PASSED + FAILED))"
echo ""

if [ "$FAILED" -gt 0 ]; then
    echo -e "${RED}Some tests failed. Review the results above.${NC}"
    exit 1
else
    echo -e "${GREEN}All smoke tests passed!${NC}"
    exit 0
fi
