#!/bin/bash
# Install Prometheus and Grafana monitoring stack for Neurectomy
# This script deploys the complete monitoring infrastructure

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  NEURECTOMY MONITORING STACK INSTALLATION                     ║"
echo "║  Prometheus + Grafana + AlertManager                          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo "Checking prerequisites..."
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}✗ kubectl not found. Please install kubectl.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ kubectl found${NC}"

if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}✗ Cannot connect to Kubernetes cluster${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Connected to Kubernetes cluster${NC}"
echo ""

# Create monitoring namespace
echo "Creating monitoring namespace..."
kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -
echo -e "${GREEN}✓ Namespace created${NC}"
echo ""

# Update Grafana credentials
echo "Setting Grafana admin password..."
read -s -p "Enter Grafana admin password (default: neurectomy-admin): " GRAFANA_PASSWORD
GRAFANA_PASSWORD=${GRAFANA_PASSWORD:-neurectomy-admin}
echo ""

# Update the secret
kubectl patch secret grafana-credentials -n monitoring -p \
  "{\"stringData\": {\"admin-password\": \"${GRAFANA_PASSWORD}\"}}" \
  --type merge 2>/dev/null || \
kubectl create secret generic grafana-credentials -n monitoring \
  --from-literal=admin-password="${GRAFANA_PASSWORD}" \
  --dry-run=client -o yaml | kubectl apply -f -
echo -e "${GREEN}✓ Grafana credentials configured${NC}"
echo ""

# Deploy Prometheus
echo "Deploying Prometheus..."
kubectl apply -f infrastructure/kubernetes/monitoring/prometheus-deployment.yaml
echo -e "${GREEN}✓ Prometheus deployment created${NC}"
echo ""

# Deploy Grafana
echo "Deploying Grafana..."
kubectl apply -f infrastructure/kubernetes/monitoring/grafana-deployment.yaml
echo -e "${GREEN}✓ Grafana deployment created${NC}"
echo ""

# Wait for Prometheus
echo "Waiting for Prometheus deployment to be ready (this may take 1-2 minutes)..."
kubectl wait --for=condition=available --timeout=300s \
  deployment/prometheus -n monitoring 2>/dev/null && \
echo -e "${GREEN}✓ Prometheus is ready${NC}" || \
echo -e "${YELLOW}⚠ Prometheus deployment pending, check with: kubectl get pods -n monitoring${NC}"
echo ""

# Wait for Grafana
echo "Waiting for Grafana deployment to be ready (this may take 1-2 minutes)..."
kubectl wait --for=condition=available --timeout=300s \
  deployment/grafana -n monitoring 2>/dev/null && \
echo -e "${GREEN}✓ Grafana is ready${NC}" || \
echo -e "${YELLOW}⚠ Grafana deployment pending, check with: kubectl get pods -n monitoring${NC}"
echo ""

# Get service information
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  INSTALLATION COMPLETE                                         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

echo "Monitoring Stack Status:"
kubectl get deployments -n monitoring
echo ""

echo "Service Information:"
echo ""
echo "Prometheus:"
echo "  kubectl port-forward -n monitoring svc/prometheus 9090:9090"
echo "  Then visit: http://localhost:9090"
echo ""
echo "Grafana:"
GRAFANA_SERVICE=$(kubectl get svc grafana -n monitoring -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
if [ "$GRAFANA_SERVICE" != "pending" ]; then
    echo "  External IP: http://$GRAFANA_SERVICE"
else
    echo "  kubectl port-forward -n monitoring svc/grafana 3000:80"
    echo "  Then visit: http://localhost:3000"
fi
echo "  Username: admin"
echo "  Password: (as configured above)"
echo ""

echo "Next Steps:"
echo "1. Access Grafana and configure dashboards"
echo "2. Deploy Neurectomy metrics: neurectomy/monitoring/metrics.py"
echo "3. Add service annotations for Prometheus scraping"
echo "4. Import Grafana dashboards"
echo ""

echo "Useful Commands:"
echo "  # View Prometheus configuration"
echo "  kubectl get configmap prometheus-config -n monitoring -o yaml"
echo ""
echo "  # View Prometheus targets"
echo "  kubectl logs -n monitoring deployment/prometheus -f"
echo ""
echo "  # Delete monitoring stack (if needed)"
echo "  kubectl delete namespace monitoring"
echo ""

echo -e "${GREEN}✓ Monitoring stack installation complete!${NC}"
