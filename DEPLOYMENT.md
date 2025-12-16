# Neurectomy Deployment Guide

## Overview

This guide covers deploying Neurectomy using Docker, Docker Compose, and Kubernetes.

---

## Prerequisites

### Local Development

- Docker 20.10+
- Docker Compose 2.0+
- Python 3.11+

### Kubernetes Deployment

- kubectl 1.24+
- Kubernetes cluster 1.24+
- Helm 3.0+ (optional)

---

## Quick Start with Docker Compose

### 1. Build and Start Services

```bash
# Build the Neurectomy API image
docker-compose build

# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

### 2. Verify Deployment

```bash
# Check API health
curl http://localhost:8000/health

# Access interactive docs
open http://localhost:8000/docs

# Check Prometheus
open http://localhost:9090

# Check Grafana
open http://localhost:3000
# Default credentials: admin/admin
```

### 3. View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f neurectomy-api
```

### 4. Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

---

## Kubernetes Deployment

### 1. Create Namespace

```bash
kubectl create namespace neurectomy
```

### 2. Deploy Configuration

```bash
# Apply all manifests
kubectl apply -f deploy/k8s/

# Or apply individually
kubectl apply -f deploy/k8s/configmap.yaml
kubectl apply -f deploy/k8s/deployment.yaml
```

### 3. Verify Deployment

```bash
# Check pods
kubectl get pods -n neurectomy

# Check services
kubectl get svc -n neurectomy

# Check HPA
kubectl get hpa -n neurectomy

# View logs
kubectl logs -f deployment/neurectomy-api -n neurectomy
```

### 4. Port Forward (for testing)

```bash
# Forward API port
kubectl port-forward svc/neurectomy-api 8000:80 -n neurectomy

# Access API
curl http://localhost:8000/health
```

### 5. Scale Deployment

```bash
# Manual scaling
kubectl scale deployment neurectomy-api --replicas=5 -n neurectomy

# HPA will auto-scale based on CPU/memory
kubectl get hpa -n neurectomy -w
```

---

## Docker Build Options

### Production Build

```bash
docker build -t neurectomy:latest .
```

### Multi-platform Build

```bash
docker buildx build --platform linux/amd64,linux/arm64 -t neurectomy:latest .
```

### Build with Cache

```bash
docker build --cache-from neurectomy:latest -t neurectomy:latest .
```

---

## Environment Variables

### API Configuration

| Variable           | Default | Description             |
| ------------------ | ------- | ----------------------- |
| `PYTHONUNBUFFERED` | `1`     | Python output buffering |
| `LOG_LEVEL`        | `INFO`  | Logging level           |
| `API_PORT`         | `8000`  | API server port         |

### Example `.env` file:

```bash
# API Settings
LOG_LEVEL=INFO
API_PORT=8000

# Performance
WORKERS=4
MAX_CONNECTIONS=1000

# Features
ENABLE_COMPRESSION=true
ENABLE_RSU=true
```

---

## Monitoring & Observability

### Prometheus Metrics

Available at: `http://localhost:9090`

**Key Metrics:**

- `neurectomy_inference_latency_ms`
- `neurectomy_compression_ratio`
- `neurectomy_cache_hit_rate`
- `neurectomy_agent_tasks_total`

### Grafana Dashboards

Available at: `http://localhost:3000`

**Dashboards:**

1. Neurectomy Overview
2. API Performance
3. Agent Metrics
4. System Resources

---

## CI/CD Pipeline

### GitHub Actions Workflow

Triggered on:

- Push to `main` or `develop`
- Pull requests

**Stages:**

1. **Lint** - Code quality checks
2. **Test** - Run test suite
3. **Build** - Build Docker image
4. **Deploy** - Deploy to Kubernetes (main only)

### Secrets Required

```yaml
GITHUB_TOKEN: # Automatic
KUBE_CONFIG: # Base64 encoded kubeconfig
CODECOV_TOKEN: # Optional for coverage
```

---

## Production Considerations

### Security

1. **Use secrets management:**

   ```bash
   kubectl create secret generic neurectomy-secrets \
     --from-literal=api-key=<your-key> \
     -n neurectomy
   ```

2. **Enable TLS:**
   - Configure Ingress with cert-manager
   - Use Let's Encrypt for certificates

3. **Network policies:**
   ```bash
   kubectl apply -f deploy/k8s/network-policy.yaml
   ```

### Performance

1. **Resource limits:**
   - Adjust CPU/memory in `deployment.yaml`
   - Monitor with `kubectl top pods -n neurectomy`

2. **Horizontal scaling:**
   - HPA configured for 3-10 replicas
   - Scales based on CPU (70%) and memory (80%)

3. **Caching:**
   - Redis for API response caching
   - Configure TTL based on use case

### Reliability

1. **Health checks:**
   - Liveness: `/live`
   - Readiness: `/ready`
   - Full health: `/health`

2. **Rolling updates:**

   ```bash
   kubectl set image deployment/neurectomy-api \
     neurectomy-api=neurectomy:v2.0.0 \
     -n neurectomy
   ```

3. **Rollback:**
   ```bash
   kubectl rollout undo deployment/neurectomy-api -n neurectomy
   ```

---

## Troubleshooting

### Container Issues

```bash
# Check container logs
docker logs neurectomy-api

# Inspect container
docker inspect neurectomy-api

# Execute command in container
docker exec -it neurectomy-api bash
```

### Kubernetes Issues

```bash
# Describe pod
kubectl describe pod <pod-name> -n neurectomy

# Check events
kubectl get events -n neurectomy --sort-by='.lastTimestamp'

# Debug pod
kubectl debug pod/<pod-name> -n neurectomy -it --image=busybox
```

### Common Problems

1. **Image pull errors:**

   ```bash
   # Check image exists
   docker images | grep neurectomy

   # Re-tag and push
   docker tag neurectomy:latest ghcr.io/user/neurectomy:latest
   docker push ghcr.io/user/neurectomy:latest
   ```

2. **Health check failures:**

   ```bash
   # Test health endpoint
   kubectl port-forward svc/neurectomy-api 8000:80 -n neurectomy
   curl http://localhost:8000/health
   ```

3. **Resource constraints:**

   ```bash
   # Check resource usage
   kubectl top pods -n neurectomy

   # Increase limits in deployment.yaml
   ```

---

## Cleanup

### Docker Compose

```bash
# Stop and remove containers
docker-compose down

# Remove volumes
docker-compose down -v

# Remove images
docker rmi neurectomy:latest
```

### Kubernetes

```bash
# Delete all resources
kubectl delete namespace neurectomy

# Or delete specific resources
kubectl delete -f deploy/k8s/
```

---

## Next Steps

1. **Configure monitoring alerts** in Prometheus
2. **Set up log aggregation** (ELK, Loki)
3. **Enable distributed tracing** (Jaeger, Tempo)
4. **Implement backup strategy** for data volumes
5. **Set up disaster recovery** procedures

---

For more information, see:

- [API Documentation](http://localhost:8000/docs)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Documentation](https://docs.docker.com/)
