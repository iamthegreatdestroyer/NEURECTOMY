# 🐳 Container Command

> **Docker/Kubernetes Orchestration**

## Purpose

Native, first-class container and orchestration capabilities that treat agents as deployable microservices with full lifecycle management.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     CONTAINER COMMAND                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              DOCKER INTEGRATION LAYER                     │   │
│  │  ┌────────────┬────────────┬────────────┬─────────────┐  │   │
│  │  │ Dockerfile │ Image      │ Container  │ Registry    │  │   │
│  │  │ Editor     │ Builder    │ Manager    │ Hub         │  │   │
│  │  └────────────┴────────────┴────────────┴─────────────┘  │   │
│  │                                                           │   │
│  │  • Visual Dockerfile Designer (drag-and-drop layers)     │   │
│  │  • One-click agent containerization                       │   │
│  │  • Multi-stage build optimization                         │   │
│  │  • Image security scanning (Trivy/Snyk integration)       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              KUBERNETES ORCHESTRATOR                      │   │
│  │  ┌─────────────────────────────────────────────────────┐ │   │
│  │  │         CLUSTER TOPOLOGY VIEW (3D)                  │ │   │
│  │  │   ┌─────────┐   ┌─────────┐   ┌─────────┐          │ │   │
│  │  │   │  Node   │───│  Node   │───│  Node   │          │ │   │
│  │  │   │ Agent-1 │   │ Agent-2 │   │ Agent-3 │          │ │   │
│  │  │   └─────────┘   └─────────┘   └─────────┘          │ │   │
│  │  └─────────────────────────────────────────────────────┘ │   │
│  │                                                           │   │
│  │  Features:                                                │   │
│  │  • Visual Helm chart designer                             │   │
│  │  • Auto-scaling policies for agent swarms                 │   │
│  │  • Service mesh configuration (Istio/Linkerd)            │   │
│  │  • GPU scheduling for ML-intensive agents                 │   │
│  │  • Canary/Blue-Green deployment strategies               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              DEPLOYMENT PIPELINES                         │   │
│  │                                                           │   │
│  │  [Local Dev] → [Sandbox] → [Staging] → [Production]      │   │
│  │       │            │           │            │             │   │
│  │    Docker      MicroVM      K8s Test    K8s Prod         │   │
│  │    Compose     Firecracker  Cluster     Cluster          │   │
│  │                                                           │   │
│  │  • GitOps integration (ArgoCD, Flux)                      │   │
│  │  • Automated rollback on agent failure                    │   │
│  │  • Multi-cloud deployment (AWS EKS, GKE, AKS, self-hosted)│   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              OBSERVABILITY DASHBOARD                      │   │
│  │                                                           │   │
│  │  ┌─────────────┬─────────────┬─────────────────────────┐ │   │
│  │  │ Metrics     │ Logs        │ Traces                  │ │   │
│  │  │ (Prometheus)│ (Loki)      │ (Jaeger/Tempo)          │ │   │
│  │  └─────────────┴─────────────┴─────────────────────────┘ │   │
│  │                                                           │   │
│  │  • Real-time agent health monitoring                      │   │
│  │  • Resource consumption analytics                         │   │
│  │  • Distributed tracing across agent communications       │   │
│  │  • Cost optimization recommendations                      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Feature Breakdown

| Feature                            | Description                                                           |
| ---------------------------------- | --------------------------------------------------------------------- |
| **One-Click Containerization**     | Automatically generate optimized Dockerfiles for any agent            |
| **Visual Dockerfile Designer**     | Drag-and-drop layer composition with best-practice suggestions        |
| **3D Cluster Topology**            | Navigate Kubernetes clusters in 3D space (inspired by Lens)           |
| **Agent Pod Templates**            | Pre-configured Kubernetes manifests for Elite Agent patterns          |
| **GPU Workload Scheduler**         | Intelligent GPU allocation for ML-heavy agents                        |
| **Sandbox-to-Production Pipeline** | Graduated deployment with automatic testing gates                     |
| **Service Mesh Integration**       | Built-in Istio/Linkerd configuration for agent-to-agent communication |
| **Secrets Management**             | Integrated Vault/SOPS for secure credential handling                  |
| **Multi-Cloud Dashboard**          | Unified view across AWS, GCP, Azure, and self-hosted clusters         |
| **Carbon Footprint Tracker**       | Sustainability metrics for container workloads                        |

---

## Docker Integration Layer

### Visual Dockerfile Designer

Create Dockerfiles visually with drag-and-drop:

```
┌─────────────────────────────────────────────────────────────┐
│  DOCKERFILE DESIGNER                                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  AVAILABLE LAYERS          YOUR DOCKERFILE                  │
│  ────────────────          ────────────────                 │
│  ┌──────────────┐          ┌──────────────────────────────┐ │
│  │ Base Images  │          │ FROM python:3.11-slim       │ │
│  │ • Python     │    ──►   │                              │ │
│  │ • Node.js    │          │ WORKDIR /app                 │ │
│  │ • Go         │          │                              │ │
│  │ • Rust       │          │ COPY requirements.txt .      │ │
│  └──────────────┘          │                              │ │
│  ┌──────────────┐          │ RUN pip install --no-cache   │ │
│  │ Commands     │          │     -r requirements.txt      │ │
│  │ • COPY       │          │                              │ │
│  │ • RUN        │          │ COPY . .                     │ │
│  │ • ENV        │          │                              │ │
│  │ • EXPOSE     │          │ CMD ["python", "agent.py"]   │ │
│  └──────────────┘          └──────────────────────────────┘ │
│                                                              │
│  [Optimize] [Scan Security] [Build] [Push to Registry]      │
└─────────────────────────────────────────────────────────────┘
```

### Container Management

- **Build** - Multi-stage optimized builds
- **Run** - Local container execution with hot-reload
- **Push** - Push to Docker Hub, ECR, GCR, ACR
- **Pull** - Fetch images with integrity verification

---

## Kubernetes Orchestrator

### 3D Cluster Topology View

Navigate your cluster in 3D space:

```
        ┌─────────────────────────────────────────────────────┐
        │              3D CLUSTER VIEW                         │
        │                                                      │
        │      ┌───────────┐                                  │
        │      │   Node 1  │                                  │
        │      │  ┌─────┐  │                                  │
        │      │  │Pod A│  │      ┌───────────┐              │
        │      │  └─────┘  │──────│   Node 2  │              │
        │      │  ┌─────┐  │      │  ┌─────┐  │              │
        │      │  │Pod B│  │      │  │Pod C│  │              │
        │      └───────────┘      │  └─────┘  │              │
        │                         └───────────┘              │
        │                              │                      │
        │                         ┌────┴────┐                │
        │                         │  Node 3 │                │
        │                         │ ┌─────┐ │                │
        │                         │ │Pod D│ │                │
        │                         │ └─────┘ │                │
        │                         └─────────┘                │
        │                                                      │
        │  [Zoom] [Rotate] [Filter] [Pod Details] [Logs]      │
        └─────────────────────────────────────────────────────┘
```

### Deployment Strategies

| Strategy           | Description                                |
| ------------------ | ------------------------------------------ |
| **Rolling Update** | Gradual replacement with zero downtime     |
| **Canary**         | Route percentage of traffic to new version |
| **Blue/Green**     | Instant switch between environments        |
| **A/B Testing**    | Route based on conditions/headers          |

---

## Deployment Pipelines

### Pipeline Stages

```
[Local Dev] → [Sandbox] → [Staging] → [Production]
     │            │           │            │
  Docker      MicroVM      K8s Test    K8s Prod
  Compose     Firecracker  Cluster     Cluster
```

### GitOps Integration

- **ArgoCD** - Declarative GitOps CD
- **Flux** - Continuous delivery for Kubernetes
- **Auto-sync** - Automatic deployment on git push
- **Auto-rollback** - Revert on health check failure

### Multi-Cloud Support

| Provider     | Service     | Status          |
| ------------ | ----------- | --------------- |
| AWS          | EKS         | ✅ Full Support |
| Google Cloud | GKE         | ✅ Full Support |
| Azure        | AKS         | ✅ Full Support |
| Self-Hosted  | kubeadm/k3s | ✅ Full Support |

---

## Observability Dashboard

### Metrics (Prometheus)

- CPU/Memory utilization
- Request latency
- Error rates
- Custom agent metrics

### Logs (Loki)

- Centralized log aggregation
- Full-text search
- Log correlation

### Traces (Jaeger/Tempo)

- Distributed tracing
- Request flow visualization
- Latency breakdown

### Sample Dashboard

```
┌─────────────────────────────────────────────────────────────┐
│  AGENT CLUSTER HEALTH                                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ACTIVE AGENTS: 12/12 ✅     PODS: 45 Running               │
│  AVG LATENCY: 23ms           ERROR RATE: 0.01%              │
│                                                              │
│  CPU USAGE                   MEMORY USAGE                   │
│  ██████████░░░░░░ 62%        ███████░░░░░░░░░ 45%           │
│                                                              │
│  REQUESTS/SEC                AGENT RESPONSES                │
│  ┌──────────────────────┐    ┌──────────────────────┐       │
│  │    ╱╲    ╱╲          │    │ Success: 99.9%       │       │
│  │   ╱  ╲  ╱  ╲   ╱╲    │    │ Timeout: 0.05%       │       │
│  │  ╱    ╲╱    ╲ ╱  ╲   │    │ Error: 0.05%         │       │
│  │ ╱            ╲    ╲  │    └──────────────────────┘       │
│  └──────────────────────┘                                   │
│                                                              │
│  [View All Metrics] [Configure Alerts] [Export]             │
└─────────────────────────────────────────────────────────────┘
```

---

## Usage Examples

### Containerizing an Agent

```python
from neurectomy.container import AgentContainer

# Define agent container
container = AgentContainer(
    name="elite-sentinel",
    base_image="python:3.11-slim",
    requirements="requirements.txt",
    entrypoint="python agent.py"
)

# Build with optimization
container.build(
    multi_stage=True,
    cache_layers=True,
    security_scan=True
)

# Push to registry
container.push("registry.example.com/elite-sentinel:v1.0.0")
```

### Deploying to Kubernetes

```python
from neurectomy.container import K8sDeployer

# Configure deployment
deployer = K8sDeployer(
    cluster="production-cluster",
    namespace="elite-agents"
)

# Deploy agent
deployer.deploy(
    image="registry.example.com/elite-sentinel:v1.0.0",
    replicas=3,
    strategy="canary",
    canary_percent=10,
    gpu_required=True
)

# Monitor rollout
deployer.watch_rollout()
```

### Auto-scaling Configuration

```yaml
# agent-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: elite-sentinel-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: elite-sentinel
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: agent_requests_per_second
        target:
          type: AverageValue
          averageValue: 100
```

---

## Integration Points

### With Dimensional Forge

- Export container topology as 3D visualization
- View pod relationships graphically

### With Intelligence Foundry

- GPU scheduling for ML training containers
- Model serving containers

### With Legal Fortress

- Container image provenance tracking
- SBOM generation for containers
- Vulnerability scanning integration

---

## Related Documentation

- [Architecture Overview](../../architecture/README.md)
- [Dimensional Forge](../dimensional-forge/README.md)
- [Experimentation Engine](../experimentation-engine/README.md)
- [Technical Stack](../../technical/stack.md)
