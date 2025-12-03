# NEURECTOMY Technical Stack

> **Recommended Technologies and Implementation Details**

## Overview

This document outlines the recommended technical stack for implementing NEURECTOMY IDE, organized by architectural layer and functional area.

---

## Complete Stack Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                  NEURECTOMY TECHNICAL STACK                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  FRONTEND                                                        │
│  ─────────────────────────────────────────────────────────────  │
│  • Framework: React 19 + TypeScript 5.5                         │
│  • 3D Rendering: Three.js + React Three Fiber + WebGPU          │
│  • Code Editor: Monaco Editor (VS Code engine)                  │
│  • State Management: Zustand + React Query                      │
│  • UI Components: Radix UI + Tailwind CSS                       │
│  • Desktop: Tauri 2.0 (Rust-based, lightweight alternative)     │
│                                                                  │
│  BACKEND                                                         │
│  ─────────────────────────────────────────────────────────────  │
│  • Runtime: Rust (core) + Python (ML pipelines)                 │
│  • API: GraphQL (Juniper) + REST fallback                       │
│  • Real-time: WebSocket + Server-Sent Events                    │
│  • Queue: NATS / Redis Streams                                  │
│                                                                  │
│  3D/4D ENGINE                                                    │
│  ─────────────────────────────────────────────────────────────  │
│  • Rendering: WebGPU (browser) / Vulkan (desktop)               │
│  • Physics: Rapier (Rust) / NVIDIA Omniverse SDK                │
│  • 4D Data: TimescaleDB + Neo4j Aura                            │
│  • AR/VR: WebXR API + OpenXR                                    │
│                                                                  │
│  CONTAINER/ORCHESTRATION                                         │
│  ─────────────────────────────────────────────────────────────  │
│  • Containers: Docker Engine API + Buildah                      │
│  • Orchestration: Kubernetes client-go + Helm SDK               │
│  • Sandbox: Firecracker (MicroVM) + Wasmtime (WASM)             │
│  • Service Mesh: Istio / Linkerd integration                    │
│                                                                  │
│  ML/AI                                                           │
│  ─────────────────────────────────────────────────────────────  │
│  • Training: PyTorch 2.5 + Lightning                            │
│  • AutoML: Optuna + Ray Tune                                    │
│  • Serving: vLLM + Triton Inference Server                      │
│  • Experiment Tracking: MLflow + Weights & Biases               │
│  • Local LLMs: Ollama + llama.cpp integration                   │
│                                                                  │
│  DISCOVERY ENGINE                                                │
│  ─────────────────────────────────────────────────────────────  │
│  • Crawler: Scrapy + GitHub API + HuggingFace API              │
│  • Analysis: Custom ML models for relevance scoring             │
│  • Package Management: uv (Python) + pnpm (Node)               │
│  • Security Scanning: Trivy + Snyk + OSV                        │
│                                                                  │
│  DATA & STORAGE                                                  │
│  ─────────────────────────────────────────────────────────────  │
│  • Primary DB: PostgreSQL 16 + pgvector                         │
│  • Graph DB: Neo4j Aura (agent relationships)                   │
│  • Time-series: TimescaleDB (4D data)                           │
│  • Cache: Redis + DragonflyDB                                   │
│  • Object Storage: MinIO / S3-compatible                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Frontend Stack

### Core Framework

| Technology     | Purpose              | Version |
| -------------- | -------------------- | ------- |
| **React**      | UI framework         | 19.x    |
| **TypeScript** | Type-safe JavaScript | 5.5+    |
| **Vite**       | Build tool           | 6.x     |

### 3D Rendering

| Technology            | Purpose                    | Notes                     |
| --------------------- | -------------------------- | ------------------------- |
| **Three.js**          | 3D graphics library        | Foundation for 3D views   |
| **React Three Fiber** | React wrapper for Three.js | Declarative 3D components |
| **@react-three/drei** | Helpers and utilities      | Pre-built 3D components   |
| **WebGPU**            | Next-gen graphics API      | Primary renderer          |
| **WebGL**             | Fallback renderer          | Browser compatibility     |

### Code Editor

| Technology               | Purpose           | Notes                  |
| ------------------------ | ----------------- | ---------------------- |
| **Monaco Editor**        | Code editing      | Same engine as VS Code |
| **@monaco-editor/react** | React integration | Easy embedding         |
| **Custom LSP**           | Language support  | Agent DSL support      |

### State Management

| Technology      | Purpose      | Notes                    |
| --------------- | ------------ | ------------------------ |
| **Zustand**     | Client state | Lightweight, simple      |
| **React Query** | Server state | Caching, synchronization |
| **Jotai**       | Atomic state | Fine-grained reactivity  |

### UI Components

| Technology        | Purpose    | Notes                |
| ----------------- | ---------- | -------------------- |
| **Radix UI**      | Primitives | Accessible, unstyled |
| **Tailwind CSS**  | Styling    | Utility-first        |
| **Framer Motion** | Animations | Smooth transitions   |
| **Lucide React**  | Icons      | Modern icon set      |

### Desktop Application

| Technology        | Purpose         | Notes                             |
| ----------------- | --------------- | --------------------------------- |
| **Tauri 2.0**     | Desktop wrapper | Rust-based, smaller than Electron |
| **Tauri Plugins** | Native features | File system, notifications        |

---

## Backend Stack

### Core Runtime

| Technology       | Purpose       | Notes                      |
| ---------------- | ------------- | -------------------------- |
| **Rust**         | Core services | Performance-critical paths |
| **Python 3.12+** | ML pipelines  | AI/ML ecosystem            |
| **Tokio**        | Async runtime | Rust async                 |

### API Layer

| Technology            | Purpose        | Notes                       |
| --------------------- | -------------- | --------------------------- |
| **GraphQL (Juniper)** | Primary API    | Type-safe, flexible queries |
| **Axum**              | HTTP framework | Rust web framework          |
| **REST**              | Fallback API   | Simple integrations         |

### Real-time Communication

| Technology             | Purpose        | Notes                      |
| ---------------------- | -------------- | -------------------------- |
| **WebSocket**          | Bidirectional  | Agent streaming            |
| **Server-Sent Events** | One-way stream | Status updates             |
| **NATS**               | Message queue  | High-performance messaging |

---

## 3D/4D Engine Stack

### Rendering

| Technology | Purpose          | Platform       |
| ---------- | ---------------- | -------------- |
| **WebGPU** | Primary renderer | Browser        |
| **Vulkan** | Desktop renderer | Desktop        |
| **wgpu**   | Rust abstraction | Cross-platform |

### Physics & Simulation

| Technology           | Purpose             | Notes       |
| -------------------- | ------------------- | ----------- |
| **Rapier**           | Physics engine      | Rust-native |
| **NVIDIA Omniverse** | Advanced simulation | Optional    |

### 4D Data

| Technology      | Purpose             | Notes                |
| --------------- | ------------------- | -------------------- |
| **TimescaleDB** | Time-series data    | PostgreSQL extension |
| **Neo4j Aura**  | Graph relationships | Cloud-hosted Neo4j   |

### AR/VR

| Technology    | Purpose       | Notes               |
| ------------- | ------------- | ------------------- |
| **WebXR API** | Browser VR/AR | Web-based immersion |
| **OpenXR**    | Native VR/AR  | Desktop headsets    |

---

## Container & Orchestration Stack

### Container Management

| Technology            | Purpose           | Notes               |
| --------------------- | ----------------- | ------------------- |
| **Docker Engine API** | Container runtime | Standard containers |
| **Buildah**           | Image building    | OCI-compliant       |
| **containerd**        | Container runtime | Production-grade    |

### Kubernetes

| Technology       | Purpose          | Notes                |
| ---------------- | ---------------- | -------------------- |
| **client-go**    | K8s Go client    | Official client      |
| **Helm SDK**     | Chart management | Template deployments |
| **Operator SDK** | Custom operators | Agent lifecycle      |

### Sandbox Isolation

| Technology      | Purpose            | Isolation Level      |
| --------------- | ------------------ | -------------------- |
| **Wasmtime**    | WebAssembly        | Tier 1 (lightweight) |
| **Docker**      | Containers         | Tier 2 (development) |
| **Firecracker** | MicroVMs           | Tier 3 (staging)     |
| **Full VM**     | Complete isolation | Tier 4 (chaos)       |

### Service Mesh

| Technology  | Purpose      | Notes                   |
| ----------- | ------------ | ----------------------- |
| **Istio**   | Service mesh | Feature-rich            |
| **Linkerd** | Service mesh | Lightweight alternative |

---

## ML/AI Stack

### Training Framework

| Technology                    | Purpose              | Notes               |
| ----------------------------- | -------------------- | ------------------- |
| **PyTorch 2.5**               | Deep learning        | Primary framework   |
| **Lightning**                 | Training abstraction | Simplified training |
| **Hugging Face Transformers** | Pre-trained models   | Model hub access    |

### AutoML & Optimization

| Technology                     | Purpose               | Notes                 |
| ------------------------------ | --------------------- | --------------------- |
| **Optuna**                     | Hyperparameter tuning | Bayesian optimization |
| **Ray Tune**                   | Distributed tuning    | Scalable HPO          |
| **Neural Architecture Search** | Model design          | Auto-architecture     |

### Model Serving

| Technology                  | Purpose         | Notes              |
| --------------------------- | --------------- | ------------------ |
| **vLLM**                    | LLM serving     | High-throughput    |
| **Triton Inference Server** | Multi-framework | Production serving |
| **ONNX Runtime**            | Cross-platform  | Portable inference |

### Experiment Tracking

| Technology           | Purpose                | Notes               |
| -------------------- | ---------------------- | ------------------- |
| **MLflow**           | Experiment tracking    | Open source         |
| **Weights & Biases** | Visualization          | Cloud-hosted option |
| **TensorBoard**      | Training visualization | PyTorch native      |

### Local LLM Integration

| Technology    | Purpose           | Notes           |
| ------------- | ----------------- | --------------- |
| **Ollama**    | Local LLM runtime | Easy deployment |
| **llama.cpp** | CPU inference     | Efficient local |
| **LM Studio** | Desktop LLMs      | User-friendly   |

---

## Discovery Engine Stack

### Web Crawling

| Technology     | Purpose            | Notes                |
| -------------- | ------------------ | -------------------- |
| **Scrapy**     | Web scraping       | Python framework     |
| **Playwright** | Browser automation | JavaScript rendering |

### API Integration

| Technology          | Purpose             | Notes           |
| ------------------- | ------------------- | --------------- |
| **GitHub API**      | Repository scanning | REST/GraphQL    |
| **HuggingFace API** | Model discovery     | Hub integration |
| **PyPI API**        | Package info        | Python packages |

### Package Management

| Technology | Purpose         | Notes             |
| ---------- | --------------- | ----------------- |
| **uv**     | Python packages | Fast resolver     |
| **pnpm**   | Node packages   | Efficient storage |
| **cargo**  | Rust packages   | Native Rust       |

### Security Scanning

| Technology | Purpose                     | Notes             |
| ---------- | --------------------------- | ----------------- |
| **Trivy**  | Container scanning          | Comprehensive     |
| **Snyk**   | Vulnerability scanning      | Commercial option |
| **OSV**    | Open source vulnerabilities | Google database   |
| **Grype**  | SBOM scanning               | Anchore project   |

---

## Data & Storage Stack

### Databases

| Technology        | Purpose          | Notes               |
| ----------------- | ---------------- | ------------------- |
| **PostgreSQL 16** | Primary database | Relational data     |
| **pgvector**      | Vector search    | Embeddings storage  |
| **Neo4j Aura**    | Graph database   | Agent relationships |
| **TimescaleDB**   | Time-series      | 4D timeline data    |

### Caching

| Technology      | Purpose                | Notes             |
| --------------- | ---------------------- | ----------------- |
| **Redis**       | Caching                | Session, cache    |
| **DragonflyDB** | High-performance cache | Redis alternative |

### Object Storage

| Technology | Purpose        | Notes             |
| ---------- | -------------- | ----------------- |
| **MinIO**  | Self-hosted S3 | Local development |
| **S3**     | Cloud storage  | Production        |

---

## Legal Fortress Stack

### Blockchain

| Technology        | Purpose              | Notes               |
| ----------------- | -------------------- | ------------------- |
| **ethers.js**     | Ethereum integration | Primary anchoring   |
| **bitcoinjs-lib** | Bitcoin integration  | Secondary anchoring |
| **IPFS**          | Distributed storage  | Evidence storage    |

### Analysis

| Technology   | Purpose              | Notes                |
| ------------ | -------------------- | -------------------- |
| **CodeBERT** | Code embeddings      | Similarity detection |
| **MOSS**     | Plagiarism detection | Academic standard    |
| **ScanCode** | License detection    | OSI-approved         |
| **Syft**     | SBOM generation      | CycloneDX/SPDX       |

### Policy Engine

| Technology | Purpose            | Notes             |
| ---------- | ------------------ | ----------------- |
| **OPA**    | Policy enforcement | Open Policy Agent |

---

## Observability Stack

### Metrics

| Technology     | Purpose            | Notes       |
| -------------- | ------------------ | ----------- |
| **Prometheus** | Metrics collection | Time-series |
| **Grafana**    | Visualization      | Dashboards  |

### Logging

| Technology        | Purpose         | Notes          |
| ----------------- | --------------- | -------------- |
| **Loki**          | Log aggregation | Grafana native |
| **OpenTelemetry** | Instrumentation | Standard       |

### Tracing

| Technology | Purpose             | Notes          |
| ---------- | ------------------- | -------------- |
| **Jaeger** | Distributed tracing | Open source    |
| **Tempo**  | Trace storage       | Grafana native |

---

## Development Tools

### Version Control

| Technology | Purpose            | Notes           |
| ---------- | ------------------ | --------------- |
| **Git**    | Source control     | Standard        |
| **GitHub** | Repository hosting | Integration hub |

### CI/CD

| Technology         | Purpose         | Notes              |
| ------------------ | --------------- | ------------------ |
| **GitHub Actions** | CI/CD pipelines | Native integration |
| **ArgoCD**         | GitOps CD       | Kubernetes         |
| **Flux**           | GitOps CD       | Alternative        |

### Testing

| Technology     | Purpose        | Notes             |
| -------------- | -------------- | ----------------- |
| **Vitest**     | Unit testing   | Fast, Vite-native |
| **Playwright** | E2E testing    | Cross-browser     |
| **pytest**     | Python testing | ML pipeline tests |

---

## Minimum Requirements

### Development Machine

| Component   | Minimum                            | Recommended      |
| ----------- | ---------------------------------- | ---------------- |
| **CPU**     | 8 cores                            | 16+ cores        |
| **RAM**     | 16 GB                              | 32+ GB           |
| **GPU**     | Optional                           | NVIDIA RTX 3060+ |
| **Storage** | 100 GB SSD                         | 500+ GB NVMe     |
| **OS**      | Windows 10, macOS 12, Ubuntu 22.04 | Latest versions  |

### Production Deployment

| Component      | Minimum  | Recommended     |
| -------------- | -------- | --------------- |
| **Kubernetes** | v1.28+   | v1.29+          |
| **Nodes**      | 3        | 5+              |
| **Node CPU**   | 4 cores  | 8+ cores        |
| **Node RAM**   | 16 GB    | 32+ GB          |
| **GPU Nodes**  | Optional | NVIDIA A10/A100 |

---

## Related Documentation

- [Architecture Overview](../architecture/README.md)
- [Implementation Roadmap](../roadmap/README.md)
- [Module Documentation](../modules/README.md)
