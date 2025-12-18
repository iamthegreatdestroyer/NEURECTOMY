# NEURECTOMY Port Assignments (16000 Series)

**Purpose:** Complete port mapping for NEURECTOMY services to avoid conflicts.

**Port Strategy:** NEURECTOMY uses the **16000-16999** range exclusively.

**Last Updated:** December 18, 2025  
**Version:** 2.0.0

---

## 🎯 Port Allocation Summary

| Port Range  | Category        | Description                    |
| ----------- | --------------- | ------------------------------ |
| 16000-16099 | Application     | Frontend, Desktop, Dev Tools   |
| 16080-16099 | API             | REST, GraphQL, WebSocket       |
| 16400-16499 | Databases       | PostgreSQL, Neo4j, TimescaleDB |
| 16500-16599 | Cache/Messaging | Redis, NATS                    |
| 16600-16699 | AI/ML           | Ollama, MLflow, ChromaDB       |
| 16900-16999 | Observability   | Prometheus, Grafana, Jaeger    |

---

## 🔵 Application Tier (16000-16099)

| Port      | Service            | Description              | Config File       |
| --------- | ------------------ | ------------------------ | ----------------- |
| **16000** | Spectrum Workspace | Vite frontend dev server | `vite.config.ts`  |
| **16001** | Tauri Dev Server   | Desktop app dev mode     | `tauri.conf.json` |

---

## 🟢 API Tier (16080-16099)

| Port      | Service        | Description               | Config File          |
| --------- | -------------- | ------------------------- | -------------------- |
| **16080** | API Gateway    | Main REST/GraphQL API     | `docker-compose.yml` |
| **16081** | ML Service     | Python FastAPI ML service | `docker-compose.yml` |
| **16082** | ML Service GPU | GPU-accelerated ML        | `docker-compose.yml` |
| **16083** | WebSocket      | Real-time events          | `docker-compose.yml` |

---

## 🟡 Databases (16400-16499)

| Port      | Service     | Internal Port | Config File          |
| --------- | ----------- | ------------- | -------------------- |
| **16432** | PostgreSQL  | 5432          | `docker-compose.yml` |
| **16433** | TimescaleDB | 5432          | `docker-compose.yml` |
| **16474** | Neo4j HTTP  | 7474          | `docker-compose.yml` |
| **16475** | Neo4j Bolt  | 7687          | `docker-compose.yml` |

---

## 🟠 Cache & Messaging (16500-16599)

| Port      | Service         | Internal Port | Config File          |
| --------- | --------------- | ------------- | -------------------- |
| **16500** | Redis           | 6379          | `docker-compose.yml` |
| **16522** | NATS Client     | 4222          | `docker-compose.yml` |
| **16523** | NATS Monitoring | 8222          | `docker-compose.yml` |

---

## 🟣 AI/ML Services (16600-16699)

| Port      | Service          | Internal Port | Config File          |
| --------- | ---------------- | ------------- | -------------------- |
| **16600** | Ollama           | 11434         | `docker-compose.yml` |
| **16610** | MLflow           | 5000          | `docker-compose.yml` |
| **16611** | Optuna Dashboard | 8080          | `docker-compose.yml` |
| **16620** | ChromaDB         | 8000          | `docker-compose.yml` |

---

## 🔴 Observability (16900-16999)

| Port      | Service          | Internal Port | Config File          |
| --------- | ---------------- | ------------- | -------------------- |
| **16900** | Prometheus       | 9090          | `docker-compose.yml` |
| **16901** | AlertManager     | 9093          | `docker-compose.yml` |
| **16910** | Grafana          | 3000          | `docker-compose.yml` |
| **16920** | Jaeger UI        | 16686         | `docker-compose.yml` |
| **16921** | Jaeger Collector | 14268         | `docker-compose.yml` |
| **16922** | Jaeger gRPC      | 14250         | `docker-compose.yml` |
| **16923** | Zipkin           | 9411          | `docker-compose.yml` |
| **16930** | Loki             | 3100          | `docker-compose.yml` |
| **16950** | MinIO API        | 9000          | `docker-compose.yml` |
| **16951** | MinIO Console    | 9001          | `docker-compose.yml` |

---

## 🔗 CORS Configuration

```
http://localhost:16000   # Spectrum Workspace
http://localhost:16080   # API Gateway
tauri://localhost        # Tauri Desktop App
http://localhost:1420    # Tauri dev (legacy)
```

---

## 🚀 Quick Start

```powershell
# Start all Docker services
docker-compose up -d

# Start frontend dev server
cd apps/spectrum-workspace
pnpm dev
# → http://localhost:16000

# Start desktop app (dev mode)
pnpm tauri:dev
```

---

## 📊 Access Points

| Service            | URL                    |
| ------------------ | ---------------------- |
| Spectrum Workspace | http://localhost:16000 |
| API Gateway        | http://localhost:16080 |
| ML Service         | http://localhost:16081 |
| Grafana            | http://localhost:16910 |
| Prometheus         | http://localhost:16900 |
| Neo4j Browser      | http://localhost:16474 |
| MLflow             | http://localhost:16610 |
| Jaeger             | http://localhost:16920 |
| MinIO Console      | http://localhost:16951 |

---

## 🌐 Related Project Port Allocations

To avoid cross-project conflicts, each project uses a distinct 10,000-series:

| Port Range      | Project             | Description         |
| --------------- | ------------------- | ------------------- |
| **10000-10999** | DOPPELGANGER-STUDIO | AI Content Platform |
| **16000-16999** | NEURECTOMY          | AI Development IDE  |
| **26000-26999** | SigmaLang (ΣLANG)   | Sub-Linear Compiler |
| **36000-36999** | SigmaVault (ΣVAULT) | Encrypted Storage   |
| **46000-46999** | Ryot LLM            | BitNet Inference    |

See `docs/MASTER_PORT_ASSIGNMENTS.md` for complete cross-project allocations.

---

## ⚠️ Ports to AVOID

Never use these ports (default/common conflicts):

| Port | Reason               |
| ---- | -------------------- |
| 3000 | DOPPELGANGER uses it |
| 5432 | Default PostgreSQL   |
| 6379 | Default Redis        |
| 8000 | Common API port      |
| 8080 | Common HTTP alt      |
| 9090 | Default Prometheus   |

---

## 📋 Conflict Resolution History

| Date       | Conflict                                 | Resolution                      |
| ---------- | ---------------------------------------- | ------------------------------- |
| 2025-12-18 | Multiple projects using 3000, 8000, 6379 | Migrated to 16xxx series        |
| 2025-12-18 | Cross-project port collisions            | Created MASTER_PORT_ASSIGNMENTS |
