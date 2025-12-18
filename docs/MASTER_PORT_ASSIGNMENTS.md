# MASTER PORT ASSIGNMENTS - All Projects

**Purpose:** Unified port allocation across all NEURECTOMY ecosystem projects to avoid conflicts.

**Strategy:** Each project uses a unique 10,000-series port range with 1,000 ports per project.

---

## 🎯 PORT RANGE ALLOCATIONS

| Port Range      | Project             | Description                    |
| --------------- | ------------------- | ------------------------------ |
| **10000-10999** | DOPPELGANGER-STUDIO | AI Content Generation Platform |
| **16000-16999** | NEURECTOMY          | AI Development IDE/Platform    |
| **26000-26999** | SigmaLang (ΣLANG)   | Sub-Linear Language Compiler   |
| **36000-36999** | SigmaVault (ΣVAULT) | Secure Encrypted Storage       |
| **46000-46999** | Ryot LLM            | BitNet AI Inference Engine     |

---

## 🔵 NEURECTOMY (16000-16999)

### Application Tier (16000-16099)

| Port      | Service            | Description              |
| --------- | ------------------ | ------------------------ |
| **16000** | Spectrum Workspace | Frontend Vite dev server |
| **16001** | Desktop App (Dev)  | Tauri dev server         |
| **16010** | GraphQL Playground | Dev tools                |

### API Tier (16080-16099)

| Port      | Service          | Description               |
| --------- | ---------------- | ------------------------- |
| **16080** | API Gateway      | Main REST/GraphQL API     |
| **16081** | ML Service       | Python FastAPI ML service |
| **16082** | Rust Core API    | Rust Axum backend         |
| **16083** | WebSocket Server | Real-time events          |

### Databases (16400-16499)

| Port      | Service     | Description             |
| --------- | ----------- | ----------------------- |
| **16432** | PostgreSQL  | Primary database        |
| **16433** | TimescaleDB | Time-series database    |
| **16474** | Neo4j HTTP  | Graph database UI       |
| **16475** | Neo4j Bolt  | Graph database protocol |

### Cache & Messaging (16500-16599)

| Port      | Service         | Description       |
| --------- | --------------- | ----------------- |
| **16500** | Redis           | Cache and pub/sub |
| **16522** | NATS Client     | Message broker    |
| **16523** | NATS Monitoring | NATS dashboard    |

### AI/ML Services (16600-16699)

| Port      | Service          | Description           |
| --------- | ---------------- | --------------------- |
| **16600** | Ollama           | Local LLM server      |
| **16610** | MLflow           | Experiment tracking   |
| **16611** | Optuna Dashboard | Hyperparameter tuning |
| **16620** | ChromaDB         | Vector database       |

### Observability (16900-16999)

| Port      | Service       | Description         |
| --------- | ------------- | ------------------- |
| **16900** | Prometheus    | Metrics collection  |
| **16901** | AlertManager  | Alert management    |
| **16910** | Grafana       | Dashboards          |
| **16920** | Jaeger UI     | Distributed tracing |
| **16930** | Loki          | Log aggregation     |
| **16950** | MinIO API     | Object storage      |
| **16951** | MinIO Console | MinIO UI            |

---

## 🟢 SigmaLang (26000-26999)

### Application Tier (26000-26099)

| Port      | Service       | Description              |
| --------- | ------------- | ------------------------ |
| **26000** | SigmaLang IDE | Web-based IDE            |
| **26001** | LSP Server    | Language Server Protocol |
| **26010** | Playground    | Interactive REPL         |

### API Tier (26080-26099)

| Port      | Service      | Description              |
| --------- | ------------ | ------------------------ |
| **26080** | Compiler API | REST compilation service |
| **26081** | Runtime API  | Execution service        |
| **26082** | Debug API    | Debugger protocol        |

### Infrastructure (26400-26499)

| Port      | Service      | Description    |
| --------- | ------------ | -------------- |
| **26432** | PostgreSQL   | Database       |
| **26500** | Redis        | Cache          |
| **26600** | Model Server | AI integration |

### Observability (26900-26999)

| Port      | Service    | Description |
| --------- | ---------- | ----------- |
| **26900** | Prometheus | Metrics     |
| **26910** | Grafana    | Dashboards  |

---

## 🟡 SigmaVault (36000-36999)

### Application Tier (36000-36099)

| Port      | Service       | Description   |
| --------- | ------------- | ------------- |
| **36000** | Vault UI      | Web interface |
| **36001** | Admin Console | Management UI |

### API Tier (36080-36099)

| Port      | Service     | Description        |
| --------- | ----------- | ------------------ |
| **36080** | Storage API | Main vault API     |
| **36081** | Crypto API  | Encryption service |
| **36082** | Sync API    | Synchronization    |

### Infrastructure (36400-36499)

| Port      | Service       | Description    |
| --------- | ------------- | -------------- |
| **36432** | PostgreSQL    | Database       |
| **36500** | Redis         | Cache          |
| **36600** | MinIO         | Object storage |
| **36601** | MinIO Console | MinIO UI       |

### Observability (36900-36999)

| Port      | Service    | Description |
| --------- | ---------- | ----------- |
| **36900** | Prometheus | Metrics     |
| **36910** | Grafana    | Dashboards  |

---

## 🟣 Ryot LLM (46000-46999)

### Application Tier (46000-46099)

| Port      | Service       | Description        |
| --------- | ------------- | ------------------ |
| **46000** | Ryot UI       | Web interface      |
| **46001** | Model Browser | Model selection UI |

### API Tier (46080-46099)

| Port      | Service        | Description                      |
| --------- | -------------- | -------------------------------- |
| **46080** | Inference API  | Main LLM API (OpenAI-compatible) |
| **46081** | Streaming API  | SSE streaming endpoint           |
| **46082** | Embeddings API | Vector embeddings                |
| **46083** | Agent API      | AI agent execution               |

### Infrastructure (46400-46499)

| Port      | Service    | Description        |
| --------- | ---------- | ------------------ |
| **46432** | PostgreSQL | Database           |
| **46500** | Redis      | Cache              |
| **46600** | Ollama     | Local model server |

### Observability (46900-46999)

| Port      | Service    | Description |
| --------- | ---------- | ----------- |
| **46900** | Prometheus | Metrics     |
| **46910** | Grafana    | Dashboards  |

---

## 🔴 DOPPELGANGER-STUDIO (10000-10999)

### Application Tier (10000-10099)

| Port      | Service       | Description   |
| --------- | ------------- | ------------- |
| **10000** | Frontend      | React web app |
| **10001** | Asset Preview | Media preview |

### API Tier (10080-10099)

| Port      | Service   | Description        |
| --------- | --------- | ------------------ |
| **10080** | Main API  | FastAPI backend    |
| **10081** | Media API | Asset processing   |
| **10082** | AI API    | Content generation |

### Infrastructure (10400-10499)

| Port      | Service    | Description      |
| --------- | ---------- | ---------------- |
| **10432** | PostgreSQL | Database         |
| **10500** | Redis      | Cache            |
| **10517** | MongoDB    | Document storage |

### Observability (10900-10999)

| Port      | Service    | Description |
| --------- | ---------- | ----------- |
| **10900** | Prometheus | Metrics     |
| **10910** | Grafana    | Dashboards  |

---

## ⚠️ RESERVED SYSTEM PORTS (NEVER USE)

| Port Range | Reserved For               |
| ---------- | -------------------------- |
| 0-1023     | System/Well-known          |
| 1024-3000  | Common apps (may conflict) |
| 3306       | MySQL default              |
| 5432       | PostgreSQL default         |
| 5672       | RabbitMQ default           |
| 6379       | Redis default              |
| 8080       | Common HTTP alt            |
| 8443       | Common HTTPS alt           |
| 9090       | Prometheus default         |
| 27017      | MongoDB default            |

---

## 🔧 MIGRATION CHECKLIST

- [ ] Update NEURECTOMY docker-compose.yml to 16xxx series
- [ ] Update SigmaLang docker-compose.yml to 26xxx series
- [ ] Update SigmaVault docker-compose.yml to 36xxx series
- [ ] Update Ryot service configs to 46xxx series
- [ ] Update DOPPELGANGER docker-compose.yml to 10xxx series
- [ ] Update all .env files with new ports
- [ ] Update CORS configurations
- [ ] Update Kubernetes manifests
- [ ] Test all services with new ports

---

**Last Updated:** December 18, 2025
**Version:** 2.0
