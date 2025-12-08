# NEURECTOMY Port Assignments (16000 Series)

**Purpose:** This document defines all port assignments for the NEURECTOMY project to avoid conflicts with other projects (e.g., Doppelganger Project on port 3000).

**Port Strategy:** All NEURECTOMY services use the 16000 series (16000-16999) for consistency and conflict avoidance.

---

## 🎯 Primary Application Ports

| Port      | Service                       | Description                             | Configuration File                       |
| --------- | ----------------------------- | --------------------------------------- | ---------------------------------------- |
| **16000** | Spectrum Workspace (Frontend) | Vite development server for main UI     | `apps/spectrum-workspace/vite.config.ts` |
| **16080** | API Gateway / GraphQL Backend | Main API and GraphQL endpoints          | `.env.example` → `API_URL`               |
| **16081** | ML Service                    | Python FastAPI machine learning service | `services/ml-service/config.py`          |

---

## 📦 Supporting Services (Non-Standard Ports)

These services use their standard ports but are documented here for completeness:

| Port | Service           | Description                   | Configuration File                   |
| ---- | ----------------- | ----------------------------- | ------------------------------------ |
| 5434 | PostgreSQL (Main) | Primary database              | `docker-compose.yml` → `postgres`    |
| 5433 | TimescaleDB       | Time-series database          | `docker-compose.yml` → `timescaledb` |
| 7474 | Neo4j HTTP        | Graph database HTTP interface | `docker-compose.yml` → `neo4j`       |
| 7687 | Neo4j Bolt        | Graph database Bolt protocol  | `docker-compose.yml` → `neo4j`       |
| 6379 | Redis             | Cache and pub/sub             | `docker-compose.yml` → `redis`       |
| 4222 | NATS              | Message broker                | `docker-compose.yml` → `nats`        |

---

## 🔗 CORS Configuration

The following origins are allowed for cross-origin requests:

```
http://localhost:16000   # Spectrum Workspace (Frontend)
http://localhost:16080   # API Gateway
tauri://localhost        # Tauri app (if applicable)
http://localhost:1420    # Tauri dev server (if applicable)
```

**Configuration:** `.env.example` → `CORS_ORIGINS`

---

## 🚀 Starting Services

### Development Mode

```powershell
# Start infrastructure services (databases, cache, messaging)
docker-compose up -d

# Start frontend (Spectrum Workspace)
pnpm dev
# Frontend will be available at: http://localhost:16000

# Start ML service (if needed)
cd services/ml-service
uvicorn main:app --host 0.0.0.0 --port 16081 --reload
# ML service will be available at: http://localhost:16081
```

### Access Points

- **Spectrum Workspace UI:** http://localhost:16000
- **API/GraphQL Endpoint:** http://localhost:16080
- **ML Service API:** http://localhost:16081
- **Neo4j Browser:** http://localhost:7474

---

## 📋 Port Conflict History

### Resolved Conflicts

| Date       | Conflict                                                            | Resolution       |
| ---------- | ------------------------------------------------------------------- | ---------------- |
| 2025-01-XX | Spectrum Workspace conflicted with Doppelganger Project (port 3000) | Changed to 16000 |
| 2025-01-XX | API/GraphQL conflicted with Doppelganger services (port 8080)       | Changed to 16080 |
| 2025-01-XX | ML Service default port (8000)                                      | Changed to 16081 |

---

## 🔧 Updating Ports

If you need to add a new service or update a port:

1. **Choose a port in the 16000 series** (16000-16999)
2. **Update this document** with the new assignment
3. **Update the relevant configuration files:**
   - `.env` and `.env.example` (if applicable)
   - `docker-compose.yml` (if Docker service)
   - Service-specific config files (e.g., `config.py`, `vite.config.ts`)
   - Update CORS configuration if needed
4. **Restart affected services**

---

## ⚠️ Reserved Ports

The following ports are reserved and should NOT be used:

- **3000:** Doppelganger Project (separate project)
- **8080:** Commonly used, avoid to prevent conflicts
- **8000:** Commonly used, avoid to prevent conflicts

---

## 📝 Notes

- All ports in the 16000 series are exclusively for NEURECTOMY
- Database ports (5432, 5433, 6379, 7474, 7687) use standard offsets to avoid conflicts with default installations
- Tauri ports (1420) are for desktop application development
- Always update `.env.example` when adding new environment variables

---

**Last Updated:** 2025-01-XX  
**Version:** 1.0.0
