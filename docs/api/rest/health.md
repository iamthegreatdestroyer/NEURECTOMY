# REST API - Health & Metrics

Non-GraphQL endpoints for system health and observability.

## Health Endpoints

### GET /health
Basic health check.

**Response:**
```json
{
  "status": "healthy",
  "service": "neurectomy-core",
  "version": "0.1.0"
}
```

**Status Codes:**
- `200 OK` - Service is healthy
- `503 Service Unavailable` - Service is unhealthy

---

### GET /health/ready
Readiness probe (checks dependencies).

**Response:**
```json
{
  "status": "ready",
  "checks": {
    "postgres": {
      "status": "up",
      "latency_ms": 2
    },
    "redis": {
      "status": "up",
      "latency_ms": 1
    },
    "neo4j": {
      "status": "up",
      "latency_ms": 5
    },
    "nats": {
      "status": "up",
      "latency_ms": 1
    }
  }
}
```

**Status Codes:**
- `200 OK` - All dependencies ready
- `503 Service Unavailable` - One or more dependencies unhealthy

---

### GET /health/live
Liveness probe (basic service check).

**Response:**
```json
{
  "status": "alive",
  "uptime_seconds": 3600
}
```

**Status Codes:**
- `200 OK` - Service is alive

---

## Metrics Endpoint

### GET /metrics
Prometheus metrics endpoint.

**Response (text/plain):**
```
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET",path="/health",status="200"} 1523
http_requests_total{method="POST",path="/graphql",status="200"} 45678

# HELP http_request_duration_seconds HTTP request latency
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{le="0.01"} 35000
http_request_duration_seconds_bucket{le="0.05"} 42000
http_request_duration_seconds_bucket{le="0.1"} 45000
http_request_duration_seconds_bucket{le="+Inf"} 45678
http_request_duration_seconds_sum 892.45
http_request_duration_seconds_count 45678

# HELP active_websocket_connections Current WebSocket connections
# TYPE active_websocket_connections gauge
active_websocket_connections 127

# HELP active_agents Number of active agents
# TYPE active_agents gauge
active_agents 45

# HELP database_pool_connections Database connection pool status
# TYPE database_pool_connections gauge
database_pool_connections{state="active"} 8
database_pool_connections{state="idle"} 12
database_pool_connections{state="max"} 20

# HELP agent_requests_total Total agent requests
# TYPE agent_requests_total counter
agent_requests_total{agent_id="550e8400-..."} 1234

# HELP agent_tokens_total Total tokens processed
# TYPE agent_tokens_total counter
agent_tokens_total{agent_id="550e8400-...",type="input"} 45000
agent_tokens_total{agent_id="550e8400-...",type="output"} 67000
```

---

## Version Endpoint

### GET /version
Service version information.

**Response:**
```json
{
  "service": "neurectomy-core",
  "version": "0.1.0",
  "build": {
    "commit": "abc1234",
    "date": "2024-01-15T10:30:00Z",
    "branch": "main"
  },
  "rust_version": "1.75.0"
}
```

---

## Debug Endpoints (Development Only)

### GET /debug/config
Current configuration (sanitized).

**Response:**
```json
{
  "environment": "development",
  "log_level": "debug",
  "database": {
    "host": "localhost",
    "port": 5432,
    "database": "neurectomy"
  },
  "redis": {
    "host": "localhost",
    "port": 6379
  },
  "features": {
    "streaming": true,
    "knowledge_base": true,
    "workflows": true
  }
}
```

**Note:** This endpoint is disabled in production.

---

## Kubernetes Probes Configuration

Example Kubernetes deployment configuration:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neurectomy-core
spec:
  template:
    spec:
      containers:
        - name: neurectomy-core
          livenessProbe:
            httpGet:
              path: /health/live
              port: 8080
            initialDelaySeconds: 10
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /health/ready
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 3
          startupProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 0
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 30
```

---

## Prometheus Scrape Configuration

```yaml
scrape_configs:
  - job_name: 'neurectomy-core'
    static_configs:
      - targets: ['neurectomy-core:8080']
    metrics_path: /metrics
    scrape_interval: 15s
```
