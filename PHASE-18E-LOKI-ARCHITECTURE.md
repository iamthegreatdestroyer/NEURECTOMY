# NEURECTOMY Phase 18E - Loki Architecture & Diagrams

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NEURECTOMY OBSERVABILITY STACK                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────  APPLICATIONS  ──────────────────────┐            │
│  │                                                              │            │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │            │
│  │  │    Ryot      │  │   ΣLANG      │  │   ΣVAULT    │   │            │
│  │  │   Service    │  │   Compiler   │  │    Vault    │   │            │
│  │  │              │  │              │  │             │   │            │
│  │  │ Logs:        │  │ Logs:        │  │ Logs:       │   │            │
│  │  │ JSON, level  │  │ JSON, code   │  │ JSON, audit │   │            │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │            │
│  │                                                              │            │
│  │  ┌──────────────────────────────────────┐                 │            │
│  │  │         Agents Service               │                 │            │
│  │  │    (Task Orchestration)              │                 │            │
│  │  │                                       │                 │            │
│  │  │  Logs: JSON, agent metrics           │                 │            │
│  │  └──────────────────────────────────────┘                 │            │
│  │                                                              │            │
│  └──────────────────────────────────────────────────────────────┘            │
│                          ↓↓↓ Log streams                                     │
│                                                                              │
│  ┌────────────────────── LOG COLLECTION ─────────────────────┐             │
│  │                                                              │             │
│  │                   PROMTAIL DAEMONSET                        │             │
│  │              (1 pod per node, all nodes)                   │             │
│  │                                                              │             │
│  │  ┌─────────────────────────────────────────────────────┐  │             │
│  │  │           Scrape Configs                           │  │             │
│  │  │                                                     │  │             │
│  │  │  • Kubernetes job: Pod log discovery via SD       │  │             │
│  │  │  • Ryot job: JSON parsing, metrics extraction    │  │             │
│  │  │  • ΣLANG job: Multiline, timestamp parsing       │  │             │
│  │  │  • ΣVAULT job: Audit label extraction            │  │             │
│  │  │  • Agents job: Agent ID + metrics                │  │             │
│  │  │  • Syslog job: UDP 1514 listener                 │  │             │
│  │  │                                                     │  │             │
│  │  │  Transformations:                                  │  │             │
│  │  │  ✓ Label parsing (service, level, tier)          │  │             │
│  │  │  ✓ JSON parsing & field extraction               │  │             │
│  │  │  ✓ Metrics generation (Counter, Gauge, Hist)    │  │             │
│  │  │  ✓ Trace ID injection (if configured)            │  │             │
│  │  │                                                     │  │             │
│  │  │  Output: HTTP push to Loki (port 3100)           │  │             │
│  │  │  Metrics: Exported on 9080 to Prometheus         │  │             │
│  │  └─────────────────────────────────────────────────────┘  │             │
│  │                                                              │             │
│  └──────────────────────────────────────────────────────────────┘             │
│                          ↓↓↓ Formatted logs                                   │
│                                                                              │
│  ┌────────────────── LOG AGGREGATION & STORAGE ──────────────┐             │
│  │                                                              │             │
│  │             LOKI STATEFULSET (3 replicas)                 │             │
│  │          ┌────────────────────────────────┐               │             │
│  │          │   DISTRIBUTOR                  │               │             │
│  │          │ (Rate limiting, request auth)  │               │             │
│  │          │ Rate limit: 100K lines/sec     │               │             │
│  │          │ Burst: 200K lines/sec          │               │             │
│  │          │ Max streams: 10K per user      │               │             │
│  │          └────────────────────────────────┘               │             │
│  │                     ↓                                       │             │
│  │          ┌────────────────────────────────┐               │             │
│  │          │   INGESTER                     │               │             │
│  │          │ (Chunk assembly, compression)  │               │             │
│  │          │ Chunk idle: 3m                 │               │             │
│  │          │ Max chunk age: 1h              │               │             │
│  │          │ Compression: Snappy (10:1)     │               │             │
│  │          └────────────────────────────────┘               │             │
│  │                     ↓                                       │             │
│  │          ┌────────────────────────────────┐               │             │
│  │          │   QUERIER                      │               │             │
│  │          │ (Query execution, caching)     │               │             │
│  │          │ Rate limit: 100 queries/sec    │               │             │
│  │          │ Cache size: 100MB default      │               │             │
│  │          │ Cache TTL: 1h                  │               │             │
│  │          └────────────────────────────────┘               │             │
│  │                                                              │             │
│  │  ┌─── STORAGE BACKEND ───────────────────────────────┐  │             │
│  │  │                                                   │  │             │
│  │  │  Index Store (24-hour periods)                  │  │             │
│  │  │  ┌─────────────────────────────────────────┐   │  │             │
│  │  │  │  BoltDB Shipper                        │   │  │             │
│  │  │  │  • Local index on each Loki pod        │   │  │             │
│  │  │  │  • Shipper uploads to S3 hourly        │   │  │             │
│  │  │  │  • Fallback index for old periods      │   │  │             │
│  │  │  └─────────────────────────────────────────┘   │  │             │
│  │  │                                                   │  │             │
│  │  │  Chunks Store                                    │  │             │
│  │  │  ┌─────────────────────────────────────────┐   │  │             │
│  │  │  │  S3/MinIO Backend                      │   │  │             │
│  │  │  │  • Immutable chunk objects             │   │  │             │
│  │  │  │  • Organized by service/date           │   │  │             │
│  │  │  │  • Snappy compressed                   │   │  │             │
│  │  │  │  • Cost: ~$1/month (compressed)        │   │  │             │
│  │  │  └─────────────────────────────────────────┘   │  │             │
│  │  │                                                   │  │             │
│  │  │  Query Cache                                     │  │             │
│  │  │  ┌─────────────────────────────────────────┐   │  │             │
│  │  │  │  Memcached Cluster (3 replicas)        │   │  │             │
│  │  │  │  • 512MB per instance                  │   │  │             │
│  │  │  │  • Hit rate: >50%                      │   │  │             │
│  │  │  │  • TTL: 1 hour                         │   │  │             │
│  │  │  │  • Improves P99 latency by 10x         │   │  │             │
│  │  │  └─────────────────────────────────────────┘   │  │             │
│  │  │                                                   │  │             │
│  │  └─────────────────────────────────────────────────┘  │             │
│  │                                                              │             │
│  │  Retention Policies (Automatic cleanup)                    │             │
│  │  ┌────────────────────────────────────────────────────┐   │             │
│  │  │ Service          │ Retention  │ Monthly Storage  │   │             │
│  │  ├──────────────────┼────────────┼──────────────────┤   │             │
│  │  │ Ryot             │ 30 days    │ ~5GB compressed  │   │             │
│  │  │ ΣLANG            │ 30 days    │ ~6GB compressed  │   │             │
│  │  │ ΣVAULT           │ 30 days    │ ~3GB compressed  │   │             │
│  │  │ Agents           │ 60 days    │ ~3GB compressed  │   │             │
│  │  │ Debug (override) │ 3 days     │ ~50MB            │   │             │
│  │  │ TOTAL            │            │ ~18GB/month      │   │             │
│  │  └────────────────────────────────────────────────────┘   │             │
│  │                                                              │             │
│  └──────────────────────────────────────────────────────────────┘             │
│                          ↓↓↓ Queries                                         │
│                                                                              │
│  ┌────────────── OBSERVABILITY FRONTENDS ─────────────┐                   │
│  │                                                      │                   │
│  │  ┌──────────────────────────────────────────────┐  │                   │
│  │  │            GRAFANA                          │  │                   │
│  │  │   (Visualization & Dashboard Engine)       │  │                   │
│  │  │                                              │  │                   │
│  │  │  Datasources Connected:                    │  │                   │
│  │  │  ✓ Loki (logs) - Port 3100                 │  │                   │
│  │  │  ✓ Prometheus (metrics) - Port 9090        │  │                   │
│  │  │  ✓ Jaeger (traces) - Port 16686            │  │                   │
│  │  │                                              │  │                   │
│  │  │  Dashboard Types:                           │  │                   │
│  │  │  • Logs: {service="ryot"} queries          │  │                   │
│  │  │  • Metrics: Error rate, latency graphs     │  │                   │
│  │  │  • Traces: Distributed trace viewer        │  │                   │
│  │  │  • Correlation: Logs ↔ Metrics ↔ Traces   │  │                   │
│  │  │                                              │  │                   │
│  │  │  Ports: 3000 (HTTP)                        │  │                   │
│  │  └──────────────────────────────────────────────┘  │                   │
│  │                                                      │                   │
│  │  ┌──────────────────────────────────────────────┐  │                   │
│  │  │         ALERTMANAGER                        │  │                   │
│  │  │    (Alert Routing & Notification)          │  │                   │
│  │  │                                              │  │                   │
│  │  │  Alert Sources:                            │  │                   │
│  │  │  • Prometheus rules (metrics-based)        │  │                   │
│  │  │  • Loki ruler (logs-based)                 │  │                   │
│  │  │                                              │  │                   │
│  │  │  Routes to:                                │  │                   │
│  │  │  • Email, Slack, PagerDuty, Teams         │  │                   │
│  │  │                                              │  │                   │
│  │  │  Ports: 9093 (HTTP)                        │  │                   │
│  │  └──────────────────────────────────────────────┘  │                   │
│  │                                                      │                   │
│  │  ┌──────────────────────────────────────────────┐  │                   │
│  │  │           JAEGER                            │  │                   │
│  │  │   (Distributed Tracing UI)                 │  │                   │
│  │  │                                              │  │                   │
│  │  │  Features:                                 │  │                   │
│  │  │  • Full request traces                     │  │                   │
│  │  │  • Service dependency graph                │  │                   │
│  │  │  • Trace-logs linking                      │  │                   │
│  │  │  • Performance analysis                    │  │                   │
│  │  │                                              │  │                   │
│  │  │  Ports: 16686 (HTTP)                       │  │                   │
│  │  └──────────────────────────────────────────────┘  │                   │
│  │                                                      │                   │
│  └──────────────────────────────────────────────────────┘                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
Applications                 Collection               Backend                    Query
─────────────────           ────────────            ─────────────────           ────

 Ryot ────┐
           ├──→ Promtail scrape ──→ HTTP/gRPC ──→ Loki Distributor ──→ BoltDB (index)
ΣLANG ────┤                                             │                     ↓
           │                                             ├──→ Ingester ───→ S3 (chunks)
ΣVAULT ───┤                                             │
           │                                             └──→ Querier ◄─── Memcached
Agents ────┘                                                  │
                                                              │
                                                              ↓
                                                         Grafana
                                                      (visualization)
                                                              │
                                                              ↓
                                                       LogQL Queries
                                                              │
                                                     ┌─────────────┐
                                                     │  Examples:  │
                                                     ├─────────────┤
                                                     │{service="ryot"}
                                                     │{level="error"}
                                                     │rate({job=~".+"})
                                                     └─────────────┘
```

---

## Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Kubernetes Cluster                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ monitoring namespace                                       │   │
│  ├────────────────────────────────────────────────────────────┤   │
│  │                                                              │   │
│  │  Loki Pod #1 ─┐                                             │   │
│  │  (StatefulSet)├─→ Shared Storage (100Gi fast-ssd)         │   │
│  │  Loki Pod #2 ─┤   │                                         │   │
│  │  Loki Pod #3 ─┘   │                                         │   │
│  │                   ↓                                          │   │
│  │            ┌─────────────────┐                             │   │
│  │            │  S3/MinIO       │                             │   │
│  │            │  Backend        │                             │   │
│  │            │  (chunks)       │                             │   │
│  │            └─────────────────┘                             │   │
│  │                                                              │   │
│  │  Memcached Pod #1 ──┐                                       │   │
│  │  Memcached Pod #2 ──├─→ Cluster mode (cache sharing)      │   │
│  │  Memcached Pod #3 ──┘                                       │   │
│  │                                                              │   │
│  │  Promtail Pod (Node 1) ──┐                                  │   │
│  │  Promtail Pod (Node 2) ──├─→ All push to Loki:3100        │   │
│  │  Promtail Pod (Node N) ──┘    (port forwarded from svc)    │   │
│  │                                                              │   │
│  │  Grafana Pod ─────────┐                                     │   │
│  │  AlertManager Pod ────├─→ All query Loki for alerts       │   │
│  │  Prometheus Pod ──────┘                                     │   │
│  │                                                              │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ Services & Network                                         │   │
│  ├────────────────────────────────────────────────────────────┤   │
│  │                                                              │   │
│  │  loki-headless (ClusterIP: None)                           │   │
│  │  └─→ Used by StatefulSet for stable DNS                   │   │
│  │                                                              │   │
│  │  loki (LoadBalancer or NodePort)                           │   │
│  │  └─→ Port 3100 exposed for external access                │   │
│  │                                                              │   │
│  │  promtail-metrics (ClusterIP)                             │   │
│  │  └─→ Port 9080 scraped by Prometheus                      │   │
│  │                                                              │   │
│  │  NetworkPolicies:                                          │   │
│  │  • loki-network-policy: Controls ingress/egress           │   │
│  │  • promtail-network-policy: Restricts traffic             │   │
│  │                                                              │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

External Connections:
  Prometheus ──→ Scrapes metrics from Loki/Promtail ports
  Grafana ──────→ Queries Loki datasource
  AlertManager ← Loki ruler fires alerts
```

---

## Log Transformation Pipeline

```
Raw Log Line (from application)
│
├─ Entry
│  ├─ Timestamp: 2024-01-15T10:30:45.123Z
│  ├─ Log content: {"level": "error", "msg": "..."}
│  └─ Source: /var/log/containers/pod-xxx.log
│
↓ Promtail Scrape Job Matches
│
├─ Service: ryot (from pod label: app=ryot)
├─ Pod: ryot-deployment-abc123-def456
├─ Namespace: neurectomy
├─ Container: ryot-container
│
↓ JSON Parsing
│
├─ level: error
├─ msg: "Database connection timeout"
├─ trace_id: "abc123def456"
├─ span_id: "xyz789"
├─ duration_ms: 5234
│
↓ Label Extraction
│
├─ service: "ryot" (from app label)
├─ level: "error" (from JSON level field)
├─ tier: "core" (from pod tier label)
├─ pod: "ryot-deployment-abc123-def456"
├─ namespace: "neurectomy"
├─ container_name: "ryot-container"
│
↓ Metrics Extraction
│
├─ log_lines_total{service="ryot", level="error"} +1
├─ log_durations_bucket{service="ryot", le="10000"} +1
├─ log_errors_total{service="ryot", module="database"} +1
│
↓ Stream Creation
│
Labels Dict:
  service: ryot
  level: error
  tier: core
  pod: ryot-deployment-abc123-def456
  namespace: neurectomy
  container_name: ryot-container

Content:
  {"level": "error", "msg": "Database connection timeout", "trace_id": "abc123def456", ...}

↓ HTTP Push to Loki (batched)
│
Loki /loki/api/v1/push
  streams:
    - labels: '{service="ryot", level="error", tier="core", ...}'
      entries:
        - ts: 1705317045123000000
          line: '{"level": "error", ...}'

↓ Loki Ingestion
│
├─ Verify rate limits: ✓
├─ Accept stream: ✓
├─ Add to buffer: ✓
│
↓ Chunk Assembly (when idle period exceeded or chunk full)
│
├─ Group lines by stream labels
├─ Compress with Snappy
├─ Create immutable chunk object
├─ Store in S3 (gzipped further)
│
↓ Index Entry (BoltDB Shipper)
│
├─ Index key: service:ryot, level:error, [date]
├─ Points to: S3://loki-chunks/[chunk-id]
├─ TTL: 24 hours local, then ship to S3
│
↓ Query Time (LogQL)
│
Query: {service="ryot", level="error"} | json | duration_ms > 5000
  1. Look up index for matching streams
  2. Fetch chunks from cache (if hit)
  3. Decompress and parse JSON
  4. Filter by duration_ms > 5000
  5. Return results to Grafana

Final Output in Grafana:
  "2024-01-15T10:30:45.123Z | {"level": "error", "msg": "Database connection timeout", ...}"
```

---

## Storage Tiering Diagram

```
Time Axis →
├────────────────────────────────────────────────────────────────────┐
│                         LOG RETENTION TIMELINE                    │
├────────────────────────────────────────────────────────────────────┤
│
│  Now
│   ↓
│   │←─────────────────────────────────────────────────────────────→│
│   │←─── HOT TIER ───→│←─── WARM TIER ───→│←─── COLD TIER ────────→│
│   │   (0-7 days)     │   (7-30 days)     │    (30-365 days)       │
│   │                  │                   │                        │
│   │ ┌──────────────┐ │ ┌──────────────┐  │ ┌──────────────────┐   │
│   │ │ SSD Storage  │ │ │ S3 Storage   │  │ │ Glacier Archive  │   │
│   │ │ (300Gi)      │ │ │ (unlimited)  │  │ │ (unlimited)      │   │
│   │ │              │ │ │              │  │ │                  │   │
│   │ │ Fast access  │ │ │ Slow access  │  │ │ Very slow access │   │
│   │ │ High cost    │ │ │ Medium cost  │  │ │ Low cost         │   │
│   │ │ ~$30/month   │ │ │ ~$3/month    │  │ │ ~$1/month        │   │
│   │ │              │ │ │              │  │ │                  │   │
│   │ │ Active index │ │ │ Shipped      │  │ │ Archive copies   │   │
│   │ │ + chunks     │ │ │ index + all  │  │ │ (optional)       │   │
│   │ │              │ │ │ chunks       │  │ │                  │   │
│   │ └──────────────┘ │ └──────────────┘  │ └──────────────────┘   │
│   │                  │                   │                        │
│   └──────────────────┴───────────────────┴────────────────────────┘
│
│ Queries:
│ • P99 latency:  50-200ms (hot) | 500ms-2s (warm) | 5-30s (cold)
│ • Cache hit:    90% (hot) | 70% (warm) | 10% (cold)
│ • Access:       Real-time | Historical | Compliance/Audit
│
│ Promotion:
│ • Hot → Warm: Automatic after 7 days (or configurable)
│ • Warm → Cold: Optional via Glacier policy
│ • BoltDB indices stay local 24h, then ship to S3

```

---

## Query Execution Path

```
Grafana Dashboard (LogQL Query)
│
│ Query: {service="ryot", level="error"} | json | rate(duration_ms[5m])
│
↓
HTTP Request to Loki API: POST /loki/api/v1/query_range
│  params:
│    query: {service="ryot", level="error"} | json | rate(duration_ms[5m])
│    start: 1705316745123000000 (5m ago)
│    end: 1705317045123000000 (now)
│    step: 60s
│
↓
Loki Querier Component
│
├─ 1. Parse LogQL expression
│    ├─ Label matchers: {service="ryot", level="error"}
│    ├─ Pipeline: | json | rate(duration_ms[5m])
│    └─ Time range: [now-5m, now]
│
├─ 2. Query index store (BoltDB)
│    ├─ Look up streams matching: service="ryot" AND level="error"
│    ├─ Find chunk references for time range
│    └─ Return: [chunk_id_1, chunk_id_2, ..., chunk_id_n]
│
├─ 3. Fetch chunks
│    ├─ Check Memcached for cached results: CACHE HIT/MISS
│    │   ├─ HIT: Return from cache (1ms)
│    │   └─ MISS: Fetch from S3 (500ms)
│    │
│    ├─ Decompress chunks (Snappy decompression)
│    └─ Return: Raw log entries
│
├─ 4. Apply pipeline
│    ├─ Filter by labels: {service="ryot", level="error"}
│    ├─ Parse JSON: Extract duration_ms field
│    ├─ Apply rate(): Calculate rate per step
│    └─ Return: Time series
│
├─ 5. Format results
│    ├─ Convert to Prometheus format
│    ├─ Add metadata (query time, result count)
│    └─ Return JSON response
│
├─ 6. Cache results (Memcached)
│    └─ Store for 1 hour for identical future queries
│
↓
Response to Grafana
│
Grafana Graph Panel
│
├─ X-axis: Time (5 minutes)
├─ Y-axis: Error rate (errors/second)
└─ Line plot showing error rate over time

Performance Metrics:
├─ Total query time: 150-500ms (depending on cache hit)
├─ Index lookup: 10-50ms
├─ Chunk fetch: 1ms (cache) or 200-500ms (S3)
├─ Pipeline execution: 50-100ms
└─ Cache write: 5-10ms
```

---

## Namespace & RBAC Isolation

```
Kubernetes Cluster
│
├─ monitoring namespace (Observability Stack)
│  │
│  ├─ ServiceAccount: loki
│  │  ├─ ClusterRole: [get, list, watch] on configmaps, pods
│  │  └─ ClusterRoleBinding: loki → ClusterRole:loki
│  │
│  ├─ ServiceAccount: promtail
│  │  ├─ ClusterRole: [get, list, watch] on pods, pods/log, nodes
│  │  └─ ClusterRoleBinding: promtail → ClusterRole:promtail
│  │
│  ├─ Secret: loki-s3-creds
│  │  ├─ access_key: (S3 credentials)
│  │  └─ secret_key: (S3 credentials)
│  │
│  ├─ ConfigMap: loki-config
│  │  └─ loki-config.yaml: (1400+ lines, mounting in Pod)
│  │
│  ├─ ConfigMap: promtail-config
│  │  └─ promtail-config.yaml: (550+ lines, mounting in Pod)
│  │
│  ├─ StorageClass: fast-ssd
│  │  └─ Used by Loki StatefulSet PVCs
│  │
│  ├─ PersistentVolumeClaim: loki-0, loki-1, loki-2
│  │  └─ 100Gi each, backed by fast-ssd StorageClass
│  │
│  ├─ StatefulSet: loki (3 replicas)
│  │  ├─ Pods: loki-0, loki-1, loki-2
│  │  ├─ Containers: loki
│  │  ├─ ServiceAccount: loki
│  │  ├─ Volumes: config (ConfigMap), data (PVC), secrets (Secret)
│  │  └─ Env: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (from Secret)
│  │
│  ├─ DaemonSet: promtail
│  │  ├─ Pods: promtail-node-1, promtail-node-2, ..., promtail-node-n
│  │  ├─ Containers: promtail
│  │  ├─ ServiceAccount: promtail
│  │  ├─ Volumes: config (ConfigMap), positions (emptyDir), mounts (host)
│  │  └─ Tolerations: tolerates all node taints
│  │
│  ├─ Deployment: memcached (3 replicas)
│  │  ├─ Pods: memcached-0, memcached-1, memcached-2
│  │  ├─ Containers: memcached
│  │  └─ Volumes: cache (emptyDir)
│  │
│  ├─ Service: loki (LoadBalancer or NodePort)
│  │  ├─ Selector: app=loki
│  │  ├─ Port 3100: HTTP
│  │  └─ Port 9096: gRPC
│  │
│  ├─ Service: loki-headless (ClusterIP: None)
│  │  ├─ Selector: app=loki
│  │  └─ Used by StatefulSet (stable DNS)
│  │
│  ├─ Service: memcached-service
│  │  ├─ Selector: app=memcached
│  │  └─ Port 11211: Memcached
│  │
│  ├─ NetworkPolicy: loki-network-policy
│  │  ├─ Ingress:
│  │  │  ├─ From: promtail (port 3100, 9096)
│  │  │  ├─ From: grafana (port 3100)
│  │  │  └─ From: alertmanager (port 3100)
│  │  └─ Egress:
│  │     ├─ To: kube-dns (DNS)
│  │     ├─ To: S3 backend (port 443)
│  │     ├─ To: memcached (port 11211)
│  │     └─ To: alertmanager (port 9093)
│  │
│  └─ NetworkPolicy: promtail-network-policy
│     ├─ Ingress: From prometheus (port 9080)
│     └─ Egress: To loki (port 3100), DNS, S3
│
├─ neurectomy namespace (Application Services)
│  │
│  ├─ Deployment: ryot
│  │  └─ Labels: app=ryot, tier=core
│  │
│  ├─ Deployment: sigmalang
│  │  └─ Labels: app=sigmalang, tier=core
│  │
│  ├─ Deployment: sigmavault
│  │  └─ Labels: app=sigmavault, tier=core
│  │
│  └─ Deployment: agents
│     └─ Labels: app=agents, tier=agents

Service-to-Service Communication (through NetworkPolicies):
  promtail → loki: Port 3100 (HTTPS if configured)
  grafana → loki: Port 3100 (HTTPS if configured)
  alertmanager → loki: Port 3100 (Query for rules firing)
  prometheus → promtail: Port 9080 (Metrics scraping)
```

---

## Monitoring Stack Architecture

```
Complete Observability Pipeline (Phase 18E)

                    ┌─────────────────────────────────────┐
                    │   DISTRIBUTED TRACING (Jaeger)      │
                    │   • Service dependency mapping      │
                    │   • Request flow visualization      │
                    │   • Latency analysis per span       │
                    │   • Port: 16686                     │
                    └─────────────────────────────────────┘
                                    ↑
                                    │
                ┌─────────────────────────────────────┐
                │  UNIFIED OBSERVABILITY FRONTEND     │
                │         (Grafana)                   │
                │                                     │
                │  • Log panel: LogQL queries         │
                │  • Metric panel: Prometheus queries │
                │  • Trace panel: Jaeger integration  │
                │  • Correlation: Logs↔Metrics↔Traces│
                │  • Port: 3000                       │
                └─────────────────────────────────────┘
                    ↑           ↑           ↑
                    │           │           │
        ┌───────────┴───────┬──┴────┬──────┴──────────┐
        │                   │       │                 │
        ↑                   ↑       ↑                 ↑

    ┌─────────────────┐ ┌──────────────┐ ┌──────────────────┐
    │  LOGS (Loki)    │ │ METRICS      │ │ TRACES (Jaeger)  │
    │                 │ │ (Prometheus) │ │                  │
    │ • Storage:      │ │              │ │ • Collectors:    │
    │   BoltDB +      │ │ • Storage:   │ │   OTEL, Jaeger   │
    │   S3 + Cache    │ │   TSDB       │ │                  │
    │                 │ │              │ │ • Frontend:      │
    │ • Collection:   │ │ • Collection:│ │   Jaeger UI      │
    │   Promtail      │ │   Prometheus│ │                  │
    │                 │ │   scrapers   │ │ • Port: 6831,   │
    │ • Retention:    │ │              │ │   14268, 16686  │
    │   30-60 days    │ │ • Retention: │ │                  │
    │                 │ │   365 days   │ │ • Retention:    │
    │ • Port: 3100    │ │              │ │   7 days        │
    │                 │ │ • Port: 9090 │ │                 │
    └─────────────────┘ │              │ └──────────────────┘
                        │ • Alerting:  │
                        │   AlertMgr   │
                        │              │
                        │ • Retention: │
                        │   365 days   │
                        └──────────────┘
            ↑                   ↑                 ↑
            │                   │                 │
            └───────────────────┴─────────────────┘
                                │
                ┌───────────────┴────────────────┐
                │                                │
                ↑                                ↑
    ┌───────────────────────┐       ┌───────────────────────┐
    │  APPLICATION SERVICES │       │   INFRASTRUCTURE      │
    │                       │       │                       │
    │ • Ryot                │       │ • Kubernetes events   │
    │ • ΣLANG               │       │ • Node metrics        │
    │ • ΣVAULT              │       │ • Network metrics     │
    │ • Agents              │       │ • Storage metrics     │
    │                       │       │                       │
    │ Instrumentation:      │       │ Exporters:            │
    │ • Logs (JSON)         │       │ • Node exporter       │
    │ • Structured tracing  │       │ • kube-state-metrics  │
    │ • OpenTelemetry       │       │ • Kubelet metrics     │
    │                       │       │                       │
    └───────────────────────┘       └───────────────────────┘
```

---

**Phase 18E Architecture Complete** ✅

For operational guidance, see:

- [PHASE-18E-LOKI-INTEGRATION.md](PHASE-18E-LOKI-INTEGRATION.md) - Complete guide
- [PHASE-18E-LOKI-QUICKSTART.md](PHASE-18E-LOKI-QUICKSTART.md) - Quick reference
- [PHASE-18E-LOKI-VALIDATION-CHECKLIST.md](PHASE-18E-LOKI-VALIDATION-CHECKLIST.md) - Deployment validation
