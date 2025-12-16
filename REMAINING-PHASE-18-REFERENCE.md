# Phase 18: Remaining Implementations - Quick Reference

## Files Already Created (4)
1. ✅ 00-README-PHASE-18.md - Master guide
2. ✅ PHASE-18-QUICK-START.md - Quick start guide
3. ✅ PHASE-18A-1-PROMETHEUS-SETUP.md - Full monitoring stack deployment
4. ✅ PHASE-18A-2-NEURECTOMY-METRICS.md - API metrics implementation

## Remaining Files Needed (10)

### Phase 18A: Metrics (4 more files)

#### PHASE-18A-3-RYOT-METRICS.md (Ryot LLM)
**File**: `ryot/monitoring/metrics.py`
**Key Metrics**:
- `ryot_inference_requests_total` (Counter)
- `ryot_inference_duration_seconds` (Histogram with buckets: 0.1, 0.5, 1, 2, 5, 10, 30)
- `ryot_tokens_generated` (Counter)
- `ryot_model_loading_duration` (Gauge)
- `ryot_batch_size` (Histogram)
- `ryot_gpu_memory_usage_bytes` (Gauge)
**Annotations**: Add to Kubernetes deployment:
```yaml
annotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "8000"
  prometheus.io/path: "/metrics"
```

#### PHASE-18A-4-SIGMALANG-METRICS.md (ΣLANG)
**File**: `sigmalang/monitoring/metrics.py`
**Key Metrics**:
- `sigmalang_compression_requests_total` (Counter)
- `sigmalang_compression_ratio` (Histogram with buckets: 5, 10, 15, 20, 25, 30, 50)
- `sigmalang_compression_duration_seconds` (Histogram)
- `sigmalang_original_size_bytes` (Histogram)
- `sigmalang_compressed_size_bytes` (Histogram)
- `sigmalang_decompression_requests_total` (Counter)
**Decorator**: `@track_compression_operation`

#### PHASE-18A-5-SIGMAVAULT-METRICS.md (ΣVAULT)
**File**: `sigmavault/monitoring/metrics.py`
**Key Metrics**:
- `sigmavault_operations_total` (Counter with labels: operation=[store, retrieve, delete])
- `sigmavault_operation_duration_seconds` (Histogram)
- `sigmavault_storage_bytes_total` (Gauge)
- `sigmavault_objects_total` (Gauge)
- `sigmavault_encryption_duration_seconds` (Histogram)
- `sigmavault_snapshot_operations_total` (Counter)
**Integration**: FUSE filesystem hooks for automatic tracking

#### PHASE-18A-6-AGENT-METRICS.md (Elite Agents)
**File**: `agents/monitoring/metrics.py`
**Key Metrics**:
- `agents_active_total` (Gauge)
- `agents_tasks_assigned_total` (Counter with labels: agent_id, task_type)
- `agents_task_duration_seconds` (Histogram with labels: agent_id, task_type)
- `agents_utilization_ratio` (Gauge with label: agent_id)
- `agents_failures_total` (Counter with labels: agent_id, error_type)
- `agents_recovery_attempts_total` (Counter)
- `agents_health_status` (Gauge: 0=healthy, 1=degraded, 2=failed)
**Special**: Track all 40 agents individually

---

### Phase 18B: Distributed Tracing (2 files)

#### PHASE-18B-1-OPENTELEMETRY.md (Infrastructure)
**File**: `infrastructure/monitoring/otel-collector-config.yaml`
**Components**:
- OpenTelemetry Collector deployment
- Jaeger backend for trace storage
- Trace sampling configuration (10% of requests)
**Exporters**: Jaeger, Prometheus, Console
**Processors**: Batch, memory_limiter, span
**Example config**:
```yaml
receivers:
  otlp:
    protocols:
      grpc:
      http:
processors:
  batch:
    timeout: 10s
    send_batch_size: 1024
exporters:
  jaeger:
    endpoint: jaeger:14250
  prometheus:
    endpoint: "0.0.0.0:8889"
```

#### PHASE-18B-2-DISTRIBUTED-TRACING.md (Neurectomy)
**File**: `neurectomy/tracing/tracer.py`
**Library**: opentelemetry-api, opentelemetry-sdk, opentelemetry-instrumentation-fastapi
**Auto-instrumentation**: FastAPI requests
**Custom spans**:
```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("ryot_inference"):
    result = await ryot_client.generate(prompt)
```
**Propagation**: Inject trace context in service-to-service calls

---

### Phase 18C: Logging (2 files)

#### PHASE-18C-1-LOGGING-STACK.md (Infrastructure)
**File**: `infrastructure/kubernetes/monitoring/loki-stack.yaml`
**Stack**: Loki + Promtail + Grafana integration
**Storage**: S3 backend for long-term retention
**Retention**: 30 days in Loki, 90 days in S3
**Promtail config**: Scrape all pod logs with labels
```yaml
scrape_configs:
  - job_name: kubernetes-pods
    kubernetes_sd_configs:
      - role: pod
    pipeline_stages:
      - docker: {}
      - labels:
          app:
          namespace:
```

#### PHASE-18C-2-LOG-FORMATTING.md (All Projects)
**Files**: Update logging config in each project
**Format**: JSON structured logging
**Fields**: timestamp, level, service, trace_id, span_id, message, extra
**Example**:
```python
import structlog

logger = structlog.get_logger()
logger.info("request_processed", 
            endpoint="/v1/complete",
            duration_ms=123.45,
            status_code=200)
```
**Output**: `{"timestamp": "...", "level": "info", "service": "neurectomy", "message": "request_processed", ...}`

---

### Phase 18D: Alerting (2 files)

#### PHASE-18D-1-ALERT-MANAGER.md (Infrastructure)
**File**: `infrastructure/monitoring/alertmanager-config.yaml`
**Alert Rules**:
- High error rate (> 5% for 5 minutes)
- High latency (p95 > 5s for 10 minutes)
- Service down (up == 0 for 1 minute)
- High CPU usage (> 90% for 5 minutes)
- Low disk space (< 10% free)
- Agent failures (> 3 in 5 minutes)
**Routes**: PagerDuty (critical), Slack (warning), Email (info)
**Silences**: Maintenance windows configuration

#### PHASE-18D-2-INCIDENT-RESPONSE.md (Infrastructure)
**File**: `infrastructure/runbooks/incident-response.md`
**Playbooks**:
1. High error rate → Check circuit breakers, recent deployments
2. High latency → Profile slow endpoints, check database
3. Service down → Check logs, restart pod, escalate
4. Out of memory → Check for memory leaks, scale up
5. Agent failure → Supervisor auto-recovery, manual intervention
**Escalation**: L1 (on-call) → L2 (engineering) → L3 (architect)
**Automation**: Auto-scaling, auto-restart, auto-rollback scripts

---

## Implementation Priority

### Week 1 (Days 1-5): Foundation
1. PHASE-18A-3-RYOT-METRICS.md
2. PHASE-18A-4-SIGMALANG-METRICS.md
3. PHASE-18A-5-SIGMAVAULT-METRICS.md
4. PHASE-18A-6-AGENT-METRICS.md

### Week 2 (Days 6-10): Advanced
5. PHASE-18B-1-OPENTELEMETRY.md
6. PHASE-18B-2-DISTRIBUTED-TRACING.md
7. PHASE-18C-1-LOGGING-STACK.md
8. PHASE-18C-2-LOG-FORMATTING.md

### Week 3 (Days 11-15): Operations
9. PHASE-18D-1-ALERT-MANAGER.md
10. PHASE-18D-2-INCIDENT-RESPONSE.md

## Quick Implementation Notes

### For Metrics (18A-3 through 18A-6)
All follow same pattern:
1. Import prometheus_client
2. Define Counter/Histogram/Gauge metrics
3. Add decorators to functions
4. Mount /metrics endpoint
5. Update Kubernetes annotations

### For Tracing (18B)
1. Install OpenTelemetry packages
2. Configure auto-instrumentation
3. Add custom spans for important operations
4. Deploy Jaeger backend
5. View traces in Jaeger UI

### For Logging (18C)
1. Deploy Loki stack
2. Configure Promtail to scrape logs
3. Switch to structured JSON logging
4. Add trace_id to correlate with traces
5. View logs in Grafana

### For Alerting (18D)
1. Define alert rules in Prometheus
2. Deploy AlertManager
3. Configure notification channels
4. Create runbooks for common issues
5. Test alert routing

## Verification Commands

```bash
# Check metrics endpoint
curl http://neurectomy-api:8000/metrics

# View Prometheus targets
kubectl port-forward -n monitoring svc/prometheus 9090:9090
# Visit: http://localhost:9090/targets

# View Grafana dashboards
kubectl port-forward -n monitoring svc/grafana 3000:80
# Visit: http://localhost:3000

# View Jaeger traces
kubectl port-forward -n monitoring svc/jaeger 16686:16686
# Visit: http://localhost:16686

# Check Loki logs
kubectl port-forward -n monitoring svc/loki 3100:3100
curl http://localhost:3100/ready
```

## Expected Outcomes

After implementing all of Phase 18:
- ✅ All services emitting metrics to Prometheus
- ✅ Grafana showing 10+ dashboards
- ✅ Distributed traces visible in Jaeger
- ✅ Logs aggregated in Loki/Grafana
- ✅ Alerts configured and routing correctly
- ✅ Runbooks documented for incidents
- ✅ Full observability across entire stack
- ✅ Ready for production operations

## Cost Considerations

**Storage**:
- Prometheus: ~100GB for 30 days retention
- Loki: ~50GB for 30 days + S3 for long-term
- Jaeger: ~20GB for traces

**Compute**:
- Prometheus: 4GB RAM, 2 CPU
- Grafana: 1GB RAM, 0.5 CPU
- Loki: 2GB RAM, 1 CPU
- Jaeger: 2GB RAM, 1 CPU
- **Total**: ~9GB RAM, 4.5 CPU

**Estimated monthly cost**: $150-300 (AWS/GCP)
