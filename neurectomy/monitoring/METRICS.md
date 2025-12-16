# Neurectomy API Metrics - PHASE 18A-2

Complete Prometheus metrics implementation for Neurectomy API with comprehensive monitoring across all integrations.

## What's Implemented

### HTTP Request Metrics

- **Total Requests** (`neurectomy_http_requests_total`)
  - Tracks by method, endpoint, status code
  - Identifies error patterns and traffic patterns
- **Request Latency** (`neurectomy_http_request_duration_seconds`)
  - Histogram with millisecond-level precision
  - Buckets: 10ms, 50ms, 100ms, 500ms, 1s, 2.5s, 5s, 10s
  - Enables latency analysis (p50, p95, p99)

- **Request/Response Sizes**
  - Tracks payload sizes
  - Identifies large request/response patterns

### Ryot LLM Integration

- **Request Tracking** - Success/error/timeout classification
- **Latency Monitoring** - Sub-second to 30-second range
- **Token Generation** - Per-model token counting
- **Model Loading** - Startup performance
- **GPU Memory** - Resource utilization
- **Batch Sizing** - Inference patterns

### ΣLANG Compression

- **Compression Requests** - Operation counting
- **Compression Ratio** - 5x to 50x tracking
- **Performance** - Duration from milliseconds to 10 seconds
- **Size Tracking** - Original and compressed sizes
- **Decompression** - Reverse operation monitoring

### ΣVAULT Storage

- **Storage Operations** - Store/retrieve/delete counting
- **Operation Latency** - Millisecond to 10-second precision
- **Storage Capacity** - Total bytes and object count
- **Encryption** - Security operation timing
- **Snapshots** - Backup/restore tracking

### Circuit Breaker Resilience

- **State Tracking** - Closed (0), Open (1), Half-Open (2)
- **Failure Counting** - Per-service failure rates
- **State Transitions** - Change tracking for alerting

### Business Metrics

- **Active Users** - Real-time user count
- **Token Generation** - Model-specific token consumption
- **API Keys** - Active credentials tracking
- **Cost Per Request** - Endpoint-specific billing
- **Per-User Request Distribution** - Usage patterns

## File Structure

```
neurectomy/monitoring/
├── __init__.py              # Module exports
├── metrics.py               # Complete metrics implementation (700+ lines)
├── test_metrics.py          # Comprehensive test suite
└── METRICS.md              # This documentation
```

## Integration with Neurectomy API

### Step 1: Add Middleware to FastAPI App

```python
# In neurectomy/main.py or neurectomy/api.py

from prometheus_client import make_asgi_app
from neurectomy.monitoring.metrics import MetricsMiddleware, system_info

# Initialize system info
system_info.info({
    'version': '1.0.0',
    'environment': 'production',
    'cluster': 'neurectomy-prod'
})

# Add metrics middleware (should be first for accurate latency)
app.add_middleware(MetricsMiddleware)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

### Step 2: Add Kubernetes Annotations

```yaml
# In Kubernetes deployment
annotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "8000"
  prometheus.io/path: "/metrics"
```

### Step 3: Add Service Integration Tracking

```python
from neurectomy.monitoring.metrics import (
    track_ryot_request,
    increment_tokens_generated,
    track_compression,
    track_storage_operation
)

# Track Ryot LLM calls
@track_ryot_request
async def call_ryot(prompt: str, model: str = "default"):
    result = await ryot_client.generate(prompt, model=model)
    increment_tokens_generated(model, result.token_count)
    return result

# Track compression
@track_compression
async def compress_data(data: bytes):
    return await sigmalang_client.compress(data)

# Track storage
@track_storage_operation("store")
async def store_encrypted(path: str, data: bytes):
    return await sigmavault_client.store_encrypted(path, data)
```

## Metrics Details

### Counter Metrics (Cumulative)

- `neurectomy_http_requests_total` - All HTTP requests
- `neurectomy_ryot_requests_total` - Ryot API calls
- `neurectomy_ryot_tokens_generated_total` - Token count
- `neurectomy_sigmalang_compression_requests_total` - Compression operations
- `neurectomy_sigmavault_operations_total` - Storage operations
- `neurectomy_circuit_breaker_failures_total` - Circuit breaker failures
- `neurectomy_tokens_generated_total` - Total tokens

### Histogram Metrics (Distributions)

- `neurectomy_http_request_duration_seconds` - Request latency
- `neurectomy_http_request_size_bytes` - Request payload size
- `neurectomy_http_response_size_bytes` - Response payload size
- `neurectomy_ryot_request_duration_seconds` - Ryot latency
- `neurectomy_ryot_batch_size` - Batch sizes
- `neurectomy_sigmalang_compression_ratio` - Compression effectiveness
- `neurectomy_sigmalang_compression_duration_seconds` - Compression speed
- `neurectomy_sigmavault_operation_duration_seconds` - Storage speed
- `neurectomy_sigmavault_encryption_duration_seconds` - Encryption overhead
- `neurectomy_requests_per_user` - Usage distribution

### Gauge Metrics (Point-in-Time)

- `neurectomy_circuit_breaker_state` - Resilience status
- `neurectomy_ryot_model_loading_duration_seconds` - Startup time
- `neurectomy_ryot_gpu_memory_usage_bytes` - Resource usage
- `neurectomy_active_users` - Current user count
- `neurectomy_api_keys_active` - Active credentials
- `neurectomy_sigmavault_storage_bytes_total` - Capacity usage
- `neurectomy_sigmavault_objects_total` - Object count
- `neurectomy_cost_per_request` - Billing rate

## Prometheus Queries

### Find High-Error Endpoints

```promql
rate(neurectomy_http_requests_total{status=~"5.."}[5m]) > 0.01
```

### Find Slow Endpoints

```promql
histogram_quantile(0.95,
  sum(rate(neurectomy_http_request_duration_seconds_bucket[5m])) by (endpoint, le)
) > 1
```

### Token Generation Rate

```promql
rate(neurectomy_ryot_tokens_generated_total[1m])
```

### Average Compression Ratio

```promql
rate(neurectomy_sigmalang_original_size_bytes_sum[5m]) /
rate(neurectomy_sigmalang_compressed_size_bytes_sum[5m])
```

### Storage Latency

```promql
histogram_quantile(0.95,
  sum(rate(neurectomy_sigmavault_operation_duration_seconds_bucket[5m])) by (operation, le)
)
```

### Circuit Breaker Status

```promql
neurectomy_circuit_breaker_state > 0
```

## Grafana Dashboard

Access at: `http://grafana:3000` (after deployment)

**Default Dashboards:**

1. **System Overview** - All requests, error rate, latency, resources
2. **Service Health** - Individual service metrics
3. **Business Metrics** - Active users, tokens, costs
4. **Agent Collective** - All agent status and utilization

## Alert Rules

Configured in `infrastructure/monitoring/alerts.yml`:

- `HighErrorRate` - Error rate > 5% for 5 minutes
- `HighLatency` - p95 latency > 5s for 10 minutes
- `RyotLLMHighErrorRate` - > 10% errors
- `RyotLLMHighLatency` - p95 > 30s
- `CircuitBreakerOpen` - Service circuit breaker open
- `HighCircuitBreakerFailures` - > 10% failure rate
- `AgentDown` - Agent unhealthy for 2+ minutes
- `PodCPUUsageHigh` - > 80% CPU for 10 minutes
- `PodMemoryUsageHigh` - > 85% memory for 10 minutes

## Performance Impact

- **Middleware Overhead**: ~1-2ms per request
- **Memory Per Metric**: ~1KB base + ~100 bytes per label combination
- **Cardinality Controlled**: Path normalization prevents explosion
- **Async-Safe**: All operations are async-compatible

## Testing

```bash
# Run tests
pytest neurectomy/monitoring/test_metrics.py -v

# Test metrics endpoint
curl http://localhost:8000/metrics

# Check specific metric
curl http://localhost:8000/metrics | grep neurectomy_http_requests_total
```

## Dependencies

```
prometheus-client>=0.17.0
```

## Next Steps

- Deploy Prometheus (Phase 18A-1) ✅
- Deploy Neurectomy Metrics (Phase 18A-2) ✅
- Add Ryot LLM metrics (Phase 18A-3)
- Add ΣLANG metrics (Phase 18A-4)
- Add ΣVAULT metrics (Phase 18A-5)
- Add agent metrics (Phase 18A-6)
- Distributed tracing (Phase 18B)
- Centralized logging (Phase 18C)
- Alerting setup (Phase 18D)
- Performance optimization (Phase 18E)
