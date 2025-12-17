# Ryot LLM Metrics: Implementation Guide

**Companion to**: [RYOT_METRICS_DESIGN.md](RYOT_METRICS_DESIGN.md)  
**Status**: Ready for Implementation  
**Target**: Python + Prometheus

---

## Quick Reference: Metric Categories

```
COUNTER METRICS (monotonically increasing):
  ✓ ryot_inference_requests_total        → Total requests
  ✓ ryot_inference_errors_total          → Error count by type
  ✓ ryot_tokens_generated_total          → Tokens generated
  ✓ ryot_model_load_attempts_total       → Model load attempts
  ✓ ryot_model_cache_evictions_total     → Cache evictions

HISTOGRAM METRICS (track distributions):
  ✓ ryot_inference_latency_seconds       → End-to-end latency
  ✓ ryot_ttft_latency_seconds            → Time to first token
  ✓ ryot_model_loading_time_seconds      → Model initialization
  ✓ ryot_batch_size_distribution         → Batch size patterns
  ✓ ryot_prompt_cache_effectiveness      → Cache reuse

GAUGE METRICS (current state):
  ✓ ryot_active_inference_requests       → In-flight requests
  ✓ ryot_gpu_memory_usage_bytes          → GPU memory usage
  ✓ ryot_gpu_memory_percentage           → Memory utilization %
  ✓ ryot_gpu_utilization_percent         → GPU compute %
  ✓ ryot_kv_cache_size_bytes             → KV cache size
  ✓ ryot_queue_depth                     → Request queue length
  ✓ ryot_tokens_per_second               → Throughput rate
  ✓ ryot_kv_cache_hit_ratio              → Cache effectiveness
```

---

## Complete Implementation Example

### Step 1: Initialize Metrics Module

**File**: `neurectomy/monitoring/ryot_metrics.py`

```python
"""
Ryot LLM Inference Metrics
Complete metrics module for production observability
"""

from prometheus_client import (
    Counter, Histogram, Gauge, Info,
    CollectorRegistry, generate_latest
)
from typing import Optional, Dict, Any
import logging
import functools
import time

logger = logging.getLogger(__name__)

# ============================================================================
# COUNTER METRICS
# ============================================================================

ryot_inference_requests_total = Counter(
    'ryot_inference_requests_total',
    'Total HTTP requests to Ryot inference',
    ['model', 'status', 'endpoint'],
    registry=None  # Use default registry
)

ryot_inference_errors_total = Counter(
    'ryot_inference_errors_total',
    'Total inference errors by type',
    ['error_type', 'model', 'severity'],
)

ryot_tokens_generated_total = Counter(
    'ryot_tokens_generated_total',
    'Total tokens generated',
    ['model', 'token_type', 'generation_mode'],
)

ryot_model_load_attempts_total = Counter(
    'ryot_model_load_attempts_total',
    'Model loading attempts',
    ['model', 'result', 'load_type'],
)

ryot_model_cache_evictions_total = Counter(
    'ryot_model_cache_evictions_total',
    'Model cache evictions',
    ['model', 'eviction_reason'],
)

# ============================================================================
# HISTOGRAM METRICS - WITH PRODUCTION BUCKETS
# ============================================================================

ryot_inference_latency_seconds = Histogram(
    'ryot_inference_latency_seconds',
    'End-to-end inference latency',
    ['model', 'batch_size', 'generation_mode'],
    buckets=(
        0.01,    # 10ms
        0.025,   # 25ms
        0.05,    # 50ms (TTFT target)
        0.1,     # 100ms
        0.25,    # 250ms
        0.5,     # 500ms
        1.0,     # 1s
        2.5,     # 2.5s
        5.0,     # 5s
        10.0,    # 10s
        30.0,    # 30s
        float('inf')  # SLA violations
    )
)

ryot_ttft_latency_seconds = Histogram(
    'ryot_ttft_latency_seconds',
    'Time to first token (prompt processing + first token)',
    ['model', 'prompt_size'],
    buckets=(
        0.005,   # 5ms
        0.01,    # 10ms
        0.025,   # 25ms
        0.05,    # 50ms (target)
        0.1,     # 100ms
        0.25,    # 250ms
        0.5,     # 500ms
        1.0,     # 1s
        2.5,     # 2.5s
        float('inf')
    )
)

ryot_model_loading_time_seconds = Histogram(
    'ryot_model_loading_time_seconds',
    'Model initialization time',
    ['model', 'load_type', 'device_type'],
    buckets=(
        0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, float('inf')
    )
)

ryot_batch_size_distribution = Histogram(
    'ryot_batch_size_distribution',
    'Batch size patterns',
    ['model', 'request_type'],
    buckets=(1, 2, 4, 8, 16, 32, 64, float('inf'))
)

ryot_tokens_per_second_histogram = Histogram(
    'ryot_tokens_per_second',
    'Token generation throughput',
    ['model'],
    buckets=(10, 25, 50, 100, 150, 250, 400, 600, float('inf'))
)

ryot_kv_cache_effectiveness = Histogram(
    'ryot_kv_cache_effectiveness',
    'KV cache reuse effectiveness',
    ['model', 'context_type'],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

ryot_gpu_memory_distribution = Histogram(
    'ryot_gpu_memory_distribution',
    'GPU memory usage patterns',
    ['model'],
    buckets=(256_000_000, 512_000_000, 1_000_000_000, 2_000_000_000,
             4_000_000_000, 8_000_000_000, 16_000_000_000, float('inf'))
)

# ============================================================================
# GAUGE METRICS - CURRENT STATE
# ============================================================================

ryot_active_inference_requests = Gauge(
    'ryot_active_inference_requests',
    'Currently in-flight inference requests',
    ['model', 'request_type']
)

ryot_gpu_memory_bytes = Gauge(
    'ryot_gpu_memory_bytes',
    'GPU memory usage (bytes)',
    ['model', 'memory_type', 'device_id']
)

ryot_gpu_memory_percentage = Gauge(
    'ryot_gpu_memory_percentage',
    'GPU memory utilization (percentage)',
    ['model', 'device_id', 'batch_size']
)

ryot_gpu_utilization_percent = Gauge(
    'ryot_gpu_utilization_percent',
    'GPU compute utilization (percentage)',
    ['model', 'device_id', 'kernel_type']
)

ryot_kv_cache_size_bytes = Gauge(
    'ryot_kv_cache_size_bytes',
    'KV cache memory usage',
    ['model', 'context_length', 'batch_size']
)

ryot_queue_depth = Gauge(
    'ryot_queue_depth',
    'Number of queued inference requests',
    ['model', 'priority_level']
)

ryot_kv_cache_hit_ratio = Gauge(
    'ryot_kv_cache_hit_ratio',
    'KV cache hit ratio (0.0-1.0)',
    ['model', 'context_type']
)

ryot_tokens_per_second_gauge = Gauge(
    'ryot_tokens_per_second_gauge',
    'Real-time token generation rate',
    ['model', 'time_window']
)

# ============================================================================
# INFO METRIC
# ============================================================================

ryot_info = Info(
    'ryot_service',
    'Ryot LLM service information',
)

# ============================================================================
# CONTEXT MANAGERS & DECORATORS
# ============================================================================

class InferenceMetricsContext:
    """Context manager for tracking inference metrics"""

    def __init__(self, model: str, batch_size: int = 1, endpoint: str = "generate"):
        self.model = model
        self.batch_size = batch_size
        self.endpoint = endpoint
        self.start_time = None
        self.ttft_time = None
        self.token_count = 0
        self.status = "success"
        self.error_type = None

    def __enter__(self):
        self.start_time = time.time()
        ryot_active_inference_requests.labels(
            model=self.model,
            request_type="standard"
        ).inc()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        latency = time.time() - self.start_time

        # Record latency
        batch_bucket = self._categorize_batch_size()
        ryot_inference_latency_seconds.labels(
            model=self.model,
            batch_size=batch_bucket,
            generation_mode="standard"
        ).observe(latency)

        # Record request status
        if exc_type is not None:
            self.status = "error"
            error_name = exc_type.__name__
            error_type_map = {
                'OutOfMemoryError': 'oom',
                'TimeoutError': 'timeout',
                'RuntimeError': 'inference_error',
                'ValueError': 'invalid_request'
            }
            self.error_type = error_type_map.get(error_name, 'unknown_error')

        ryot_inference_requests_total.labels(
            model=self.model,
            status=self.status,
            endpoint=self.endpoint
        ).inc()

        # Record errors if occurred
        if self.status == "error":
            ryot_inference_errors_total.labels(
                error_type=self.error_type,
                model=self.model,
                severity="warning"
            ).inc()

        # Record token metrics
        if self.token_count > 0:
            ryot_tokens_generated_total.labels(
                model=self.model,
                token_type="completion",
                generation_mode="standard"
            ).inc(self.token_count)

        # Decrement active requests
        ryot_active_inference_requests.labels(
            model=self.model,
            request_type="standard"
        ).dec()

        return False  # Don't suppress exceptions

    def record_ttft(self):
        """Record time to first token"""
        if self.ttft_time is None:
            self.ttft_time = time.time() - self.start_time
            prompt_size = self._categorize_prompt_size()
            ryot_ttft_latency_seconds.labels(
                model=self.model,
                prompt_size=prompt_size
            ).observe(self.ttft_time)

    def record_tokens(self, count: int):
        """Record token generation"""
        self.token_count += count

    def _categorize_batch_size(self) -> str:
        if self.batch_size == 1:
            return "1"
        elif self.batch_size <= 4:
            return "2-4"
        elif self.batch_size <= 8:
            return "5-8"
        elif self.batch_size <= 16:
            return "9-16"
        elif self.batch_size <= 32:
            return "17-32"
        else:
            return "33+"

    def _categorize_prompt_size(self) -> str:
        # This would come from actual prompt in real implementation
        return "medium"


def track_inference_metric(model_name: Optional[str] = None):
    """Decorator for automatic inference metrics tracking"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            model = model_name or kwargs.get('model', 'unknown')
            batch_size = kwargs.get('batch_size', 1)

            with InferenceMetricsContext(model, batch_size) as metrics:
                try:
                    result = await func(*args, **kwargs)
                    if hasattr(result, 'tokens'):
                        metrics.record_tokens(len(result.tokens))
                    return result
                except Exception as e:
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            model = model_name or kwargs.get('model', 'unknown')
            batch_size = kwargs.get('batch_size', 1)

            with InferenceMetricsContext(model, batch_size) as metrics:
                try:
                    result = func(*args, **kwargs)
                    if hasattr(result, 'tokens'):
                        metrics.record_tokens(len(result.tokens))
                    return result
                except Exception as e:
                    raise

        # Return appropriate wrapper
        if hasattr(func, '__await__'):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# ============================================================================
# GPU METRICS HELPER
# ============================================================================

class GPUMetricsCollector:
    """Collect GPU-specific metrics"""

    def __init__(self):
        try:
            import pynvml
            self.pynvml = pynvml
            self.pynvml.nvmlInit()
            self.available = True
        except ImportError:
            logger.warning("pynvml not available - GPU metrics disabled")
            self.available = False

    def update_gpu_memory_metrics(self, model: str, device_id: int = 0):
        """Update GPU memory metrics for a model"""
        if not self.available:
            return

        try:
            handle = self.pynvml.nvmlDeviceGetHandleByIndex(device_id)
            mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)

            # Memory metrics
            ryot_gpu_memory_bytes.labels(
                model=model,
                memory_type="used",
                device_id=str(device_id)
            ).set(mem_info.used)

            memory_percentage = (mem_info.used / mem_info.total) * 100
            ryot_gpu_memory_percentage.labels(
                model=model,
                device_id=str(device_id),
                batch_size="current"
            ).set(memory_percentage)

            # GPU utilization
            ryot_gpu_utilization_percent.labels(
                model=model,
                device_id=str(device_id),
                kernel_type="mixed"
            ).set(util.gpu)

        except Exception as e:
            logger.error(f"Error collecting GPU metrics: {e}")

    def get_gpu_memory_info(self, device_id: int = 0) -> Dict[str, int]:
        """Get current GPU memory info"""
        if not self.available:
            return {}

        try:
            handle = self.pynvml.nvmlDeviceGetHandleByIndex(device_id)
            mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)

            return {
                'total': mem_info.total,
                'used': mem_info.used,
                'free': mem_info.free,
                'percent_used': (mem_info.used / mem_info.total) * 100
            }
        except Exception as e:
            logger.error(f"Error getting GPU memory info: {e}")
            return {}


# ============================================================================
# METRIC INITIALIZATION
# ============================================================================

def initialize_metrics(service_version: str, service_name: str = "ryot"):
    """Initialize metrics with service info"""
    ryot_info.info({
        'version': service_version,
        'service': service_name,
        'component': 'inference'
    })
    logger.info("Ryot metrics initialized")


def get_metrics_snapshot() -> Dict[str, Any]:
    """Get current metrics snapshot for logging/debugging"""
    return {
        'active_requests': dict(ryot_active_inference_requests._metrics),
        'queue_depth': dict(ryot_queue_depth._metrics),
        'gpu_memory_percent': dict(ryot_gpu_memory_percentage._metrics),
    }
```

---

### Step 2: FastAPI Integration

**File**: `neurectomy/api/middleware/metrics_middleware.py`

```python
"""
FastAPI middleware for automatic metrics collection
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse
from neurectomy.monitoring.ryot_metrics import (
    InferenceMetricsContext,
    ryot_inference_latency_seconds,
    ryot_inference_requests_total,
)
import time
import logging

logger = logging.getLogger(__name__)

class RyotMetricsMiddleware(BaseHTTPMiddleware):
    """Automatic metrics collection for Ryot endpoints"""

    async def dispatch(self, request: Request, call_next):
        # Only track inference endpoints
        if not request.url.path.startswith("/v1/inference"):
            return await call_next(request)

        start_time = time.time()

        # Extract inference parameters
        model = request.query_params.get("model", "unknown")
        batch_size = int(request.query_params.get("batch_size", 1))

        try:
            response = await call_next(request)

            # Track metrics
            latency = time.time() - start_time
            status_code = response.status_code

            batch_bucket = self._categorize_batch_size(batch_size)

            ryot_inference_latency_seconds.labels(
                model=model,
                batch_size=batch_bucket,
                generation_mode="standard"
            ).observe(latency)

            status = "success" if status_code < 400 else "error"
            ryot_inference_requests_total.labels(
                model=model,
                status=status,
                endpoint="generate"
            ).inc()

            return response

        except Exception as e:
            # Record error metrics
            latency = time.time() - start_time
            ryot_inference_requests_total.labels(
                model=model,
                status="error",
                endpoint="generate"
            ).inc()
            raise

    @staticmethod
    def _categorize_batch_size(batch_size: int) -> str:
        if batch_size == 1:
            return "1"
        elif batch_size <= 4:
            return "2-4"
        elif batch_size <= 8:
            return "5-8"
        elif batch_size <= 16:
            return "9-16"
        elif batch_size <= 32:
            return "17-32"
        else:
            return "33+"
```

---

### Step 3: Streaming Response Metrics

**File**: `neurectomy/api/endpoints/inference_stream.py`

```python
"""
Streaming inference with token-level metrics
"""

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
import asyncio
from neurectomy.monitoring.ryot_metrics import (
    InferenceMetricsContext,
    GPUMetricsCollector,
)

router = APIRouter(prefix="/v1/inference", tags=["inference"])
gpu_metrics = GPUMetricsCollector()

@router.post("/stream")
async def stream_inference(
    model: str = Query(...),
    prompt: str = Query(...),
    max_tokens: int = Query(100),
    batch_size: int = Query(1),
):
    """Stream tokens with per-token metrics"""

    async def generate_tokens():
        with InferenceMetricsContext(model, batch_size, "stream") as metrics:
            # Simulate token generation
            token_count = 0

            for token_idx in range(max_tokens):
                # Record TTFT after first token
                if token_idx == 0:
                    metrics.record_ttft()

                # Generate token (simulated)
                token = f"token_{token_idx}"
                token_count += 1

                # Yield token
                yield f"data: {token}\n"

                # Add small delay between tokens
                await asyncio.sleep(0.01)

            # Record total tokens generated
            metrics.record_tokens(token_count)

            # Update GPU metrics periodically
            gpu_metrics.update_gpu_memory_metrics(model)

    return StreamingResponse(
        generate_tokens(),
        media_type="text/event-stream"
    )
```

---

### Step 4: Metrics Export Endpoint

**File**: `neurectomy/api/endpoints/metrics.py`

```python
"""
Prometheus metrics export endpoint
"""

from fastapi import APIRouter
from prometheus_client import generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST

router = APIRouter(tags=["monitoring"])

@router.get("/metrics", include_in_schema=False)
async def metrics():
    """Prometheus metrics endpoint"""
    from fastapi.responses import Response

    metrics_output = generate_latest()

    return Response(
        content=metrics_output,
        media_type=CONTENT_TYPE_LATEST
    )

@router.get("/health/metrics")
async def metrics_health():
    """Health check for metrics collection"""
    return {
        "status": "healthy",
        "metrics_enabled": True,
        "scrape_endpoint": "/metrics"
    }
```

---

### Step 5: Recording Rules

**File**: `config/prometheus_recording_rules.yml`

```yaml
groups:
  - name: ryot_inference_rules
    interval: 30s
    rules:
      # Token rate (tokens per second, 1-minute window)
      - record: ryot:tokens_per_second:rate1m
        expr: rate(ryot_tokens_generated_total[1m])
        labels:
          window: "1m"

      # Token rate (5-minute window)
      - record: ryot:tokens_per_second:rate5m
        expr: rate(ryot_tokens_generated_total[5m])
        labels:
          window: "5m"

      # Error rate (percentage)
      - record: ryot:error_rate:rate5m
        expr: |-
          (rate(ryot_inference_errors_total[5m]) / 
           rate(ryot_inference_requests_total[5m]) * 100)

      # P95 latency
      - record: ryot:inference_latency:p95:5m
        expr: histogram_quantile(0.95, rate(ryot_inference_latency_seconds_bucket[5m]))

      # P99 latency
      - record: ryot:inference_latency:p99:5m
        expr: histogram_quantile(0.99, rate(ryot_inference_latency_seconds_bucket[5m]))

      # Average batch size
      - record: ryot:batch_size:avg:5m
        expr: |-
          (rate(ryot_batch_size_distribution_sum[5m]) / 
           rate(ryot_batch_size_distribution_count[5m]))

      # GPU efficiency ratio
      - record: ryot:gpu_efficiency:ratio
        expr: |-
          (rate(ryot_tokens_generated_total[5m]) / 
           (ryot_gpu_memory_percentage / 100))
```

---

### Step 6: Alert Rules

**File**: `config/prometheus_alert_rules.yml`

```yaml
groups:
  - name: ryot_inference_alerts
    interval: 30s
    rules:
      - alert: RyotHighErrorRate
        expr: ryot:error_rate:rate5m > 5
        for: 5m
        labels:
          severity: warning
          component: ryot
        annotations:
          summary: "Ryot error rate > 5%"
          description: "Error rate is {{ $value }}% for the last 5 minutes"

      - alert: RyotHighLatency
        expr: ryot:inference_latency:p95:5m > 2.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Ryot P95 latency {{ $value }}s (target: <2.5s)"

      - alert: RyotGPUMemoryPressure
        expr: ryot_gpu_memory_percentage > 85
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "GPU memory utilization > 85%"
          description: "GPU memory is {{ $value }}% full"

      - alert: RyotLowTokenThroughput
        expr: ryot:tokens_per_second:rate1m < 50
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Token generation rate < 50 tps"
          description: "Current rate: {{ $value }} tps"

      - alert: RyotQueueBacklog
        expr: ryot_queue_depth > 100
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Request queue depth: {{ $value }}"
          description: "SLA at risk due to request backlog"
```

---

## Testing Metrics Implementation

**File**: `tests/test_ryot_metrics.py`

```python
"""
Unit tests for Ryot metrics
"""

import pytest
from neurectomy.monitoring.ryot_metrics import (
    InferenceMetricsContext,
    track_inference_metric,
    ryot_inference_requests_total,
    ryot_tokens_generated_total,
)
import time

def test_inference_metrics_context():
    """Test basic metrics tracking"""

    with InferenceMetricsContext("test-model", batch_size=4) as metrics:
        metrics.record_tokens(50)
        time.sleep(0.01)  # Simulate 10ms latency

    # Verify metrics were recorded
    assert ryot_inference_requests_total.labels(
        model="test-model",
        status="success",
        endpoint="generate"
    )._value.get() > 0


def test_error_tracking():
    """Test error metrics"""

    try:
        with InferenceMetricsContext("test-model") as metrics:
            raise RuntimeError("Simulated error")
    except RuntimeError:
        pass

    # Verify error was recorded


def test_token_metrics():
    """Test token generation metrics"""

    with InferenceMetricsContext("test-model") as metrics:
        metrics.record_tokens(100)

    # Verify tokens counted
    assert ryot_tokens_generated_total.labels(
        model="test-model",
        token_type="completion",
        generation_mode="standard"
    )._value.get() > 0


@pytest.mark.asyncio
async def test_decorator():
    """Test @track_inference_metric decorator"""

    @track_inference_metric("test-model")
    async def mock_inference():
        class Result:
            tokens = list(range(50))
        return Result()

    result = await mock_inference()
    assert len(result.tokens) == 50
```

---

## Deployment Checklist

- [ ] Deploy metrics module to staging
- [ ] Verify /metrics endpoint returns data
- [ ] Configure Prometheus scrape job
- [ ] Deploy recording rules to Prometheus
- [ ] Deploy alert rules to Alertmanager
- [ ] Create Grafana dashboards
- [ ] Test alert notifications
- [ ] Performance baseline established
- [ ] SLOs defined with stakeholders
- [ ] Production rollout with gradual increase
- [ ] Monitor metrics overhead (<0.01%)
- [ ] Set up on-call runbooks for alerts
