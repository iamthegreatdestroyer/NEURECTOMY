# PHASE 18A-2: Neurectomy API Metrics

## Target Project
**Neurectomy**

## Objective
Add Prometheus metrics to Neurectomy API for monitoring requests, latency, errors, and business metrics.

## Prerequisites
- Phase 18A-1 complete (Prometheus deployed)
- Neurectomy API running

## File to Create

### `neurectomy/monitoring/metrics.py`

```python
"""
Prometheus Metrics for Neurectomy API
"""

from prometheus_client import Counter, Histogram, Gauge, Info
from functools import wraps
import time
from typing import Callable

# Request metrics
http_requests_total = Counter(
    'neurectomy_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'neurectomy_http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint'],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0)
)

# Service integration metrics
ryot_requests_total = Counter(
    'neurectomy_ryot_requests_total',
    'Total requests to Ryot LLM',
    ['status']
)

ryot_request_duration_seconds = Histogram(
    'neurectomy_ryot_request_duration_seconds',
    'Ryot LLM request latency',
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
)

sigmalang_compression_ratio = Histogram(
    'neurectomy_sigmalang_compression_ratio',
    'ΣLANG compression ratios',
    buckets=(5, 10, 15, 20, 25, 30)
)

sigmavault_operations_total = Counter(
    'neurectomy_sigmavault_operations_total',
    'Total ΣVAULT operations',
    ['operation', 'status']
)

# Circuit breaker metrics
circuit_breaker_state = Gauge(
    'neurectomy_circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=open, 2=half-open)',
    ['service']
)

circuit_breaker_failures = Counter(
    'neurectomy_circuit_breaker_failures_total',
    'Total circuit breaker failures',
    ['service']
)

# Business metrics
active_users = Gauge(
    'neurectomy_active_users',
    'Current active users'
)

tokens_generated_total = Counter(
    'neurectomy_tokens_generated_total',
    'Total tokens generated',
    ['model']
)

api_keys_active = Gauge(
    'neurectomy_api_keys_active',
    'Number of active API keys'
)

# System metrics
system_info = Info(
    'neurectomy_system',
    'System information'
)


class MetricsMiddleware:
    """FastAPI middleware for automatic metrics collection"""
    
    def __init__(self, app):
        self.app = app
        
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
            
        method = scope["method"]
        path = scope["path"]
        
        start_time = time.time()
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_code = message["status"]
                duration = time.time() - start_time
                
                # Record metrics
                http_requests_total.labels(
                    method=method,
                    endpoint=path,
                    status=status_code
                ).inc()
                
                http_request_duration_seconds.labels(
                    method=method,
                    endpoint=path
                ).observe(duration)
                
            await send(message)
            
        await self.app(scope, receive, send_wrapper)


def track_ryot_request(func: Callable):
    """Decorator to track Ryot LLM requests"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        status = "success"
        
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time
            ryot_requests_total.labels(status=status).inc()
            ryot_request_duration_seconds.observe(duration)
            
    return wrapper


def track_compression(func: Callable):
    """Decorator to track compression operations"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        result = await func(*args, **kwargs)
        
        if hasattr(result, 'compression_ratio'):
            sigmalang_compression_ratio.observe(result.compression_ratio)
            
        return result
        
    return wrapper


def track_storage_operation(operation: str):
    """Decorator to track ΣVAULT operations"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            status = "success"
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                sigmavault_operations_total.labels(
                    operation=operation,
                    status=status
                ).inc()
                
        return wrapper
    return decorator
```

### Add to `neurectomy/main.py`

```python
from prometheus_client import make_asgi_app
from neurectomy.monitoring.metrics import MetricsMiddleware, system_info

# Initialize system info
system_info.info({
    'version': '1.0.0',
    'environment': 'production',
    'cluster': 'neurectomy-prod'
})

# Add metrics middleware
app.add_middleware(MetricsMiddleware)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

## Usage Examples

```python
from neurectomy.monitoring.metrics import (
    track_ryot_request,
    track_compression,
    track_storage_operation,
    active_users,
    tokens_generated_total
)

# Track Ryot requests
@track_ryot_request
async def call_ryot_inference(prompt: str):
    result = await ryot_client.generate(prompt)
    tokens_generated_total.labels(model="ryot-bitnet-7b").inc(result.tokens)
    return result

# Track compression
@track_compression
async def compress_text(text: str):
    return await sigmalang_client.compress(text)

# Track storage
@track_storage_operation("store")
async def store_file(path: str, data: bytes):
    return await sigmavault_client.store(path, data)

# Update active users
active_users.set(len(get_active_sessions()))
```

## Success Criteria

- [ ] Metrics endpoint accessible at /metrics
- [ ] HTTP request metrics captured
- [ ] Service integration metrics working
- [ ] Circuit breaker metrics tracked
- [ ] Business metrics updated
- [ ] Prometheus scraping metrics

## Next Steps

Proceed to PHASE-18A-3-RYOT-METRICS.md
