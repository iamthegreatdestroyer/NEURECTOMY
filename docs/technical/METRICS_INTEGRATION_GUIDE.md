# Metrics Integration Guide

**Location:** `docs/technical/METRICS_INTEGRATION_GUIDE.md`

## Overview

This guide provides step-by-step instructions for integrating Prometheus metrics into new services within the NEURECTOMY ecosystem. It covers the complete lifecycle: setup, instrumentation, deployment, and validation.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Metric Types and Usage](#metric-types-and-usage)
3. [Integration Patterns](#integration-patterns)
4. [Common Implementation Patterns](#common-implementation-patterns)
5. [Performance Considerations](#performance-considerations)
6. [Testing Metrics](#testing-metrics)
7. [Troubleshooting](#troubleshooting)
8. [Checklist](#integration-checklist)

---

## Quick Start

### 1. Install Prometheus Client

```bash
# Python
pip install prometheus-client

# Node.js
npm install prom-client

# Go
go get github.com/prometheus/client_golang/prometheus
```

### 2. Initialize Metrics Module

```python
# File: yourservice/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry

# Create registry (optional, auto-registers if omitted)
registry = CollectorRegistry()

# Define metrics
operations_total = Counter(
    'operations_total',
    'Total operations',
    ['operation_type', 'status'],
    registry=registry
)

operation_duration_seconds = Histogram(
    'operation_duration_seconds',
    'Operation duration',
    ['operation_type'],
    buckets=(0.1, 0.5, 1.0, 5.0, 10.0),
    registry=registry
)

active_connections = Gauge(
    'active_connections',
    'Active connections',
    ['service'],
    registry=registry
)

system_info = Info(
    'system',
    'System information',
    registry=registry
)
```

### 3. Expose Metrics Endpoint

```python
# File: yourservice/server.py
from prometheus_client import start_http_server, generate_latest, REGISTRY
import time
from flask import Flask, Response

app = Flask(__name__)

@app.route('/metrics')
def metrics():
    return Response(generate_latest(REGISTRY), mimetype='text/plain')

if __name__ == '__main__':
    # Start HTTP server on port 9090
    start_http_server(8000)
    app.run(port=5000)
```

### 4. Instrument Code

```python
# File: yourservice/service.py
from monitoring.metrics import operations_total, operation_duration_seconds
import time

def process_request(request_type):
    start_time = time.time()

    try:
        # Perform operation
        result = handle_request(request_type)

        # Track success
        operations_total.labels(operation_type=request_type, status='success').inc()

        return result
    except Exception as e:
        # Track failure
        operations_total.labels(operation_type=request_type, status='error').inc()
        raise
    finally:
        # Record duration
        duration = time.time() - start_time
        operation_duration_seconds.labels(operation_type=request_type).observe(duration)
```

### 5. Configure Prometheus

```yaml
# File: prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: "yourservice"
    static_configs:
      - targets: ["localhost:8000"]
    metrics_path: "/metrics"
```

### 6. Verify Metrics

```bash
# Check metrics endpoint
curl http://localhost:8000/metrics

# Expected output:
# # HELP operations_total Total operations
# # TYPE operations_total counter
# operations_total{operation_type="request",status="success"} 42
```

---

## Metric Types and Usage

### Counter

**Use Case:** Monotonically increasing values (never decrease)

**Examples:**

- Total requests
- Total errors
- Total bytes transferred
- Total operations completed

**Example:**

```python
requests_total = Counter(
    'requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

# Usage
requests_total.labels(method='GET', endpoint='/api/data', status='200').inc()

# Multiple increments
requests_total.labels(method='POST', endpoint='/api/data', status='201').inc(5)
```

### Gauge

**Use Case:** Point-in-time values (can increase or decrease)

**Examples:**

- Current memory usage
- Active connections
- Queue length
- Temperature

**Example:**

```python
active_connections = Gauge(
    'active_connections',
    'Number of active connections',
    ['service', 'type']
)

# Usage
active_connections.labels(service='api', type='websocket').set(42)
active_connections.labels(service='api', type='http').inc()
active_connections.labels(service='api', type='http').dec()
```

### Histogram

**Use Case:** Distribution of values within buckets

**Examples:**

- Request latency
- Response size
- Processing duration
- Memory allocation size

**Example:**

```python
request_duration_seconds = Histogram(
    'request_duration_seconds',
    'Request processing time',
    ['endpoint'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0)
)

# Usage
request_duration_seconds.labels(endpoint='/api/users').observe(0.234)

# Provides:
# - request_duration_seconds_bucket (count of observations <= bucket value)
# - request_duration_seconds_sum (total of all observations)
# - request_duration_seconds_count (total number of observations)
```

### Summary

**Use Case:** Similar to histogram but with pre-computed quantiles

**Difference:** Histogram computes quantiles server-side (Prometheus), Summary computes client-side

**Example:**

```python
from prometheus_client import Summary

response_size_bytes = Summary(
    'response_size_bytes',
    'Response size',
    ['endpoint']
)

# Usage
response_size_bytes.labels(endpoint='/api/data').observe(1024)

# Provides quantiles: 0.5 (median), 0.9 (P90), 0.99 (P99)
```

### Info

**Use Case:** Metadata about the system (static labels)

**Example:**

```python
system_info = Info(
    'system',
    'System information'
)

system_info.info({
    'version': '1.2.3',
    'environment': 'production',
    'region': 'us-west-2'
})
```

---

## Integration Patterns

### Pattern 1: Decorator-Based Instrumentation

**Best for:** Function/method-level tracking

```python
from functools import wraps
import time
from monitoring.metrics import function_duration_seconds, function_calls_total

def track_function(func_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                function_calls_total.labels(
                    function=func_name,
                    status='success'
                ).inc()
                return result
            except Exception as e:
                function_calls_total.labels(
                    function=func_name,
                    status='error'
                ).inc()
                raise
            finally:
                duration = time.time() - start
                function_duration_seconds.labels(
                    function=func_name
                ).observe(duration)
        return wrapper
    return decorator

# Usage
@track_function('process_data')
def process_data(data):
    # Your code here
    pass
```

### Pattern 2: Context Manager Instrumentation

**Best for:** Block-level tracking

```python
from contextlib import contextmanager
import time
from monitoring.metrics import block_duration_seconds, block_errors_total

@contextmanager
def track_block(block_name):
    start = time.time()
    try:
        yield
    except Exception as e:
        block_errors_total.labels(block=block_name).inc()
        raise
    finally:
        duration = time.time() - start
        block_duration_seconds.labels(block=block_name).observe(duration)

# Usage
with track_block('data_processing'):
    result = process_data()
    analyze_results(result)
```

### Pattern 3: Middleware Instrumentation

**Best for:** Request/response tracking

```python
from flask import Flask, request
import time

app = Flask(__name__)

@app.before_request
def before_request():
    request.start_time = time.time()

@app.after_request
def after_request(response):
    duration = time.time() - request.start_time

    http_request_duration_seconds.labels(
        method=request.method,
        endpoint=request.path,
        status=response.status_code
    ).observe(duration)

    http_requests_total.labels(
        method=request.method,
        endpoint=request.path,
        status=response.status_code
    ).inc()

    return response
```

### Pattern 4: Batch Operation Tracking

**Best for:** Operations on multiple items

```python
from monitoring.metrics import batch_operations_total, batch_items_processed

class BatchProcessor:
    def __init__(self, operation_name):
        self.operation_name = operation_name
        self.items_processed = 0
        self.errors = 0

    def process_batch(self, items):
        for item in items:
            try:
                self.process_item(item)
                self.items_processed += 1
            except Exception:
                self.errors += 1

        batch_operations_total.labels(
            operation=self.operation_name,
            status='success' if self.errors == 0 else 'partial'
        ).inc()

        batch_items_processed.labels(
            operation=self.operation_name
        ).inc(self.items_processed)

# Usage
processor = BatchProcessor('import_users')
processor.process_batch(user_list)
```

---

## Common Implementation Patterns

### Pattern: Service Health Check

```python
from prometheus_client import Gauge
import asyncio

service_health = Gauge(
    'service_health_status',
    'Service health (0=healthy, 1=degraded, 2=down)',
    ['service_name']
)

async def health_check_loop(service_name, interval=10):
    while True:
        try:
            # Check service health
            is_healthy = await verify_service_health()
            status = 0 if is_healthy else 1
            service_health.labels(service_name=service_name).set(status)
        except Exception:
            service_health.labels(service_name=service_name).set(2)

        await asyncio.sleep(interval)
```

### Pattern: Resource Monitoring

```python
from prometheus_client import Gauge
import psutil

memory_usage_bytes = Gauge(
    'memory_usage_bytes',
    'Memory usage',
    ['type']
)

cpu_usage_percent = Gauge(
    'cpu_usage_percent',
    'CPU usage',
    ['cpu_id']
)

def update_resource_metrics():
    # Memory
    mem = psutil.virtual_memory()
    memory_usage_bytes.labels(type='used').set(mem.used)
    memory_usage_bytes.labels(type='total').set(mem.total)

    # CPU
    cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
    for i, percent in enumerate(cpu_percent):
        cpu_usage_percent.labels(cpu_id=str(i)).set(percent)
```

### Pattern: Queue Monitoring

```python
from prometheus_client import Gauge, Counter

queue_depth = Gauge(
    'queue_depth',
    'Items in queue',
    ['queue_name']
)

queue_items_processed = Counter(
    'queue_items_processed_total',
    'Total items processed from queue',
    ['queue_name']
)

class MetricsAwareQueue:
    def __init__(self, name):
        self.name = name
        self.queue = []

    def put(self, item):
        self.queue.append(item)
        queue_depth.labels(queue_name=self.name).set(len(self.queue))

    def get(self):
        item = self.queue.pop(0)
        queue_depth.labels(queue_name=self.name).set(len(self.queue))
        queue_items_processed.labels(queue_name=self.name).inc()
        return item
```

### Pattern: Database Query Tracking

```python
from prometheus_client import Histogram, Counter

query_duration_seconds = Histogram(
    'db_query_duration_seconds',
    'Database query duration',
    ['query_type']
)

query_errors_total = Counter(
    'db_query_errors_total',
    'Database query errors',
    ['query_type', 'error_type']
)

class MetricsAwareDB:
    def execute_query(self, query_type, query_string):
        import time
        start = time.time()

        try:
            result = self._execute(query_string)
            return result
        except Exception as e:
            query_errors_total.labels(
                query_type=query_type,
                error_type=type(e).__name__
            ).inc()
            raise
        finally:
            duration = time.time() - start
            query_duration_seconds.labels(query_type=query_type).observe(duration)
```

---

## Performance Considerations

### Label Strategy

**Best Practice:** Use static, low-cardinality labels

```python
# GOOD: Low cardinality
http_requests_total = Counter(
    'http_requests_total',
    'HTTP requests',
    ['method', 'status']  # Limited values
)

# BAD: High cardinality (one label per user)
requests_by_user = Counter(
    'requests_by_user',
    'Requests by user',
    ['user_id']  # Millions of unique values!
)
```

### Metric Naming

**Best Practice:** Use descriptive names with units

```python
# GOOD
request_duration_seconds = Histogram('request_duration_seconds', ...)
memory_usage_bytes = Gauge('memory_usage_bytes', ...)
cache_hit_ratio = Gauge('cache_hit_ratio', ...)

# AVOID
latency = Histogram('latency', ...)  # Unit unclear
mem = Gauge('mem', ...)  # Unit/scope unclear
hits = Gauge('hits', ...)  # Ambiguous
```

### Histogram Buckets

**Best Practice:** Align buckets with SLA targets

```python
# For API with 100ms SLA
request_duration_seconds = Histogram(
    'request_duration_seconds',
    'Request duration',
    buckets=[
        0.01,    # 10ms
        0.025,   # 25ms
        0.05,    # 50ms
        0.075,   # 75ms (below SLA)
        0.1,     # 100ms (SLA target)
        0.25,    # 250ms
        0.5,     # 500ms
        1.0,     # 1 second
        2.5,     # 2.5 seconds
        5.0      # 5 seconds
    ]
)
```

### Memory Impact

**Guideline:** Estimate memory per metric series

```
Memory per series ≈ 1-2 KB

Example:
- 100 metrics × 5 label combinations = 500 series
- 500 series × 1.5 KB = ~750 KB per service

Total for 10 services: ~7.5 MB (acceptable)
```

### Cardinality Limits

**Recommendation:** Keep total cardinality < 10,000 series per service

```python
# Calculate cardinality
from itertools import product

labels = {
    'method': ['GET', 'POST', 'PUT', 'DELETE'],          # 4
    'endpoint': ['/api/users', '/api/data', '/api/health'],  # 3
    'status': ['200', '400', '500']                        # 3
}

cardinality = 4 * 3 * 3  # = 36 series
```

---

## Testing Metrics

### Unit Tests

```python
import pytest
from prometheus_client import CollectorRegistry, Counter

def test_counter_increment():
    registry = CollectorRegistry()
    counter = Counter('test_counter', 'Test', registry=registry)

    counter.inc()
    assert counter._value.get() == 1.0

    counter.inc(5)
    assert counter._value.get() == 6.0

def test_labels():
    registry = CollectorRegistry()
    counter = Counter('test_counter', 'Test', ['method', 'status'], registry=registry)

    counter.labels(method='GET', status='200').inc()
    counter.labels(method='POST', status='201').inc(2)

    # Verify values
    assert counter.labels(method='GET', status='200')._value.get() == 1.0
    assert counter.labels(method='POST', status='201')._value.get() == 2.0
```

### Integration Tests

```python
import requests

def test_metrics_endpoint():
    # Start service
    service = start_test_service()

    # Generate some activity
    requests.get('http://localhost:5000/api/data')
    requests.post('http://localhost:5000/api/data', json={'data': 'test'})

    # Check metrics
    response = requests.get('http://localhost:8000/metrics')
    assert response.status_code == 200

    metrics_text = response.text
    assert 'http_requests_total' in metrics_text
    assert 'method="GET"' in metrics_text
    assert 'method="POST"' in metrics_text
```

### Metric Value Assertions

```python
def test_operation_tracking():
    from monitoring.metrics import operations_total, operation_duration_seconds

    # Execute operation
    result = perform_operation('test_op')

    # Assert metrics
    assert operations_total.labels(
        operation_type='test_op',
        status='success'
    )._value.get() >= 1

    # Check histogram
    metrics = operation_duration_seconds.collect()
    for family in metrics:
        for sample in family.samples:
            if 'test_op' in str(sample):
                assert sample.value >= 0  # Duration recorded
```

---

## Troubleshooting

### Issue: Metrics Endpoint Not Accessible

**Symptoms:** `curl http://localhost:8000/metrics` returns connection error

**Solutions:**

1. Verify service is running
2. Check if port 8000 is bound: `netstat -tuln | grep 8000`
3. Check firewall rules
4. Verify endpoint routing configuration

### Issue: Metrics Not Recording

**Symptoms:** Prometheus scrape succeeds but no metrics appear

**Solutions:**

1. Verify code is executing (add logging)
2. Check metric registration: `prometheus_client.REGISTRY.collect()`
3. Ensure labels match expected values
4. Verify histogram is being called with `.observe()`

### Issue: High Memory Usage

**Symptoms:** Service memory grows over time

**Potential Causes:**

1. Unbounded label cardinality
2. Memory leak in custom instrumentation
3. Too many histogram buckets

**Investigation:**

```python
from prometheus_client import REGISTRY

def get_metric_stats():
    total_series = 0
    for collector in REGISTRY._collector_to_names:
        for metric in collector.collect():
            for sample in metric.samples:
                total_series += 1
    print(f"Total metric series: {total_series}")
```

### Issue: High CPU Usage

**Symptoms:** Metric collection causing CPU spikes

**Solutions:**

1. Reduce scrape frequency in prometheus.yml
2. Optimize histogram bucket computation
3. Use Summary instead of Histogram for expensive percentile calculations
4. Batch metric updates

---

## Integration Checklist

### Planning Phase

- [ ] Identify service name and type
- [ ] List key business metrics to track
- [ ] List key operational metrics (latency, errors, resources)
- [ ] Estimate cardinality (label combinations)
- [ ] Review SLA targets for latency/availability
- [ ] Identify alert thresholds

### Implementation Phase

- [ ] Install prometheus-client library
- [ ] Create monitoring/metrics.py module
- [ ] Define metric types (Counter, Gauge, Histogram)
- [ ] Choose appropriate labels (low cardinality)
- [ ] Select histogram buckets aligned with SLA
- [ ] Implement instrumentation patterns
- [ ] Add metrics endpoint to service
- [ ] Test metrics collection locally

### Deployment Phase

- [ ] Add Prometheus scrape configuration
- [ ] Configure metrics retention policy (30+ days)
- [ ] Test metrics collection in staging
- [ ] Verify cardinality is acceptable (< 10k series)
- [ ] Create Grafana dashboards
- [ ] Set up alert rules
- [ ] Configure notification channels
- [ ] Document metrics in runbook

### Validation Phase

- [ ] Verify metrics appear in Prometheus
- [ ] Verify dashboard displays correct values
- [ ] Test alert firing (simulate conditions)
- [ ] Verify no cardinality explosion
- [ ] Monitor memory/CPU impact
- [ ] Review metrics with team
- [ ] Update documentation

### Ongoing Operations

- [ ] Monitor for cardinality growth
- [ ] Audit unused metrics
- [ ] Review and update SLA thresholds
- [ ] Refine alert rules based on false positives
- [ ] Collect team feedback on dashboards
- [ ] Update runbooks with new metrics

---

## Best Practices Summary

1. **Label Design:** Use low-cardinality, static labels
2. **Naming:** Include unit in metric name (e.g., `_seconds`, `_bytes`, `_ratio`)
3. **Histogram Buckets:** Align with SLA targets
4. **Instrumentation:** Use decorators/context managers for consistency
5. **Testing:** Include metric assertions in tests
6. **Documentation:** Document all custom metrics
7. **Monitoring:** Alert on cardinality growth
8. **Performance:** Measure memory/CPU impact
9. **Granularity:** Balance detail with cardinality
10. **Consistency:** Follow established patterns across services

---

## References

- **Prometheus Client Library:** https://github.com/prometheus/client_python
- **Metric Naming:** https://prometheus.io/docs/practices/naming/
- **Instrumentation:** https://prometheus.io/docs/practices/instrumentation/
- **Query Language:** https://prometheus.io/docs/prometheus/latest/querying/
- **Grafana:** https://grafana.com/docs/grafana/latest/

---

**Version:** 1.0  
**Last Updated:** December 16, 2025  
**Maintained By:** @SCRIBE
