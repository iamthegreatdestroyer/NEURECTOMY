# ============================================================================
# NEURECTOMY ML Service - Prometheus Metrics
# Custom metrics for monitoring and observability
# ============================================================================

"""
Prometheus metrics configuration for ML Service.

This module provides:
- Custom metrics for training jobs
- API latency histograms
- Business metrics (predictions, active models)
- Resource utilization metrics
"""

from typing import Optional, Callable
from functools import wraps
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# Metric Labels
# ============================================================================

class MetricLabel(str, Enum):
    """Standard metric labels."""
    ENVIRONMENT = "environment"
    SERVICE = "service"
    VERSION = "version"
    ENDPOINT = "endpoint"
    METHOD = "method"
    STATUS = "status"
    MODEL_TYPE = "model_type"
    AGENT_ID = "agent_id"
    JOB_TYPE = "job_type"


# ============================================================================
# Mock Prometheus Metrics (for when prometheus_client is not installed)
# ============================================================================

@dataclass
class MockMetric:
    """Mock metric for when prometheus_client is not available."""
    name: str
    documentation: str
    labelnames: tuple = ()
    
    def labels(self, *args, **kwargs) -> "MockMetric":
        return self
    
    def inc(self, amount: float = 1) -> None:
        pass
    
    def dec(self, amount: float = 1) -> None:
        pass
    
    def set(self, value: float) -> None:
        pass
    
    def observe(self, value: float) -> None:
        pass
    
    def time(self) -> "MockMetric":
        return self
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary, Info,
        generate_latest, CONTENT_TYPE_LATEST,
        CollectorRegistry, REGISTRY, multiprocess
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Gauge = Histogram = Summary = Info = MockMetric
    REGISTRY = None
    CONTENT_TYPE_LATEST = "text/plain"
    
    def generate_latest(registry=None):
        return b"# Prometheus client not installed"


# ============================================================================
# Service Info
# ============================================================================

if PROMETHEUS_AVAILABLE:
    SERVICE_INFO = Info(
        "neurectomy_ml_service",
        "ML Service information"
    )
    SERVICE_INFO.info({
        "version": "1.0.0",
        "service": "ml-service",
        "framework": "fastapi"
    })
else:
    SERVICE_INFO = MockMetric("neurectomy_ml_service", "ML Service information")


# ============================================================================
# Request Metrics
# ============================================================================

# HTTP request counter
REQUEST_COUNT = Counter(
    "neurectomy_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
) if PROMETHEUS_AVAILABLE else MockMetric(
    "neurectomy_http_requests_total",
    "Total HTTP requests",
    labelnames=("method", "endpoint", "status")
)

# HTTP request latency
REQUEST_LATENCY = Histogram(
    "neurectomy_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
) if PROMETHEUS_AVAILABLE else MockMetric(
    "neurectomy_http_request_duration_seconds",
    "HTTP request duration in seconds",
    labelnames=("method", "endpoint")
)

# Request size
REQUEST_SIZE = Histogram(
    "neurectomy_http_request_size_bytes",
    "HTTP request size in bytes",
    ["method", "endpoint"],
    buckets=(100, 1000, 10000, 100000, 1000000)
) if PROMETHEUS_AVAILABLE else MockMetric(
    "neurectomy_http_request_size_bytes",
    "HTTP request size in bytes",
    labelnames=("method", "endpoint")
)

# Response size
RESPONSE_SIZE = Histogram(
    "neurectomy_http_response_size_bytes",
    "HTTP response size in bytes",
    ["method", "endpoint"],
    buckets=(100, 1000, 10000, 100000, 1000000)
) if PROMETHEUS_AVAILABLE else MockMetric(
    "neurectomy_http_response_size_bytes",
    "HTTP response size in bytes",
    labelnames=("method", "endpoint")
)

# Active requests gauge
ACTIVE_REQUESTS = Gauge(
    "neurectomy_http_requests_active",
    "Number of active HTTP requests",
    ["method", "endpoint"]
) if PROMETHEUS_AVAILABLE else MockMetric(
    "neurectomy_http_requests_active",
    "Number of active HTTP requests",
    labelnames=("method", "endpoint")
)


# ============================================================================
# Training Metrics
# ============================================================================

# Training jobs counter
TRAINING_JOBS_TOTAL = Counter(
    "neurectomy_training_jobs_total",
    "Total training jobs",
    ["model_type", "status"]
) if PROMETHEUS_AVAILABLE else MockMetric(
    "neurectomy_training_jobs_total",
    "Total training jobs",
    labelnames=("model_type", "status")
)

# Active training jobs
ACTIVE_TRAINING_JOBS = Gauge(
    "neurectomy_training_jobs_active",
    "Number of active training jobs",
    ["model_type"]
) if PROMETHEUS_AVAILABLE else MockMetric(
    "neurectomy_training_jobs_active",
    "Number of active training jobs",
    labelnames=("model_type",)
)

# Training duration
TRAINING_DURATION = Histogram(
    "neurectomy_training_duration_seconds",
    "Training job duration in seconds",
    ["model_type"],
    buckets=(60, 300, 600, 1800, 3600, 7200, 14400, 28800, 86400)
) if PROMETHEUS_AVAILABLE else MockMetric(
    "neurectomy_training_duration_seconds",
    "Training job duration in seconds",
    labelnames=("model_type",)
)

# Training loss
TRAINING_LOSS = Gauge(
    "neurectomy_training_loss",
    "Current training loss",
    ["job_id", "model_type"]
) if PROMETHEUS_AVAILABLE else MockMetric(
    "neurectomy_training_loss",
    "Current training loss",
    labelnames=("job_id", "model_type")
)

# Training accuracy
TRAINING_ACCURACY = Gauge(
    "neurectomy_training_accuracy",
    "Current training accuracy",
    ["job_id", "model_type"]
) if PROMETHEUS_AVAILABLE else MockMetric(
    "neurectomy_training_accuracy",
    "Current training accuracy",
    labelnames=("job_id", "model_type")
)

# Epoch progress
TRAINING_EPOCH = Gauge(
    "neurectomy_training_epoch_current",
    "Current training epoch",
    ["job_id"]
) if PROMETHEUS_AVAILABLE else MockMetric(
    "neurectomy_training_epoch_current",
    "Current training epoch",
    labelnames=("job_id",)
)


# ============================================================================
# Inference Metrics
# ============================================================================

# Inference counter
INFERENCE_COUNT = Counter(
    "neurectomy_inference_total",
    "Total inference requests",
    ["model_type", "status"]
) if PROMETHEUS_AVAILABLE else MockMetric(
    "neurectomy_inference_total",
    "Total inference requests",
    labelnames=("model_type", "status")
)

# Inference latency
INFERENCE_LATENCY = Histogram(
    "neurectomy_inference_duration_seconds",
    "Inference duration in seconds",
    ["model_type"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
) if PROMETHEUS_AVAILABLE else MockMetric(
    "neurectomy_inference_duration_seconds",
    "Inference duration in seconds",
    labelnames=("model_type",)
)

# Model load time
MODEL_LOAD_TIME = Histogram(
    "neurectomy_model_load_duration_seconds",
    "Model load duration in seconds",
    ["model_type"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
) if PROMETHEUS_AVAILABLE else MockMetric(
    "neurectomy_model_load_duration_seconds",
    "Model load duration in seconds",
    labelnames=("model_type",)
)


# ============================================================================
# Resource Metrics
# ============================================================================

# GPU utilization
GPU_UTILIZATION = Gauge(
    "neurectomy_gpu_utilization_percent",
    "GPU utilization percentage",
    ["gpu_id"]
) if PROMETHEUS_AVAILABLE else MockMetric(
    "neurectomy_gpu_utilization_percent",
    "GPU utilization percentage",
    labelnames=("gpu_id",)
)

# GPU memory
GPU_MEMORY_USED = Gauge(
    "neurectomy_gpu_memory_used_bytes",
    "GPU memory used in bytes",
    ["gpu_id"]
) if PROMETHEUS_AVAILABLE else MockMetric(
    "neurectomy_gpu_memory_used_bytes",
    "GPU memory used in bytes",
    labelnames=("gpu_id",)
)

# Model cache size
MODEL_CACHE_SIZE = Gauge(
    "neurectomy_model_cache_size_bytes",
    "Model cache size in bytes"
) if PROMETHEUS_AVAILABLE else MockMetric(
    "neurectomy_model_cache_size_bytes",
    "Model cache size in bytes"
)

# Loaded models count
LOADED_MODELS_COUNT = Gauge(
    "neurectomy_loaded_models_total",
    "Number of loaded models",
    ["model_type"]
) if PROMETHEUS_AVAILABLE else MockMetric(
    "neurectomy_loaded_models_total",
    "Number of loaded models",
    labelnames=("model_type",)
)


# ============================================================================
# Security Metrics
# ============================================================================

# Authentication attempts
AUTH_ATTEMPTS = Counter(
    "neurectomy_auth_attempts_total",
    "Total authentication attempts",
    ["status", "method"]
) if PROMETHEUS_AVAILABLE else MockMetric(
    "neurectomy_auth_attempts_total",
    "Total authentication attempts",
    labelnames=("status", "method")
)

# Rate limit hits
RATE_LIMIT_HITS = Counter(
    "neurectomy_rate_limit_hits_total",
    "Total rate limit hits",
    ["endpoint"]
) if PROMETHEUS_AVAILABLE else MockMetric(
    "neurectomy_rate_limit_hits_total",
    "Total rate limit hits",
    labelnames=("endpoint",)
)

# Active sessions
ACTIVE_SESSIONS = Gauge(
    "neurectomy_active_sessions",
    "Number of active user sessions"
) if PROMETHEUS_AVAILABLE else MockMetric(
    "neurectomy_active_sessions",
    "Number of active user sessions"
)


# ============================================================================
# Analytics Metrics
# ============================================================================

# Anomalies detected
ANOMALIES_DETECTED = Counter(
    "neurectomy_anomalies_detected_total",
    "Total anomalies detected",
    ["metric_type", "severity"]
) if PROMETHEUS_AVAILABLE else MockMetric(
    "neurectomy_anomalies_detected_total",
    "Total anomalies detected",
    labelnames=("metric_type", "severity")
)

# Forecasts generated
FORECASTS_GENERATED = Counter(
    "neurectomy_forecasts_generated_total",
    "Total forecasts generated",
    ["method"]
) if PROMETHEUS_AVAILABLE else MockMetric(
    "neurectomy_forecasts_generated_total",
    "Total forecasts generated",
    labelnames=("method",)
)


# ============================================================================
# Utility Functions
# ============================================================================

@contextmanager
def track_request_metrics(method: str, endpoint: str):
    """Context manager to track request metrics."""
    ACTIVE_REQUESTS.labels(method=method, endpoint=endpoint).inc()
    start_time = time.perf_counter()
    status = "success"
    
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        duration = time.perf_counter() - start_time
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
        ACTIVE_REQUESTS.labels(method=method, endpoint=endpoint).dec()


def track_training_job(model_type: str):
    """Decorator to track training job metrics."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            ACTIVE_TRAINING_JOBS.labels(model_type=model_type).inc()
            start_time = time.perf_counter()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception:
                status = "failed"
                raise
            finally:
                duration = time.perf_counter() - start_time
                TRAINING_DURATION.labels(model_type=model_type).observe(duration)
                TRAINING_JOBS_TOTAL.labels(model_type=model_type, status=status).inc()
                ACTIVE_TRAINING_JOBS.labels(model_type=model_type).dec()
        
        return wrapper
    return decorator


def track_inference(model_type: str):
    """Decorator to track inference metrics."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception:
                status = "error"
                raise
            finally:
                duration = time.perf_counter() - start_time
                INFERENCE_LATENCY.labels(model_type=model_type).observe(duration)
                INFERENCE_COUNT.labels(model_type=model_type, status=status).inc()
        
        return wrapper
    return decorator


def get_metrics() -> bytes:
    """Get all metrics in Prometheus format."""
    return generate_latest(REGISTRY) if PROMETHEUS_AVAILABLE else b"# Prometheus client not installed"


def get_content_type() -> str:
    """Get the content type for metrics endpoint."""
    return CONTENT_TYPE_LATEST
