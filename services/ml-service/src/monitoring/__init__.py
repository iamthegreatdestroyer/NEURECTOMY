# ============================================================================
# NEURECTOMY ML Service - Monitoring Package
# ============================================================================

"""
Monitoring package for ML Service.

Provides:
- Prometheus metrics
- Health checks
- Tracing
- Logging configuration
"""

from .metrics import (
    # Request metrics
    REQUEST_COUNT,
    REQUEST_LATENCY,
    REQUEST_SIZE,
    RESPONSE_SIZE,
    ACTIVE_REQUESTS,
    
    # Training metrics
    TRAINING_JOBS_TOTAL,
    ACTIVE_TRAINING_JOBS,
    TRAINING_DURATION,
    TRAINING_LOSS,
    TRAINING_ACCURACY,
    TRAINING_EPOCH,
    
    # Inference metrics
    INFERENCE_COUNT,
    INFERENCE_LATENCY,
    MODEL_LOAD_TIME,
    
    # Resource metrics
    GPU_UTILIZATION,
    GPU_MEMORY_USED,
    MODEL_CACHE_SIZE,
    LOADED_MODELS_COUNT,
    
    # Security metrics
    AUTH_ATTEMPTS,
    RATE_LIMIT_HITS,
    ACTIVE_SESSIONS,
    
    # Analytics metrics
    ANOMALIES_DETECTED,
    FORECASTS_GENERATED,
    
    # Utility functions
    track_request_metrics,
    track_training_job,
    track_inference,
    get_metrics,
    get_content_type,
    
    # Constants
    PROMETHEUS_AVAILABLE,
)

__all__ = [
    # Request metrics
    "REQUEST_COUNT",
    "REQUEST_LATENCY",
    "REQUEST_SIZE",
    "RESPONSE_SIZE",
    "ACTIVE_REQUESTS",
    
    # Training metrics
    "TRAINING_JOBS_TOTAL",
    "ACTIVE_TRAINING_JOBS",
    "TRAINING_DURATION",
    "TRAINING_LOSS",
    "TRAINING_ACCURACY",
    "TRAINING_EPOCH",
    
    # Inference metrics
    "INFERENCE_COUNT",
    "INFERENCE_LATENCY",
    "MODEL_LOAD_TIME",
    
    # Resource metrics
    "GPU_UTILIZATION",
    "GPU_MEMORY_USED",
    "MODEL_CACHE_SIZE",
    "LOADED_MODELS_COUNT",
    
    # Security metrics
    "AUTH_ATTEMPTS",
    "RATE_LIMIT_HITS",
    "ACTIVE_SESSIONS",
    
    # Analytics metrics
    "ANOMALIES_DETECTED",
    "FORECASTS_GENERATED",
    
    # Utility functions
    "track_request_metrics",
    "track_training_job",
    "track_inference",
    "get_metrics",
    "get_content_type",
    
    # Constants
    "PROMETHEUS_AVAILABLE",
]
