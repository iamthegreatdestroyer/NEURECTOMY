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

# Import tracing module
from .tracing import (
    # Initialization
    init_tracer,
    shutdown_tracer,
    get_service_tracer,
    instrument_fastapi,
    uninstrument_fastapi,
    
    # Span creation
    create_span,
    traced,
    
    # Span utilities
    add_span_attributes,
    add_span_event,
    set_span_error,
    get_trace_context,
    extract_trace_context,
    
    # ML-specific tracing
    trace_inference,
    trace_training,
    trace_embedding,
    trace_llm_call,
    
    # Database tracing
    trace_db_operation,
    trace_cache_operation,
    
    # Context propagation
    inject_context_to_headers,
    create_context_from_headers,
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
    
    # Tracing - Initialization
    "init_tracer",
    "shutdown_tracer",
    "get_service_tracer",
    "instrument_fastapi",
    "uninstrument_fastapi",
    
    # Tracing - Span creation
    "create_span",
    "traced",
    
    # Tracing - Span utilities
    "add_span_attributes",
    "add_span_event",
    "set_span_error",
    "get_trace_context",
    "extract_trace_context",
    
    # Tracing - ML-specific
    "trace_inference",
    "trace_training",
    "trace_embedding",
    "trace_llm_call",
    
    # Tracing - Database
    "trace_db_operation",
    "trace_cache_operation",
    
    # Tracing - Context propagation
    "inject_context_to_headers",
    "create_context_from_headers",
]
