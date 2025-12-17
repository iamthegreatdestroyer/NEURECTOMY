"""
Ryot LLM Monitoring Module
Prometheus metrics for LLM inference and token generation
"""

from .metrics import (
    # Inference Request Metrics
    inference_requests_total,
    inference_request_duration_seconds,
    inference_ttft_seconds,
    inference_inter_token_latency_seconds,
    
    # Token Generation Metrics
    tokens_generated_total,
    tokens_per_request,
    tokens_per_second,
    average_tokens_per_minute,
    
    # Model Performance
    model_loading_duration_seconds,
    model_cache_hit_ratio,
    batch_size,
    batches_processed_total,
    
    # Resource Utilization
    gpu_memory_usage_bytes,
    gpu_memory_reserved_bytes,
    gpu_utilization_percent,
    memory_efficiency_ratio,
    
    # Error and Quality
    inference_errors_total,
    inference_retries_total,
    inference_queue_size,
    inference_queue_wait_seconds,
    
    # System Info
    system_info,
    model_info,
    
    # Decorators
    track_inference_request,
    track_token_generation,
    InferenceContext,
    
    # Helpers
    update_gpu_metrics,
    record_inference_error,
    update_queue_metrics,
    record_queue_wait,
    update_token_throughput,
    record_batch_processed,
    record_model_load_time,
    update_cache_metrics,
)

__all__ = [
    "inference_requests_total",
    "inference_request_duration_seconds",
    "inference_ttft_seconds",
    "inference_inter_token_latency_seconds",
    "tokens_generated_total",
    "tokens_per_request",
    "tokens_per_second",
    "average_tokens_per_minute",
    "model_loading_duration_seconds",
    "model_cache_hit_ratio",
    "batch_size",
    "batches_processed_total",
    "gpu_memory_usage_bytes",
    "gpu_memory_reserved_bytes",
    "gpu_utilization_percent",
    "memory_efficiency_ratio",
    "inference_errors_total",
    "inference_retries_total",
    "inference_queue_size",
    "inference_queue_wait_seconds",
    "system_info",
    "model_info",
    "track_inference_request",
    "track_token_generation",
    "InferenceContext",
    "update_gpu_metrics",
    "record_inference_error",
    "update_queue_metrics",
    "record_queue_wait",
    "update_token_throughput",
    "record_batch_processed",
    "record_model_load_time",
    "update_cache_metrics",
]
