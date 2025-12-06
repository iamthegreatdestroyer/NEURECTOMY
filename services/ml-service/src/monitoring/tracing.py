# ============================================================================
# NEURECTOMY OpenTelemetry Tracing Module
# Distributed tracing with Jaeger export for FastAPI
# ============================================================================
"""
Distributed tracing configuration for the ML Service.

This module provides OpenTelemetry instrumentation for:
- FastAPI endpoints (automatic HTTP span creation)
- Database operations
- ML inference calls
- External API calls

Traces are exported to Jaeger for visualization and analysis.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, ParamSpec, TypeVar

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.trace import (
    Span,
    SpanKind,
    Status,
    StatusCode,
    get_current_span,
    get_tracer,
)
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

if TYPE_CHECKING:
    from fastapi import FastAPI

# Type variables for decorator typing
P = ParamSpec("P")
R = TypeVar("R")

# ============================================================================
# Configuration
# ============================================================================

JAEGER_HOST = os.getenv("JAEGER_HOST", "jaeger")
JAEGER_PORT = int(os.getenv("JAEGER_PORT", "6831"))
SERVICE_NAME = os.getenv("SERVICE_NAME", "neurectomy-ml-service")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
OTEL_ENABLED = os.getenv("OTEL_ENABLED", "true").lower() == "true"


# ============================================================================
# Tracer Provider Setup
# ============================================================================

_tracer_provider: TracerProvider | None = None
_tracer: trace.Tracer | None = None


def init_tracer(
    service_name: str = SERVICE_NAME,
    jaeger_host: str = JAEGER_HOST,
    jaeger_port: int = JAEGER_PORT,
    environment: str = ENVIRONMENT,
    use_batch_processor: bool = True,
) -> TracerProvider:
    """
    Initialize the OpenTelemetry tracer provider with Jaeger export.

    Args:
        service_name: Name of the service for trace identification
        jaeger_host: Jaeger agent hostname
        jaeger_port: Jaeger agent UDP port (typically 6831)
        environment: Deployment environment name
        use_batch_processor: Use batch export (True) or simple export (False)

    Returns:
        Configured TracerProvider instance
    """
    global _tracer_provider, _tracer

    if not OTEL_ENABLED:
        # Return a no-op tracer provider
        _tracer_provider = TracerProvider()
        trace.set_tracer_provider(_tracer_provider)
        _tracer = get_tracer(service_name)
        return _tracer_provider

    # Create resource with service information
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": os.getenv("SERVICE_VERSION", "0.2.0"),
            "deployment.environment": environment,
            "service.namespace": "neurectomy",
            "host.name": os.getenv("HOSTNAME", "localhost"),
        }
    )

    # Create tracer provider with resource
    _tracer_provider = TracerProvider(resource=resource)

    # Configure Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name=jaeger_host,
        agent_port=jaeger_port,
    )

    # Add span processor
    if use_batch_processor:
        span_processor = BatchSpanProcessor(
            jaeger_exporter,
            max_queue_size=2048,
            max_export_batch_size=512,
            schedule_delay_millis=5000,
        )
    else:
        span_processor = SimpleSpanProcessor(jaeger_exporter)

    _tracer_provider.add_span_processor(span_processor)

    # Set as global tracer provider
    trace.set_tracer_provider(_tracer_provider)

    # Get tracer for this service
    _tracer = get_tracer(service_name)

    return _tracer_provider


def get_service_tracer() -> trace.Tracer:
    """Get the service-level tracer, initializing if necessary."""
    global _tracer
    if _tracer is None:
        init_tracer()
    return _tracer or get_tracer(SERVICE_NAME)


def shutdown_tracer() -> None:
    """Shutdown the tracer provider and flush remaining spans."""
    global _tracer_provider
    if _tracer_provider:
        _tracer_provider.shutdown()
        _tracer_provider = None


# ============================================================================
# FastAPI Instrumentation
# ============================================================================


def instrument_fastapi(app: "FastAPI") -> None:
    """
    Instrument a FastAPI application with OpenTelemetry.

    This automatically creates spans for all HTTP requests with:
    - HTTP method, URL, and status code
    - Request/response headers (configurable)
    - Route path template
    - Client IP and user agent

    Args:
        app: FastAPI application instance
    """
    if not OTEL_ENABLED:
        return

    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=_tracer_provider,
        excluded_urls="/health,/ready,/metrics",
    )


def uninstrument_fastapi(app: "FastAPI") -> None:
    """Remove OpenTelemetry instrumentation from FastAPI app."""
    FastAPIInstrumentor.uninstrument_app(app)


# ============================================================================
# Span Creation Utilities
# ============================================================================


@contextmanager
def create_span(
    name: str,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: dict[str, Any] | None = None,
    links: list[Any] | None = None,
    context: Context | None = None,
):
    """
    Context manager for creating a traced span.

    Usage:
        with create_span("database.query", attributes={"db.table": "users"}) as span:
            result = execute_query()
            span.set_attribute("db.row_count", len(result))

    Args:
        name: Span name (should be descriptive, e.g., "db.query", "ml.inference")
        kind: Span kind (INTERNAL, CLIENT, SERVER, PRODUCER, CONSUMER)
        attributes: Initial span attributes
        links: Links to other spans
        context: Parent context (uses current context if not provided)

    Yields:
        The created span
    """
    tracer = get_service_tracer()
    with tracer.start_as_current_span(
        name,
        kind=kind,
        attributes=attributes,
        links=links,
        context=context,
    ) as span:
        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


def traced(
    name: str | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: dict[str, Any] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator to trace a function execution.

    Usage:
        @traced("ml.inference", attributes={"model.name": "embeddings"})
        async def generate_embeddings(texts: list[str]) -> list[Embedding]:
            ...

    Args:
        name: Span name (defaults to function name)
        kind: Span kind
        attributes: Static span attributes

    Returns:
        Decorated function with automatic span creation
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        span_name = name or f"{func.__module__}.{func.__name__}"

        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with create_span(span_name, kind=kind, attributes=attributes) as span:
                span.set_attribute("function.name", func.__name__)
                return await func(*args, **kwargs)  # type: ignore

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with create_span(span_name, kind=kind, attributes=attributes) as span:
                span.set_attribute("function.name", func.__name__)
                return func(*args, **kwargs)

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


# ============================================================================
# Span Attribute Helpers
# ============================================================================


def add_span_attributes(**attributes: Any) -> None:
    """
    Add attributes to the current span.

    Usage:
        add_span_attributes(
            user_id="123",
            model_version="1.0.0",
            inference_time_ms=45.2
        )
    """
    span = get_current_span()
    for key, value in attributes.items():
        span.set_attribute(key, value)


def add_span_event(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> None:
    """
    Add an event to the current span.

    Events are useful for recording points in time within a span.

    Usage:
        add_span_event("cache.miss", {"cache.key": "user:123"})
    """
    span = get_current_span()
    span.add_event(name, attributes=attributes)


def set_span_error(error: Exception, message: str | None = None) -> None:
    """
    Mark the current span as errored.

    Args:
        error: The exception that occurred
        message: Optional custom error message
    """
    span = get_current_span()
    span.set_status(Status(StatusCode.ERROR, message or str(error)))
    span.record_exception(error)


def get_trace_context() -> dict[str, str]:
    """
    Get the current trace context as a dictionary.

    Useful for propagating context to external services.

    Returns:
        Dictionary containing trace context headers
    """
    carrier: dict[str, str] = {}
    inject(carrier)
    return carrier


def extract_trace_context(headers: dict[str, str]) -> Context:
    """
    Extract trace context from headers.

    Useful for continuing a trace from an external service.

    Args:
        headers: Dictionary containing trace context headers

    Returns:
        Context object for continuing the trace
    """
    return extract(headers)


# ============================================================================
# ML-Specific Tracing Helpers
# ============================================================================


@contextmanager
def trace_inference(
    model_name: str,
    model_version: str | None = None,
    batch_size: int | None = None,
):
    """
    Context manager for tracing ML inference operations.

    Usage:
        with trace_inference("embedding-model", "v1.0", batch_size=32) as span:
            embeddings = model.encode(texts)
            span.set_attribute("output.dimensions", embeddings.shape[-1])

    Args:
        model_name: Name of the model
        model_version: Model version string
        batch_size: Input batch size
    """
    attributes = {
        "ml.model.name": model_name,
        "ml.operation": "inference",
    }
    if model_version:
        attributes["ml.model.version"] = model_version
    if batch_size is not None:
        attributes["ml.batch_size"] = batch_size

    with create_span(f"ml.inference.{model_name}", attributes=attributes) as span:
        yield span


@contextmanager
def trace_training(
    model_name: str,
    experiment_id: str | None = None,
    run_id: str | None = None,
):
    """
    Context manager for tracing ML training operations.

    Args:
        model_name: Name of the model being trained
        experiment_id: MLflow experiment ID
        run_id: MLflow run ID
    """
    attributes = {
        "ml.model.name": model_name,
        "ml.operation": "training",
    }
    if experiment_id:
        attributes["ml.experiment_id"] = experiment_id
    if run_id:
        attributes["ml.run_id"] = run_id

    with create_span(f"ml.training.{model_name}", attributes=attributes) as span:
        yield span


@contextmanager
def trace_embedding(
    model_name: str,
    text_count: int,
    dimensions: int | None = None,
):
    """
    Context manager for tracing embedding generation.

    Args:
        model_name: Name of the embedding model
        text_count: Number of texts being embedded
        dimensions: Embedding dimensions (if known)
    """
    attributes = {
        "ml.model.name": model_name,
        "ml.operation": "embedding",
        "ml.input_count": text_count,
    }
    if dimensions:
        attributes["ml.embedding_dimensions"] = dimensions

    with create_span(f"ml.embedding.{model_name}", attributes=attributes) as span:
        yield span


@contextmanager
def trace_llm_call(
    provider: str,
    model: str,
    prompt_tokens: int | None = None,
    max_tokens: int | None = None,
):
    """
    Context manager for tracing LLM API calls.

    Args:
        provider: LLM provider (openai, anthropic, ollama)
        model: Model name/ID
        prompt_tokens: Number of prompt tokens
        max_tokens: Maximum completion tokens
    """
    attributes = {
        "llm.provider": provider,
        "llm.model": model,
        "ml.operation": "llm_call",
    }
    if prompt_tokens is not None:
        attributes["llm.prompt_tokens"] = prompt_tokens
    if max_tokens is not None:
        attributes["llm.max_tokens"] = max_tokens

    with create_span(
        f"llm.{provider}.{model}",
        kind=SpanKind.CLIENT,
        attributes=attributes,
    ) as span:
        yield span


# ============================================================================
# Database Tracing Helpers
# ============================================================================


@contextmanager
def trace_db_operation(
    operation: str,
    table: str | None = None,
    statement: str | None = None,
):
    """
    Context manager for tracing database operations.

    Args:
        operation: DB operation (SELECT, INSERT, UPDATE, DELETE)
        table: Table name
        statement: SQL statement (will be truncated for safety)
    """
    attributes = {
        "db.system": "postgresql",
        "db.operation": operation,
    }
    if table:
        attributes["db.table"] = table
    if statement:
        # Truncate for safety - don't log full queries
        attributes["db.statement"] = statement[:100] + ("..." if len(statement) > 100 else "")

    with create_span(
        f"db.{operation.lower()}.{table or 'unknown'}",
        kind=SpanKind.CLIENT,
        attributes=attributes,
    ) as span:
        yield span


@contextmanager
def trace_cache_operation(
    operation: str,
    key: str | None = None,
    hit: bool | None = None,
):
    """
    Context manager for tracing cache operations.

    Args:
        operation: Cache operation (GET, SET, DELETE)
        key: Cache key (will be sanitized)
        hit: Whether it was a cache hit (for GET operations)
    """
    attributes = {
        "cache.system": "redis",
        "cache.operation": operation,
    }
    if key:
        # Sanitize key - just use prefix
        attributes["cache.key_prefix"] = key.split(":")[0] if ":" in key else key[:20]
    if hit is not None:
        attributes["cache.hit"] = hit

    with create_span(
        f"cache.{operation.lower()}",
        kind=SpanKind.CLIENT,
        attributes=attributes,
    ) as span:
        yield span


# ============================================================================
# Context Propagation
# ============================================================================

propagator = TraceContextTextMapPropagator()


def inject_context_to_headers(headers: dict[str, str]) -> dict[str, str]:
    """
    Inject current trace context into HTTP headers.

    Use this when making outgoing HTTP requests to propagate trace context.

    Args:
        headers: Existing headers dictionary

    Returns:
        Headers with trace context injected
    """
    inject(headers)
    return headers


def create_context_from_headers(headers: dict[str, str]) -> Context:
    """
    Extract trace context from incoming HTTP headers.

    Use this to continue a trace from an incoming request.

    Args:
        headers: Incoming request headers

    Returns:
        Context object to use for creating child spans
    """
    return propagator.extract(headers)


# ============================================================================
# Span Sampling Helpers
# ============================================================================


def should_sample_span(name: str, attributes: dict[str, Any] | None = None) -> bool:
    """
    Determine if a span should be sampled based on name and attributes.

    Implement custom sampling logic here. By default, always sample.

    Args:
        name: Span name
        attributes: Span attributes

    Returns:
        True if span should be sampled
    """
    # Always sample errors
    if attributes and attributes.get("error"):
        return True

    # Sample health checks at lower rate
    if "health" in name.lower():
        import random

        return random.random() < 0.01  # 1% sampling

    # Sample everything else
    return True
