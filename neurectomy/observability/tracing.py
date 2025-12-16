"""OpenTelemetry distributed tracing."""

from contextlib import contextmanager
from functools import wraps
import time
from typing import Dict, Any, Optional, Generator
from dataclasses import dataclass, field


@dataclass
class Span:
    """Represents a span in a trace."""
    name: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value
    
    def end(self) -> None:
        """End the span."""
        self.end_time = time.time()


class Tracer:
    """Tracer for distributed tracing."""
    
    def __init__(self, service_name: str = "neurectomy"):
        self.service_name = service_name
        self._spans: list = []
    
    @contextmanager
    def start_span(self, name: str) -> Generator[Span, None, None]:
        """Start a new span."""
        span = Span(name)
        self._spans.append(span)
        try:
            yield span
        finally:
            span.end()
    
    def get_spans(self) -> list:
        """Get all recorded spans."""
        return self._spans
    
    def clear_spans(self) -> None:
        """Clear recorded spans."""
        self._spans.clear()


# Global tracer instance
_tracer: Optional[Tracer] = None


def get_tracer(service_name: str = "neurectomy") -> Tracer:
    """Get the global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = Tracer(service_name)
    return _tracer


def trace(name: str = None):
    """Decorator to trace a function."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            span_name = name or func.__name__
            with tracer.start_span(span_name) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("status", "success")
                    return result
                except Exception as e:
                    span.set_attribute("status", "error")
                    span.set_attribute("error_type", type(e).__name__)
                    span.set_attribute("error_message", str(e))
                    raise
        return wrapper
    return decorator
