"""Prometheus metrics collection."""

from collections import defaultdict
from typing import Dict, List, Optional, Any


class Counter:
    """Counter metric."""
    
    def __init__(self, name: str, labels: List[str] = None):
        self.name = name
        self.labels = labels or []
        self._values: Dict[tuple, float] = defaultdict(float)
    
    def inc(self, value: float = 1, **label_values) -> None:
        """Increment counter."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        self._values[key] += value
    
    def get_value(self, **label_values) -> float:
        """Get counter value."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        return self._values.get(key, 0)


class Histogram:
    """Histogram metric."""
    
    def __init__(self, name: str, labels: List[str] = None, buckets: List[float] = None):
        self.name = name
        self.labels = labels or []
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        self._observations: Dict[tuple, list] = defaultdict(list)
    
    def observe(self, value: float, **label_values) -> None:
        """Observe a value."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        self._observations[key].append(value)
    
    def get_observations(self, **label_values) -> list:
        """Get observations."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        return self._observations.get(key, [])
    
    def get_stats(self, **label_values) -> Dict[str, float]:
        """Get histogram statistics."""
        observations = self.get_observations(**label_values)
        if not observations:
            return {"count": 0, "sum": 0, "min": 0, "max": 0}
        
        return {
            "count": len(observations),
            "sum": sum(observations),
            "min": min(observations),
            "max": max(observations),
            "mean": sum(observations) / len(observations),
        }


class Gauge:
    """Gauge metric."""
    
    def __init__(self, name: str, labels: List[str] = None):
        self.name = name
        self.labels = labels or []
        self._values: Dict[tuple, float] = defaultdict(float)
    
    def set(self, value: float, **label_values) -> None:
        """Set gauge value."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        self._values[key] = value
    
    def get_value(self, **label_values) -> float:
        """Get gauge value."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        return self._values.get(key, 0)


class MetricsRegistry:
    """Central registry for metrics."""
    
    def __init__(self):
        self._counters: Dict[str, Counter] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._setup_defaults()
    
    def _setup_defaults(self) -> None:
        """Setup default metrics."""
        self._counters["neurectomy_requests_total"] = Counter(
            "neurectomy_requests_total",
            ["endpoint", "status", "tenant_id"]
        )
        self._histograms["neurectomy_latency_seconds"] = Histogram(
            "neurectomy_latency_seconds",
            ["endpoint", "tenant_id"]
        )
        self._gauges["neurectomy_active_requests"] = Gauge(
            "neurectomy_active_requests",
            ["tenant_id"]
        )
    
    def get_counter(self, name: str) -> Optional[Counter]:
        """Get a counter metric."""
        return self._counters.get(name)
    
    def get_histogram(self, name: str) -> Optional[Histogram]:
        """Get a histogram metric."""
        return self._histograms.get(name)
    
    def get_gauge(self, name: str) -> Optional[Gauge]:
        """Get a gauge metric."""
        return self._gauges.get(name)
    
    def register_counter(self, name: str, labels: List[str] = None) -> Counter:
        """Register a new counter."""
        counter = Counter(name, labels)
        self._counters[name] = counter
        return counter
    
    def register_histogram(self, name: str, labels: List[str] = None) -> Histogram:
        """Register a new histogram."""
        histogram = Histogram(name, labels)
        self._histograms[name] = histogram
        return histogram
    
    def register_gauge(self, name: str, labels: List[str] = None) -> Gauge:
        """Register a new gauge."""
        gauge = Gauge(name, labels)
        self._gauges[name] = gauge
        return gauge


# Global metrics registry
metrics = MetricsRegistry()
