"""
Performance Monitoring
======================

Real-time performance tracking for the Neurectomy system.
"""

import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from contextlib import contextmanager
import threading


@dataclass
class MetricPoint:
    """Single metric measurement."""
    value: float
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""
    count: int = 0
    total: float = 0.0
    min_value: float = float('inf')
    max_value: float = float('-inf')
    
    @property
    def avg(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0
    
    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "count": self.count,
            "total": self.total,
            "avg": self.avg,
            "min": self.min_value if self.count > 0 else 0,
            "max": self.max_value if self.count > 0 else 0,
        }


class MetricsCollector:
    """
    Collects and aggregates metrics.
    """
    
    def __init__(self, retention_minutes: int = 60):
        self._metrics: Dict[str, List[MetricPoint]] = defaultdict(list)
        self._summaries: Dict[str, MetricSummary] = defaultdict(
            MetricSummary
        )
        self._retention = timedelta(minutes=retention_minutes)
        self._lock = threading.Lock()
    
    def record(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a metric value."""
        point = MetricPoint(value=value, labels=labels or {})
        
        with self._lock:
            self._metrics[name].append(point)
            self._summaries[name].add(value)
            self._cleanup(name)
    
    def get_summary(self, name: str) -> MetricSummary:
        """Get summary for a metric."""
        return self._summaries.get(name, MetricSummary())
    
    def get_recent(
        self,
        name: str,
        minutes: int = 5,
    ) -> List[MetricPoint]:
        """Get recent metric points."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        
        with self._lock:
            return [
                p for p in self._metrics.get(name, [])
                if p.timestamp >= cutoff
            ]
    
    def get_all_summaries(self) -> Dict[str, Dict[str, float]]:
        """Get all metric summaries."""
        return {
            name: summary.to_dict()
            for name, summary in self._summaries.items()
        }
    
    def _cleanup(self, name: str) -> None:
        """Remove old metric points."""
        cutoff = datetime.now(timezone.utc) - self._retention
        self._metrics[name] = [
            p for p in self._metrics[name]
            if p.timestamp >= cutoff
        ]


class PerformanceMonitor:
    """
    High-level performance monitoring.
    
    Tracks:
    - Inference latency
    - Compression ratios
    - Cache hit rates
    - Agent performance
    - System resources
    """
    
    def __init__(self):
        self._collector = MetricsCollector()
        self._start_time = datetime.now(timezone.utc)
    
    @contextmanager
    def measure(
        self,
        operation: str,
        labels: Optional[Dict[str, str]] = None
    ):
        """Context manager to measure operation duration."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self._collector.record(
                f"{operation}_latency_ms",
                duration_ms,
                labels
            )
    
    def record_inference(
        self,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float,
        model: str = "default",
    ) -> None:
        """Record inference metrics."""
        labels = {"model": model}
        self._collector.record("inference_latency_ms", latency_ms, labels)
        self._collector.record("inference_tokens_in", tokens_in, labels)
        self._collector.record("inference_tokens_out", tokens_out, labels)
        
        if latency_ms > 0:
            tokens_per_sec = (tokens_out / latency_ms) * 1000
            self._collector.record(
                "inference_tokens_per_sec",
                tokens_per_sec,
                labels
            )
    
    def record_compression(
        self,
        original_tokens: int,
        compressed_glyphs: int,
        latency_ms: float,
    ) -> None:
        """Record compression metrics."""
        ratio = (
            original_tokens / compressed_glyphs
            if compressed_glyphs > 0 else 1.0
        )
        
        self._collector.record("compression_ratio", ratio)
        self._collector.record("compression_latency_ms", latency_ms)
        self._collector.record("compression_tokens", original_tokens)
    
    def record_cache_access(self, hit: bool) -> None:
        """Record cache access."""
        self._collector.record("cache_hit", 1.0 if hit else 0.0)
    
    def record_agent_task(
        self,
        agent_id: str,
        task_type: str,
        latency_ms: float,
        success: bool,
    ) -> None:
        """Record agent task metrics."""
        labels = {"agent_id": agent_id, "task_type": task_type}
        
        self._collector.record("agent_task_latency_ms", latency_ms, labels)
        self._collector.record(
            "agent_task_success",
            1.0 if success else 0.0,
            labels
        )
    
    def record_storage_operation(
        self,
        operation: str,
        latency_ms: float,
        bytes_transferred: int = 0,
    ) -> None:
        """Record storage operation metrics."""
        self._collector.record(f"storage_{operation}_latency_ms", latency_ms)
        if bytes_transferred > 0:
            self._collector.record(
                f"storage_{operation}_bytes",
                bytes_transferred
            )
    
    def get_dashboard(self) -> Dict[str, Any]:
        """Get metrics dashboard."""
        summaries = self._collector.get_all_summaries()
        uptime = (
            datetime.now(timezone.utc) - self._start_time
        ).total_seconds()
        
        # Calculate derived metrics
        cache_summary = summaries.get("cache_hit", {})
        cache_hit_rate = cache_summary.get("avg", 0.0) if cache_summary else 0.0
        
        return {
            "uptime_seconds": uptime,
            "inference": {
                "latency_ms": summaries.get("inference_latency_ms", {}),
                "tokens_per_sec": summaries.get("inference_tokens_per_sec", {}),
            },
            "compression": {
                "ratio": summaries.get("compression_ratio", {}),
                "latency_ms": summaries.get("compression_latency_ms", {}),
            },
            "cache": {
                "hit_rate": cache_hit_rate,
            },
            "agents": {
                "task_latency_ms": summaries.get("agent_task_latency_ms", {}),
                "success_rate": summaries.get(
                    "agent_task_success", {}
                ).get("avg", 0.0),
            },
            "storage": {
                "read_latency_ms": summaries.get(
                    "storage_read_latency_ms", {}
                ),
                "write_latency_ms": summaries.get(
                    "storage_write_latency_ms", {}
                ),
            },
        }


# Global monitor instance
_monitor = PerformanceMonitor()


def get_monitor() -> PerformanceMonitor:
    """Get the global performance monitor."""
    return _monitor
