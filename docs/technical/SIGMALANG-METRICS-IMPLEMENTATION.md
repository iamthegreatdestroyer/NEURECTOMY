# ΣLANG Performance Metrics - Implementation Guide

## Production-Ready Code and Configuration

**Date**: December 16, 2025  
**Status**: Ready for Implementation  
**Target Files**: `sigmalang/monitoring/metrics.py`, Prometheus configs, Grafana dashboards

---

## Part 1: Complete Metrics Implementation

### 1.1 Core Metrics Registry (`sigmalang/monitoring/metrics.py`)

```python
"""
ΣLANG Compression Service - Performance Metrics

Comprehensive metrics design for identifying optimization opportunities
and monitoring compression performance from a performance engineering perspective.

Key Design Principle: Every metric answers "Where should we optimize next?"
"""

from prometheus_client import Counter, Histogram, Gauge, Summary, Info
from functools import wraps
from typing import Optional, Dict, Any, Callable
import time
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# SECTION 1: COMPRESSION OPERATION METRICS
# ============================================================================

# Compression request tracking
sigmalang_compression_requests_total = Counter(
    'sigmalang_compression_requests_total',
    'Total compression requests with outcome categorization',
    ['status', 'algorithm', 'data_type', 'encoding_mode'],
    help='status: success|failure|cache_hit|cache_miss; '
         'algorithm: semantic_primitive|learned_pattern|delta_encode|hybrid; '
         'data_type: token_sequence|text|semantic_tree; '
         'encoding_mode: balanced|aggressive|streaming'
)

sigmalang_compression_outcome_counts = Counter(
    'sigmalang_compression_outcome_counts',
    'Granular compression outcome tracking for bottleneck identification',
    ['outcome', 'algorithm_family', 'compression_quality'],
    help='outcome: encode_success|decode_success|encode_failure|decode_failure|'
         'ratio_excellent|ratio_good|ratio_poor|ratio_uncompressible'
)

# Compression ratio tracking
sigmalang_compression_ratio = Histogram(
    'sigmalang_compression_ratio',
    'Compression ratio distribution (original_size / compressed_size)',
    ['algorithm', 'data_category', 'size_range'],
    buckets=[1.5, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100],
    help='Buckets target sub-linear optimization opportunities: '
         'size_range: tiny_<1kb|small_1-10kb|medium_10-100kb|large_>100kb'
)

sigmalang_compression_ratio_trend = Gauge(
    'sigmalang_compression_ratio_trend',
    'Moving average compression ratio to detect degradation or improvement trends',
    ['algorithm', 'time_window'],
    help='time_window: 1h|1d|1w - detect ratio trends for learning engine'
)

# Compression speed/duration metrics
sigmalang_compression_duration_seconds = Histogram(
    'sigmalang_compression_duration_seconds',
    'Time to compress data with ΣLANG (total operation)',
    ['algorithm', 'compression_level', 'input_size_range'],
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
    help='compression_level: 1_fast|5_balanced|9_maximum; '
         'input_size_range: tiny|small|medium|large|huge'
)

sigmalang_compression_phase_duration_seconds = Histogram(
    'sigmalang_compression_phase_duration_seconds',
    'Compression phase-level timing for bottleneck identification',
    ['phase', 'algorithm'],
    buckets=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
    help='phase: parse|encode|optimize|serialize - identify slow phases'
)

sigmalang_compression_throughput_bytes_per_second = Gauge(
    'sigmalang_compression_throughput_bytes_per_second',
    'Real-time throughput in bytes/second during compression',
    ['algorithm', 'mode'],
    help='mode: streaming|batch|adaptive - algorithm throughput performance'
)

# Decompression metrics
sigmalang_decompression_requests_total = Counter(
    'sigmalang_decompression_requests_total',
    'Total decompression operations and outcomes',
    ['status', 'decode_algorithm'],
    help='status: success|failure|cache_hit; '
         'decode_algorithm: direct|pattern_lookup|delta_reversal'
)

sigmalang_decompression_duration_seconds = Histogram(
    'sigmalang_decompression_duration_seconds',
    'Decompression latency (should be 2-10x faster than compression)',
    ['decode_algorithm', 'output_size_range'],
    buckets=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1.0, 10.0],
    help='output_size_range: tiny|small|medium|large|huge'
)

sigmalang_encode_decode_consistency = Gauge(
    'sigmalang_encode_decode_consistency',
    'Round-trip encode→decode consistency (should be 100%)',
    ['metric'],
    help='metric: fidelity_percentage|token_match_rate - target 99.9%+'
)

# Effectiveness metrics
sigmalang_compression_effectiveness_score = Gauge(
    'sigmalang_compression_effectiveness_score',
    'Composite score: (ratio - 1) / (expected_ratio - 1), normalized [0-1]',
    ['algorithm', 'data_category'],
    help='1.0 = meeting target, <1.0 = below target, >1.0 = exceeding target'
)

sigmalang_semantic_redundancy_detected = Counter(
    'sigmalang_semantic_redundancy_detected',
    'Count of optimization opportunities discovered',
    ['redundancy_type', 'reduction_potential_percent'],
    help='redundancy_type: repeated_pattern|common_substructure|'
         'learned_primitive|rsu_reuse; reduction_potential_percent: 10|25|50|75'
)

# ============================================================================
# SECTION 2: SIZE TRACKING METRICS
# ============================================================================

sigmalang_input_size_bytes = Histogram(
    'sigmalang_input_size_bytes',
    'Distribution of input data sizes before compression',
    ['data_type', 'source'],
    buckets=[100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 10000000],
    help='data_type: token_sequence|text|semantic_tree; '
         'source: user_input|context_window|conversation_history'
)

sigmalang_input_size_percentiles = Gauge(
    'sigmalang_input_size_percentiles',
    'Percentile distribution of input sizes',
    ['percentile'],
    help='percentile: p50|p75|p90|p95|p99 - workload characterization'
)

sigmalang_output_size_bytes = Histogram(
    'sigmalang_output_size_bytes',
    'Distribution of compressed data sizes',
    ['algorithm', 'compression_level'],
    buckets=[10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000],
    help='Guide storage planning and cache sizing'
)

sigmalang_space_savings_bytes = Gauge(
    'sigmalang_space_savings_bytes',
    'Cumulative bytes saved: sum(original_size - compressed_size)',
    ['time_window'],
    help='time_window: last_hour|last_day|since_start - business impact metric'
)

sigmalang_cumulative_compression_ratio = Gauge(
    'sigmalang_cumulative_compression_ratio',
    'Overall compression ratio across entire time window',
    ['time_window'],
    help='time_window: 1h|1d|1w|all_time - trend detection for learning engine'
)

# ============================================================================
# SECTION 3: PERFORMANCE CHARACTERISTICS METRICS
# ============================================================================

sigmalang_algorithm_selection_accuracy = Gauge(
    'sigmalang_algorithm_selection_accuracy',
    'Percentage of algorithm selections achieving target ratio',
    ['selection_strategy'],
    help='selection_strategy: heuristic|learned|hybrid - guide heuristic tuning'
)

sigmalang_algorithm_efficiency_comparison = Histogram(
    'sigmalang_algorithm_efficiency_comparison',
    'Relative efficiency between algorithm pairs',
    ['algorithm_pair', 'metric'],
    buckets=[0.5, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 2.0],
    help='metric: ratio_delta|speed_delta|efficiency_ratio'
)

sigmalang_throughput_categories = Counter(
    'sigmalang_throughput_categories',
    'Categorize compression throughput into performance buckets',
    ['category'],
    help='category: excellent_>100mb|good_10-100mb|acceptable_1-10mb|slow_<1mb'
)

# Cache performance metrics
sigmalang_cache_statistics = Gauge(
    'sigmalang_cache_statistics',
    'Cache performance at each tier',
    ['cache_level', 'metric'],
    help='cache_level: l1_rsu|l2_pattern|l3_lru; '
         'metric: hit_rate|miss_rate|eviction_rate|size_percent'
)

sigmalang_cache_eviction_reason_distribution = Counter(
    'sigmalang_cache_eviction_reason_distribution',
    'Why items were evicted from cache',
    ['reason'],
    help='reason: lru|size_pressure|age_threshold|manual_clear'
)

# Compression level tradeoffs
sigmalang_compression_level_tradeoff = Histogram(
    'sigmalang_compression_level_tradeoff',
    'Compression level impact on ratio and speed',
    ['level', 'metric'],
    buckets=[Varies by metric],
    help='level: 1_fast|5_balanced|9_maximum; metric: ratio|speed_ms|efficiency'
)

# ============================================================================
# SECTION 4: RESOURCE UTILIZATION METRICS
# ============================================================================

sigmalang_cpu_usage_percent = Gauge(
    'sigmalang_cpu_usage_percent',
    'CPU utilization percentage during compression',
    ['algorithm', 'cpu_core'],
    help='Identify CPU bottlenecks and parallelization opportunities'
)

sigmalang_cpu_cycles_per_byte = Gauge(
    'sigmalang_cpu_cycles_per_byte',
    'CPU cycles required to compress one byte',
    ['algorithm'],
    help='Target ranges: semantic 100-500, pattern 200-1000, delta 50-300'
)

sigmalang_branch_misprediction_rate = Gauge(
    'sigmalang_branch_misprediction_rate',
    'CPU branch misprediction rate from perf profiling',
    ['algorithm'],
    help='Red flag >10%: indicates unpredictable hot path branches'
)

sigmalang_memory_usage_bytes = Gauge(
    'sigmalang_memory_usage_bytes',
    'Memory consumption by component',
    ['component', 'algorithm'],
    help='component: rsu_cache|pattern_cache|lru_cache|working_set'
)

sigmalang_cache_memory_efficiency = Gauge(
    'sigmalang_cache_memory_efficiency',
    'Bytes stored per cache hit (lower = more efficient)',
    ['cache_level'],
    help='Target: L1 1-10KB/hit, L2 50-500B/hit, L3 100-1000B/hit'
)

sigmalang_memory_allocation_pattern = Counter(
    'sigmalang_memory_allocation_pattern',
    'Classification of memory allocation behavior',
    ['pattern'],
    help='pattern: small_frequent|large_burst|steady_growth - detect leaks'
)

# I/O metrics
sigmalang_io_operations_total = Counter(
    'sigmalang_io_operations_total',
    'I/O operations for compression service',
    ['operation', 'io_type'],
    help='operation: read_codebook|write_cache|read_cache|checkpoint_save; '
         'io_type: memory|disk|network'
)

sigmalang_io_latency_milliseconds = Histogram(
    'sigmalang_io_latency_milliseconds',
    'I/O operation latency',
    ['operation'],
    buckets=[0.1, 1, 10, 100, 1000, 10000],
    help='Identify I/O contention or network bottlenecks'
)

sigmalang_io_throughput_mb_per_second = Gauge(
    'sigmalang_io_throughput_mb_per_second',
    'I/O throughput for streaming operations',
    ['operation'],
    help='Baseline: disk read 100-500MB/s, disk write 50-300MB/s'
)

# Parallelization metrics
sigmalang_parallel_efficiency = Gauge(
    'sigmalang_parallel_efficiency',
    'Parallel speedup as fraction of theoretical (1.0 = linear)',
    ['algorithm', 'worker_count'],
    help='Target >80%: minimal synchronization overhead'
)

sigmalang_worker_utilization_percent = Gauge(
    'sigmalang_worker_utilization_percent',
    'Per-worker utilization to detect load imbalance',
    ['worker_id', 'metric'],
    help='metric: cpu_percent|idle_time_percent - detect imbalance'
)

sigmalang_thread_contention_events = Counter(
    'sigmalang_thread_contention_events',
    'Number of lock contention events detected',
    ['lock_name'],
    help='Red flag >1000/sec: switch to lock-free or RwLock'
)

# ============================================================================
# SECTION 5: OPTIMIZATION OPPORTUNITY METRICS
# ============================================================================

sigmalang_bottleneck_detection = Gauge(
    'sigmalang_bottleneck_detection',
    'Identified bottleneck and its contribution to total latency',
    ['bottleneck_type', 'severity'],
    help='bottleneck_type: memory_bandwidth|cpu_compute|cache_misses|io_wait|'
         'lock_contention; severity: critical|high|medium|low'
)

sigmalang_optimization_opportunity_score = Gauge(
    'sigmalang_optimization_opportunity_score',
    'Estimated impact if optimization completed (0-1 scale)',
    ['opportunity'],
    help='opportunity: cache_miss_reduction|algorithm_selection|compression_level|'
         'memory_optimization|parallelization|io_batching'
)

sigmalang_compression_ratio_anomaly = Gauge(
    'sigmalang_compression_ratio_anomaly',
    'Deviation from expected compression ratio in standard deviations',
    ['anomaly_type', 'algorithm'],
    help='anomaly_type: below_expected|above_expected|high_variance - '
         'z_score for statistical detection'
)

sigmalang_compression_ratio_forecast = Gauge(
    'sigmalang_compression_ratio_forecast',
    'Predicted compression ratio using trend + seasonality',
    ['algorithm', 'forecast_horizon'],
    help='forecast_horizon: next_hour|next_day - predict ratio degradation'
)

sigmalang_algorithm_selection_efficiency = Gauge(
    'sigmalang_algorithm_selection_efficiency',
    'How well algorithm selection matches optimal for scenario',
    ['scenario'],
    help='scenario: token_sequences|code_snippets|natural_language|mixed_content'
)

sigmalang_suboptimal_scenario_detection = Counter(
    'sigmalang_suboptimal_scenario_detection',
    'Detected scenarios where compression could be improved',
    ['scenario', 'suboptimal_reason'],
    help='suboptimal_reason: algorithm_mismatch|compression_level_too_low|'
         'cache_miss|pattern_not_learned'
)

# ============================================================================
# SECTION 6: DECORATOR FOR AUTOMATIC METRIC COLLECTION
# ============================================================================

def track_compression_operation(
    algorithm: str,
    compression_level: str = '5_balanced',
    data_type: str = 'token_sequence'
):
    """
    Decorator to automatically track compression metrics.

    Usage:
        @track_compression_operation('semantic_primitive', '5_balanced')
        def compress_data(data: bytes) -> Tuple[bytes, dict]:
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Start timing
            start_time = time.time()
            start_phases = {
                'parse': 0,
                'encode': 0,
                'optimize': 0,
                'serialize': 0
            }

            status = 'success'
            ratio = 0.0
            input_size = 0
            output_size = 0

            try:
                # Execute compression
                result = func(*args, **kwargs)

                # Extract metrics from result
                if isinstance(result, dict):
                    output_data = result.get('compressed_data', b'')
                    metadata = result.get('metadata', {})
                    input_size = metadata.get('original_size', 0)
                    output_size = len(output_data)

                    # Calculate ratio
                    if output_size > 0:
                        ratio = input_size / output_size
                else:
                    output_size = len(result) if isinstance(result, bytes) else 0

                # Determine outcome
                if ratio >= 10:
                    outcome = 'ratio_excellent'
                elif ratio >= 5:
                    outcome = 'ratio_good'
                elif ratio > 1.5:
                    outcome = 'ratio_poor'
                else:
                    outcome = 'ratio_uncompressible'

                status = 'success'

                return result

            except Exception as e:
                status = 'failure'
                outcome = 'encode_failure'
                logger.error(f"Compression error: {e}")
                raise

            finally:
                # Record metrics
                duration = time.time() - start_time

                # Ratio histogram
                if ratio > 1.0:
                    sigmalang_compression_ratio.labels(
                        algorithm=algorithm,
                        data_category='general',
                        size_range='medium_10-100kb' if 10000 <= input_size < 100000 else 'other'
                    ).observe(ratio)

                # Duration histogram
                sigmalang_compression_duration_seconds.labels(
                    algorithm=algorithm,
                    compression_level=compression_level,
                    input_size_range='small_1-10kb' if input_size < 10000 else 'medium'
                ).observe(duration)

                # Request counter
                sigmalang_compression_requests_total.labels(
                    status=status,
                    algorithm=algorithm,
                    data_type=data_type,
                    encoding_mode='balanced'
                ).inc()

                # Outcome counter
                if status == 'success':
                    sigmalang_compression_outcome_counts.labels(
                        outcome=outcome,
                        algorithm_family='semantic',
                        compression_quality='balanced'
                    ).inc()

                # Throughput
                if duration > 0 and input_size > 0:
                    throughput = input_size / duration
                    sigmalang_compression_throughput_bytes_per_second.labels(
                        algorithm=algorithm,
                        mode='batch'
                    ).set(throughput)

        return wrapper
    return decorator


def track_decompression_operation(decode_algorithm: str = 'direct'):
    """
    Decorator to track decompression metrics.

    Usage:
        @track_decompression_operation('direct')
        def decompress_data(compressed: bytes) -> bytes:
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'
            output_size = 0

            try:
                result = func(*args, **kwargs)
                output_size = len(result) if isinstance(result, bytes) else 0
                return result

            except Exception as e:
                status = 'failure'
                logger.error(f"Decompression error: {e}")
                raise

            finally:
                duration = time.time() - start_time

                sigmalang_decompression_requests_total.labels(
                    status=status,
                    decode_algorithm=decode_algorithm
                ).inc()

                sigmalang_decompression_duration_seconds.labels(
                    decode_algorithm=decode_algorithm,
                    output_size_range='small' if output_size < 10000 else 'large'
                ).observe(duration)

        return wrapper
    return decorator
```

---

## Part 2: Prometheus Configuration

### 2.1 Scrape Configuration (`prometheus/prometheus.yml`)

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: "sigmalang"
    static_configs:
      - targets: ["localhost:8000"] # ΣLANG metrics port

    # Reduce cardinality for high-volume metrics
    metric_relabeling:
      - source_labels: [__name__]
        regex: "sigmalang_compression_ratio"
        action: keep
      - source_labels: [__name__]
        regex: "sigmalang_compression_duration_seconds"
        action: keep
      - source_labels: [__name__]
        regex: "sigmalang_.*_total"
        action: keep
      - source_labels: [__name__]
        regex: "sigmalang_.*_bytes"
        action: keep
      - source_labels: [__name__]
        regex: "sigmalang_cache_statistics"
        action: keep
      - source_labels: [__name__]
        regex: "sigmalang_.*_percent"
        action: keep
```

### 2.2 Alert Rules (`prometheus/rules/sigmalang_alerts.yml`)

```yaml
groups:
  - name: sigmalang_alerts
    interval: 30s
    rules:
      # Performance Alerts
      - alert: CompressionRatioDegraded
        expr: >
          sigmalang_compression_ratio_trend < 5
          and 
          sigmalang_compression_requests_total > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "ΣLANG compression ratio degraded below 5x"
          description: "Current ratio: {{ $value }}"

      - alert: HighCompressionLatency
        expr: >
          histogram_quantile(0.99, 
            sigmalang_compression_duration_seconds_bucket) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "ΣLANG compression p99 latency > 100ms"
          description: "P99 latency: {{ $value }}s"

      - alert: LowCacheHitRate
        expr: >
          (
            increase(sigmalang_compression_requests_total{status="cache_hit"}[5m])
            /
            increase(sigmalang_compression_requests_total[5m])
          ) < 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "ΣLANG cache hit rate < 50%"
          description: "Hit rate: {{ $value | humanizePercentage }}"

      # Resource Alerts
      - alert: HighMemoryUsage
        expr: >
          sigmalang_memory_usage_bytes{component="rsu_cache"} > 
          (70 * 1024 * 1024)
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "ΣLANG RSU cache memory > 70MB"
          description: "Memory: {{ $value | humanize1024 }}B"

      - alert: HighCPUUsage
        expr: >
          sigmalang_cpu_usage_percent > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "ΣLANG CPU usage > 80%"
          description: "CPU: {{ $value }}%"

      # Failure Alerts
      - alert: CompressionFailureRate
        expr: >
          (
            increase(sigmalang_compression_requests_total{status="failure"}[5m])
            /
            increase(sigmalang_compression_requests_total[5m])
          ) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "ΣLANG failure rate > 1%"
          description: "Failure rate: {{ $value | humanizePercentage }}"

      - alert: DecompressionFailureRate
        expr: >
          (
            increase(sigmalang_decompression_requests_total{status="failure"}[5m])
            /
            increase(sigmalang_decompression_requests_total[5m])
          ) > 0.001
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "ΣLANG decompression failure rate > 0.1%"
          description: "Failure rate: {{ $value | humanizePercentage }}"

      # Optimization Alerts
      - alert: SuboptimalCompressionDetected
        expr: >
          increase(sigmalang_suboptimal_scenario_detection[1h]) > 10
        for: 5m
        labels:
          severity: info
        annotations:
          summary: "{{ $value }} suboptimal compression scenarios in last hour"
          description: "Review algorithm selection heuristic"

      - alert: LockContentionHigh
        expr: >
          increase(sigmalang_thread_contention_events[1m]) > 1000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "ΣLANG lock contention > 1000 events/sec"
          description: "Consider lock-free structures"
```

---

## Part 3: Grafana Dashboard JSON

### 3.1 Compression Overview Dashboard

```json
{
  "dashboard": {
    "title": "ΣLANG Compression Overview",
    "tags": ["compression", "performance"],
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(sigmalang_compression_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Avg Compression Ratio",
        "type": "gauge",
        "targets": [
          {
            "expr": "avg(sigmalang_compression_ratio)"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "thresholds": {
              "mode": "absolute",
              "steps": [
                { "color": "red", "value": 2 },
                { "color": "yellow", "value": 5 },
                { "color": "green", "value": 10 }
              ]
            }
          }
        }
      },
      {
        "title": "Space Saved (GB)",
        "type": "stat",
        "targets": [
          {
            "expr": "sigmalang_space_savings_bytes / 1024 / 1024 / 1024"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "type": "gauge",
        "targets": [
          {
            "expr": "(\n  increase(sigmalang_compression_requests_total{status=\"cache_hit\"}[5m])\n  /\n  increase(sigmalang_compression_requests_total[5m])\n) * 100"
          }
        ]
      },
      {
        "title": "Compression Ratio by Algorithm",
        "type": "heatmap",
        "targets": [
          {
            "expr": "sigmalang_compression_ratio",
            "format": "heatmap",
            "legendFormat": "{{ algorithm }}"
          }
        ]
      },
      {
        "title": "Latency Percentiles",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, sigmalang_compression_duration_seconds_bucket)",
            "legendFormat": "p50"
          },
          {
            "expr": "histogram_quantile(0.95, sigmalang_compression_duration_seconds_bucket)",
            "legendFormat": "p95"
          },
          {
            "expr": "histogram_quantile(0.99, sigmalang_compression_duration_seconds_bucket)",
            "legendFormat": "p99"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "(\n  increase(sigmalang_compression_requests_total{status=\"failure\"}[5m])\n  /\n  increase(sigmalang_compression_requests_total[5m])\n) * 100"
          }
        ]
      }
    ]
  }
}
```

---

## Part 4: Integration Testing

### 4.1 Metrics Collection Test

```python
"""
Test metrics collection and validation.

File: tests/test_compression_metrics.py
"""

import pytest
from prometheus_client import REGISTRY, CollectorRegistry
from sigmalang.monitoring.metrics import (
    sigmalang_compression_requests_total,
    sigmalang_compression_ratio,
    sigmalang_compression_duration_seconds,
    track_compression_operation
)

@pytest.fixture
def isolated_registry():
    """Use isolated registry for testing."""
    return CollectorRegistry()

def test_compression_ratio_tracking():
    """Test compression ratio histogram."""
    sigmalang_compression_ratio.labels(
        algorithm='semantic_primitive',
        data_category='text',
        size_range='medium_10-100kb'
    ).observe(12.5)

    # Verify bucket was recorded
    samples = list(sigmalang_compression_ratio.collect())[0].samples
    assert any(s.value > 0 for s in samples)

def test_decorator_tracks_metrics():
    """Test automatic metric tracking with decorator."""

    @track_compression_operation('semantic_primitive', '5_balanced')
    def mock_compress(data: bytes):
        return {
            'compressed_data': b'compressed',
            'metadata': {
                'original_size': len(data)
            }
        }

    # Execute compression
    input_data = b'x' * 10000
    result = mock_compress(input_data)

    # Verify metrics collected
    samples = list(sigmalang_compression_requests_total.collect())[0].samples
    assert any(s.value > 0 for s in samples)

def test_phase_duration_tracking():
    """Test phase-level timing."""
    sigmalang_compression_phase_duration_seconds.labels(
        phase='encode',
        algorithm='semantic_primitive'
    ).observe(0.005)

    samples = list(sigmalang_compression_phase_duration_seconds.collect())[0].samples
    assert any(s.value > 0 for s in samples)

def test_cache_statistics():
    """Test cache hit/miss rate calculation."""
    from sigmalang.monitoring.metrics import sigmalang_cache_statistics

    sigmalang_cache_statistics.labels(
        cache_level='l1_rsu',
        metric='hit_rate'
    ).set(0.75)

    samples = list(sigmalang_cache_statistics.collect())[0].samples
    assert any(abs(s.value - 0.75) < 0.01 for s in samples)
```

---

## Part 5: Configuration Reference

### 5.1 Metric Naming Convention

```
sigmalang_<metric_type>_<unit>

Prefixes:
- sigmalang_compression_* : Compression operations
- sigmalang_decompression_* : Decompression operations
- sigmalang_cache_* : Cache performance
- sigmalang_memory_* : Memory usage
- sigmalang_cpu_* : CPU metrics
- sigmalang_io_* : I/O operations
- sigmalang_thread_* : Threading/parallelization
- sigmalang_algorithm_* : Algorithm performance
- sigmalang_optimization_* : Optimization opportunities

Units:
- _total : Counter (increment only)
- _seconds : Histogram/Gauge (latency)
- _bytes : Histogram/Gauge (sizes)
- _percent : Gauge (percentages)
- _ratio : Histogram (dimensionless ratio)
```

### 5.2 Label Strategy

```
Standard labels across related metrics:
- algorithm: semantic_primitive|learned_pattern|delta_encode|hybrid
- status: success|failure|cache_hit|cache_miss
- data_type: token_sequence|text|semantic_tree
- component: rsu_cache|pattern_cache|lru_cache|working_set
- time_window: 1h|1d|1w|all_time

Optimization queries use labels as filters:
rate(sigmalang_compression_requests_total{status="cache_miss"}[5m])
```

---

## Deployment Checklist

- [ ] Deploy metrics code to `sigmalang/monitoring/metrics.py`
- [ ] Configure Prometheus scrape job for ΣLANG endpoint
- [ ] Import alert rules into Prometheus
- [ ] Create Grafana dashboards from JSON templates
- [ ] Add metrics endpoint to ΣLANG API server
- [ ] Verify metric collection in dev environment
- [ ] Load test to establish performance baselines
- [ ] Set up Grafana alerts for critical thresholds
- [ ] Document metric interpretation for team
- [ ] Create runbook for responding to alerts

---

**Implementation Ready**: Use this design document + code as complete specification for ΣLANG performance metrics production deployment.
