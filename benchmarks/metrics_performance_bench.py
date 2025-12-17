"""
Performance Benchmarks for Metrics System
Phase 18A: Performance validation for ΣVAULT (18A-5) and Elite Agents (18A-6)
Validates decorator overhead (<150μs), context manager overhead (<80μs)
"""

import pytest
import time
import asyncio
import statistics
from decimal import Decimal
from typing import List, Callable
from unittest.mock import Mock, patch, AsyncMock

# Benchmark result tracking
class BenchmarkResult:
    """Container for benchmark results"""
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.times: List[float] = []
    
    def add_time(self, duration: float):
        """Add a timing measurement"""
        self.times.append(duration)
    
    @property
    def mean_us(self) -> float:
        """Mean duration in microseconds"""
        if not self.times:
            return 0.0
        return statistics.mean(self.times) * 1_000_000
    
    @property
    def median_us(self) -> float:
        """Median duration in microseconds"""
        if not self.times:
            return 0.0
        return statistics.median(self.times) * 1_000_000
    
    @property
    def p95_us(self) -> float:
        """95th percentile duration in microseconds"""
        if len(self.times) < 20:
            return max(self.times) * 1_000_000 if self.times else 0.0
        sorted_times = sorted(self.times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[idx] * 1_000_000
    
    @property
    def p99_us(self) -> float:
        """99th percentile duration in microseconds"""
        if len(self.times) < 100:
            return max(self.times) * 1_000_000 if self.times else 0.0
        sorted_times = sorted(self.times)
        idx = int(len(sorted_times) * 0.99)
        return sorted_times[idx] * 1_000_000


def benchmark_operation(operation: Callable, iterations: int = 1000) -> BenchmarkResult:
    """Benchmark an operation"""
    result = BenchmarkResult(operation.__name__)
    
    for _ in range(iterations):
        start = time.perf_counter()
        operation()
        duration = time.perf_counter() - start
        result.add_time(duration)
    
    return result


class TestCounterPerformance:
    """Benchmark counter metric performance"""
    
    def test_counter_increment_baseline(self):
        """Baseline counter increment performance"""
        from prometheus_client import Counter
        
        counter = Counter('test_counter', 'Test counter')
        
        def increment():
            counter.inc()
        
        result = benchmark_operation(increment, iterations=1000)
        
        print(f"\nCounter Increment Performance:")
        print(f"  Mean: {result.mean_us:.2f} μs")
        print(f"  Median: {result.median_us:.2f} μs")
        print(f"  P95: {result.p95_us:.2f} μs")
        
        # Should be very fast (< 10 μs)
        assert result.mean_us < 10
    
    def test_counter_with_labels(self):
        """Counter increment with labels"""
        from prometheus_client import Counter
        
        counter = Counter('test_counter_labels', 'Test counter', ['operation', 'status'])
        
        def increment():
            counter.labels(operation='store', status='success').inc()
        
        result = benchmark_operation(increment, iterations=1000)
        
        print(f"\nCounter with Labels:")
        print(f"  Mean: {result.mean_us:.2f} μs")
        print(f"  Median: {result.median_us:.2f} μs")
        print(f"  P95: {result.p95_us:.2f} μs")
        
        # With labels slightly slower but still < 20 μs
        assert result.mean_us < 20
    
    def test_counter_high_cardinality_labels(self):
        """Counter performance with high cardinality labels"""
        from prometheus_client import Counter
        
        counter = Counter('test_counter_hc', 'Test counter', ['id', 'type'])
        
        call_count = [0]
        
        def increment():
            # Rotate through 100 different label combinations
            idx = call_count[0] % 100
            counter.labels(id=f'id_{idx}', type=f'type_{idx % 5}').inc()
            call_count[0] += 1
        
        result = benchmark_operation(increment, iterations=1000)
        
        print(f"\nCounter High Cardinality:")
        print(f"  Mean: {result.mean_us:.2f} μs")
        print(f"  Median: {result.median_us:.2f} μs")
        print(f"  P95: {result.p95_us:.2f} μs")
        
        # Should still be reasonable (< 50 μs)
        assert result.mean_us < 50


class TestHistogramPerformance:
    """Benchmark histogram metric performance"""
    
    def test_histogram_observation_baseline(self):
        """Baseline histogram observation performance"""
        from prometheus_client import Histogram
        
        histogram = Histogram('test_histogram', 'Test histogram')
        
        call_count = [0]
        
        def observe():
            value = (call_count[0] % 10) * 0.1
            histogram.observe(value)
            call_count[0] += 1
        
        result = benchmark_operation(observe, iterations=1000)
        
        print(f"\nHistogram Observation:")
        print(f"  Mean: {result.mean_us:.2f} μs")
        print(f"  Median: {result.median_us:.2f} μs")
        print(f"  P95: {result.p95_us:.2f} μs")
        
        # Should be reasonably fast (< 30 μs)
        assert result.mean_us < 30
    
    def test_histogram_with_labels(self):
        """Histogram observation with labels"""
        from prometheus_client import Histogram
        
        histogram = Histogram('test_histogram_labels', 'Test histogram', ['operation'])
        
        call_count = [0]
        
        def observe():
            value = (call_count[0] % 10) * 0.1
            histogram.labels(operation='store').observe(value)
            call_count[0] += 1
        
        result = benchmark_operation(observe, iterations=1000)
        
        print(f"\nHistogram with Labels:")
        print(f"  Mean: {result.mean_us:.2f} μs")
        print(f"  Median: {result.median_us:.2f} μs")
        print(f"  P95: {result.p95_us:.2f} μs")
        
        # With labels, slightly slower but still < 50 μs
        assert result.mean_us < 50


class TestGaugePerformance:
    """Benchmark gauge metric performance"""
    
    def test_gauge_set_baseline(self):
        """Baseline gauge set performance"""
        from prometheus_client import Gauge
        
        gauge = Gauge('test_gauge', 'Test gauge')
        
        call_count = [0]
        
        def set_value():
            value = (call_count[0] % 100) / 100.0
            gauge.set(value)
            call_count[0] += 1
        
        result = benchmark_operation(set_value, iterations=1000)
        
        print(f"\nGauge Set:")
        print(f"  Mean: {result.mean_us:.2f} μs")
        print(f"  Median: {result.median_us:.2f} μs")
        print(f"  P95: {result.p95_us:.2f} μs")
        
        # Should be fast (< 10 μs)
        assert result.mean_us < 10
    
    def test_gauge_with_labels(self):
        """Gauge set with labels"""
        from prometheus_client import Gauge
        
        gauge = Gauge('test_gauge_labels', 'Test gauge', ['tier'])
        
        call_count = [0]
        
        def set_value():
            value = (call_count[0] % 100) / 100.0
            tier = f'TIER_{(call_count[0] % 8) + 1}'
            gauge.labels(tier=tier).set(value)
            call_count[0] += 1
        
        result = benchmark_operation(set_value, iterations=1000)
        
        print(f"\nGauge with Labels:")
        print(f"  Mean: {result.mean_us:.2f} μs")
        print(f"  Median: {result.median_us:.2f} μs")
        print(f"  P95: {result.p95_us:.2f} μs")
        
        # With labels < 20 μs
        assert result.mean_us < 20


class TestDecoratorOverhead:
    """Benchmark decorator overhead"""
    
    @pytest.mark.asyncio
    async def test_storage_operation_decorator_overhead(self):
        """Measure track_storage_operation decorator overhead"""
        from sigmavault.monitoring.metrics import track_storage_operation
        
        @track_storage_operation('store')
        async def mock_storage_op():
            """Mock storage operation"""
            await asyncio.sleep(0.001)  # 1ms operation
            return {'status': 'success', 'size_bytes': 1024, 'cost_usd': Decimal('0.001')}
        
        times = []
        for _ in range(100):
            start = time.perf_counter()
            await mock_storage_op()
            duration = time.perf_counter() - start
            times.append(duration)
        
        mean_us = statistics.mean(times) * 1_000_000
        p95_us = sorted(times)[int(len(times) * 0.95)] * 1_000_000
        
        print(f"\nStorage Decorator Overhead:")
        print(f"  Mean: {mean_us:.2f} μs")
        print(f"  P95: {p95_us:.2f} μs")
        
        # Decorator overhead should be < 150 μs (requirement)
        # Total time ~1ms + decorator overhead should be < 1.15ms
        decorator_overhead = mean_us - 1000  # Subtract base operation
        assert decorator_overhead < 150 or mean_us < 1150
    
    @pytest.mark.asyncio
    async def test_agent_task_decorator_overhead(self):
        """Measure track_agent_task decorator overhead"""
        from agents.monitoring.metrics import track_agent_task
        
        @track_agent_task('APEX-001', 'APEX', 'analysis')
        async def mock_agent_task():
            """Mock agent task"""
            await asyncio.sleep(0.001)  # 1ms task
            return {'result': 'complete', 'success': True}
        
        times = []
        for _ in range(100):
            start = time.perf_counter()
            await mock_agent_task()
            duration = time.perf_counter() - start
            times.append(duration)
        
        mean_us = statistics.mean(times) * 1_000_000
        p95_us = sorted(times)[int(len(times) * 0.95)] * 1_000_000
        
        print(f"\nAgent Task Decorator Overhead:")
        print(f"  Mean: {mean_us:.2f} μs")
        print(f"  P95: {p95_us:.2f} μs")
        
        # Decorator overhead should be < 150 μs (requirement)
        decorator_overhead = mean_us - 1000  # Subtract base operation
        assert decorator_overhead < 150 or mean_us < 1150


class TestContextManagerOverhead:
    """Benchmark context manager overhead"""
    
    def test_storage_context_overhead(self):
        """Measure StorageContext overhead"""
        from sigmavault.monitoring.metrics import StorageContext
        
        def with_context():
            with StorageContext('store', Decimal('0.001')):
                pass
        
        result = benchmark_operation(with_context, iterations=1000)
        
        print(f"\nStorageContext Manager:")
        print(f"  Mean: {result.mean_us:.2f} μs")
        print(f"  Median: {result.median_us:.2f} μs")
        print(f"  P95: {result.p95_us:.2f} μs")
        
        # Context manager overhead should be < 80 μs (requirement)
        assert result.mean_us < 80
    
    def test_storage_context_with_operations(self):
        """StorageContext with size/cost tracking"""
        from sigmavault.monitoring.metrics import StorageContext
        
        def with_context_and_tracking():
            with StorageContext('store', Decimal('0.001')) as ctx:
                ctx.set_size(1024 * 1024)
                ctx.set_cost(Decimal('0.001'))
        
        result = benchmark_operation(with_context_and_tracking, iterations=1000)
        
        print(f"\nStorageContext with Tracking:")
        print(f"  Mean: {result.mean_us:.2f} μs")
        print(f"  Median: {result.median_us:.2f} μs")
        print(f"  P95: {result.p95_us:.2f} μs")
        
        # Even with tracking, should stay reasonable (< 100 μs)
        assert result.mean_us < 100


class TestConcurrentMetricsPerformance:
    """Benchmark concurrent metrics operations"""
    
    def test_concurrent_counter_increments(self):
        """Performance of concurrent counter updates"""
        import threading
        
        from prometheus_client import Counter
        counter = Counter('concurrent_test', 'Test counter')
        
        def increment_many():
            for _ in range(100):
                counter.inc()
        
        start = time.perf_counter()
        
        threads = [threading.Thread(target=increment_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        duration = time.perf_counter() - start
        total_ops = 10 * 100
        mean_us_per_op = (duration / total_ops) * 1_000_000
        
        print(f"\nConcurrent Counter Increments (10 threads × 100 ops):")
        print(f"  Total: {duration:.3f}s")
        print(f"  Mean per op: {mean_us_per_op:.2f} μs")
        
        # Should handle concurrent access efficiently
        assert duration < 1.0  # Should complete within 1 second
    
    def test_concurrent_mixed_operations(self):
        """Performance of concurrent mixed metric operations"""
        import threading
        
        from prometheus_client import Counter, Gauge, Histogram
        
        counter = Counter('mixed_counter', 'Test counter')
        gauge = Gauge('mixed_gauge', 'Test gauge')
        histogram = Histogram('mixed_histogram', 'Test histogram')
        
        def mixed_ops():
            for i in range(100):
                counter.inc()
                gauge.set(i / 100.0)
                histogram.observe((i % 10) * 0.1)
        
        start = time.perf_counter()
        
        threads = [threading.Thread(target=mixed_ops) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        duration = time.perf_counter() - start
        
        print(f"\nConcurrent Mixed Operations (5 threads):")
        print(f"  Total: {duration:.3f}s")
        
        # Should handle mixed concurrent access
        assert duration < 1.0


class TestMetricsMemoryEfficiency:
    """Benchmark memory efficiency"""
    
    def test_large_label_cardinality_memory(self):
        """Memory impact of large label cardinality"""
        from prometheus_client import Counter, REGISTRY, CollectorRegistry
        
        # Use separate registry to not pollute global
        registry = CollectorRegistry()
        counter = Counter('test_memory', 'Test counter', ['id'], registry=registry)
        
        # Generate metrics with 1000 label combinations
        for i in range(1000):
            counter.labels(id=f'id_{i}').inc()
        
        # Export and measure size
        from prometheus_client import generate_latest
        metrics = generate_latest(registry)
        
        size_kb = len(metrics) / 1024
        
        print(f"\nLarge Label Cardinality (1000 combinations):")
        print(f"  Export size: {size_kb:.2f} KB")
        
        # Should be reasonable size
        assert size_kb < 50  # 50KB threshold


class TestPrometheusExportPerformance:
    """Benchmark Prometheus format export"""
    
    def test_export_generation_speed(self):
        """Speed of generating Prometheus format"""
        from prometheus_client import Counter, Gauge, Histogram, REGISTRY
        
        # Pre-populate with many metrics
        for i in range(100):
            Counter(f'counter_{i}', f'Counter {i}', registry=REGISTRY).inc()
            Gauge(f'gauge_{i}', f'Gauge {i}', registry=REGISTRY).set(0.5)
            Histogram(f'histogram_{i}', f'Histogram {i}', registry=REGISTRY).observe(0.5)
        
        times = []
        for _ in range(100):
            start = time.perf_counter()
            from prometheus_client import generate_latest
            metrics = generate_latest(REGISTRY)
            duration = time.perf_counter() - start
            times.append(duration)
        
        mean_ms = statistics.mean(times) * 1000
        p95_ms = sorted(times)[int(len(times) * 0.95)] * 1000
        
        print(f"\nPrometheus Export Generation (300 metrics):")
        print(f"  Mean: {mean_ms:.2f} ms")
        print(f"  P95: {p95_ms:.2f} ms")
        
        # Should be fast (< 100ms)
        assert mean_ms < 100


class TestEndToEndPerformance:
    """End-to-end performance benchmarks"""
    
    @pytest.mark.asyncio
    async def test_storage_operation_complete_flow(self):
        """Complete storage operation flow performance"""
        from sigmavault.monitoring.metrics import track_storage_operation
        
        @track_storage_operation('store')
        async def complete_storage_op():
            """Simulate complete storage operation"""
            await asyncio.sleep(0.005)  # 5ms simulated I/O
            return {
                'status': 'success',
                'size_bytes': 10 * 1024 * 1024,
                'cost_usd': Decimal('0.05'),
                'storage_class': 'hot',
                'cost_center': 'prod'
            }
        
        times = []
        for _ in range(50):
            start = time.perf_counter()
            result = await complete_storage_op()
            duration = time.perf_counter() - start
            times.append(duration)
        
        mean_ms = statistics.mean(times) * 1000
        
        print(f"\nComplete Storage Operation Flow:")
        print(f"  Mean: {mean_ms:.2f} ms")
        print(f"  Base: ~5 ms, Overhead: ~{(mean_ms - 5):.2f} ms")
    
    @pytest.mark.asyncio
    async def test_agent_task_complete_flow(self):
        """Complete agent task flow performance"""
        from agents.monitoring.metrics import track_agent_task
        
        @track_agent_task('APEX-001', 'APEX', 'analysis')
        async def complete_agent_task():
            """Simulate complete agent task"""
            await asyncio.sleep(0.010)  # 10ms simulated processing
            return {
                'result': 'optimized',
                'success': True,
                'processing_time_ms': 10
            }
        
        times = []
        for _ in range(50):
            start = time.perf_counter()
            result = await complete_agent_task()
            duration = time.perf_counter() - start
            times.append(duration)
        
        mean_ms = statistics.mean(times) * 1000
        
        print(f"\nComplete Agent Task Flow:")
        print(f"  Mean: {mean_ms:.2f} ms")
        print(f"  Base: ~10 ms, Overhead: ~{(mean_ms - 10):.2f} ms")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '--tb=short'])
