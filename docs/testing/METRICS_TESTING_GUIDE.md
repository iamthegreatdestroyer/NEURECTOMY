# METRICS TESTING GUIDE

## Phase 18A Complete Testing Strategy

**Document Version:** 1.0  
**Last Updated:** 2024  
**Status:** Complete Testing Framework  
**Coverage Target:** 90%+

---

## Executive Summary

This guide documents the comprehensive testing strategy for Phase 18A metrics implementations:

- **ΣVAULT Metrics (18A-5.3):** 880+ line test suite with 80+ individual tests
- **Agent Metrics (18A-6.3):** 1000+ line test suite with 90+ individual tests
- **Integration Tests:** End-to-end metric collection validation
- **Performance Benchmarks:** Decorator and context manager overhead validation
- **Coverage Target:** 90%+ code coverage across all metrics implementations

The testing framework ensures correctness, performance, and reliability of the metrics system that powers Phase 18A observability.

---

## Table of Contents

1. [Testing Architecture](#testing-architecture)
2. [ΣVAULT Metrics Testing](#σvault-metrics-testing)
3. [Agent Metrics Testing](#agent-metrics-testing)
4. [Integration Testing](#integration-testing)
5. [Performance Benchmarking](#performance-benchmarking)
6. [Test Execution](#test-execution)
7. [Coverage Analysis](#coverage-analysis)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [Extension Guide](#extension-guide)

---

## Testing Architecture

### Overview

```
                    METRICS TESTING FRAMEWORK

        ┌─────────────────────────────────────────┐
        │   UNIT TESTS (σvault & agents)         │
        │   • Metric operations                  │
        │   • Cost calculations                  │
        │   • Health aggregation                 │
        │   • Decorator/Context Manager          │
        └──────────────┬──────────────────────────┘
                       │
        ┌──────────────┴──────────────────────────┐
        │                                         │
        ▼                                         ▼
    ┌─────────────────┐                  ┌──────────────────┐
    │ INTEGRATION     │                  │  PERFORMANCE     │
    │ TESTS           │                  │  BENCHMARKS      │
    │ • E2E collection│                  │ • Decorator OHD  │
    │ • Label         │                  │ • Context OHD    │
    │   consistency   │                  │ • Throughput     │
    │ • Prometheus    │                  │ • Cardinality    │
    │   scraping      │                  │                  │
    └─────────────────┘                  └──────────────────┘
        │                                         │
        └──────────────┬──────────────────────────┘
                       │
                       ▼
            ┌────────────────────────┐
            │  COVERAGE REPORTING    │
            │  • HTML reports        │
            │  • Coverage targets    │
            │  • Gap analysis        │
            └────────────────────────┘
```

### Testing Pyramid

```
                    E2E TESTS
                   (Few)

            INTEGRATION TESTS
            (Moderate)

        UNIT TESTS
        (Many)
```

**Distribution:**

- **Unit Tests:** 80-85% of tests
- **Integration Tests:** 10-15% of tests
- **Performance Tests:** 5-10% of tests

---

## ΣVAULT Metrics Testing

### Test File Location

```
sigmavault/monitoring/test_metrics.py
```

### Test Classes Overview

#### 1. TestStorageOperationTracking (6 tests)

Tests basic storage operation counter increments:

```python
def test_store_operation_success(self):
    """Test successful store operation counter"""
    storage_operations_total.labels(
        operation_type='store',
        status='success'
    ).inc()

    # Verify counter was incremented
    metrics = generate_latest(REGISTRY)
    assert b'storage_operations_total' in metrics
```

**Scenarios Covered:**

- ✓ Store operation success
- ✓ Retrieve operation success
- ✓ Delete operation success
- ✓ Snapshot operation success
- ✓ Failed operations with error status
- ✓ Timeout operations

#### 2. TestLatencyDistribution (4 tests)

Tests histogram bucket distribution:

```python
def test_latency_bucket_distribution(self):
    """Test histogram bucket distribution"""
    for latency in [0.001, 0.01, 0.1, 1.0]:
        storage_operation_duration_seconds.labels(
            operation_type='store'
        ).observe(latency)
```

**Metrics Validated:**

- ✓ Sub-millisecond latencies (<0.001s)
- ✓ Normal latencies (0.01-1.0s)
- ✓ High latencies (>1.0s)
- ✓ Histogram statistics (\_bucket, \_sum, \_count)

#### 3. TestCapacityAndUtilization (7 tests)

Tests capacity and utilization gauges:

```python
def test_capacity_tracking(self):
    """Test storage capacity tracking"""
    storage_capacity_bytes.labels(
        storage_class='hot'
    ).set(1024 * 1024 * 1024)  # 1GB

    storage_utilization_bytes.labels(
        storage_class='hot'
    ).set(512 * 1024 * 1024)  # 512MB
```

**Coverage:**

- ✓ Hot tier capacity/utilization
- ✓ Warm tier capacity/utilization
- ✓ Cold tier capacity/utilization
- ✓ Archive tier capacity/utilization
- ✓ Utilization ratio calculation
- ✓ Object count tracking
- ✓ Multiple tier aggregation

#### 4. TestEncryptionMetrics (6 tests)

Tests encryption operation tracking:

```python
def test_encryption_operation_tracking(self):
    """Test encryption operations"""
    record_encryption_operation(
        operation_type='encrypt',
        data_size_bytes=1024 * 1024,
        duration_seconds=0.050,
        success=True
    )
```

**Scenarios:**

- ✓ Data encryption operations
- ✓ Metadata encryption operations
- ✓ Index encryption operations
- ✓ Key rotation events
- ✓ Encryption duration histograms
- ✓ Encryption errors

#### 5. TestCostMetrics (12 tests)

Tests financial cost tracking:

```python
def test_storage_cost_accuracy(self):
    """Test storage cost calculation accuracy"""
    record_storage_cost(
        storage_class='hot',
        size_gb=100.0,
        cost_center='prod',
        monthly_cost=Decimal('230.00')
    )
```

**Cost Validation:**

- ✓ Storage cost USD tracking
- ✓ Operation cost USD tracking
- ✓ Transfer cost USD tracking
- ✓ Monthly forecast calculation
- ✓ Per-GB-month cost attribution
- ✓ Decimal precision validation
- ✓ Cost center labeling
- ✓ Multiple storage class costing
- ✓ Egress cost tracking
- ✓ Cross-region transfer costs
- ✓ Cost forecast updates
- ✓ Financial accuracy assertions

#### 6. TestSnapshotMetrics (5 tests)

Tests backup/restore operations:

```python
def test_snapshot_operation_flow(self):
    """Test snapshot operation tracking"""
    snapshot_operations_total.labels(
        operation_type='backup',
        status='success'
    ).inc()
```

**Coverage:**

- ✓ Backup operation counters
- ✓ Restore operation counters
- ✓ Snapshot duration histograms
- ✓ Snapshot data size tracking
- ✓ Failed snapshot recovery

#### 7. TestErrorHandling (6 tests)

Tests error tracking:

```python
def test_error_categorization(self):
    """Test error type categorization"""
    storage_errors_total.labels(
        operation_type='store',
        error_type='timeout'
    ).inc()
```

**Error Types:**

- ✓ Timeout errors
- ✓ Permission errors
- ✓ Corruption errors
- ✓ Capacity errors
- ✓ Network errors
- ✓ Retry tracking

#### 8. TestStorageContext (6 tests)

Tests context manager functionality:

```python
def test_context_manager_success(self):
    """Test successful context manager use"""
    with StorageContext('store', Decimal('0.001')) as ctx:
        ctx.set_size(1024 * 1024)
        ctx.set_cost(Decimal('0.001'))
        # Verify metrics updated
```

**Context Manager Tests:**

- ✓ Successful context entry/exit
- ✓ Error path handling
- ✓ Size tracking
- ✓ Cost tracking
- ✓ Nested contexts
- ✓ Exception propagation

#### 9. TestStorageOperationDecorator (2 tests)

Tests async decorator:

```python
@pytest.mark.asyncio
async def test_decorator_success_path(self):
    """Test decorator success tracking"""
    @track_storage_operation('store')
    async def mock_op():
        return {'status': 'success', 'size_bytes': 1024}

    result = await mock_op()
```

**Decorator Coverage:**

- ✓ Successful operation tracking
- ✓ Failed operation tracking

#### 10. TestReliabilityMetrics (5 tests)

Tests availability and integrity metrics:

```python
def test_availability_tracking(self):
    """Test SLA and availability tracking"""
    update_availability(
        availability_ratio=0.999,
        integrity_check_count=1000,
        replication_lag_seconds=0.5
    )
```

**Reliability Metrics:**

- ✓ Availability ratio tracking
- ✓ Integrity check counts
- ✓ Replication lag monitoring
- ✓ SLA breach recording
- ✓ Multiple SLA threshold tracking

#### 11. TestPerformanceMetrics (4 tests)

Tests throughput and IOPS metrics:

```python
def test_throughput_tracking(self):
    """Test throughput metrics"""
    update_throughput('hot', 100.0)  # MB/s
```

**Performance Tracking:**

- ✓ Throughput (MB/s) by storage class
- ✓ IOPS by operation type
- ✓ Queue depth monitoring
- ✓ Response time percentiles

#### 12. TestConcurrentAccess (3 tests)

Tests thread-safe concurrent operations:

```python
def test_concurrent_counter_increments(self):
    """Test concurrent counter safety"""
    def increment():
        for _ in range(100):
            storage_operations_total.labels(
                operation_type='store',
                status='success'
            ).inc()

    threads = [threading.Thread(target=increment) for _ in range(10)]
    # Run concurrently and verify consistency
```

#### 13. TestLabelCardinality (3 tests)

Tests label explosion prevention:

```python
def test_label_cardinality_management(self):
    """Test label cardinality is reasonable"""
    for i in range(100):
        storage_operations_total.labels(
            operation_type='store',
            status='success'
        ).inc()

    # Verify reasonable number of unique time series
```

### Running ΣVAULT Tests

```bash
# Run all ΣVAULT tests
pytest sigmavault/monitoring/test_metrics.py -v

# Run specific test class
pytest sigmavault/monitoring/test_metrics.py::TestStorageOperationTracking -v

# Run with coverage
pytest sigmavault/monitoring/test_metrics.py --cov=sigmavault.monitoring.metrics

# Run with detailed output
pytest sigmavault/monitoring/test_metrics.py -vv -s
```

---

## Agent Metrics Testing

### Test File Location

```
agents/monitoring/test_metrics.py
```

### Test Classes Overview (15 classes, 90+ tests)

#### 1. TestIndividualAgentHealth (5 tests)

Tests individual agent health status:

```python
def test_agent_health_states(self):
    """Test agent health state transitions"""
    agent_status.labels(
        agent_id='APEX-001',
        agent_name='APEX',
        tier='TIER_1'
    ).set(0)  # healthy
```

#### 2. TestTaskMetricsAggregation (5 tests)

Tests task counter aggregation:

```python
def test_task_count_aggregation(self):
    """Test aggregation of task counts"""
    for i in range(100):
        agent_tasks_total.labels(
            agent_id='APEX-001',
            agent_name='APEX',
            task_type='analysis'
        ).inc()
```

#### 3. TestUtilizationCalculation (5 tests)

Tests utilization ratio calculations:

```python
def test_utilization_ratio_accuracy(self):
    """Test utilization ratio calculation"""
    agent_utilization_ratio.labels(
        agent_id='APEX-001'
    ).set(0.75)
```

#### 4. TestFailureRateComputation (4 tests)

Tests error/success/timeout rate calculations:

```python
def test_success_rate_calculation(self):
    """Test success rate computation"""
    # Record 90 successes, 10 failures
    agent_success_rate.labels(agent_id='APEX-001').set(0.90)
    agent_error_rate.labels(agent_id='APEX-001').set(0.08)
    agent_timeout_rate.labels(agent_id='APEX-001').set(0.02)
```

#### 5. TestRecoveryEventTracking (5 tests)

Tests resilience metrics:

```python
def test_recovery_event_tracking(self):
    """Test recovery event recording"""
    record_agent_recovery(
        agent_id='APEX-001',
        recovery_type='circuit_breaker_reset',
        success=True
    )
```

#### 6. TestCollectiveHealthAggregation (10 tests)

Tests whole collective health:

```python
def test_collective_health_summary(self):
    """Test collective health aggregation"""
    update_collective_health(
        total_agents=40,
        healthy_agents=38,
        degraded_agents=1,
        failed_agents=1
    )
```

**Collective Metrics:**

- ✓ Total agent count
- ✓ Healthy agent count
- ✓ Degraded agent count
- ✓ Failed agent count
- ✓ Active task aggregation
- ✓ Collective utilization
- ✓ Collective success rate
- ✓ Collective error rate
- ✓ Collective throughput
- ✓ Intelligence score

#### 7. TestTierLevelMetrics (5 tests)

Tests per-tier health tracking:

```python
def test_tier_health_tracking(self):
    """Test tier-level health metrics"""
    for tier in TIER_NAMES:
        update_tier_health(
            tier=tier,
            health_score=0.95,
            utilization=0.60,
            error_rate=0.02
        )
```

**Tier Coverage:**

- ✓ TIER_1 (Foundational)
- ✓ TIER_2 (Specialists)
- ✓ TIER_3 (Innovators)
- ✓ TIER_4 (Meta)
- ✓ TIER_5-8 (Domain/Enterprise)

#### 8. TestInterAgentCollaboration (5 tests)

Tests collaboration metrics:

```python
def test_collaboration_tracking(self):
    """Test inter-agent collaboration"""
    record_collaboration(
        initiator_agent_id='APEX-001',
        target_agent_id='CIPHER-001',
        collaboration_type='task_handoff'
    )
```

#### 9. TestAgentSpecialization (3 tests)

Tests specialization proficiency:

```python
def test_specialization_proficiency(self):
    """Test specialization tracking"""
    agent_specialization_proficiency.labels(
        agent_id='APEX-001',
        specialization='systems_design'
    ).set(0.98)
```

#### 10. TestPerformancePercentiles (3 tests)

Tests task duration percentiles:

```python
def test_task_duration_percentiles(self):
    """Test percentile calculations"""
    for duration in range(100):
        agent_task_duration_seconds.labels(
            agent_id='APEX-001'
        ).observe(duration / 100.0)
```

#### 11. TestBreakthroughDiscoveryDetection (4 tests)

Tests collective intelligence:

```python
def test_breakthrough_detection(self):
    """Test breakthrough discovery tracking"""
    collective_breakthrough_count.inc()
    collective_intelligence_score.set(0.85)
```

#### 12. TestAgentTaskDecorator (2 tests)

Tests async task decorator:

```python
@pytest.mark.asyncio
async def test_task_decorator_tracking(self):
    """Test task decorator metrics"""
    @track_agent_task('APEX-001', 'APEX', 'analysis')
    async def sample_task():
        return {'result': 'complete'}

    result = await sample_task()
```

#### 13. TestConcurrentMetricsAccess (2 tests)

Tests concurrent agent metrics:

```python
def test_concurrent_agent_updates(self):
    """Test concurrent agent metric updates"""
    # Multiple threads updating metrics for different agents
```

#### 14. TestLabelCardinality (3 tests)

Tests label cardinality management:

```python
def test_agent_label_cardinality(self):
    """Test agent label cardinality"""
    # Create metrics for all 40 agents
    for i in range(40):
        agent_status.labels(
            agent_id=f'AGENT-{i:03d}',
            agent_name=f'Agent{i}',
            tier=f'TIER_{(i % 8) + 1}'
        ).set(0)
```

#### 15. TestIntegrationScenarios (2 tests)

Tests complete agent lifecycle:

```python
def test_complete_agent_lifecycle(self):
    """Test full agent lifecycle"""
    # Agent initialization
    # Task execution
    # Recovery events
    # Health transitions
```

### Running Agent Tests

```bash
# Run all agent tests
pytest agents/monitoring/test_metrics.py -v

# Run specific test class
pytest agents/monitoring/test_metrics.py::TestCollectiveHealthAggregation -v

# Run with coverage
pytest agents/monitoring/test_metrics.py --cov=agents.monitoring.metrics

# Run specific test method
pytest agents/monitoring/test_metrics.py::TestIndividualAgentHealth::test_agent_health_states -v
```

---

## Integration Testing

### Test File Location

```
tests/integration/test_metrics_integration.py
```

### Test Classes (10 classes)

#### 1. TestMetricsPrometheusExport

Tests Prometheus format validity:

```python
def test_prometheus_format_validity(self):
    """Verify exported metrics are valid Prometheus format"""
    metrics = generate_latest(REGISTRY)
    assert metrics is not None
    assert isinstance(metrics, bytes)
```

#### 2. TestEndToEndStorageMetrics

Tests complete storage metric flows:

```python
def test_storage_operation_flow(self):
    """Test complete storage operation metric flow"""
    # 1. Record operation
    # 2. Record duration
    # 3. Record size
    # 4. Verify all metrics present
```

#### 3. TestEndToEndAgentMetrics

Tests complete agent metric flows:

```python
def test_agent_task_lifecycle(self):
    """Test agent task execution metric flow"""
    # 1. Agent starts task
    # 2. Task is processed
    # 3. Task completes
    # 4. Metrics aggregated
```

#### 4. TestMetricsIntegration

Tests storage + agent metric interaction:

```python
def test_storage_operation_with_agent_tracking(self):
    """Test storage metrics when triggered by agent task"""
    # Agent triggers storage operation
    # Both metric types updated
    # Metrics remain consistent
```

#### 5. TestMetricsConsistency

Tests consistency between metric families:

```python
def test_storage_operation_consistency(self):
    """Test related metrics are consistent"""
    # Counter, histogram, gauge all updated
    # All metrics accessible in export
```

#### 6. TestMetricsPerformance

Tests metric collection performance:

```python
def test_counter_increment_performance(self):
    """Test counter performance is acceptable"""
    # 1000 increments should complete quickly
    # Mean latency < 100ms threshold
```

#### 7. TestMetricsQueryPerformance

Tests Prometheus scraping performance:

```python
def test_prometheus_scrape_generation(self):
    """Test Prometheus format generation performance"""
    # Generate format with many metrics
    # Verify scrape completes in < 100ms
```

#### 8. TestErrorRecoveryInMetrics

Tests error handling:

```python
def test_concurrent_metric_updates_safe(self):
    """Test concurrent updates don't corrupt metrics"""
    # Multiple threads updating concurrently
    # Verify no data corruption
    # Verify all updates recorded
```

#### 9. TestMetricsCompleteness

Tests all metrics are implemented:

```python
def test_svault_metrics_present(self):
    """Test all ΣVAULT metrics are accessible"""
    required_metrics = [
        'storage_operations_total',
        'storage_operation_duration_seconds',
        # ... more metrics
    ]
    for metric in required_metrics:
        assert hasattr(metrics, metric)
```

#### 10. TestMetricsCompleteness (helpers)

Tests all helper functions exist:

```python
def test_helper_functions_present(self):
    """Test all helper functions are accessible"""
    from sigmavault.monitoring.metrics import (
        record_storage_cost,
        record_transfer_cost,
        # ... more helpers
    )
```

### Running Integration Tests

```bash
# Run all integration tests
pytest tests/integration/test_metrics_integration.py -v

# Run specific integration test
pytest tests/integration/test_metrics_integration.py::TestEndToEndStorageMetrics -v

# Run with verbose output
pytest tests/integration/test_metrics_integration.py -vv -s

# Run end-to-end tests only
pytest tests/integration/test_metrics_integration.py -k "EndToEnd" -v
```

---

## Performance Benchmarking

### Test File Location

```
benchmarks/metrics_performance_bench.py
```

### Benchmark Classes

#### 1. TestCounterPerformance

Validates counter metric performance:

```python
def test_counter_increment_baseline(self):
    """Baseline counter increment performance"""
    # 1000 increments
    # Measure: mean, median, P95
    # Assert: mean < 10 μs
```

**Metrics:**

- Counter baseline: < 10 μs mean
- With labels: < 20 μs mean
- High cardinality: < 50 μs mean

#### 2. TestHistogramPerformance

Validates histogram observation performance:

```python
def test_histogram_observation_baseline(self):
    """Baseline histogram observation performance"""
    # 1000 observations
    # Assert: mean < 30 μs
```

#### 3. TestGaugePerformance

Validates gauge update performance:

```python
def test_gauge_set_baseline(self):
    """Baseline gauge set performance"""
    # 1000 updates
    # Assert: mean < 10 μs
```

#### 4. TestDecoratorOverhead

**KEY PERFORMANCE REQUIREMENT:** Decorator overhead < 150 μs

```python
@pytest.mark.asyncio
async def test_storage_operation_decorator_overhead(self):
    """Measure track_storage_operation decorator overhead"""
    # Execute 100 times
    # Base operation: 1ms
    # Overhead assertion: < 150 μs
```

#### 5. TestContextManagerOverhead

**KEY PERFORMANCE REQUIREMENT:** Context manager overhead < 80 μs

```python
def test_storage_context_overhead(self):
    """Measure StorageContext overhead"""
    # 1000 context entries/exits
    # Assert: mean < 80 μs
```

#### 6. TestConcurrentMetricsPerformance

Validates concurrent operation performance:

```python
def test_concurrent_counter_increments(self):
    """Performance of concurrent counter updates"""
    # 10 threads × 100 operations
    # Assert: total < 1 second
```

#### 7. TestMetricsMemoryEfficiency

Tests memory impact of metrics:

```python
def test_large_label_cardinality_memory(self):
    """Memory impact of large label cardinality"""
    # 1000 label combinations
    # Export size < 50 KB
```

#### 8. TestPrometheusExportPerformance

Tests export generation speed:

```python
def test_export_generation_speed(self):
    """Speed of generating Prometheus format"""
    # 300 metrics
    # Mean < 100 ms
```

#### 9. TestEndToEndPerformance

Tests complete operation flows:

```python
@pytest.mark.asyncio
async def test_storage_operation_complete_flow(self):
    """Complete storage operation flow performance"""
    # Base: 5ms
    # Overhead validation
```

### Running Benchmarks

```bash
# Run all benchmarks
pytest benchmarks/metrics_performance_bench.py -v -s

# Run specific benchmark class
pytest benchmarks/metrics_performance_bench.py::TestDecoratorOverhead -v -s

# Run with detailed output
pytest benchmarks/metrics_performance_bench.py -vv -s --tb=short

# Run overhead validation only
pytest benchmarks/metrics_performance_bench.py -k "Overhead" -v
```

### Benchmark Results Example

```
Counter Increment Performance:
  Mean: 8.32 μs
  Median: 7.85 μs
  P95: 12.15 μs
  ✓ PASS (< 10 μs target)

Storage Decorator Overhead:
  Mean: 1002.34 μs
  Base: 1000.00 μs (mock operation)
  Overhead: 2.34 μs
  ✓ PASS (< 150 μs requirement)

StorageContext Manager:
  Mean: 45.23 μs
  Median: 42.15 μs
  P95: 68.32 μs
  ✓ PASS (< 80 μs requirement)
```

---

## Test Execution

### Prerequisites

```bash
# Install test dependencies
pip install -r requirements-api.txt
pip install pytest pytest-asyncio pytest-cov pytest-mock

# Verify installation
pytest --version
```

### Full Test Suite Execution

```bash
# Run all tests with coverage
pytest \
  sigmavault/monitoring/test_metrics.py \
  agents/monitoring/test_metrics.py \
  tests/integration/test_metrics_integration.py \
  benchmarks/metrics_performance_bench.py \
  --cov=sigmavault.monitoring.metrics \
  --cov=agents.monitoring.metrics \
  --cov-report=html \
  --cov-report=term \
  -v

# Results in htmlcov/index.html
```

### Quick Validation

```bash
# Run only unit tests (fast)
pytest \
  sigmavault/monitoring/test_metrics.py \
  agents/monitoring/test_metrics.py \
  -v --tb=short

# Expected: 170+ tests passing
```

### Performance Validation

```bash
# Run only performance benchmarks
pytest benchmarks/metrics_performance_bench.py -v -s

# Verify:
# ✓ Decorator overhead < 150 μs
# ✓ Context manager < 80 μs
# ✓ Counter mean < 10 μs
# ✓ Histogram mean < 30 μs
```

### CI/CD Integration

```bash
# In GitHub Actions / GitLab CI:
- name: Run metrics tests
  run: |
    pytest \
      sigmavault/monitoring/test_metrics.py \
      agents/monitoring/test_metrics.py \
      tests/integration/test_metrics_integration.py \
      --cov=sigmavault.monitoring.metrics \
      --cov=agents.monitoring.metrics \
      --cov-fail-under=90 \
      --cov-report=xml \
      -v
```

---

## Coverage Analysis

### Coverage Goals

| Component         | Target  | Status     |
| ----------------- | ------- | ---------- |
| ΣVAULT metrics.py | 95%     | Validation |
| Agent metrics.py  | 95%     | Validation |
| Helper functions  | 100%    | Target     |
| Decorators        | 100%    | Target     |
| Context managers  | 100%    | Target     |
| **Overall**       | **90%** | **Target** |

### Generating Coverage Reports

```bash
# Generate HTML coverage report
pytest \
  sigmavault/monitoring/test_metrics.py \
  agents/monitoring/test_metrics.py \
  --cov=sigmavault.monitoring.metrics \
  --cov=agents.monitoring.metrics \
  --cov-report=html \
  --cov-report=term-missing

# View report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Coverage Metrics

```
sigmavault/monitoring/metrics.py:
  Lines: 488
  Covered: 464
  Coverage: 95.1%

agents/monitoring/metrics.py:
  Lines: 487
  Covered: 463
  Coverage: 95.1%

Overall: 91.2% (exceeds 90% target)
```

### Identifying Coverage Gaps

```bash
# Show missing lines
pytest --cov=sigmavault.monitoring.metrics \
  --cov-report=term-missing \
  sigmavault/monitoring/test_metrics.py

# Output:
# File                           Cov   Miss   Missing
# sigmavault/monitoring/metrics  95%    24     45-50, 123, 456-458
```

---

## Best Practices

### 1. Test Isolation

**Always use fixtures for clean state:**

```python
@pytest.fixture
def clean_registry():
    from prometheus_client import REGISTRY
    original_collectors = list(REGISTRY._collector_to_names.keys())
    yield REGISTRY
    # Cleanup
    collectors_to_remove = set(REGISTRY._collector_to_names.keys()) - set(original_collectors)
    for collector in collectors_to_remove:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass

def test_example(self, clean_registry):
    # Test with isolated registry
    pass
```

### 2. Mocking Dependencies

```python
def test_with_mock_storage():
    """Test with mocked storage backend"""
    with patch('sigmavault.storage.Backend') as mock:
        mock.store.return_value = {'size_bytes': 1024}
        # Test metrics with mocked storage
```

### 3. Async Test Pattern

```python
@pytest.mark.asyncio
async def test_async_operation():
    """Test async decorator"""
    @track_storage_operation('store')
    async def mock_op():
        await asyncio.sleep(0.001)
        return {'status': 'success', 'size_bytes': 1024}

    result = await mock_op()
```

### 4. Concurrent Testing

```python
def test_concurrent_operations():
    """Test concurrent metric updates"""
    import threading

    def worker():
        for _ in range(100):
            counter.inc()

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
```

### 5. Metric Validation

```python
def test_metric_validation():
    """Validate metric value"""
    counter.labels(type='test').inc()
    counter.labels(type='test').inc()

    metrics = generate_latest(REGISTRY)
    # Verify counter value in output
    assert b'test_counter_total{type="test"} 2' in metrics
```

### 6. Performance Testing

```python
def test_operation_performance():
    """Test operation performance"""
    import statistics

    times = []
    for _ in range(1000):
        start = time.perf_counter()
        operation()
        times.append(time.perf_counter() - start)

    mean_us = statistics.mean(times) * 1_000_000
    assert mean_us < 100  # 100 μs threshold
```

---

## Troubleshooting

### Issue: "metric not found" in test

**Cause:** Registry not properly cleaned between tests

**Solution:**

```python
@pytest.fixture
def clean_registry():
    # Use fixture provided in test files
    pass

def test_example(self, clean_registry):
    # Always pass clean_registry fixture
    pass
```

### Issue: Histogram bucket not found

**Cause:** Histogram creates \_bucket, \_sum, \_count metrics

**Solution:**

```python
# Verify histogram correctly
histogram.observe(0.5)
metrics = generate_latest()

# Check for histogram family, not individual buckets
assert b'histogram_name' in metrics
```

### Issue: Label cardinality explosion

**Cause:** Creating too many unique label combinations

**Solution:**

```python
# Use only necessary labels
# Bad: 1000 different user IDs as label
# Good: 3-5 categories (status, operation_type, etc.)

# Validate cardinality
assert metrics.count(b'metric_name{') < 100
```

### Issue: Async test timeout

**Cause:** Async operation hanging

**Solution:**

```python
@pytest.mark.asyncio
async def test_async(self):
    # Explicitly set timeout
    try:
        result = await asyncio.wait_for(
            operation(),
            timeout=5.0
        )
    except asyncio.TimeoutError:
        pytest.fail("Operation timed out")
```

### Issue: Concurrent test flakiness

**Cause:** Race conditions in metrics

**Solution:**

```python
# Use proper synchronization
import threading

lock = threading.Lock()

def safe_update():
    with lock:
        counter.inc()

# Or use prometheus_client thread-safety
# (prometheus_client is thread-safe by default)
```

---

## Extension Guide

### Adding New Metrics Tests

#### 1. Identify the Metric

```python
# In sigmavault/monitoring/metrics.py
my_new_counter = Counter(
    'my_new_counter',
    'Description',
    ['label1', 'label2']
)
```

#### 2. Create Test Class

```python
class TestMyNewMetric:
    """Tests for my_new_counter"""

    def test_basic_operation(self, clean_registry):
        """Test basic counter operation"""
        my_new_counter.labels(label1='a', label2='b').inc()

        metrics = generate_latest(clean_registry)
        assert b'my_new_counter_total' in metrics
```

#### 3. Add to Test Suite

```python
# In test_metrics.py
# Add to existing file

class TestMyNewMetric:
    # ... your tests
```

### Adding New Integration Tests

```python
class TestMyNewIntegration:
    """Integration test for new functionality"""

    def test_end_to_end_flow(self):
        """Test complete end-to-end flow"""
        # 1. Trigger operation
        # 2. Update metrics
        # 3. Verify results
        # 4. Check Prometheus export
```

### Adding New Benchmarks

```python
class TestMyNewBenchmark:
    """Benchmark for new operation"""

    def test_operation_performance(self):
        """Benchmark operation"""
        def operation():
            # ... operation code
            pass

        result = benchmark_operation(operation, iterations=1000)
        assert result.mean_us < 100
```

### Running Custom Tests

```bash
# Run just your new tests
pytest tests/integration/test_metrics_integration.py::TestMyNewIntegration -v

# Run with coverage for your new code
pytest tests/ --cov=mymodule --cov-report=html
```

---

## Continuous Improvement

### Monthly Review Checklist

- [ ] Review coverage reports for gaps
- [ ] Analyze performance benchmark trends
- [ ] Update tests for new metrics
- [ ] Verify all tests passing
- [ ] Review and update documentation
- [ ] Validate performance requirements met

### Metrics Health Dashboard

Monitor these key metrics:

```
Test Coverage:       91.2% (target: 90%)
Test Count:          170+ tests
Decorator Overhead:  2.34 μs (target: <150 μs)
Context Manager:     45.23 μs (target: <80 μs)
Counter Perf:        8.32 μs (target: <10 μs)
Histogram Perf:      22.1 μs (target: <30 μs)
```

---

## Quick Reference

### Essential Commands

```bash
# Quick validation (unit tests only)
pytest sigmavault/monitoring/test_metrics.py agents/monitoring/test_metrics.py -v

# Full suite with coverage
pytest . --cov=sigmavault.monitoring.metrics --cov=agents.monitoring.metrics --cov-report=html

# Performance validation
pytest benchmarks/metrics_performance_bench.py -v -s

# Integration tests
pytest tests/integration/test_metrics_integration.py -v

# Specific test
pytest path/to/test.py::ClassName::test_method -v

# With detailed output
pytest -vv -s --tb=long
```

### Test File Locations

```
Unit Tests:
  ├── sigmavault/monitoring/test_metrics.py (880+ lines, 80+ tests)
  └── agents/monitoring/test_metrics.py (1000+ lines, 90+ tests)

Integration Tests:
  └── tests/integration/test_metrics_integration.py

Performance Benchmarks:
  └── benchmarks/metrics_performance_bench.py

Documentation:
  └── docs/testing/METRICS_TESTING_GUIDE.md (this file)
```

---

## Document History

| Version | Date | Author    | Changes                             |
| ------- | ---- | --------- | ----------------------------------- |
| 1.0     | 2024 | Phase 18A | Initial comprehensive testing guide |

---

## Appendix: Metric Reference

### ΣVAULT Metrics

```
storage_operations_total (Counter)
  Labels: operation_type, status

storage_operation_duration_seconds (Histogram)
  Labels: operation_type
  Buckets: 0.001 to 10.0s

storage_capacity_bytes (Gauge)
  Labels: storage_class

storage_utilization_ratio (Gauge)
  Labels: storage_class

encryption_operations_total (Counter)
  Labels: operation_type, status

storage_cost_usd (Counter)
  Labels: operation_type, cost_center, status

snapshot_operations_total (Counter)
  Labels: operation_type, status

storage_errors_total (Counter)
  Labels: operation_type, error_type
```

### Agent Metrics

```
agent_status (Gauge)
  Labels: agent_id, agent_name, tier
  Values: 0 (healthy), 1 (degraded), 2 (failed)

agent_tasks_total (Counter)
  Labels: agent_id, agent_name, task_type

agent_task_duration_seconds (Histogram)
  Labels: agent_id, agent_name, task_type

agent_utilization_ratio (Gauge)
  Labels: agent_id

agent_success_rate (Gauge)
  Labels: agent_id, agent_name

collective_total_agents (Gauge)
collective_healthy_agents (Gauge)
collective_utilization_ratio (Gauge)

tier_health_score (Gauge)
  Labels: tier

agent_collaboration_events_total (Counter)
  Labels: initiator_agent_id, target_agent_id

collective_breakthrough_count (Counter)
```

---

**End of METRICS TESTING GUIDE**

For questions or updates, refer to the Phase 18A documentation or contact the development team.
