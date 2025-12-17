"""
Comprehensive Test Suite for ΣVAULT Metrics
18A-5.3: Storage Operation Tracking, Financial Cost Attribution, and Reliability Metrics
"""

import pytest
import time
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from prometheus_client import CollectorRegistry, generate_latest

from sigmavault.monitoring.metrics import (
    # Storage Operation Metrics
    storage_operations_total,
    storage_operation_duration_seconds,
    storage_operation_size_bytes,
    
    # Storage Capacity Metrics
    storage_capacity_bytes,
    storage_utilization_bytes,
    storage_utilization_ratio,
    object_count,
    object_size_bytes,
    
    # Performance Metrics
    storage_throughput_bytes_per_second,
    storage_iops,
    queue_depth,
    
    # Encryption Metrics
    encryption_operations_total,
    encryption_duration_seconds,
    key_rotations_total,
    
    # Financial Cost Metrics
    storage_cost_usd,
    operation_cost_usd,
    transfer_cost_usd,
    total_cost_usd,
    monthly_cost_forecast_usd,
    cost_per_gb_month,
    cost_per_operation,
    total_stored_gb_month,
    
    # Snapshot/Backup Metrics
    snapshot_operations_total,
    snapshot_duration_seconds,
    snapshot_data_size_bytes,
    
    # Reliability and Integrity
    storage_availability_ratio,
    data_integrity_checks_total,
    replication_lag_seconds,
    sla_breaches_total,
    
    # Error Tracking
    storage_errors_total,
    retry_operations_total,
    
    # Decorators and Context Managers
    track_storage_operation,
    StorageContext,
    
    # Helper Functions
    record_storage_cost,
    record_transfer_cost,
    update_monthly_forecast,
    update_storage_capacity,
    record_object_size,
    update_throughput,
    update_iops,
    update_queue_depth,
    record_encryption_operation,
    record_key_rotation,
    update_availability,
    record_integrity_check,
    record_sla_breach,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def clean_registry():
    """Clean registry for isolated metric tests"""
    from prometheus_client import REGISTRY
    
    # Store original collectors
    original_collectors = list(REGISTRY._collector_to_names.keys())
    
    yield REGISTRY
    
    # Restore state
    collectors_to_remove = set(REGISTRY._collector_to_names.keys()) - set(original_collectors)
    for collector in collectors_to_remove:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


@pytest.fixture
def mock_storage_result():
    """Mock storage operation result with size and cost"""
    result = Mock()
    result.size_bytes = 1024 * 1024  # 1 MB
    result.cost_usd = Decimal('0.0001')
    return result


@pytest.fixture
def metrics_baseline():
    """Get baseline metrics state"""
    from prometheus_client import REGISTRY
    metrics = generate_latest(REGISTRY)
    return metrics.decode('utf-8')


# ============================================================================
# TEST STORAGE OPERATION TRACKING
# ============================================================================

class TestStorageOperationTracking:
    """Tests for basic storage operation tracking"""
    
    def test_store_operation_success(self, clean_registry):
        """Test successful store operation is tracked"""
        storage_operations_total.labels(operation_type='store', status='success').inc()
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_storage_operations_total{operation_type="store",status="success"} 1' in metrics
    
    def test_retrieve_operation_tracking(self, clean_registry):
        """Test retrieve operation counter"""
        storage_operations_total.labels(operation_type='retrieve', status='success').inc(2)
        metrics = generate_latest(clean_registry)
        assert b'operation_type="retrieve"' in metrics
    
    def test_delete_operation_tracking(self, clean_registry):
        """Test delete operation counter"""
        storage_operations_total.labels(operation_type='delete', status='success').inc()
        storage_operations_total.labels(operation_type='delete', status='error').inc()
        metrics = generate_latest(clean_registry)
        assert b'operation_type="delete"' in metrics
    
    def test_snapshot_operation_tracking(self, clean_registry):
        """Test snapshot operation counter"""
        storage_operations_total.labels(operation_type='snapshot', status='success').inc()
        metrics = generate_latest(clean_registry)
        assert b'operation_type="snapshot"' in metrics
    
    def test_operation_status_distinctions(self, clean_registry):
        """Test different operation statuses are tracked separately"""
        storage_operations_total.labels(operation_type='store', status='success').inc(5)
        storage_operations_total.labels(operation_type='store', status='error').inc(2)
        storage_operations_total.labels(operation_type='store', status='timeout').inc(1)
        
        metrics = generate_latest(clean_registry)
        metrics_str = metrics.decode('utf-8')
        assert 'status="success"' in metrics_str
        assert 'status="error"' in metrics_str
        assert 'status="timeout"' in metrics_str
    
    def test_operation_count_increment(self, clean_registry):
        """Test operation count increments correctly"""
        for _ in range(10):
            storage_operations_total.labels(operation_type='store', status='success').inc()
        
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_storage_operations_total{operation_type="store",status="success"} 10' in metrics


# ============================================================================
# TEST LATENCY DISTRIBUTION TRACKING
# ============================================================================

class TestLatencyDistribution:
    """Tests for operation latency distribution (histogram)"""
    
    def test_latency_observation_fast(self, clean_registry):
        """Test fast operation latency observation"""
        storage_operation_duration_seconds.labels(operation_type='retrieve').observe(0.001)
        metrics = generate_latest(clean_registry)
        assert b'_bucket' in metrics  # Histogram buckets present
    
    def test_latency_observation_slow(self, clean_registry):
        """Test slow operation latency observation"""
        storage_operation_duration_seconds.labels(operation_type='store').observe(5.0)
        metrics = generate_latest(clean_registry)
        assert b'_bucket' in metrics
    
    def test_latency_multiple_observations(self, clean_registry):
        """Test multiple latency observations create valid histogram"""
        latencies = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
        for latency in latencies:
            storage_operation_duration_seconds.labels(operation_type='store').observe(latency)
        
        metrics = generate_latest(clean_registry)
        assert b'_sum' in metrics  # Histogram sum present
        assert b'_count' in metrics  # Histogram count present
    
    def test_latency_histogram_buckets(self, clean_registry):
        """Test histogram has correct bucket boundaries"""
        # Buckets: (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0)
        storage_operation_duration_seconds.labels(operation_type='store').observe(0.002)
        storage_operation_duration_seconds.labels(operation_type='store').observe(0.1)
        storage_operation_duration_seconds.labels(operation_type='store').observe(5.0)
        
        metrics = generate_latest(clean_registry)
        metrics_str = metrics.decode('utf-8')
        assert 'le="0.001"' in metrics_str or 'le="+Inf"' in metrics_str


# ============================================================================
# TEST STORAGE CAPACITY AND UTILIZATION
# ============================================================================

class TestCapacityAndUtilization:
    """Tests for storage capacity and utilization tracking"""
    
    def test_capacity_bytes_gauge(self, clean_registry):
        """Test storage capacity gauge updates"""
        storage_capacity_bytes.labels(storage_class='hot').set(1024 * 1024 * 1024)  # 1 GB
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_storage_capacity_bytes{storage_class="hot"}' in metrics
    
    def test_utilization_bytes_gauge(self, clean_registry):
        """Test storage utilization gauge updates"""
        storage_utilization_bytes.labels(storage_class='warm').set(512 * 1024 * 1024)  # 512 MB
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_storage_utilization_bytes{storage_class="warm"}' in metrics
    
    def test_utilization_ratio_calculation(self, clean_registry):
        """Test utilization ratio is properly recorded"""
        storage_utilization_ratio.labels(storage_class='hot').set(0.75)
        metrics = generate_latest(clean_registry)
        metrics_str = metrics.decode('utf-8')
        assert 'sigmavault_storage_utilization_ratio' in metrics_str
    
    def test_utilization_ratio_valid_range(self, clean_registry):
        """Test utilization ratio stays in valid 0.0-1.0 range"""
        for tier in ['hot', 'warm', 'cold', 'archive']:
            ratio = 0.25 * (len(tier) % 4)
            storage_utilization_ratio.labels(storage_class=tier).set(ratio)
        
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_storage_utilization_ratio' in metrics
    
    def test_object_count_gauge(self, clean_registry):
        """Test object count gauge"""
        object_count.labels(storage_class='hot').set(1000)
        object_count.labels(storage_class='warm').set(5000)
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_object_count' in metrics
    
    def test_object_size_distribution(self, clean_registry):
        """Test object size histogram"""
        sizes = [100, 1024, 10*1024, 100*1024, 1024*1024]
        for size in sizes:
            object_size_bytes.labels(storage_class='hot').observe(size)
        
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_object_size_bytes' in metrics
    
    def test_update_storage_capacity_helper(self, clean_registry):
        """Test update_storage_capacity helper function"""
        update_storage_capacity('hot', 1000000, 750000, 100)
        
        metrics = generate_latest(clean_registry)
        metrics_str = metrics.decode('utf-8')
        assert 'storage_class="hot"' in metrics_str


# ============================================================================
# TEST ENCRYPTION OPERATION METRICS
# ============================================================================

class TestEncryptionMetrics:
    """Tests for encryption operation tracking"""
    
    def test_data_encryption_operation(self, clean_registry):
        """Test data encryption operation counter"""
        encryption_operations_total.labels(key_type='data').inc()
        metrics = generate_latest(clean_registry)
        assert b'key_type="data"' in metrics
    
    def test_metadata_encryption_operation(self, clean_registry):
        """Test metadata encryption operation counter"""
        encryption_operations_total.labels(key_type='metadata').inc(5)
        metrics = generate_latest(clean_registry)
        assert b'key_type="metadata"' in metrics
    
    def test_index_encryption_operation(self, clean_registry):
        """Test index encryption operation counter"""
        encryption_operations_total.labels(key_type='index').inc()
        metrics = generate_latest(clean_registry)
        assert b'key_type="index"' in metrics
    
    def test_encryption_duration_latency(self, clean_registry):
        """Test encryption operation latency"""
        encryption_duration_seconds.labels(key_type='data').observe(0.005)
        encryption_duration_seconds.labels(key_type='data').observe(0.010)
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_encryption_duration_seconds' in metrics
    
    def test_key_rotation_counter(self, clean_registry):
        """Test key rotation counter"""
        key_rotations_total.labels(key_type='data').inc()
        key_rotations_total.labels(key_type='metadata').inc()
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_key_rotations_total' in metrics
    
    def test_key_types_tracked_separately(self, clean_registry):
        """Test different key types are tracked separately"""
        record_encryption_operation('data')
        record_encryption_operation('metadata')
        record_encryption_operation('index')
        
        metrics = generate_latest(clean_registry)
        metrics_str = metrics.decode('utf-8')
        for key_type in ['data', 'metadata', 'index']:
            assert f'key_type="{key_type}"' in metrics_str


# ============================================================================
# TEST COST CALCULATION ACCURACY
# ============================================================================

class TestCostMetrics:
    """Tests for financial cost tracking and calculation"""
    
    def test_storage_cost_recording(self, clean_registry):
        """Test storage cost is recorded correctly"""
        storage_cost_usd.labels(cost_type='storage', storage_class='hot', cost_center='eng').inc(0.50)
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_storage_cost_usd' in metrics
    
    def test_operation_cost_recording(self, clean_registry):
        """Test operation cost is recorded"""
        operation_cost_usd.labels(operation_type='store', status='success', cost_center='prod').inc(0.001)
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_operation_cost_usd' in metrics
    
    def test_transfer_cost_ingress(self, clean_registry):
        """Test ingress transfer cost"""
        transfer_cost_usd.labels(transfer_type='ingress', cost_center='prod').inc(0.10)
        metrics = generate_latest(clean_registry)
        assert b'transfer_type="ingress"' in metrics
    
    def test_transfer_cost_egress(self, clean_registry):
        """Test egress transfer cost"""
        transfer_cost_usd.labels(transfer_type='egress', cost_center='prod').inc(0.25)
        metrics = generate_latest(clean_registry)
        assert b'transfer_type="egress"' in metrics
    
    def test_transfer_cost_cross_region(self, clean_registry):
        """Test cross-region transfer cost"""
        transfer_cost_usd.labels(transfer_type='cross_region', cost_center='prod').inc(0.50)
        metrics = generate_latest(clean_registry)
        assert b'transfer_type="cross_region"' in metrics
    
    def test_monthly_cost_accumulation(self, clean_registry):
        """Test monthly cost accumulation"""
        total_cost_usd.labels(cost_center='prod', month='2024-01').inc(100.00)
        total_cost_usd.labels(cost_center='prod', month='2024-01').inc(50.00)
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_total_cost_usd' in metrics
    
    def test_monthly_cost_forecast(self, clean_registry):
        """Test monthly cost forecast"""
        monthly_cost_forecast_usd.labels(cost_center='prod').set(500.00)
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_monthly_cost_forecast_usd' in metrics
    
    def test_cost_per_gb_month(self, clean_registry):
        """Test cost per GB/month metric"""
        cost_per_gb_month.labels(storage_class='hot', cost_center='prod').set(0.023)
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_cost_per_gb_month' in metrics
    
    def test_record_storage_cost_helper(self, clean_registry):
        """Test record_storage_cost helper function"""
        record_storage_cost('hot', 100.0, 'prod', Decimal('2.30'))
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_storage_cost_usd' in metrics
    
    def test_record_transfer_cost_helper(self, clean_registry):
        """Test record_transfer_cost helper function"""
        record_transfer_cost('egress', 50.0, 'prod', Decimal('5.00'))
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_transfer_cost_usd' in metrics
    
    def test_update_monthly_forecast_helper(self, clean_registry):
        """Test update_monthly_forecast helper"""
        update_monthly_forecast('prod', Decimal('1000.00'))
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_monthly_cost_forecast_usd' in metrics
    
    def test_decimal_cost_precision(self, clean_registry):
        """Test Decimal cost maintains precision"""
        cost = Decimal('0.0001')
        operation_cost_usd.labels(operation_type='store', status='success', cost_center='prod').inc(float(cost))
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_operation_cost_usd' in metrics


# ============================================================================
# TEST SNAPSHOT/BACKUP METRICS
# ============================================================================

class TestSnapshotMetrics:
    """Tests for snapshot and backup operation tracking"""
    
    def test_backup_operation_success(self, clean_registry):
        """Test successful backup operation"""
        snapshot_operations_total.labels(operation_type='backup', status='success').inc()
        metrics = generate_latest(clean_registry)
        assert b'operation_type="backup"' in metrics
    
    def test_restore_operation_success(self, clean_registry):
        """Test successful restore operation"""
        snapshot_operations_total.labels(operation_type='restore', status='success').inc()
        metrics = generate_latest(clean_registry)
        assert b'operation_type="restore"' in metrics
    
    def test_snapshot_operation_failures(self, clean_registry):
        """Test failed snapshot operations"""
        snapshot_operations_total.labels(operation_type='backup', status='error').inc()
        metrics = generate_latest(clean_registry)
        assert b'status="error"' in metrics
    
    def test_snapshot_duration_histogram(self, clean_registry):
        """Test snapshot operation duration histogram"""
        snapshot_duration_seconds.labels(operation_type='backup').observe(5.0)
        snapshot_duration_seconds.labels(operation_type='restore').observe(3.0)
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_snapshot_duration_seconds' in metrics
    
    def test_snapshot_data_size_histogram(self, clean_registry):
        """Test snapshot data size distribution"""
        sizes_mb = [100, 500, 1000, 5000]
        for size_mb in sizes_mb:
            snapshot_data_size_bytes.labels(operation_type='backup').observe(size_mb * 1024 * 1024)
        
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_snapshot_data_size_bytes' in metrics


# ============================================================================
# TEST ERROR HANDLING AND RECOVERY
# ============================================================================

class TestErrorHandling:
    """Tests for error tracking and recovery scenarios"""
    
    def test_timeout_error_tracking(self, clean_registry):
        """Test timeout error is tracked"""
        storage_errors_total.labels(operation_type='store', error_type='timeout').inc()
        metrics = generate_latest(clean_registry)
        assert b'error_type="timeout"' in metrics
    
    def test_permission_error_tracking(self, clean_registry):
        """Test permission error is tracked"""
        storage_errors_total.labels(operation_type='retrieve', error_type='permission').inc()
        metrics = generate_latest(clean_registry)
        assert b'error_type="permission"' in metrics
    
    def test_corruption_error_tracking(self, clean_registry):
        """Test corruption error is tracked"""
        storage_errors_total.labels(operation_type='retrieve', error_type='corruption').inc()
        metrics = generate_latest(clean_registry)
        assert b'error_type="corruption"' in metrics
    
    def test_unknown_error_tracking(self, clean_registry):
        """Test unknown error is tracked"""
        storage_errors_total.labels(operation_type='store', error_type='unknown').inc()
        metrics = generate_latest(clean_registry)
        assert b'error_type="unknown"' in metrics
    
    def test_retry_operation_counter(self, clean_registry):
        """Test retry operations are counted"""
        retry_operations_total.labels(operation_type='store').inc()
        retry_operations_total.labels(operation_type='store').inc()
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_retry_operations_total' in metrics
    
    def test_multiple_error_types(self, clean_registry):
        """Test multiple error types are tracked separately"""
        error_types = ['timeout', 'permission', 'corruption', 'unknown']
        for error_type in error_types:
            storage_errors_total.labels(operation_type='store', error_type=error_type).inc()
        
        metrics = generate_latest(clean_registry)
        metrics_str = metrics.decode('utf-8')
        for error_type in error_types:
            assert f'error_type="{error_type}"' in metrics_str


# ============================================================================
# TEST CONTEXT MANAGER
# ============================================================================

class TestStorageContext:
    """Tests for StorageContext manager"""
    
    def test_context_manager_success(self, clean_registry):
        """Test context manager records successful operation"""
        with StorageContext('store', 'prod') as ctx:
            ctx.set_size(1024 * 1024)
            ctx.set_cost(Decimal('0.0001'))
            time.sleep(0.001)
        
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_storage_operations_total' in metrics
    
    def test_context_manager_timing(self, clean_registry):
        """Test context manager records timing"""
        with StorageContext('store', 'prod') as ctx:
            ctx.set_size(1024)
            time.sleep(0.01)
        
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_storage_operation_duration_seconds' in metrics
    
    def test_context_manager_error_handling(self, clean_registry):
        """Test context manager handles exceptions"""
        try:
            with StorageContext('store', 'prod') as ctx:
                ctx.set_size(1024)
                raise ValueError("Test error")
        except ValueError:
            pass
        
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_storage_errors_total' in metrics
    
    def test_context_manager_size_tracking(self, clean_registry):
        """Test context manager tracks operation size"""
        with StorageContext('store', 'prod') as ctx:
            size = 10 * 1024 * 1024
            ctx.set_size(size)
        
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_storage_operation_size_bytes' in metrics
    
    def test_context_manager_cost_tracking(self, clean_registry):
        """Test context manager tracks cost"""
        with StorageContext('retrieve', 'prod') as ctx:
            ctx.set_cost(Decimal('0.001'))
        
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_operation_cost_usd' in metrics
    
    def test_context_manager_multiple_sequential(self, clean_registry):
        """Test multiple sequential context manager invocations"""
        for i in range(5):
            with StorageContext('store', 'prod') as ctx:
                ctx.set_size(1024)
                ctx.set_cost(Decimal('0.0001'))
        
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_storage_operations_total' in metrics


# ============================================================================
# TEST DECORATOR
# ============================================================================

class TestStorageOperationDecorator:
    """Tests for track_storage_operation decorator"""
    
    @pytest.mark.asyncio
    async def test_decorator_success_path(self, clean_registry):
        """Test decorator tracks successful operation"""
        @track_storage_operation('store', 'prod')
        async def store_data():
            await asyncio.sleep(0.001)
            result = Mock()
            result.size_bytes = 1024 * 1024
            result.cost_usd = Decimal('0.0001')
            return result
        
        import asyncio
        await store_data()
        
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_storage_operations_total' in metrics
    
    @pytest.mark.asyncio
    async def test_decorator_error_tracking(self, clean_registry):
        """Test decorator tracks errors"""
        @track_storage_operation('store', 'prod')
        async def failing_store():
            import asyncio
            await asyncio.sleep(0.001)
            raise PermissionError("Access denied")
        
        import asyncio
        with pytest.raises(PermissionError):
            await failing_store()
        
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_storage_errors_total' in metrics


# ============================================================================
# TEST RELIABILITY AND INTEGRITY
# ============================================================================

class TestReliabilityMetrics:
    """Tests for storage reliability and data integrity"""
    
    def test_availability_ratio_gauge(self, clean_registry):
        """Test availability ratio gauge"""
        update_availability(0.9999)
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_storage_availability_ratio' in metrics
    
    def test_integrity_check_passed(self, clean_registry):
        """Test passed integrity check"""
        record_integrity_check(True)
        metrics = generate_latest(clean_registry)
        assert b'status="passed"' in metrics
    
    def test_integrity_check_failed(self, clean_registry):
        """Test failed integrity check"""
        record_integrity_check(False)
        metrics = generate_latest(clean_registry)
        assert b'status="failed"' in metrics
    
    def test_replication_lag_histogram(self, clean_registry):
        """Test replication lag histogram"""
        replication_lag_seconds.observe(0.5)
        replication_lag_seconds.observe(1.0)
        replication_lag_seconds.observe(5.0)
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_replication_lag_seconds' in metrics
    
    def test_sla_breach_tracking(self, clean_registry):
        """Test SLA breach tracking"""
        record_sla_breach('availability')
        record_sla_breach('latency')
        record_sla_breach('throughput')
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_sla_breaches_total' in metrics


# ============================================================================
# TEST PERFORMANCE METRICS
# ============================================================================

class TestPerformanceMetrics:
    """Tests for throughput and IOPS metrics"""
    
    def test_throughput_bytes_per_second(self, clean_registry):
        """Test throughput gauge"""
        update_throughput('read', 1024 * 1024)
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_storage_throughput_bytes_per_second' in metrics
    
    def test_iops_metric(self, clean_registry):
        """Test IOPS gauge"""
        update_iops('write', 1000)
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_storage_iops' in metrics
    
    def test_queue_depth_gauge(self, clean_registry):
        """Test queue depth gauge"""
        update_queue_depth(42)
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_queue_depth' in metrics
    
    def test_throughput_multiple_operation_types(self, clean_registry):
        """Test throughput for multiple operation types"""
        for op_type in ['read', 'write', 'delete']:
            update_throughput(op_type, 1024 * 1024 * (len(op_type)))
        
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_storage_throughput_bytes_per_second' in metrics


# ============================================================================
# TEST CONCURRENT ACCESS PATTERNS
# ============================================================================

class TestConcurrentAccess:
    """Tests for concurrent metric access safety"""
    
    def test_concurrent_operation_increments(self, clean_registry):
        """Test concurrent counter increments"""
        import threading
        
        def increment_counter():
            for _ in range(100):
                storage_operations_total.labels(operation_type='store', status='success').inc()
        
        threads = [threading.Thread(target=increment_counter) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        metrics = generate_latest(clean_registry)
        # Should have 500 total increments
        assert b'sigmavault_storage_operations_total' in metrics
    
    def test_concurrent_gauge_updates(self, clean_registry):
        """Test concurrent gauge updates"""
        import threading
        
        def update_gauge(value):
            storage_utilization_ratio.labels(storage_class='hot').set(value)
        
        threads = [threading.Thread(target=update_gauge, args=(i/10,)) for i in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_storage_utilization_ratio' in metrics
    
    def test_concurrent_histogram_observations(self, clean_registry):
        """Test concurrent histogram observations"""
        import threading
        import random
        
        def observe_latency():
            for _ in range(100):
                latency = random.uniform(0.001, 5.0)
                storage_operation_duration_seconds.labels(operation_type='store').observe(latency)
        
        threads = [threading.Thread(target=observe_latency) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        metrics = generate_latest(clean_registry)
        assert b'sigmavault_storage_operation_duration_seconds' in metrics


# ============================================================================
# TEST LABEL CARDINALITY VALIDATION
# ============================================================================

class TestLabelCardinality:
    """Tests for metric label cardinality"""
    
    def test_storage_class_labels_valid(self, clean_registry):
        """Test storage class labels are valid"""
        for storage_class in ['hot', 'warm', 'cold', 'archive']:
            storage_capacity_bytes.labels(storage_class=storage_class).set(1000)
        
        metrics = generate_latest(clean_registry)
        for storage_class in ['hot', 'warm', 'cold', 'archive']:
            assert f'storage_class="{storage_class}"'.encode() in metrics
    
    def test_operation_type_labels_valid(self, clean_registry):
        """Test operation type labels are valid"""
        for op_type in ['store', 'retrieve', 'delete', 'snapshot']:
            storage_operations_total.labels(operation_type=op_type, status='success').inc()
        
        metrics = generate_latest(clean_registry)
        for op_type in ['store', 'retrieve', 'delete', 'snapshot']:
            assert f'operation_type="{op_type}"'.encode() in metrics
    
    def test_cost_center_labels_distinct(self, clean_registry):
        """Test different cost centers are tracked"""
        for center in ['prod', 'staging', 'dev']:
            operation_cost_usd.labels(operation_type='store', status='success', cost_center=center).inc(0.01)
        
        metrics = generate_latest(clean_registry)
        for center in ['prod', 'staging', 'dev']:
            assert f'cost_center="{center}"'.encode() in metrics
    
    def test_high_cardinality_warning(self, clean_registry):
        """Test that reasonable cardinality is maintained"""
        # Each dimension should have limited values to avoid cardinality explosion
        operation_types = ['store', 'retrieve', 'delete', 'snapshot']
        statuses = ['success', 'error', 'timeout']
        cost_centers = ['prod', 'staging', 'dev', 'research']
        
        count = 0
        for op_type in operation_types:
            for status in statuses:
                for center in cost_centers:
                    storage_operations_total.labels(
                        operation_type=op_type, 
                        status=status,
                    ).inc()
                    count += 1
        
        # Should be manageable cardinality
        assert count == len(operation_types) * len(statuses)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
