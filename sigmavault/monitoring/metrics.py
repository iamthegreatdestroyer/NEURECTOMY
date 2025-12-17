"""
Prometheus Metrics for ΣVAULT Storage Service
Complete storage operation tracking with financial cost attribution
"""

from prometheus_client import Counter, Histogram, Gauge, Info
from functools import wraps
import time
from typing import Callable, Optional, Dict, Any, Literal
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Storage Operation Metrics
# ============================================================================

storage_operations_total = Counter(
    'sigmavault_storage_operations_total',
    'Total storage operations',
    ['operation_type', 'status'],  # operation_type: store, retrieve, delete, snapshot
    help='Count of storage operations by type and status'
)

storage_operation_duration_seconds = Histogram(
    'sigmavault_storage_operation_duration_seconds',
    'Storage operation latency in seconds',
    ['operation_type'],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0),
    help='End-to-end operation latency'
)

storage_operation_size_bytes = Histogram(
    'sigmavault_storage_operation_size_bytes',
    'Size of storage operation in bytes',
    ['operation_type'],
    buckets=(100, 1024, 10*1024, 100*1024, 1024*1024, 10*1024*1024, 100*1024*1024, 1024*1024*1024),
    help='Data size for operation'
)


# ============================================================================
# Storage Capacity Metrics
# ============================================================================

storage_capacity_bytes = Gauge(
    'sigmavault_storage_capacity_bytes',
    'Total storage capacity in bytes',
    ['storage_class'],  # storage_class: hot, warm, cold, archive
    help='Total storage capacity by tier'
)

storage_utilization_bytes = Gauge(
    'sigmavault_storage_utilization_bytes',
    'Current storage utilization in bytes',
    ['storage_class'],
    help='Used storage by tier'
)

storage_utilization_ratio = Gauge(
    'sigmavault_storage_utilization_ratio',
    'Storage utilization ratio (0.0 to 1.0)',
    ['storage_class'],
    help='Used/total ratio by tier'
)

object_count = Gauge(
    'sigmavault_object_count',
    'Number of stored objects',
    ['storage_class'],
    help='Object count by tier'
)

object_size_bytes = Histogram(
    'sigmavault_object_size_bytes',
    'Size of stored objects in bytes',
    ['storage_class'],
    buckets=(100, 1024, 10*1024, 100*1024, 1024*1024, 10*1024*1024, 100*1024*1024, 1024*1024*1024),
    help='Distribution of object sizes'
)


# ============================================================================
# Performance Metrics
# ============================================================================

storage_throughput_bytes_per_second = Gauge(
    'sigmavault_storage_throughput_bytes_per_second',
    'Storage throughput (bytes/second)',
    ['operation_type'],
    help='Real-time read/write speed'
)

storage_iops = Gauge(
    'sigmavault_storage_iops',
    'Storage I/O operations per second',
    ['operation_type'],
    help='Operations per second'
)

queue_depth = Gauge(
    'sigmavault_queue_depth',
    'Current operation queue depth',
    help='Number of pending operations'
)


# ============================================================================
# Encryption Metrics
# ============================================================================

encryption_operations_total = Counter(
    'sigmavault_encryption_operations_total',
    'Total encryption operations',
    ['key_type'],  # key_type: data, metadata, index
    help='Encryption operation count'
)

encryption_duration_seconds = Histogram(
    'sigmavault_encryption_duration_seconds',
    'Encryption operation duration',
    ['key_type'],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5),
    help='Time for encryption/decryption'
)

key_rotations_total = Counter(
    'sigmavault_key_rotations_total',
    'Total key rotations',
    ['key_type'],
    help='Key rotation operations'
)


# ============================================================================
# Financial Cost Metrics
# ============================================================================

storage_cost_usd = Counter(
    'sigmavault_storage_cost_usd',
    'Storage costs in USD',
    ['cost_type', 'storage_class', 'cost_center'],  # cost_type: storage, compute, transfer
    help='Cumulative storage costs'
)

operation_cost_usd = Counter(
    'sigmavault_operation_cost_usd',
    'Operation cost in USD',
    ['operation_type', 'status', 'cost_center'],
    help='Cost per operation'
)

transfer_cost_usd = Counter(
    'sigmavault_transfer_cost_usd',
    'Data transfer costs in USD',
    ['transfer_type', 'cost_center'],  # transfer_type: ingress, egress, cross_region
    help='Transfer costs'
)

total_cost_usd = Counter(
    'sigmavault_total_cost_usd',
    'Total costs in USD',
    ['cost_center', 'month'],
    help='Monthly cost tracking'
)

monthly_cost_forecast_usd = Gauge(
    'sigmavault_monthly_cost_forecast_usd',
    'Forecasted monthly cost in USD',
    ['cost_center'],
    help='Projected month-end costs'
)


# ============================================================================
# Cost Attribution Metrics
# ============================================================================

cost_per_gb_month = Gauge(
    'sigmavault_cost_per_gb_month',
    'Monthly cost per GB',
    ['storage_class', 'cost_center'],
    help='Unit cost metric'
)

cost_per_operation = Histogram(
    'sigmavault_cost_per_operation',
    'Cost per storage operation',
    ['operation_type'],
    buckets=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0),
    help='Operation cost distribution'
)

total_stored_gb_month = Gauge(
    'sigmavault_total_stored_gb_month',
    'Total GB-months of storage',
    ['storage_class', 'cost_center'],
    help='Storage volume metric'
)


# ============================================================================
# Snapshot/Backup Metrics
# ============================================================================

snapshot_operations_total = Counter(
    'sigmavault_snapshot_operations_total',
    'Total snapshot operations',
    ['operation_type', 'status'],  # operation_type: backup, restore
    help='Snapshot operation count'
)

snapshot_duration_seconds = Histogram(
    'sigmavault_snapshot_duration_seconds',
    'Snapshot operation duration',
    ['operation_type'],
    buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0),
    help='Time for snapshot operations'
)

snapshot_data_size_bytes = Histogram(
    'sigmavault_snapshot_data_size_bytes',
    'Size of snapshot data',
    ['operation_type'],
    buckets=(1024*1024, 10*1024*1024, 100*1024*1024, 1024*1024*1024),
    help='Snapshot size distribution'
)


# ============================================================================
# Reliability and Integrity Metrics
# ============================================================================

storage_availability_ratio = Gauge(
    'sigmavault_storage_availability_ratio',
    'Storage availability (0.0 to 1.0)',
    help='Uptime/availability ratio'
)

data_integrity_checks_total = Counter(
    'sigmavault_data_integrity_checks_total',
    'Data integrity checks performed',
    ['status'],  # status: passed, failed
    help='Integrity check count'
)

replication_lag_seconds = Histogram(
    'sigmavault_replication_lag_seconds',
    'Replication lag between copies',
    buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0),
    help='Time to replicate data'
)

sla_breaches_total = Counter(
    'sigmavault_sla_breaches_total',
    'SLA breaches',
    ['sla_type'],  # sla_type: availability, latency, throughput
    help='SLA violation count'
)


# ============================================================================
# Error Tracking
# ============================================================================

storage_errors_total = Counter(
    'sigmavault_storage_errors_total',
    'Storage errors',
    ['operation_type', 'error_type'],  # error_type: timeout, permission, corruption, etc.
    help='Error count by type'
)

retry_operations_total = Counter(
    'sigmavault_retry_operations_total',
    'Retry operations',
    ['operation_type'],
    help='Retry count'
)


# ============================================================================
# System Information
# ============================================================================

system_info = Info(
    'sigmavault_system',
    'ΣVAULT storage system information',
)


# ============================================================================
# Decorators and Context Managers
# ============================================================================

def track_storage_operation(operation_type: str, cost_center: str = 'default'):
    """Decorator to track storage operations with cost attribution"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            size_bytes = 0
            cost_usd = Decimal('0')
            
            try:
                result = await func(*args, **kwargs)
                
                if hasattr(result, 'size_bytes'):
                    size_bytes = result.size_bytes
                if hasattr(result, 'cost_usd'):
                    cost_usd = result.cost_usd
                
                return result
            except TimeoutError:
                status = "timeout"
                storage_errors_total.labels(operation_type=operation_type, error_type='timeout').inc()
                raise
            except PermissionError:
                status = "error"
                storage_errors_total.labels(operation_type=operation_type, error_type='permission').inc()
                raise
            except Exception as e:
                status = "error"
                storage_errors_total.labels(operation_type=operation_type, error_type='unknown').inc()
                logger.error(f"Storage operation error: {e}")
                raise
            finally:
                duration = time.time() - start_time
                storage_operations_total.labels(operation_type=operation_type, status=status).inc()
                storage_operation_duration_seconds.labels(operation_type=operation_type).observe(duration)
                
                if size_bytes > 0:
                    storage_operation_size_bytes.labels(operation_type=operation_type).observe(size_bytes)
                    throughput = size_bytes / duration if duration > 0 else 0
                    storage_throughput_bytes_per_second.labels(operation_type=operation_type).set(throughput)
                
                if cost_usd > 0:
                    operation_cost_usd.labels(
                        operation_type=operation_type,
                        status=status,
                        cost_center=cost_center
                    ).inc(float(cost_usd))
        
        return wrapper
    return decorator


class StorageContext:
    """Context manager for storage operations with cost tracking"""
    
    def __init__(self, operation_type: str, cost_center: str = 'default'):
        self.operation_type = operation_type
        self.cost_center = cost_center
        self.start_time = None
        self.size_bytes = 0
        self.cost_usd = Decimal('0')
        self.status = "success"
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is not None:
            self.status = "error"
            error_type = exc_type.__name__.lower()
            storage_errors_total.labels(operation_type=self.operation_type, error_type=error_type).inc()
        
        storage_operations_total.labels(operation_type=self.operation_type, status=self.status).inc()
        storage_operation_duration_seconds.labels(operation_type=self.operation_type).observe(duration)
        
        if self.size_bytes > 0:
            storage_operation_size_bytes.labels(operation_type=self.operation_type).observe(self.size_bytes)
            throughput = self.size_bytes / duration if duration > 0 else 0
            storage_throughput_bytes_per_second.labels(operation_type=self.operation_type).set(throughput)
        
        if self.cost_usd > 0:
            operation_cost_usd.labels(
                operation_type=self.operation_type,
                status=self.status,
                cost_center=self.cost_center
            ).inc(float(self.cost_usd))
        
        return False
    
    def set_size(self, size_bytes: int):
        """Set operation size"""
        self.size_bytes = size_bytes
    
    def set_cost(self, cost_usd: Decimal):
        """Set operation cost"""
        self.cost_usd = cost_usd


# ============================================================================
# Helper Functions - Financial
# ============================================================================

def record_storage_cost(storage_class: str, gb_month: float, cost_center: str, cost_usd: Decimal):
    """Record storage cost by class"""
    storage_cost_usd.labels(cost_type='storage', storage_class=storage_class, cost_center=cost_center).inc(float(cost_usd))
    cost_per_gb_month.labels(storage_class=storage_class, cost_center=cost_center).set(float(cost_usd / Decimal(gb_month)))
    total_stored_gb_month.labels(storage_class=storage_class, cost_center=cost_center).set(gb_month)


def record_transfer_cost(transfer_type: str, gb: float, cost_center: str, cost_usd: Decimal):
    """Record data transfer cost"""
    transfer_cost_usd.labels(transfer_type=transfer_type, cost_center=cost_center).inc(float(cost_usd))


def update_monthly_forecast(cost_center: str, forecast_usd: Decimal):
    """Update monthly cost forecast"""
    monthly_cost_forecast_usd.labels(cost_center=cost_center).set(float(forecast_usd))


# ============================================================================
# Helper Functions - Storage
# ============================================================================

def update_storage_capacity(storage_class: str, capacity_bytes: int, used_bytes: int, object_count: int):
    """Update storage capacity metrics"""
    storage_capacity_bytes.labels(storage_class=storage_class).set(capacity_bytes)
    storage_utilization_bytes.labels(storage_class=storage_class).set(used_bytes)
    
    ratio = used_bytes / capacity_bytes if capacity_bytes > 0 else 0
    storage_utilization_ratio.labels(storage_class=storage_class).set(ratio)
    
    object_count_gauge = Gauge('sigmavault_object_count_temp', 'temp')
    object_count_gauge.labels(storage_class=storage_class).set(object_count)


def record_object_size(storage_class: str, size_bytes: int):
    """Record object size"""
    object_size_bytes.labels(storage_class=storage_class).observe(size_bytes)


def update_throughput(operation_type: str, bytes_per_sec: float):
    """Update throughput metric"""
    storage_throughput_bytes_per_second.labels(operation_type=operation_type).set(bytes_per_sec)


def update_iops(operation_type: str, ops_per_sec: float):
    """Update IOPS metric"""
    storage_iops.labels(operation_type=operation_type).set(ops_per_sec)


def update_queue_depth(depth: int):
    """Update queue depth"""
    queue_depth.set(depth)


# ============================================================================
# Helper Functions - Encryption
# ============================================================================

def record_encryption_operation(key_type: str):
    """Record encryption operation"""
    encryption_operations_total.labels(key_type=key_type).inc()


def record_key_rotation(key_type: str):
    """Record key rotation"""
    key_rotations_total.labels(key_type=key_type).inc()


# ============================================================================
# Helper Functions - Reliability
# ============================================================================

def update_availability(availability_ratio: float):
    """Update storage availability"""
    storage_availability_ratio.set(availability_ratio)


def record_integrity_check(passed: bool):
    """Record integrity check result"""
    status = "passed" if passed else "failed"
    data_integrity_checks_total.labels(status=status).inc()


def record_sla_breach(sla_type: str):
    """Record SLA breach"""
    sla_breaches_total.labels(sla_type=sla_type).inc()
