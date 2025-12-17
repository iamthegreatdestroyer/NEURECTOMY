"""
ΣVAULT Storage Monitoring Module
Storage operations with comprehensive cost attribution and financial tracking
"""

from .metrics import (
    # Operation metrics
    storage_operations_total,
    storage_operation_duration_seconds,
    storage_operation_size_bytes,
    
    # Capacity metrics
    storage_capacity_bytes,
    storage_utilization_bytes,
    storage_utilization_ratio,
    object_count,
    object_size_bytes,
    
    # Performance
    storage_throughput_bytes_per_second,
    storage_iops,
    queue_depth,
    
    # Encryption
    encryption_operations_total,
    encryption_duration_seconds,
    key_rotations_total,
    
    # Financial
    storage_cost_usd,
    operation_cost_usd,
    transfer_cost_usd,
    total_cost_usd,
    monthly_cost_forecast_usd,
    cost_per_gb_month,
    cost_per_operation,
    total_stored_gb_month,
    
    # Snapshots
    snapshot_operations_total,
    snapshot_duration_seconds,
    snapshot_data_size_bytes,
    
    # Reliability
    storage_availability_ratio,
    data_integrity_checks_total,
    replication_lag_seconds,
    sla_breaches_total,
    
    # Errors
    storage_errors_total,
    retry_operations_total,
    
    # System info
    system_info,
    
    # Context managers and decorators
    track_storage_operation,
    StorageContext,
    
    # Helpers
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

__all__ = [
    "storage_operations_total",
    "storage_operation_duration_seconds",
    "storage_operation_size_bytes",
    "storage_capacity_bytes",
    "storage_utilization_bytes",
    "storage_utilization_ratio",
    "object_count",
    "object_size_bytes",
    "storage_throughput_bytes_per_second",
    "storage_iops",
    "queue_depth",
    "encryption_operations_total",
    "encryption_duration_seconds",
    "key_rotations_total",
    "storage_cost_usd",
    "operation_cost_usd",
    "transfer_cost_usd",
    "total_cost_usd",
    "monthly_cost_forecast_usd",
    "cost_per_gb_month",
    "cost_per_operation",
    "total_stored_gb_month",
    "snapshot_operations_total",
    "snapshot_duration_seconds",
    "snapshot_data_size_bytes",
    "storage_availability_ratio",
    "data_integrity_checks_total",
    "replication_lag_seconds",
    "sla_breaches_total",
    "storage_errors_total",
    "retry_operations_total",
    "system_info",
    "track_storage_operation",
    "StorageContext",
    "record_storage_cost",
    "record_transfer_cost",
    "update_monthly_forecast",
    "update_storage_capacity",
    "record_object_size",
    "update_throughput",
    "update_iops",
    "update_queue_depth",
    "record_encryption_operation",
    "record_key_rotation",
    "update_availability",
    "record_integrity_check",
    "record_sla_breach",
]
