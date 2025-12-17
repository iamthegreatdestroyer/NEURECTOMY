# ΣVAULT Storage Metrics - Complete Reference

## Overview

ΣVAULT (Sigma Vault) is Neurectomy's enterprise-grade encrypted storage service with comprehensive financial cost attribution and audit capabilities. This document provides complete reference documentation for all 50+ Prometheus metrics, integration patterns, and operational guidelines.

**File Location:** `sigmavault/monitoring/metrics.py` (488 lines)

### Key Characteristics

- **Complete Coverage:** Storage operations, capacity, performance, encryption, costs, snapshots, reliability
- **Financial Tracking:** Real-time cost attribution across storage classes and cost centers
- **Encryption Metrics:** Key rotations, operation times, and encryption/decryption tracking
- **SLA Monitoring:** Availability, latency, throughput, and breach detection
- **Cost Analysis:** Per-GB, per-operation, transfer, and monthly forecasting

---

## Metrics Categories

### 1. Storage Operation Metrics

Track all storage operations with detailed performance data.

#### `sigmavault_storage_operations_total` (Counter)

**Type:** Counter  
**Labels:** `operation_type`, `status`  
**Help:** Count of storage operations by type and status

**Operation Types:**

- `store` - Write/upload operations
- `retrieve` - Read/download operations
- `delete` - Deletion operations
- `snapshot` - Backup operations

**Status Values:**

- `success` - Operation completed successfully
- `error` - Operation failed
- `timeout` - Operation exceeded timeout
- `partial` - Partial completion

**Example Query:**

```promql
# Total operations per operation type
rate(sigmavault_storage_operations_total[5m])

# Success rate percentage
(rate(sigmavault_storage_operations_total{status="success"}[5m]) /
 rate(sigmavault_storage_operations_total[5m])) * 100
```

**Usage in Code:**

```python
from sigmavault.monitoring.metrics import storage_operations_total

@track_storage_operation(operation_type='store', cost_center='default')
async def store_object(data: bytes) -> StorageResult:
    # Operation automatically tracked
    pass

# Manual tracking
storage_operations_total.labels(
    operation_type='retrieve',
    status='success'
).inc()
```

---

#### `sigmavault_storage_operation_duration_seconds` (Histogram)

**Type:** Histogram  
**Labels:** `operation_type`  
**Buckets:** 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0 (seconds)  
**Help:** End-to-end operation latency

Measures complete operation time including:

- Network overhead
- Encryption/decryption
- Storage backend I/O
- Serialization

**Bucket Selection Rationale:**

- 1ms-5ms: Cache/network latency
- 10ms-50ms: Small object operations
- 100ms-500ms: Medium object operations
- 1s+: Large object or slow backend

**Example Queries:**

```promql
# P95 latency for store operations
histogram_quantile(0.95, rate(sigmavault_storage_operation_duration_seconds_bucket{operation_type="store"}[5m]))

# Average latency
rate(sigmavault_storage_operation_duration_seconds_sum[5m]) /
rate(sigmavault_storage_operation_duration_seconds_count[5m])

# Operations slower than 1 second
rate(sigmavault_storage_operation_duration_seconds_bucket{le="5.0"}[5m])
```

---

#### `sigmavault_storage_operation_size_bytes` (Histogram)

**Type:** Histogram  
**Labels:** `operation_type`  
**Buckets:** 100B, 1KB, 10KB, 100KB, 1MB, 10MB, 100MB, 1GB  
**Help:** Size of storage operation in bytes

Tracks data volume patterns by operation type.

**Example Queries:**

```promql
# Average object size by operation type
(rate(sigmavault_storage_operation_size_bytes_sum[5m]) /
 rate(sigmavault_storage_operation_size_bytes_count[5m])) / 1024 / 1024

# Objects larger than 100MB
rate(sigmavault_storage_operation_size_bytes_bucket{le="1024*1024*1024"}[5m]) -
rate(sigmavault_storage_operation_size_bytes_bucket{le="104857600"}[5m])
```

---

### 2. Storage Capacity Metrics

Monitor storage utilization across multiple tiers.

#### `sigmavault_storage_capacity_bytes` (Gauge)

**Type:** Gauge  
**Labels:** `storage_class`  
**Help:** Total storage capacity in bytes

**Storage Classes:**

- `hot` - Frequently accessed (SSD-backed, ~$0.023/GB/month)
- `warm` - Occasional access (HDD-backed, ~$0.01/GB/month)
- `cold` - Archival access (Tape/object, ~$0.004/GB/month)
- `archive` - Rare access (Deep archive, ~$0.0013/GB/month)

**Example Query:**

```promql
# Total capacity across all tiers
sum(sigmavault_storage_capacity_bytes) / 1024 / 1024 / 1024
```

**Manual Updates (from provisioning):**

```python
storage_capacity_bytes.labels(storage_class='hot').set(10 * 1024**3)  # 10GB hot
storage_capacity_bytes.labels(storage_class='warm').set(100 * 1024**3)  # 100GB warm
storage_capacity_bytes.labels(storage_class='cold').set(500 * 1024**3)  # 500GB cold
```

---

#### `sigmavault_storage_utilization_bytes` (Gauge)

**Type:** Gauge  
**Labels:** `storage_class`  
**Help:** Current storage utilization in bytes

**Update Mechanism:**

- Real-time tracking during operations
- Periodic sync from storage backend (every 5 minutes)
- Reconciliation during snapshot operations

**Example Queries:**

```promql
# Utilization by tier
sigmavault_storage_utilization_bytes / 1024 / 1024 / 1024

# Total utilized storage
sum(sigmavault_storage_utilization_bytes) / 1024 / 1024 / 1024
```

---

#### `sigmavault_storage_utilization_ratio` (Gauge)

**Type:** Gauge  
**Labels:** `storage_class`  
**Range:** 0.0 to 1.0  
**Help:** Storage utilization ratio (used/total)

**Threshold Guidance:**

- 0.0-0.5: Healthy (comfortable headroom)
- 0.5-0.8: Caution (plan expansion)
- 0.8-0.95: Warning (expansion needed)
- 0.95-1.0: Critical (immediate action)

**Example Queries:**

```promql
# Alert when hot storage exceeds 85% utilization
sigmavault_storage_utilization_ratio{storage_class="hot"} > 0.85

# Average utilization across all classes
avg(sigmavault_storage_utilization_ratio)
```

**Alert Rules:**

```yaml
- alert: StorageCapacityWarning
  expr: sigmavault_storage_utilization_ratio{storage_class="hot"} > 0.8
  for: 10m
  annotations:
    summary: "Hot storage at {{ $value | humanizePercentage }} capacity"

- alert: StorageCapacityCritical
  expr: sigmavault_storage_utilization_ratio{storage_class="hot"} > 0.95
  for: 5m
  annotations:
    summary: "Hot storage CRITICAL at {{ $value | humanizePercentage }} capacity"
```

---

#### `sigmavault_object_count` (Gauge)

**Type:** Gauge  
**Labels:** `storage_class`  
**Help:** Number of stored objects

Useful for tracking:

- Average object size: `used_bytes / object_count`
- Fragmentation patterns
- API quotas

**Example Query:**

```promql
# Average object size by tier (in MB)
(sigmavault_storage_utilization_bytes / sigmavault_object_count) / 1024 / 1024
```

---

#### `sigmavault_object_size_bytes` (Histogram)

**Type:** Histogram  
**Labels:** `storage_class`  
**Buckets:** 100B, 1KB, 10KB, 100KB, 1MB, 10MB, 100MB, 1GB  
**Help:** Distribution of object sizes

Informs:

- API batch sizing
- Transfer optimization
- Cost analysis

---

### 3. Performance Metrics

Real-time throughput and I/O monitoring.

#### `sigmavault_storage_throughput_bytes_per_second` (Gauge)

**Type:** Gauge  
**Labels:** `operation_type`  
**Help:** Storage throughput in bytes/second

**Tracking Points:**

```python
# Automatically updated by @track_storage_operation decorator
throughput = size_bytes / duration if duration > 0 else 0
storage_throughput_bytes_per_second.labels(operation_type=operation_type).set(throughput)
```

**Example Queries:**

```promql
# Read/write throughput in MB/s
sigmavault_storage_throughput_bytes_per_second / 1024 / 1024

# Bottleneck detection
min(sigmavault_storage_throughput_bytes_per_second)
```

**Baseline Performance:**

- SSD (hot): 100-500 MB/s
- HDD (warm): 50-200 MB/s
- Archive (cold): 1-10 MB/s

---

#### `sigmavault_storage_iops` (Gauge)

**Type:** Gauge  
**Labels:** `operation_type`  
**Help:** Storage I/O operations per second

**Calculation:**

```python
iops = operations_count / time_window_seconds
storage_iops.labels(operation_type=operation_type).set(iops)
```

**Baseline IOPS:**

- SSD (hot): 5,000-20,000 IOPS
- HDD (warm): 100-500 IOPS
- Archive (cold): 10-50 IOPS

---

#### `sigmavault_queue_depth` (Gauge)

**Type:** Gauge  
**Help:** Current operation queue depth

**Usage:**

```python
queue_depth.set(len(pending_operations))
```

**Alert Threshold:**

```yaml
- alert: HighQueueDepth
  expr: sigmavault_queue_depth > 1000
  for: 5m
  annotations:
    summary: "Storage queue depth: {{ $value }}"
```

---

### 4. Encryption Metrics

Track encryption operations and key management.

#### `sigmavault_encryption_operations_total` (Counter)

**Type:** Counter  
**Labels:** `key_type`  
**Help:** Total encryption operations

**Key Types:**

- `data` - Data encryption/decryption
- `metadata` - Metadata encryption
- `index` - Index encryption

**Example Query:**

```promql
# Encryption operation rate (ops/sec)
rate(sigmavault_encryption_operations_total[1m])
```

---

#### `sigmavault_encryption_duration_seconds` (Histogram)

**Type:** Histogram  
**Labels:** `key_type`  
**Buckets:** 0.001, 0.005, 0.01, 0.05, 0.1, 0.5 (seconds)  
**Help:** Encryption operation duration

**Performance Impact Analysis:**

```promql
# Encryption overhead as % of total operation time
(rate(sigmavault_encryption_duration_seconds_sum[5m]) /
 rate(sigmavault_storage_operation_duration_seconds_sum[5m])) * 100
```

---

#### `sigmavault_key_rotations_total` (Counter)

**Type:** Counter  
**Labels:** `key_type`  
**Help:** Total key rotations

**Audit Trail:**

```promql
# Track key rotation frequency
increase(sigmavault_key_rotations_total[24h])
```

**Compliance Requirements:**

- Data keys: Rotate annually (or per policy)
- Metadata keys: Rotate monthly
- Index keys: Rotate as needed

---

### 5. Financial Cost Metrics

Complete cost attribution across operations and time periods.

#### `sigmavault_storage_cost_usd` (Counter)

**Type:** Counter  
**Labels:** `cost_type`, `storage_class`, `cost_center`  
**Help:** Cumulative storage costs in USD

**Cost Types:**

- `storage` - Per-GB-month charges
- `compute` - Processing/API charges
- `transfer` - Egress/cross-region charges

**Example Queries:**

```promql
# Total storage costs this month
increase(sigmavault_storage_cost_usd{cost_type="storage"}[30d])

# Costs by storage class
sum by(storage_class) (increase(sigmavault_storage_cost_usd{cost_type="storage"}[30d]))

# Costs by cost center
sum by(cost_center) (increase(sigmavault_storage_cost_usd[30d]))
```

**Cost Calculation Example:**

```python
# Store 100GB in hot storage
storage_gb = 100
monthly_rate_per_gb = 0.023  # $0.023/GB/month for hot storage
cost = storage_gb * monthly_rate_per_gb

storage_cost_usd.labels(
    cost_type='storage',
    storage_class='hot',
    cost_center='engineering'
).inc(cost)
```

---

#### `sigmavault_operation_cost_usd` (Counter)

**Type:** Counter  
**Labels:** `operation_type`, `status`, `cost_center`  
**Help:** Cost per operation

**Cost Mapping:**

- Store: $0.0005 per operation
- Retrieve: $0.00001 per operation
- Delete: $0.00001 per operation
- Snapshot: $0.01 per snapshot

**Tracking:**

```python
@track_storage_operation(operation_type='retrieve', cost_center='analytics')
async def get_object(key: str):
    # Cost automatically tracked based on operation success
    pass
```

---

#### `sigmavault_transfer_cost_usd` (Counter)

**Type:** Counter  
**Labels:** `transfer_type`, `cost_center`  
**Help:** Data transfer costs in USD

**Transfer Types:**

- `ingress` - Inbound data (typically free or $0.01/GB)
- `egress` - Outbound data (typically $0.09/GB)
- `cross_region` - Cross-region transfer (typically $0.02/GB)

**Cost Calculation:**

```python
# 50GB egress at $0.09/GB
egress_gb = 50
transfer_cost_usd.labels(
    transfer_type='egress',
    cost_center='analytics'
).inc(egress_gb * 0.09)
```

---

#### `sigmavault_total_cost_usd` (Counter)

**Type:** Counter  
**Labels:** `cost_center`, `month`  
**Help:** Monthly cost tracking

**Monthly Reporting:**

```promql
# Total cost by month
sum by(month) (increase(sigmavault_total_cost_usd[30d]))

# Month-over-month growth
month_over_month_growth = (current_month - previous_month) / previous_month * 100
```

**Alert Rules:**

```yaml
- alert: MonthlyBudgetExceeded
  expr: increase(sigmavault_total_cost_usd[30d]) > 10000
  annotations:
    summary: "Monthly storage costs exceed $10k budget"
```

---

#### `sigmavault_monthly_cost_forecast_usd` (Gauge)

**Type:** Gauge  
**Labels:** `cost_center`  
**Help:** Forecasted monthly cost in USD

**Calculation:**

```python
# Extrapolate current rate for full month
current_cost = sum(increase(sigmavault_total_cost_usd[current_day_of_month*24h]))
days_passed = datetime.now().day
days_in_month = calendar.monthrange(datetime.now().year, datetime.now().month)[1]
forecasted_cost = current_cost * (days_in_month / days_passed)

monthly_cost_forecast_usd.labels(cost_center='analytics').set(forecasted_cost)
```

---

### 6. Cost Attribution Metrics

Detailed cost analysis by unit.

#### `sigmavault_cost_per_gb_month` (Gauge)

**Type:** Gauge  
**Labels:** `storage_class`, `cost_center`  
**Unit:** USD  
**Help:** Monthly cost per GB

**Pricing Tiers:**

```
Hot:     $0.023/GB/month
Warm:    $0.010/GB/month
Cold:    $0.004/GB/month
Archive: $0.0013/GB/month
```

**Chargeback Example:**

```python
hot_cost = 0.023
warm_cost = 0.010

cost_per_gb_month.labels(
    storage_class='hot',
    cost_center='engineering'
).set(hot_cost)

cost_per_gb_month.labels(
    storage_class='warm',
    cost_center='analytics'
).set(warm_cost)
```

---

#### `sigmavault_cost_per_operation` (Histogram)

**Type:** Histogram  
**Labels:** `operation_type`  
**Buckets:** $0.00001, $0.0001, $0.001, $0.01, $0.1, $1.0  
**Help:** Cost per operation distribution

**Example Queries:**

```promql
# Average cost per retrieve operation
(rate(sigmavault_cost_per_operation_sum{operation_type="retrieve"}[5m]) /
 rate(sigmavault_cost_per_operation_count{operation_type="retrieve"}[5m]))

# Cost distribution (P50, P95, P99)
histogram_quantile(0.5, rate(sigmavault_cost_per_operation_bucket{operation_type="store"}[5m]))
histogram_quantile(0.95, rate(sigmavault_cost_per_operation_bucket{operation_type="store"}[5m]))
histogram_quantile(0.99, rate(sigmavault_cost_per_operation_bucket{operation_type="store"}[5m]))
```

---

#### `sigmavault_total_stored_gb_month` (Gauge)

**Type:** Gauge  
**Labels:** `storage_class`, `cost_center`  
**Unit:** GB-months  
**Help:** Storage volume metric

**Calculation:**

```python
# GB-months = average GB stored over month
average_gb = mean(storage_utilization_bytes / 1024**3)
gb_months = average_gb * 1  # for 1 month

total_stored_gb_month.labels(
    storage_class='hot',
    cost_center='analytics'
).set(gb_months)
```

---

### 7. Snapshot/Backup Metrics

Track backup operations and recovery capabilities.

#### `sigmavault_snapshot_operations_total` (Counter)

**Type:** Counter  
**Labels:** `operation_type`, `status`  
**Help:** Snapshot operation count

**Operation Types:**

- `backup` - Backup creation
- `restore` - Restore from backup

**Status Values:**

- `success` - Backup/restore completed
- `failed` - Operation failed
- `partial` - Partial completion

**Example Queries:**

```promql
# Backup success rate
(rate(sigmavault_snapshot_operations_total{operation_type="backup", status="success"}[24h]) /
 rate(sigmavault_snapshot_operations_total{operation_type="backup"}[24h])) * 100

# Total backups created today
increase(sigmavault_snapshot_operations_total{operation_type="backup", status="success"}[24h])
```

---

#### `sigmavault_snapshot_duration_seconds` (Histogram)

**Type:** Histogram  
**Labels:** `operation_type`  
**Buckets:** 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0 (seconds)  
**Help:** Snapshot operation duration

**SLA Targets:**

- Backup: < 30 seconds for < 100GB
- Restore: < 60 seconds for < 100GB

**Example Alerts:**

```yaml
- alert: SlowBackup
  expr: histogram_quantile(0.95, rate(sigmavault_snapshot_duration_seconds_bucket{operation_type="backup"}[5m])) > 30
  annotations:
    summary: "Backup P95 latency exceeds 30s"

- alert: SlowRestore
  expr: histogram_quantile(0.95, rate(sigmavault_snapshot_duration_seconds_bucket{operation_type="restore"}[5m])) > 60
  annotations:
    summary: "Restore P95 latency exceeds 60s"
```

---

#### `sigmavault_snapshot_data_size_bytes` (Histogram)

**Type:** Histogram  
**Labels:** `operation_type`  
**Buckets:** 1MB, 10MB, 100MB, 1GB  
**Help:** Snapshot data size distribution

---

### 8. Reliability and Integrity Metrics

Monitor storage reliability and data integrity.

#### `sigmavault_storage_availability_ratio` (Gauge)

**Type:** Gauge  
**Range:** 0.0 to 1.0  
**Help:** Storage availability (uptime ratio)

**Calculation:**

```python
uptime_seconds = (now - last_downtime_start).total_seconds()
downtime_seconds = sum(downtime_windows)
total_seconds = 24 * 3600  # daily

availability = (total_seconds - downtime_seconds) / total_seconds
storage_availability_ratio.set(availability)
```

**SLA Targets:**

- 99.9% = 8.6 hours downtime/month
- 99.95% = 4.3 hours downtime/month
- 99.99% = 52 minutes downtime/month

**Alert Rules:**

```yaml
- alert: LowAvailability
  expr: storage_availability_ratio < 0.9999
  for: 5m
  annotations:
    summary: "Storage availability below 99.99%: {{ $value | humanizePercentage }}"
```

---

#### `sigmavault_data_integrity_checks_total` (Counter)

**Type:** Counter  
**Labels:** `status`  
**Help:** Data integrity checks performed

**Status Values:**

- `passed` - Checksum/parity verified
- `failed` - Corruption detected

**Schedule:**

```python
# Daily integrity checks
@periodic_task(crontab(hour=2, minute=0))
async def daily_integrity_check():
    passed = await verify_all_checksums()
    failed = await detect_corruption()

    data_integrity_checks_total.labels(status='passed').inc(passed)
    data_integrity_checks_total.labels(status='failed').inc(failed)
```

**Example Query:**

```promql
# Corruption detection rate
rate(sigmavault_data_integrity_checks_total{status="failed"}[24h])

# Pass rate
(rate(sigmavault_data_integrity_checks_total{status="passed"}[24h]) /
 rate(sigmavault_data_integrity_checks_total[24h])) * 100
```

---

#### `sigmavault_replication_lag_seconds` (Histogram)

**Type:** Histogram  
**Labels:** None  
**Buckets:** 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0 (seconds)  
**Help:** Replication lag between copies

**Tracks:**

- Primary to secondary replica delay
- Geographic replication lag
- Consistency guarantee

**Alert Rules:**

```yaml
- alert: HighReplicationLag
  expr: histogram_quantile(0.95, rate(sigmavault_replication_lag_seconds_bucket[5m])) > 10
  annotations:
    summary: "Replication lag P95: {{ $value }}s"
```

---

#### `sigmavault_sla_breaches_total` (Counter)

**Type:** Counter  
**Labels:** `sla_type`  
**Help:** SLA violation count

**SLA Types:**

- `availability` - Availability SLA breach
- `latency` - Latency SLA breach
- `throughput` - Throughput SLA breach

**Example Query:**

```promql
# SLA breaches this month
increase(sigmavault_sla_breaches_total[30d])
```

---

### 9. Error Tracking

Monitor and track errors.

#### `sigmavault_storage_errors_total` (Counter)

**Type:** Counter  
**Labels:** `operation_type`, `error_type`  
**Help:** Storage errors

**Error Types:**

- `timeout` - Operation timeout
- `permission` - Permission denied
- `corruption` - Data corruption detected
- `network` - Network error
- `backend` - Backend storage error
- `unknown` - Unknown error

**Example Queries:**

```promql
# Error rate by type
rate(sigmavault_storage_errors_total[5m])

# Timeout errors
rate(sigmavault_storage_errors_total{error_type="timeout"}[5m])

# Error distribution
sum by(error_type) (increase(sigmavault_storage_errors_total[1h]))
```

---

#### `sigmavault_retry_operations_total` (Counter)

**Type:** Counter  
**Labels:** `operation_type`  
**Help:** Retry operations

**Tracks:**

- Automatic retry attempts
- Transient failure recovery
- Retry exhaustion

**Example Query:**

```promql
# Operations requiring retry
rate(sigmavault_retry_operations_total[5m])
```

---

### 10. System Information

Metadata about the storage system.

#### `sigmavault_system` (Info)

**Type:** Info  
**Labels:** `version`, `environment`, `region`  
**Help:** ΣVAULT system information

**Usage:**

```python
system_info.info({
    'version': '2.1.0',
    'environment': 'production',
    'region': 'us-east-1',
    'provider': 'aws'
})
```

---

## Storage Operation Patterns

### Best Practices

#### 1. Decorator Pattern (Recommended)

```python
from sigmavault.monitoring.metrics import track_storage_operation

@track_storage_operation(operation_type='store', cost_center='engineering')
async def store_user_data(user_id: str, data: bytes) -> StorageResult:
    """Store user data with automatic metrics tracking"""
    result = await storage_client.put(f"users/{user_id}", data)

    # Decorator automatically tracks:
    # - Operation start/end time
    # - Success/failure status
    # - Data size
    # - Cost attribution

    return result
```

#### 2. Context Manager Pattern

```python
from sigmavault.monitoring.metrics import StorageContext

async def batch_import_objects():
    """Batch operation with aggregate tracking"""
    async with StorageContext(operation_type='store', cost_center='import') as ctx:
        for obj in objects:
            await ctx.track_operation(
                lambda: storage_client.put(obj.key, obj.data),
                size_bytes=len(obj.data)
            )
```

#### 3. Manual Tracking

```python
from sigmavault.monitoring.metrics import (
    storage_operations_total,
    storage_operation_duration_seconds,
    storage_operation_size_bytes
)
import time

async def retrieve_object(key: str):
    start = time.time()
    try:
        data = await storage_client.get(key)

        storage_operations_total.labels(
            operation_type='retrieve',
            status='success'
        ).inc()

        storage_operation_duration_seconds.labels(
            operation_type='retrieve'
        ).observe(time.time() - start)

        storage_operation_size_bytes.labels(
            operation_type='retrieve'
        ).observe(len(data))

        return data
    except Exception as e:
        storage_operations_total.labels(
            operation_type='retrieve',
            status='error'
        ).inc()
        raise
```

---

## Cost Attribution Methodology

### Tiered Pricing Model

ΣVAULT uses a tiered pricing model with automatic cost tracking:

```
Storage Tier          Cost/GB/Month    Use Case              Response Time
─────────────────────────────────────────────────────────────────────────
Hot (SSD)             $0.023           Frequent access       < 100ms
Warm (HDD)            $0.010           Occasional access     < 1s
Cold (Archive)        $0.004           Rare access           < 5m
Archive (Deep)        $0.0013          Compliance/legal      < 24h
```

### Cost Calculation Example

**Scenario:** Store 500GB of data, with access pattern:

- 100GB hot (accessed daily)
- 200GB warm (accessed weekly)
- 150GB cold (accessed monthly)
- 50GB archive (legal hold)

```python
# Monthly storage cost
hot_cost = 100 * 0.023      # $2.30
warm_cost = 200 * 0.010     # $2.00
cold_cost = 150 * 0.004     # $0.60
archive_cost = 50 * 0.0013  # $0.065

total_storage_cost = 5.00 + 0.065  # ~$5.07/month

# Store the costs
storage_cost_usd.labels(
    cost_type='storage',
    storage_class='hot',
    cost_center='engineering'
).inc(hot_cost)

storage_cost_usd.labels(
    cost_type='storage',
    storage_class='warm',
    cost_center='engineering'
).inc(warm_cost)

# ... repeat for cold and archive
```

### Operation Costs

```
Operation    Cost         Frequency Impact
─────────────────────────────────────────
Store        $0.0005      Large batches reduce per-op cost
Retrieve     $0.00001     Query optimization important
Delete       $0.00001     Minimal cost
Snapshot     $0.01        Monthly/quarterly only
```

### Transfer Costs

```
Transfer Type      Cost/GB    Notes
──────────────────────────────────────────
Ingress           $0.01      One-time import
Egress            $0.09      Continuous drain
Cross-Region      $0.02      DR replication
```

---

## Encryption and Security Metrics

### Key Rotation Tracking

```python
async def rotate_encryption_keys():
    """Track key rotation for compliance"""

    start = time.time()

    for key_type in ['data', 'metadata', 'index']:
        try:
            await kms_client.rotate_key(key_type)
            key_rotations_total.labels(key_type=key_type).inc()

            duration = time.time() - start
            encryption_duration_seconds.labels(key_type=key_type).observe(duration)

        except Exception as e:
            logger.error(f"Key rotation failed: {e}")
```

### Encryption Performance Impact

```promql
# Measure encryption overhead
(rate(sigmavault_encryption_duration_seconds_sum[5m]) /
 rate(sigmavault_storage_operation_duration_seconds_sum[5m])) * 100
```

**Expected Impact:**

- AES-256-GCM: 5-15% overhead
- ChaCha20-Poly1305: 3-10% overhead

---

## Prometheus Queries Reference

### Operational Queries

```promql
# 1. Current operations/second
rate(sigmavault_storage_operations_total[1m])

# 2. Success rate percentage
(rate(sigmavault_storage_operations_total{status="success"}[5m]) /
 rate(sigmavault_storage_operations_total[5m])) * 100

# 3. Error rate by type
rate(sigmavault_storage_errors_total[5m]) by (error_type)

# 4. P95 latency by operation type
histogram_quantile(0.95, rate(sigmavault_storage_operation_duration_seconds_bucket[5m])) by (operation_type)

# 5. P99 latency
histogram_quantile(0.99, rate(sigmavault_storage_operation_duration_seconds_bucket[5m])) by (operation_type)

# 6. Average data size by operation
(rate(sigmavault_storage_operation_size_bytes_sum[5m]) /
 rate(sigmavault_storage_operation_size_bytes_count[5m])) / 1024 / 1024

# 7. Queue depth
sigmavault_queue_depth

# 8. Throughput in MB/s
rate(sigmavault_storage_throughput_bytes_per_second[1m]) / 1024 / 1024
```

### Capacity Queries

```promql
# 9. Storage utilization percentage by tier
(sigmavault_storage_utilization_bytes / sigmavault_storage_capacity_bytes) * 100

# 10. Available capacity by tier
sigmavault_storage_capacity_bytes - sigmavault_storage_utilization_bytes

# 11. Object count by tier
sigmavault_object_count by (storage_class)

# 12. Average object size
(sigmavault_storage_utilization_bytes / sigmavault_object_count) / 1024 / 1024
```

### Financial Queries

```promql
# 13. Monthly storage costs by tier
sum by(storage_class) (increase(sigmavault_storage_cost_usd{cost_type="storage"}[30d]))

# 14. Monthly costs by cost center
sum by(cost_center) (increase(sigmavault_storage_cost_usd[30d]))

# 15. Total monthly cost projection
sum(sigmavault_monthly_cost_forecast_usd)

# 16. Operation costs (hourly)
increase(sigmavault_operation_cost_usd[1h])

# 17. Transfer costs (daily)
increase(sigmavault_transfer_cost_usd[24h])

# 18. Cost per GB-month by tier
(increase(sigmavault_storage_cost_usd{cost_type="storage"}[30d]) /
 sum by(storage_class) (sigmavault_total_stored_gb_month))
```

### Reliability Queries

```promql
# 19. Storage availability
sigmavault_storage_availability_ratio

# 20. Data integrity check pass rate
(rate(sigmavault_data_integrity_checks_total{status="passed"}[24h]) /
 rate(sigmavault_data_integrity_checks_total[24h])) * 100

# 21. Replication lag P95
histogram_quantile(0.95, rate(sigmavault_replication_lag_seconds_bucket[5m]))

# 22. SLA breaches (monthly)
increase(sigmavault_sla_breaches_total[30d])

# 23. Backup success rate
(rate(sigmavault_snapshot_operations_total{operation_type="backup", status="success"}[24h]) /
 rate(sigmavault_snapshot_operations_total{operation_type="backup"}[24h])) * 100
```

---

## Alert Rules

### Performance Alerts

```yaml
groups:
  - name: sigmavault.performance
    interval: 30s
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(sigmavault_storage_operation_duration_seconds_bucket[5m])) > 1
        for: 10m
        annotations:
          summary: "Storage latency P95 > 1s"
          description: "{{ $value }}s"

      - alert: LowThroughput
        expr: rate(sigmavault_storage_throughput_bytes_per_second[5m]) < 10*1024*1024 # 10MB/s
        for: 5m
        annotations:
          summary: "Storage throughput < 10MB/s"

      - alert: HighErrorRate
        expr: (rate(sigmavault_storage_errors_total[5m]) / rate(sigmavault_storage_operations_total[5m])) > 0.05
        for: 5m
        annotations:
          summary: "Storage error rate > 5%"
```

### Capacity Alerts

```yaml
- name: sigmavault.capacity
  interval: 5m
  rules:
    - alert: StorageNearCapacity
      expr: (sigmavault_storage_utilization_bytes / sigmavault_storage_capacity_bytes) > 0.8
      for: 30m
      annotations:
        summary: "{{ $labels.storage_class }} storage 80% full"

    - alert: StorageCriticalCapacity
      expr: (sigmavault_storage_utilization_bytes / sigmavault_storage_capacity_bytes) > 0.95
      for: 5m
      annotations:
        summary: "{{ $labels.storage_class }} storage 95% full - CRITICAL"
```

### Financial Alerts

```yaml
- name: sigmavault.financial
  interval: 1h
  rules:
    - alert: MonthlyBudgetExceeded
      expr: sum(sigmavault_monthly_cost_forecast_usd) > 10000
      annotations:
        summary: "Forecasted monthly costs exceed $10k"
        description: "Projected: ${{ $value }}"

    - alert: UnexpectedCostSpike
      expr: (increase(sigmavault_storage_cost_usd[1d]) / avg_over_time(increase(sigmavault_storage_cost_usd[1d])[7d:1d])) > 1.5
      for: 1h
      annotations:
        summary: "Daily storage costs 50% above average"
```

### Reliability Alerts

```yaml
- name: sigmavault.reliability
  interval: 5m
  rules:
    - alert: LowAvailability
      expr: sigmavault_storage_availability_ratio < 0.9999
      for: 10m
      annotations:
        summary: "Storage availability < 99.99%"

    - alert: DataIntegrityFailure
      expr: rate(sigmavault_data_integrity_checks_total{status="failed"}[24h]) > 0
      for: 1m
      annotations:
        summary: "Data integrity check failures detected"

    - alert: HighReplicationLag
      expr: histogram_quantile(0.95, rate(sigmavault_replication_lag_seconds_bucket[5m])) > 10
      for: 5m
      annotations:
        summary: "Replication lag P95 > 10s"

    - alert: BackupFailure
      expr: rate(sigmavault_snapshot_operations_total{operation_type="backup", status="failed"}[24h]) > 0
      for: 1m
      annotations:
        summary: "Backup failures detected"
```

---

## Capacity Planning Guidelines

### Storage Tier Projections

**Calculate growth rate:**

```promql
# Storage growth rate (GB/day)
(sigmavault_storage_utilization_bytes - sigmoid_vault_utilization_bytes_1d_ago) / 1024 / 1024 / 1024
```

**Project capacity needs (6 months):**

```
Current capacity:          1000 GB
Current utilization:       650 GB (65%)
Growth rate:               10 GB/day
6-month growth:            10 * 180 = 1800 GB
Projected utilization:     650 + 1800 = 2450 GB

Recommended capacity:      2450 GB / 0.75 = 3266 GB (75% target)
Expansion needed:          3266 - 1000 = 2266 GB
```

### Cost Projections

**Annual cost forecast:**

```python
# Current monthly cost
current_monthly = increase(sigmavault_storage_cost_usd[30d])

# Projected annual cost (assuming 10% growth)
months_remaining = 12 - (current_month)
annual_cost = (current_monthly * growth_factor ** months_remaining) * 12
```

---

## Cost Optimization Tips

### 1. Right-Sizing Storage Tiers

```
Audit object access patterns:
- Accessed daily?         → Use hot storage
- Accessed weekly?        → Use warm storage
- Accessed monthly?       → Use cold storage
- Never accessed?         → Move to archive/delete

Savings opportunity: Moving 100GB cold → archive saves $0.30/month
```

### 2. Batch Operations

```python
# Batch stores reduce per-operation cost
# Cost without batching: 1000 ops * $0.0005 = $0.50
# Cost with batching:    1 batch * $0.0005 = $0.0005
# Savings:               99.9%
```

### 3. Lifecycle Policies

```python
# Example: Auto-archive old data
class LifecyclePolicy:
    def apply(self, object_key, last_accessed_date):
        days_old = (now - last_accessed_date).days

        if days_old < 30:
            return 'hot'
        elif days_old < 90:
            return 'warm'
        elif days_old < 365:
            return 'cold'
        else:
            return 'archive'  # or delete
```

### 4. Compression

```python
# Compressing 500GB data with 3:1 ratio
original_size = 500  # GB
compressed_size = 500 / 3  # ~167 GB
monthly_savings = (original_size - compressed_size) * 0.023
# Savings: (500 - 167) * $0.023 = $7.60/month
```

---

## Troubleshooting Guide

### High Latency

**Symptoms:** P95 latency > 1s

**Diagnostic Queries:**

```promql
# By operation type
histogram_quantile(0.95, rate(sigmavault_storage_operation_duration_seconds_bucket[5m])) by (operation_type)

# By data size
histogram_quantile(0.95, rate(sigmavault_storage_operation_duration_seconds_bucket[5m])) by (le=">1MB")

# Queue depth correlation
sigmavault_queue_depth
```

**Solutions:**

1. Check queue depth - add capacity if > 1000
2. Check storage tier utilization - expand if > 80%
3. Monitor network throughput - check for saturation
4. Analyze data size distribution - batch small ops

---

### High Error Rate

**Symptoms:** Error rate > 5%

**Diagnostic Queries:**

```promql
# Error breakdown
rate(sigmavault_storage_errors_total[5m]) by (error_type, operation_type)

# Recent error trend
increase(sigmavault_storage_errors_total[1h])
```

**Solutions by Error Type:**

| Error Type | Cause                  | Solution                         |
| ---------- | ---------------------- | -------------------------------- |
| timeout    | Slow backend/network   | Check network health, scale up   |
| permission | Auth/policy issue      | Review IAM policies              |
| corruption | Data integrity failure | Verify checksums, restore backup |
| network    | Network connectivity   | Check DNS, routes, firewall      |
| backend    | Storage backend issue  | Contact provider, check status   |

---

### Storage Capacity Issues

**Symptoms:** Utilization > 80%

**Diagnostic Queries:**

```promql
# Utilization by tier
(sigmavault_storage_utilization_bytes / sigmavault_storage_capacity_bytes) by (storage_class)

# Growth rate
rate(sigmavault_storage_utilization_bytes[24h]) / 1024 / 1024

# Time to full
(sigmavault_storage_capacity_bytes - sigmavault_storage_utilization_bytes) /
(rate(sigmavault_storage_utilization_bytes[24h])) / 86400
```

**Solutions:**

1. Delete old data
2. Migrate to lower-cost tiers
3. Expand capacity
4. Enable compression

---

### Unexpected Cost Growth

**Symptoms:** Monthly costs up 50%+ vs. baseline

**Diagnostic Queries:**

```promql
# Cost breakdown
increase(sigmavault_storage_cost_usd[30d]) by (cost_type, storage_class)

# Operation cost trend
increase(sigmavault_operation_cost_usd[24h])

# Transfer cost breakdown
increase(sigmavault_transfer_cost_usd[24h]) by (transfer_type)
```

**Solutions:**

1. Audit operation volume - batch if possible
2. Check transfer costs - optimize egress
3. Review storage growth - apply lifecycle
4. Verify tier distribution - right-size

---

## Integration Checklist

- [ ] Deploy Prometheus scrape config for ΣVAULT endpoint (`:9091/metrics`)
- [ ] Configure retention policy (minimum 30 days for cost tracking)
- [ ] Create Grafana dashboards (see GRAFANA_DASHBOARD_GUIDE.md)
- [ ] Set up alert rules (copy from Alert Rules section)
- [ ] Configure Slack/PagerDuty for critical alerts
- [ ] Test metrics collection in staging
- [ ] Document cost center codes for your organization
- [ ] Train team on reading cost metrics
- [ ] Set up monthly cost reporting
- [ ] Review and optimize storage tiers quarterly

---

## References

- [ΣVAULT Documentation](../../../sigmavault/README.md)
- [Prometheus Query Language](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [Histogram Quantile Function](https://prometheus.io/docs/prometheus/latest/querying/functions/#histogram_quantile)
- [AWS Storage Pricing](https://aws.amazon.com/s3/pricing/)
- [SLA Definitions](https://en.wikipedia.org/wiki/Service-level_agreement)

---

**Version:** 2.0  
**Last Updated:** December 16, 2025  
**Maintained By:** @SCRIBE
