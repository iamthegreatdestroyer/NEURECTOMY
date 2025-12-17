# ΣVAULT Storage Metrics - Prometheus Configuration & Implementation

## 1. PROMETHEUS SCRAPE CONFIGURATION

```yaml
# prometheus.yml
global:
  scrape_interval: 10s # Collect metrics every 10 seconds
  evaluation_interval: 10s
  external_labels:
    cluster: "σvault-prod"
    service: "storage-service"

scrape_configs:
  # ΣVAULT Storage Service Metrics
  - job_name: "svault-storage"
    static_configs:
      - targets: ["localhost:9090"]
    metric_path: "/metrics"
    scrape_interval: 10s
    scrape_timeout: 5s
    relabel_configs:
      # Add service labels
      - source_labels: [__address__]
        target_label: instance
      - source_labels: []
        target_label: service
        replacement: "svault-storage"

  # ΣVAULT Billing/Cost Attribution Agent
  - job_name: "svault-billing"
    static_configs:
      - targets: ["localhost:9091"]
    scrape_interval: 60s # Less frequent for batch jobs

  # ΣVAULT Key Management (HSM Operations)
  - job_name: "svault-keymgmt"
    static_configs:
      - targets: ["localhost:9092"]
    scrape_interval: 30s

# Remote write to long-term storage
remote_write:
  - url: http://victoriametrics:8428/api/v1/write
    queue_config:
      capacity: 100000
      max_shards: 10
    write_relabel_configs:
      # Keep cost-related metrics longer
      - source_labels: [__name__]
        regex: "svault_.*_cost_usd"
        action: keep

# Remote read for historical queries
remote_read:
  - url: http://victoriametrics:8428/api/v1/read
    read_recent: true
```

## 2. RECORDING RULES (Pre-calculated Metrics)

```yaml
# recording_rules.yml
groups:
  - name: svault_cost_aggregation
    interval: 1m
    rules:
      # Hourly Cost Rollup
      - record: "svault:storage_cost:hourly"
        expr: |
          sum by (cost_center, storage_class) (
            increase(svault_storage_cost_usd[1h])
          )

      # Daily Cost Rollup
      - record: "svault:total_cost:daily"
        expr: |
          sum by (cost_center) (
            increase(svault_total_cost_usd[24h])
          )

      # Monthly Cost Forecast (Linear Extrapolation)
      - record: "svault:monthly_cost_forecast"
        expr: |
          (increase(svault_total_cost_usd[1h])) * 730

      # Cost per GB Stored
      - record: "svault:cost_per_gb:instantaneous"
        expr: |
          sum by (cost_center, storage_class) (svault_storage_cost_usd) /
          (sum by (cost_center, storage_class) (svault_storage_utilization_bytes) / 1e9)

      # Cost per Operation
      - record: "svault:cost_per_operation:hourly"
        expr: |
          increase(svault_operation_cost_usd[1h]) /
          increase(svault_store_operations_total[1h])

      # Storage Efficiency (Utilization / Capacity)
      - record: "svault:storage_efficiency"
        expr: |
          sum by (storage_class) (svault_storage_utilization_bytes) /
          sum by (storage_class) (svault_storage_capacity_bytes) * 100

      # Cost Anomaly Detection (Deviation from 7-day average)
      - record: "svault:cost_anomaly_score"
        expr: |
          abs(rate(svault_total_cost_usd[5m]) - 
              avg_over_time(rate(svault_total_cost_usd[5m])[7d:5m])) >
          (stddev_over_time(rate(svault_total_cost_usd[5m])[7d:5m]) * 2)

  - name: svault_performance_slas
    interval: 1m
    rules:
      # Operation Success Rate
      - record: "svault:operation_success_rate"
        expr: |
          100 * (
            sum by (operation_type, cost_center) (
              rate(svault_store_operations_total{result="success"}[5m])
            ) /
            sum by (operation_type, cost_center) (
              rate(svault_store_operations_total[5m])
            )
          )

      # P99 Latency for SLA Tracking
      - record: "svault:store_latency:p99"
        expr: |
          histogram_quantile(0.99, 
            sum by (le, storage_class) (
              rate(svault_store_latency_seconds_bucket[5m])
            )
          )

      # P95 Latency for SLA Tracking
      - record: "svault:store_latency:p95"
        expr: |
          histogram_quantile(0.95,
            sum by (le, storage_class) (
              rate(svault_store_latency_seconds_bucket[5m])
            )
          )

  - name: svault_financial_reporting
    interval: 5m
    rules:
      # Cost breakdown by dimension
      - record: "svault:cost_by_operation_type:5m"
        expr: |
          sum by (operation_type, cost_center) (
            increase(svault_operation_cost_usd[5m])
          )

      # Cost trend (Hour over Hour)
      - record: "svault:cost_hoh_growth"
        expr: |
          (
            sum(increase(svault_total_cost_usd[1h])) -
            sum(increase(svault_total_cost_usd[1h] offset 1h))
          ) / sum(increase(svault_total_cost_usd[1h] offset 1h)) * 100

      # Cost trend (Week over Week)
      - record: "svault:cost_wow_growth"
        expr: |
          (
            sum(increase(svault_total_cost_usd[7d])) -
            sum(increase(svault_total_cost_usd[7d] offset 7d))
          ) / sum(increase(svault_total_cost_usd[7d] offset 7d)) * 100
```

## 3. ALERTING RULES

```yaml
# alerts.yml
groups:
  - name: svault_cost_alerts
    rules:
      # Alert: Unusual Cost Spike (> 50% deviation from average)
      - alert: SVaultCostSpike
        expr: |
          abs(rate(svault_total_cost_usd[5m]) - 
              avg_over_time(rate(svault_total_cost_usd[5m])[7d:5m])) >
          (stddev_over_time(rate(svault_total_cost_usd[5m])[7d:5m]) * 2)
        for: 5m
        labels:
          severity: warning
          component: billing
        annotations:
          summary: "Unusual cost spike detected"
          description: "Cost for {{ $labels.cost_center }} deviated >2σ from 7-day average"
          dashboard: "http://grafana:3000/d/svault-costs"

      # Alert: Monthly Cost Forecast Exceeds Budget (if budget defined)
      - alert: SVaultBudgetAtRisk
        expr: |
          (increase(svault_total_cost_usd[1h])) * 730 > 
          (budget_usd * 0.80)  # Alert at 80% of budget
        for: 15m
        labels:
          severity: critical
          component: billing
        annotations:
          summary: "Monthly cost forecast exceeds 80% of budget"
          description: "Projected monthly cost for {{ $labels.cost_center }}: ${{ $value }}"

      # Alert: Storage Capacity Near Full (> 90%)
      - alert: SVaultStorageCapacityWarning
        expr: |
          svault_storage_utilization_percent > 90
        for: 10m
        labels:
          severity: warning
          component: capacity
        annotations:
          summary: "Storage capacity {{ $labels.storage_class }} at {{ $value }}%"
          description: "Immediate action required to avoid service disruption"

      # Alert: Replication Factor Below Requirement
      - alert: SVaultReplicationFailure
        expr: |
          svault_replication_success_rate{replication_factor="3"} < 99.9
        for: 5m
        labels:
          severity: critical
          component: reliability
        annotations:
          summary: "Replication failure detected"
          description: "{{ $labels.cost_center }} has only {{ $value }}% replicated"

      # Alert: Data Integrity Check Failed
      - alert: SVaultIntegrityCheckFailed
        expr: |
          rate(svault_integrity_check_failures[5m]) > 0
        for: 1m
        labels:
          severity: critical
          component: data_integrity
        annotations:
          summary: "Data integrity check failed"
          description: "{{ $value }} integrity failures detected in last 5 minutes"

      # Alert: SLA Breach (Availability < SLA Target)
      - alert: SVaultSLABreach
        expr: |
          svault_availability_percent{sla_tier="standard"} < 99.9
        for: 5m
        labels:
          severity: critical
          component: sla
        annotations:
          summary: "SLA breach for {{ $labels.cost_center }}"
          description: "Availability: {{ $value }}%, Target: 99.9%"

  - name: svault_performance_alerts
    rules:
      # Alert: P99 Latency Exceeds Threshold
      - alert: SVaultHighLatency
        expr: |
          svault_store_latency_p99 > 5  # 5 seconds threshold
        for: 5m
        labels:
          severity: warning
          component: performance
        annotations:
          summary: "Store operation P99 latency {{ $value }}s"
          description: "{{ $labels.storage_class }} storage class affected"

      # Alert: Operation Failure Rate Too High
      - alert: SVaultHighFailureRate
        expr: |
          (1 - svault_operation_success_rate) > 0.01  # > 1% failure
        for: 5m
        labels:
          severity: critical
          component: reliability
        annotations:
          summary: "Operation failure rate {{ $value }}% (threshold: 1%)"
          description: "{{ $labels.operation_type }} operations failing"
```

## 4. GRAFANA DASHBOARD DEFINITIONS

```json
{
  "dashboard": {
    "title": "ΣVAULT Cost & Financial Tracking",
    "uid": "svault-financial",
    "tags": ["storage", "billing", "financial"],
    "timezone": "UTC",
    "refresh": "30s",
    "panels": [
      {
        "title": "Daily Cost Trend",
        "type": "graph",
        "targets": [
          {
            "expr": "sum by (cost_center) (increase(svault_total_cost_usd[24h]))",
            "legendFormat": "{{ cost_center }}"
          }
        ],
        "yaxes": [
          {
            "label": "Cost (USD)",
            "format": "short"
          }
        ]
      },
      {
        "title": "Cost Breakdown by Dimension",
        "type": "piechart",
        "targets": [
          {
            "expr": "sum by (cost_dimension) (rate(svault_total_cost_usd[1h]))",
            "legendFormat": "{{ cost_dimension }}"
          }
        ]
      },
      {
        "title": "Storage Utilization vs Cost",
        "type": "graph",
        "targets": [
          {
            "expr": "sum by (storage_class) (svault_storage_utilization_bytes / 1e9)",
            "legendFormat": "{{ storage_class }} (GB)"
          },
          {
            "expr": "sum by (storage_class) (rate(svault_storage_cost_usd[1h]))",
            "legendFormat": "{{ storage_class }} Cost (USD/h)"
          }
        ]
      },
      {
        "title": "Cost Per GB Stored",
        "type": "stat",
        "targets": [
          {
            "expr": "svault:cost_per_gb:instantaneous"
          }
        ],
        "unit": "USD"
      },
      {
        "title": "Monthly Cost Forecast",
        "type": "stat",
        "targets": [
          {
            "expr": "svault:monthly_cost_forecast"
          }
        ],
        "unit": "USD"
      },
      {
        "title": "Cost Anomaly Detection",
        "type": "graph",
        "targets": [
          {
            "expr": "svault:cost_anomaly_score"
          }
        ]
      },
      {
        "title": "Storage Class Distribution",
        "type": "graph",
        "targets": [
          {
            "expr": "sum by (storage_class) (svault_storage_class_distribution_bytes / 1e9)",
            "legendFormat": "{{ storage_class }}"
          }
        ],
        "yaxes": [
          {
            "label": "Bytes (GB)"
          }
        ]
      },
      {
        "title": "Cost Optimization Opportunity",
        "type": "gauge",
        "targets": [
          {
            "expr": "sum(svault_cold_storage_potential_savings_usd)"
          }
        ],
        "unit": "USD"
      },
      {
        "title": "Operation Success Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "svault:operation_success_rate"
          }
        ],
        "unit": "percent"
      },
      {
        "title": "Top 10 Cost Centers",
        "type": "table",
        "targets": [
          {
            "expr": "topk(10, sum by (cost_center) (rate(svault_total_cost_usd[1h])))"
          }
        ]
      }
    ]
  }
}
```

## 5. IMPLEMENTATION CODE EXAMPLES

### Python Prometheus Client Example

```python
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from datetime import datetime
from typing import Dict, Any
import time

class SVaultMetrics:
    def __init__(self, registry=None):
        self.registry = registry or CollectorRegistry()
        self._init_counters()
        self._init_histograms()
        self._init_gauges()

    def _init_counters(self):
        """Initialize counter metrics"""
        self.store_ops_total = Counter(
            'svault_store_operations_total',
            'Total store operations completed',
            ['operation_id', 'cost_center', 'storage_class', 'region',
             'encryption_type', 'replication_factor', 'result', 'error_code'],
            registry=self.registry
        )

        self.store_cost = Counter(
            'svault_operation_cost_usd',
            'Cumulative cost of operations',
            ['operation_type', 'cost_center', 'region', 'storage_class'],
            registry=self.registry
        )

        self.audit_log_ops = Counter(
            'svault_audit_log_operations_total',
            'Audit trail entries (immutable)',
            ['operation_type', 'cost_center', 'action', 'user_type'],
            registry=self.registry
        )

    def _init_histograms(self):
        """Initialize histogram metrics"""
        self.store_latency = Histogram(
            'svault_store_latency_seconds',
            'Store operation latency distribution',
            ['operation_id', 'cost_center', 'storage_class',
             'object_size_bucket', 'encryption_type'],
            buckets=(0.001, 0.005, 0.010, 0.050, 0.100, 0.250, 0.500,
                    1.0, 2.5, 5.0, 10.0, float('inf')),
            registry=self.registry
        )

        self.object_size = Histogram(
            'svault_object_size_bytes',
            'Object size distribution',
            ['storage_class', 'cost_center', 'object_type'],
            buckets=(1000, 10000, 100000, 1000000, 10000000, 100000000,
                    1000000000, 10000000000, 100000000000, float('inf')),
            registry=self.registry
        )

    def _init_gauges(self):
        """Initialize gauge metrics"""
        self.storage_capacity = Gauge(
            'svault_storage_capacity_bytes',
            'Total provisioned storage capacity',
            ['storage_class', 'region', 'cost_center'],
            registry=self.registry
        )

        self.storage_utilization = Gauge(
            'svault_storage_utilization_bytes',
            'Current bytes stored',
            ['storage_class', 'region', 'cost_center'],
            registry=self.registry
        )

        self.total_objects = Gauge(
            'svault_objects_total',
            'Total number of stored objects',
            ['storage_class', 'cost_center'],
            registry=self.registry
        )

        self.monthly_cost_forecast = Gauge(
            'svault_monthly_cost_forecast_usd',
            'Projected monthly cost',
            ['cost_center'],
            registry=self.registry
        )

    def record_store_operation(
        self,
        operation_id: str,
        cost_center: str,
        storage_class: str,
        region: str,
        encryption_type: str,
        replication_factor: int,
        data_size_bytes: int,
        latency_seconds: float,
        result: str = 'success',
        error_code: str = '0',
        cost_usd: float = None
    ):
        """Record a store operation with all metrics"""

        # Record operation count
        self.store_ops_total.labels(
            operation_id=operation_id,
            cost_center=cost_center,
            storage_class=storage_class,
            region=region,
            encryption_type=encryption_type,
            replication_factor=str(replication_factor),
            result=result,
            error_code=error_code
        ).inc()

        # Record latency
        self.store_latency.labels(
            operation_id=operation_id,
            cost_center=cost_center,
            storage_class=storage_class,
            object_size_bucket=self._get_size_bucket(data_size_bytes),
            encryption_type=encryption_type
        ).observe(latency_seconds)

        # Record object size
        self.object_size.labels(
            storage_class=storage_class,
            cost_center=cost_center,
            object_type='binary'
        ).observe(data_size_bytes)

        # Record cost (if calculated)
        if cost_usd is not None:
            self.store_cost.labels(
                operation_type='store',
                cost_center=cost_center,
                region=region,
                storage_class=storage_class
            ).inc(cost_usd)

        # Record audit trail (immutable)
        self.audit_log_ops.labels(
            operation_type='store',
            cost_center=cost_center,
            action='create',
            user_type='service_account'
        ).inc()

    @staticmethod
    def _get_size_bucket(size_bytes: int) -> str:
        """Categorize object size into buckets"""
        if size_bytes < 1000000:  # 1MB
            return 'small'
        elif size_bytes < 100000000:  # 100MB
            return 'medium'
        elif size_bytes < 1000000000:  # 1GB
            return 'large'
        else:
            return 'huge'

    def update_storage_utilization(
        self,
        storage_class: str,
        region: str,
        cost_center: str,
        bytes_used: int,
        bytes_capacity: int
    ):
        """Update storage utilization metrics"""
        self.storage_utilization.labels(
            storage_class=storage_class,
            region=region,
            cost_center=cost_center
        ).set(bytes_used)

        self.storage_capacity.labels(
            storage_class=storage_class,
            region=region,
            cost_center=cost_center
        ).set(bytes_capacity)


# Example usage
if __name__ == '__main__':
    metrics = SVaultMetrics()

    # Simulate store operation
    import uuid

    operation_id = str(uuid.uuid4())[:8]
    cost_usd = 0.0001  # Cost calculation

    metrics.record_store_operation(
        operation_id=operation_id,
        cost_center='acme_corp/engineering/ml_pipeline',
        storage_class='warm',
        region='us-east-1',
        encryption_type='AES256',
        replication_factor=3,
        data_size_bytes=1073741824,  # 1GB
        latency_seconds=0.125,
        result='success',
        cost_usd=cost_usd
    )

    # Update storage metrics
    metrics.update_storage_utilization(
        storage_class='warm',
        region='us-east-1',
        cost_center='acme_corp/engineering/ml_pipeline',
        bytes_used=10737418240,  # 10GB
        bytes_capacity=107374182400  # 100GB
    )

    # Export metrics
    print(generate_latest(metrics.registry).decode('utf-8'))
```

## 6. COST CALCULATION SERVICE

```python
# cost_calculator.py
from dataclasses import dataclass
from typing import Dict
from enum import Enum

class StorageClass(Enum):
    HOT = 'hot'
    WARM = 'warm'
    COLD = 'cold'
    ARCHIVE = 'archive'

class OperationType(Enum):
    STORE = 'store'
    RETRIEVE = 'retrieve'
    DELETE = 'delete'
    SNAPSHOT = 'snapshot'

@dataclass
class CostModel:
    """Financial cost model for ΣVAULT"""

    storage_costs_per_gb_month: Dict[StorageClass, float] = None
    operation_costs: Dict[OperationType, Dict[StorageClass, float]] = None
    transfer_costs_per_gb: Dict[str, float] = None
    encryption_costs: Dict[str, float] = None

    def __post_init__(self):
        if self.storage_costs_per_gb_month is None:
            self.storage_costs_per_gb_month = {
                StorageClass.HOT: 0.023,
                StorageClass.WARM: 0.0115,
                StorageClass.COLD: 0.00575,
                StorageClass.ARCHIVE: 0.00115
            }

        if self.operation_costs is None:
            self.operation_costs = {
                OperationType.STORE: {
                    StorageClass.HOT: 0.0001,
                    StorageClass.WARM: 0.0001,
                    StorageClass.COLD: 0.0001,
                    StorageClass.ARCHIVE: 0.001
                },
                OperationType.RETRIEVE: {
                    StorageClass.HOT: 0.00001,
                    StorageClass.WARM: 0.00001,
                    StorageClass.COLD: 0.00001,
                    StorageClass.ARCHIVE: 0.005
                },
                OperationType.DELETE: {
                    StorageClass.HOT: 0.00001,
                    StorageClass.WARM: 0.00001,
                    StorageClass.COLD: 0.00001,
                    StorageClass.ARCHIVE: 0.00001
                }
            }

        if self.transfer_costs_per_gb is None:
            self.transfer_costs_per_gb = {
                'ingress': 0.0,
                'egress': 0.01,
                'cross_region': 0.02,
                'cross_account': 0.02
            }

        if self.encryption_costs is None:
            self.encryption_costs = {
                'key_derivation': 0.00001,
                'key_rotation': 0.001,
                'hsm_operation': 0.0001,
                'crypto_per_second': 0.000001
            }

class CostCalculator:
    def __init__(self, cost_model: CostModel):
        self.model = cost_model

    def calculate_storage_cost(
        self,
        storage_class: StorageClass,
        bytes_stored: int,
        days: int = 30
    ) -> float:
        """Calculate monthly storage cost"""
        gb_stored = bytes_stored / 1e9
        monthly_rate = self.model.storage_costs_per_gb_month[storage_class]
        days_factor = days / 30.0
        return gb_stored * monthly_rate * days_factor

    def calculate_operation_cost(
        self,
        operation_type: OperationType,
        storage_class: StorageClass,
        data_size_bytes: int,
        latency_seconds: float,
        encryption_enabled: bool = False,
        cross_region: bool = False
    ) -> float:
        """Calculate operation cost with all components"""

        # Base operation cost
        base_cost = self.model.operation_costs[operation_type][storage_class]

        # Data transfer component
        transfer_cost = 0.0
        if operation_type == OperationType.RETRIEVE:
            if cross_region:
                transfer_cost = (data_size_bytes / 1e9) * self.model.transfer_costs_per_gb['cross_region']
            else:
                transfer_cost = (data_size_bytes / 1e9) * self.model.transfer_costs_per_gb['egress']

        # Encryption cost
        encryption_cost = 0.0
        if encryption_enabled:
            # Crypto ops per second (estimated)
            estimated_ops = latency_seconds * 1000  # Rough estimate
            encryption_cost = estimated_ops * self.model.encryption_costs['crypto_per_second']

        total_cost = base_cost + transfer_cost + encryption_cost
        return round(total_cost, 6)  # Precision to microseconds

    def calculate_monthly_cost_forecast(
        self,
        daily_operations: Dict[OperationType, int],
        avg_data_size_bytes: int,
        avg_latency_seconds: float,
        storage_class: StorageClass,
        bytes_stored: int
    ) -> float:
        """Forecast total monthly cost"""

        # Storage cost
        storage_cost = self.calculate_storage_cost(storage_class, bytes_stored, 30)

        # Operation costs
        operation_cost = 0.0
        for op_type, count in daily_operations.items():
            monthly_count = count * 30
            per_op_cost = self.calculate_operation_cost(
                op_type,
                storage_class,
                avg_data_size_bytes,
                avg_latency_seconds
            )
            operation_cost += monthly_count * per_op_cost

        total = storage_cost + operation_cost
        return round(total, 2)


# Example usage
if __name__ == '__main__':
    cost_model = CostModel()
    calculator = CostCalculator(cost_model)

    # Calculate operation cost
    op_cost = calculator.calculate_operation_cost(
        operation_type=OperationType.STORE,
        storage_class=StorageClass.WARM,
        data_size_bytes=1073741824,  # 1GB
        latency_seconds=0.125,
        encryption_enabled=True,
        cross_region=False
    )
    print(f"Operation cost: ${op_cost:.6f}")

    # Calculate monthly forecast
    forecast = calculator.calculate_monthly_cost_forecast(
        daily_operations={
            OperationType.STORE: 1000,
            OperationType.RETRIEVE: 5000,
            OperationType.DELETE: 100
        },
        avg_data_size_bytes=1073741824,
        avg_latency_seconds=0.125,
        storage_class=StorageClass.WARM,
        bytes_stored=10737418240  # 10GB
    )
    print(f"Forecasted monthly cost: ${forecast:.2f}")
```

---

## DEPLOYMENT CHECKLIST

- [ ] Deploy Prometheus with scrape configs
- [ ] Activate recording rules
- [ ] Configure alert rules with notification channels
- [ ] Deploy Grafana dashboards
- [ ] Set up remote write to long-term storage
- [ ] Initialize financial ledger database
- [ ] Deploy cost calculator service
- [ ] Configure billing event stream (Kafka)
- [ ] Set up monthly reconciliation job
- [ ] Enable audit trail collection (immutable)
- [ ] Configure SLA refund calculations
- [ ] Create billing reports (scheduled daily/weekly/monthly)
- [ ] Train on cost queries and analysis
