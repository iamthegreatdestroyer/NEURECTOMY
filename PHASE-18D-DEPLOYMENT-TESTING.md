# Phase 18D: OpenTelemetry + Jaeger Deployment & Testing Guide

**Status:** Complete Implementation Plan  
**Date:** 2024-2025  
**Target:** Neurectomy Phase 18D

---

## 📋 Quick Start (5 Minutes)

```bash
# 1. Deploy Jaeger stack
kubectl apply -f deploy/k8s/13-jaeger-configmap.yaml
kubectl apply -f deploy/k8s/14-jaeger-deployment.yaml
kubectl apply -f deploy/k8s/15-jaeger-services.yaml
kubectl apply -f deploy/k8s/16-jaeger-rbac.yaml

# 2. Deploy OpenTelemetry Collector
kubectl apply -f deploy/k8s/17-otel-collector.yaml

# 3. Verify deployment
kubectl get pods -n monitoring | grep jaeger
kubectl get pods -n monitoring | grep otel

# 4. Access Jaeger UI
kubectl port-forward -n monitoring svc/jaeger-query 16686:16686
# Open: http://localhost:16686
```

---

## 🚀 Full Deployment Procedure

### Phase 1: Prerequisites (Day 1)

**Step 1.1: Verify Kubernetes Cluster**

```bash
# Check cluster version
kubectl version --short

# Verify namespaces
kubectl get namespace monitoring
# If not exists:
kubectl create namespace monitoring
kubectl label namespace monitoring name=monitoring
```

**Step 1.2: Create Storage Classes**

```bash
cat > deploy/k8s/storage-class-fast-ssd.yaml << 'EOF'
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
EOF

kubectl apply -f deploy/k8s/storage-class-fast-ssd.yaml
```

**Step 1.3: Verify Prometheus Stack**

```bash
# Ensure Prometheus is running
kubectl get deployment -n monitoring prometheus

# Check if ServiceMonitor CRD exists
kubectl get crd servicemonitors.monitoring.coreos.com
```

---

### Phase 2: Deploy Jaeger (Day 2)

**Step 2.1: Deploy Jaeger ConfigMap**

```bash
kubectl apply -f deploy/k8s/13-jaeger-configmap.yaml

# Verify
kubectl get cm -n monitoring jaeger-config
kubectl describe cm -n monitoring jaeger-config | head -30
```

**Step 2.2: Deploy Jaeger Components**

```bash
# Deploy all Jaeger components (agent, collector, query)
kubectl apply -f deploy/k8s/14-jaeger-deployment.yaml

# Wait for rollout
kubectl rollout status statefulset/jaeger-collector -n monitoring
kubectl rollout status deployment/jaeger-agent -n monitoring
kubectl rollout status deployment/jaeger-query -n monitoring

# Verify all pods running
kubectl get pods -n monitoring -l app=jaeger
```

**Step 2.3: Create Services**

```bash
kubectl apply -f deploy/k8s/15-jaeger-services.yaml

# Verify services
kubectl get svc -n monitoring -l app=jaeger
```

**Step 2.4: Create RBAC and Security**

```bash
kubectl apply -f deploy/k8s/16-jaeger-rbac.yaml

# Verify
kubectl get sa -n monitoring jaeger
kubectl get clusterrole jaeger
```

**Step 2.5: Verify Jaeger Deployment**

```bash
# Check all pods are running
kubectl get pods -n monitoring | grep jaeger

# Check persistent volumes
kubectl get pvc -n monitoring | grep jaeger

# Port-forward to access UI
kubectl port-forward -n monitoring svc/jaeger-query 16686:16686

# In another terminal, open: http://localhost:16686
```

---

### Phase 3: Deploy OpenTelemetry Collector (Day 2-3)

**Step 3.1: Deploy OTEL Collector**

```bash
kubectl apply -f deploy/k8s/17-otel-collector.yaml

# Wait for DaemonSet rollout
kubectl rollout status daemonset/otel-collector -n monitoring

# Verify pods on all nodes
kubectl get pods -n monitoring -l app=otel-collector -o wide
```

**Step 3.2: Verify OTEL Collector is Receiving Traces**

```bash
# Check OTEL Collector logs
kubectl logs -n monitoring -l app=otel-collector --tail=50

# Look for "receiver_accepted_spans" in output
```

**Step 3.3: Verify Jaeger is Receiving Traces**

```bash
# Check Jaeger Collector logs
kubectl logs -n monitoring -l app=jaeger,component=collector --tail=50

# Look for "received_spans" in output
```

---

### Phase 4: Service Instrumentation (Day 3-4)

**Step 4.1: Install OTEL SDK in Services**

For RYOT (Python):

```bash
# In ryot service directory
pip install -r requirements-otel.txt

# Create requirements-otel.txt:
cat > ryot/requirements-otel.txt << 'EOF'
opentelemetry-api==1.20.0
opentelemetry-sdk==1.20.0
opentelemetry-exporter-jaeger-thrift==1.20.0
opentelemetry-exporter-prometheus==0.41b0
opentelemetry-instrumentation==0.41b0
opentelemetry-instrumentation-flask==0.41b0
opentelemetry-instrumentation-requests==0.41b0
opentelemetry-instrumentation-sqlalchemy==0.41b0
opentelemetry-instrumentation-redis==0.41b0
EOF

pip install -r requirements-otel.txt
```

**Step 4.2: Initialize Tracing in Each Service**

```python
# In ryot/app.py
from ryot.monitoring.tracing import RyotTracingConfig

# Initialize before creating Flask app
tracer = RyotTracingConfig.initialize(
    service_name="ryot",
    jaeger_agent_host="otel-collector.monitoring.svc.cluster.local",
    jaeger_agent_port=14250
)

app = create_app()
```

**Step 4.3: Update Service Deployment Manifests**

```yaml
# In deploy/k8s/neurectomy-deployments.yaml
env:
  - name: OTEL_EXPORTER_JAEGER_ENDPOINT
    value: "http://jaeger-collector.monitoring.svc.cluster.local:14250"
  - name: OTEL_EXPORTER_JAEGER_AGENT_HOST
    value: "otel-collector.monitoring.svc.cluster.local"
  - name: OTEL_EXPORTER_JAEGER_AGENT_PORT
    value: "6831"
  - name: OTEL_SERVICE_NAME
    value: "ryot"
  - name: OTEL_TRACES_EXPORTER
    value: "jaeger"
```

---

### Phase 5: Grafana Integration (Day 4-5)

**Step 5.1: Add Jaeger Data Source to Grafana**

```bash
# Get Grafana admin password
GRAFANA_PASSWORD=$(kubectl get secret -n monitoring grafana -o jsonpath="{.data.admin-password}" | base64 -d)

# Port-forward to Grafana
kubectl port-forward -n monitoring svc/grafana 3000:3000

# Open http://localhost:3000
# Login with admin / $GRAFANA_PASSWORD
```

**Step 5.2: Create Data Source**

```yaml
# In Grafana UI:
# Configuration → Data Sources → Add
# - Name: Jaeger - Traces
# - Type: Jaeger
# - URL: http://jaeger-query.monitoring.svc.cluster.local:16686
# - Save & Test
```

**Step 5.3: Import Dashboards**

```bash
# Copy dashboard JSON files to Grafana provisioning
kubectl create configmap grafana-dashboard-traces \
  --from-file=deploy/k8s/grafana-dashboards/ \
  -n monitoring

# Restart Grafana
kubectl rollout restart deployment/grafana -n monitoring
```

---

## 🧪 Testing & Validation

### Test 1: Generate Sample Traces

**File:** `tests/distributed-tracing/test-trace-generation.py`

```python
#!/usr/bin/env python
"""Generate sample traces to test Jaeger integration."""

import sys
import time
import random
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource

def generate_traces(service_name: str, num_traces: int = 10):
    """Generate sample traces."""

    # Setup
    jaeger_exporter = JaegerExporter(
        agent_host_name="localhost",
        agent_port=6831,
    )

    trace_provider = TracerProvider(resource=Resource({SERVICE_NAME: service_name}))
    trace_provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
    trace.set_tracer_provider(trace_provider)

    tracer = trace.get_tracer(__name__)

    # Generate traces
    for i in range(num_traces):
        with tracer.start_as_current_span(f"operation_{i}") as span:
            span.set_attribute("request.id", f"req-{i}")
            span.set_attribute("user.id", f"user-{random.randint(1, 10)}")

            # Simulate child spans
            with tracer.start_as_current_span("database_query") as db_span:
                db_span.set_attribute("db.statement", "SELECT * FROM users")
                time.sleep(random.uniform(0.01, 0.1))

            with tracer.start_as_current_span("cache_lookup") as cache_span:
                cache_span.set_attribute("cache.key", f"key-{i}")
                cache_span.set_attribute("cache.hit", random.choice([True, False]))
                time.sleep(random.uniform(0.001, 0.01))

            print(f"Generated trace {i+1}/{num_traces}")
            time.sleep(0.5)

    print(f"Generated {num_traces} traces for {service_name}")

if __name__ == "__main__":
    services = ["ryot", "sigmalang", "sigmavault", "agents"]

    for service in services:
        print(f"\nGenerating traces for {service}...")
        generate_traces(service, num_traces=5)
        time.sleep(2)

    print("\n✅ All traces generated! Check Jaeger UI at http://localhost:16686")
```

**Run Test:**

```bash
# Port-forward Jaeger agent
kubectl port-forward -n monitoring svc/jaeger-agent 6831:6831/udp

# In another terminal
python tests/distributed-tracing/test-trace-generation.py

# Check Jaeger UI
open http://localhost:16686
```

---

### Test 2: Verify Trace Reception

```bash
# Check OTEL Collector received spans
kubectl logs -n monitoring -l app=otel-collector | grep "receiver_accepted_spans"

# Check Jaeger Collector stored spans
kubectl logs -n monitoring -l app=jaeger,component=collector | grep "received_spans"

# Query Jaeger for services
curl http://localhost:16686/api/services | jq .
```

**Expected Output:**

```json
{
  "data": ["ryot", "sigmalang", "sigmavault", "agents"]
}
```

---

### Test 3: Trace-to-Metrics Correlation

```promql
# In Prometheus:
# Query: span_duration_seconds
histogram_quantile(0.95,
  sum(rate(span_duration_seconds_bucket[5m])) by (service, operation, le)
)

# Should return data for all services
```

---

### Test 4: Sampling Validation

```bash
# Check sampling rates
kubectl exec -n monitoring jaeger-collector-0 -c jaeger-collector -- \
  curl http://localhost:14269/metrics | grep sampling

# Expected: metrics like jaeger_sampler_queries_total
```

---

### Test 5: Error Trace Capture

```bash
# Inject error trace
python -c "
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource

jaeger_exporter = JaegerExporter(agent_host_name='localhost', agent_port=6831)
trace_provider = TracerProvider(resource=Resource({SERVICE_NAME: 'test-error'}))
trace_provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
trace.set_tracer_provider(trace_provider)

tracer = trace.get_tracer(__name__)
with tracer.start_as_current_span('error_operation') as span:
    span.record_exception(Exception('Test error'))
    span.set_attribute('error', True)
"

# Check Jaeger UI - error trace should be visible with 100% sampling
```

---

## 📊 Monitoring the Monitoring System

### Key Metrics to Track

```promql
# Jaeger health
jaeger_collector_spans_received_total
jaeger_collector_traces_received_total
jaeger_sampler_queries_total

# OTEL Collector health
otelcontrib_receiver_accepted_spans{receiver="jaeger"}
otelcontrib_receiver_refused_spans{receiver="jaeger"}
otelcontrib_exporter_sent_spans{exporter="jaeger"}
otelcontrib_exporter_send_failed_spans{exporter="jaeger"}

# Trace processing
span_processing_latency_seconds
sampler_sampling_rate

# Storage
jaeger_storage_latency_seconds
badger_disk_size_bytes
```

### Alert Rules

```yaml
- alert: JaegerHighErrorRate
  expr: (rate(jaeger_collector_errors_total[5m]) / rate(jaeger_collector_operations_total[5m])) > 0.01
  annotations:
    summary: "Jaeger error rate > 1%"

- alert: JaegerStorageAlmostFull
  expr: (jaeger_storage_bytes_used / jaeger_storage_bytes_max) > 0.9
  annotations:
    summary: "Jaeger storage 90% full"

- alert: OTELCollectorHighMemory
  expr: container_memory_usage_bytes{pod=~"otel-collector-.*"} > 800000000
  annotations:
    summary: "OTEL Collector memory > 800MB"
```

---

## ✅ Verification Checklist

### Infrastructure

- [ ] Jaeger Agent running on all nodes (DaemonSet)
- [ ] Jaeger Collector StatefulSet with 3 replicas
- [ ] Jaeger Query deployment accessible
- [ ] OTEL Collector DaemonSet running
- [ ] Persistent volumes mounted correctly
- [ ] Service discovery working

### Integration

- [ ] Services sending traces to OTEL Collector
- [ ] OTEL Collector forwarding to Jaeger
- [ ] Jaeger storing traces in Badger DB
- [ ] Prometheus scraping Jaeger metrics
- [ ] Grafana connected to Jaeger data source

### Observability

- [ ] Sample traces visible in Jaeger UI
- [ ] Services appearing in Jaeger service list
- [ ] Trace latency distributions correct
- [ ] Error traces captured with 100% sampling
- [ ] Sampling rates matching policy
- [ ] Trace-to-metrics correlation working

### Performance

- [ ] Trace processing latency < 500ms
- [ ] No spans dropped due to memory limits
- [ ] Storage utilization healthy (< 80%)
- [ ] Query latency < 1 second
- [ ] Sampling reducing volume effectively

---

## 📚 Additional Resources

- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Jaeger Getting Started](https://www.jaegertracing.io/docs/latest/getting-started/)
- [OTEL Collector Configuration](https://opentelemetry.io/docs/reference/specification/protocol/exporter/)
- [Kubernetes Observability](https://kubernetes.io/docs/tasks/debug-application-cluster/)

---

## 🆘 Troubleshooting

### Issue: No traces appearing in Jaeger

**Solution:**

```bash
# Check OTEL Collector is running
kubectl get pods -n monitoring -l app=otel-collector

# Check Jaeger Agent is receiving traces
kubectl logs -n monitoring -l app=otel-collector | grep "receiver_accepted"

# Check network connectivity
kubectl exec -n monitoring otel-collector-xxx -- \
  curl http://jaeger-collector:14250/health
```

### Issue: High memory usage

**Solution:**

```bash
# Check memory limits in OTEL Collector
kubectl describe pod -n monitoring -l app=otel-collector | grep -i memory

# Reduce batch size in collector config
# Change send_batch_size: 512 → 256
kubectl edit cm -n monitoring otel-collector-config
```

### Issue: Traces disappearing after 72 hours

**Solution:**

- This is expected - traces are retained 72 hours in Badger DB
- For longer retention, export to object storage:
  ```yaml
  remote_storage:
    type: s3
    s3:
      bucket: neurectomy-traces-archive
      endpoint: s3.amazonaws.com
  ```

---

## 🎉 Success Criteria

✅ **Phase 18D Complete When:**

1. All 4 services (RYOT, ΣLANG, ΣVAULT, Agents) sending traces
2. Jaeger displaying service topology and traces
3. Sampling strategy reducing volume by 70%+
4. Trace-to-metric correlation working in Grafana
5. Error traces captured with 100% sampling
6. Query latency < 1 second
7. Storage efficiency > 80%
8. No dropped spans in 72-hour period
