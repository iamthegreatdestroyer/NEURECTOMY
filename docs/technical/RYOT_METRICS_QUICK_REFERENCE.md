# Ryot LLM Metrics: Quick Reference & Operations Guide

**Quick Links:**

- [Full Design Specification](RYOT_METRICS_DESIGN.md)
- [Implementation Guide](RYOT_METRICS_IMPLEMENTATION.md)

---

## 🚀 Quick Deployment

```bash
# 1. Copy metrics module
cp neurectomy/monitoring/ryot_metrics.py /your/project/path/

# 2. Add to FastAPI app
from neurectomy.api.middleware.metrics_middleware import RyotMetricsMiddleware
app.add_middleware(RyotMetricsMiddleware)

# 3. Export metrics endpoint
from neurectomy.api.endpoints.metrics import router
app.include_router(router)

# 4. Deploy Prometheus scrape config
# See config/prometheus.yml

# 5. Import metrics in inference code
from neurectomy.monitoring.ryot_metrics import InferenceMetricsContext
```

---

## 📊 Essential Prometheus Queries

### Real-Time Monitoring

```promql
# Request success rate (%)
(1 - (rate(ryot_inference_requests_total{status="error"}[5m]) /
      rate(ryot_inference_requests_total[5m]))) * 100

# Average latency
histogram_quantile(0.5, rate(ryot_inference_latency_seconds_bucket[5m]))

# P95 latency (SLO target: <500ms)
histogram_quantile(0.95, rate(ryot_inference_latency_seconds_bucket[5m]))

# Token generation rate
sum(rate(ryot_tokens_generated_total[1m])) by (model)

# GPU memory utilization
avg(ryot_gpu_memory_percentage) by (model)

# Active inference requests
sum(ryot_active_inference_requests)
```

### SLO Dashboard

```promql
# Latency SLO status (1 = meeting, 0 = violated)
(histogram_quantile(0.95, rate(ryot_inference_latency_seconds_bucket[5m])) < 0.5) * 1

# Error rate SLO status
((rate(ryot_inference_errors_total[5m]) / rate(ryot_inference_requests_total[5m])) < 0.001) * 1

# Throughput SLO status (>100 tps)
(sum(rate(ryot_tokens_generated_total[1m])) > 100) * 1

# GPU headroom SLO (>15% free)
((100 - avg(ryot_gpu_memory_percentage)) > 15) * 1
```

### Debugging Queries

```promql
# Which models are slowest?
topk(5, histogram_quantile(0.95, rate(ryot_inference_latency_seconds_bucket[5m])) by (model))

# Which models have highest error rates?
topk(5, rate(ryot_inference_errors_total[5m]) by (model))

# GPU memory hot spots
topk(5, ryot_gpu_memory_percentage)

# Batch size effectiveness
avg(rate(ryot_batch_size_distribution_sum[5m]) / rate(ryot_batch_size_distribution_count[5m])) by (model)
```

---

## 🚨 Alert Thresholds

| Alert Name          | Condition           | Threshold | Action                      |
| ------------------- | ------------------- | --------- | --------------------------- |
| High Error Rate     | Error % > threshold | 5%        | Check logs, restart service |
| High Latency        | P95 latency         | > 2.5s    | Check GPU load, scale       |
| GPU Memory Pressure | Memory usage        | > 85%     | Reduce batch size, scale    |
| Low Throughput      | Token rate          | < 50 tps  | Check GPU, restart          |
| Queue Backlog       | Queue depth         | > 100     | Scale horizontally          |
| TTFT Degradation    | TTFT latency        | > 500ms   | Investigate model loading   |

---

## 📈 Expected Baselines

### BitNet 1.58b (CPU)

```
TTFT:           50-100ms
Total latency:  100-500ms
Token rate:     100-150 tps
Success rate:   > 99.5%
```

### Llama 7B (Single GPU)

```
TTFT:           20-50ms
Total latency:  50-200ms
Token rate:     150-250 tps
GPU memory:     8-12GB
Success rate:   > 99.9%
```

### Multi-GPU Setup

```
TTFT:           10-30ms (improved)
Total latency:  30-150ms
Token rate:     300-600 tps
GPU memory:     Distributed
Success rate:   > 99.95%
```

---

## 🔍 Common Issues & Diagnostics

### Issue: High Latency

**Check metrics:**

```promql
histogram_quantile(0.95, rate(ryot_inference_latency_seconds_bucket[5m]))
histogram_quantile(0.95, rate(ryot_ttft_latency_seconds_bucket[5m]))
rate(ryot_model_loading_time_seconds_bucket{load_type="cold_start"}[5m])
```

**Likely causes:**

1. Cold model loads → Monitor `model_loading_time_seconds`
2. KV cache misses → Check `kv_cache_hit_ratio`
3. GPU memory swapping → Check `gpu_memory_percentage`
4. Queue buildup → Monitor `queue_depth`

**Solutions:**

- Increase model cache size
- Use smaller batch sizes
- Pre-warm models
- Scale GPU resources

---

### Issue: High Error Rate

**Check metrics:**

```promql
rate(ryot_inference_errors_total[5m]) by (error_type)
rate(ryot_inference_errors_total[5m]) by (model)
```

**Likely causes:**

1. Out of Memory → Look for `oom` errors
2. Timeout → Check `timeout` error spike
3. Invalid requests → Check `invalid_request` errors
4. GPU failures → Check `cuda_error` errors

**Solutions:**

- Reduce batch sizes (OOM)
- Increase timeout (timeout)
- Validate prompts (invalid_request)
- Check GPU health (cuda_error)

---

### Issue: Low Token Throughput

**Check metrics:**

```promql
sum(rate(ryot_tokens_generated_total[1m]))
avg(ryot_batch_size_distribution_sum/ryot_batch_size_distribution_count)
ryot_gpu_utilization_percent
```

**Likely causes:**

1. Small batch sizes → Increase batching
2. Low GPU utilization → Check for I/O bottlenecks
3. Memory contention → Monitor `gpu_memory_percentage`
4. Model inefficiency → Profile inference code

**Solutions:**

- Enable automatic batching
- Optimize model for hardware
- Use appropriate quantization
- Profile with GPU profiler

---

### Issue: GPU Memory Pressure

**Check metrics:**

```promql
ryot_gpu_memory_percentage
ryot_kv_cache_size_bytes
rate(ryot_model_cache_evictions_total[5m])
```

**Likely causes:**

1. Large KV cache → Reduce context length
2. Large batch size → Reduce batch
3. Model too large → Use quantization
4. Memory leak → Monitor over time

**Solutions:**

- Enable KV cache quantization
- Use sliding window attention
- Reduce batch size
- Apply model quantization (int8/fp16)

---

## 📋 Daily Operations Checklist

**Morning (9 AM)**

- [ ] Check error rate: `rate(ryot_inference_errors_total[24h])`
- [ ] Review P95 latency: `histogram_quantile(0.95, rate(ryot_inference_latency_seconds_bucket[24h]))`
- [ ] Verify GPU health: `avg(ryot_gpu_utilization_percent)`
- [ ] Check for any alerts fired overnight

**Throughout Day**

- [ ] Monitor queue depth: `ryot_queue_depth > 50` triggers investigation
- [ ] Watch token rate: `sum(rate(ryot_tokens_generated_total[5m])) < 80` alert
- [ ] Verify GPU memory: `ryot_gpu_memory_percentage < 85`
- [ ] Spot-check request patterns: Batch size distribution

**Weekly Review**

- [ ] Generate capacity report
  ```promql
  sum(rate(ryot_tokens_generated_total[7d])) / 604800  # tokens/sec avg
  ```
- [ ] Analyze error trends: `rate(ryot_inference_errors_total[7d]) by (error_type)`
- [ ] Review SLO compliance:
  - Latency: P95 < 500ms
  - Success: > 99.9%
  - Throughput: > 100 tps
- [ ] Plan scaling if needed

---

## 🎯 Performance Tuning Based on Metrics

### Scenario 1: Latency Too High

```
Observed: P95 = 2s (target: 0.5s)

Investigation:
  1. TTFT high? → Model loading slow, enable caching
  2. Generation slow? → Check token rate, may indicate GPU saturation
  3. Batch size? → Too large batch, increase parallelism

Action:
  - Set: max_concurrent_requests = 100 (was 50)
  - Result: P95 latency reduced to 400ms
  - Trade-off: GPU utilization increased 15%
```

### Scenario 2: Throughput Low

```
Observed: Token rate = 50 tps (target: 150 tps)

Investigation:
  1. Batch size small? → avg batch = 2 (should be 8-16)
  2. GPU memory? → 45% utilized (headroom available)
  3. Model loading? → Cold loads taking 5s

Action:
  - Enable: automatic_batching = true
  - Set: batch_size_target = 16
  - Result: Token rate increased to 180 tps
  - Side effect: Slight latency increase (200ms → 250ms)
```

### Scenario 3: GPU Memory Pressure

```
Observed: GPU memory = 92% (target: <80%)

Investigation:
  1. KV cache size? → 6GB for 8k context
  2. Model size? → 12GB for 7B params (no quantization)
  3. Batch size? → batch=32 (too large)

Action:
  - Enable: int8_quantization = true (model: 8GB)
  - Set: max_context_length = 4096 (was 8192)
  - Set: max_batch_size = 16 (was 32)
  - Result: GPU memory at 68%, latency decreased 10%
```

---

## 📊 Grafana Dashboard Queries

### Panel 1: Requests/Sec

```promql
rate(ryot_inference_requests_total[1m]) - rate(ryot_inference_requests_total{status="error"}[1m])
```

### Panel 2: Error Rate %

```promql
(rate(ryot_inference_requests_total{status="error"}[1m]) / rate(ryot_inference_requests_total[1m])) * 100
```

### Panel 3: P50/P95/P99 Latency

```promql
histogram_quantile(0.5, rate(ryot_inference_latency_seconds_bucket[5m]))
histogram_quantile(0.95, rate(ryot_inference_latency_seconds_bucket[5m]))
histogram_quantile(0.99, rate(ryot_inference_latency_seconds_bucket[5m]))
```

### Panel 4: Token Generation Rate

```promql
sum(rate(ryot_tokens_generated_total[1m])) by (model)
```

### Panel 5: GPU Memory %

```promql
avg(ryot_gpu_memory_percentage) by (model)
```

### Panel 6: Active Requests

```promql
sum(ryot_active_inference_requests)
```

### Panel 7: Queue Depth

```promql
sum(ryot_queue_depth) by (model)
```

### Panel 8: TTFT Latency

```promql
histogram_quantile(0.95, rate(ryot_ttft_latency_seconds_bucket[5m]))
```

---

## 🔐 Security & Compliance

**Metric Retention**

- Store raw metrics: 2 weeks
- Store aggregates: 1 year
- Comply with data retention policies

**Access Control**

- Prometheus: Internal only (restricted network)
- Grafana: Authenticated users only
- Alerts: Ops team access

**Privacy**

- Metrics: No PII included ✓
- Labels: Model names, error types only ✓
- Querying: Audited for compliance ✓

---

## 📞 Support Matrix

| Issue          | Metrics to Check      | Query                                | Escalation       |
| -------------- | --------------------- | ------------------------------------ | ---------------- |
| Slow inference | Latency histogram     | `histogram_quantile(0.95, ...)`      | Performance team |
| Errors         | Error counter by type | `rate(...errors_total[5m])`          | Dev team         |
| GPU issues     | GPU memory/util       | `ryot_gpu_*`                         | Infrastructure   |
| Scaling needed | Throughput, queue     | `rate(tokens_total)` + `queue_depth` | Cloud ops        |

---

## 📚 Related Documentation

- [Full Design Specification](RYOT_METRICS_DESIGN.md)
- [Implementation Guide with Code](RYOT_METRICS_IMPLEMENTATION.md)
- [Prometheus Documentation](https://prometheus.io/docs)
- [Grafana Dashboard Guide](https://grafana.com/docs/grafana/latest/dashboards/)
