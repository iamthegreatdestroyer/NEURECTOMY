# Phase 18D: OpenTelemetry + Jaeger Distributed Tracing

**Status:** Design & Implementation Guide  
**Target:** Neurectomy Phase 18D  
**Date:** 2024-2025

---

## 📊 Executive Overview

This document provides the complete implementation framework for distributed tracing across Neurectomy's 4 core services using OpenTelemetry (OTEL) and Jaeger. Enables end-to-end request tracing, latency analysis, and bottleneck identification with cost-optimized sampling.

### Key Metrics

- **Trace Coverage:** 100% of request flows
- **Sampling Rate (Adaptive):** 10% baseline → 100% on errors/high latency
- **Trace Retention:** 72 hours hot, 30 days archived
- **Infrastructure Cost Impact:** +15-20% (optimized with sampling)

---

## 🏗️ Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                    NEURECTOMY SERVICES                         │
├────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │    RYOT      │  │   ΣLANG      │  │   ΣVAULT     │  ...     │
│  │   (Python)   │  │  (Rust/WASM) │  │   (Rust)     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          │                                      │
│                   OpenTelemetry Instrumentation                 │
│                   (Traces + Metrics + Logs)                     │
│                          │                                      │
│  ┌────────────────────────────────────────────────────┐         │
│  │      OpenTelemetry Collector (Daemonset)          │         │
│  │  • Trace processors (sampling, batching)          │         │
│  │  • Metric exporters                               │         │
│  │  • Log aggregation                                │         │
│  └────────────────────────────────────────────────────┘         │
│         │                    │                  │               │
│         ├────────────────────┼──────────────────┤               │
│         ▼                    ▼                  ▼               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │    Jaeger    │  │ Prometheus   │  │  Loki/ELK    │          │
│  │   (Traces)   │  │  (Metrics)   │  │   (Logs)     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                    │                  │               │
│         └────────────────────┼──────────────────┘               │
│                              │                                  │
│                       ┌──────▼────────┐                         │
│                       │    Grafana    │                         │
│                       │ (Visualization│                         │
│                       │ + Alerting)   │                         │
│                       └───────────────┘                         │
└────────────────────────────────────────────────────────────────┘
```

---

## 📡 Service Instrumentation Guides

### 1. RYOT (Python LLM Service)

#### Installation

```bash
# Install OpenTelemetry packages
pip install \
  opentelemetry-api \
  opentelemetry-sdk \
  opentelemetry-exporter-jaeger-thrift \
  opentelemetry-exporter-prometheus \
  opentelemetry-instrumentation \
  opentelemetry-instrumentation-requests \
  opentelemetry-instrumentation-flask \
  opentelemetry-instrumentation-sqlalchemy \
  opentelemetry-instrumentation-redis \
  opentelemetry-instrumentation-grpc
```

#### Implementation Pattern

**File:** `ryot/monitoring/tracing.py`

```python
# OpenTelemetry Initialization for RYOT
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
import logging

class RyotTracingConfig:
    """Configure tracing for RYOT service."""

    @staticmethod
    def initialize(service_name="ryot", jaeger_agent_host="localhost", jaeger_agent_port=6831):
        """Initialize OpenTelemetry tracing."""

        # Create Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name=jaeger_agent_host,
            agent_port=jaeger_agent_port,
        )

        # Create resource
        resource = Resource(attributes={
            SERVICE_NAME: service_name,
            "service.version": "18D",
            "deployment.environment": "production",
        })

        # Create TracerProvider
        trace_provider = TracerProvider(resource=resource)
        trace_provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
        trace.set_tracer_provider(trace_provider)

        # Auto-instrument libraries
        FlaskInstrumentor().instrument()
        RequestsInstrumentor().instrument()
        SQLAlchemyInstrumentor().instrument()
        RedisInstrumentor().instrument()

        return trace.get_tracer(__name__)

# Usage in Flask app
from flask import Flask

app = Flask(__name__)
tracer = RyotTracingConfig.initialize()

@app.route("/inference", methods=["POST"])
def inference():
    """LLM inference endpoint with tracing."""
    with tracer.start_as_current_span("llm_inference") as span:
        span.set_attribute("model.name", "gpt-4")
        span.set_attribute("request.tokens", 2048)

        # Automatic instrumentation of sub-calls
        response = make_llm_request()

        span.set_attribute("response.tokens", len(response))
        return response
```

#### Span Strategy

| Operation         | Span Name        | Attributes                 | Duration   |
| ----------------- | ---------------- | -------------------------- | ---------- |
| LLM Inference     | `llm.inference`  | model, tokens, temperature | 100-5000ms |
| Prompt Processing | `prompt.process` | tokens_in, tokens_out      | 10-100ms   |
| Cache Lookup      | `cache.get`      | key, hit/miss              | 1-10ms     |
| Database Query    | `db.query`       | query, rows, duration      | 5-500ms    |
| Vector Search     | `vector.search`  | dimensions, k              | 50-1000ms  |

---

### 2. ΣLANG (Rust/WASM Language Service)

#### Installation

```bash
# Add Cargo dependencies
[dependencies]
opentelemetry = "0.21"
opentelemetry-jaeger-thrift = { version = "0.20", features = ["rt-tokio"] }
opentelemetry-prometheus = "0.14"
opentelemetry-otlp = { version = "0.14", features = ["tonic"] }
tokio = { version = "1", features = ["full"] }
tracing = "0.1"
tracing-opentelemetry = "0.21"
tracing-subscriber = "0.3"
```

#### Implementation Pattern

**File:** `sigmalang/src/tracing.rs`

```rust
// OpenTelemetry Tracing for ΣLANG
use opentelemetry::global;
use opentelemetry_jaeger_thrift::new_agent_pipeline;
use tracing::{info, span, Level};
use tracing_opentelemetry::OpenTelemetryLayer;
use tracing_subscriber::Registry;

pub fn init_tracing(service_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Create Jaeger exporter
    let tracer = new_agent_pipeline()
        .install_simple()?;

    // Create tracing subscriber
    let subscriber = Registry::default()
        .with(OpenTelemetryLayer::new(tracer));

    tracing::subscriber::set_global_default(subscriber)?;

    info!("Tracing initialized for {}", service_name);
    Ok(())
}

// Usage in Rust code
pub async fn compile_expression(source: &str) -> Result<Program, CompileError> {
    let span = span!(Level::INFO, "sigma_compile", tokens = 0);
    let _enter = span.enter();

    // Parsing phase
    let tokens = {
        let parse_span = span!(Level::DEBUG, "sigma_parse");
        let _parse_enter = parse_span.enter();
        tokenize(source)?
    };

    // Type checking phase
    let typed_program = {
        let typecheck_span = span!(Level::DEBUG, "sigma_typecheck");
        let _typecheck_enter = typecheck_span.enter();
        type_check_program(&tokens)?
    };

    // Code generation phase
    let program = {
        let codegen_span = span!(Level::DEBUG, "sigma_codegen");
        let _codegen_enter = codegen_span.enter();
        codegen(&typed_program)?
    };

    drop(_enter);
    Ok(program)
}
```

#### Span Strategy

| Operation        | Span Name            | Attributes                 | Duration |
| ---------------- | -------------------- | -------------------------- | -------- |
| Language Parsing | `sigma.parse`        | input_size, token_count    | 10-100ms |
| Type Checking    | `sigma.typecheck`    | type_errors, warnings      | 5-50ms   |
| Code Generation  | `sigma.codegen`      | target, optimization_level | 20-200ms |
| WASM Compilation | `sigma.wasm_compile` | wasm_size                  | 50-500ms |
| VM Execution     | `sigma.vm_exec`      | instructions, stack_depth  | 1-1000ms |

---

### 3. ΣVAULT (Rust Storage Service)

#### Installation

```bash
# Cargo.toml additions (same as ΣLANG)
```

#### Implementation Pattern

**File:** `sigmavault/src/tracing.rs`

```rust
// OpenTelemetry Tracing for ΣVAULT
use opentelemetry::trace::{Tracer, Status};
use std::sync::Arc;

pub struct VaultTracer {
    tracer: Arc<dyn Tracer>,
}

impl VaultTracer {
    pub fn new(tracer: Arc<dyn Tracer>) -> Self {
        Self { tracer }
    }

    pub async fn trace_storage_operation<F, T>(
        &self,
        operation: &str,
        tier: &str,
        f: F,
    ) -> Result<T, Box<dyn std::error::Error>>
    where
        F: std::future::Future<Output = Result<T, Box<dyn std::error::Error>>>,
    {
        let mut span = self.tracer.start(format!("storage.{}", operation));
        span.set_attribute("storage.tier".into(), tier.into());

        match f.await {
            Ok(result) => {
                span.set_status(Status::Ok);
                Ok(result)
            }
            Err(e) => {
                span.set_status(Status::error(e.to_string()));
                Err(e)
            }
        }
    }
}

// Usage
pub async fn store_encrypted(
    vault: &VaultTracer,
    data: &[u8],
    tier: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    vault
        .trace_storage_operation("encrypt_store", tier, async {
            let encrypted = encrypt_aes256_gcm(data)?;
            let object_id = persist_to_tier(&encrypted, tier).await?;
            Ok(object_id)
        })
        .await
}
```

#### Span Strategy

| Operation          | Span Name            | Attributes                    | Duration    |
| ------------------ | -------------------- | ----------------------------- | ----------- |
| Store Operation    | `storage.store`      | tier, size_bytes, encryption  | 5-1000ms    |
| Retrieve Operation | `storage.retrieve`   | tier, object_id, decrypt_time | 10-500ms    |
| Encryption         | `encryption.encrypt` | algorithm, size               | 1-100ms     |
| Decryption         | `encryption.decrypt` | algorithm, size               | 1-100ms     |
| Tier Migration     | `storage.migrate`    | from_tier, to_tier, size      | 100-5000ms  |
| Snapshot           | `storage.snapshot`   | object_count, data_size       | 500-30000ms |

---

### 4. Agent Collective (Multi-Language)

#### Installation

```bash
# Python agents
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-jaeger-thrift

# JavaScript agents
npm install @opentelemetry/api @opentelemetry/sdk-trace-node \
  @opentelemetry/exporter-trace-otlp-http @opentelemetry/auto-instrumentations-node
```

#### Implementation Pattern - Python Agents

**File:** `agents/monitoring/agent_tracing.py`

```python
# OpenTelemetry Tracing for Agent Collective
from opentelemetry import trace
from functools import wraps
from contextlib import contextmanager

tracer = trace.get_tracer("agent-collective")

def agent_traced(agent_name: str, task_type: str):
    """Decorator for tracing agent executions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(
                f"agent.execute",
                attributes={
                    "agent.name": agent_name,
                    "task.type": task_type,
                }
            ) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("execution.status", "success")
                    return result
                except Exception as e:
                    span.set_attribute("execution.status", "error")
                    span.set_attribute("error.type", type(e).__name__)
                    raise
        return wrapper
    return decorator

# Agent Class Integration
class EliteAgent:
    """Base agent with tracing support."""

    def __init__(self, name: str):
        self.name = name
        self.tracer = trace.get_tracer(name)

    @contextmanager
    def trace_operation(self, operation: str, **attributes):
        """Context manager for tracing operations."""
        with self.tracer.start_as_current_span(
            f"{self.name}.{operation}",
            attributes=attributes
        ) as span:
            yield span

    async def execute_task(self, task: dict):
        """Execute task with automatic tracing."""
        with self.trace_operation("execute", task_id=task.get("id")) as span:
            # Retrieve from memory
            with self.tracer.start_as_current_span("memory.retrieve"):
                memories = self.retrieve_memories(task)

            # Process
            with self.tracer.start_as_current_span("task.process"):
                result = await self.process(task, memories)

            # Store result
            with self.tracer.start_as_current_span("memory.store"):
                self.store_result(result)

            return result

# Usage Example
@agent_traced("APEX", "code_generation")
def generate_code(problem_statement: str) -> str:
    """Generate code with tracing."""
    # Implementation
    return code
```

#### Span Strategy

| Operation       | Span Name           | Attributes                 | Duration    |
| --------------- | ------------------- | -------------------------- | ----------- |
| Agent Init      | `agent.init`        | agent_name, tier           | 10-100ms    |
| Memory Retrieve | `memory.retrieve`   | query_type, result_count   | 5-50ms      |
| Task Process    | `agent.process`     | task_type, complexity      | 100-10000ms |
| Memory Store    | `memory.store`      | data_size, fitness_score   | 1-100ms     |
| Agent Collab    | `agent.collaborate` | primary, secondary, result | 50-5000ms   |

---

## 🎯 Span Instrumentation Patterns

### Pattern 1: Automatic Database Instrumentation

```python
# Automatic with SQLAlchemy instrumentation
SQLAlchemyInstrumentor().instrument(
    engine=database_engine,
    capture_parameters=True
)

# Spans generated automatically:
# - db.query.execute
# - db.connection.execute
# - db.statement.execute
```

### Pattern 2: Manual Request Tracing

```python
def trace_request(tracer, func):
    """Trace incoming HTTP request."""
    @wraps(func)
    def wrapper(request, *args, **kwargs):
        with tracer.start_as_current_span(
            "http.request",
            attributes={
                "http.method": request.method,
                "http.url": request.url,
                "http.target": request.path,
            }
        ) as span:
            response = func(request, *args, **kwargs)
            span.set_attribute("http.status_code", response.status_code)
            return response
    return wrapper
```

### Pattern 3: Distributed Context Propagation

```python
# Incoming request
from opentelemetry.propagate import extract

def handle_request(headers):
    ctx = extract(headers)
    with trace.use_span(trace.get_current_span(), end_on_exit=True):
        # Process request - trace context maintained
        pass

# Outgoing request
from opentelemetry.propagate import inject

def make_downstream_call(downstream_service):
    headers = {}
    inject(headers)  # Inject trace context
    return requests.get(downstream_service, headers=headers)
```

---

## 🔄 Trace-to-Metric Correlation

### Automatic Correlation Mapping

**File:** `deploy/monitoring/trace-metric-correlation.yaml`

```yaml
# Trace Signals → Prometheus Metrics
correlations:
  # Latency Traces → Histogram Metrics
  - trace_signal: "span.duration"
    metric: "request_duration_seconds"
    mapping:
      operation: "span.name"
      service: "resource.service.name"
      status: "span.status"
    aggregation: "histogram"
    buckets: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

  # Error Traces → Error Rate Metrics
  - trace_signal: "span.error"
    metric: "errors_total"
    mapping:
      error_type: "span.attributes.error.type"
      service: "resource.service.name"
      operation: "span.name"
    aggregation: "counter"

  # Database Operations → DB Metrics
  - trace_signal: "db.statement.execute"
    metric: "db_query_duration_seconds"
    mapping:
      query_type: "span.attributes.db.operation"
      database: "span.attributes.db.name"
      table: "span.attributes.db.sql.table"
    aggregation: "histogram"

  # Cache Operations → Cache Metrics
  - trace_signal: "cache.get"
    metric: "cache_hits_total"
    mapping:
      cache_name: "span.attributes.cache.name"
      status: "span.attributes.cache.hit"
    aggregation: "counter"

  # Vector Search → Search Metrics
  - trace_signal: "vector.search"
    metric: "vector_search_duration_seconds"
    mapping:
      dimensions: "span.attributes.vector.dimensions"
      k_neighbors: "span.attributes.vector.k"
      index_name: "span.attributes.vector.index_name"
    aggregation: "histogram"

  # Agent Execution → Agent Metrics
  - trace_signal: "agent.execute"
    metric: "agent_execution_duration_seconds"
    mapping:
      agent_name: "span.attributes.agent.name"
      task_type: "span.attributes.task.type"
      status: "span.attributes.execution.status"
    aggregation: "histogram"

# Service Correlation Matrix
service_correlations:
  ryot:
    metrics:
      - llm_inference_duration_seconds
      - llm_tokens_generated_total
      - prompt_cache_hit_ratio
    traces:
      - llm.inference
      - prompt.process
      - cache.get

  sigmalang:
    metrics:
      - sigma_compilation_duration_seconds
      - sigma_parse_errors_total
      - wasm_size_bytes
    traces:
      - sigma.parse
      - sigma.typecheck
      - sigma.codegen

  sigmavault:
    metrics:
      - storage_operation_duration_seconds
      - encryption_duration_seconds
      - storage_tier_utilization
    traces:
      - storage.store
      - storage.retrieve
      - encryption.encrypt

  agents:
    metrics:
      - agent_execution_duration_seconds
      - agent_task_success_ratio
      - memory_retrieval_latency_seconds
    traces:
      - agent.execute
      - memory.retrieve
      - agent.collaborate
```

---

## 💡 Query Examples

### PromQL Queries Using Trace Data

```promql
# 1. 95th percentile latency for LLM inference
histogram_quantile(0.95,
  llm_inference_duration_seconds_bucket
)

# 2. Error rate for storage operations
(
  sum(rate(storage_operation_errors_total[5m]))
  /
  sum(rate(storage_operation_total[5m]))
) * 100

# 3. Cache hit ratio
(
  sum(rate(cache_hits_total[5m]))
  /
  sum(rate(cache_total[5m]))
)

# 4. Agent execution success rate
(
  sum(rate(agent_execution_success_total[5m]))
  /
  sum(rate(agent_execution_total[5m]))
)
```

### Jaeger Queries for Trace Analysis

```json
{
  "service": "ryot",
  "operation": "llm.inference",
  "minDuration": "100ms",
  "maxDuration": "5s",
  "tags": {
    "error": "true"
  }
}
```

---

## 📊 Implementation Checklist

- [ ] OpenTelemetry SDK installed in all 4 services
- [ ] Jaeger exporter configured for each service
- [ ] Instrumentation libraries auto-enabled (requests, SQLAlchemy, Flask, etc.)
- [ ] Custom span generators implemented for domain-specific operations
- [ ] Trace context propagation configured for inter-service calls
- [ ] Sampling strategy deployed and tested
- [ ] Trace-to-metric correlation mapping implemented
- [ ] Grafana dashboards integrated with trace backends
- [ ] Alerting rules created for trace-based SLOs
- [ ] Documentation and runbooks generated
