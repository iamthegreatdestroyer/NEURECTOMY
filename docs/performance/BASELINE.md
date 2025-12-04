# Performance Baseline

This document establishes the performance baseline for NEURECTOMY, including target metrics, benchmarking methodology, and optimization guidelines.

## Overview

Performance is a critical quality attribute for NEURECTOMY. This document defines:
- Target performance metrics
- Benchmark methodology
- Baseline measurements
- Optimization strategies

## Performance Targets

### API Response Times (P99)

| Endpoint Type | Target | Maximum | Notes |
|---------------|--------|---------|-------|
| Health Check | < 10ms | 50ms | Simple status check |
| GraphQL Simple Query | < 50ms | 100ms | Single entity fetch |
| GraphQL Complex Query | < 200ms | 500ms | Joins, aggregations |
| GraphQL Mutation | < 100ms | 250ms | Single write operation |
| WebSocket Message | < 20ms | 50ms | Real-time updates |
| Authentication | < 100ms | 200ms | JWT validation |

### Throughput Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Health Endpoint | 10,000 RPS | Per instance |
| GraphQL Queries | 1,000 RPS | Per instance |
| GraphQL Mutations | 500 RPS | Per instance |
| WebSocket Connections | 10,000 concurrent | Per instance |
| WebSocket Messages | 50,000/sec | Per instance |

### Database Performance

| Operation | Target | Notes |
|-----------|--------|-------|
| Simple SELECT | < 5ms | Single row by PK |
| Complex SELECT | < 50ms | Joins, WHERE clauses |
| INSERT | < 10ms | Single row |
| Batch INSERT | 10,000 rows/sec | Bulk operations |
| Vector Search | < 100ms | pgvector similarity |
| Graph Traversal | < 50ms | Neo4j 3-hop query |

### Memory & Resources

| Resource | Target | Maximum |
|----------|--------|---------|
| Memory per instance | 512 MB | 2 GB |
| CPU per instance | 0.5 cores | 2 cores |
| Connection pool | 20 connections | 50 connections |
| Cache hit rate | > 80% | - |

## Benchmark Suite

### Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench --bench api_benchmarks
cargo bench --bench auth_benchmarks
cargo bench --bench database_benchmarks

# Run with specific feature
cargo bench --features "benchmark"

# Generate HTML reports
cargo bench -- --save-baseline main
```

### Benchmark Categories

#### API Benchmarks (`api_benchmarks.rs`)

- **Request Parsing**: JSON deserialization performance
- **Response Serialization**: JSON serialization for various payload sizes
- **UUID Generation**: v4 and v7 UUID creation
- **Timestamp Operations**: DateTime creation and formatting
- **String Operations**: Common string manipulations
- **GraphQL Patterns**: Query depth estimation, operation detection

#### Authentication Benchmarks (`auth_benchmarks.rs`)

- **Password Hashing**: Argon2id hash generation
- **Password Verification**: Hash comparison
- **JWT Operations**: Token encoding/decoding
- **API Key Operations**: Key generation and validation
- **Session Operations**: Session management
- **Permission Checks**: Authorization validation

#### Database Benchmarks (`database_benchmarks.rs`)

- **Query Building**: SQL/Cypher query construction
- **Data Serialization**: Entity to/from JSON
- **Connection Pool**: Pool acquire/release simulation
- **Caching**: Cache operations and TTL checks
- **Batch Operations**: Bulk data handling
- **Vector Operations**: Embedding calculations (pgvector)
- **Graph Queries**: Cypher query patterns

## Baseline Measurements

### Initial Baseline (Phase 1)

These are the expected baseline measurements established during Phase 1:

| Benchmark | Expected Range | Status |
|-----------|---------------|--------|
| JSON parse (small) | 100-500ns | Baseline |
| JSON parse (medium) | 500ns-2µs | Baseline |
| JSON parse (large) | 2-10µs | Baseline |
| UUID v4 generation | 50-100ns | Baseline |
| UUID v7 generation | 100-200ns | Baseline |
| Argon2id hash | 200-500ms | Baseline |
| JWT encode | 50-200µs | Baseline |
| JWT decode | 50-200µs | Baseline |
| Cache lookup (hit) | 10-50ns | Baseline |
| Vector cosine (1536d) | 1-5µs | Baseline |

### Tracking Baseline Changes

Baselines should be updated:
1. After major architectural changes
2. When upgrading dependencies
3. After performance optimization work
4. At each milestone completion

```bash
# Save new baseline
cargo bench -- --save-baseline phase_1

# Compare against baseline
cargo bench -- --baseline phase_1
```

## Performance Optimization Guidelines

### 1. Measure First

Never optimize without measurements:

```rust
// Use tracing for timing
use tracing::instrument;

#[instrument(skip(self))]
async fn get_agent(&self, id: Uuid) -> Result<Agent> {
    // Method automatically timed
}
```

### 2. Profile Before Optimizing

```bash
# CPU profiling with perf
perf record -g ./target/release/neurectomy-core
perf report

# Memory profiling with heaptrack
heaptrack ./target/release/neurectomy-core
heaptrack_gui heaptrack.*.gz
```

### 3. Optimization Priority

1. **Algorithm Complexity**: O(n) → O(log n) → O(1)
2. **Data Structures**: Choose appropriate structures
3. **Caching**: Add caching for repeated computations
4. **Batching**: Combine multiple operations
5. **Async/Parallel**: Utilize concurrency
6. **Code-level**: Micro-optimizations last

### 4. Common Optimizations

#### Caching

```rust
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;

struct Cache<T> {
    data: Arc<RwLock<HashMap<String, (T, Instant)>>>,
    ttl: Duration,
}

impl<T: Clone> Cache<T> {
    async fn get(&self, key: &str) -> Option<T> {
        let data = self.data.read().await;
        data.get(key)
            .filter(|(_, created)| created.elapsed() < self.ttl)
            .map(|(value, _)| value.clone())
    }
}
```

#### Connection Pooling

```rust
// Already configured in SQLx
let pool = PgPoolOptions::new()
    .max_connections(20)
    .min_connections(5)
    .acquire_timeout(Duration::from_secs(3))
    .connect(&database_url)
    .await?;
```

#### Batch Operations

```rust
// Instead of N separate inserts
for agent in agents {
    sqlx::query("INSERT INTO agents ...").execute(&pool).await?;
}

// Use batch insert
let query = format!(
    "INSERT INTO agents (id, name, status) VALUES {}",
    values.join(", ")
);
sqlx::query(&query).execute(&pool).await?;
```

### 5. Memory Optimization

```rust
// Pre-allocate vectors
let mut results = Vec::with_capacity(expected_size);

// Use references where possible
fn process(data: &[Agent]) -> Vec<&Agent> {
    data.iter().filter(|a| a.is_active()).collect()
}

// Use Cow for optional cloning
use std::borrow::Cow;
fn process_name(name: Cow<'_, str>) -> String {
    name.into_owned()
}
```

## Monitoring & Alerting

### Prometheus Metrics

Key metrics exported:

```
# Request latency histogram
http_request_duration_seconds_bucket{method="GET",path="/health",le="0.01"}

# Request rate
http_requests_total{method="POST",path="/graphql",status="200"}

# Active connections
database_connections_active

# Cache metrics
cache_hits_total
cache_misses_total
```

### Alert Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| P99 Latency | > 500ms | > 1s |
| Error Rate | > 1% | > 5% |
| CPU Usage | > 70% | > 90% |
| Memory Usage | > 70% | > 90% |
| DB Connections | > 80% | > 95% |

## Load Testing

### Tools

- **Criterion**: Micro-benchmarks (included)
- **k6**: Load testing HTTP endpoints
- **wrk**: High-performance load generation
- **locust**: Python-based load testing

### k6 Example

```javascript
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '30s', target: 100 },
    { duration: '1m', target: 100 },
    { duration: '10s', target: 0 },
  ],
};

export default function () {
  const res = http.post('http://localhost:4000/graphql', JSON.stringify({
    query: 'query { agents { id name } }'
  }), {
    headers: { 'Content-Type': 'application/json' },
  });
  
  check(res, {
    'status is 200': (r) => r.status === 200,
    'latency < 200ms': (r) => r.timings.duration < 200,
  });
  
  sleep(0.1);
}
```

## Continuous Performance Testing

### CI/CD Integration

```yaml
# .github/workflows/benchmark.yml
name: Benchmarks

on:
  push:
    branches: [main]
  pull_request:

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run benchmarks
        run: cargo bench -- --save-baseline pr
        
      - name: Compare with main
        run: cargo bench -- --baseline main
        
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: target/criterion
```

### Performance Regression Detection

Fail CI if:
- P99 latency increases by > 20%
- Throughput decreases by > 10%
- Memory usage increases by > 30%

## References

- [Criterion.rs Documentation](https://bheisler.github.io/criterion.rs/book/)
- [Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/naming/)
- [k6 Documentation](https://k6.io/docs/)
