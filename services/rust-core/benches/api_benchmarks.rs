//! API Performance Benchmarks
//!
//! This module contains benchmarks for measuring API endpoint performance.
//! Run with: cargo bench --bench api_benchmarks

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use serde_json::{json, Value};
use std::time::Duration;

/// Benchmark configuration constants
mod config {
    /// Target latency for health check endpoint (microseconds)
    pub const HEALTH_CHECK_TARGET_US: u64 = 100;

    /// Target latency for GraphQL simple query (milliseconds)
    pub const GRAPHQL_SIMPLE_QUERY_TARGET_MS: u64 = 10;

    /// Target latency for GraphQL complex query (milliseconds)
    pub const GRAPHQL_COMPLEX_QUERY_TARGET_MS: u64 = 50;

    /// Target latency for authentication (milliseconds)
    pub const AUTH_TARGET_MS: u64 = 100;

    /// Target requests per second for health endpoint
    pub const HEALTH_RPS_TARGET: u64 = 10000;

    /// Target requests per second for GraphQL endpoint
    pub const GRAPHQL_RPS_TARGET: u64 = 1000;
}

/// Simulate request parsing benchmark
fn bench_request_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("request_parsing");
    group.measurement_time(Duration::from_secs(10));

    // Small JSON payload (typical health check)
    let small_json = json!({
        "status": "healthy"
    });

    // Medium JSON payload (typical API request)
    let medium_json = json!({
        "query": "query { agents { id name status } }",
        "variables": {},
        "operationName": "GetAgents"
    });

    // Large JSON payload (complex mutation)
    let large_json = json!({
        "query": "mutation CreateAgent($input: CreateAgentInput!) { createAgent(input: $input) { id name description capabilities tools status createdAt } }",
        "variables": {
            "input": {
                "name": "Test Agent",
                "description": "A test agent for benchmarking",
                "capabilities": ["text-generation", "code-analysis", "data-processing"],
                "tools": [
                    {"name": "python", "version": "3.11"},
                    {"name": "rust", "version": "1.75"},
                    {"name": "nodejs", "version": "20"}
                ],
                "config": {
                    "maxTokens": 4096,
                    "temperature": 0.7,
                    "topP": 0.9,
                    "frequencyPenalty": 0.0,
                    "presencePenalty": 0.0
                }
            }
        },
        "operationName": "CreateAgent"
    });

    group.throughput(Throughput::Bytes(small_json.to_string().len() as u64));
    group.bench_with_input(
        BenchmarkId::new("json_parse", "small"),
        &small_json.to_string(),
        |b, json_str| {
            b.iter(|| {
                let _: Value = serde_json::from_str(black_box(json_str)).unwrap();
            });
        },
    );

    group.throughput(Throughput::Bytes(medium_json.to_string().len() as u64));
    group.bench_with_input(
        BenchmarkId::new("json_parse", "medium"),
        &medium_json.to_string(),
        |b, json_str| {
            b.iter(|| {
                let _: Value = serde_json::from_str(black_box(json_str)).unwrap();
            });
        },
    );

    group.throughput(Throughput::Bytes(large_json.to_string().len() as u64));
    group.bench_with_input(
        BenchmarkId::new("json_parse", "large"),
        &large_json.to_string(),
        |b, json_str| {
            b.iter(|| {
                let _: Value = serde_json::from_str(black_box(json_str)).unwrap();
            });
        },
    );

    group.finish();
}

/// Simulate response serialization benchmark
fn bench_response_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("response_serialization");
    group.measurement_time(Duration::from_secs(10));

    // Health response
    let health_response = json!({
        "status": "healthy",
        "version": "0.1.0",
        "uptime": 86400,
        "services": {
            "database": "connected",
            "cache": "connected",
            "queue": "connected"
        }
    });

    // Agent list response (10 agents)
    let agents: Vec<Value> = (0..10)
        .map(|i| {
            json!({
                "id": format!("agent-{}", i),
                "name": format!("Agent {}", i),
                "description": "A sophisticated AI agent",
                "status": "active",
                "capabilities": ["text-generation", "code-analysis"],
                "createdAt": "2025-01-01T00:00:00Z",
                "updatedAt": "2025-01-15T12:00:00Z"
            })
        })
        .collect();

    let agent_list_response = json!({
        "data": {
            "agents": agents
        }
    });

    // Large response (100 agents with detailed info)
    let large_agents: Vec<Value> = (0..100)
        .map(|i| {
            json!({
                "id": format!("agent-{}", i),
                "name": format!("Agent {}", i),
                "description": "A sophisticated AI agent with many capabilities",
                "status": if i % 3 == 0 { "active" } else if i % 3 == 1 { "idle" } else { "offline" },
                "capabilities": ["text-generation", "code-analysis", "data-processing", "image-recognition"],
                "tools": [
                    {"name": "python", "version": "3.11", "status": "available"},
                    {"name": "rust", "version": "1.75", "status": "available"},
                    {"name": "nodejs", "version": "20", "status": "available"}
                ],
                "config": {
                    "maxTokens": 4096,
                    "temperature": 0.7,
                    "model": "gpt-4-turbo"
                },
                "metrics": {
                    "totalRequests": 1000 + i * 100,
                    "successRate": 0.99,
                    "avgLatencyMs": 150
                },
                "createdAt": "2025-01-01T00:00:00Z",
                "updatedAt": "2025-01-15T12:00:00Z"
            })
        })
        .collect();

    let large_response = json!({
        "data": {
            "agents": large_agents,
            "pagination": {
                "total": 100,
                "page": 1,
                "perPage": 100
            }
        }
    });

    group.bench_function("serialize_health", |b| {
        b.iter(|| serde_json::to_string(black_box(&health_response)).unwrap());
    });

    group.bench_function("serialize_agent_list_10", |b| {
        b.iter(|| serde_json::to_string(black_box(&agent_list_response)).unwrap());
    });

    group.bench_function("serialize_agent_list_100", |b| {
        b.iter(|| serde_json::to_string(black_box(&large_response)).unwrap());
    });

    group.finish();
}

/// Benchmark UUID generation (used extensively in API)
fn bench_uuid_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("uuid_generation");
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("uuid_v4", |b| {
        b.iter(|| black_box(uuid::Uuid::new_v4()));
    });

    group.bench_function("uuid_v7", |b| {
        b.iter(|| black_box(uuid::Uuid::now_v7()));
    });

    group.bench_function("uuid_to_string", |b| {
        let id = uuid::Uuid::new_v4();
        b.iter(|| black_box(id.to_string()));
    });

    group.bench_function("uuid_from_string", |b| {
        let id_str = uuid::Uuid::new_v4().to_string();
        b.iter(|| black_box(uuid::Uuid::parse_str(&id_str).unwrap()));
    });

    group.finish();
}

/// Benchmark timestamp operations
fn bench_timestamp_operations(c: &mut Criterion) {
    use chrono::{DateTime, Utc};

    let mut group = c.benchmark_group("timestamp_operations");
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("now_utc", |b| {
        b.iter(|| black_box(Utc::now()));
    });

    group.bench_function("format_rfc3339", |b| {
        let now = Utc::now();
        b.iter(|| black_box(now.to_rfc3339()));
    });

    group.bench_function("parse_rfc3339", |b| {
        let timestamp = "2025-01-15T12:00:00Z";
        b.iter(|| black_box(DateTime::parse_from_rfc3339(timestamp).unwrap()));
    });

    group.finish();
}

/// Benchmark string operations common in API processing
fn bench_string_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_operations");
    group.measurement_time(Duration::from_secs(5));

    // Email validation simulation
    let emails = vec![
        "user@example.com",
        "test.user+tag@subdomain.example.org",
        "a@b.co",
    ];

    group.bench_function("email_lowercase", |b| {
        b.iter(|| {
            for email in &emails {
                black_box(email.to_lowercase());
            }
        });
    });

    // Username normalization
    let usernames = vec!["TestUser", "UPPERCASE", "lowercase", "MixedCase123"];

    group.bench_function("username_normalize", |b| {
        b.iter(|| {
            for username in &usernames {
                let normalized = username.trim().to_lowercase();
                black_box(normalized);
            }
        });
    });

    // Query sanitization simulation
    let queries = vec![
        "SELECT * FROM users",
        "query { agents { id } }",
        "mutation { createAgent(input: { name: \"Test\" }) { id } }",
    ];

    group.bench_function("query_length_check", |b| {
        b.iter(|| {
            for query in &queries {
                let valid = query.len() < 10000 && !query.is_empty();
                black_box(valid);
            }
        });
    });

    group.finish();
}

/// Benchmark GraphQL query validation patterns
fn bench_graphql_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphql_patterns");
    group.measurement_time(Duration::from_secs(5));

    // Simple query detection
    let queries = vec![
        ("simple", "query { agents { id } }"),
        ("with_args", "query GetAgent($id: ID!) { agent(id: $id) { id name } }"),
        ("nested", "query { agents { id name executions { id status } } }"),
        ("complex", "query GetAgentWithDetails($id: ID!) { agent(id: $id) { id name description status capabilities tools { name version } executions(limit: 10) { id status startedAt completedAt } } }"),
    ];

    for (name, query) in queries {
        group.bench_with_input(
            BenchmarkId::new("query_depth_estimate", name),
            &query,
            |b, q| {
                b.iter(|| {
                    // Simple depth estimation by counting braces
                    let depth = q.chars().filter(|c| *c == '{').count();
                    black_box(depth)
                });
            },
        );
    }

    // Operation type detection
    let operations = vec![
        "query { agents { id } }",
        "mutation { createAgent(input: {}) { id } }",
        "subscription { agentStatus { id status } }",
    ];

    group.bench_function("operation_type_detect", |b| {
        b.iter(|| {
            for op in &operations {
                let trimmed = op.trim();
                let op_type = if trimmed.starts_with("query") || trimmed.starts_with('{') {
                    "query"
                } else if trimmed.starts_with("mutation") {
                    "mutation"
                } else if trimmed.starts_with("subscription") {
                    "subscription"
                } else {
                    "unknown"
                };
                black_box(op_type);
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_request_parsing,
    bench_response_serialization,
    bench_uuid_generation,
    bench_timestamp_operations,
    bench_string_operations,
    bench_graphql_patterns,
);

criterion_main!(benches);
