//! Database Performance Benchmarks
//!
//! This module contains benchmarks for measuring database-related operations.
//! Run with: cargo bench --bench database_benchmarks

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;

/// Database performance targets
mod targets {
    /// Simple query target latency (ms)
    pub const SIMPLE_QUERY_TARGET_MS: u64 = 5;
    
    /// Complex query target latency (ms)
    pub const COMPLEX_QUERY_TARGET_MS: u64 = 50;
    
    /// Batch insert target throughput (rows/sec)
    pub const BATCH_INSERT_TARGET_RPS: u64 = 10000;
    
    /// Cache hit target latency (microseconds)
    pub const CACHE_HIT_TARGET_US: u64 = 100;
}

/// Benchmark SQL query building
fn bench_query_building(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_building");
    group.measurement_time(Duration::from_secs(10));
    
    // Simple query building
    group.bench_function("simple_select", |b| {
        b.iter(|| {
            let table = "agents";
            let columns = &["id", "name", "status"];
            let query = format!(
                "SELECT {} FROM {}",
                columns.join(", "),
                table
            );
            black_box(query)
        });
    });
    
    // Parameterized query building
    group.bench_function("parameterized_select", |b| {
        let id = uuid::Uuid::new_v4();
        b.iter(|| {
            let query = format!(
                "SELECT id, name, status, created_at, updated_at FROM agents WHERE id = $1",
            );
            black_box((query, id))
        });
    });
    
    // Complex query with joins
    group.bench_function("complex_join_query", |b| {
        b.iter(|| {
            let query = r#"
                SELECT 
                    a.id, a.name, a.status,
                    COUNT(e.id) as execution_count,
                    AVG(e.duration_ms) as avg_duration
                FROM agents a
                LEFT JOIN executions e ON e.agent_id = a.id
                WHERE a.user_id = $1
                    AND a.created_at > $2
                    AND a.status IN ('active', 'idle')
                GROUP BY a.id, a.name, a.status
                ORDER BY a.created_at DESC
                LIMIT $3 OFFSET $4
            "#;
            black_box(query)
        });
    });
    
    // Dynamic query building with filters
    group.bench_function("dynamic_filter_query", |b| {
        let filters = vec![
            ("status", "active"),
            ("type", "assistant"),
            ("created_after", "2025-01-01"),
        ];
        b.iter(|| {
            let mut query = String::from("SELECT * FROM agents WHERE 1=1");
            let mut param_idx = 1;
            
            for (field, _) in &filters {
                query.push_str(&format!(" AND {} = ${}", field, param_idx));
                param_idx += 1;
            }
            
            black_box(query)
        });
    });
    
    group.finish();
}

/// Benchmark data serialization for database operations
fn bench_data_serialization(c: &mut Criterion) {
    use serde::{Deserialize, Serialize};
    use chrono::{DateTime, Utc};
    
    #[derive(Debug, Serialize, Deserialize)]
    struct Agent {
        id: uuid::Uuid,
        name: String,
        description: Option<String>,
        status: String,
        config: serde_json::Value,
        created_at: DateTime<Utc>,
        updated_at: DateTime<Utc>,
    }
    
    let mut group = c.benchmark_group("data_serialization");
    group.measurement_time(Duration::from_secs(10));
    
    let sample_agent = Agent {
        id: uuid::Uuid::new_v4(),
        name: "Test Agent".to_string(),
        description: Some("A test agent for benchmarking database operations".to_string()),
        status: "active".to_string(),
        config: serde_json::json!({
            "model": "gpt-4-turbo",
            "maxTokens": 4096,
            "temperature": 0.7,
            "tools": ["code_interpreter", "retrieval"]
        }),
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };
    
    // Serialize to JSON (common for JSONB columns)
    group.bench_function("agent_to_json", |b| {
        b.iter(|| {
            serde_json::to_string(black_box(&sample_agent)).unwrap()
        });
    });
    
    // Deserialize from JSON
    let agent_json = serde_json::to_string(&sample_agent).unwrap();
    group.bench_function("agent_from_json", |b| {
        b.iter(|| {
            serde_json::from_str::<Agent>(black_box(&agent_json)).unwrap()
        });
    });
    
    // Batch serialization
    let agents: Vec<Agent> = (0..100)
        .map(|i| Agent {
            id: uuid::Uuid::new_v4(),
            name: format!("Agent {}", i),
            description: Some(format!("Description for agent {}", i)),
            status: if i % 2 == 0 { "active" } else { "idle" }.to_string(),
            config: serde_json::json!({"index": i}),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        })
        .collect();
    
    group.throughput(Throughput::Elements(100));
    group.bench_function("batch_agents_to_json", |b| {
        b.iter(|| {
            serde_json::to_string(black_box(&agents)).unwrap()
        });
    });
    
    group.finish();
}

/// Benchmark connection pool operations simulation
fn bench_connection_pool(c: &mut Criterion) {
    use std::sync::Arc;
    use std::collections::VecDeque;
    use std::sync::Mutex;
    
    let mut group = c.benchmark_group("connection_pool");
    group.measurement_time(Duration::from_secs(10));
    
    // Simulate a connection pool
    struct MockConnection {
        id: usize,
    }
    
    struct MockPool {
        connections: Mutex<VecDeque<MockConnection>>,
        size: usize,
    }
    
    impl MockPool {
        fn new(size: usize) -> Self {
            let connections: VecDeque<MockConnection> = (0..size)
                .map(|i| MockConnection { id: i })
                .collect();
            Self {
                connections: Mutex::new(connections),
                size,
            }
        }
        
        fn acquire(&self) -> Option<MockConnection> {
            self.connections.lock().unwrap().pop_front()
        }
        
        fn release(&self, conn: MockConnection) {
            self.connections.lock().unwrap().push_back(conn);
        }
    }
    
    let pool = Arc::new(MockPool::new(10));
    
    group.bench_function("pool_acquire_release", |b| {
        b.iter(|| {
            if let Some(conn) = pool.acquire() {
                // Simulate using connection
                black_box(conn.id);
                pool.release(conn);
            }
        });
    });
    
    // Benchmark connection validation (ping simulation)
    group.bench_function("connection_validate", |b| {
        b.iter(|| {
            // Simulate a simple validation check
            let is_valid = true; // Would actually ping database
            black_box(is_valid)
        });
    });
    
    group.finish();
}

/// Benchmark caching operations
fn bench_caching(c: &mut Criterion) {
    use std::collections::HashMap;
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;
    
    let mut group = c.benchmark_group("caching");
    group.measurement_time(Duration::from_secs(10));
    
    // Cache key generation
    group.bench_function("cache_key_generation", |b| {
        let user_id = uuid::Uuid::new_v4();
        let entity_type = "agent";
        let entity_id = uuid::Uuid::new_v4();
        
        b.iter(|| {
            let key = format!("{}:{}:{}", user_id, entity_type, entity_id);
            black_box(key)
        });
    });
    
    // Hash-based cache key
    group.bench_function("cache_key_hash", |b| {
        let params = ("user-123", "agent", "agent-456", "v1");
        b.iter(|| {
            let mut hasher = DefaultHasher::new();
            params.hash(&mut hasher);
            let hash = hasher.finish();
            black_box(hash)
        });
    });
    
    // In-memory cache operations
    let mut cache: HashMap<String, String> = HashMap::new();
    for i in 0..1000 {
        cache.insert(
            format!("key_{}", i),
            format!("value_{}", i),
        );
    }
    
    let existing_key = "key_500".to_string();
    let nonexistent_key = "key_9999".to_string();
    
    group.bench_function("cache_hit", |b| {
        b.iter(|| {
            cache.get(black_box(&existing_key))
        });
    });
    
    group.bench_function("cache_miss", |b| {
        b.iter(|| {
            cache.get(black_box(&nonexistent_key))
        });
    });
    
    // Cache with TTL check simulation
    use chrono::{DateTime, Utc, Duration as ChronoDuration};
    
    struct CacheEntry {
        value: String,
        expires_at: DateTime<Utc>,
    }
    
    let mut ttl_cache: HashMap<String, CacheEntry> = HashMap::new();
    let now = Utc::now();
    
    for i in 0..1000 {
        ttl_cache.insert(
            format!("key_{}", i),
            CacheEntry {
                value: format!("value_{}", i),
                expires_at: now + ChronoDuration::hours(1),
            },
        );
    }
    
    group.bench_function("cache_with_ttl_check", |b| {
        let key = "key_500".to_string();
        b.iter(|| {
            if let Some(entry) = ttl_cache.get(black_box(&key)) {
                if entry.expires_at > Utc::now() {
                    Some(&entry.value)
                } else {
                    None
                }
            } else {
                None
            }
        });
    });
    
    group.finish();
}

/// Benchmark batch operations
fn bench_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations");
    group.measurement_time(Duration::from_secs(10));
    
    // Batch ID generation
    let batch_sizes = [10, 100, 1000];
    
    for size in batch_sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("uuid_batch_generate", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let ids: Vec<uuid::Uuid> = (0..size)
                        .map(|_| uuid::Uuid::new_v4())
                        .collect();
                    black_box(ids)
                });
            },
        );
    }
    
    // Batch INSERT value building
    for size in batch_sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("insert_values_build", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut values = String::with_capacity(size * 50);
                    for i in 0..size {
                        if i > 0 {
                            values.push_str(", ");
                        }
                        let idx = i * 4;
                        values.push_str(&format!(
                            "(${}, ${}, ${}, ${})",
                            idx + 1, idx + 2, idx + 3, idx + 4
                        ));
                    }
                    black_box(values)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark vector operations for pgvector
fn bench_vector_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_operations");
    group.measurement_time(Duration::from_secs(10));
    
    // Vector dimensions commonly used
    let dimensions = [384, 768, 1536, 3072]; // Common embedding sizes
    
    // Vector generation
    for dim in dimensions {
        group.bench_with_input(
            BenchmarkId::new("vector_generate", dim),
            &dim,
            |b, &dim| {
                b.iter(|| {
                    let vec: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
                    black_box(vec)
                });
            },
        );
    }
    
    // Vector normalization (L2)
    let vec_1536: Vec<f32> = (0..1536).map(|_| rand::random::<f32>()).collect();
    
    group.bench_function("vector_l2_normalize_1536", |b| {
        b.iter(|| {
            let magnitude: f32 = vec_1536.iter().map(|x| x * x).sum::<f32>().sqrt();
            let normalized: Vec<f32> = vec_1536.iter().map(|x| x / magnitude).collect();
            black_box(normalized)
        });
    });
    
    // Cosine similarity calculation
    let vec_a: Vec<f32> = (0..1536).map(|_| rand::random::<f32>()).collect();
    let vec_b: Vec<f32> = (0..1536).map(|_| rand::random::<f32>()).collect();
    
    group.bench_function("vector_cosine_similarity_1536", |b| {
        b.iter(|| {
            let dot_product: f32 = vec_a.iter().zip(&vec_b).map(|(a, b)| a * b).sum();
            let mag_a: f32 = vec_a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let mag_b: f32 = vec_b.iter().map(|x| x * x).sum::<f32>().sqrt();
            let similarity = dot_product / (mag_a * mag_b);
            black_box(similarity)
        });
    });
    
    // Vector to string for storage
    group.bench_function("vector_to_pgvector_string", |b| {
        let vec: Vec<f32> = (0..1536).map(|i| i as f32 * 0.001).collect();
        b.iter(|| {
            let s = format!(
                "[{}]",
                vec.iter()
                    .map(|v| format!("{:.6}", v))
                    .collect::<Vec<_>>()
                    .join(",")
            );
            black_box(s)
        });
    });
    
    group.finish();
}

/// Benchmark graph query patterns (for Neo4j)
fn bench_graph_query_building(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_queries");
    group.measurement_time(Duration::from_secs(10));
    
    // Simple node match
    group.bench_function("cypher_simple_match", |b| {
        b.iter(|| {
            let query = "MATCH (a:Agent {id: $id}) RETURN a";
            black_box(query)
        });
    });
    
    // Relationship query
    group.bench_function("cypher_relationship_query", |b| {
        b.iter(|| {
            let query = r#"
                MATCH (a:Agent)-[r:EXECUTES]->(e:Execution)
                WHERE a.id = $agentId
                RETURN a, r, e
                ORDER BY e.startedAt DESC
                LIMIT $limit
            "#;
            black_box(query)
        });
    });
    
    // Complex traversal
    group.bench_function("cypher_complex_traversal", |b| {
        b.iter(|| {
            let query = r#"
                MATCH path = (start:Agent {id: $startId})-[:CALLS*1..5]->(end:Agent)
                WHERE end.status = 'active'
                WITH path, relationships(path) as rels
                RETURN 
                    [n IN nodes(path) | n.name] as agentNames,
                    length(path) as depth,
                    reduce(cost = 0, r IN rels | cost + r.weight) as totalCost
                ORDER BY totalCost ASC
                LIMIT 10
            "#;
            black_box(query)
        });
    });
    
    // Dynamic Cypher query building
    group.bench_function("cypher_dynamic_build", |b| {
        let labels = vec!["Agent", "Tool"];
        let properties = vec![("status", "active"), ("type", "assistant")];
        
        b.iter(|| {
            let labels_str = labels.join(":");
            let props_str: String = properties
                .iter()
                .enumerate()
                .map(|(i, (k, _))| format!("{}: ${}", k, i))
                .collect::<Vec<_>>()
                .join(", ");
            
            let query = format!(
                "MATCH (n:{} {{{}}}) RETURN n",
                labels_str, props_str
            );
            black_box(query)
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_query_building,
    bench_data_serialization,
    bench_connection_pool,
    bench_caching,
    bench_batch_operations,
    bench_vector_operations,
    bench_graph_query_building,
);

criterion_main!(benches);
