//! Database Module for NEURECTOMY
//! @VERTEX Database Connection Management
//! 
//! This module provides connection pools and utilities for:
//! - PostgreSQL (primary relational database)
//! - Neo4j (graph database for knowledge graphs)
//! - Redis (caching and session storage)

pub mod postgres;
pub mod neo4j;
pub mod redis;
pub mod models;

pub use postgres::PostgresPool;
pub use neo4j::Neo4jPool;
pub use redis::RedisPool;

use std::sync::Arc;
use thiserror::Error;

/// Database errors
#[derive(Error, Debug)]
pub enum DatabaseError {
    #[error("PostgreSQL error: {0}")]
    Postgres(#[from] sqlx::Error),
    
    #[error("Neo4j error: {0}")]
    Neo4j(#[from] neo4rs::Error),
    
    #[error("Redis error: {0}")]
    Redis(#[from] ::redis::RedisError),
    
    #[error("Connection pool error: {0}")]
    Pool(String),
    
    #[error("Migration error: {0}")]
    Migration(String),
    
    #[error("Configuration error: {0}")]
    Config(String),
}

/// Result type for database operations
pub type DbResult<T> = Result<T, DatabaseError>;

/// Unified database connections struct
#[derive(Clone)]
pub struct DatabaseConnections {
    pub postgres: Arc<PostgresPool>,
    pub neo4j: Arc<Neo4jPool>,
    pub redis: Arc<RedisPool>,
}

impl DatabaseConnections {
    /// Create new database connections from configuration
    pub async fn new(config: &crate::config::DatabaseConfig) -> DbResult<Self> {
        tracing::info!("Initializing database connections...");
        
        let postgres = Arc::new(PostgresPool::new(&config.postgres_url).await?);
        tracing::info!("PostgreSQL connection pool established");
        
        let neo4j = Arc::new(Neo4jPool::new(&config.neo4j_url).await?);
        tracing::info!("Neo4j connection pool established");
        
        let redis = Arc::new(RedisPool::new(&config.redis_url).await?);
        tracing::info!("Redis connection pool established");
        
        Ok(Self {
            postgres,
            neo4j,
            redis,
        })
    }
    
    /// Run PostgreSQL migrations
    pub async fn run_migrations(&self) -> DbResult<()> {
        tracing::info!("Running PostgreSQL migrations...");
        self.postgres.run_migrations().await?;
        tracing::info!("PostgreSQL migrations completed");
        Ok(())
    }
    
    /// Health check for all database connections
    pub async fn health_check(&self) -> DbResult<DatabaseHealth> {
        let postgres_healthy = self.postgres.health_check().await.is_ok();
        let neo4j_healthy = self.neo4j.health_check().await.is_ok();
        let redis_healthy = self.redis.health_check().await.is_ok();
        
        Ok(DatabaseHealth {
            postgres: postgres_healthy,
            neo4j: neo4j_healthy,
            redis: redis_healthy,
            all_healthy: postgres_healthy && neo4j_healthy && redis_healthy,
        })
    }
    
    /// Close all database connections gracefully
    pub async fn close(&self) -> DbResult<()> {
        tracing::info!("Closing database connections...");
        
        self.postgres.close().await;
        self.redis.close().await;
        // Neo4j doesn't have explicit close
        
        tracing::info!("Database connections closed");
        Ok(())
    }
}

/// Database health status
#[derive(Debug, Clone, serde::Serialize)]
pub struct DatabaseHealth {
    pub postgres: bool,
    pub neo4j: bool,
    pub redis: bool,
    pub all_healthy: bool,
}
