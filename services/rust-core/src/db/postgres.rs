//! PostgreSQL Connection Pool
//! @VERTEX PostgreSQL Database Management

use sqlx::{postgres::PgPoolOptions, PgPool, Row};
use std::time::Duration;

use super::{DatabaseError, DbResult};

/// PostgreSQL connection pool wrapper
#[derive(Clone)]
pub struct PostgresPool {
    pool: PgPool,
}

impl PostgresPool {
    /// Create a new PostgreSQL connection pool
    pub async fn new(database_url: &str) -> DbResult<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(20)
            .min_connections(5)
            .acquire_timeout(Duration::from_secs(10))
            .idle_timeout(Duration::from_secs(600))
            .max_lifetime(Duration::from_secs(1800))
            .connect(database_url)
            .await
            .map_err(DatabaseError::Postgres)?;
        
        Ok(Self { pool })
    }
    
    /// Get a reference to the underlying pool
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }
    
    /// Run database migrations
    pub async fn run_migrations(&self) -> DbResult<()> {
        sqlx::migrate!("./migrations")
            .run(&self.pool)
            .await
            .map_err(|e| DatabaseError::Migration(e.to_string()))?;
        Ok(())
    }
    
    /// Health check
    pub async fn health_check(&self) -> DbResult<()> {
        sqlx::query("SELECT 1")
            .fetch_one(&self.pool)
            .await
            .map_err(DatabaseError::Postgres)?;
        Ok(())
    }
    
    /// Get connection pool statistics
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            size: self.pool.size(),
            idle: self.pool.num_idle(),
        }
    }
    
    /// Close the connection pool
    pub async fn close(&self) {
        self.pool.close().await;
    }
}

/// Pool statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct PoolStats {
    pub size: u32,
    pub idle: usize,
}

// Re-export sqlx types for convenience
pub use sqlx::{Error as SqlxError, FromRow, Row as SqlxRow};
