//! Database Module
//!
//! Provides embedded SQLite database for local desktop use.
//! Can be extended to support PostgreSQL for production deployments.

use anyhow::Result;
use sqlx::{SqlitePool, sqlite::SqlitePoolOptions};
use std::path::PathBuf;
use tauri::Manager;
use tracing::info;

#[derive(Clone)]
pub struct Database {
    pool: SqlitePool,
}

impl Database {
    /// Initialize embedded SQLite database
    pub async fn init(app_handle: &tauri::AppHandle) -> Result<Self> {
        let db_path = Self::get_database_path(app_handle)?;
        info!("Initializing database at: {:?}", db_path);

        // Ensure parent directory exists
        if let Some(parent) = db_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        // Create database file with read-write-create mode
        let database_url = format!("sqlite:{}?mode=rwc", db_path.display());
        
        let pool = SqlitePoolOptions::new()
            .max_connections(10)
            .connect(&database_url)
            .await?;

        // Run migrations
        Self::run_migrations(&pool).await?;

        Ok(Self { pool })
    }

    /// Get the database file path (in app data directory)
    fn get_database_path(app_handle: &tauri::AppHandle) -> Result<PathBuf> {
        let mut path = app_handle.path().app_data_dir()
            .map_err(|e| anyhow::anyhow!("Could not determine data directory: {}", e))?;
        
        path.push("neurectomy");
        path.push("neurectomy.db");
        
        Ok(path)
    }

    /// Run database migrations
    async fn run_migrations(pool: &SqlitePool) -> Result<()> {
        info!("Running database migrations...");

        // Create tables
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                metadata TEXT
            )
            "#,
        )
        .execute(pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                config TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            )
            "#,
        )
        .execute(pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS agent_runs (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                status TEXT NOT NULL,
                input TEXT,
                output TEXT,
                started_at INTEGER NOT NULL,
                completed_at INTEGER,
                error TEXT,
                FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE
            )
            "#,
        )
        .execute(pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                metadata TEXT,
                created_at INTEGER NOT NULL
            )
            "#,
        )
        .execute(pool)
        .await?;

        info!("✓ Database migrations completed");
        Ok(())
    }

    /// Get the connection pool
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }
}
