//! ML Service Integration
//!
//! Provides ML/AI capabilities either by:
//! 1. Spawning Python ML service as subprocess (full features)
//! 2. Providing Rust-native ML capabilities (lightweight)
//!
//! For now, we spawn the Python service. Future: migrate to Rust ML libraries.

use axum::{
    routing::post,
    Router,
};
use std::process::{Child, Command};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn, error};

use super::database::Database;

pub struct MlService {
    db: Database,
    python_process: Arc<Mutex<Option<Child>>>,
}

impl MlService {
    pub fn new(db: Database) -> Self {
        Self {
            db,
            python_process: Arc::new(Mutex::new(None)),
        }
    }

    /// Start ML service (spawns Python subprocess or starts Rust server)
    pub async fn start(self) -> anyhow::Result<()> {
        info!("Starting ML Service...");

        // Try to spawn Python ML service as subprocess
        if let Ok(child) = self.spawn_python_service().await {
            let mut process_lock = self.python_process.lock().await;
            *process_lock = Some(child);
            info!("✓ Python ML Service spawned successfully");
        } else {
            warn!("Could not spawn Python ML service, starting lightweight Rust server");
            self.start_rust_server().await?;
        }

        Ok(())
    }

    /// Spawn Python ML service as subprocess
    async fn spawn_python_service(&self) -> anyhow::Result<Child> {
        // Find Python executable
        let python_cmd = if cfg!(windows) {
            "python"
        } else {
            "python3"
        };

        // Get the ML service directory
        let ml_service_dir = std::env::current_exe()?
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Could not determine exe directory"))?
            .join("../../services/ml-service");

        info!("Spawning Python ML service from: {:?}", ml_service_dir);

        // Spawn Python process
        let child = Command::new(python_cmd)
            .arg("-m")
            .arg("uvicorn")
            .arg("main:app")
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg("16081")
            .arg("--reload")
            .current_dir(&ml_service_dir)
            .spawn()?;

        Ok(child)
    }

    /// Start lightweight Rust-based ML server (fallback)
    async fn start_rust_server(self) -> anyhow::Result<()> {
        info!("Starting lightweight Rust ML server on http://localhost:16081");

        let app = Router::new()
            .route("/health", axum::routing::get(ml_health_handler))
            .route("/api/embeddings", post(ml_embeddings_handler))
            .route("/api/completions", post(ml_completions_handler));

        let listener = tokio::net::TcpListener::bind("127.0.0.1:16081").await?;
        
        axum::serve(listener, app)
            .await
            .map_err(|e| anyhow::anyhow!("ML Service server error: {}", e))?;

        Ok(())
    }
}

impl Drop for MlService {
    fn drop(&mut self) {
        // Kill Python subprocess if running
        if let Ok(mut process_lock) = self.python_process.try_lock() {
            if let Some(mut child) = process_lock.take() {
                info!("Stopping Python ML service...");
                let _ = child.kill();
            }
        }
    }
}

// Lightweight Rust ML handlers (fallback)

async fn ml_health_handler() -> &'static str {
    "ML Service OK (Rust fallback mode)"
}

async fn ml_embeddings_handler() -> &'static str {
    "Embeddings endpoint (not yet implemented in Rust)"
}

async fn ml_completions_handler() -> &'static str {
    "Completions endpoint (not yet implemented in Rust)"
}
