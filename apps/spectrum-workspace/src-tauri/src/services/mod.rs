//! Embedded Backend Services
//!
//! This module provides embedded HTTP API server, database connections,
//! and ML service integration that runs directly within the Tauri desktop app.
//!
//! Everything starts automatically when the desktop app launches - no separate
//! backend services needed!

mod api_gateway;
mod database;
mod ml_service;

pub use api_gateway::ApiGateway;
pub use database::Database;
pub use ml_service::MlService;

use anyhow::Result;
use tokio::task::JoinHandle;
use tracing::{error, info};

/// Manages all embedded backend services
pub struct ServiceManager {
    api_gateway: Option<JoinHandle<()>>,
    ml_service: Option<JoinHandle<()>>,
}

impl ServiceManager {
    pub fn new() -> Self {
        Self {
            api_gateway: None,
            ml_service: None,
        }
    }

    /// Start all embedded services
    pub async fn start_all(&mut self, app_handle: &tauri::AppHandle) -> Result<()> {
        info!("Starting embedded backend services...");

        // Initialize database
        let db = Database::init(app_handle).await?;
        info!("✓ Database initialized");

        // Start API Gateway on port 16080
        let api_gateway = ApiGateway::new(db.clone());
        let api_handle = tokio::spawn(async move {
            if let Err(e) = api_gateway.start().await {
                error!("API Gateway error: {}", e);
            }
        });
        self.api_gateway = Some(api_handle);
        info!("✓ API Gateway started on http://localhost:16080");

        // Start ML Service on port 16081
        let ml_service = MlService::new(db);
        let ml_handle = tokio::spawn(async move {
            if let Err(e) = ml_service.start().await {
                error!("ML Service error: {}", e);
            }
        });
        self.ml_service = Some(ml_handle);
        info!("✓ ML Service started on http://localhost:16081");

        info!("🚀 All backend services running!");
        Ok(())
    }

    /// Gracefully shutdown all services
    pub async fn shutdown(&mut self) {
        info!("Shutting down embedded backend services...");

        if let Some(handle) = self.api_gateway.take() {
            handle.abort();
        }

        if let Some(handle) = self.ml_service.take() {
            handle.abort();
        }

        info!("✓ All services stopped");
    }
}

impl Drop for ServiceManager {
    fn drop(&mut self) {
        if self.api_gateway.is_some() || self.ml_service.is_some() {
            tracing::warn!("ServiceManager dropped while services still running");
        }
    }
}
