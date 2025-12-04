//! NEURECTOMY Core Backend Service
//!
//! This is the main entry point for the NEURECTOMY backend, providing:
//! - GraphQL API for agent management
//! - WebSocket connections for real-time updates
//! - REST API for health checks and metrics
//! - Event-driven architecture with NATS

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::get,
    Router,
};
use serde::Serialize;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod config;
pub mod db;
pub mod graphql;
pub mod auth;
pub mod ws;

use config::AppConfig;
use db::DatabaseConnections;
use graphql::create_graphql_router;

/// Application state shared across all handlers
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<AppConfig>,
    pub db: Arc<DatabaseConnections>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    init_tracing();

    info!("🧠 Starting NEURECTOMY Core Backend...");

    // Load configuration
    let config = AppConfig::load()?;
    let config = Arc::new(config);
    info!("✅ Configuration loaded");
    info!("   Environment: {}", config.server.env);

    // Initialize database connections
    info!("📡 Connecting to databases...");
    let db = match DatabaseConnections::new(&config.database).await {
        Ok(db) => {
            info!("✅ Database connections established");
            Arc::new(db)
        }
        Err(e) => {
            tracing::warn!("⚠️  Database connection failed (will retry): {}", e);
            // In development, we can continue without DB for testing
            if config.server.env != "development" {
                return Err(e.into());
            }
            // Create a placeholder for development
            tracing::warn!("   Running in development mode without database");
            return start_server_without_db(config).await;
        }
    };

    // Run migrations in development
    if config.server.env == "development" {
        if let Err(e) = db.run_migrations().await {
            tracing::warn!("⚠️  Migration warning: {}", e);
        }
    }

    // Create application state
    let state = AppState {
        config: config.clone(),
        db,
    };

    // Build router
    let app = create_router(state);

    // Start server
    let addr: SocketAddr = format!("{}:{}", config.server.host, config.server.port)
        .parse()
        .expect("Invalid server address");

    info!("🚀 NEURECTOMY Core listening on http://{}", addr);
    info!("   Health:   http://{}/health", addr);
    info!("   Ready:    http://{}/ready", addr);
    info!("   GraphQL:  http://{}/graphql", addr);
    info!("   GraphQL WS: ws://{}/graphql/ws", addr);
    info!("   WebSocket: ws://{}/ws", addr);
    info!("   WS Agent:  ws://{}/ws/agent/:agent_id", addr);
    info!("   WS Logs:   ws://{}/ws/container/:id/logs", addr);
    info!("   WS Train:  ws://{}/ws/training/:job_id", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    
    // Graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("👋 NEURECTOMY Core shutting down...");
    Ok(())
}

/// Start server without database (for development/testing)
async fn start_server_without_db(config: Arc<AppConfig>) -> Result<()> {
    let app = Router::new()
        .route("/health", get(health_check_simple))
        .route("/ready", get(|| async { 
            (StatusCode::SERVICE_UNAVAILABLE, "Database not connected")
        }))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .layer(TraceLayer::new_for_http());

    let addr: SocketAddr = format!("{}:{}", config.server.host, config.server.port)
        .parse()
        .expect("Invalid server address");

    info!("🚀 NEURECTOMY Core (limited mode) listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

fn create_router(state: AppState) -> Router {
    // Create GraphQL router with database connections
    let graphql_router = create_graphql_router((*state.db).clone());
    
    // Create WebSocket router
    let ws_manager = ws::ConnectionManager::new();  // Already returns Arc<ConnectionManager>
    let jwt_config = auth::jwt::JwtConfig {
        secret: state.config.server.jwt_secret.clone().unwrap_or_else(|| "neurectomy-dev-secret-key".to_string()),
        ..Default::default()
    };
    let jwt_service = auth::JwtService::new(jwt_config).expect("Failed to initialize JWT service");
    let ws_router = ws::create_ws_router(ws_manager.clone(), jwt_service, state.db.clone());
    
    // Start WebSocket background tasks
    ws_manager.clone().start_heartbeat_task();
    ws_manager.clone().start_cleanup_task();
    
    // Build router with state-dependent routes first
    let app_router = Router::new()
        // Health & Metrics
        .route("/health", get(health_check))
        .route("/ready", get(readiness_check))
        // API info
        .route("/", get(api_info))
        .route("/api", get(api_info))
        .route("/api/v1", get(api_info))
        .with_state(state);
    
    // Merge routers that have their own state
    app_router
        // GraphQL API (has its own GraphQLState)
        .nest("/graphql", graphql_router)
        // WebSocket API (has its own WsState)
        .nest("/ws", ws_router)
        // Middleware
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .layer(TraceLayer::new_for_http())
}

// ===========================================
// Health Check Handlers
// ===========================================

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    version: &'static str,
    service: &'static str,
}

#[derive(Serialize)]
struct ReadinessResponse {
    status: &'static str,
    databases: db::DatabaseHealth,
}

#[derive(Serialize)]
struct ApiInfoResponse {
    name: &'static str,
    version: &'static str,
    description: &'static str,
    endpoints: Vec<&'static str>,
}

async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy",
        version: env!("CARGO_PKG_VERSION"),
        service: "neurectomy-core",
    })
}

async fn health_check_simple() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy",
        version: env!("CARGO_PKG_VERSION"),
        service: "neurectomy-core",
    })
}

async fn readiness_check(
    State(state): State<AppState>,
) -> Result<Json<ReadinessResponse>, (StatusCode, String)> {
    let health = state.db.health_check().await
        .map_err(|e| (StatusCode::SERVICE_UNAVAILABLE, e.to_string()))?;
    
    if !health.all_healthy {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "One or more databases are unhealthy".to_string(),
        ));
    }
    
    Ok(Json(ReadinessResponse {
        status: "ready",
        databases: health,
    }))
}

async fn api_info() -> Json<ApiInfoResponse> {
    Json(ApiInfoResponse {
        name: "NEURECTOMY Core API",
        version: env!("CARGO_PKG_VERSION"),
        description: "Backend service for NEURECTOMY - The Ultimate Agent Development Platform",
        endpoints: vec![
            "GET  /health - Health check",
            "GET  /ready - Readiness check with database status",
            "POST /graphql - GraphQL API",
            "GET  /graphql - GraphQL Playground",
            "WS   /graphql/ws - GraphQL Subscriptions (WebSocket)",
        ],
    })
}

// ===========================================
// Graceful Shutdown
// ===========================================

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    info!("🛑 Shutdown signal received, starting graceful shutdown...");
}

fn init_tracing() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "neurectomy_core=debug,tower_http=debug,sqlx=warn".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
}
