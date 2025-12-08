//! API Gateway
//!
//! Embedded HTTP/REST API server running on port 16080
//! Provides all backend endpoints needed by the frontend

use axum::{
    extract::State,
    http::{header, Method, StatusCode},
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tracing::{info, error};

use super::database::Database;

#[derive(Clone)]
pub struct ApiGateway {
    db: Database,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    status: String,
    version: String,
    timestamp: i64,
}

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct Project {
    id: String,
    name: String,
    description: Option<String>,
    created_at: i64,
    updated_at: i64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateProjectRequest {
    name: String,
    description: Option<String>,
}

impl ApiGateway {
    pub fn new(db: Database) -> Self {
        Self { db }
    }

    /// Start the API gateway server
    pub async fn start(self) -> anyhow::Result<()> {
        let app_state = Arc::new(self);

        // Configure CORS
        let cors = CorsLayer::new()
            .allow_origin(Any)
            .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE, Method::OPTIONS])
            .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION]);

        // Build router
        let app = Router::new()
            // Health check
            .route("/health", get(health_handler))
            .route("/api/health", get(health_handler))
            
            // Project endpoints
            .route("/api/projects", get(list_projects).post(create_project))
            .route("/api/projects/:id", get(get_project))
            
            // Agent endpoints
            .route("/api/agents", get(list_agents))
            .route("/api/agents/:id", get(get_agent))
            
            // GraphQL endpoint (placeholder)
            .route("/graphql", post(graphql_handler))
            
            .layer(cors)
            .with_state(app_state);

        info!("Starting API Gateway on http://localhost:16080");

        // Bind to localhost:16080
        let listener = tokio::net::TcpListener::bind("127.0.0.1:16080").await?;
        
        axum::serve(listener, app)
            .await
            .map_err(|e| anyhow::anyhow!("API Gateway server error: {}", e))?;

        Ok(())
    }
}

// Handler functions

async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: chrono::Utc::now().timestamp(),
    })
}

async fn list_projects(
    State(gateway): State<Arc<ApiGateway>>,
) -> Result<Json<Vec<Project>>, StatusCode> {
    let projects = sqlx::query_as::<_, Project>(
        r#"
        SELECT id, name, description, created_at, updated_at
        FROM projects
        ORDER BY created_at DESC
        "#
    )
    .fetch_all(gateway.db.pool())
    .await
    .map_err(|e| {
        error!("Failed to fetch projects: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    Ok(Json(projects))
}

async fn create_project(
    State(gateway): State<Arc<ApiGateway>>,
    Json(req): Json<CreateProjectRequest>,
) -> Result<Json<Project>, StatusCode> {
    let id = uuid::Uuid::new_v4().to_string();
    let now = chrono::Utc::now().timestamp();

    let project = sqlx::query_as::<_, Project>(
        r#"
        INSERT INTO projects (id, name, description, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        RETURNING id, name, description, created_at, updated_at
        "#
    )
    .bind(&id)
    .bind(&req.name)
    .bind(&req.description)
    .bind(&now)
    .bind(&now)
    .fetch_one(gateway.db.pool())
    .await
    .map_err(|e| {
        error!("Failed to create project: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    Ok(Json(project))
}

async fn get_project(
    State(gateway): State<Arc<ApiGateway>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<Project>, StatusCode> {
    let project = sqlx::query_as::<_, Project>(
        r#"
        SELECT id, name, description, created_at, updated_at
        FROM projects
        WHERE id = $1
        "#
    )
    .bind(&id)
    .fetch_one(gateway.db.pool())
    .await
    .map_err(|e| {
        error!("Failed to fetch project: {}", e);
        StatusCode::NOT_FOUND
    })?;

    Ok(Json(project))
}

async fn list_agents(
    State(_gateway): State<Arc<ApiGateway>>,
) -> Result<Json<Vec<String>>, StatusCode> {
    // Placeholder - implement agent listing
    Ok(Json(vec![]))
}

async fn get_agent(
    State(_gateway): State<Arc<ApiGateway>>,
    axum::extract::Path(_id): axum::extract::Path<String>,
) -> Result<Json<String>, StatusCode> {
    // Placeholder - implement agent retrieval
    Ok(Json("Agent details".to_string()))
}

async fn graphql_handler(
    State(_gateway): State<Arc<ApiGateway>>,
    body: String,
) -> impl IntoResponse {
    // Placeholder for GraphQL endpoint
    // TODO: Integrate with async-graphql or similar
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "data": null,
            "errors": [{
                "message": "GraphQL endpoint not yet implemented"
            }]
        }))
    )
}
