//! GraphQL API module for NEURECTOMY Core
//!
//! Implements the GraphQL schema using async-graphql with:
//! - Queries for agents, containers, training jobs, conversations
//! - Mutations for CRUD operations
//! - Subscriptions for real-time updates
//!
//! @APEX @SYNAPSE - Integration Engineering & API Design

pub mod schema;
pub mod types;
pub mod queries;
pub mod mutations;
pub mod subscriptions;
pub mod dataloaders;
pub mod context;

pub use schema::{build_schema, NeurectomySchema};
pub use context::GraphQLContext;

use async_graphql_axum::{GraphQLRequest, GraphQLResponse, GraphQL, GraphQLProtocol};
use axum::{
    extract::State,
    response::{Html, IntoResponse},
    routing::get,
    Router,
};

use crate::db::DatabaseConnections;

/// GraphQL state containing schema and database connections
#[derive(Clone)]
pub struct GraphQLState {
    pub schema: NeurectomySchema,
}

impl GraphQLState {
    pub fn new(db: DatabaseConnections) -> Self {
        let schema = build_schema(db);
        Self { schema }
    }
}

/// GraphQL query handler
pub async fn graphql_handler(
    State(state): State<GraphQLState>,
    req: GraphQLRequest,
) -> GraphQLResponse {
    state.schema.execute(req.into_inner()).await.into()
}

/// GraphQL subscription handler (WebSocket)
pub async fn graphql_subscription_handler(
    State(state): State<GraphQLState>,
    protocol: GraphQLProtocol,
    ws: axum::extract::WebSocketUpgrade,
) -> impl IntoResponse {
    let schema = state.schema.clone();
    ws.protocols(["graphql-transport-ws", "graphql-ws"])
        .on_upgrade(move |socket| async move {
            let _ = async_graphql_axum::GraphQLWebSocket::new(socket, schema, protocol)
                .serve()
                .await;
        })
}

/// GraphQL Playground HTML interface
pub async fn graphql_playground() -> impl IntoResponse {
    Html(
        async_graphql::http::playground_source(
            async_graphql::http::GraphQLPlaygroundConfig::new("/graphql")
                .subscription_endpoint("/graphql/ws"),
        ),
    )
}

/// Create GraphQL router
pub fn create_graphql_router(db: DatabaseConnections) -> Router {
    let state = GraphQLState::new(db);

    Router::new()
        .route("/", get(graphql_playground).post(graphql_handler))
        .route("/ws", get(graphql_subscription_handler))
        .with_state(state)
}
