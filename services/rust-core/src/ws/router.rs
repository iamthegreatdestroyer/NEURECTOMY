//! WebSocket Router
//!
//! Axum router for WebSocket endpoints:
//! - Connection upgrade handling
//! - Message routing
//! - Error handling
//!
//! @STREAM - HTTP to WebSocket bridge

use std::sync::Arc;

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        ConnectInfo, State,
    },
    http::HeaderMap,
    response::IntoResponse,
    routing::get,
    Router,
};
use futures_util::{SinkExt, StreamExt};
use std::net::SocketAddr;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use super::connection::WsConnection;
use super::handlers::process_command;
use super::manager::ConnectionManager;
use super::messages::{WsCommand, WsEvent};
use crate::auth::JwtService;
use crate::db::DatabaseConnections;

/// WebSocket router state
#[derive(Clone)]
pub struct WsState {
    pub manager: Arc<ConnectionManager>,
    pub jwt_service: JwtService,
    pub db: Arc<DatabaseConnections>,
}

/// Create WebSocket router
pub fn create_ws_router(
    manager: Arc<ConnectionManager>,
    jwt_service: JwtService,
    db: Arc<DatabaseConnections>,
) -> Router {
    let state = WsState {
        manager,
        jwt_service,
        db,
    };

    Router::new()
        .route("/", get(ws_handler))
        .route("/agent/:agent_id", get(ws_agent_handler))
        .route("/container/:container_id/logs", get(ws_container_logs_handler))
        .route("/training/:job_id", get(ws_training_handler))
        .with_state(state)
}

// =============================================
// WebSocket Handlers
// =============================================

/// Main WebSocket handler
async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<WsState>,
    headers: HeaderMap,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
) -> impl IntoResponse {
    let user_agent = headers
        .get("user-agent")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let client_ip = Some(addr.ip().to_string());

    ws.on_upgrade(move |socket| handle_socket(socket, state, client_ip, user_agent))
}

/// Agent-specific WebSocket handler
async fn ws_agent_handler(
    ws: WebSocketUpgrade,
    State(state): State<WsState>,
    axum::extract::Path(agent_id): axum::extract::Path<uuid::Uuid>,
    headers: HeaderMap,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
) -> impl IntoResponse {
    let user_agent = headers
        .get("user-agent")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let client_ip = Some(addr.ip().to_string());

    ws.on_upgrade(move |socket| {
        handle_socket_with_subscription(
            socket,
            state,
            client_ip,
            user_agent,
            super::messages::SubscriptionType::Agent,
            agent_id,
        )
    })
}

/// Container logs WebSocket handler
async fn ws_container_logs_handler(
    ws: WebSocketUpgrade,
    State(state): State<WsState>,
    axum::extract::Path(container_id): axum::extract::Path<uuid::Uuid>,
    headers: HeaderMap,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
) -> impl IntoResponse {
    let user_agent = headers
        .get("user-agent")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let client_ip = Some(addr.ip().to_string());

    ws.on_upgrade(move |socket| {
        handle_socket_with_subscription(
            socket,
            state,
            client_ip,
            user_agent,
            super::messages::SubscriptionType::ContainerLogs,
            container_id,
        )
    })
}

/// Training progress WebSocket handler
async fn ws_training_handler(
    ws: WebSocketUpgrade,
    State(state): State<WsState>,
    axum::extract::Path(job_id): axum::extract::Path<uuid::Uuid>,
    headers: HeaderMap,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
) -> impl IntoResponse {
    let user_agent = headers
        .get("user-agent")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let client_ip = Some(addr.ip().to_string());

    ws.on_upgrade(move |socket| {
        handle_socket_with_subscription(
            socket,
            state,
            client_ip,
            user_agent,
            super::messages::SubscriptionType::Training,
            job_id,
        )
    })
}

// =============================================
// Socket Handlers
// =============================================

/// Handle WebSocket connection
async fn handle_socket(
    socket: WebSocket,
    state: WsState,
    client_ip: Option<String>,
    user_agent: Option<String>,
) {
    // Create outbound channel for sending to client
    let (tx, rx) = mpsc::unbounded_channel::<WsEvent>();

    // Create connection
    let connection = WsConnection::new(tx).with_metadata(client_ip.clone(), user_agent.clone());
    let connection_id = connection.id;

    // Register connection
    state.manager.register(connection).await;

    info!(
        "WebSocket connected: {} from {:?}",
        connection_id, client_ip
    );

    // Process connection
    process_socket(socket, state.clone(), connection_id, rx).await;

    // Unregister on disconnect
    state.manager.unregister(connection_id).await;

    info!("WebSocket disconnected: {}", connection_id);
}

/// Handle WebSocket with auto-subscription
async fn handle_socket_with_subscription(
    socket: WebSocket,
    state: WsState,
    client_ip: Option<String>,
    user_agent: Option<String>,
    subscription_type: super::messages::SubscriptionType,
    resource_id: uuid::Uuid,
) {
    // Create outbound channel
    let (tx, rx) = mpsc::unbounded_channel::<WsEvent>();

    // Create connection with auto-subscription
    let mut connection = WsConnection::new(tx).with_metadata(client_ip.clone(), user_agent.clone());
    let connection_id = connection.id;

    // Auto-subscribe (will require auth before actually receiving events)
    connection.subscribe(subscription_type, resource_id);

    // Register connection
    state.manager.register(connection).await;

    info!(
        "WebSocket connected: {} (auto-sub {:?}/{})",
        connection_id, subscription_type, resource_id
    );

    // Process connection
    process_socket(socket, state.clone(), connection_id, rx).await;

    // Unregister on disconnect
    state.manager.unregister(connection_id).await;
}

/// Main socket processing loop
async fn process_socket(
    socket: WebSocket,
    state: WsState,
    connection_id: uuid::Uuid,
    mut rx: mpsc::UnboundedReceiver<WsEvent>,
) {
    let (mut ws_sender, mut ws_receiver) = socket.split();

    // Spawn task to forward events to WebSocket
    let send_task = tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            match serde_json::to_string(&event) {
                Ok(json) => {
                    if ws_sender.send(Message::Text(json.into())).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    error!("Failed to serialize event: {}", e);
                }
            }
        }
    });

    // Process incoming messages
    while let Some(result) = ws_receiver.next().await {
        match result {
            Ok(msg) => {
                if let Err(e) = handle_message(msg, &state, connection_id).await {
                    warn!("Message handling error: {}", e);
                }
            }
            Err(e) => {
                warn!("WebSocket error: {}", e);
                break;
            }
        }
    }

    // Clean up
    send_task.abort();
}

/// Handle incoming WebSocket message
async fn handle_message(
    msg: Message,
    state: &WsState,
    connection_id: uuid::Uuid,
) -> Result<(), String> {
    match msg {
        Message::Text(text) => {
            state.manager.record_message_received().await;

            // Parse command
            let command: WsCommand = serde_json::from_str(&text)
                .map_err(|e| format!("Invalid message format: {}", e))?;

            debug!("Received command from {}: {:?}", connection_id, command);

            // Get connection for processing
            let conn = state
                .manager
                .get(connection_id)
                .await
                .ok_or("Connection not found")?;

            // Process command
            let mut conn_guard = conn.write().await;
            let result = process_command(
                command,
                &mut conn_guard,
                &state.manager,
                &state.jwt_service,
            )
            .await;

            // Send response
            match result {
                Ok(Some(event)) => {
                    let _ = conn_guard.send(event);
                }
                Ok(None) => {
                    // No response needed
                }
                Err(event) => {
                    let _ = conn_guard.send(event);
                }
            }

            Ok(())
        }
        Message::Binary(data) => {
            // Binary messages not supported yet
            warn!(
                "Received unsupported binary message ({} bytes)",
                data.len()
            );
            Ok(())
        }
        Message::Ping(data) => {
            // Handled automatically by axum
            debug!("Received ping ({} bytes)", data.len());
            Ok(())
        }
        Message::Pong(_) => {
            // Update connection pong timestamp
            if let Some(conn) = state.manager.get(connection_id).await {
                conn.write().await.record_pong();
            }
            Ok(())
        }
        Message::Close(_) => {
            info!("WebSocket close requested: {}", connection_id);
            Err("Connection closed".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    // WebSocket tests would require more complex setup with actual WebSocket connections
    // These are typically done as integration tests
}
