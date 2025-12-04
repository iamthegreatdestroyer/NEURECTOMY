//! WebSocket Message Handlers
//!
//! Handlers for different WebSocket message types:
//! - Agent streaming
//! - Container logs
//! - Training progress
//! - Notifications
//!
//! @STREAM - Event-driven processing

use std::sync::Arc;

use tracing::{debug, error, info, warn};
use uuid::Uuid;

use super::connection::WsConnection;
use super::manager::ConnectionManager;
use super::messages::{
    WsCommand, WsEvent, SubscriptionType, NotificationLevel,
};
use crate::auth::{AuthenticatedUser, JwtService};

/// Handler result type
pub type HandlerResult = Result<Option<WsEvent>, WsEvent>;

/// Process incoming WebSocket command
pub async fn process_command(
    command: WsCommand,
    connection: &mut WsConnection,
    manager: &Arc<ConnectionManager>,
    jwt_service: &JwtService,
) -> HandlerResult {
    match command {
        WsCommand::Authenticate { token } => {
            handle_authenticate(token, connection, jwt_service).await
        }
        
        WsCommand::SubscribeAgent { agent_id, conversation_id } => {
            handle_subscribe_agent(agent_id, conversation_id, connection).await
        }
        
        WsCommand::UnsubscribeAgent { agent_id } => {
            handle_unsubscribe_agent(agent_id, connection).await
        }
        
        WsCommand::SendMessage { agent_id, conversation_id, content, metadata } => {
            handle_send_message(agent_id, conversation_id, content, metadata, connection, manager).await
        }
        
        WsCommand::SubscribeContainerLogs { container_id, follow, tail } => {
            handle_subscribe_container_logs(container_id, follow, tail, connection).await
        }
        
        WsCommand::UnsubscribeContainerLogs { container_id } => {
            handle_unsubscribe_container_logs(container_id, connection).await
        }
        
        WsCommand::SubscribeTraining { job_id } => {
            handle_subscribe_training(job_id, connection).await
        }
        
        WsCommand::UnsubscribeTraining { job_id } => {
            handle_unsubscribe_training(job_id, connection).await
        }
        
        WsCommand::GetState => {
            Ok(Some(connection.state_response()))
        }
        
        WsCommand::Ping { timestamp } => {
            connection.record_pong();
            Ok(Some(WsEvent::Pong {
                client_timestamp: timestamp,
                server_timestamp: chrono::Utc::now().timestamp(),
            }))
        }
        
        WsCommand::Cancel { request_id } => {
            handle_cancel(request_id, connection).await
        }
    }
}

// =============================================
// Authentication Handler
// =============================================

async fn handle_authenticate(
    token: String,
    connection: &mut WsConnection,
    jwt_service: &JwtService,
) -> HandlerResult {
    match jwt_service.validate_token(&token) {
        Ok(claims) => {
            let session_id = Uuid::new_v4().to_string();
            let user_uuid = Uuid::parse_str(&claims.sub)
                .map_err(|e| WsEvent::AuthenticationFailed {
                    reason: format!("Invalid user ID: {}", e),
                })?;
            connection.authenticate(user_uuid, session_id.clone());
            
            info!("WebSocket authenticated for user: {}", claims.sub);
            
            Ok(Some(WsEvent::Authenticated {
                user_id: user_uuid,
                session_id,
            }))
        }
        Err(e) => {
            warn!("WebSocket authentication failed: {}", e);
            Ok(Some(WsEvent::AuthenticationFailed {
                reason: e.to_string(),
            }))
        }
    }
}

// =============================================
// Agent Handlers
// =============================================

async fn handle_subscribe_agent(
    agent_id: Uuid,
    _conversation_id: Option<Uuid>,
    connection: &mut WsConnection,
) -> HandlerResult {
    if !connection.is_authenticated() {
        return Err(WsEvent::error("AUTH_REQUIRED", "Authentication required"));
    }

    // TODO: Verify user has access to this agent
    
    connection.subscribe(SubscriptionType::Agent, agent_id);
    
    debug!("Connection {} subscribed to agent {}", connection.id, agent_id);
    
    Ok(Some(WsEvent::Subscribed {
        subscription_type: SubscriptionType::Agent,
        resource_id: agent_id,
    }))
}

async fn handle_unsubscribe_agent(
    agent_id: Uuid,
    connection: &mut WsConnection,
) -> HandlerResult {
    connection.unsubscribe(SubscriptionType::Agent, agent_id);
    
    debug!("Connection {} unsubscribed from agent {}", connection.id, agent_id);
    
    Ok(Some(WsEvent::Unsubscribed {
        subscription_type: SubscriptionType::Agent,
        resource_id: agent_id,
    }))
}

async fn handle_send_message(
    agent_id: Uuid,
    conversation_id: Uuid,
    content: String,
    metadata: Option<serde_json::Value>,
    connection: &WsConnection,
    manager: &Arc<ConnectionManager>,
) -> HandlerResult {
    if !connection.is_authenticated() {
        return Err(WsEvent::error("AUTH_REQUIRED", "Authentication required"));
    }

    let user_id = connection.user_id.unwrap();
    let message_id = Uuid::new_v4();

    // TODO: Actually process the message through the agent system
    // For now, we'll simulate a response
    
    debug!(
        "User {} sending message to agent {} in conversation {}",
        user_id, agent_id, conversation_id
    );

    // Spawn agent processing task
    let manager_clone = manager.clone();
    let content_clone = content.clone();
    
    tokio::spawn(async move {
        // Simulate streaming response
        let response_parts = vec![
            "I ",
            "understand ",
            "your ",
            "message. ",
            "Processing...",
        ];

        for (i, part) in response_parts.iter().enumerate() {
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            
            let is_complete = i == response_parts.len() - 1;
            let event = WsEvent::agent_chunk(
                agent_id,
                conversation_id,
                message_id,
                *part,
                is_complete,
            );
            
            manager_clone.broadcast_to_subscribers(
                SubscriptionType::Agent,
                agent_id,
                event,
            ).await;
        }

        // Send completion event
        let full_response: String = response_parts.join("");
        let complete_event = WsEvent::AgentComplete {
            agent_id,
            conversation_id,
            message_id,
            full_response,
            usage: Some(super::messages::TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                cost_usd: Some(0.0001),
            }),
        };
        
        manager_clone.broadcast_to_subscribers(
            SubscriptionType::Agent,
            agent_id,
            complete_event,
        ).await;
    });

    // Acknowledge message received
    Ok(None) // No immediate response, will stream
}

/// Public handler for agent streaming (called from streaming service)
pub async fn agent_stream_handler(
    manager: Arc<ConnectionManager>,
    agent_id: Uuid,
    conversation_id: Uuid,
    message_id: Uuid,
    chunk: String,
    is_complete: bool,
) {
    let event = WsEvent::agent_chunk(agent_id, conversation_id, message_id, chunk, is_complete);
    manager.broadcast_to_subscribers(SubscriptionType::Agent, agent_id, event).await;
}

// =============================================
// Container Logs Handlers
// =============================================

async fn handle_subscribe_container_logs(
    container_id: Uuid,
    follow: bool,
    tail: Option<u32>,
    connection: &mut WsConnection,
) -> HandlerResult {
    if !connection.is_authenticated() {
        return Err(WsEvent::error("AUTH_REQUIRED", "Authentication required"));
    }

    // TODO: Verify user has access to this container
    // TODO: Start log streaming if follow=true, get tail logs
    
    connection.subscribe(SubscriptionType::ContainerLogs, container_id);
    
    debug!(
        "Connection {} subscribed to container {} logs (follow={}, tail={:?})",
        connection.id, container_id, follow, tail
    );
    
    Ok(Some(WsEvent::Subscribed {
        subscription_type: SubscriptionType::ContainerLogs,
        resource_id: container_id,
    }))
}

async fn handle_unsubscribe_container_logs(
    container_id: Uuid,
    connection: &mut WsConnection,
) -> HandlerResult {
    connection.unsubscribe(SubscriptionType::ContainerLogs, container_id);
    
    Ok(Some(WsEvent::Unsubscribed {
        subscription_type: SubscriptionType::ContainerLogs,
        resource_id: container_id,
    }))
}

/// Public handler for container log streaming
pub async fn container_logs_handler(
    manager: Arc<ConnectionManager>,
    container_id: Uuid,
    stream: super::messages::LogStream,
    message: String,
) {
    let event = WsEvent::container_log(container_id, stream, message);
    manager.broadcast_to_subscribers(SubscriptionType::ContainerLogs, container_id, event).await;
}

// =============================================
// Training Progress Handlers
// =============================================

async fn handle_subscribe_training(
    job_id: Uuid,
    connection: &mut WsConnection,
) -> HandlerResult {
    if !connection.is_authenticated() {
        return Err(WsEvent::error("AUTH_REQUIRED", "Authentication required"));
    }

    // TODO: Verify user has access to this training job
    
    connection.subscribe(SubscriptionType::Training, job_id);
    
    debug!("Connection {} subscribed to training job {}", connection.id, job_id);
    
    Ok(Some(WsEvent::Subscribed {
        subscription_type: SubscriptionType::Training,
        resource_id: job_id,
    }))
}

async fn handle_unsubscribe_training(
    job_id: Uuid,
    connection: &mut WsConnection,
) -> HandlerResult {
    connection.unsubscribe(SubscriptionType::Training, job_id);
    
    Ok(Some(WsEvent::Unsubscribed {
        subscription_type: SubscriptionType::Training,
        resource_id: job_id,
    }))
}

/// Public handler for training progress updates
pub async fn training_progress_handler(
    manager: Arc<ConnectionManager>,
    job_id: Uuid,
    epoch: u32,
    total_epochs: u32,
    step: u64,
    total_steps: u64,
    loss: f64,
    metrics: Option<serde_json::Value>,
    eta_seconds: Option<i64>,
) {
    let event = WsEvent::TrainingProgress {
        job_id,
        epoch,
        total_epochs,
        step,
        total_steps,
        loss,
        metrics,
        eta_seconds,
    };
    manager.broadcast_to_subscribers(SubscriptionType::Training, job_id, event).await;
}

// =============================================
// Notification Handler
// =============================================

/// Send notification to specific user
pub async fn notification_handler(
    manager: Arc<ConnectionManager>,
    user_id: Uuid,
    level: NotificationLevel,
    title: String,
    message: String,
    data: Option<serde_json::Value>,
) {
    let event = WsEvent::Notification {
        id: Uuid::new_v4(),
        level,
        title,
        message,
        timestamp: chrono::Utc::now(),
        data,
    };
    manager.send_to_user(user_id, event).await;
}

/// Broadcast notification to all authenticated users
pub async fn broadcast_notification(
    manager: Arc<ConnectionManager>,
    level: NotificationLevel,
    title: String,
    message: String,
) {
    let event = WsEvent::notification(level, title, message);
    manager.broadcast_authenticated(event).await;
}

// =============================================
// Cancel Handler
// =============================================

async fn handle_cancel(
    request_id: Uuid,
    _connection: &WsConnection,
) -> HandlerResult {
    // TODO: Implement cancellation of ongoing operations
    debug!("Cancel requested for request: {}", request_id);
    
    Ok(Some(WsEvent::notification(
        NotificationLevel::Info,
        "Cancellation Requested",
        "The operation cancellation has been requested",
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    fn create_test_connection() -> WsConnection {
        let (tx, _rx) = mpsc::unbounded_channel();
        WsConnection::new(tx)
    }

    #[tokio::test]
    async fn test_authenticate_required_for_subscriptions() {
        let mut conn = create_test_connection();
        let agent_id = Uuid::new_v4();

        let result = handle_subscribe_agent(agent_id, None, &mut conn).await;
        
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_subscribe_agent() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut conn = WsConnection::new(tx);
        let user_id = Uuid::new_v4();
        conn.authenticate(user_id, "session".to_string());
        
        let agent_id = Uuid::new_v4();
        let result = handle_subscribe_agent(agent_id, None, &mut conn).await;
        
        assert!(result.is_ok());
        assert!(conn.is_subscribed(SubscriptionType::Agent, agent_id));
    }

    #[tokio::test]
    async fn test_ping_pong() {
        let mut conn = create_test_connection();
        let timestamp = chrono::Utc::now().timestamp();
        
        let result = process_command(
            WsCommand::Ping { timestamp },
            &mut conn,
            &ConnectionManager::new(),
            &JwtService::new("test_secret"),
        ).await;
        
        assert!(result.is_ok());
        if let Ok(Some(WsEvent::Pong { client_timestamp, server_timestamp })) = result {
            assert_eq!(client_timestamp, timestamp);
            assert!(server_timestamp >= timestamp);
        } else {
            panic!("Expected Pong response");
        }
    }
}
