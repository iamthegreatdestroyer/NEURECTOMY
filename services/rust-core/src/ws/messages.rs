//! WebSocket Message Types
//!
//! Defines the protocol for client-server WebSocket communication:
//! - Commands (client → server)
//! - Events (server → client)
//! - Bidirectional messages
//!
//! @STREAM @SYNAPSE - Protocol design

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Client to server commands
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload")]
pub enum WsCommand {
    /// Authenticate the connection
    Authenticate { token: String },

    /// Subscribe to agent responses
    SubscribeAgent {
        agent_id: Uuid,
        conversation_id: Option<Uuid>,
    },

    /// Unsubscribe from agent
    UnsubscribeAgent { agent_id: Uuid },

    /// Send message to agent
    SendMessage {
        agent_id: Uuid,
        conversation_id: Uuid,
        content: String,
        #[serde(default)]
        metadata: Option<serde_json::Value>,
    },

    /// Subscribe to container logs
    SubscribeContainerLogs {
        container_id: Uuid,
        #[serde(default)]
        follow: bool,
        #[serde(default)]
        tail: Option<u32>,
    },

    /// Unsubscribe from container logs
    UnsubscribeContainerLogs { container_id: Uuid },

    /// Subscribe to training job progress
    SubscribeTraining { job_id: Uuid },

    /// Unsubscribe from training job
    UnsubscribeTraining { job_id: Uuid },

    /// Request current state
    GetState,

    /// Ping for keepalive
    Ping { timestamp: i64 },

    /// Cancel ongoing operation
    Cancel { request_id: Uuid },
}

/// Server to client events
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload")]
pub enum WsEvent {
    /// Authentication result
    Authenticated { user_id: Uuid, session_id: String },

    /// Authentication failed
    AuthenticationFailed { reason: String },

    /// Subscription confirmed
    Subscribed {
        subscription_type: SubscriptionType,
        resource_id: Uuid,
    },

    /// Unsubscription confirmed
    Unsubscribed {
        subscription_type: SubscriptionType,
        resource_id: Uuid,
    },

    /// Agent response chunk (streaming)
    AgentChunk {
        agent_id: Uuid,
        conversation_id: Uuid,
        message_id: Uuid,
        chunk: String,
        #[serde(default)]
        is_complete: bool,
        #[serde(default)]
        metadata: Option<AgentChunkMetadata>,
    },

    /// Agent response complete
    AgentComplete {
        agent_id: Uuid,
        conversation_id: Uuid,
        message_id: Uuid,
        full_response: String,
        usage: Option<TokenUsage>,
    },

    /// Agent error
    AgentError {
        agent_id: Uuid,
        conversation_id: Option<Uuid>,
        error: String,
        error_code: Option<String>,
    },

    /// Container log entry
    ContainerLog {
        container_id: Uuid,
        timestamp: DateTime<Utc>,
        stream: LogStream,
        message: String,
    },

    /// Container status change
    ContainerStatus {
        container_id: Uuid,
        status: ContainerStatusType,
        message: Option<String>,
    },

    /// Training progress update
    TrainingProgress {
        job_id: Uuid,
        epoch: u32,
        total_epochs: u32,
        step: u64,
        total_steps: u64,
        loss: f64,
        metrics: Option<serde_json::Value>,
        eta_seconds: Option<i64>,
    },

    /// Training checkpoint saved
    TrainingCheckpoint {
        job_id: Uuid,
        checkpoint_path: String,
        epoch: u32,
        metrics: serde_json::Value,
    },

    /// Training completed
    TrainingComplete {
        job_id: Uuid,
        final_metrics: serde_json::Value,
        model_path: String,
        duration_seconds: i64,
    },

    /// Training error
    TrainingError {
        job_id: Uuid,
        error: String,
        recoverable: bool,
    },

    /// System notification
    Notification {
        id: Uuid,
        level: NotificationLevel,
        title: String,
        message: String,
        timestamp: DateTime<Utc>,
        #[serde(default)]
        data: Option<serde_json::Value>,
    },

    /// Current connection state
    State {
        authenticated: bool,
        user_id: Option<Uuid>,
        subscriptions: Vec<ActiveSubscription>,
        connected_at: DateTime<Utc>,
    },

    /// Pong response
    Pong {
        client_timestamp: i64,
        server_timestamp: i64,
    },

    /// Generic error
    Error {
        code: String,
        message: String,
        #[serde(default)]
        request_id: Option<Uuid>,
    },

    /// Heartbeat (server keepalive)
    Heartbeat { timestamp: i64 },
}

/// Unified WebSocket message (either direction)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum WsMessage {
    Command(WsCommand),
    Event(WsEvent),
}

// =============================================
// Supporting Types
// =============================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SubscriptionType {
    Agent,
    ContainerLogs,
    Training,
    Notifications,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveSubscription {
    pub subscription_type: SubscriptionType,
    pub resource_id: Uuid,
    pub subscribed_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentChunkMetadata {
    #[serde(default)]
    pub thinking: bool,
    #[serde(default)]
    pub tool_call: Option<ToolCallInfo>,
    #[serde(default)]
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallInfo {
    pub name: String,
    pub arguments: serde_json::Value,
    pub result: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(default)]
    pub cost_usd: Option<f64>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum LogStream {
    Stdout,
    Stderr,
    System,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContainerStatusType {
    Creating,
    Running,
    Paused,
    Stopping,
    Stopped,
    Error,
    Restarting,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum NotificationLevel {
    Info,
    Success,
    Warning,
    Error,
}

// =============================================
// Message Construction Helpers
// =============================================

impl WsEvent {
    /// Create an error event
    pub fn error(code: impl Into<String>, message: impl Into<String>) -> Self {
        WsEvent::Error {
            code: code.into(),
            message: message.into(),
            request_id: None,
        }
    }

    /// Create an error with request ID
    pub fn error_for_request(
        code: impl Into<String>,
        message: impl Into<String>,
        request_id: Uuid,
    ) -> Self {
        WsEvent::Error {
            code: code.into(),
            message: message.into(),
            request_id: Some(request_id),
        }
    }

    /// Create a notification
    pub fn notification(
        level: NotificationLevel,
        title: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        WsEvent::Notification {
            id: Uuid::new_v4(),
            level,
            title: title.into(),
            message: message.into(),
            timestamp: Utc::now(),
            data: None,
        }
    }

    /// Create agent chunk event
    pub fn agent_chunk(
        agent_id: Uuid,
        conversation_id: Uuid,
        message_id: Uuid,
        chunk: impl Into<String>,
        is_complete: bool,
    ) -> Self {
        WsEvent::AgentChunk {
            agent_id,
            conversation_id,
            message_id,
            chunk: chunk.into(),
            is_complete,
            metadata: None,
        }
    }

    /// Create container log event
    pub fn container_log(
        container_id: Uuid,
        stream: LogStream,
        message: impl Into<String>,
    ) -> Self {
        WsEvent::ContainerLog {
            container_id,
            timestamp: Utc::now(),
            stream,
            message: message.into(),
        }
    }

    /// Create training progress event
    pub fn training_progress(
        job_id: Uuid,
        epoch: u32,
        total_epochs: u32,
        step: u64,
        total_steps: u64,
        loss: f64,
    ) -> Self {
        WsEvent::TrainingProgress {
            job_id,
            epoch,
            total_epochs,
            step,
            total_steps,
            loss,
            metrics: None,
            eta_seconds: None,
        }
    }
}

impl WsMessage {
    /// Parse from JSON string
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Serialize to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_serialization() {
        let cmd = WsCommand::SubscribeAgent {
            agent_id: Uuid::new_v4(),
            conversation_id: None,
        };

        let json = serde_json::to_string(&cmd).expect("Serialization failed");
        assert!(json.contains("SubscribeAgent"));
    }

    #[test]
    fn test_event_serialization() {
        let event = WsEvent::agent_chunk(
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
            "Hello, ",
            false,
        );

        let json = serde_json::to_string(&event).expect("Serialization failed");
        assert!(json.contains("AgentChunk"));
        assert!(json.contains("Hello, "));
    }

    #[test]
    fn test_error_construction() {
        let error = WsEvent::error("AUTH_FAILED", "Invalid token");

        if let WsEvent::Error { code, message, .. } = error {
            assert_eq!(code, "AUTH_FAILED");
            assert_eq!(message, "Invalid token");
        } else {
            panic!("Expected Error event");
        }
    }
}
