//! Database Models
//! @VERTEX SQLx Models for PostgreSQL

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use uuid::Uuid;

// ============================================================
// User Models
// ============================================================

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct User {
    pub id: Uuid,
    pub email: String,
    pub username: String,
    pub display_name: Option<String>,
    pub avatar_url: Option<String>,
    pub password_hash: Option<String>,
    pub auth_provider: String,
    pub external_id: Option<String>,
    pub email_verified: bool,
    pub role: String,
    pub permissions: serde_json::Value,
    pub status: String,
    pub mfa_enabled: bool,
    pub failed_login_attempts: i32,
    pub locked_until: Option<DateTime<Utc>>,
    pub last_login_at: Option<DateTime<Utc>>,
    pub preferences: serde_json::Value,
    pub quota: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateUser {
    pub email: String,
    pub username: String,
    pub display_name: Option<String>,
    pub password_hash: Option<String>,
    pub auth_provider: String,
    pub external_id: Option<String>,
    pub role: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateUser {
    pub display_name: Option<String>,
    pub avatar_url: Option<String>,
    pub preferences: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPublic {
    pub id: Uuid,
    pub email: String,
    pub username: String,
    pub display_name: Option<String>,
    pub avatar_url: Option<String>,
    pub role: String,
    pub status: String,
    pub created_at: DateTime<Utc>,
}

impl From<User> for UserPublic {
    fn from(user: User) -> Self {
        Self {
            id: user.id,
            email: user.email,
            username: user.username,
            display_name: user.display_name,
            avatar_url: user.avatar_url,
            role: user.role,
            status: user.status,
            created_at: user.created_at,
        }
    }
}

// ============================================================
// Session Models
// ============================================================

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct UserSession {
    pub id: Uuid,
    pub user_id: Uuid,
    pub token_hash: String,
    pub refresh_token_hash: Option<String>,
    pub device_id: Option<String>,
    pub device_name: Option<String>,
    pub user_agent: Option<String>,
    pub is_revoked: bool,
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub last_active_at: DateTime<Utc>,
}

// ============================================================
// Agent Models
// ============================================================

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Agent {
    pub id: Uuid,
    pub user_id: Uuid,
    pub name: String,
    pub slug: String,
    pub description: Option<String>,
    pub avatar_url: Option<String>,
    pub tags: serde_json::Value,
    pub agent_type: String,
    pub category: Option<String>,
    pub model_provider: String,
    pub model_name: String,
    pub model_config: serde_json::Value,
    pub system_prompt: String,
    pub persona: Option<serde_json::Value>,
    pub capabilities: serde_json::Value,
    pub tools: serde_json::Value,
    pub mcp_servers: serde_json::Value,
    pub memory_config: serde_json::Value,
    pub rag_config: Option<serde_json::Value>,
    pub status: String,
    pub status_message: Option<String>,
    pub metrics: serde_json::Value,
    pub version: i32,
    pub is_published: bool,
    pub published_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_active_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateAgent {
    pub name: String,
    pub description: Option<String>,
    pub agent_type: Option<String>,
    pub model_provider: Option<String>,
    pub model_name: Option<String>,
    pub system_prompt: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateAgent {
    pub name: Option<String>,
    pub description: Option<String>,
    pub avatar_url: Option<String>,
    pub tags: Option<Vec<String>>,
    pub model_provider: Option<String>,
    pub model_name: Option<String>,
    pub model_config: Option<serde_json::Value>,
    pub system_prompt: Option<String>,
    pub capabilities: Option<serde_json::Value>,
    pub tools: Option<serde_json::Value>,
    pub memory_config: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSummary {
    pub id: Uuid,
    pub name: String,
    pub slug: String,
    pub description: Option<String>,
    pub avatar_url: Option<String>,
    pub agent_type: String,
    pub model_provider: String,
    pub model_name: String,
    pub status: String,
    pub is_published: bool,
    pub created_at: DateTime<Utc>,
}

// ============================================================
// Container Models
// ============================================================

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Container {
    pub id: Uuid,
    pub user_id: Uuid,
    pub agent_id: Option<Uuid>,
    pub name: String,
    pub description: Option<String>,
    pub container_type: String,
    pub docker_id: Option<String>,
    pub image: String,
    pub resources: serde_json::Value,
    pub resource_usage: serde_json::Value,
    pub environment: serde_json::Value,
    pub ports: serde_json::Value,
    pub status: String,
    pub status_message: Option<String>,
    pub health_status: Option<String>,
    pub metrics: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub stopped_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateContainer {
    pub name: String,
    pub description: Option<String>,
    pub container_type: Option<String>,
    pub agent_id: Option<Uuid>,
    pub image: Option<String>,
    pub resources: Option<serde_json::Value>,
    pub environment: Option<serde_json::Value>,
}

// ============================================================
// Training Models
// ============================================================

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct TrainingJob {
    pub id: Uuid,
    pub user_id: Uuid,
    pub agent_id: Option<Uuid>,
    pub name: String,
    pub description: Option<String>,
    pub method: String,
    pub base_model: String,
    pub base_model_provider: String,
    pub hyperparameters: serde_json::Value,
    pub dataset_config: serde_json::Value,
    pub resources: serde_json::Value,
    pub status: String,
    pub status_message: Option<String>,
    pub progress: serde_json::Value,
    pub results: Option<serde_json::Value>,
    pub output_model_path: Option<String>,
    pub metrics_history: serde_json::Value,
    pub container_id: Option<Uuid>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateTrainingJob {
    pub name: String,
    pub description: Option<String>,
    pub method: Option<String>,
    pub base_model: String,
    pub base_model_provider: String,
    pub agent_id: Option<Uuid>,
    pub hyperparameters: Option<serde_json::Value>,
    pub dataset_config: Option<serde_json::Value>,
}

// ============================================================
// Conversation Models
// ============================================================

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Conversation {
    pub id: Uuid,
    pub user_id: Uuid,
    pub agent_id: Uuid,
    pub title: Option<String>,
    pub summary: Option<String>,
    pub tags: serde_json::Value,
    pub status: String,
    pub message_count: i32,
    pub total_tokens: i32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_message_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Message {
    pub id: Uuid,
    pub conversation_id: Uuid,
    pub role: String,
    pub content: String,
    pub tool_calls: Option<serde_json::Value>,
    pub tool_call_id: Option<String>,
    pub token_count: Option<i32>,
    pub model: Option<String>,
    pub latency_ms: Option<i32>,
    pub user_rating: Option<i32>,
    pub position: i32,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateMessage {
    pub role: String,
    pub content: String,
    pub tool_calls: Option<serde_json::Value>,
    pub tool_call_id: Option<String>,
}

// ============================================================
// Audit Models
// ============================================================

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct AuditLog {
    pub id: Uuid,
    pub user_id: Option<Uuid>,
    pub user_email: Option<String>,
    pub action: String,
    pub resource_type: String,
    pub resource_id: Option<Uuid>,
    pub resource_name: Option<String>,
    pub description: String,
    pub old_values: Option<serde_json::Value>,
    pub new_values: Option<serde_json::Value>,
    pub success: bool,
    pub error_message: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateAuditLog {
    pub user_id: Option<Uuid>,
    pub action: String,
    pub resource_type: String,
    pub resource_id: Option<Uuid>,
    pub resource_name: Option<String>,
    pub description: String,
    pub old_values: Option<serde_json::Value>,
    pub new_values: Option<serde_json::Value>,
    pub success: bool,
    pub error_message: Option<String>,
}
