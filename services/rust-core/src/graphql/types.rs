//! GraphQL type definitions
//!
//! Defines GraphQL object types for all domain entities

use async_graphql::{ComplexObject, Enum, InputObject, Object, SimpleObject, ID};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ============================================================================
// ENUMS
// ============================================================================

/// User role enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum, Serialize, Deserialize)]
#[graphql(rename_items = "SCREAMING_SNAKE_CASE")]
pub enum UserRole {
    Admin,
    Developer,
    Viewer,
    ApiUser,
}

impl Default for UserRole {
    fn default() -> Self {
        Self::Developer
    }
}

/// Agent status enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum, Serialize, Deserialize)]
#[graphql(rename_items = "SCREAMING_SNAKE_CASE")]
pub enum AgentStatus {
    Draft,
    Active,
    Paused,
    Training,
    Archived,
    Error,
}

impl Default for AgentStatus {
    fn default() -> Self {
        Self::Draft
    }
}

/// Agent type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum, Serialize, Deserialize)]
#[graphql(rename_items = "SCREAMING_SNAKE_CASE")]
pub enum AgentType {
    Chat,
    Task,
    Assistant,
    Specialist,
    Multi,
}

impl Default for AgentType {
    fn default() -> Self {
        Self::Chat
    }
}

/// Model provider enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum, Serialize, Deserialize)]
#[graphql(rename_items = "SCREAMING_SNAKE_CASE")]
pub enum ModelProvider {
    OpenAI,
    Anthropic,
    Google,
    Meta,
    Mistral,
    Local,
    Custom,
}

impl Default for ModelProvider {
    fn default() -> Self {
        Self::OpenAI
    }
}

/// Container status enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum, Serialize, Deserialize)]
#[graphql(rename_items = "SCREAMING_SNAKE_CASE")]
pub enum ContainerStatus {
    Creating,
    Running,
    Paused,
    Stopped,
    Error,
    Terminated,
}

impl Default for ContainerStatus {
    fn default() -> Self {
        Self::Creating
    }
}

/// Training status enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum, Serialize, Deserialize)]
#[graphql(rename_items = "SCREAMING_SNAKE_CASE")]
pub enum TrainingStatus {
    Pending,
    Preparing,
    Training,
    Evaluating,
    Completed,
    Failed,
    Cancelled,
}

impl Default for TrainingStatus {
    fn default() -> Self {
        Self::Pending
    }
}

/// Training method enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum, Serialize, Deserialize)]
#[graphql(rename_items = "SCREAMING_SNAKE_CASE")]
pub enum TrainingMethod {
    Rlhf,
    Sft,
    Dpo,
    Ppo,
    LoRa,
    QLoRa,
    FullFinetune,
}

impl Default for TrainingMethod {
    fn default() -> Self {
        Self::LoRa
    }
}

/// Conversation status enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum, Serialize, Deserialize)]
#[graphql(rename_items = "SCREAMING_SNAKE_CASE")]
pub enum ConversationStatus {
    Active,
    Archived,
    Deleted,
}

impl Default for ConversationStatus {
    fn default() -> Self {
        Self::Active
    }
}

/// Message role enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum, Serialize, Deserialize)]
#[graphql(rename_items = "SCREAMING_SNAKE_CASE")]
pub enum MessageRole {
    User,
    Assistant,
    System,
    Tool,
    Function,
}

impl Default for MessageRole {
    fn default() -> Self {
        Self::User
    }
}

// ============================================================================
// OUTPUT TYPES
// ============================================================================

/// User type for GraphQL
#[derive(Debug, Clone, SimpleObject)]
#[graphql(complex)]
pub struct User {
    pub id: ID,
    pub email: String,
    pub username: String,
    #[graphql(skip)]
    pub password_hash: Option<String>,
    pub display_name: Option<String>,
    pub avatar_url: Option<String>,
    pub role: UserRole,
    pub is_active: bool,
    pub email_verified: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_login_at: Option<DateTime<Utc>>,
}

#[ComplexObject]
impl User {
    /// Get user's agents
    async fn agents(
        &self,
        ctx: &async_graphql::Context<'_>,
        #[graphql(default = 10)] limit: i32,
        #[graphql(default = 0)] offset: i32,
    ) -> async_graphql::Result<Vec<Agent>> {
        // Will be implemented with dataloader
        Ok(vec![])
    }

    /// Get user's conversations count
    async fn conversations_count(&self, ctx: &async_graphql::Context<'_>) -> async_graphql::Result<i32> {
        Ok(0)
    }
}

/// Agent type for GraphQL
#[derive(Debug, Clone, SimpleObject)]
#[graphql(complex)]
pub struct Agent {
    pub id: ID,
    pub user_id: ID,
    pub name: String,
    pub description: Option<String>,
    pub agent_type: AgentType,
    pub status: AgentStatus,
    pub model_provider: ModelProvider,
    pub model_name: String,
    pub system_prompt: Option<String>,
    pub temperature: f64,
    pub max_tokens: i32,
    pub version: i32,
    pub is_public: bool,
    pub tags: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[ComplexObject]
impl Agent {
    /// Get agent's tools
    async fn tools(&self, ctx: &async_graphql::Context<'_>) -> async_graphql::Result<Vec<AgentTool>> {
        Ok(vec![])
    }

    /// Get agent's conversations count
    async fn conversations_count(&self, ctx: &async_graphql::Context<'_>) -> async_graphql::Result<i32> {
        Ok(0)
    }

    /// Get agent's active container
    async fn active_container(&self, ctx: &async_graphql::Context<'_>) -> async_graphql::Result<Option<Container>> {
        Ok(None)
    }
}

/// Agent tool configuration
#[derive(Debug, Clone, SimpleObject)]
pub struct AgentTool {
    pub id: ID,
    pub name: String,
    pub description: String,
    pub tool_type: String,
    pub schema: serde_json::Value,
    pub is_enabled: bool,
}

/// Container type for GraphQL
#[derive(Debug, Clone, SimpleObject)]
#[graphql(complex)]
pub struct Container {
    pub id: ID,
    pub agent_id: ID,
    pub name: String,
    pub container_id: Option<String>,
    pub image: String,
    pub status: ContainerStatus,
    pub cpu_limit: f64,
    pub memory_limit_mb: i32,
    pub gpu_enabled: bool,
    pub port_mappings: serde_json::Value,
    pub environment: serde_json::Value,
    pub started_at: Option<DateTime<Utc>>,
    pub stopped_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[ComplexObject]
impl Container {
    /// Get container's agent
    async fn agent(&self, ctx: &async_graphql::Context<'_>) -> async_graphql::Result<Option<Agent>> {
        Ok(None)
    }

    /// Get container logs
    async fn logs(
        &self,
        ctx: &async_graphql::Context<'_>,
        #[graphql(default = 100)] limit: i32,
        level: Option<String>,
    ) -> async_graphql::Result<Vec<ContainerLog>> {
        Ok(vec![])
    }
}

/// Container log entry
#[derive(Debug, Clone, SimpleObject)]
pub struct ContainerLog {
    pub id: ID,
    pub container_id: ID,
    pub timestamp: DateTime<Utc>,
    pub level: String,
    pub message: String,
    pub source: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

/// Training job type for GraphQL
#[derive(Debug, Clone, SimpleObject)]
#[graphql(complex)]
pub struct TrainingJob {
    pub id: ID,
    pub agent_id: ID,
    pub name: String,
    pub description: Option<String>,
    pub method: TrainingMethod,
    pub status: TrainingStatus,
    pub base_model: String,
    pub hyperparameters: serde_json::Value,
    pub progress: f64,
    pub current_epoch: Option<i32>,
    pub total_epochs: Option<i32>,
    pub current_step: Option<i32>,
    pub total_steps: Option<i32>,
    pub training_loss: Option<f64>,
    pub validation_loss: Option<f64>,
    pub error_message: Option<String>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[ComplexObject]
impl TrainingJob {
    /// Get training job's agent
    async fn agent(&self, ctx: &async_graphql::Context<'_>) -> async_graphql::Result<Option<Agent>> {
        Ok(None)
    }

    /// Get training job's checkpoints
    async fn checkpoints(&self, ctx: &async_graphql::Context<'_>) -> async_graphql::Result<Vec<TrainingCheckpoint>> {
        Ok(vec![])
    }

    /// Get training job's evaluations
    async fn evaluations(&self, ctx: &async_graphql::Context<'_>) -> async_graphql::Result<Vec<EvaluationRun>> {
        Ok(vec![])
    }
}

/// Training checkpoint
#[derive(Debug, Clone, SimpleObject)]
pub struct TrainingCheckpoint {
    pub id: ID,
    pub training_job_id: ID,
    pub epoch: i32,
    pub step: i32,
    pub checkpoint_path: String,
    pub metrics: serde_json::Value,
    pub is_best: bool,
    pub created_at: DateTime<Utc>,
}

/// Evaluation run
#[derive(Debug, Clone, SimpleObject)]
pub struct EvaluationRun {
    pub id: ID,
    pub training_job_id: ID,
    pub checkpoint_id: Option<ID>,
    pub eval_type: String,
    pub metrics: serde_json::Value,
    pub created_at: DateTime<Utc>,
}

/// Conversation type for GraphQL
#[derive(Debug, Clone, SimpleObject)]
#[graphql(complex)]
pub struct Conversation {
    pub id: ID,
    pub user_id: ID,
    pub agent_id: ID,
    pub title: String,
    pub status: ConversationStatus,
    pub message_count: i32,
    pub total_tokens: i32,
    pub metadata: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[ComplexObject]
impl Conversation {
    /// Get conversation's agent
    async fn agent(&self, ctx: &async_graphql::Context<'_>) -> async_graphql::Result<Option<Agent>> {
        Ok(None)
    }

    /// Get conversation's messages
    async fn messages(
        &self,
        ctx: &async_graphql::Context<'_>,
        #[graphql(default = 50)] limit: i32,
        #[graphql(default = 0)] offset: i32,
    ) -> async_graphql::Result<Vec<Message>> {
        Ok(vec![])
    }
}

/// Message type for GraphQL
#[derive(Debug, Clone, SimpleObject)]
pub struct Message {
    pub id: ID,
    pub conversation_id: ID,
    pub role: MessageRole,
    pub content: String,
    pub token_count: Option<i32>,
    pub model_used: Option<String>,
    pub metadata: serde_json::Value,
    pub created_at: DateTime<Utc>,
}

/// Message attachment
#[derive(Debug, Clone, SimpleObject)]
pub struct MessageAttachment {
    pub id: ID,
    pub message_id: ID,
    pub filename: String,
    pub content_type: String,
    pub size_bytes: i64,
    pub url: String,
}

// ============================================================================
// INPUT TYPES
// ============================================================================

/// Input for creating a new agent
#[derive(Debug, Clone, InputObject)]
pub struct CreateAgentInput {
    pub name: String,
    #[graphql(default)]
    pub description: Option<String>,
    #[graphql(default)]
    pub agent_type: AgentType,
    #[graphql(default)]
    pub model_provider: ModelProvider,
    #[graphql(default_with = "String::from(\"gpt-4-turbo\")")]
    pub model_name: String,
    #[graphql(default)]
    pub system_prompt: Option<String>,
    #[graphql(default_with = "0.7")]
    pub temperature: f64,
    #[graphql(default_with = "4096")]
    pub max_tokens: i32,
    #[graphql(default)]
    pub is_public: bool,
    #[graphql(default)]
    pub tags: Vec<String>,
}

/// Input for updating an agent
#[derive(Debug, Clone, InputObject)]
pub struct UpdateAgentInput {
    pub id: ID,
    pub name: Option<String>,
    pub description: Option<String>,
    pub agent_type: Option<AgentType>,
    pub status: Option<AgentStatus>,
    pub model_provider: Option<ModelProvider>,
    pub model_name: Option<String>,
    pub system_prompt: Option<String>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<i32>,
    pub is_public: Option<bool>,
    pub tags: Option<Vec<String>>,
}

/// Input for creating a container
#[derive(Debug, Clone, InputObject)]
pub struct CreateContainerInput {
    pub agent_id: ID,
    pub name: String,
    #[graphql(default_with = "String::from(\"neurectomy/agent-runtime:latest\")")]
    pub image: String,
    #[graphql(default_with = "1.0")]
    pub cpu_limit: f64,
    #[graphql(default_with = "2048")]
    pub memory_limit_mb: i32,
    #[graphql(default)]
    pub gpu_enabled: bool,
    #[graphql(default)]
    pub environment: Option<serde_json::Value>,
}

/// Input for starting a training job
#[derive(Debug, Clone, InputObject)]
pub struct StartTrainingInput {
    pub agent_id: ID,
    pub name: String,
    #[graphql(default)]
    pub description: Option<String>,
    #[graphql(default)]
    pub method: TrainingMethod,
    pub base_model: String,
    pub dataset_ids: Vec<ID>,
    #[graphql(default)]
    pub hyperparameters: Option<serde_json::Value>,
}

/// Input for creating a conversation
#[derive(Debug, Clone, InputObject)]
pub struct CreateConversationInput {
    pub agent_id: ID,
    #[graphql(default)]
    pub title: Option<String>,
    #[graphql(default)]
    pub initial_message: Option<String>,
}

/// Input for sending a message
#[derive(Debug, Clone, InputObject)]
pub struct SendMessageInput {
    pub conversation_id: ID,
    pub content: String,
    #[graphql(default)]
    pub attachments: Vec<AttachmentInput>,
}

/// Attachment input
#[derive(Debug, Clone, InputObject)]
pub struct AttachmentInput {
    pub filename: String,
    pub content_type: String,
    pub data: String, // Base64 encoded
}

// ============================================================================
// PAGINATION & FILTERING
// ============================================================================

/// Pagination input
#[derive(Debug, Clone, InputObject)]
pub struct PaginationInput {
    #[graphql(default_with = "20")]
    pub limit: i32,
    #[graphql(default_with = "0")]
    pub offset: i32,
}

/// Agent filter input
#[derive(Debug, Clone, InputObject, Default)]
pub struct AgentFilterInput {
    pub status: Option<AgentStatus>,
    pub agent_type: Option<AgentType>,
    pub model_provider: Option<ModelProvider>,
    pub is_public: Option<bool>,
    pub search: Option<String>,
    pub tags: Option<Vec<String>>,
}

/// Container filter input
#[derive(Debug, Clone, InputObject, Default)]
pub struct ContainerFilterInput {
    pub status: Option<ContainerStatus>,
    pub agent_id: Option<ID>,
}

/// Training job filter input
#[derive(Debug, Clone, InputObject, Default)]
pub struct TrainingJobFilterInput {
    pub status: Option<TrainingStatus>,
    pub method: Option<TrainingMethod>,
    pub agent_id: Option<ID>,
}

/// Conversation filter input
#[derive(Debug, Clone, InputObject, Default)]
pub struct ConversationFilterInput {
    pub status: Option<ConversationStatus>,
    pub agent_id: Option<ID>,
    pub search: Option<String>,
}

// ============================================================================
// CONNECTION TYPES (Relay-style pagination)
// ============================================================================

/// Page info for pagination
#[derive(Debug, Clone, SimpleObject)]
pub struct PageInfo {
    pub has_next_page: bool,
    pub has_previous_page: bool,
    pub start_cursor: Option<String>,
    pub end_cursor: Option<String>,
    pub total_count: i32,
}

/// Generic edge type
#[derive(Debug, Clone, SimpleObject)]
pub struct Edge<T: async_graphql::OutputType> {
    pub cursor: String,
    pub node: T,
}

/// Agent connection for pagination
#[derive(Debug, Clone, SimpleObject)]
pub struct AgentConnection {
    pub edges: Vec<Edge<Agent>>,
    pub page_info: PageInfo,
}

/// Container connection for pagination
#[derive(Debug, Clone, SimpleObject)]
pub struct ContainerConnection {
    pub edges: Vec<Edge<Container>>,
    pub page_info: PageInfo,
}

/// Training job connection for pagination
#[derive(Debug, Clone, SimpleObject)]
pub struct TrainingJobConnection {
    pub edges: Vec<Edge<TrainingJob>>,
    pub page_info: PageInfo,
}

/// Conversation connection for pagination
#[derive(Debug, Clone, SimpleObject)]
pub struct ConversationConnection {
    pub edges: Vec<Edge<Conversation>>,
    pub page_info: PageInfo,
}

// ============================================================================
// RESPONSE TYPES
// ============================================================================

/// Generic mutation response
#[derive(Debug, Clone, SimpleObject)]
pub struct MutationResponse {
    pub success: bool,
    pub message: String,
}

/// Delete response
#[derive(Debug, Clone, SimpleObject)]
pub struct DeleteResponse {
    pub success: bool,
    pub deleted_id: ID,
}

/// Container action response
#[derive(Debug, Clone, SimpleObject)]
pub struct ContainerActionResponse {
    pub success: bool,
    pub container: Option<Container>,
    pub message: String,
}

/// Training action response
#[derive(Debug, Clone, SimpleObject)]
pub struct TrainingActionResponse {
    pub success: bool,
    pub training_job: Option<TrainingJob>,
    pub message: String,
}
