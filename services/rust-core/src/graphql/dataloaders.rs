//! DataLoaders for efficient batched data fetching
//!
//! Uses async-graphql's dataloader to avoid N+1 query problems

use async_graphql::dataloader::{DataLoader, Loader};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use crate::db::DatabaseConnections;
use crate::graphql::types::*;

// ============================================================================
// AGENT DATALOADER
// ============================================================================

/// Loader for fetching agents by ID
pub struct AgentLoader {
    db: DatabaseConnections,
}

impl AgentLoader {
    pub fn new(db: DatabaseConnections) -> DataLoader<Self> {
        DataLoader::new(Self { db }, tokio::spawn)
            .max_batch_size(100)
            .delay(std::time::Duration::from_millis(5))
    }
}

impl Loader<Uuid> for AgentLoader {
    type Value = Agent;
    type Error = Arc<anyhow::Error>;

    async fn load(&self, keys: &[Uuid]) -> Result<HashMap<Uuid, Self::Value>, Self::Error> {
        // TODO: Batch fetch agents from database
        // let agents = sqlx::query_as!(
        //     AgentRow,
        //     "SELECT * FROM agents WHERE id = ANY($1)",
        //     &keys
        // )
        // .fetch_all(&self.db.postgres.pool)
        // .await
        // .map_err(|e| Arc::new(anyhow::anyhow!(e)))?;
        
        // Placeholder
        Ok(HashMap::new())
    }
}

// ============================================================================
// USER DATALOADER
// ============================================================================

/// Loader for fetching users by ID
pub struct UserLoader {
    db: DatabaseConnections,
}

impl UserLoader {
    pub fn new(db: DatabaseConnections) -> DataLoader<Self> {
        DataLoader::new(Self { db }, tokio::spawn)
            .max_batch_size(100)
            .delay(std::time::Duration::from_millis(5))
    }
}

impl Loader<Uuid> for UserLoader {
    type Value = User;
    type Error = Arc<anyhow::Error>;

    async fn load(&self, keys: &[Uuid]) -> Result<HashMap<Uuid, Self::Value>, Self::Error> {
        // TODO: Batch fetch users from database
        Ok(HashMap::new())
    }
}

// ============================================================================
// CONTAINER DATALOADER
// ============================================================================

/// Loader for fetching containers by ID
pub struct ContainerLoader {
    db: DatabaseConnections,
}

impl ContainerLoader {
    pub fn new(db: DatabaseConnections) -> DataLoader<Self> {
        DataLoader::new(Self { db }, tokio::spawn)
            .max_batch_size(50)
            .delay(std::time::Duration::from_millis(5))
    }
}

impl Loader<Uuid> for ContainerLoader {
    type Value = Container;
    type Error = Arc<anyhow::Error>;

    async fn load(&self, keys: &[Uuid]) -> Result<HashMap<Uuid, Self::Value>, Self::Error> {
        // TODO: Batch fetch containers from database
        Ok(HashMap::new())
    }
}

// ============================================================================
// CONVERSATION DATALOADER
// ============================================================================

/// Loader for fetching conversations by ID
pub struct ConversationLoader {
    db: DatabaseConnections,
}

impl ConversationLoader {
    pub fn new(db: DatabaseConnections) -> DataLoader<Self> {
        DataLoader::new(Self { db }, tokio::spawn)
            .max_batch_size(50)
            .delay(std::time::Duration::from_millis(5))
    }
}

impl Loader<Uuid> for ConversationLoader {
    type Value = Conversation;
    type Error = Arc<anyhow::Error>;

    async fn load(&self, keys: &[Uuid]) -> Result<HashMap<Uuid, Self::Value>, Self::Error> {
        // TODO: Batch fetch conversations from database
        Ok(HashMap::new())
    }
}

// ============================================================================
// TRAINING JOB DATALOADER
// ============================================================================

/// Loader for fetching training jobs by ID
pub struct TrainingJobLoader {
    db: DatabaseConnections,
}

impl TrainingJobLoader {
    pub fn new(db: DatabaseConnections) -> DataLoader<Self> {
        DataLoader::new(Self { db }, tokio::spawn)
            .max_batch_size(50)
            .delay(std::time::Duration::from_millis(5))
    }
}

impl Loader<Uuid> for TrainingJobLoader {
    type Value = TrainingJob;
    type Error = Arc<anyhow::Error>;

    async fn load(&self, keys: &[Uuid]) -> Result<HashMap<Uuid, Self::Value>, Self::Error> {
        // TODO: Batch fetch training jobs from database
        Ok(HashMap::new())
    }
}

// ============================================================================
// RELATED DATA LOADERS (for resolving relationships)
// ============================================================================

/// Loader for fetching agents by user ID
pub struct AgentsByUserLoader {
    db: DatabaseConnections,
}

impl AgentsByUserLoader {
    pub fn new(db: DatabaseConnections) -> DataLoader<Self> {
        DataLoader::new(Self { db }, tokio::spawn)
            .max_batch_size(50)
            .delay(std::time::Duration::from_millis(5))
    }
}

impl Loader<Uuid> for AgentsByUserLoader {
    type Value = Vec<Agent>;
    type Error = Arc<anyhow::Error>;

    async fn load(&self, user_ids: &[Uuid]) -> Result<HashMap<Uuid, Self::Value>, Self::Error> {
        // TODO: Batch fetch agents grouped by user
        Ok(HashMap::new())
    }
}

/// Loader for fetching containers by agent ID
pub struct ContainersByAgentLoader {
    db: DatabaseConnections,
}

impl ContainersByAgentLoader {
    pub fn new(db: DatabaseConnections) -> DataLoader<Self> {
        DataLoader::new(Self { db }, tokio::spawn)
            .max_batch_size(50)
            .delay(std::time::Duration::from_millis(5))
    }
}

impl Loader<Uuid> for ContainersByAgentLoader {
    type Value = Vec<Container>;
    type Error = Arc<anyhow::Error>;

    async fn load(&self, agent_ids: &[Uuid]) -> Result<HashMap<Uuid, Self::Value>, Self::Error> {
        // TODO: Batch fetch containers grouped by agent
        Ok(HashMap::new())
    }
}

/// Loader for fetching conversations by agent ID
pub struct ConversationsByAgentLoader {
    db: DatabaseConnections,
}

impl ConversationsByAgentLoader {
    pub fn new(db: DatabaseConnections) -> DataLoader<Self> {
        DataLoader::new(Self { db }, tokio::spawn)
            .max_batch_size(50)
            .delay(std::time::Duration::from_millis(5))
    }
}

impl Loader<Uuid> for ConversationsByAgentLoader {
    type Value = Vec<Conversation>;
    type Error = Arc<anyhow::Error>;

    async fn load(&self, agent_ids: &[Uuid]) -> Result<HashMap<Uuid, Self::Value>, Self::Error> {
        // TODO: Batch fetch conversations grouped by agent
        Ok(HashMap::new())
    }
}

/// Loader for fetching messages by conversation ID
pub struct MessagesByConversationLoader {
    db: DatabaseConnections,
}

impl MessagesByConversationLoader {
    pub fn new(db: DatabaseConnections) -> DataLoader<Self> {
        DataLoader::new(Self { db }, tokio::spawn)
            .max_batch_size(50)
            .delay(std::time::Duration::from_millis(5))
    }
}

impl Loader<Uuid> for MessagesByConversationLoader {
    type Value = Vec<Message>;
    type Error = Arc<anyhow::Error>;

    async fn load(&self, conversation_ids: &[Uuid]) -> Result<HashMap<Uuid, Self::Value>, Self::Error> {
        // TODO: Batch fetch messages grouped by conversation
        Ok(HashMap::new())
    }
}

// ============================================================================
// DATALOADER REGISTRY
// ============================================================================

/// Container for all dataloaders
pub struct DataLoaders {
    pub agents: DataLoader<AgentLoader>,
    pub users: DataLoader<UserLoader>,
    pub containers: DataLoader<ContainerLoader>,
    pub conversations: DataLoader<ConversationLoader>,
    pub training_jobs: DataLoader<TrainingJobLoader>,
    pub agents_by_user: DataLoader<AgentsByUserLoader>,
    pub containers_by_agent: DataLoader<ContainersByAgentLoader>,
    pub conversations_by_agent: DataLoader<ConversationsByAgentLoader>,
    pub messages_by_conversation: DataLoader<MessagesByConversationLoader>,
}

impl DataLoaders {
    pub fn new(db: DatabaseConnections) -> Self {
        Self {
            agents: AgentLoader::new(db.clone()),
            users: UserLoader::new(db.clone()),
            containers: ContainerLoader::new(db.clone()),
            conversations: ConversationLoader::new(db.clone()),
            training_jobs: TrainingJobLoader::new(db.clone()),
            agents_by_user: AgentsByUserLoader::new(db.clone()),
            containers_by_agent: ContainersByAgentLoader::new(db.clone()),
            conversations_by_agent: ConversationsByAgentLoader::new(db.clone()),
            messages_by_conversation: MessagesByConversationLoader::new(db),
        }
    }
}
