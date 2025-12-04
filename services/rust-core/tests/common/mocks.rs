//! Mock Implementations
//!
//! Mock traits and implementations for isolated unit testing.
//! Uses mockall for auto-generated mocks.
//!
//! @ECLIPSE - Mock infrastructure

use async_trait::async_trait;
use uuid::Uuid;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Mock database trait
#[async_trait]
pub trait DatabaseMock: Send + Sync {
    async fn get_user(&self, id: Uuid) -> Option<MockUser>;
    async fn create_user(&self, user: MockUser) -> Result<MockUser, MockError>;
    async fn get_agent(&self, id: Uuid) -> Option<MockAgent>;
    async fn create_agent(&self, agent: MockAgent) -> Result<MockAgent, MockError>;
}

/// Simple in-memory mock database
#[derive(Default)]
pub struct InMemoryDb {
    users: Arc<RwLock<HashMap<Uuid, MockUser>>>,
    agents: Arc<RwLock<HashMap<Uuid, MockAgent>>>,
}

impl InMemoryDb {
    pub fn new() -> Self {
        Self::default()
    }

    /// Seed with test data
    pub fn with_user(self, user: MockUser) -> Self {
        self.users.write().unwrap().insert(user.id, user);
        self
    }

    pub fn with_agent(self, agent: MockAgent) -> Self {
        self.agents.write().unwrap().insert(agent.id, agent);
        self
    }
}

#[async_trait]
impl DatabaseMock for InMemoryDb {
    async fn get_user(&self, id: Uuid) -> Option<MockUser> {
        self.users.read().unwrap().get(&id).cloned()
    }

    async fn create_user(&self, user: MockUser) -> Result<MockUser, MockError> {
        let mut users = self.users.write().unwrap();
        if users.contains_key(&user.id) {
            return Err(MockError::DuplicateKey);
        }
        users.insert(user.id, user.clone());
        Ok(user)
    }

    async fn get_agent(&self, id: Uuid) -> Option<MockAgent> {
        self.agents.read().unwrap().get(&id).cloned()
    }

    async fn create_agent(&self, agent: MockAgent) -> Result<MockAgent, MockError> {
        let mut agents = self.agents.write().unwrap();
        if agents.contains_key(&agent.id) {
            return Err(MockError::DuplicateKey);
        }
        agents.insert(agent.id, agent.clone());
        Ok(agent)
    }
}

#[derive(Debug, Clone)]
pub struct MockUser {
    pub id: Uuid,
    pub email: String,
    pub username: String,
    pub role: String,
}

#[derive(Debug, Clone)]
pub struct MockAgent {
    pub id: Uuid,
    pub user_id: Uuid,
    pub name: String,
    pub status: String,
}

#[derive(Debug, Clone)]
pub enum MockError {
    NotFound,
    DuplicateKey,
    ConnectionError,
    ValidationError(String),
}

/// Mock LLM client for testing AI interactions
pub struct MockLLMClient {
    responses: Arc<RwLock<Vec<String>>>,
    call_count: Arc<RwLock<usize>>,
}

impl MockLLMClient {
    pub fn new() -> Self {
        Self {
            responses: Arc::new(RwLock::new(Vec::new())),
            call_count: Arc::new(RwLock::new(0)),
        }
    }

    /// Set up mock responses (FIFO)
    pub fn with_responses(self, responses: Vec<String>) -> Self {
        *self.responses.write().unwrap() = responses;
        self
    }

    /// Get call count
    pub fn call_count(&self) -> usize {
        *self.call_count.read().unwrap()
    }

    /// Simulate completion
    pub async fn complete(&self, _prompt: &str) -> Result<String, MockError> {
        *self.call_count.write().unwrap() += 1;

        let mut responses = self.responses.write().unwrap();
        if responses.is_empty() {
            Ok("Default mock response".to_string())
        } else {
            Ok(responses.remove(0))
        }
    }

    /// Simulate streaming completion
    pub async fn complete_stream(
        &self,
        _prompt: &str,
    ) -> Result<MockStream, MockError> {
        *self.call_count.write().unwrap() += 1;

        let responses = self.responses.read().unwrap();
        let response = responses.first().cloned().unwrap_or_else(|| "Default".to_string());

        Ok(MockStream {
            chunks: response.chars().map(|c| c.to_string()).collect(),
            index: 0,
        })
    }
}

impl Default for MockLLMClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Mock stream for testing streaming responses
pub struct MockStream {
    chunks: Vec<String>,
    index: usize,
}

impl MockStream {
    pub async fn next(&mut self) -> Option<String> {
        if self.index < self.chunks.len() {
            let chunk = self.chunks[self.index].clone();
            self.index += 1;
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            Some(chunk)
        } else {
            None
        }
    }
}

/// Mock event bus for testing pub/sub
pub struct MockEventBus {
    events: Arc<RwLock<Vec<MockEvent>>>,
}

impl MockEventBus {
    pub fn new() -> Self {
        Self {
            events: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub fn publish(&self, event: MockEvent) {
        self.events.write().unwrap().push(event);
    }

    pub fn events(&self) -> Vec<MockEvent> {
        self.events.read().unwrap().clone()
    }

    pub fn event_count(&self) -> usize {
        self.events.read().unwrap().len()
    }

    pub fn clear(&self) {
        self.events.write().unwrap().clear();
    }
}

impl Default for MockEventBus {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct MockEvent {
    pub topic: String,
    pub payload: serde_json::Value,
}

/// Mock cache for testing caching behavior
pub struct MockCache {
    data: Arc<RwLock<HashMap<String, (String, Option<std::time::Instant>)>>>,
}

impl MockCache {
    pub fn new() -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn get(&self, key: &str) -> Option<String> {
        let data = self.data.read().unwrap();
        data.get(key).and_then(|(value, expiry)| {
            if let Some(exp) = expiry {
                if *exp < std::time::Instant::now() {
                    return None;
                }
            }
            Some(value.clone())
        })
    }

    pub async fn set(&self, key: &str, value: &str, ttl_secs: Option<u64>) {
        let expiry = ttl_secs.map(|ttl| {
            std::time::Instant::now() + std::time::Duration::from_secs(ttl)
        });
        self.data
            .write()
            .unwrap()
            .insert(key.to_string(), (value.to_string(), expiry));
    }

    pub async fn delete(&self, key: &str) -> bool {
        self.data.write().unwrap().remove(key).is_some()
    }
}

impl Default for MockCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_in_memory_db() {
        let db = InMemoryDb::new();
        
        let user = MockUser {
            id: Uuid::new_v4(),
            email: "test@test.com".to_string(),
            username: "testuser".to_string(),
            role: "user".to_string(),
        };

        let created = db.create_user(user.clone()).await.unwrap();
        assert_eq!(created.email, "test@test.com");

        let fetched = db.get_user(user.id).await.unwrap();
        assert_eq!(fetched.email, "test@test.com");
    }

    #[tokio::test]
    async fn test_mock_llm_client() {
        let client = MockLLMClient::new()
            .with_responses(vec![
                "First response".to_string(),
                "Second response".to_string(),
            ]);

        let r1 = client.complete("test").await.unwrap();
        let r2 = client.complete("test").await.unwrap();
        let r3 = client.complete("test").await.unwrap();

        assert_eq!(r1, "First response");
        assert_eq!(r2, "Second response");
        assert_eq!(r3, "Default mock response");
        assert_eq!(client.call_count(), 3);
    }

    #[tokio::test]
    async fn test_mock_cache() {
        let cache = MockCache::new();

        cache.set("key1", "value1", None).await;
        assert_eq!(cache.get("key1").await, Some("value1".to_string()));

        cache.delete("key1").await;
        assert_eq!(cache.get("key1").await, None);
    }
}
