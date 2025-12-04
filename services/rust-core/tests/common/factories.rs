//! Test Data Factories
//!
//! Factory functions for generating test data.
//! Uses fake-rs for realistic random data generation.
//!
//! @ECLIPSE - Test data generation

use chrono::{DateTime, Utc};
use fake::faker::internet::en::*;
use fake::faker::lorem::en::*;
use fake::faker::name::en::*;
use fake::{Fake, Faker};
use uuid::Uuid;

/// User factory for generating test users
pub struct UserFactory;

impl UserFactory {
    /// Create a random user
    pub fn random() -> TestUser {
        TestUser {
            id: Uuid::new_v4(),
            email: SafeEmail().fake(),
            username: Username().fake(),
            full_name: Some(Name().fake()),
            password_hash: "$argon2id$v=19$m=65536,t=3,p=4$fake_salt$fake_hash".to_string(),
            role: "user".to_string(),
            is_active: true,
            is_verified: true,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    /// Create an admin user
    pub fn admin() -> TestUser {
        let mut user = Self::random();
        user.role = "admin".to_string();
        user
    }

    /// Create an unverified user
    pub fn unverified() -> TestUser {
        let mut user = Self::random();
        user.is_verified = false;
        user
    }

    /// Create a specific user with builder pattern
    pub fn build() -> TestUserBuilder {
        TestUserBuilder::default()
    }
}

#[derive(Debug, Clone)]
pub struct TestUser {
    pub id: Uuid,
    pub email: String,
    pub username: String,
    pub full_name: Option<String>,
    pub password_hash: String,
    pub role: String,
    pub is_active: bool,
    pub is_verified: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Default)]
pub struct TestUserBuilder {
    id: Option<Uuid>,
    email: Option<String>,
    username: Option<String>,
    role: Option<String>,
    is_active: Option<bool>,
    is_verified: Option<bool>,
}

impl TestUserBuilder {
    pub fn id(mut self, id: Uuid) -> Self {
        self.id = Some(id);
        self
    }

    pub fn email(mut self, email: impl Into<String>) -> Self {
        self.email = Some(email.into());
        self
    }

    pub fn username(mut self, username: impl Into<String>) -> Self {
        self.username = Some(username.into());
        self
    }

    pub fn role(mut self, role: impl Into<String>) -> Self {
        self.role = Some(role.into());
        self
    }

    pub fn active(mut self, is_active: bool) -> Self {
        self.is_active = Some(is_active);
        self
    }

    pub fn verified(mut self, is_verified: bool) -> Self {
        self.is_verified = Some(is_verified);
        self
    }

    pub fn build(self) -> TestUser {
        let mut user = UserFactory::random();

        if let Some(id) = self.id {
            user.id = id;
        }
        if let Some(email) = self.email {
            user.email = email;
        }
        if let Some(username) = self.username {
            user.username = username;
        }
        if let Some(role) = self.role {
            user.role = role;
        }
        if let Some(is_active) = self.is_active {
            user.is_active = is_active;
        }
        if let Some(is_verified) = self.is_verified {
            user.is_verified = is_verified;
        }

        user
    }
}

/// Agent factory for generating test agents
pub struct AgentFactory;

impl AgentFactory {
    /// Create a random agent
    pub fn random() -> TestAgent {
        let words: Vec<String> = Words(2..4).fake();
        TestAgent {
            id: Uuid::new_v4(),
            user_id: Uuid::new_v4(),
            name: words.join(" "),
            description: Some(Sentence(5..15).fake()),
            system_prompt: Paragraph(2..5).fake(),
            model_provider: "openai".to_string(),
            model_name: "gpt-4".to_string(),
            temperature: 0.7,
            max_tokens: 4096,
            is_public: false,
            status: "active".to_string(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    /// Create a public agent
    pub fn public() -> TestAgent {
        let mut agent = Self::random();
        agent.is_public = true;
        agent
    }

    /// Create agent with specific user
    pub fn for_user(user_id: Uuid) -> TestAgent {
        let mut agent = Self::random();
        agent.user_id = user_id;
        agent
    }

    /// Builder pattern
    pub fn build() -> TestAgentBuilder {
        TestAgentBuilder::default()
    }
}

#[derive(Debug, Clone)]
pub struct TestAgent {
    pub id: Uuid,
    pub user_id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub system_prompt: String,
    pub model_provider: String,
    pub model_name: String,
    pub temperature: f64,
    pub max_tokens: i32,
    pub is_public: bool,
    pub status: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Default)]
pub struct TestAgentBuilder {
    user_id: Option<Uuid>,
    name: Option<String>,
    model_provider: Option<String>,
    model_name: Option<String>,
    is_public: Option<bool>,
}

impl TestAgentBuilder {
    pub fn user_id(mut self, user_id: Uuid) -> Self {
        self.user_id = Some(user_id);
        self
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn model(mut self, provider: impl Into<String>, name: impl Into<String>) -> Self {
        self.model_provider = Some(provider.into());
        self.model_name = Some(name.into());
        self
    }

    pub fn public(mut self, is_public: bool) -> Self {
        self.is_public = Some(is_public);
        self
    }

    pub fn build(self) -> TestAgent {
        let mut agent = AgentFactory::random();

        if let Some(user_id) = self.user_id {
            agent.user_id = user_id;
        }
        if let Some(name) = self.name {
            agent.name = name;
        }
        if let Some(provider) = self.model_provider {
            agent.model_provider = provider;
        }
        if let Some(model_name) = self.model_name {
            agent.model_name = model_name;
        }
        if let Some(is_public) = self.is_public {
            agent.is_public = is_public;
        }

        agent
    }
}

/// Conversation factory
pub struct ConversationFactory;

impl ConversationFactory {
    pub fn random() -> TestConversation {
        let words: Vec<String> = Words(2..5).fake();
        TestConversation {
            id: Uuid::new_v4(),
            agent_id: Uuid::new_v4(),
            user_id: Uuid::new_v4(),
            title: words.join(" "),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    pub fn for_agent(agent_id: Uuid, user_id: Uuid) -> TestConversation {
        let mut conv = Self::random();
        conv.agent_id = agent_id;
        conv.user_id = user_id;
        conv
    }
}

#[derive(Debug, Clone)]
pub struct TestConversation {
    pub id: Uuid,
    pub agent_id: Uuid,
    pub user_id: Uuid,
    pub title: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Message factory
pub struct MessageFactory;

impl MessageFactory {
    pub fn user_message(conversation_id: Uuid) -> TestMessage {
        TestMessage {
            id: Uuid::new_v4(),
            conversation_id,
            role: "user".to_string(),
            content: Paragraph(1..3).fake(),
            token_count: Some((10..500).fake()),
            created_at: Utc::now(),
        }
    }

    pub fn assistant_message(conversation_id: Uuid) -> TestMessage {
        TestMessage {
            id: Uuid::new_v4(),
            conversation_id,
            role: "assistant".to_string(),
            content: Paragraph(2..6).fake(),
            token_count: Some((50..2000).fake()),
            created_at: Utc::now(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TestMessage {
    pub id: Uuid,
    pub conversation_id: Uuid,
    pub role: String,
    pub content: String,
    pub token_count: Option<i32>,
    pub created_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_factory_random() {
        let user = UserFactory::random();
        assert!(!user.email.is_empty());
        assert!(user.is_active);
        assert!(user.is_verified);
    }

    #[test]
    fn test_user_factory_admin() {
        let user = UserFactory::admin();
        assert_eq!(user.role, "admin");
    }

    #[test]
    fn test_user_builder() {
        let user = UserFactory::build()
            .email("test@example.com")
            .role("admin")
            .build();

        assert_eq!(user.email, "test@example.com");
        assert_eq!(user.role, "admin");
    }

    #[test]
    fn test_agent_factory() {
        let agent = AgentFactory::random();
        assert!(!agent.name.is_empty());
        assert!(!agent.is_public);
    }

    #[test]
    fn test_conversation_factory() {
        let conv = ConversationFactory::random();
        assert!(!conv.title.is_empty());
    }
}
