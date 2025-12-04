//! Test Fixtures
//!
//! Static test data and fixture management.
//!
//! @ECLIPSE - Static test fixtures

use once_cell::sync::Lazy;
use uuid::Uuid;

/// Fixed UUIDs for deterministic testing
pub mod ids {
    use super::*;

    pub static TEST_USER_ID: Lazy<Uuid> =
        Lazy::new(|| Uuid::parse_str("11111111-1111-1111-1111-111111111111").unwrap());

    pub static TEST_ADMIN_ID: Lazy<Uuid> =
        Lazy::new(|| Uuid::parse_str("22222222-2222-2222-2222-222222222222").unwrap());

    pub static TEST_AGENT_ID: Lazy<Uuid> =
        Lazy::new(|| Uuid::parse_str("33333333-3333-3333-3333-333333333333").unwrap());

    pub static TEST_CONVERSATION_ID: Lazy<Uuid> =
        Lazy::new(|| Uuid::parse_str("44444444-4444-4444-4444-444444444444").unwrap());

    pub static TEST_CONTAINER_ID: Lazy<Uuid> =
        Lazy::new(|| Uuid::parse_str("55555555-5555-5555-5555-555555555555").unwrap());
}

/// Test JWT tokens
pub mod tokens {
    /// Valid test JWT (for development only, expires far future)
    pub const VALID_JWT: &str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMTExMTExMS0xMTExLTExMTEtMTExMS0xMTExMTExMTExMTEiLCJlbWFpbCI6InRlc3RAdGVzdC5jb20iLCJyb2xlIjoidXNlciIsImV4cCI6OTk5OTk5OTk5OX0.test_signature";

    /// Expired test JWT
    pub const EXPIRED_JWT: &str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMTExMTExMS0xMTExLTExMTEtMTExMS0xMTExMTExMTExMTEiLCJlbWFpbCI6InRlc3RAdGVzdC5jb20iLCJyb2xlIjoidXNlciIsImV4cCI6MH0.test_signature";

    /// Admin JWT
    pub const ADMIN_JWT: &str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIyMjIyMjIyMi0yMjIyLTIyMjItMjIyMi0yMjIyMjIyMjIyMjIiLCJlbWFpbCI6ImFkbWluQHRlc3QuY29tIiwicm9sZSI6ImFkbWluIiwiZXhwIjo5OTk5OTk5OTk5fQ.test_signature";
}

/// Test API keys
pub mod api_keys {
    pub const VALID_API_KEY: &str = "nrct_test_1234567890abcdef1234567890abcdef";
    pub const INVALID_API_KEY: &str = "invalid_key_format";
    pub const REVOKED_API_KEY: &str = "nrct_test_revoked12345678901234567890";
}

/// Test configuration values
pub mod config {
    pub const TEST_DATABASE_URL: &str = "postgres://test:test@localhost:5432/neurectomy_test";
    pub const TEST_REDIS_URL: &str = "redis://localhost:6379/1";
    pub const TEST_NEO4J_URI: &str = "bolt://localhost:7687";
    pub const TEST_JWT_SECRET: &str = "test_jwt_secret_for_development_only_32bytes";
}

/// GraphQL test queries
pub mod graphql {
    pub const INTROSPECTION_QUERY: &str = r#"
        query {
            __schema {
                types {
                    name
                }
            }
        }
    "#;

    pub const GET_AGENTS_QUERY: &str = r#"
        query GetAgents($limit: Int, $offset: Int) {
            agents(limit: $limit, offset: $offset) {
                id
                name
                status
            }
        }
    "#;

    pub const CREATE_AGENT_MUTATION: &str = r#"
        mutation CreateAgent($input: CreateAgentInput!) {
            createAgent(input: $input) {
                id
                name
                status
            }
        }
    "#;

    pub const AGENT_UPDATES_SUBSCRIPTION: &str = r#"
        subscription AgentUpdates($agentId: UUID!) {
            agentUpdates(agentId: $agentId) {
                id
                status
                lastActivity
            }
        }
    "#;
}

/// Sample model configurations
pub mod models {
    use serde_json::json;

    pub fn openai_config() -> serde_json::Value {
        json!({
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 1.0
        })
    }

    pub fn anthropic_config() -> serde_json::Value {
        json!({
            "model": "claude-3-opus",
            "temperature": 0.5,
            "max_tokens": 8192
        })
    }

    pub fn local_config() -> serde_json::Value {
        json!({
            "model": "llama-2-70b",
            "temperature": 0.8,
            "max_tokens": 2048
        })
    }
}

/// Sample system prompts
pub mod prompts {
    pub const ASSISTANT_PROMPT: &str = r#"
        You are a helpful AI assistant. Be concise and accurate.
        Always be respectful and professional.
    "#;

    pub const CODE_REVIEW_PROMPT: &str = r#"
        You are an expert code reviewer. Focus on:
        - Security vulnerabilities
        - Performance issues
        - Code style and best practices
        - Potential bugs
    "#;

    pub const DATA_ANALYST_PROMPT: &str = r#"
        You are a data analyst. Provide insights based on data.
        Use statistical methods and visualizations when appropriate.
    "#;
}
