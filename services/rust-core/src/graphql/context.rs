//! GraphQL Context for request-scoped data
//!
//! Provides access to database connections and authenticated user info

use crate::db::DatabaseConnections;
use uuid::Uuid;

/// Request-scoped GraphQL context
#[derive(Clone)]
pub struct GraphQLContext {
    /// Database connections
    pub db: DatabaseConnections,
    /// Authenticated user ID (if any)
    pub user_id: Option<Uuid>,
    /// User role for authorization
    pub user_role: Option<String>,
    /// Request ID for tracing
    pub request_id: String,
}

impl GraphQLContext {
    /// Create new context with database connections
    pub fn new(db: DatabaseConnections) -> Self {
        Self {
            db,
            user_id: None,
            user_role: None,
            request_id: Uuid::new_v4().to_string(),
        }
    }

    /// Create authenticated context
    pub fn authenticated(db: DatabaseConnections, user_id: Uuid, role: String) -> Self {
        Self {
            db,
            user_id: Some(user_id),
            user_role: Some(role),
            request_id: Uuid::new_v4().to_string(),
        }
    }

    /// Check if user is authenticated
    pub fn is_authenticated(&self) -> bool {
        self.user_id.is_some()
    }

    /// Check if user has admin role
    pub fn is_admin(&self) -> bool {
        self.user_role.as_deref() == Some("admin")
    }

    /// Get user ID or return unauthorized error
    pub fn require_auth(&self) -> Result<Uuid, async_graphql::Error> {
        self.user_id
            .ok_or_else(|| async_graphql::Error::new("Authentication required"))
    }

    /// Require admin role
    pub fn require_admin(&self) -> Result<Uuid, async_graphql::Error> {
        let user_id = self.require_auth()?;
        if self.is_admin() {
            Ok(user_id)
        } else {
            Err(async_graphql::Error::new("Admin access required"))
        }
    }
}
