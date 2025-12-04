//! Authentication & Authorization Module
//!
//! Implements comprehensive security for NEURECTOMY:
//! - JWT token generation and validation
//! - Argon2 password hashing
//! - Role-Based Access Control (RBAC)
//! - Session management with Redis
//! - API key authentication
//!
//! @CIPHER @FORTRESS - Security First Design

pub mod api_key;
pub mod jwt;
pub mod middleware;
pub mod password;
pub mod rbac;
pub mod session;

pub use api_key::ApiKeyService;
pub use jwt::{Claims, JwtConfig, JwtService, TokenPair};
pub use middleware::{AuthMiddleware, RequireAuth, RequireRole};
pub use password::PasswordService;
pub use rbac::{Permission, RbacService, Role};
pub use session::SessionService;

use thiserror::Error;

/// Authentication errors
#[derive(Debug, Error)]
pub enum AuthError {
    #[error("Invalid credentials")]
    InvalidCredentials,

    #[error("Token expired")]
    TokenExpired,

    #[error("Invalid token")]
    InvalidToken,

    #[error("Token not found")]
    TokenNotFound,

    #[error("Session expired")]
    SessionExpired,

    #[error("Session not found")]
    SessionNotFound,

    #[error("Invalid API key")]
    InvalidApiKey,

    #[error("API key expired")]
    ApiKeyExpired,

    #[error("API key revoked")]
    ApiKeyRevoked,

    #[error("Insufficient permissions")]
    InsufficientPermissions,

    #[error("Account locked")]
    AccountLocked,

    #[error("Account not verified")]
    AccountNotVerified,

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Password too weak")]
    PasswordTooWeak,

    #[error("Password hash error: {0}")]
    PasswordHashError(String),

    #[error("Database error: {0}")]
    DatabaseError(String),

    #[error("Redis error: {0}")]
    RedisError(String),

    #[error("Internal error: {0}")]
    InternalError(String),
}

/// Authentication result type
pub type AuthResult<T> = Result<T, AuthError>;

/// User authentication status
#[derive(Debug, Clone)]
pub struct AuthenticatedUser {
    pub user_id: uuid::Uuid,
    pub email: String,
    pub role: Role,
    pub permissions: Vec<Permission>,
    pub session_id: Option<String>,
}

impl AuthenticatedUser {
    /// Check if user has a specific permission
    pub fn has_permission(&self, permission: &Permission) -> bool {
        self.permissions.contains(permission) || self.role == Role::Admin
    }

    /// Check if user has any of the given permissions
    pub fn has_any_permission(&self, permissions: &[Permission]) -> bool {
        permissions.iter().any(|p| self.has_permission(p))
    }

    /// Check if user has all of the given permissions
    pub fn has_all_permissions(&self, permissions: &[Permission]) -> bool {
        permissions.iter().all(|p| self.has_permission(p))
    }
}
