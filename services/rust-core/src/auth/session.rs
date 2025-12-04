//! Session Management Service
//!
//! Manages user sessions with Redis backend:
//! - Session creation and validation
//! - Session timeout and extension
//! - Multi-device session tracking
//! - Session revocation
//!
//! @FORTRESS - Secure session handling

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{AuthError, AuthResult, rbac::Role};

/// Session data stored in Redis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionData {
    /// Session ID
    pub id: String,
    /// User ID
    pub user_id: Uuid,
    /// User email
    pub email: String,
    /// User role
    pub role: Role,
    /// Session creation time
    pub created_at: DateTime<Utc>,
    /// Last activity time
    pub last_activity: DateTime<Utc>,
    /// Session expiration time
    pub expires_at: DateTime<Utc>,
    /// Client IP address
    pub ip_address: Option<String>,
    /// User agent string
    pub user_agent: Option<String>,
    /// Device identifier
    pub device_id: Option<String>,
    /// Whether session is active
    pub is_active: bool,
}

impl SessionData {
    /// Check if session has expired
    pub fn is_expired(&self) -> bool {
        Utc::now() > self.expires_at
    }

    /// Check if session is valid (active and not expired)
    pub fn is_valid(&self) -> bool {
        self.is_active && !self.is_expired()
    }
}

/// Session configuration
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Session timeout in seconds (default: 24 hours)
    pub timeout_seconds: i64,
    /// Extend session on activity
    pub extend_on_activity: bool,
    /// Maximum sessions per user
    pub max_sessions_per_user: usize,
    /// Session ID prefix for Redis
    pub key_prefix: String,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            timeout_seconds: 86400, // 24 hours
            extend_on_activity: true,
            max_sessions_per_user: 5,
            key_prefix: "neurectomy:session:".to_string(),
        }
    }
}

/// Session service for managing user sessions
#[derive(Clone)]
pub struct SessionService {
    config: SessionConfig,
    // In production, this would be a Redis connection pool
    // For now, we use an in-memory store for development
    store: std::sync::Arc<tokio::sync::RwLock<std::collections::HashMap<String, SessionData>>>,
    // Track sessions by user for multi-device support
    user_sessions: std::sync::Arc<tokio::sync::RwLock<std::collections::HashMap<Uuid, Vec<String>>>>,
}

impl SessionService {
    /// Create new session service
    pub fn new(config: SessionConfig) -> Self {
        Self {
            config,
            store: std::sync::Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
            user_sessions: std::sync::Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
        }
    }

    /// Create a new session for a user
    pub async fn create_session(
        &self,
        user_id: Uuid,
        email: &str,
        role: Role,
        ip_address: Option<String>,
        user_agent: Option<String>,
        device_id: Option<String>,
    ) -> AuthResult<SessionData> {
        // Check max sessions limit
        let user_session_count = self.get_user_session_count(user_id).await;
        if user_session_count >= self.config.max_sessions_per_user {
            // Revoke oldest session
            self.revoke_oldest_session(user_id).await?;
        }

        let now = Utc::now();
        let session = SessionData {
            id: Uuid::new_v4().to_string(),
            user_id,
            email: email.to_string(),
            role,
            created_at: now,
            last_activity: now,
            expires_at: now + Duration::seconds(self.config.timeout_seconds),
            ip_address,
            user_agent,
            device_id,
            is_active: true,
        };

        // Store session
        self.store.write().await.insert(session.id.clone(), session.clone());

        // Track user session
        self.user_sessions
            .write()
            .await
            .entry(user_id)
            .or_default()
            .push(session.id.clone());

        Ok(session)
    }

    /// Get session by ID
    pub async fn get_session(&self, session_id: &str) -> AuthResult<SessionData> {
        let session = self
            .store
            .read()
            .await
            .get(session_id)
            .cloned()
            .ok_or(AuthError::SessionNotFound)?;

        if !session.is_valid() {
            return Err(AuthError::SessionExpired);
        }

        Ok(session)
    }

    /// Validate and optionally extend session
    pub async fn validate_session(&self, session_id: &str) -> AuthResult<SessionData> {
        let mut session = self.get_session(session_id).await?;

        // Extend session if configured
        if self.config.extend_on_activity {
            session.last_activity = Utc::now();
            session.expires_at = Utc::now() + Duration::seconds(self.config.timeout_seconds);
            self.store.write().await.insert(session_id.to_string(), session.clone());
        }

        Ok(session)
    }

    /// Update session activity timestamp
    pub async fn touch_session(&self, session_id: &str) -> AuthResult<()> {
        let mut store = self.store.write().await;
        
        if let Some(session) = store.get_mut(session_id) {
            if !session.is_valid() {
                return Err(AuthError::SessionExpired);
            }

            session.last_activity = Utc::now();
            if self.config.extend_on_activity {
                session.expires_at = Utc::now() + Duration::seconds(self.config.timeout_seconds);
            }
            Ok(())
        } else {
            Err(AuthError::SessionNotFound)
        }
    }

    /// Revoke a specific session
    pub async fn revoke_session(&self, session_id: &str) -> AuthResult<()> {
        let mut store = self.store.write().await;
        
        if let Some(session) = store.get_mut(session_id) {
            let user_id = session.user_id;
            session.is_active = false;
            
            // Remove from user sessions
            if let Some(sessions) = self.user_sessions.write().await.get_mut(&user_id) {
                sessions.retain(|s| s != session_id);
            }
            
            Ok(())
        } else {
            Err(AuthError::SessionNotFound)
        }
    }

    /// Revoke all sessions for a user
    pub async fn revoke_all_user_sessions(&self, user_id: Uuid) -> AuthResult<usize> {
        let session_ids: Vec<String> = self
            .user_sessions
            .read()
            .await
            .get(&user_id)
            .cloned()
            .unwrap_or_default();

        let count = session_ids.len();

        for session_id in session_ids {
            let _ = self.revoke_session(&session_id).await;
        }

        // Clear user sessions list
        self.user_sessions.write().await.remove(&user_id);

        Ok(count)
    }

    /// Revoke all sessions except current
    pub async fn revoke_other_sessions(&self, user_id: Uuid, current_session_id: &str) -> AuthResult<usize> {
        let session_ids: Vec<String> = self
            .user_sessions
            .read()
            .await
            .get(&user_id)
            .cloned()
            .unwrap_or_default();

        let mut count = 0;
        for session_id in session_ids {
            if session_id != current_session_id {
                if self.revoke_session(&session_id).await.is_ok() {
                    count += 1;
                }
            }
        }

        Ok(count)
    }

    /// Get all active sessions for a user
    pub async fn get_user_sessions(&self, user_id: Uuid) -> Vec<SessionData> {
        let session_ids: Vec<String> = self
            .user_sessions
            .read()
            .await
            .get(&user_id)
            .cloned()
            .unwrap_or_default();

        let store = self.store.read().await;
        
        session_ids
            .iter()
            .filter_map(|id| store.get(id).cloned())
            .filter(|s| s.is_valid())
            .collect()
    }

    /// Get session count for a user
    async fn get_user_session_count(&self, user_id: Uuid) -> usize {
        self.get_user_sessions(user_id).await.len()
    }

    /// Revoke oldest session for a user
    async fn revoke_oldest_session(&self, user_id: Uuid) -> AuthResult<()> {
        let sessions = self.get_user_sessions(user_id).await;
        
        if let Some(oldest) = sessions.iter().min_by_key(|s| s.created_at) {
            self.revoke_session(&oldest.id).await
        } else {
            Ok(())
        }
    }

    /// Cleanup expired sessions (should be run periodically)
    pub async fn cleanup_expired(&self) -> usize {
        let mut store = self.store.write().await;
        let mut user_sessions = self.user_sessions.write().await;

        let expired: Vec<String> = store
            .iter()
            .filter(|(_, s)| s.is_expired())
            .map(|(id, _)| id.clone())
            .collect();

        let count = expired.len();

        for session_id in expired {
            if let Some(session) = store.remove(&session_id) {
                if let Some(sessions) = user_sessions.get_mut(&session.user_id) {
                    sessions.retain(|s| s != &session_id);
                }
            }
        }

        count
    }
}

impl Default for SessionService {
    fn default() -> Self {
        Self::new(SessionConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_and_validate_session() {
        let service = SessionService::default();
        let user_id = Uuid::new_v4();

        let session = service
            .create_session(user_id, "test@example.com", Role::Developer, None, None, None)
            .await
            .expect("Session creation should succeed");

        assert!(session.is_valid());
        assert_eq!(session.user_id, user_id);

        let validated = service
            .validate_session(&session.id)
            .await
            .expect("Session validation should succeed");

        assert_eq!(validated.user_id, user_id);
    }

    #[tokio::test]
    async fn test_revoke_session() {
        let service = SessionService::default();
        let user_id = Uuid::new_v4();

        let session = service
            .create_session(user_id, "test@example.com", Role::Developer, None, None, None)
            .await
            .expect("Session creation should succeed");

        service
            .revoke_session(&session.id)
            .await
            .expect("Session revocation should succeed");

        let result = service.validate_session(&session.id).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_max_sessions_limit() {
        let config = SessionConfig {
            max_sessions_per_user: 2,
            ..Default::default()
        };
        let service = SessionService::new(config);
        let user_id = Uuid::new_v4();

        // Create max sessions
        let session1 = service
            .create_session(user_id, "test@example.com", Role::Developer, None, None, None)
            .await
            .expect("Session 1 creation should succeed");

        let _session2 = service
            .create_session(user_id, "test@example.com", Role::Developer, None, None, None)
            .await
            .expect("Session 2 creation should succeed");

        // Create one more - should revoke oldest
        let _session3 = service
            .create_session(user_id, "test@example.com", Role::Developer, None, None, None)
            .await
            .expect("Session 3 creation should succeed");

        // Oldest session should be revoked
        let result = service.validate_session(&session1.id).await;
        assert!(result.is_err());

        // User should have exactly max_sessions
        let sessions = service.get_user_sessions(user_id).await;
        assert_eq!(sessions.len(), 2);
    }

    #[tokio::test]
    async fn test_revoke_all_sessions() {
        let service = SessionService::default();
        let user_id = Uuid::new_v4();

        // Create multiple sessions
        for _ in 0..3 {
            service
                .create_session(user_id, "test@example.com", Role::Developer, None, None, None)
                .await
                .expect("Session creation should succeed");
        }

        let count = service
            .revoke_all_user_sessions(user_id)
            .await
            .expect("Revoke all should succeed");

        assert_eq!(count, 3);

        let sessions = service.get_user_sessions(user_id).await;
        assert!(sessions.is_empty());
    }
}
