//! API Key Authentication Service
//!
//! Implements API key authentication for programmatic access:
//! - Key generation with secure random bytes
//! - Key validation and expiration
//! - Key scoping and permissions
//! - Key revocation and rotation
//!
//! @CIPHER - Cryptographic security
//! @FORTRESS - Defense in depth

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
use uuid::Uuid;

use super::{AuthError, AuthResult, rbac::Permission};

/// API Key prefix for identification
const API_KEY_PREFIX: &str = "nrctmy";

/// API Key data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    /// Unique key identifier
    pub id: Uuid,
    /// User who owns this key
    pub user_id: Uuid,
    /// Key name (for identification)
    pub name: String,
    /// Hashed key value (never store plaintext!)
    pub key_hash: String,
    /// Key prefix (visible part for identification)
    pub key_prefix: String,
    /// Scoped permissions
    pub permissions: Vec<Permission>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last used timestamp
    pub last_used_at: Option<DateTime<Utc>>,
    /// Expiration timestamp (None = never expires)
    pub expires_at: Option<DateTime<Utc>>,
    /// Whether the key is active
    pub is_active: bool,
    /// Rate limit (requests per minute, None = default)
    pub rate_limit: Option<u32>,
    /// Allowed IP addresses (empty = any)
    pub allowed_ips: Vec<String>,
}

impl ApiKey {
    /// Check if key is expired
    pub fn is_expired(&self) -> bool {
        self.expires_at.map(|e| Utc::now() > e).unwrap_or(false)
    }

    /// Check if key is valid (active and not expired)
    pub fn is_valid(&self) -> bool {
        self.is_active && !self.is_expired()
    }

    /// Check if key has a specific permission
    pub fn has_permission(&self, permission: &Permission) -> bool {
        self.permissions.contains(permission)
    }

    /// Check if IP is allowed
    pub fn is_ip_allowed(&self, ip: &str) -> bool {
        self.allowed_ips.is_empty() || self.allowed_ips.contains(&ip.to_string())
    }
}

/// Generated API key (returned only once)
#[derive(Debug, Clone, Serialize)]
pub struct GeneratedApiKey {
    /// Full API key (only shown once!)
    pub key: String,
    /// Key metadata
    pub metadata: ApiKey,
}

/// API Key configuration
#[derive(Debug, Clone)]
pub struct ApiKeyConfig {
    /// Default expiration days (None = never)
    pub default_expiration_days: Option<i64>,
    /// Default rate limit
    pub default_rate_limit: u32,
    /// Key length in bytes
    pub key_length: usize,
}

impl Default for ApiKeyConfig {
    fn default() -> Self {
        Self {
            default_expiration_days: Some(365), // 1 year
            default_rate_limit: 1000,           // 1000 requests/minute
            key_length: 32,                     // 256 bits
        }
    }
}

/// API Key service
#[derive(Clone)]
pub struct ApiKeyService {
    config: ApiKeyConfig,
    // In production, this would be database-backed
    store: std::sync::Arc<tokio::sync::RwLock<std::collections::HashMap<String, ApiKey>>>,
}

impl ApiKeyService {
    /// Create new API key service
    pub fn new() -> Self {
        Self::with_config(ApiKeyConfig::default())
    }

    /// Create with custom config
    pub fn with_config(config: ApiKeyConfig) -> Self {
        Self {
            config,
            store: std::sync::Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
        }
    }

    /// Generate a new API key
    pub async fn generate_key(
        &self,
        user_id: Uuid,
        name: &str,
        permissions: Vec<Permission>,
        expires_in_days: Option<i64>,
        rate_limit: Option<u32>,
        allowed_ips: Vec<String>,
    ) -> AuthResult<GeneratedApiKey> {
        // Generate random bytes
        let mut key_bytes = vec![0u8; self.config.key_length];
        getrandom::getrandom(&mut key_bytes)
            .map_err(|e| AuthError::InternalError(e.to_string()))?;

        // Create human-readable key
        let key_hex = hex::encode(&key_bytes);
        let full_key = format!("{}_{}", API_KEY_PREFIX, key_hex);
        let key_prefix = format!("{}_{}", API_KEY_PREFIX, &key_hex[..8]);

        // Hash the key for storage
        let key_hash = hash_api_key(&full_key);

        let now = Utc::now();
        let expires_at = expires_in_days
            .or(self.config.default_expiration_days)
            .map(|days| now + Duration::days(days));

        let api_key = ApiKey {
            id: Uuid::new_v4(),
            user_id,
            name: name.to_string(),
            key_hash: key_hash.clone(),
            key_prefix,
            permissions,
            created_at: now,
            last_used_at: None,
            expires_at,
            is_active: true,
            rate_limit: rate_limit.or(Some(self.config.default_rate_limit)),
            allowed_ips,
        };

        // Store the key (indexed by hash)
        self.store.write().await.insert(key_hash.clone(), api_key.clone());

        Ok(GeneratedApiKey {
            key: full_key,
            metadata: api_key,
        })
    }

    /// Validate an API key
    pub async fn validate_key(&self, key: &str) -> AuthResult<ApiKey> {
        // Validate format
        if !key.starts_with(&format!("{}_", API_KEY_PREFIX)) {
            return Err(AuthError::InvalidApiKey);
        }

        let key_hash = hash_api_key(key);
        
        let mut store = self.store.write().await;
        
        let api_key = store
            .get_mut(&key_hash)
            .ok_or(AuthError::InvalidApiKey)?;

        // Check if valid
        if !api_key.is_active {
            return Err(AuthError::ApiKeyRevoked);
        }

        if api_key.is_expired() {
            return Err(AuthError::ApiKeyExpired);
        }

        // Update last used
        api_key.last_used_at = Some(Utc::now());

        Ok(api_key.clone())
    }

    /// Validate key with IP check
    pub async fn validate_key_with_ip(&self, key: &str, ip: &str) -> AuthResult<ApiKey> {
        let api_key = self.validate_key(key).await?;

        if !api_key.is_ip_allowed(ip) {
            return Err(AuthError::InvalidApiKey);
        }

        Ok(api_key)
    }

    /// Get all keys for a user
    pub async fn get_user_keys(&self, user_id: Uuid) -> Vec<ApiKey> {
        self.store
            .read()
            .await
            .values()
            .filter(|k| k.user_id == user_id && k.is_active)
            .cloned()
            .collect()
    }

    /// Revoke an API key
    pub async fn revoke_key(&self, key_id: Uuid, user_id: Uuid) -> AuthResult<()> {
        let mut store = self.store.write().await;

        for api_key in store.values_mut() {
            if api_key.id == key_id && api_key.user_id == user_id {
                api_key.is_active = false;
                return Ok(());
            }
        }

        Err(AuthError::InvalidApiKey)
    }

    /// Revoke all keys for a user
    pub async fn revoke_all_user_keys(&self, user_id: Uuid) -> AuthResult<usize> {
        let mut store = self.store.write().await;
        let mut count = 0;

        for api_key in store.values_mut() {
            if api_key.user_id == user_id && api_key.is_active {
                api_key.is_active = false;
                count += 1;
            }
        }

        Ok(count)
    }

    /// Rotate a key (revoke old, generate new with same settings)
    pub async fn rotate_key(&self, key_id: Uuid, user_id: Uuid) -> AuthResult<GeneratedApiKey> {
        let store = self.store.read().await;
        
        let old_key = store
            .values()
            .find(|k| k.id == key_id && k.user_id == user_id)
            .cloned()
            .ok_or(AuthError::InvalidApiKey)?;

        drop(store);

        // Revoke old key
        self.revoke_key(key_id, user_id).await?;

        // Generate new key with same settings
        let expires_in_days = old_key.expires_at.map(|e| {
            (e - Utc::now()).num_days()
        });

        self.generate_key(
            user_id,
            &format!("{} (rotated)", old_key.name),
            old_key.permissions,
            expires_in_days,
            old_key.rate_limit,
            old_key.allowed_ips,
        )
        .await
    }
}

impl Default for ApiKeyService {
    fn default() -> Self {
        Self::new()
    }
}

/// Hash an API key for storage
fn hash_api_key(key: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(key.as_bytes());
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_generate_and_validate_key() {
        let service = ApiKeyService::new();
        let user_id = Uuid::new_v4();

        let generated = service
            .generate_key(
                user_id,
                "Test Key",
                vec![Permission::AgentRead, Permission::AgentCreate],
                Some(30),
                None,
                vec![],
            )
            .await
            .expect("Key generation should succeed");

        // Key should have correct prefix
        assert!(generated.key.starts_with("nrctmy_"));

        // Validate the key
        let validated = service
            .validate_key(&generated.key)
            .await
            .expect("Key validation should succeed");

        assert_eq!(validated.user_id, user_id);
        assert_eq!(validated.name, "Test Key");
        assert!(validated.has_permission(&Permission::AgentRead));
        assert!(!validated.has_permission(&Permission::UserDelete));
    }

    #[tokio::test]
    async fn test_revoke_key() {
        let service = ApiKeyService::new();
        let user_id = Uuid::new_v4();

        let generated = service
            .generate_key(user_id, "Test Key", vec![], None, None, vec![])
            .await
            .expect("Key generation should succeed");

        // Revoke the key
        service
            .revoke_key(generated.metadata.id, user_id)
            .await
            .expect("Revocation should succeed");

        // Validation should fail
        let result = service.validate_key(&generated.key).await;
        assert!(matches!(result, Err(AuthError::ApiKeyRevoked)));
    }

    #[tokio::test]
    async fn test_ip_restriction() {
        let service = ApiKeyService::new();
        let user_id = Uuid::new_v4();

        let generated = service
            .generate_key(
                user_id,
                "IP Restricted Key",
                vec![],
                None,
                None,
                vec!["192.168.1.1".to_string(), "10.0.0.1".to_string()],
            )
            .await
            .expect("Key generation should succeed");

        // Allowed IP should work
        let result = service
            .validate_key_with_ip(&generated.key, "192.168.1.1")
            .await;
        assert!(result.is_ok());

        // Disallowed IP should fail
        let result = service
            .validate_key_with_ip(&generated.key, "1.2.3.4")
            .await;
        assert!(matches!(result, Err(AuthError::InvalidApiKey)));
    }

    #[tokio::test]
    async fn test_rotate_key() {
        let service = ApiKeyService::new();
        let user_id = Uuid::new_v4();

        let original = service
            .generate_key(
                user_id,
                "Original Key",
                vec![Permission::AgentRead],
                Some(30),
                Some(500),
                vec![],
            )
            .await
            .expect("Key generation should succeed");

        // Rotate the key
        let rotated = service
            .rotate_key(original.metadata.id, user_id)
            .await
            .expect("Rotation should succeed");

        // Original key should be invalid
        let result = service.validate_key(&original.key).await;
        assert!(result.is_err());

        // New key should work
        let validated = service
            .validate_key(&rotated.key)
            .await
            .expect("New key should be valid");

        assert!(validated.has_permission(&Permission::AgentRead));
    }
}
