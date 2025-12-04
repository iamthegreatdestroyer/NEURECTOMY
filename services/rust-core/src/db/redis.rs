//! Redis Connection Pool
//! @VERTEX Redis Cache and Session Management

use redis::{aio::ConnectionManager, Client, AsyncCommands};
use std::time::Duration;

use super::{DatabaseError, DbResult};

/// Redis connection pool wrapper
#[derive(Clone)]
pub struct RedisPool {
    client: Client,
    manager: ConnectionManager,
}

impl RedisPool {
    /// Create a new Redis connection pool
    pub async fn new(redis_url: &str) -> DbResult<Self> {
        let client = Client::open(redis_url)
            .map_err(DatabaseError::Redis)?;
        
        let manager = ConnectionManager::new(client.clone())
            .await
            .map_err(DatabaseError::Redis)?;
        
        Ok(Self { client, manager })
    }
    
    /// Get a connection manager reference
    pub fn connection(&self) -> ConnectionManager {
        self.manager.clone()
    }
    
    /// Health check
    pub async fn health_check(&self) -> DbResult<()> {
        let mut conn = self.manager.clone();
        let _: String = redis::cmd("PING")
            .query_async(&mut conn)
            .await
            .map_err(DatabaseError::Redis)?;
        Ok(())
    }
    
    /// Close connections (no-op for Redis as ConnectionManager handles this)
    pub async fn close(&self) {
        // ConnectionManager handles cleanup automatically
    }
    
    // ========================================
    // Key-Value Operations
    // ========================================
    
    /// Set a key-value pair with optional expiration
    pub async fn set(&self, key: &str, value: &str, ttl_seconds: Option<u64>) -> DbResult<()> {
        let mut conn = self.manager.clone();
        
        if let Some(ttl) = ttl_seconds {
            conn.set_ex(key, value, ttl)
                .await
                .map_err(DatabaseError::Redis)?;
        } else {
            conn.set(key, value)
                .await
                .map_err(DatabaseError::Redis)?;
        }
        
        Ok(())
    }
    
    /// Get a value by key
    pub async fn get(&self, key: &str) -> DbResult<Option<String>> {
        let mut conn = self.manager.clone();
        let result: Option<String> = conn.get(key)
            .await
            .map_err(DatabaseError::Redis)?;
        Ok(result)
    }
    
    /// Delete a key
    pub async fn delete(&self, key: &str) -> DbResult<bool> {
        let mut conn = self.manager.clone();
        let result: i64 = conn.del(key)
            .await
            .map_err(DatabaseError::Redis)?;
        Ok(result > 0)
    }
    
    /// Check if key exists
    pub async fn exists(&self, key: &str) -> DbResult<bool> {
        let mut conn = self.manager.clone();
        let result: bool = conn.exists(key)
            .await
            .map_err(DatabaseError::Redis)?;
        Ok(result)
    }
    
    /// Set expiration on a key
    pub async fn expire(&self, key: &str, seconds: i64) -> DbResult<bool> {
        let mut conn = self.manager.clone();
        let result: bool = conn.expire(key, seconds)
            .await
            .map_err(DatabaseError::Redis)?;
        Ok(result)
    }
    
    // ========================================
    // Session Management
    // ========================================
    
    /// Store a session
    pub async fn set_session(&self, session_id: &str, user_id: &str, ttl_seconds: u64) -> DbResult<()> {
        let key = format!("session:{}", session_id);
        self.set(&key, user_id, Some(ttl_seconds)).await
    }
    
    /// Get session user ID
    pub async fn get_session(&self, session_id: &str) -> DbResult<Option<String>> {
        let key = format!("session:{}", session_id);
        self.get(&key).await
    }
    
    /// Delete a session
    pub async fn delete_session(&self, session_id: &str) -> DbResult<bool> {
        let key = format!("session:{}", session_id);
        self.delete(&key).await
    }
    
    /// Refresh session TTL
    pub async fn refresh_session(&self, session_id: &str, ttl_seconds: i64) -> DbResult<bool> {
        let key = format!("session:{}", session_id);
        self.expire(&key, ttl_seconds).await
    }
    
    // ========================================
    // Caching Operations
    // ========================================
    
    /// Cache a JSON value
    pub async fn cache_json<T: serde::Serialize>(
        &self,
        key: &str,
        value: &T,
        ttl_seconds: Option<u64>,
    ) -> DbResult<()> {
        let json = serde_json::to_string(value)
            .map_err(|e| DatabaseError::Pool(e.to_string()))?;
        self.set(key, &json, ttl_seconds).await
    }
    
    /// Get a cached JSON value
    pub async fn get_cached_json<T: serde::de::DeserializeOwned>(
        &self,
        key: &str,
    ) -> DbResult<Option<T>> {
        match self.get(key).await? {
            Some(json) => {
                let value = serde_json::from_str(&json)
                    .map_err(|e| DatabaseError::Pool(e.to_string()))?;
                Ok(Some(value))
            }
            None => Ok(None),
        }
    }
    
    // ========================================
    // Rate Limiting
    // ========================================
    
    /// Increment rate limit counter
    pub async fn increment_rate_limit(
        &self,
        key: &str,
        window_seconds: u64,
    ) -> DbResult<i64> {
        let mut conn = self.manager.clone();
        
        let count: i64 = conn.incr(key, 1)
            .await
            .map_err(DatabaseError::Redis)?;
        
        // Set expiration on first increment
        if count == 1 {
            conn.expire(key, window_seconds as i64)
                .await
                .map_err(DatabaseError::Redis)?;
        }
        
        Ok(count)
    }
    
    /// Check if rate limit exceeded
    pub async fn is_rate_limited(&self, key: &str, limit: i64) -> DbResult<bool> {
        let count: Option<i64> = {
            let mut conn = self.manager.clone();
            conn.get(key).await.map_err(DatabaseError::Redis)?
        };
        
        Ok(count.unwrap_or(0) >= limit)
    }
    
    // ========================================
    // Pub/Sub Operations
    // ========================================
    
    /// Publish a message to a channel
    pub async fn publish(&self, channel: &str, message: &str) -> DbResult<i64> {
        let mut conn = self.manager.clone();
        let subscribers: i64 = conn.publish(channel, message)
            .await
            .map_err(DatabaseError::Redis)?;
        Ok(subscribers)
    }
    
    // ========================================
    // List Operations (for queues)
    // ========================================
    
    /// Push to a list (queue)
    pub async fn lpush(&self, key: &str, value: &str) -> DbResult<i64> {
        let mut conn = self.manager.clone();
        let len: i64 = conn.lpush(key, value)
            .await
            .map_err(DatabaseError::Redis)?;
        Ok(len)
    }
    
    /// Pop from a list (queue)
    pub async fn rpop(&self, key: &str) -> DbResult<Option<String>> {
        let mut conn = self.manager.clone();
        let value: Option<String> = conn.rpop(key, None)
            .await
            .map_err(DatabaseError::Redis)?;
        Ok(value)
    }
    
    /// Get list length
    pub async fn llen(&self, key: &str) -> DbResult<i64> {
        let mut conn = self.manager.clone();
        let len: i64 = conn.llen(key)
            .await
            .map_err(DatabaseError::Redis)?;
        Ok(len)
    }
}
