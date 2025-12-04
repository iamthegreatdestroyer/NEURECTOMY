//! WebSocket Connection Manager
//!
//! Global manager for all WebSocket connections:
//! - Connection registry
//! - Broadcast capabilities
//! - Subscription-based routing
//! - Connection cleanup
//!
//! @STREAM - Connection orchestration

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, RwLock};
use tokio::time::interval;
use tracing::{debug, info, warn};
use uuid::Uuid;

use super::connection::{WsConnection, ConnectionInfo, WsConnectionState};
use super::messages::{WsEvent, SubscriptionType};

/// Connection manager for all WebSocket connections
pub struct ConnectionManager {
    /// All active connections (connection_id -> connection)
    connections: RwLock<HashMap<Uuid, Arc<RwLock<WsConnection>>>>,
    /// User to connections mapping (user_id -> [connection_ids])
    user_connections: RwLock<HashMap<Uuid, Vec<Uuid>>>,
    /// Statistics
    stats: RwLock<ConnectionStats>,
}

#[derive(Debug, Default, Clone)]
pub struct ConnectionStats {
    pub total_connections: u64,
    pub total_disconnections: u64,
    pub current_connections: usize,
    pub authenticated_connections: usize,
    pub total_messages_sent: u64,
    pub total_messages_received: u64,
}

impl ConnectionManager {
    /// Create new connection manager
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            connections: RwLock::new(HashMap::new()),
            user_connections: RwLock::new(HashMap::new()),
            stats: RwLock::new(ConnectionStats::default()),
        })
    }

    /// Register a new connection
    pub async fn register(&self, connection: WsConnection) -> Uuid {
        let id = connection.id;
        let connection = Arc::new(RwLock::new(connection));

        {
            let mut connections = self.connections.write().await;
            connections.insert(id, connection);
        }

        {
            let mut stats = self.stats.write().await;
            stats.total_connections += 1;
            stats.current_connections += 1;
        }

        info!("WebSocket connection registered: {}", id);
        id
    }

    /// Unregister a connection
    pub async fn unregister(&self, connection_id: Uuid) {
        // Get connection to check user_id
        let user_id = {
            let connections = self.connections.read().await;
            if let Some(conn) = connections.get(&connection_id) {
                conn.read().await.user_id
            } else {
                None
            }
        };

        // Remove from user_connections if authenticated
        if let Some(user_id) = user_id {
            let mut user_conns = self.user_connections.write().await;
            if let Some(conns) = user_conns.get_mut(&user_id) {
                conns.retain(|&id| id != connection_id);
                if conns.is_empty() {
                    user_conns.remove(&user_id);
                }
            }
        }

        // Remove from connections
        {
            let mut connections = self.connections.write().await;
            if let Some(conn) = connections.remove(&connection_id) {
                conn.write().await.disconnect();
            }
        }

        {
            let mut stats = self.stats.write().await;
            stats.total_disconnections += 1;
            stats.current_connections = stats.current_connections.saturating_sub(1);
        }

        info!("WebSocket connection unregistered: {}", connection_id);
    }

    /// Mark connection as authenticated
    pub async fn authenticate(&self, connection_id: Uuid, user_id: Uuid, session_id: String) {
        let connections = self.connections.read().await;
        
        if let Some(conn) = connections.get(&connection_id) {
            conn.write().await.authenticate(user_id, session_id);
        }

        // Update user_connections mapping
        drop(connections);
        
        let mut user_conns = self.user_connections.write().await;
        user_conns
            .entry(user_id)
            .or_insert_with(Vec::new)
            .push(connection_id);

        let mut stats = self.stats.write().await;
        stats.authenticated_connections += 1;

        debug!("Connection {} authenticated for user {}", connection_id, user_id);
    }

    /// Get connection by ID
    pub async fn get(&self, connection_id: Uuid) -> Option<Arc<RwLock<WsConnection>>> {
        self.connections.read().await.get(&connection_id).cloned()
    }

    /// Get all connections for a user
    pub async fn get_user_connections(&self, user_id: Uuid) -> Vec<Arc<RwLock<WsConnection>>> {
        let user_conns = self.user_connections.read().await;
        let connections = self.connections.read().await;

        user_conns
            .get(&user_id)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| connections.get(id).cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Send event to specific connection
    pub async fn send_to(&self, connection_id: Uuid, event: WsEvent) -> bool {
        if let Some(conn) = self.get(connection_id).await {
            let result = conn.read().await.send(event);
            if result.is_ok() {
                let mut stats = self.stats.write().await;
                stats.total_messages_sent += 1;
            }
            result.is_ok()
        } else {
            false
        }
    }

    /// Send event to all connections of a user
    pub async fn send_to_user(&self, user_id: Uuid, event: WsEvent) -> usize {
        let connections = self.get_user_connections(user_id).await;
        let mut sent = 0;

        for conn in connections {
            if conn.read().await.send(event.clone()).is_ok() {
                sent += 1;
            }
        }

        if sent > 0 {
            let mut stats = self.stats.write().await;
            stats.total_messages_sent += sent as u64;
        }

        sent
    }

    /// Broadcast to all subscribed connections
    pub async fn broadcast_to_subscribers(
        &self,
        subscription_type: SubscriptionType,
        resource_id: Uuid,
        event: WsEvent,
    ) -> usize {
        let connections = self.connections.read().await;
        let mut sent = 0;

        for conn in connections.values() {
            let conn_guard = conn.read().await;
            if conn_guard.is_subscribed(subscription_type, resource_id) {
                if conn_guard.send(event.clone()).is_ok() {
                    sent += 1;
                }
            }
        }

        if sent > 0 {
            let mut stats = self.stats.write().await;
            stats.total_messages_sent += sent as u64;
        }

        debug!(
            "Broadcast {:?}/{} to {} connections",
            subscription_type, resource_id, sent
        );

        sent
    }

    /// Broadcast to all authenticated connections
    pub async fn broadcast_authenticated(&self, event: WsEvent) -> usize {
        let connections = self.connections.read().await;
        let mut sent = 0;

        for conn in connections.values() {
            let conn_guard = conn.read().await;
            if conn_guard.is_authenticated() {
                if conn_guard.send(event.clone()).is_ok() {
                    sent += 1;
                }
            }
        }

        if sent > 0 {
            let mut stats = self.stats.write().await;
            stats.total_messages_sent += sent as u64;
        }

        sent
    }

    /// Broadcast to all connections
    pub async fn broadcast_all(&self, event: WsEvent) -> usize {
        let connections = self.connections.read().await;
        let mut sent = 0;

        for conn in connections.values() {
            if conn.read().await.send(event.clone()).is_ok() {
                sent += 1;
            }
        }

        if sent > 0 {
            let mut stats = self.stats.write().await;
            stats.total_messages_sent += sent as u64;
        }

        sent
    }

    /// Get all connection info
    pub async fn list_connections(&self) -> Vec<ConnectionInfo> {
        let connections = self.connections.read().await;
        let mut infos = Vec::with_capacity(connections.len());

        for conn in connections.values() {
            infos.push(ConnectionInfo::from(&*conn.read().await));
        }

        infos
    }

    /// Get statistics
    pub async fn stats(&self) -> ConnectionStats {
        self.stats.read().await.clone()
    }

    /// Record message received
    pub async fn record_message_received(&self) {
        let mut stats = self.stats.write().await;
        stats.total_messages_received += 1;
    }

    /// Start cleanup task
    pub fn start_cleanup_task(self: Arc<Self>) {
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30));

            loop {
                interval.tick().await;
                self.cleanup_stale_connections().await;
            }
        });
    }

    /// Start heartbeat task
    pub fn start_heartbeat_task(self: Arc<Self>) {
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30));

            loop {
                interval.tick().await;
                self.send_heartbeats().await;
            }
        });
    }

    /// Clean up stale/timed out connections
    async fn cleanup_stale_connections(&self) {
        let connections = self.connections.read().await;
        let mut to_remove = Vec::new();

        for (id, conn) in connections.iter() {
            let conn_guard = conn.read().await;
            if conn_guard.is_timed_out() || conn_guard.state == WsConnectionState::Disconnected {
                to_remove.push(*id);
            }
        }

        drop(connections);

        for id in to_remove {
            warn!("Cleaning up stale connection: {}", id);
            self.unregister(id).await;
        }
    }

    /// Send heartbeats to all connections
    async fn send_heartbeats(&self) {
        let connections = self.connections.read().await;
        let timestamp = chrono::Utc::now().timestamp();

        for conn in connections.values() {
            let conn_guard = conn.read().await;
            if conn_guard.needs_heartbeat() {
                let _ = conn_guard.send(WsEvent::Heartbeat { timestamp });
            }
        }
    }
}

impl Default for ConnectionManager {
    fn default() -> Self {
        Self {
            connections: RwLock::new(HashMap::new()),
            user_connections: RwLock::new(HashMap::new()),
            stats: RwLock::new(ConnectionStats::default()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_register_unregister() {
        let manager = ConnectionManager::new();
        let (tx, _rx) = mpsc::unbounded_channel();
        let conn = WsConnection::new(tx);
        let conn_id = conn.id;

        // Register
        let id = manager.register(conn).await;
        assert_eq!(id, conn_id);

        let stats = manager.stats().await;
        assert_eq!(stats.current_connections, 1);

        // Unregister
        manager.unregister(conn_id).await;
        
        let stats = manager.stats().await;
        assert_eq!(stats.current_connections, 0);
        assert_eq!(stats.total_disconnections, 1);
    }

    #[tokio::test]
    async fn test_authentication() {
        let manager = ConnectionManager::new();
        let (tx, _rx) = mpsc::unbounded_channel();
        let conn = WsConnection::new(tx);
        let conn_id = conn.id;

        manager.register(conn).await;

        let user_id = Uuid::new_v4();
        manager.authenticate(conn_id, user_id, "session123".to_string()).await;

        let connections = manager.get_user_connections(user_id).await;
        assert_eq!(connections.len(), 1);
    }

    #[tokio::test]
    async fn test_broadcast_to_subscribers() {
        let manager = ConnectionManager::new();
        let agent_id = Uuid::new_v4();

        // Create and register connection 1 (subscribed)
        let (tx1, mut rx1) = mpsc::unbounded_channel();
        let mut conn1 = WsConnection::new(tx1);
        conn1.subscribe(SubscriptionType::Agent, agent_id);
        manager.register(conn1).await;

        // Create and register connection 2 (not subscribed)
        let (tx2, mut rx2) = mpsc::unbounded_channel();
        let conn2 = WsConnection::new(tx2);
        manager.register(conn2).await;

        // Broadcast
        let event = WsEvent::agent_chunk(agent_id, Uuid::new_v4(), Uuid::new_v4(), "test", false);
        let sent = manager.broadcast_to_subscribers(SubscriptionType::Agent, agent_id, event).await;

        assert_eq!(sent, 1);

        // Only conn1 should receive
        assert!(rx1.try_recv().is_ok());
        assert!(rx2.try_recv().is_err());
    }
}
