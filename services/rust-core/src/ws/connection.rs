//! WebSocket Connection Management
//!
//! Handles individual WebSocket connections:
//! - Connection state tracking
//! - Subscription management
//! - Authentication state
//! - Heartbeat/keepalive
//!
//! @STREAM - Connection lifecycle

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use tokio::sync::mpsc;
use uuid::Uuid;

use super::messages::{ActiveSubscription, SubscriptionType, WsEvent};

/// Heartbeat interval
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(30);

/// Connection timeout (no pong received)
const CONNECTION_TIMEOUT: Duration = Duration::from_secs(60);

/// Connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WsConnectionState {
    /// Newly connected, not yet authenticated
    Connected,
    /// Authentication in progress
    Authenticating,
    /// Fully authenticated
    Authenticated,
    /// Disconnecting (graceful)
    Disconnecting,
    /// Disconnected
    Disconnected,
}

/// Subscription entry
#[derive(Debug, Clone)]
pub struct Subscription {
    pub subscription_type: SubscriptionType,
    pub resource_id: Uuid,
    pub subscribed_at: DateTime<Utc>,
}

impl From<&Subscription> for ActiveSubscription {
    fn from(sub: &Subscription) -> Self {
        ActiveSubscription {
            subscription_type: sub.subscription_type,
            resource_id: sub.resource_id,
            subscribed_at: sub.subscribed_at,
        }
    }
}

/// WebSocket connection
pub struct WsConnection {
    /// Unique connection ID
    pub id: Uuid,
    /// User ID (if authenticated)
    pub user_id: Option<Uuid>,
    /// Session ID (if authenticated)
    pub session_id: Option<String>,
    /// Connection state
    pub state: WsConnectionState,
    /// Active subscriptions
    subscriptions: HashSet<Subscription>,
    /// Connected timestamp
    pub connected_at: DateTime<Utc>,
    /// Last activity timestamp
    pub last_activity: Instant,
    /// Last pong received
    pub last_pong: Instant,
    /// Sender channel to this connection
    pub sender: mpsc::UnboundedSender<WsEvent>,
    /// Client IP address
    pub client_ip: Option<String>,
    /// User agent
    pub user_agent: Option<String>,
}

impl WsConnection {
    /// Create new connection
    pub fn new(sender: mpsc::UnboundedSender<WsEvent>) -> Self {
        let now = Instant::now();
        Self {
            id: Uuid::new_v4(),
            user_id: None,
            session_id: None,
            state: WsConnectionState::Connected,
            subscriptions: HashSet::new(),
            connected_at: Utc::now(),
            last_activity: now,
            last_pong: now,
            sender,
            client_ip: None,
            user_agent: None,
        }
    }

    /// Set connection metadata
    pub fn with_metadata(mut self, client_ip: Option<String>, user_agent: Option<String>) -> Self {
        self.client_ip = client_ip;
        self.user_agent = user_agent;
        self
    }

    /// Check if authenticated
    pub fn is_authenticated(&self) -> bool {
        self.state == WsConnectionState::Authenticated && self.user_id.is_some()
    }

    /// Authenticate the connection
    pub fn authenticate(&mut self, user_id: Uuid, session_id: String) {
        self.user_id = Some(user_id);
        self.session_id = Some(session_id);
        self.state = WsConnectionState::Authenticated;
        self.touch();
    }

    /// Add subscription
    pub fn subscribe(&mut self, subscription_type: SubscriptionType, resource_id: Uuid) -> bool {
        let subscription = Subscription {
            subscription_type,
            resource_id,
            subscribed_at: Utc::now(),
        };
        let added = self.subscriptions.insert(subscription);
        self.touch();
        added
    }

    /// Remove subscription
    pub fn unsubscribe(&mut self, subscription_type: SubscriptionType, resource_id: Uuid) -> bool {
        let subscription = Subscription {
            subscription_type,
            resource_id,
            subscribed_at: Utc::now(), // Not used for comparison
        };
        let removed = self.subscriptions.remove(&subscription);
        self.touch();
        removed
    }

    /// Check if subscribed to a resource
    pub fn is_subscribed(&self, subscription_type: SubscriptionType, resource_id: Uuid) -> bool {
        self.subscriptions
            .iter()
            .any(|s| s.subscription_type == subscription_type && s.resource_id == resource_id)
    }

    /// Get all subscriptions of a type
    pub fn get_subscriptions(&self, subscription_type: SubscriptionType) -> Vec<Uuid> {
        self.subscriptions
            .iter()
            .filter(|s| s.subscription_type == subscription_type)
            .map(|s| s.resource_id)
            .collect()
    }

    /// Get all active subscriptions
    pub fn all_subscriptions(&self) -> Vec<ActiveSubscription> {
        self.subscriptions.iter().map(|s| s.into()).collect()
    }

    /// Remove all subscriptions of a type
    pub fn unsubscribe_all(&mut self, subscription_type: SubscriptionType) -> usize {
        let before = self.subscriptions.len();
        self.subscriptions
            .retain(|s| s.subscription_type != subscription_type);
        let removed = before - self.subscriptions.len();
        self.touch();
        removed
    }

    /// Clear all subscriptions
    pub fn clear_subscriptions(&mut self) {
        self.subscriptions.clear();
        self.touch();
    }

    /// Update last activity
    pub fn touch(&mut self) {
        self.last_activity = Instant::now();
    }

    /// Record pong received
    pub fn record_pong(&mut self) {
        self.last_pong = Instant::now();
        self.touch();
    }

    /// Check if connection needs heartbeat
    pub fn needs_heartbeat(&self) -> bool {
        self.last_activity.elapsed() >= HEARTBEAT_INTERVAL
    }

    /// Check if connection has timed out
    pub fn is_timed_out(&self) -> bool {
        self.last_pong.elapsed() >= CONNECTION_TIMEOUT
    }

    /// Send event to connection
    pub fn send(&self, event: WsEvent) -> Result<(), mpsc::error::SendError<WsEvent>> {
        self.sender.send(event)
    }

    /// Build state response
    pub fn state_response(&self) -> WsEvent {
        WsEvent::State {
            authenticated: self.is_authenticated(),
            user_id: self.user_id,
            subscriptions: self.all_subscriptions(),
            connected_at: self.connected_at,
        }
    }

    /// Begin disconnect
    pub fn begin_disconnect(&mut self) {
        self.state = WsConnectionState::Disconnecting;
    }

    /// Mark as disconnected
    pub fn disconnect(&mut self) {
        self.state = WsConnectionState::Disconnected;
        self.clear_subscriptions();
    }
}

/// Connection info (safe to share)
#[derive(Debug, Clone)]
pub struct ConnectionInfo {
    pub id: Uuid,
    pub user_id: Option<Uuid>,
    pub state: WsConnectionState,
    pub subscription_count: usize,
    pub connected_at: DateTime<Utc>,
    pub client_ip: Option<String>,
}

impl From<&WsConnection> for ConnectionInfo {
    fn from(conn: &WsConnection) -> Self {
        ConnectionInfo {
            id: conn.id,
            user_id: conn.user_id,
            state: conn.state,
            subscription_count: conn.subscriptions.len(),
            connected_at: conn.connected_at,
            client_ip: conn.client_ip.clone(),
        }
    }
}

// Custom Eq/Hash for Subscription (only compare type and resource_id)
impl PartialEq for Subscription {
    fn eq(&self, other: &Self) -> bool {
        self.subscription_type == other.subscription_type && self.resource_id == other.resource_id
    }
}

impl Eq for Subscription {}

impl std::hash::Hash for Subscription {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.subscription_type.hash(state);
        self.resource_id.hash(state);
    }
}

// Make SubscriptionType hashable
impl std::hash::Hash for SubscriptionType {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connection_lifecycle() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut conn = WsConnection::new(tx);

        assert!(!conn.is_authenticated());
        assert_eq!(conn.state, WsConnectionState::Connected);

        // Authenticate
        let user_id = Uuid::new_v4();
        conn.authenticate(user_id, "session123".to_string());

        assert!(conn.is_authenticated());
        assert_eq!(conn.user_id, Some(user_id));
    }

    #[test]
    fn test_subscriptions() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut conn = WsConnection::new(tx);
        let agent_id = Uuid::new_v4();

        // Subscribe
        assert!(conn.subscribe(SubscriptionType::Agent, agent_id));
        assert!(conn.is_subscribed(SubscriptionType::Agent, agent_id));

        // Duplicate subscription
        assert!(!conn.subscribe(SubscriptionType::Agent, agent_id));

        // Unsubscribe
        assert!(conn.unsubscribe(SubscriptionType::Agent, agent_id));
        assert!(!conn.is_subscribed(SubscriptionType::Agent, agent_id));
    }

    #[test]
    fn test_multiple_subscription_types() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut conn = WsConnection::new(tx);

        let agent_id = Uuid::new_v4();
        let container_id = Uuid::new_v4();
        let job_id = Uuid::new_v4();

        conn.subscribe(SubscriptionType::Agent, agent_id);
        conn.subscribe(SubscriptionType::ContainerLogs, container_id);
        conn.subscribe(SubscriptionType::Training, job_id);

        assert_eq!(conn.get_subscriptions(SubscriptionType::Agent).len(), 1);
        assert_eq!(
            conn.get_subscriptions(SubscriptionType::ContainerLogs)
                .len(),
            1
        );
        assert_eq!(conn.get_subscriptions(SubscriptionType::Training).len(), 1);

        // Unsubscribe all of one type
        conn.unsubscribe_all(SubscriptionType::Agent);
        assert_eq!(conn.get_subscriptions(SubscriptionType::Agent).len(), 0);
        assert_eq!(
            conn.get_subscriptions(SubscriptionType::ContainerLogs)
                .len(),
            1
        );
    }
}
