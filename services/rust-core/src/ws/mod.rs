//! WebSocket Module for Real-Time Communication
//!
//! Provides real-time bidirectional communication for:
//! - Agent response streaming
//! - Container log streaming
//! - Training progress updates
//! - System notifications
//!
//! @STREAM - Real-time data processing

pub mod connection;
pub mod handlers;
pub mod manager;
pub mod messages;
pub mod router;

pub use connection::{WsConnection, WsConnectionState};
pub use handlers::{
    agent_stream_handler, container_logs_handler, notification_handler, training_progress_handler,
};
pub use manager::ConnectionManager;
pub use messages::{WsCommand, WsEvent, WsMessage};
pub use router::create_ws_router;
