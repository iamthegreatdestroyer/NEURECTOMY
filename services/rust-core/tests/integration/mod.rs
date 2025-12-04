//! Integration Test Suite
//!
//! Full-stack integration tests for NEURECTOMY core services.
//! Uses testcontainers for isolated database instances.
//!
//! @ECLIPSE @SYNAPSE - Integration testing

mod api;
mod auth;
mod database;
mod websocket;

pub use api::*;
pub use auth::*;
pub use database::*;
pub use websocket::*;
