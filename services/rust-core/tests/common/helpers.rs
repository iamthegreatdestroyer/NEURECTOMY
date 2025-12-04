//! Test Helpers
//!
//! Utility functions for setting up and running tests.
//!
//! @ECLIPSE - Test infrastructure utilities

use std::net::SocketAddr;
use tokio::sync::OnceCell;

/// Test application instance
static TEST_APP: OnceCell<TestApp> = OnceCell::const_new();

/// Test application wrapper
pub struct TestApp {
    pub addr: SocketAddr,
    pub client: reqwest::Client,
}

impl TestApp {
    /// Get or initialize the test application
    pub async fn get() -> &'static TestApp {
        TEST_APP
            .get_or_init(|| async {
                // Initialize test app (would start server in background)
                TestApp {
                    addr: "127.0.0.1:0".parse().unwrap(),
                    client: reqwest::Client::new(),
                }
            })
            .await
    }

    /// Make GET request to test server
    pub async fn get_json<T>(&self, path: &str) -> Result<T, TestError>
    where
        T: serde::de::DeserializeOwned,
    {
        let url = format!("http://{}{}", self.addr, path);
        let response = self.client.get(&url).send().await?;
        let json = response.json().await?;
        Ok(json)
    }

    /// Make POST request with JSON body
    pub async fn post_json<B, R>(&self, path: &str, body: &B) -> Result<R, TestError>
    where
        B: serde::Serialize,
        R: serde::de::DeserializeOwned,
    {
        let url = format!("http://{}{}", self.addr, path);
        let response = self.client.post(&url).json(body).send().await?;
        let json = response.json().await?;
        Ok(json)
    }

    /// Make authenticated request
    pub async fn get_authed<T>(&self, path: &str, token: &str) -> Result<T, TestError>
    where
        T: serde::de::DeserializeOwned,
    {
        let url = format!("http://{}{}", self.addr, path);
        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", token))
            .send()
            .await?;
        let json = response.json().await?;
        Ok(json)
    }

    /// Execute GraphQL query
    pub async fn graphql<T>(
        &self,
        query: &str,
        variables: Option<serde_json::Value>,
    ) -> Result<GraphQLResponse<T>, TestError>
    where
        T: serde::de::DeserializeOwned,
    {
        let url = format!("http://{}/graphql", self.addr);
        let body = serde_json::json!({
            "query": query,
            "variables": variables.unwrap_or(serde_json::json!({}))
        });

        let response = self.client.post(&url).json(&body).send().await?;
        let json = response.json().await?;
        Ok(json)
    }
}

#[derive(Debug, serde::Deserialize)]
pub struct GraphQLResponse<T> {
    pub data: Option<T>,
    pub errors: Option<Vec<GraphQLError>>,
}

#[derive(Debug, serde::Deserialize)]
pub struct GraphQLError {
    pub message: String,
    pub locations: Option<Vec<GraphQLLocation>>,
    pub path: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, serde::Deserialize)]
pub struct GraphQLLocation {
    pub line: i32,
    pub column: i32,
}

#[derive(Debug, thiserror::Error)]
pub enum TestError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Assertion failed: {0}")]
    Assertion(String),
}

/// Assert that a GraphQL response has no errors
pub fn assert_no_graphql_errors<T>(response: &GraphQLResponse<T>) {
    if let Some(errors) = &response.errors {
        panic!(
            "GraphQL errors: {:?}",
            errors.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
    }
}

/// Test context manager for setup/teardown
pub struct TestContext {
    cleanup_fns: Vec<Box<dyn FnOnce() + Send>>,
}

impl TestContext {
    pub fn new() -> Self {
        TestContext {
            cleanup_fns: Vec::new(),
        }
    }

    pub fn on_cleanup<F>(&mut self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.cleanup_fns.push(Box::new(f));
    }
}

impl Default for TestContext {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for TestContext {
    fn drop(&mut self) {
        for cleanup in self.cleanup_fns.drain(..).rev() {
            cleanup();
        }
    }
}

/// Retry helper for flaky operations
pub async fn retry<F, T, E>(
    max_attempts: u32,
    delay_ms: u64,
    mut f: F,
) -> Result<T, E>
where
    F: FnMut() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T, E>> + Send>>,
    E: std::fmt::Debug,
{
    let mut attempts = 0;
    loop {
        attempts += 1;
        match f().await {
            Ok(result) => return Ok(result),
            Err(e) if attempts >= max_attempts => return Err(e),
            Err(_) => {
                tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;
            }
        }
    }
}

/// Timeout wrapper for async tests
pub async fn with_timeout<F, T>(duration_secs: u64, f: F) -> Result<T, &'static str>
where
    F: std::future::Future<Output = T>,
{
    tokio::time::timeout(
        tokio::time::Duration::from_secs(duration_secs),
        f,
    )
    .await
    .map_err(|_| "Test timed out")
}

/// Assert that a future completes within timeout
#[macro_export]
macro_rules! assert_completes {
    ($future:expr) => {
        assert_completes!($future, 5)
    };
    ($future:expr, $timeout_secs:expr) => {
        tokio::time::timeout(
            tokio::time::Duration::from_secs($timeout_secs),
            $future,
        )
        .await
        .expect("Operation did not complete within timeout")
    };
}

/// Assert JSON equality with pretty diff
#[macro_export]
macro_rules! assert_json_eq {
    ($left:expr, $right:expr) => {
        let left_str = serde_json::to_string_pretty(&$left).unwrap();
        let right_str = serde_json::to_string_pretty(&$right).unwrap();
        pretty_assertions::assert_eq!(left_str, right_str);
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_cleanup() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let cleaned_up = Arc::new(AtomicBool::new(false));
        let cleaned_up_clone = cleaned_up.clone();

        {
            let mut ctx = TestContext::new();
            ctx.on_cleanup(move || {
                cleaned_up_clone.store(true, Ordering::SeqCst);
            });
        }

        assert!(cleaned_up.load(Ordering::SeqCst));
    }
}
