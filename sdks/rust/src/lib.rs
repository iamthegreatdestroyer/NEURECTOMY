//! Neurectomy Rust SDK
//!
//! A type-safe, production-grade Rust SDK for the Neurectomy API.
//! Provides async/await support with tokio and comprehensive error handling.
//!
//! # Example
//!
//! ```rust,no_run
//! use neurectomy_sdk::NeurectomyClient;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = NeurectomyClient::new("your-api-key".to_string())?;
//!
//!     let response = client.complete(
//!         "Explain quantum computing".to_string(),
//!         Some(200),
//!         Some(0.7),
//!     ).await?;
//!
//!     println!("{}", response.text);
//!     Ok(())
//! }
//! ```

use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

/// Neurectomy SDK error types
#[derive(Error, Debug)]
pub enum NeurectomyError {
    /// HTTP client error
    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),

    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// API error response
    #[error("API error {code}: {message}")]
    ApiError { code: String, message: String },

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Timeout error
    #[error("Request timeout")]
    TimeoutError,

    /// Authentication error
    #[error("Authentication failed: {0}")]
    AuthError(String),
}

pub type Result<T> = std::result::Result<T, NeurectomyError>;

/// Client configuration
#[derive(Debug, Clone)]
pub struct NeurectomyConfig {
    /// API key for authentication
    pub api_key: String,
    /// Base URL for API (default: https://api.neurectomy.ai)
    pub base_url: String,
    /// Request timeout in seconds (default: 30)
    pub timeout_secs: u64,
}

impl NeurectomyConfig {
    /// Create a new config with API key
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            base_url: "https://api.neurectomy.ai".to_string(),
            timeout_secs: 30,
        }
    }

    /// Set custom base URL
    pub fn with_base_url(mut self, url: String) -> Self {
        self.base_url = url;
        self
    }

    /// Set custom timeout
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }
}

/// Text completion request
#[derive(Debug, Clone, Serialize)]
pub struct CompletionRequest {
    pub prompt: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub model: Option<String>,
    pub top_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
}

/// Text completion response
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionResponse {
    pub text: String,
    pub tokens_generated: u32,
    pub finish_reason: String,
    #[serde(default)]
    pub usage: Option<TokenUsage>,
}

/// Token usage information
#[derive(Debug, Clone, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Text compression request
#[derive(Debug, Clone, Serialize)]
pub struct CompressionRequest {
    pub text: String,
    pub target_ratio: Option<f32>,
    pub compression_level: Option<u8>,
    pub algorithm: Option<String>,
}

/// Text compression response
#[derive(Debug, Clone, Deserialize)]
pub struct CompressionResponse {
    pub compressed_data: String,
    pub compression_ratio: f32,
    pub original_size: u64,
    pub compressed_size: u64,
    pub algorithm: String,
}

/// File storage response
#[derive(Debug, Clone, Deserialize)]
pub struct StorageResponse {
    pub object_id: String,
    pub path: String,
    pub size: u64,
    pub timestamp: String,
}

/// Retrieved file data
#[derive(Debug, Clone, Deserialize)]
pub struct RetrievedFile {
    pub data: String,
    pub path: String,
    pub size: u64,
    pub timestamp: String,
}

/// API status response
#[derive(Debug, Clone, Deserialize)]
pub struct StatusResponse {
    pub status: String,
    pub version: String,
}

/// API error response
#[derive(Debug, Clone, Deserialize)]
struct ErrorResponse {
    code: String,
    message: String,
}

/// Neurectomy API Client
///
/// A type-safe, async client for the Neurectomy API.
/// All methods are async and return Results.
///
/// # Example
///
/// ```rust,no_run
/// use neurectomy_sdk::NeurectomyClient;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let client = NeurectomyClient::new("api-key".to_string())?;
///     let response = client.complete("Hello".to_string(), None, None).await?;
///     println!("{}", response.text);
///     Ok(())
/// }
/// ```
pub struct NeurectomyClient {
    http_client: HttpClient,
    config: NeurectomyConfig,
}

impl NeurectomyClient {
    /// Create a new client with API key
    pub fn new(api_key: String) -> Result<Self> {
        if api_key.is_empty() {
            return Err(NeurectomyError::ConfigError(
                "API key is required".to_string(),
            ));
        }

        let config = NeurectomyConfig::new(api_key);
        let http_client = HttpClient::new();

        Ok(Self {
            http_client,
            config,
        })
    }

    /// Create a new client with custom config
    pub fn with_config(config: NeurectomyConfig) -> Result<Self> {
        if config.api_key.is_empty() {
            return Err(NeurectomyError::ConfigError(
                "API key is required".to_string(),
            ));
        }

        let http_client = HttpClient::new();

        Ok(Self {
            http_client,
            config,
        })
    }

    /// Generate text completion
    pub async fn complete(
        &self,
        prompt: String,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) -> Result<CompletionResponse> {
        let request = CompletionRequest {
            prompt,
            max_tokens: max_tokens.or(Some(100)),
            temperature: temperature.or(Some(0.7)),
            model: Some("ryot-bitnet-7b".to_string()),
            top_p: Some(1.0),
            frequency_penalty: Some(0.0),
            presence_penalty: Some(0.0),
        };

        let url = format!("{}/v1/completions", self.config.base_url);

        let response = self
            .http_client
            .post(&url)
            .bearer_auth(&self.config.api_key)
            .json(&request)
            .timeout(std::time::Duration::from_secs(self.config.timeout_secs))
            .send()
            .await?;

        if !response.status().is_success() {
            let error: ErrorResponse = response.json().await?;
            return Err(NeurectomyError::ApiError {
                code: error.code,
                message: error.message,
            });
        }

        let body = response.json().await?;
        Ok(body)
    }

    /// Compress text
    pub async fn compress(
        &self,
        text: String,
        target_ratio: Option<f32>,
        compression_level: Option<u8>,
    ) -> Result<CompressionResponse> {
        let request = CompressionRequest {
            text,
            target_ratio: target_ratio.or(Some(0.1)),
            compression_level: compression_level.or(Some(2)),
            algorithm: Some("lz4".to_string()),
        };

        let url = format!("{}/v1/compress", self.config.base_url);

        let response = self
            .http_client
            .post(&url)
            .bearer_auth(&self.config.api_key)
            .json(&request)
            .timeout(std::time::Duration::from_secs(self.config.timeout_secs))
            .send()
            .await?;

        if !response.status().is_success() {
            let error: ErrorResponse = response.json().await?;
            return Err(NeurectomyError::ApiError {
                code: error.code,
                message: error.message,
            });
        }

        let body = response.json().await?;
        Ok(body)
    }

    /// Store file in ΣVAULT
    pub async fn store_file(
        &self,
        path: String,
        data: String,
    ) -> Result<StorageResponse> {
        let url = format!("{}/v1/storage/store", self.config.base_url);

        let payload = serde_json::json!({
            "path": path,
            "data": data,
        });

        let response = self
            .http_client
            .post(&url)
            .bearer_auth(&self.config.api_key)
            .json(&payload)
            .timeout(std::time::Duration::from_secs(self.config.timeout_secs))
            .send()
            .await?;

        if !response.status().is_success() {
            let error: ErrorResponse = response.json().await?;
            return Err(NeurectomyError::ApiError {
                code: error.code,
                message: error.message,
            });
        }

        let body = response.json().await?;
        Ok(body)
    }

    /// Retrieve file from ΣVAULT
    pub async fn retrieve_file(&self, object_id: String) -> Result<RetrievedFile> {
        let url = format!("{}/v1/storage/{}", self.config.base_url, object_id);

        let response = self
            .http_client
            .get(&url)
            .bearer_auth(&self.config.api_key)
            .timeout(std::time::Duration::from_secs(self.config.timeout_secs))
            .send()
            .await?;

        if !response.status().is_success() {
            let error: ErrorResponse = response.json().await?;
            return Err(NeurectomyError::ApiError {
                code: error.code,
                message: error.message,
            });
        }

        let body = response.json().await?;
        Ok(body)
    }

    /// Delete file from ΣVAULT
    pub async fn delete_file(&self, object_id: String) -> Result<()> {
        let url = format!("{}/v1/storage/{}", self.config.base_url, object_id);

        let response = self
            .http_client
            .delete(&url)
            .bearer_auth(&self.config.api_key)
            .timeout(std::time::Duration::from_secs(self.config.timeout_secs))
            .send()
            .await?;

        if !response.status().is_success() {
            let error: ErrorResponse = response.json().await?;
            return Err(NeurectomyError::ApiError {
                code: error.code,
                message: error.message,
            });
        }

        Ok(())
    }

    /// Get API status
    pub async fn get_status(&self) -> Result<StatusResponse> {
        let url = format!("{}/v1/status", self.config.base_url);

        let response = self
            .http_client
            .get(&url)
            .timeout(std::time::Duration::from_secs(self.config.timeout_secs))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(NeurectomyError::ApiError {
                code: "status_error".to_string(),
                message: "Failed to get API status".to_string(),
            });
        }

        let body = response.json().await?;
        Ok(body)
    }
}

impl fmt::Debug for NeurectomyClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NeurectomyClient")
            .field("config", &self.config)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = NeurectomyConfig::new("test-key".to_string());
        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.base_url, "https://api.neurectomy.ai");
        assert_eq!(config.timeout_secs, 30);
    }

    #[test]
    fn test_config_builder() {
        let config = NeurectomyConfig::new("test-key".to_string())
            .with_base_url("https://custom.api".to_string())
            .with_timeout(60);

        assert_eq!(config.base_url, "https://custom.api");
        assert_eq!(config.timeout_secs, 60);
    }

    #[test]
    fn test_client_creation() {
        let result = NeurectomyClient::new("test-key".to_string());
        assert!(result.is_ok());
    }

    #[test]
    fn test_client_missing_api_key() {
        let result = NeurectomyClient::new("".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_completion_request_serialization() {
        let request = CompletionRequest {
            prompt: "test".to_string(),
            max_tokens: Some(100),
            temperature: Some(0.7),
            model: Some("model".to_string()),
            top_p: Some(1.0),
            frequency_penalty: Some(0.0),
            presence_penalty: Some(0.0),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"prompt\":\"test\""));
        assert!(json.contains("\"max_tokens\":100"));
    }
}
