//! Integration tests for Neurectomy SDK
#![cfg(test)]

use neurectomy_sdk::{NeurectomyClient, NeurectomyConfig};

#[test]
fn test_client_initialization() {
    let result = NeurectomyClient::new("test-api-key".to_string());
    assert!(result.is_ok());
}

#[test]
fn test_client_with_custom_config() {
    let config = NeurectomyConfig::new("test-key".to_string())
        .with_base_url("https://custom.api.com".to_string())
        .with_timeout(60);

    let result = NeurectomyClient::with_config(config);
    assert!(result.is_ok());
}

#[test]
fn test_client_rejects_empty_key() {
    let result = NeurectomyClient::new("".to_string());
    assert!(result.is_err());
}

#[tokio::test]
async fn test_config_builder_chain() {
    let config = NeurectomyConfig::new("api-key".to_string())
        .with_base_url("https://test.api".to_string())
        .with_timeout(45);

    assert_eq!(config.api_key, "api-key");
    assert_eq!(config.base_url, "https://test.api");
    assert_eq!(config.timeout_secs, 45);
}
