//! Basic usage example
//!
//! This example demonstrates how to use the Neurectomy SDK

use neurectomy_sdk::NeurectomyClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize client
    let client = NeurectomyClient::new(
        std::env::var("NEURECTOMY_API_KEY")
            .unwrap_or_else(|_| "your-api-key".to_string()),
    )?;

    // Example 1: Check API status
    println!("Checking API status...");
    match client.get_status().await {
        Ok(status) => println!("API Status: {} (v{})", status.status, status.version),
        Err(e) => println!("Status check failed: {}", e),
    }

    // Example 2: Generate text completion
    println!("\nGenerating text completion...");
    match client
        .complete(
            "Explain quantum computing in one sentence".to_string(),
            Some(50),
            Some(0.7),
        )
        .await
    {
        Ok(response) => {
            println!("Completion: {}", response.text);
            println!("Tokens: {}", response.tokens_generated);
        }
        Err(e) => println!("Completion failed: {}", e),
    }

    // Example 3: Compress text
    println!("\nCompressing text...");
    let long_text = "This is a sample text that will be compressed. ".repeat(100);
    match client
        .compress(long_text, Some(0.1), Some(5))
        .await
    {
        Ok(response) => {
            println!("Original size: {} bytes", response.original_size);
            println!("Compressed size: {} bytes", response.compressed_size);
            println!("Compression ratio: {:.2}x", response.compression_ratio);
        }
        Err(e) => println!("Compression failed: {}", e),
    }

    Ok(())
}
