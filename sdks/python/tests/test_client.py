"""
Test suite for Neurectomy Python SDK
"""

import pytest
from neurectomy import (
    NeurectomyClient,
    ConfigError,
    NeurectomyError,
    APIError,
    CompletionResponse,
    TokenUsage,
)


def test_client_initialization():
    """Test client initialization"""
    client = NeurectomyClient(api_key="test-key")
    assert client.api_key == "test-key"
    assert client.base_url == "https://api.neurectomy.ai"
    assert client.timeout == 30


def test_client_initialization_empty_key():
    """Test client rejects empty API key"""
    with pytest.raises(ConfigError):
        NeurectomyClient(api_key="")


def test_client_custom_config():
    """Test client with custom configuration"""
    client = NeurectomyClient(
        api_key="test-key",
        base_url="https://custom.api.com",
        timeout=60,
    )
    assert client.base_url == "https://custom.api.com"
    assert client.timeout == 60


def test_completion_response():
    """Test CompletionResponse dataclass"""
    usage = TokenUsage(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
    )
    response = CompletionResponse(
        text="test response",
        tokens_generated=20,
        finish_reason="stop",
        usage=usage,
    )
    assert response.text == "test response"
    assert response.tokens_generated == 20
    assert response.usage.total_tokens == 30


def test_session_headers():
    """Test that session headers are properly set"""
    client = NeurectomyClient(api_key="test-key")
    assert "Authorization" in client.session.headers
    assert "Bearer test-key" in client.session.headers["Authorization"]
    assert client.session.headers["Content-Type"] == "application/json"


def test_client_url_normalization():
    """Test that base URL is properly normalized"""
    client = NeurectomyClient(
        api_key="test-key",
        base_url="https://api.neurectomy.ai/",
    )
    assert client.base_url == "https://api.neurectomy.ai"
