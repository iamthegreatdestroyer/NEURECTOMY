"""
Application configuration using Pydantic Settings.

@CIPHER - SECURITY: API keys use SecretStr to prevent accidental logging.
All secrets should be loaded from environment variables or secrets manager.
"""

from functools import lru_cache
from typing import List, Optional
import os

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    @CIPHER - SECURITY NOTES:
    - All API keys use SecretStr to prevent accidental logging
    - No default values for secrets - must be explicitly provided
    - Validation ensures keys are present when features are enabled
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        secrets_dir="/run/secrets",  # Docker secrets support
    )
    
    # Environment
    environment: str = "development"  # development, staging, production
    
    # Server
    host: str = "0.0.0.0"
    port: int = 16081
    debug: bool = True
    
    # Database
    database_url: str = "postgresql+asyncpg://neurectomy:neurectomy@localhost:16432/neurectomy"
    redis_url: str = "redis://localhost:16500"
    
    # CORS - @CIPHER: Explicit origins only, no wildcards in production
    cors_origins: List[str] = ["http://localhost:16000", "http://localhost:16080"]
    cors_allow_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    cors_allow_headers: List[str] = [
        "Authorization",
        "Content-Type",
        "X-Request-ID",
        "X-API-Key",
        "Accept",
        "Origin",
    ]
    cors_expose_headers: List[str] = ["X-Request-ID", "X-RateLimit-Remaining", "X-RateLimit-Reset"]
    cors_max_age: int = 3600  # 1 hour preflight cache
    
    # AI/ML - Local models (no secrets required)
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    
    # OpenAI - @CIPHER: SecretStr prevents accidental logging
    openai_api_key: Optional[SecretStr] = None
    openai_model: str = "gpt-4-turbo-preview"
    
    # Anthropic - @CIPHER: SecretStr prevents accidental logging  
    anthropic_api_key: Optional[SecretStr] = None
    anthropic_model: str = "claude-3-opus-20240229"
    
    # vLLM - Production inference server
    # @TENSOR @VELOCITY - vLLM provides 2-4x throughput vs Ollama
    vllm_url: str = "http://localhost:16081"  # vLLM OpenAI-compatible endpoint
    vllm_model: str = "meta-llama/Llama-3.2-8B-Instruct"
    vllm_api_key: Optional[SecretStr] = None  # Optional API key for vLLM
    
    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    
    # MLflow
    mlflow_tracking_uri: str = "http://localhost:16610"
    
    # Inference optimization settings
    # @VELOCITY - Performance tuning for real-time agent intelligence
    inference_max_batch_size: int = 8  # Max requests to batch together
    inference_batch_timeout_ms: int = 50  # Max wait time for batch fill
    inference_max_concurrent: int = 16  # Max concurrent LLM requests
    ollama_connection_pool_size: int = 10  # HTTP connection pool
    ollama_keepalive_timeout: int = 30  # Keep-alive seconds
    ollama_max_retries: int = 3  # Retry failed requests
    
    # Feature flags
    enable_openai: bool = False
    enable_anthropic: bool = False
    enable_ollama: bool = True
    enable_vllm: bool = False  # Enable for production high-throughput
    
    # @CIPHER - Validation: Ensure API keys are provided when features are enabled
    @model_validator(mode='after')
    def validate_api_keys(self) -> 'Settings':
        """Validate that API keys are provided when their features are enabled."""
        if self.enable_openai and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when enable_openai=True. "
                "Set OPENAI_API_KEY environment variable or disable OpenAI."
            )
        
        if self.enable_anthropic and not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required when enable_anthropic=True. "
                "Set ANTHROPIC_API_KEY environment variable or disable Anthropic."
            )
        
        # Warn about debug mode in non-development environments
        if self.environment == "production" and self.debug:
            import warnings
            warnings.warn(
                "DEBUG mode is enabled in production! This may expose sensitive information.",
                RuntimeWarning
            )
        
        return self
    
    @field_validator('cors_origins')
    @classmethod
    def validate_cors_origins(cls, v: List[str], info) -> List[str]:
        """Validate CORS origins - no wildcards in production."""
        # We'll check environment after model is fully constructed
        for origin in v:
            if '*' in origin:
                import warnings
                warnings.warn(
                    f"Wildcard CORS origin detected: {origin}. "
                    "This is a security risk in production.",
                    RuntimeWarning
                )
        return v
    
    def get_openai_key(self) -> str:
        """Safely get OpenAI API key value."""
        if self.openai_api_key:
            return self.openai_api_key.get_secret_value()
        return ""
    
    def get_anthropic_key(self) -> str:
        """Safely get Anthropic API key value."""
        if self.anthropic_api_key:
            return self.anthropic_api_key.get_secret_value()
        return ""
    
    def get_vllm_key(self) -> str:
        """Safely get vLLM API key value."""
        if self.vllm_api_key:
            return self.vllm_api_key.get_secret_value()
        return ""


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
