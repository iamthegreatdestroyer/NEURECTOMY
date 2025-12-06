"""
Application configuration using Pydantic Settings.
"""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8081
    debug: bool = True
    
    # Database
    database_url: str = "postgresql+asyncpg://neurectomy:neurectomy@localhost:5432/neurectomy"
    redis_url: str = "redis://localhost:6379"
    
    # CORS
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    
    # AI/ML
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    
    openai_api_key: str = ""
    openai_model: str = "gpt-4-turbo-preview"
    
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-opus-20240229"
    
    # vLLM - Production inference server
    # @TENSOR @VELOCITY - vLLM provides 2-4x throughput vs Ollama
    vllm_url: str = "http://localhost:8000"  # vLLM OpenAI-compatible endpoint
    vllm_model: str = "meta-llama/Llama-3.2-8B-Instruct"
    vllm_api_key: str = ""  # Optional API key for vLLM
    
    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    
    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    
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


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
