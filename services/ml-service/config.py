"""
ML Service Configuration

Environment-based configuration for MLflow, Optuna, and training services.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Service Configuration
    service_name: str = "ml-service"
    service_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 16081
    debug: bool = False
    
    # MLflow Configuration
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_artifact_root: str = "s3://mlflow-artifacts"
    mlflow_backend_store_uri: str = "postgresql://mlflow:mlflow@postgres:5432/mlflow"
    
    # Optuna Configuration
    optuna_storage: str = "postgresql://optuna:optuna@postgres:5432/optuna"
    optuna_default_sampler: str = "tpe"
    optuna_default_pruner: str = "median"
    
    # MinIO/S3 Configuration
    s3_endpoint_url: str = "http://minio:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "mlflow-artifacts"
    
    # PostgreSQL Configuration
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_user: str = "mlflow"
    postgres_password: str = "mlflow"
    postgres_db: str = "mlflow"
    
    # Training Configuration
    default_batch_size: int = 32
    default_epochs: int = 10
    default_learning_rate: float = 0.001
    max_parallel_trials: int = 4
    gpu_memory_fraction: float = 0.9
    
    # WebSocket Configuration
    ws_heartbeat_interval: int = 30
    ws_max_connections: int = 100
    
    # CORS Configuration
    cors_origins: str = "http://localhost:16000,http://localhost:16080,tauri://localhost,http://localhost:1420"
    
    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string"""
        return [origin.strip() for origin in self.cors_origins.split(',')]
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
