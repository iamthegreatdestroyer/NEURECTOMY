"""
ML Services module.

@TENSOR @LINGUA @NEURAL - Core AI/ML services for Intelligence Foundry.
"""

from src.services.llm import LLMService, ModelRouter
from src.services.embeddings import EmbeddingService
from src.services.training import TrainingOrchestrator
from src.services.mlflow_tracker import MLflowTracker
from src.services.agent_intelligence import AgentIntelligenceService
from src.services.analytics import PredictiveAnalyticsService
from src.services.inference_optimizer import (
    InferenceMetrics,
    InferenceBatcher,
    OptimizedHttpClient,
    ConcurrentExecutor,
    get_inference_metrics,
    create_ollama_client,
    create_vllm_client,
    create_executor,
    parallel_memory_and_llm,
)

__all__ = [
    "LLMService",
    "ModelRouter",
    "EmbeddingService",
    "TrainingOrchestrator",
    "MLflowTracker",
    "AgentIntelligenceService",
    "PredictiveAnalyticsService",
    # Inference optimization
    "InferenceMetrics",
    "InferenceBatcher",
    "OptimizedHttpClient",
    "ConcurrentExecutor",
    "get_inference_metrics",
    "create_ollama_client",
    "create_vllm_client",
    "create_executor",
    "parallel_memory_and_llm",
]
