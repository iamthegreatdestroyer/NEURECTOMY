"""
ML Service API Layer.

@SYNAPSE @APEX - FastAPI routes for ML services.

Exposes endpoints for:
- LLM inference and chat
- Embedding generation and search
- Training orchestration
- Agent intelligence operations
"""

from src.api.routes import router

__all__ = ["router"]
