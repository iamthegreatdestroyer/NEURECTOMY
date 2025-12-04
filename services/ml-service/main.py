"""
NEURECTOMY ML Service

Intelligence Foundry backend providing:
- Custom model training pipelines
- LLM integration (OpenAI, Anthropic, Ollama)
- Embedding generation and similarity search
- Model serving and inference
- Agent intelligence and memory systems
- IDE Copilot functionality

@TENSOR @FLUX @ARCHITECT - Phase 2: Intelligence Layer & AI Integration
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import structlog

from src.config import settings
from src.api import router as api_router
from src.db import init_db
from src.services import (
    LLMService,
    EmbeddingService,
    TrainingOrchestrator,
    MLflowTracker,
    AgentIntelligenceService,
)
from src.monitoring import get_metrics, get_content_type, PROMETHEUS_AVAILABLE

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("🧠 Starting NEURECTOMY ML Service...")
    logger.info(f"Environment: {settings.environment}")
    
    # Initialize database connections
    await init_db()
    logger.info("✅ Database initialized (PostgreSQL + pgvector)")
    
    # Initialize ML services
    app.state.llm_service = LLMService()
    await app.state.llm_service.initialize()
    logger.info("✅ LLM Service initialized")
    
    app.state.embedding_service = EmbeddingService()
    await app.state.embedding_service.initialize()
    logger.info("✅ Embedding Service initialized")
    
    app.state.training_orchestrator = TrainingOrchestrator()
    await app.state.training_orchestrator.initialize()
    logger.info("✅ Training Orchestrator initialized")
    
    app.state.mlflow_tracker = MLflowTracker()
    await app.state.mlflow_tracker.initialize()
    logger.info("✅ MLflow Tracker initialized")
    
    app.state.agent_intelligence = AgentIntelligenceService()
    await app.state.agent_intelligence.initialize()
    logger.info("✅ Agent Intelligence Service initialized")
    
    logger.info("🚀 NEURECTOMY ML Service ready!")
    
    yield
    
    # Cleanup
    logger.info("Shutting down ML Service...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="NEURECTOMY ML Service",
        description="""
Intelligence Foundry - AI/ML Backend

## Features
- **LLM Integration**: Multi-provider support (Ollama, OpenAI, Anthropic)
- **Embeddings**: Vector generation and semantic search
- **Training**: PyTorch training orchestration with Optuna HPO
- **MLflow**: Experiment tracking and model registry
- **Agent Intelligence**: Memory systems, behavior modeling, learning

@TENSOR @FLUX @ARCHITECT - Phase 2: Intelligence Layer & AI Integration
        """,
        version="0.2.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(api_router, prefix="/api/v1")
    
    # Prometheus metrics endpoint
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(
            content=get_metrics(),
            media_type=get_content_type()
        )
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "ml-service"}
    
    @app.get("/ready")
    async def readiness_check():
        # TODO: Add actual readiness checks
        return {"status": "ready"}
    
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )
