"""
NEURECTOMY ML Service

Intelligence Foundry backend providing:
- Custom model training pipelines
- LLM integration (OpenAI, Anthropic, Ollama)
- Embedding generation and similarity search
- Model serving and inference
- IDE Copilot functionality
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from src.config import settings
from src.api import router as api_router
from src.db import init_db
from src.services.llm import LLMService
from src.services.embeddings import EmbeddingService

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("🧠 Starting NEURECTOMY ML Service...")
    
    # Initialize database connections
    await init_db()
    logger.info("✅ Database initialized")
    
    # Initialize ML services
    app.state.llm_service = LLMService()
    app.state.embedding_service = EmbeddingService()
    logger.info("✅ ML services initialized")
    
    yield
    
    # Cleanup
    logger.info("Shutting down ML Service...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="NEURECTOMY ML Service",
        description="Intelligence Foundry - AI/ML Backend",
        version="0.1.0",
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
