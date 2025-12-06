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
@CIPHER @FORTRESS - Security hardening with OWASP best practices
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse
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
from src.monitoring import (
    get_metrics,
    get_content_type,
    PROMETHEUS_AVAILABLE,
    # OpenTelemetry tracing
    init_tracer,
    shutdown_tracer,
    instrument_fastapi,
)

logger = structlog.get_logger()


# ==============================================================================
# Security Middleware
# ==============================================================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    @CIPHER @FORTRESS - Add security headers to all responses.
    
    Headers added:
    - X-Content-Type-Options: nosniff
    - X-Frame-Options: DENY (clickjacking protection)
    - X-XSS-Protection: 0 (deprecated but harmless)
    - Referrer-Policy: strict-origin-when-cross-origin
    - Content-Security-Policy: Restrictive CSP
    - Permissions-Policy: Restrict browser features
    - Cross-Origin-Opener-Policy: same-origin
    - Cross-Origin-Embedder-Policy: require-corp
    - Cross-Origin-Resource-Policy: same-origin
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> StarletteResponse:
        response = await call_next(request)
        
        # @CIPHER - Standard security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "0"  # Modern browsers don't need this
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # @CIPHER - HSTS: Force HTTPS for 2 years with preload
        # Only set in production to avoid development issues
        if settings.environment == "production":
            response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
        
        # Content Security Policy (CSP)
        # Restrictive for API service - adjust if serving HTML
        csp_directives = [
            "default-src 'none'",
            "frame-ancestors 'none'",
            "form-action 'none'",
            "base-uri 'none'",
        ]
        response.headers["Content-Security-Policy"] = "; ".join(csp_directives)
        
        # Permissions Policy (formerly Feature-Policy)
        permissions = [
            "accelerometer=()",
            "camera=()",
            "geolocation=()",
            "gyroscope=()",
            "magnetometer=()",
            "microphone=()",
            "payment=()",
            "usb=()",
        ]
        response.headers["Permissions-Policy"] = ", ".join(permissions)
        
        # Cross-Origin headers
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        
        # Cache control for API responses
        if "Cache-Control" not in response.headers:
            response.headers["Cache-Control"] = "no-store, max-age=0"
        
        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    @SENTRY - Add X-Request-ID for request correlation and tracing.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> StarletteResponse:
        # Use existing request ID or generate new one
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        
        # Add to request state for use in handlers
        request.state.request_id = request_id
        
        # Add to structlog context
        structlog.contextvars.bind_contextvars(request_id=request_id)
        
        response = await call_next(request)
        
        # Add to response for client correlation
        response.headers["X-Request-ID"] = request_id
        
        # Clear context
        structlog.contextvars.unbind_contextvars("request_id")
        
        return response


class ProductionErrorMiddleware(BaseHTTPMiddleware):
    """
    @FORTRESS - Sanitize error responses in production.
    
    Prevents information leakage through stack traces and detailed errors.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> StarletteResponse:
        try:
            return await call_next(request)
        except Exception as e:
            request_id = getattr(request.state, "request_id", "unknown")
            
            # Log full error internally
            logger.error(
                "Unhandled exception",
                request_id=request_id,
                path=request.url.path,
                method=request.method,
                exc_info=True,
            )
            
            # Return sanitized error in production
            if settings.environment == "production":
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": "Internal server error",
                        "request_id": request_id,
                        "message": "An unexpected error occurred. Please contact support with your request ID.",
                    },
                )
            
            # Re-raise in development for detailed errors
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("🧠 Starting NEURECTOMY ML Service...")
    logger.info(f"Environment: {settings.environment}")
    
    # Initialize OpenTelemetry tracing
    init_tracer(
        service_name="neurectomy-ml-service",
        jaeger_host=getattr(settings, "jaeger_host", "jaeger"),
        jaeger_port=getattr(settings, "jaeger_port", 6831),
        environment=settings.environment,
    )
    logger.info("✅ OpenTelemetry tracing initialized (Jaeger export)")
    
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
    shutdown_tracer()
    logger.info("✅ OpenTelemetry tracing shutdown complete")


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
- **Observability**: OpenTelemetry tracing with Jaeger, Prometheus metrics

@TENSOR @FLUX @ARCHITECT @SENTRY - Phase 2: Intelligence Layer & AI Integration
        """,
        version="0.2.0",
        docs_url="/docs" if settings.debug else None,  # @CIPHER - Disable docs in production
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
    )
    
    # Instrument FastAPI with OpenTelemetry
    instrument_fastapi(app)
    
    # @CIPHER @FORTRESS - Security middleware (order matters - first added = last executed)
    # Add production error handling first (catches errors from all other middleware)
    if settings.environment == "production":
        app.add_middleware(ProductionErrorMiddleware)
    
    # Add security headers to all responses
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Add request ID for correlation
    app.add_middleware(RequestIDMiddleware)
    
    # CORS middleware
    # @CIPHER - SECURITY: Explicit allow lists, no wildcards
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
        expose_headers=settings.cors_expose_headers,
        max_age=settings.cors_max_age,
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
