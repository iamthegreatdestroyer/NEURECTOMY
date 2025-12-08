"""
Intelligence Foundry ML Service - Main Application

FastAPI application entry point specifically for Intelligence Foundry backend.
Integrates MLflow, Optuna, and Training Engine with WebSocket support.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime
import json

from config import settings
import mlflow_server
import optuna_service

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================================
# WebSocket Connection Manager
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket):
        """Accept and store a new WebSocket connection"""
        await websocket.accept()
        async with self.lock:
            self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    async def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        async with self.lock:
            self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
        
        # Convert message to JSON
        message_json = json.dumps(message)
        
        # Send to all connections
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                logger.error(f"Failed to send message to client: {e}")
                disconnected.add(connection)
        
        # Remove disconnected clients
        if disconnected:
            async with self.lock:
                self.active_connections -= disconnected
    
    async def send_personal(self, websocket: WebSocket, message: dict):
        """Send message to a specific client"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send personal message: {e}")
    
    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)


# Global connection manager
manager = ConnectionManager()

# ============================================================================
# Application Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("=" * 80)
    logger.info(f"Starting {settings.service_name} v{settings.service_version}")
    logger.info("=" * 80)
    
    # Test MLflow connection
    try:
        import mlflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        logger.info(f"✓ Connected to MLflow: {settings.mlflow_tracking_uri}")
    except Exception as e:
        logger.error(f"✗ Failed to connect to MLflow: {e}")
    
    # Test Optuna connection
    try:
        import optuna
        storage = optuna.storages.RDBStorage(url=settings.optuna_storage)
        logger.info(f"✓ Connected to Optuna: {settings.optuna_storage}")
    except Exception as e:
        logger.error(f"✗ Failed to connect to Optuna: {e}")
    
    # Log configuration
    logger.info(f"Host: {settings.host}:{settings.port}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"CORS origins: {settings.cors_origins_list}")
    logger.info(f"Max WebSocket connections: {settings.ws_max_connections}")
    logger.info("=" * 80)
    
    yield
    
    # Shutdown
    logger.info("=" * 80)
    logger.info("Shutting down ML service")
    logger.info("=" * 80)
    
    # Close WebSocket connections
    if manager.active_connections:
        logger.info(f"Closing {len(manager.active_connections)} WebSocket connections")
        for connection in list(manager.active_connections):
            try:
                await connection.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
    
    logger.info("Shutdown complete")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title=settings.service_name,
    version=settings.service_version,
    description="ML Service for Intelligence Foundry - MLflow, Optuna, and Training Engine",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(mlflow_server.router)
app.include_router(optuna_service.router)

# ============================================================================
# Root Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "status": "running",
        "endpoints": {
            "mlflow": "/api/mlflow",
            "optuna": "/api/optuna",
            "websocket": "/ws/intelligence-foundry",
            "health": "/health",
            "docs": "/docs",
        },
        "websocket_connections": manager.get_connection_count(),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "service": settings.service_name,
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "websocket_connections": manager.get_connection_count(),
    }
    
    # Check MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.search_experiments(max_results=1)
        health_status["mlflow"] = "healthy"
    except Exception as e:
        health_status["mlflow"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Optuna
    try:
        import optuna
        storage = optuna.storages.RDBStorage(url=settings.optuna_storage)
        health_status["optuna"] = "healthy"
    except Exception as e:
        health_status["optuna"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Return appropriate status code
    status_code = status.HTTP_200_OK if health_status["status"] == "healthy" else status.HTTP_503_SERVICE_UNAVAILABLE
    
    return JSONResponse(content=health_status, status_code=status_code)


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws/intelligence-foundry")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates
    
    Events:
    - training:metrics - Training metrics updates
    - experiment:status - Experiment status changes
    - trial:complete - Optuna trial completion
    - optimization:progress - Optimization progress updates
    - connection:open - Connection established
    - connection:close - Connection closed
    - ping/pong - Heartbeat messages
    """
    # Check connection limit
    if manager.get_connection_count() >= settings.ws_max_connections:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Max connections reached")
        return
    
    # Accept connection
    await manager.connect(websocket)
    
    try:
        # Send connection confirmation
        await manager.send_personal(websocket, {
            "type": "connection:open",
            "data": {
                "message": "Connected to Intelligence Foundry WebSocket",
                "timestamp": datetime.now().isoformat(),
            },
        })
        
        # Message loop
        while True:
            try:
                # Receive message
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle ping
                if message.get("type") == "ping":
                    await manager.send_personal(websocket, {
                        "type": "pong",
                        "data": {
                            "timestamp": datetime.now().isoformat(),
                        },
                    })
                    continue
                
                # Log other messages
                logger.debug(f"Received WebSocket message: {message.get('type')}")
                
            except json.JSONDecodeError:
                logger.error("Invalid JSON received")
                await manager.send_personal(websocket, {
                    "type": "error",
                    "data": {
                        "message": "Invalid JSON format",
                        "timestamp": datetime.now().isoformat(),
                    },
                })
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                break
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await manager.disconnect(websocket)


# ============================================================================
# Utility Endpoints
# ============================================================================

@app.get("/stats")
async def get_stats():
    """Get service statistics"""
    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "websocket_connections": manager.get_connection_count(),
        "max_websocket_connections": settings.ws_max_connections,
        "heartbeat_interval": settings.ws_heartbeat_interval,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/broadcast")
async def broadcast_message(message: dict):
    """
    Broadcast message to all WebSocket clients
    
    For testing and debugging purposes.
    """
    await manager.broadcast(message)
    return {
        "message": "Broadcast sent",
        "recipients": manager.get_connection_count(),
    }


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An unexpected error occurred",
            "timestamp": datetime.now().isoformat(),
        },
    )


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "intelligence_foundry_main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
        access_log=True,
    )
