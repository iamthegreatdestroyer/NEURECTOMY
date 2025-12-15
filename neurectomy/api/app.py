"""
Neurectomy REST API
===================

FastAPI application exposing Neurectomy capabilities.
"""

from typing import Optional, AsyncGenerator
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import json
import uuid
from datetime import datetime, timezone

from .types import (
    GenerationRequest, GenerationResponse,
    AgentTaskRequest, AgentTaskResponse,
    HealthResponse, MetricsResponse,
    AgentListResponse, ConversationRequest,
    RSUSearchRequest, RSUSearchResponse,
)
from ..core import NeurectomyOrchestrator, TaskRequest, TaskStatus
from ..core.health import create_default_health_checker, HealthStatus
from ..core.monitoring import get_monitor
from ..core.logging import get_logger
from ..elite import EliteCollective


# Global instances
_orchestrator: Optional[NeurectomyOrchestrator] = None
_collective: Optional[EliteCollective] = None
_health_checker = None
_logger = get_logger("api", component="fastapi")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _orchestrator, _collective, _health_checker
    
    _logger.info("Starting Neurectomy API...")
    
    # Initialize components
    _orchestrator = NeurectomyOrchestrator()
    _collective = EliteCollective()
    _health_checker = create_default_health_checker()
    
    _logger.info("Neurectomy API started")
    
    yield
    
    _logger.info("Shutting down Neurectomy API...")


# Create FastAPI app
app = FastAPI(
    title="Neurectomy API",
    description="Unified AI Development Ecosystem API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency injection
def get_orchestrator() -> NeurectomyOrchestrator:
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return _orchestrator


def get_collective() -> EliteCollective:
    if _collective is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return _collective


# =============================================================================
# GENERATION ENDPOINTS
# =============================================================================

@app.post("/v1/generate", response_model=GenerationResponse)
async def generate(
    request: GenerationRequest,
    orchestrator: NeurectomyOrchestrator = Depends(get_orchestrator),
):
    """Generate text from a prompt."""
    start_time = datetime.now(timezone.utc)
    
    _logger.info(
        f"Generation request: {len(request.prompt)} chars",
        task_id=str(uuid.uuid4())[:8],
    )
    
    try:
        result = orchestrator.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            conversation_id=request.conversation_id,
        )
        
        latency_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000
        
        return GenerationResponse(
            id=result.task_id,
            created_at=datetime.now(timezone.utc),
            text=result.generated_text or "",
            finish_reason=(
                "stop" if result.status == TaskStatus.COMPLETED else "error"
            ),
            prompt_tokens=result.tokens_processed,
            completion_tokens=result.tokens_generated,
            total_tokens=result.tokens_processed + result.tokens_generated,
            latency_ms=latency_ms,
            compression_ratio=result.compression_ratio,
            agent_id=result.executing_agent,
            rsu_reference=result.rsu_reference,
        )
        
    except Exception as e:
        _logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/generate/stream")
async def generate_stream(
    request: GenerationRequest,
    orchestrator: NeurectomyOrchestrator = Depends(get_orchestrator),
):
    """Stream text generation."""
    
    async def stream_generator() -> AsyncGenerator[str, None]:
        try:
            for chunk in orchestrator.stream_generate(
                request.prompt,
                request.max_tokens,
                request.temperature,
            ):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            
            yield f"data: {json.dumps({'done': True})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
    )


# =============================================================================
# AGENT ENDPOINTS
# =============================================================================

@app.post("/v1/agents/task", response_model=AgentTaskResponse)
async def execute_agent_task(
    request: AgentTaskRequest,
    collective: EliteCollective = Depends(get_collective),
):
    """Execute a task using the Elite Collective."""
    start_time = datetime.now(timezone.utc)
    
    task = TaskRequest(
        task_id=f"api_{uuid.uuid4().hex[:8]}",
        task_type=request.task_type.value,
        payload=request.payload,
        conversation_id=request.conversation_id,
        timeout_seconds=request.timeout_seconds,
    )
    
    if request.agent_id:
        task.preferred_agent = request.agent_id
    
    try:
        result = collective.execute(task)
        
        latency_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000
        
        return AgentTaskResponse(
            task_id=result.task_id,
            status=result.status.name.lower(),
            result=result.output,
            error=result.error_message,
            agent_id=result.executing_agent or "unknown",
            team=request.team or "auto",
            latency_ms=latency_ms,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/agents", response_model=AgentListResponse)
async def list_agents(
    collective: EliteCollective = Depends(get_collective),
):
    """List all available agents."""
    agents = []
    teams = {}
    
    for agent_id in collective.list_agents():
        agent = collective.get_agent(agent_id)
        if agent:
            team = agent.team_id
            
            if team not in teams:
                teams[team] = []
            teams[team].append(agent_id)
            
            agents.append({
                "id": agent_id,
                "name": agent.config.agent_name,
                "type": agent.config.agent_type,
                "team": team,
                "capabilities": [c.name for c in agent.config.capabilities],
            })
    
    return AgentListResponse(
        total=len(agents),
        teams=teams,
        agents=agents,
    )


@app.get("/v1/agents/{agent_id}")
async def get_agent(
    agent_id: str,
    collective: EliteCollective = Depends(get_collective),
):
    """Get details for a specific agent."""
    agent = collective.get_agent(agent_id)
    
    if agent is None:
        raise HTTPException(
            status_code=404,
            detail=f"Agent not found: {agent_id}"
        )
    
    return {
        "id": agent_id,
        "name": agent.config.agent_name,
        "type": agent.config.agent_type,
        "team": agent.team_id,
        "role": agent.role.name,
        "capabilities": [c.name for c in agent.config.capabilities],
        "state": {
            "is_active": agent.state.is_active,
            "is_busy": agent.state.is_busy,
            "tasks_completed": agent.state.tasks_completed,
        },
    }


# =============================================================================
# HEALTH & MONITORING ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check."""
    health = _health_checker.check_all()
    monitor = get_monitor()
    
    return HealthResponse(
        status=health.status.value,
        timestamp=health.timestamp,
        components={c.name: c.to_dict() for c in health.components},
        uptime_seconds=monitor.get_dashboard()["uptime_seconds"],
        version="1.0.0",
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get performance metrics."""
    monitor = get_monitor()
    dashboard = monitor.get_dashboard()
    
    return MetricsResponse(
        timestamp=datetime.now(timezone.utc),
        **dashboard,
    )


@app.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe."""
    health = _health_checker.check_all()
    
    if health.status == HealthStatus.UNHEALTHY:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {"status": "ready"}


@app.get("/live")
async def liveness_check():
    """Kubernetes liveness probe."""
    return {"status": "alive"}


# =============================================================================
# CONVERSATION ENDPOINTS
# =============================================================================

@app.post("/v1/conversations/{conversation_id}/message")
async def conversation_message(
    conversation_id: str,
    request: ConversationRequest,
    orchestrator: NeurectomyOrchestrator = Depends(get_orchestrator),
):
    """Send message in a conversation context."""
    # Build prompt from conversation history
    prompt_parts = []
    for msg in request.messages[-10:]:  # Last 10 messages
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prompt_parts.append(f"{role.capitalize()}: {content}")
    
    prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"
    
    result = orchestrator.generate(
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        conversation_id=conversation_id,
    )
    
    return {
        "conversation_id": conversation_id,
        "message": result.generated_text,
        "rsu_reference": result.rsu_reference,
    }


# =============================================================================
# RSU ENDPOINTS
# =============================================================================

@app.post("/v1/rsu/search", response_model=RSUSearchResponse)
async def search_rsu(
    request: RSUSearchRequest,
    orchestrator: NeurectomyOrchestrator = Depends(get_orchestrator),
):
    """Search for similar RSUs."""
    # This would integrate with ΣVAULT storage
    # Placeholder implementation
    
    import hashlib
    query_hash = int.from_bytes(
        hashlib.sha256(request.query.encode()).digest()[:8],
        'little'
    )
    
    return RSUSearchResponse(
        results=[],  # Would come from ΣVAULT
        total=0,
        query_hash=query_hash,
    )


# =============================================================================
# ADMIN ENDPOINTS
# =============================================================================

@app.get("/v1/admin/stats")
async def admin_stats(
    orchestrator: NeurectomyOrchestrator = Depends(get_orchestrator),
    collective: EliteCollective = Depends(get_collective),
):
    """Get admin statistics."""
    return {
        "orchestrator": orchestrator.get_state().__dict__,
        "collective": collective.get_stats().__dict__,
        "monitor": get_monitor().get_dashboard(),
    }


# Entry point for running directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
