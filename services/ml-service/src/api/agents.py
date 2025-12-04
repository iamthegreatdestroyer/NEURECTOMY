"""
Agent Intelligence API Routes.

@NEURAL @OMNISCIENT - Agent management, memory, and intelligence endpoints.
"""

from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from src.models.agents import (
    AgentConfig,
    AgentState,
    AgentMemory,
    AgentAction,
    AgentFeedback,
    AgentType,
    ActionType,
    MemoryType,
    BehaviorModel,
)
from src.services.agent_intelligence import AgentIntelligenceService

router = APIRouter()

# Service dependency
_agent_service: Optional[AgentIntelligenceService] = None


async def get_agent_service() -> AgentIntelligenceService:
    """Dependency to get agent intelligence service."""
    global _agent_service
    if _agent_service is None:
        _agent_service = AgentIntelligenceService()
        await _agent_service.initialize()
    return _agent_service


# =============================================================================
# Request/Response Models
# =============================================================================

class CreateAgentRequest(BaseModel):
    """Request to create an agent."""
    name: str
    agent_type: AgentType = AgentType.CONVERSATIONAL
    system_prompt: str = "You are a helpful AI assistant."
    description: str = ""
    tools: list[str] = Field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 2048
    learning_enabled: bool = True


class AgentResponse(BaseModel):
    """Agent response."""
    agent: AgentConfig


class AgentListResponse(BaseModel):
    """List of agents."""
    agents: list[AgentConfig]
    total: int


class ThinkRequest(BaseModel):
    """Request for agent thinking."""
    input: str
    max_steps: int = 5


class ThinkResponse(BaseModel):
    """Thinking response."""
    actions: list[AgentAction]
    final_response: Optional[str] = None


class ChatRequest(BaseModel):
    """Simple chat request."""
    message: str


class ChatResponse(BaseModel):
    """Chat response."""
    response: str
    action: AgentAction


class MemoryStoreRequest(BaseModel):
    """Request to store memory."""
    content: str
    memory_type: MemoryType = MemoryType.SHORT_TERM
    importance: float = 0.5
    metadata: dict = Field(default_factory=dict)


class MemoryResponse(BaseModel):
    """Memory response."""
    id: str
    stored: bool


class MemoryRetrieveRequest(BaseModel):
    """Request to retrieve memories."""
    query: str
    memory_types: Optional[list[MemoryType]] = None
    limit: int = 10


class MemoriesResponse(BaseModel):
    """Multiple memories response."""
    memories: list[dict]


class FeedbackRequest(BaseModel):
    """User feedback request."""
    action_id: str
    rating: int = Field(ge=1, le=5)
    feedback_text: Optional[str] = None
    expected_output: Optional[str] = None


class EvolutionResponse(BaseModel):
    """Evolution metrics response."""
    metrics: dict


# =============================================================================
# Agent CRUD Endpoints
# =============================================================================

@router.post("", response_model=AgentResponse)
async def create_agent(
    request: CreateAgentRequest,
    service: AgentIntelligenceService = Depends(get_agent_service),
) -> AgentResponse:
    """
    Create a new AI agent.
    
    @NEURAL - Creates agent with behavior model and memory systems.
    
    Example:
        ```json
        {
            "name": "ResearchAssistant",
            "agent_type": "researcher",
            "system_prompt": "You are a research assistant specializing in AI.",
            "tools": ["web_search", "document_reader"],
            "learning_enabled": true
        }
        ```
    """
    config = AgentConfig(
        name=request.name,
        agent_type=request.agent_type,
        system_prompt=request.system_prompt,
        description=request.description,
        tools=request.tools,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        learning_enabled=request.learning_enabled,
    )
    
    agent = await service.create_agent(config)
    return AgentResponse(agent=agent)


@router.get("", response_model=AgentListResponse)
async def list_agents(
    agent_type: Optional[AgentType] = None,
    limit: int = 50,
    offset: int = 0,
    service: AgentIntelligenceService = Depends(get_agent_service),
) -> AgentListResponse:
    """
    List all agents.
    
    Optionally filter by agent type.
    """
    # Get all agents from service
    agents = list(service._agents.values())
    
    if agent_type:
        agents = [a for a in agents if a.agent_type == agent_type]
    
    total = len(agents)
    agents = agents[offset:offset + limit]
    
    return AgentListResponse(agents=agents, total=total)


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str,
    service: AgentIntelligenceService = Depends(get_agent_service),
) -> AgentResponse:
    """
    Get agent by ID.
    """
    agent = await service.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return AgentResponse(agent=agent)


@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: str,
    updates: dict,
    service: AgentIntelligenceService = Depends(get_agent_service),
) -> AgentResponse:
    """
    Update agent configuration.
    """
    agent = await service.update_agent(agent_id, updates)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return AgentResponse(agent=agent)


@router.delete("/{agent_id}")
async def delete_agent(
    agent_id: str,
    service: AgentIntelligenceService = Depends(get_agent_service),
) -> dict:
    """
    Delete an agent.
    """
    success = await service.delete_agent(agent_id)
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"deleted": True, "agent_id": agent_id}


# =============================================================================
# Agent Interaction Endpoints
# =============================================================================

@router.post("/{agent_id}/chat", response_model=ChatResponse)
async def chat_with_agent(
    agent_id: str,
    request: ChatRequest,
    service: AgentIntelligenceService = Depends(get_agent_service),
) -> ChatResponse:
    """
    Chat with an agent.
    
    @NEURAL @LINGUA - Simple chat interface with memory integration.
    """
    agent = await service.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Store user message as memory
    await service.store_memory(
        agent_id,
        f"User: {request.message}",
        MemoryType.SHORT_TERM,
        importance=0.5,
    )
    
    # Execute speak action
    action = await service.execute_action(
        agent_id,
        ActionType.SPEAK,
        request.message,
    )
    
    # Store response as memory
    if action.result:
        await service.store_memory(
            agent_id,
            f"Assistant: {action.result}",
            MemoryType.SHORT_TERM,
            importance=0.6,
        )
    
    return ChatResponse(
        response=action.result or "",
        action=action,
    )


@router.post("/{agent_id}/think", response_model=ThinkResponse)
async def agent_think(
    agent_id: str,
    request: ThinkRequest,
    service: AgentIntelligenceService = Depends(get_agent_service),
) -> ThinkResponse:
    """
    Trigger agent reasoning process.
    
    @NEURAL - Multi-step chain-of-thought reasoning.
    
    The agent will:
    1. Retrieve relevant memories
    2. Reason through the problem
    3. Take actions as needed
    4. Generate a response
    """
    agent = await service.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    actions = await service.think(
        agent_id,
        request.input,
        request.max_steps,
    )
    
    # Get final response if available
    final_response = None
    for action in reversed(actions):
        if action.action_type == ActionType.SPEAK and action.success:
            final_response = action.result
            break
    
    return ThinkResponse(
        actions=actions,
        final_response=final_response,
    )


@router.post("/{agent_id}/action")
async def execute_agent_action(
    agent_id: str,
    action_type: ActionType,
    content: str,
    tool_name: Optional[str] = None,
    tool_args: Optional[dict] = None,
    service: AgentIntelligenceService = Depends(get_agent_service),
) -> AgentAction:
    """
    Execute a specific agent action.
    
    @NEURAL - Direct action execution.
    """
    agent = await service.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return await service.execute_action(
        agent_id,
        action_type,
        content,
        tool_name,
        tool_args,
    )


# =============================================================================
# Memory Endpoints
# =============================================================================

@router.post("/{agent_id}/memory", response_model=MemoryResponse)
async def store_memory(
    agent_id: str,
    request: MemoryStoreRequest,
    service: AgentIntelligenceService = Depends(get_agent_service),
) -> MemoryResponse:
    """
    Store a memory for an agent.
    
    @NEURAL @VERTEX - Hierarchical memory storage.
    """
    memory_id = await service.store_memory(
        agent_id,
        request.content,
        request.memory_type,
        request.importance,
        request.metadata,
    )
    return MemoryResponse(id=memory_id, stored=True)


@router.post("/{agent_id}/memory/retrieve", response_model=MemoriesResponse)
async def retrieve_memories(
    agent_id: str,
    request: MemoryRetrieveRequest,
    service: AgentIntelligenceService = Depends(get_agent_service),
) -> MemoriesResponse:
    """
    Retrieve memories for an agent.
    
    @NEURAL @VERTEX - Semantic memory retrieval.
    """
    memories = await service.retrieve_memories(
        agent_id,
        request.query,
        request.memory_types,
        request.limit,
    )
    
    return MemoriesResponse(
        memories=[m.model_dump() for m in memories]
    )


@router.get("/{agent_id}/memory", response_model=dict)
async def get_memory_state(
    agent_id: str,
    service: AgentIntelligenceService = Depends(get_agent_service),
) -> dict:
    """
    Get agent's memory state.
    """
    memory = await service.get_memory(agent_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Agent not found")
    return memory.model_dump()


@router.post("/{agent_id}/memory/summarize")
async def summarize_context(
    agent_id: str,
    service: AgentIntelligenceService = Depends(get_agent_service),
) -> dict:
    """
    Summarize agent's current context.
    
    @NEURAL - Compresses working memory.
    """
    summary = await service.summarize_context(agent_id)
    return {"summary": summary}


# =============================================================================
# Learning & Evolution Endpoints
# =============================================================================

@router.post("/{agent_id}/feedback")
async def submit_feedback(
    agent_id: str,
    request: FeedbackRequest,
    service: AgentIntelligenceService = Depends(get_agent_service),
) -> dict:
    """
    Submit feedback for agent learning.
    
    @NEURAL @OMNISCIENT - Feedback for reinforcement learning.
    """
    agent = await service.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    feedback = AgentFeedback(
        agent_id=agent_id,
        action_id=request.action_id,
        rating=request.rating,
        feedback_text=request.feedback_text,
        expected_output=request.expected_output,
    )
    
    await service.record_feedback(feedback)
    return {"recorded": True, "agent_id": agent_id}


@router.post("/{agent_id}/evolve", response_model=EvolutionResponse)
async def trigger_evolution(
    agent_id: str,
    service: AgentIntelligenceService = Depends(get_agent_service),
) -> EvolutionResponse:
    """
    Trigger agent evolution.
    
    @OMNISCIENT @GENESIS - Evolve agent based on accumulated feedback.
    """
    agent = await service.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    metrics = await service.evolve_agent(agent_id)
    return EvolutionResponse(metrics=metrics)


@router.get("/{agent_id}/behavior")
async def get_behavior_model(
    agent_id: str,
    service: AgentIntelligenceService = Depends(get_agent_service),
) -> dict:
    """
    Get agent's behavior model.
    
    @NEURAL - Returns learned patterns and metrics.
    """
    behavior = await service.get_behavior_model(agent_id)
    if not behavior:
        raise HTTPException(status_code=404, detail="Agent not found or no behavior model")
    return behavior.model_dump()


# =============================================================================
# State Endpoints
# =============================================================================

@router.get("/{agent_id}/state")
async def get_agent_state(
    agent_id: str,
    service: AgentIntelligenceService = Depends(get_agent_service),
) -> dict:
    """
    Get agent's current state.
    """
    state = await service.get_state(agent_id)
    if not state:
        raise HTTPException(status_code=404, detail="Agent not found")
    return state.model_dump()
