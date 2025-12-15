"""
API Request/Response Types
==========================

Pydantic models for API validation.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class TaskType(str, Enum):
    """Supported task types."""
    GENERATE = "generate"
    SUMMARIZE = "summarize"
    ANALYZE = "analyze"
    CODE = "code"
    TRANSLATE = "translate"
    CUSTOM = "custom"


class GenerationRequest(BaseModel):
    """Request for text generation."""
    
    prompt: str = Field(..., min_length=1, max_length=100000)
    max_tokens: int = Field(default=256, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1, le=1000)
    
    # Compression settings
    use_compression: bool = Field(default=True)
    compression_level: int = Field(default=2, ge=1, le=3)
    
    # RSU settings
    use_rsu: bool = Field(default=True)
    conversation_id: Optional[str] = None
    
    # Agent routing
    preferred_agent: Optional[str] = None
    required_capabilities: List[str] = Field(default_factory=list)
    
    # Streaming
    stream: bool = Field(default=False)
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "Explain quantum computing in simple terms",
                "max_tokens": 500,
                "temperature": 0.7,
            }
        }


class GenerationResponse(BaseModel):
    """Response from text generation."""
    
    id: str
    created_at: datetime
    
    # Output
    text: str
    finish_reason: str  # "stop", "length", "error"
    
    # Usage
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    
    # Performance
    latency_ms: float
    compression_ratio: float = 1.0
    
    # Metadata
    model: str = "neurectomy-v1"
    agent_id: Optional[str] = None
    rsu_reference: Optional[str] = None


class AgentTaskRequest(BaseModel):
    """Request for agent-specific task."""
    
    task_type: TaskType
    payload: Dict[str, Any]
    
    # Routing
    agent_id: Optional[str] = None
    team: Optional[str] = None
    
    # Settings
    timeout_seconds: float = Field(default=60.0, ge=1.0, le=300.0)
    priority: int = Field(default=3, ge=1, le=5)
    
    # Context
    conversation_id: Optional[str] = None


class AgentTaskResponse(BaseModel):
    """Response from agent task."""
    
    task_id: str
    status: str  # "completed", "failed", "timeout"
    
    # Output
    result: Optional[Any] = None
    error: Optional[str] = None
    
    # Execution info
    agent_id: str
    team: str
    latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: datetime
    
    components: Dict[str, Dict[str, Any]]
    
    # System info
    uptime_seconds: float
    version: str


class MetricsResponse(BaseModel):
    """Metrics dashboard response."""
    
    timestamp: datetime
    
    inference: Dict[str, Any]
    compression: Dict[str, Any]
    cache: Dict[str, Any]
    agents: Dict[str, Any]
    storage: Dict[str, Any]


class AgentListResponse(BaseModel):
    """List of available agents."""
    
    total: int
    teams: Dict[str, List[str]]
    agents: List[Dict[str, Any]]


class ConversationRequest(BaseModel):
    """Request with conversation context."""
    
    conversation_id: str
    messages: List[Dict[str, str]]
    
    # Generation settings
    max_tokens: int = Field(default=256)
    temperature: float = Field(default=0.7)


class RSUSearchRequest(BaseModel):
    """Search for similar RSUs."""
    
    query: str
    semantic_hash: Optional[int] = None
    threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    max_results: int = Field(default=10, ge=1, le=100)


class RSUSearchResponse(BaseModel):
    """RSU search results."""
    
    results: List[Dict[str, Any]]
    total: int
    query_hash: int
