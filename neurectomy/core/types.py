"""
Neurectomy Core Types
=====================

Types for orchestration and agent management.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum, auto
from datetime import datetime


class TaskPriority(Enum):
    """Task execution priority."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = auto()
    QUEUED = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class AgentCapability(Enum):
    """Agent capability types."""
    INFERENCE = auto()
    COMPRESSION = auto()
    STORAGE = auto()
    ANALYSIS = auto()
    SYNTHESIS = auto()
    TRANSLATION = auto()
    SUMMARIZATION = auto()
    CODE_GENERATION = auto()
    REASONING = auto()
    PLANNING = auto()


@dataclass
class TaskRequest:
    """Request for task execution."""
    
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    
    # Routing hints
    required_capabilities: List[AgentCapability] = field(default_factory=list)
    preferred_agent: Optional[str] = None
    
    # Context
    conversation_id: Optional[str] = None
    parent_task_id: Optional[str] = None
    
    # Constraints
    timeout_seconds: float = 60.0
    max_tokens: int = 4096
    use_compression: bool = True
    use_rsu: bool = True
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result of task execution."""
    
    task_id: str
    status: TaskStatus
    
    # Output
    output: Optional[Any] = None
    generated_text: Optional[str] = None
    
    # Metrics
    execution_time_ms: float = 0.0
    tokens_processed: int = 0
    tokens_generated: int = 0
    compression_ratio: float = 1.0
    
    # RSU info
    rsu_reference: Optional[str] = None
    cache_hit: bool = False
    
    # Error handling
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    
    # Agent info
    executing_agent: Optional[str] = None
    
    # Metadata
    completed_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AgentState:
    """Current state of an agent."""
    
    agent_id: str
    agent_type: str
    capabilities: List[AgentCapability]
    
    # Status
    is_active: bool = True
    is_busy: bool = False
    current_task_id: Optional[str] = None
    
    # Statistics
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_tokens_processed: int = 0
    average_latency_ms: float = 0.0
    
    # Resources
    memory_usage_mb: float = 0.0
    last_active: datetime = field(default_factory=datetime.utcnow)


@dataclass
class OrchestratorState:
    """Current state of the orchestrator."""
    
    # Components
    inference_ready: bool = False
    compression_ready: bool = False
    storage_ready: bool = False
    
    # Agents
    active_agents: int = 0
    total_agents: int = 0
    
    # Tasks
    pending_tasks: int = 0
    running_tasks: int = 0
    completed_tasks: int = 0
    
    # Performance
    total_tokens_processed: int = 0
    average_compression_ratio: float = 1.0
    cache_hit_rate: float = 0.0
    
    # System
    uptime_seconds: float = 0.0
    memory_usage_mb: float = 0.0
