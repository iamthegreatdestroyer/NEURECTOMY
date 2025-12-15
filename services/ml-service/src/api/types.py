"""
Neurectomy Core Type Definitions
================================

Types for IDE orchestration and agent coordination.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from datetime import datetime


# =============================================================================
# ENUMERATIONS
# =============================================================================

class AgentRole(Enum):
    """Roles in the Elite Agent Collective."""
    # Coordination
    NEXUS = auto()          # Cross-domain synthesis, master coordinator
    OMNISCIENT = auto()     # Meta-coordination, system health
    
    # Development
    APEX = auto()           # Core implementation, architecture
    ARCHITECT = auto()      # System design, patterns
    VELOCITY = auto()       # Performance optimization
    TENSOR = auto()         # ML/AI model integration
    ECLIPSE = auto()        # Testing, verification
    FLUX = auto()           # DevOps, infrastructure
    
    # Specialized
    CIPHER = auto()         # Security, cryptography
    VERTEX = auto()         # Database, storage
    PARSE = auto()          # Parsing, compilation
    SYNAPSE = auto()        # Integration, APIs
    
    # Creative
    MUSE = auto()           # Creative ideation
    SCRIBE = auto()         # Documentation
    HERALD = auto()         # Communication


class TaskStatus(Enum):
    """Status of a task in the orchestration pipeline."""
    PENDING = auto()
    QUEUED = auto()
    ASSIGNED = auto()
    IN_PROGRESS = auto()
    BLOCKED = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class ProjectType(Enum):
    """Types of projects Neurectomy can manage."""
    PYTHON = auto()
    TYPESCRIPT = auto()
    RUST = auto()
    CPP = auto()
    MIXED = auto()
    DOCUMENTATION = auto()


class ContextScope(Enum):
    """Scope of context for operations."""
    FILE = auto()           # Single file
    DIRECTORY = auto()      # Directory and contents
    PROJECT = auto()        # Entire project
    WORKSPACE = auto()      # Multiple projects
    CONVERSATION = auto()   # Conversation history


class InferenceBackend(Enum):
    """Available inference backends."""
    RYOT_LOCAL = auto()     # Local Ryot LLM
    OLLAMA = auto()         # Ollama server
    CLOUD_API = auto()      # Cloud API fallback
    MOCK = auto()           # Mock for testing


class CompressionLevel(Enum):
    """ΣLANG compression aggressiveness."""
    NONE = 0
    LIGHT = 1
    BALANCED = 2
    AGGRESSIVE = 3


# =============================================================================
# AGENT STRUCTURES
# =============================================================================

@dataclass
class AgentCapability:
    """A specific capability an agent possesses."""
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    estimated_time_ms: int = 1000
    requires_context: bool = True


@dataclass
class AgentProfile:
    """Profile describing an Elite Agent."""
    agent_id: str
    role: AgentRole
    name: str
    description: str
    capabilities: List[AgentCapability]
    
    # Performance characteristics
    avg_response_time_ms: float = 0.0
    success_rate: float = 1.0
    total_tasks_completed: int = 0
    
    # Resource requirements
    requires_inference: bool = True
    requires_storage: bool = False
    max_context_tokens: int = 4096


@dataclass
class AgentMessage:
    """Message between agents or from user to agent."""
    message_id: str
    sender: str                    # Agent ID or "USER"
    recipients: List[str]          # Agent IDs
    
    # Content
    intent: str                    # What the message is asking for
    content: str                   # Detailed content
    context_reference: Optional[str] = None  # Reference to shared context
    
    # Metadata
    priority: TaskPriority = TaskPriority.NORMAL
    timestamp: float = 0.0
    requires_response: bool = True
    
    # Threading
    conversation_id: str = ""
    parent_message_id: Optional[str] = None


@dataclass
class AgentResponse:
    """Response from an agent."""
    response_id: str
    agent_id: str
    request_message_id: str
    
    # Content
    content: str
    artifacts: List['Artifact'] = field(default_factory=list)
    
    # Status
    success: bool = True
    error_message: Optional[str] = None
    
    # Continuation
    should_continue: bool = False
    next_agent_hint: Optional[str] = None
    
    # Metrics
    processing_time_ms: float = 0.0
    tokens_used: int = 0
    compression_ratio: float = 1.0


# =============================================================================
# TASK STRUCTURES
# =============================================================================

@dataclass
class TaskDefinition:
    """Definition of a task to be executed."""
    task_id: str
    title: str
    description: str
    
    # Assignment
    assigned_agents: List[AgentRole] = field(default_factory=list)
    primary_agent: Optional[AgentRole] = None
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)  # Task IDs
    blocks: List[str] = field(default_factory=list)      # Task IDs
    
    # Context
    context_scope: ContextScope = ContextScope.PROJECT
    context_files: List[str] = field(default_factory=list)
    
    # Execution
    priority: TaskPriority = TaskPriority.NORMAL
    estimated_time_ms: int = 5000
    max_retries: int = 2
    
    # Output
    expected_artifacts: List[str] = field(default_factory=list)


@dataclass
class TaskResult:
    """Result of task execution."""
    task_id: str
    status: TaskStatus
    
    # Output
    artifacts: List['Artifact'] = field(default_factory=list)
    output_message: str = ""
    
    # Metrics
    actual_time_ms: float = 0.0
    agents_used: List[str] = field(default_factory=list)
    tokens_consumed: int = 0
    
    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    
    # Timestamps
    started_at: float = 0.0
    completed_at: float = 0.0


@dataclass
class TaskPlan:
    """A plan consisting of multiple tasks."""
    plan_id: str
    title: str
    description: str
    
    # Tasks
    tasks: List[TaskDefinition] = field(default_factory=list)
    task_order: List[str] = field(default_factory=list)  # Execution order
    
    # Execution state
    current_task_index: int = 0
    completed_tasks: List[str] = field(default_factory=list)
    failed_tasks: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: float = 0.0
    estimated_total_time_ms: int = 0


# =============================================================================
# ARTIFACT STRUCTURES
# =============================================================================

@dataclass
class Artifact:
    """An artifact produced by agent work."""
    artifact_id: str
    artifact_type: str           # "code", "document", "diagram", "test", etc.
    
    # Content
    name: str
    content: Union[str, bytes]
    
    # File info
    file_path: Optional[str] = None
    language: Optional[str] = None
    
    # Metadata
    created_by: str = ""         # Agent ID
    created_at: float = 0.0
    
    # Versioning
    version: int = 1
    parent_artifact_id: Optional[str] = None


@dataclass
class CodeArtifact(Artifact):
    """Specialized artifact for code."""
    language: str = "python"
    imports: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    
    # Quality
    has_tests: bool = False
    test_artifact_id: Optional[str] = None
    lint_passed: bool = False
    type_check_passed: bool = False


# =============================================================================
# CONTEXT STRUCTURES
# =============================================================================

@dataclass
class ContextWindow:
    """Represents the current context available to agents."""
    context_id: str
    scope: ContextScope
    
    # Content
    files: Dict[str, str] = field(default_factory=dict)  # path -> content
    conversation_history: List[AgentMessage] = field(default_factory=list)
    
    # Compression
    is_compressed: bool = False
    compression_ratio: float = 1.0
    sigma_encoded: Optional[bytes] = None
    
    # Metrics
    total_tokens: int = 0
    effective_tokens: int = 0    # After compression


@dataclass
class ProjectContext:
    """Context about the current project."""
    project_id: str
    project_name: str
    project_type: ProjectType
    project_path: str
    
    # Structure
    source_directories: List[str] = field(default_factory=list)
    test_directories: List[str] = field(default_factory=list)
    config_files: List[str] = field(default_factory=list)
    
    # Dependencies
    dependencies: Dict[str, str] = field(default_factory=dict)
    dev_dependencies: Dict[str, str] = field(default_factory=dict)
    
    # Git
    git_remote: Optional[str] = None
    current_branch: str = "main"
    has_uncommitted_changes: bool = False
    
    # Storage
    is_vault_protected: bool = False
    vault_path: Optional[str] = None


# =============================================================================
# CONFIGURATION STRUCTURES
# =============================================================================

@dataclass
class InferenceConfig:
    """Configuration for inference backend."""
    backend: InferenceBackend = InferenceBackend.RYOT_LOCAL
    
    # Ryot-specific
    ryot_endpoint: str = "http://localhost:8000"
    ryot_model: str = "bitnet-7b"
    
    # Generation
    max_tokens: int = 1024
    temperature: float = 0.7
    
    # Compression
    use_sigma_compression: bool = True
    compression_level: CompressionLevel = CompressionLevel.BALANCED
    
    # RSU
    enable_rsu_caching: bool = True
    rsu_storage_path: Optional[str] = None


@dataclass
class StorageConfig:
    """Configuration for ΣVAULT storage."""
    enabled: bool = True
    vault_path: str = "~/.neurectomy/vault"
    
    # Security
    device_binding: bool = True
    auto_lock_minutes: int = 30
    
    # Tiers
    hot_cache_mb: int = 512
    warm_cache_mb: int = 2048


@dataclass
class NeurectomyConfig:
    """Complete Neurectomy configuration."""
    # Core
    workspace_path: str = "~/.neurectomy"
    
    # Components
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    
    # Agents
    enabled_agents: List[AgentRole] = field(default_factory=lambda: list(AgentRole))
    max_concurrent_agents: int = 3
    
    # UI
    auto_save: bool = True
    theme: str = "dark"


# =============================================================================
# ORCHESTRATION STRUCTURES
# =============================================================================

@dataclass
class OrchestratorState:
    """Current state of the orchestrator."""
    # Status
    is_running: bool = False
    current_plan: Optional[TaskPlan] = None
    active_tasks: List[str] = field(default_factory=list)
    
    # Agents
    available_agents: List[str] = field(default_factory=list)
    busy_agents: Dict[str, str] = field(default_factory=dict)  # agent_id -> task_id
    
    # Context
    current_context: Optional[ContextWindow] = None
    
    # Metrics
    tasks_completed_today: int = 0
    tokens_used_today: int = 0
    compression_savings_bytes: int = 0


@dataclass
class OrchestratorStatistics:
    """Statistics about orchestrator performance."""
    # Tasks
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    average_task_time_ms: float = 0.0
    
    # Agents
    agent_utilization: Dict[str, float] = field(default_factory=dict)
    agent_success_rates: Dict[str, float] = field(default_factory=dict)
    
    # Resources
    total_tokens_used: int = 0
    total_compression_savings: int = 0
    average_compression_ratio: float = 1.0
    
    # Uptime
    started_at: float = 0.0
    uptime_seconds: float = 0.0
