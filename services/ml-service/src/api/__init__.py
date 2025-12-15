"""
ML Service API Layer & Neurectomy Phase 0 Interface Contracts

@SYNAPSE @APEX - FastAPI routes for ML services.

Exposes endpoints for:
- LLM inference and chat
- Embedding generation and search
- Training orchestration
- Agent intelligence operations

Phase 0 Interfaces:
- Protocols for orchestration, context management, inference, storage
- Type definitions for agents, tasks, artifacts, configuration
- Custom exception hierarchy
"""

from src.api.routes import router

# Phase 0 Interface Contracts
from .interfaces import (
    AgentProtocol,
    OrchestratorProtocol,
    ContextManagerProtocol,
    InferenceBridge,
    StorageBridge,
    ProjectManager,
    AgentCollective,
    NeurectomyFactory,
)

from .types import (
    # Enums
    AgentRole,
    TaskStatus,
    TaskPriority,
    ProjectType,
    ContextScope,
    InferenceBackend,
    CompressionLevel,
    # Agent structures
    AgentCapability,
    AgentProfile,
    AgentMessage,
    AgentResponse,
    # Task structures
    TaskDefinition,
    TaskResult,
    TaskPlan,
    # Artifact structures
    Artifact,
    CodeArtifact,
    # Context structures
    ContextWindow,
    ProjectContext,
    # Configuration
    InferenceConfig,
    StorageConfig,
    NeurectomyConfig,
    # Orchestration
    OrchestratorState,
    OrchestratorStatistics,
)

from .exceptions import (
    NeurectomyError,
    AgentNotFoundError,
    AgentBusyError,
    TaskExecutionError,
    PlanExecutionError,
    ContextBuildError,
    InferenceError,
    StorageError,
    ProjectError,
    CompressionError,
)

__all__ = [
    # FastAPI router
    "router",
    # Protocols
    "AgentProtocol",
    "OrchestratorProtocol",
    "ContextManagerProtocol",
    "InferenceBridge",
    "StorageBridge",
    "ProjectManager",
    "AgentCollective",
    "NeurectomyFactory",
    # Enums
    "AgentRole",
    "TaskStatus",
    "TaskPriority",
    "ProjectType",
    "ContextScope",
    "InferenceBackend",
    "CompressionLevel",
    # Agent types
    "AgentCapability",
    "AgentProfile",
    "AgentMessage",
    "AgentResponse",
    # Task types
    "TaskDefinition",
    "TaskResult",
    "TaskPlan",
    # Artifact types
    "Artifact",
    "CodeArtifact",
    # Context types
    "ContextWindow",
    "ProjectContext",
    # Configuration
    "InferenceConfig",
    "StorageConfig",
    "NeurectomyConfig",
    # Orchestration
    "OrchestratorState",
    "OrchestratorStatistics",
    # Exceptions
    "NeurectomyError",
    "AgentNotFoundError",
    "AgentBusyError",
    "TaskExecutionError",
    "PlanExecutionError",
    "ContextBuildError",
    "InferenceError",
    "StorageError",
    "ProjectError",
    "CompressionError",
]

__version__ = "0.1.0"
