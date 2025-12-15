"""
Neurectomy - AI Development Ecosystem
=====================================

Unified orchestration of:
- Ryot LLM (CPU-native inference)
- ΣLANG (semantic compression)
- ΣVAULT (8D encrypted storage)
- Elite Agent Collective (40 specialized agents)
"""

from .core import (
    NeurectomyOrchestrator,
    OrchestratorConfig,
    TaskRequest,
    TaskResult,
    TaskStatus,
)

from .elite import (
    EliteCollective,
    CollectiveConfig,
    CollectiveStats,
    create_elite_collective,
    get_all_agent_ids,
    EliteAgent,
    TeamCommander,
    TeamConfig,
    TeamRole,
)

__all__ = [
    # Core
    "NeurectomyOrchestrator",
    "OrchestratorConfig",
    "TaskRequest",
    "TaskResult",
    "TaskStatus",
    # Elite Collective
    "EliteCollective",
    "CollectiveConfig",
    "CollectiveStats",
    "create_elite_collective",
    "get_all_agent_ids",
    "EliteAgent",
    "TeamCommander",
    "TeamConfig",
    "TeamRole",
]

__version__ = "0.1.0"
