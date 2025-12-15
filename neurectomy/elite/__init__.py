"""
Neurectomy Elite Agent Collective
=================================

40 specialized AI agents organized into 5 functional teams.
"""

from .collective import (
    EliteCollective,
    CollectiveConfig,
    CollectiveStats,
    create_elite_collective,
    get_all_agent_ids,
)
from .teams import (
    EliteAgent,
    TeamCommander,
    TeamConfig,
    TeamRole,
    create_inference_team,
    create_compression_team,
    create_storage_team,
    create_analysis_team,
    create_synthesis_team,
)

__all__ = [
    # Collective
    "EliteCollective",
    "CollectiveConfig",
    "CollectiveStats",
    "create_elite_collective",
    "get_all_agent_ids",
    # Team base
    "EliteAgent",
    "TeamCommander",
    "TeamConfig",
    "TeamRole",
    # Team factories
    "create_inference_team",
    "create_compression_team",
    "create_storage_team",
    "create_analysis_team",
    "create_synthesis_team",
]
