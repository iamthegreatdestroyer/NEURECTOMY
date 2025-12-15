"""
Elite Agent Collective
======================

Unified management of all 40 Elite Agents.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .teams import (
    EliteAgent, TeamCommander,
    create_inference_team,
    create_compression_team,
    create_storage_team,
    create_analysis_team,
    create_synthesis_team,
)
from ..agents.registry import AgentRegistry
from ..core.types import TaskRequest, TaskResult, TaskStatus, AgentCapability


@dataclass
class CollectiveConfig:
    """Configuration for the Elite Collective."""
    
    # Teams to activate
    enable_inference_team: bool = True
    enable_compression_team: bool = True
    enable_storage_team: bool = True
    enable_analysis_team: bool = True
    enable_synthesis_team: bool = True
    
    # Coordination
    enable_cross_team_collaboration: bool = True
    max_delegation_depth: int = 3
    
    # Performance
    parallel_team_execution: bool = True
    task_timeout_seconds: float = 60.0


@dataclass
class CollectiveStats:
    """Statistics for the collective."""
    
    total_agents: int = 0
    active_agents: int = 0
    total_tasks_completed: int = 0
    total_tokens_processed: int = 0
    average_task_time_ms: float = 0.0
    team_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class EliteCollective:
    """
    The Elite Agent Collective.
    
    Manages 40 specialized AI agents organized into 5 teams:
    - Inference Team (8 agents)
    - Compression Team (8 agents)
    - Storage Team (8 agents)
    - Analysis Team (8 agents)
    - Synthesis Team (8 agents)
    """
    
    def __init__(self, config: Optional[CollectiveConfig] = None):
        self.config = config or CollectiveConfig()
        
        # Agent registry
        self._registry = AgentRegistry()
        
        # Team commanders
        self._commanders: Dict[str, TeamCommander] = {}
        
        # All agents
        self._agents: Dict[str, EliteAgent] = {}
        
        # Statistics (initialize before _init_teams)
        self._stats = CollectiveStats()
        self._start_time = datetime.now(timezone.utc)
        
        # Initialize teams
        self._init_teams()
    
    def _init_teams(self) -> None:
        """Initialize all teams."""
        teams = []
        
        if self.config.enable_inference_team:
            teams.extend(self._init_team("inference", create_inference_team))
        
        if self.config.enable_compression_team:
            teams.extend(self._init_team("compression", create_compression_team))
        
        if self.config.enable_storage_team:
            teams.extend(self._init_team("storage", create_storage_team))
        
        if self.config.enable_analysis_team:
            teams.extend(self._init_team("analysis", create_analysis_team))
        
        if self.config.enable_synthesis_team:
            teams.extend(self._init_team("synthesis", create_synthesis_team))
        
        # Register all agents
        for agent in teams:
            self._agents[agent.agent_id] = agent
            self._registry.register(type(agent), agent.config)
        
        # Enable cross-team collaboration
        if self.config.enable_cross_team_collaboration:
            self._setup_collaboration()
        
        # Update stats
        self._stats.total_agents = len(self._agents)
        self._stats.active_agents = len(self._agents)
    
    def _init_team(self, team_name: str, factory) -> List[EliteAgent]:
        """Initialize a single team."""
        agents = factory()
        
        # First agent is commander
        if agents and hasattr(agents[0], 'team_config'):
            self._commanders[team_name] = agents[0]
        
        return agents
    
    def _setup_collaboration(self) -> None:
        """Enable cross-team collaboration."""
        commanders = list(self._commanders.values())
        
        # Each commander can collaborate with others
        for cmd in commanders:
            for other_cmd in commanders:
                if cmd.agent_id != other_cmd.agent_id:
                    cmd.add_collaborator(other_cmd)
    
    def execute(self, request: TaskRequest) -> TaskResult:
        """
        Execute a task using the collective.
        
        Routes to appropriate team based on capabilities.
        """
        start_time = datetime.now(timezone.utc)
        
        # Find best team
        team_name = self._route_to_team(request)
        
        if team_name and team_name in self._commanders:
            commander = self._commanders[team_name]
            result = commander.route_task(request)
        else:
            # Find any capable agent
            agent = self._registry.find_for_task(request)
            if agent:
                result = agent.process(request)
            else:
                result = TaskResult(
                    task_id=request.task_id,
                    status=TaskStatus.FAILED,
                    error_message="No capable agent found",
                )
        
        # Update stats
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        self._update_stats(result, elapsed)
        
        return result
    
    def _route_to_team(self, request: TaskRequest) -> Optional[str]:
        """Determine which team should handle the request."""
        # Check required capabilities
        caps = request.required_capabilities
        
        if not caps:
            # Infer from task type
            task_type = request.task_type
            if task_type in ["generate", "inference", "complete"]:
                return "inference"
            elif task_type in ["compress", "encode", "decode"]:
                return "compression"
            elif task_type in ["store", "retrieve", "search"]:
                return "storage"
            elif task_type in ["analyze", "summarize", "classify"]:
                return "analysis"
            elif task_type in ["synthesize", "create", "translate", "code"]:
                return "synthesis"
            return None
        
        # Route by capability
        if AgentCapability.INFERENCE in caps:
            return "inference"
        elif AgentCapability.COMPRESSION in caps:
            return "compression"
        elif AgentCapability.STORAGE in caps:
            return "storage"
        elif AgentCapability.ANALYSIS in caps or AgentCapability.SUMMARIZATION in caps:
            return "analysis"
        elif AgentCapability.SYNTHESIS in caps or AgentCapability.CODE_GENERATION in caps:
            return "synthesis"
        
        return None
    
    def _update_stats(self, result: TaskResult, elapsed_ms: float) -> None:
        """Update collective statistics."""
        self._stats.total_tasks_completed += 1
        
        # Rolling average
        n = self._stats.total_tasks_completed
        old_avg = self._stats.average_task_time_ms
        self._stats.average_task_time_ms = old_avg + (elapsed_ms - old_avg) / n
        
        self._stats.total_tokens_processed += (
            result.tokens_processed + result.tokens_generated
        )
    
    def get_agent(self, agent_id: str) -> Optional[EliteAgent]:
        """Get agent by ID."""
        return self._agents.get(agent_id)
    
    def get_team(self, team_name: str) -> Optional[TeamCommander]:
        """Get team commander."""
        return self._commanders.get(team_name)
    
    def list_agents(self) -> List[str]:
        """List all agent IDs."""
        return list(self._agents.keys())
    
    def list_teams(self) -> List[str]:
        """List all team names."""
        return list(self._commanders.keys())
    
    def get_stats(self) -> CollectiveStats:
        """Get collective statistics."""
        # Update team stats
        for team_name, commander in self._commanders.items():
            self._stats.team_stats[team_name] = commander.get_team_status()
        
        return self._stats
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of collective."""
        return {
            "status": "healthy",
            "total_agents": self._stats.total_agents,
            "active_agents": self._stats.active_agents,
            "teams": {
                name: cmd.get_team_status()
                for name, cmd in self._commanders.items()
            },
        }


# Convenience functions
def create_elite_collective(
    config: Optional[CollectiveConfig] = None,
) -> EliteCollective:
    """Create a fully initialized Elite Collective."""
    return EliteCollective(config)


def get_all_agent_ids() -> List[str]:
    """Get list of all Elite Agent IDs."""
    return [
        # Inference Team
        "inference_commander", "prompt_architect", "context_manager",
        "token_optimizer", "stream_controller", "batch_processor",
        "cache_strategist", "latency_minimizer",
        # Compression Team
        "compression_commander", "glyph_master", "semantic_hasher",
        "rsu_architect", "delta_encoder", "pattern_miner",
        "compression_analyst", "decompression_expert",
        # Storage Team
        "storage_commander", "vault_navigator", "encryption_guard",
        "retrieval_optimizer", "manifold_mapper", "data_integrity_checker",
        "cache_coordinator", "garbage_collector",
        # Analysis Team
        "analysis_commander", "sentiment_analyst", "entity_extractor",
        "topic_modeler", "summary_expert", "classification_agent",
        "similarity_matcher", "trend_detector",
        # Synthesis Team
        "synthesis_commander", "content_creator", "code_crafter",
        "translation_expert", "style_adapter", "format_converter",
        "quality_assurer", "output_polisher",
    ]
