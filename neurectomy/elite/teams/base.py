"""
Elite Team Base Classes
=======================

Foundation for Elite Agent teams.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum, auto

from ...agents.base import BaseAgent, AgentConfig
from ...core.types import TaskRequest, TaskResult, TaskStatus, AgentCapability


class TeamRole(Enum):
    """Agent roles within a team."""
    COMMANDER = auto()      # Team lead
    SPECIALIST = auto()     # Domain expert
    SUPPORT = auto()        # Support function
    COORDINATOR = auto()    # Cross-team coordination


@dataclass
class TeamConfig:
    """Configuration for an elite team."""
    
    team_id: str
    team_name: str
    description: str
    
    # Capabilities
    primary_capabilities: List[AgentCapability] = field(default_factory=list)
    
    # Resource limits
    max_concurrent_tasks: int = 4
    max_context_tokens: int = 8192


class EliteAgent(BaseAgent):
    """
    Enhanced base class for Elite agents.
    
    Adds team coordination and advanced features.
    """
    
    def __init__(
        self,
        config: AgentConfig,
        team_id: str,
        role: TeamRole = TeamRole.SPECIALIST,
    ):
        super().__init__(config)
        self.team_id = team_id
        self.role = role
        
        # Cross-agent communication
        self._collaborators: Dict[str, 'EliteAgent'] = {}
        self._pending_collaborations: List[str] = []
    
    def add_collaborator(self, agent: 'EliteAgent') -> None:
        """Add a collaborating agent."""
        self._collaborators[agent.agent_id] = agent
    
    def request_collaboration(
        self,
        collaborator_id: str,
        request: TaskRequest,
    ) -> Optional[TaskResult]:
        """Request collaboration from another agent."""
        if collaborator_id not in self._collaborators:
            return None
        
        collaborator = self._collaborators[collaborator_id]
        return collaborator.process(request)
    
    def delegate(
        self,
        request: TaskRequest,
        target_capability: AgentCapability,
    ) -> Optional[TaskResult]:
        """Delegate task to agent with capability."""
        for agent in self._collaborators.values():
            if target_capability in agent.config.capabilities:
                return agent.process(request)
        return None


class TeamCommander(EliteAgent):
    """
    Base class for team commanders.
    
    Coordinates team members and routes tasks.
    """
    
    def __init__(
        self,
        config: AgentConfig,
        team_id: str,
        team_config: TeamConfig,
    ):
        super().__init__(config, team_id, TeamRole.COMMANDER)
        self.team_config = team_config
        self._team_members: Dict[str, EliteAgent] = {}
    
    def add_team_member(self, agent: EliteAgent) -> None:
        """Add agent to team."""
        self._team_members[agent.agent_id] = agent
        agent.add_collaborator(self)
        self.add_collaborator(agent)
    
    def route_task(self, request: TaskRequest) -> TaskResult:
        """Route task to best team member."""
        # Find best member for task
        best_agent = self._find_best_agent(request)
        
        if best_agent is None:
            # Handle ourselves
            return self.process(request)
        
        return best_agent.process(request)
    
    def _find_best_agent(self, request: TaskRequest) -> Optional[EliteAgent]:
        """Find best agent for task."""
        for agent in self._team_members.values():
            if agent.can_handle(request) and not agent.state.is_busy:
                return agent
        return None
    
    def get_team_status(self) -> Dict[str, Any]:
        """Get team status summary."""
        return {
            "team_id": self.team_id,
            "team_name": self.team_config.team_name,
            "member_count": len(self._team_members),
            "busy_members": sum(1 for a in self._team_members.values() if a.state.is_busy),
            "total_tasks": sum(a.state.tasks_completed for a in self._team_members.values()),
        }
