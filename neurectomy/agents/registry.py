"""
Agent Registry
==============

Manages agent registration and discovery.
"""

from typing import Optional, List, Dict, Type
from dataclasses import dataclass, field

from .base import BaseAgent, AgentConfig
from ..core.types import TaskRequest, AgentCapability


@dataclass
class AgentRegistration:
    """Registration info for an agent."""
    
    agent_class: Type[BaseAgent]
    config: AgentConfig
    instance: Optional[BaseAgent] = None
    is_singleton: bool = True


class AgentRegistry:
    """
    Registry for agent discovery and instantiation.
    """
    
    def __init__(self):
        self._registrations: Dict[str, AgentRegistration] = {}
        self._capability_index: Dict[AgentCapability, List[str]] = {}
        self._type_index: Dict[str, List[str]] = {}
    
    def register(
        self,
        agent_class: Type[BaseAgent],
        config: AgentConfig,
        is_singleton: bool = True,
    ) -> str:
        """
        Register an agent class.
        
        Returns agent ID.
        """
        registration = AgentRegistration(
            agent_class=agent_class,
            config=config,
            is_singleton=is_singleton,
        )
        
        agent_id = config.agent_id or f"agent_{len(self._registrations)}"
        config.agent_id = agent_id
        
        self._registrations[agent_id] = registration
        
        # Index by capability
        for cap in config.capabilities:
            if cap not in self._capability_index:
                self._capability_index[cap] = []
            self._capability_index[cap].append(agent_id)
        
        # Index by type
        agent_type = config.agent_type
        if agent_type not in self._type_index:
            self._type_index[agent_type] = []
        self._type_index[agent_type].append(agent_id)
        
        return agent_id
    
    def get(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent instance by ID."""
        if agent_id not in self._registrations:
            return None
        
        reg = self._registrations[agent_id]
        
        if reg.is_singleton:
            if reg.instance is None:
                reg.instance = reg.agent_class(reg.config)
            return reg.instance
        else:
            return reg.agent_class(reg.config)
    
    def find_by_capability(
        self,
        capability: AgentCapability,
    ) -> List[BaseAgent]:
        """Find agents with a specific capability."""
        agent_ids = self._capability_index.get(capability, [])
        return [self.get(aid) for aid in agent_ids if self.get(aid)]
    
    def find_by_capabilities(
        self,
        capabilities: List[AgentCapability],
    ) -> List[BaseAgent]:
        """Find agents with all specified capabilities."""
        if not capabilities:
            return list(self.list_all())
        
        # Start with first capability
        candidates = set(self._capability_index.get(capabilities[0], []))
        
        # Intersect with remaining capabilities
        for cap in capabilities[1:]:
            candidates &= set(self._capability_index.get(cap, []))
        
        return [self.get(aid) for aid in candidates if self.get(aid)]
    
    def find_by_type(self, agent_type: str) -> List[BaseAgent]:
        """Find agents of a specific type."""
        agent_ids = self._type_index.get(agent_type, [])
        return [self.get(aid) for aid in agent_ids if self.get(aid)]
    
    def find_for_task(self, request: TaskRequest) -> Optional[BaseAgent]:
        """
        Find best agent for a task.
        
        Considers:
        - Required capabilities
        - Preferred agent
        - Agent availability
        """
        # Check preferred agent first
        if request.preferred_agent:
            agent = self.get(request.preferred_agent)
            if agent and agent.can_handle(request):
                return agent
        
        # Find by capabilities
        if request.required_capabilities:
            candidates = self.find_by_capabilities(request.required_capabilities)
            if candidates:
                # Return first available
                for agent in candidates:
                    if not agent.state.is_busy:
                        return agent
                # All busy, return first anyway
                return candidates[0]
        
        # Return any available agent
        for agent in self.list_all():
            if agent.can_handle(request) and not agent.state.is_busy:
                return agent
        
        return None
    
    def list_all(self) -> List[BaseAgent]:
        """List all registered agents."""
        return [self.get(aid) for aid in self._registrations.keys()]
    
    def list_ids(self) -> List[str]:
        """List all agent IDs."""
        return list(self._registrations.keys())
    
    def unregister(self, agent_id: str) -> bool:
        """Unregister an agent."""
        if agent_id not in self._registrations:
            return False
        
        reg = self._registrations.pop(agent_id)
        
        # Remove from indexes
        for cap in reg.config.capabilities:
            if cap in self._capability_index:
                self._capability_index[cap] = [
                    aid for aid in self._capability_index[cap]
                    if aid != agent_id
                ]
        
        agent_type = reg.config.agent_type
        if agent_type in self._type_index:
            self._type_index[agent_type] = [
                aid for aid in self._type_index[agent_type]
                if aid != agent_id
            ]
        
        return True
