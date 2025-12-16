"""
Dynamic Task Delegation for Elite Agent Collective
Intelligently assigns tasks to agents based on capability and load
"""

import asyncio
from typing import Dict, Set, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AgentCapability:
    """Agent capability profile"""
    agent_id: str
    capabilities: Set[str]  # Task types this agent can handle
    max_concurrent_tasks: int = 5
    active_tasks: int = 0
    specialization_score: Dict[str, float] = field(default_factory=dict)
    success_rate: Dict[str, float] = field(default_factory=dict)  # Task type -> success rate
    last_task_completed: Optional[datetime] = None
    total_tasks_completed: int = 0
    
    def __post_init__(self):
        """Initialize default specialization scores"""
        if not self.specialization_score:
            # Default: equal proficiency in all capabilities
            self.specialization_score = {cap: 1.0 for cap in self.capabilities}
        
        if not self.success_rate:
            # Default: 100% success rate for all capabilities
            self.success_rate = {cap: 1.0 for cap in self.capabilities}
    
    def get_utilization(self) -> float:
        """Get current utilization percentage"""
        if self.max_concurrent_tasks == 0:
            return 0.0
        return self.active_tasks / self.max_concurrent_tasks
    
    def has_capacity(self) -> bool:
        """Check if agent has available capacity"""
        return self.active_tasks < self.max_concurrent_tasks
    
    def can_handle(self, required_capabilities: Set[str]) -> bool:
        """Check if agent can handle task requirements"""
        return required_capabilities.issubset(self.capabilities)


@dataclass
class Task:
    """Task to be delegated"""
    task_id: str
    task_type: str
    priority: TaskPriority
    payload: Dict
    required_capabilities: Set[str]
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate task on creation"""
        if not self.task_id:
            raise ValueError("task_id is required")
        if not self.task_type:
            raise ValueError("task_type is required")


@dataclass
class DelegationResult:
    """Result of task delegation"""
    task_id: str
    agent_id: Optional[str]
    success: bool
    score: Optional[float] = None
    reason: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class TaskDelegator:
    """
    Intelligent task delegation system for Elite Agent Collective
    
    Features:
    - Capability-based task matching
    - Load-aware agent selection
    - Specialization scoring
    - Success rate tracking
    - Pending task queue
    - Collective status monitoring
    
    Delegation strategy:
    1. Filter agents with required capabilities
    2. Verify agents have capacity
    3. Score agents based on:
       - Current load (60% weight)
       - Specialization (30% weight)
       - Success rate (10% weight)
    4. Apply priority multiplier
    5. Assign to highest-scoring agent
    """
    
    def __init__(self, max_pending_tasks: int = 1000):
        """
        Initialize task delegator
        
        Args:
            max_pending_tasks: Maximum tasks to keep pending
        """
        self.agents: Dict[str, AgentCapability] = {}
        self.pending_tasks: List[Task] = []
        self.max_pending_tasks = max_pending_tasks
        self.delegation_history: List[DelegationResult] = []
        self.task_to_agent: Dict[str, str] = {}  # task_id -> agent_id mapping
    
    def register_agent(self, agent: AgentCapability):
        """
        Register an agent with capabilities
        
        Args:
            agent: AgentCapability with agent configuration
        """
        self.agents[agent.agent_id] = agent
        logger.info(
            f"Registered agent: {agent.agent_id} "
            f"with capabilities: {agent.capabilities} "
            f"(max tasks: {agent.max_concurrent_tasks})"
        )
    
    async def delegate_task(self, task: Task) -> DelegationResult:
        """
        Delegate task to best available agent
        
        Args:
            task: Task to delegate
            
        Returns:
            DelegationResult with assignment details
        """
        logger.info(f"Delegating task: {task.task_id} (type={task.task_type}, "
                   f"priority={task.priority.name})")
        
        # Find suitable agents
        suitable_agents = self._find_suitable_agents(task)
        
        if not suitable_agents:
            logger.warning(
                f"No suitable agents for task: {task.task_id}. "
                f"Required: {task.required_capabilities}"
            )
            
            # Queue task if under limit
            if len(self.pending_tasks) < self.max_pending_tasks:
                self.pending_tasks.append(task)
                result = DelegationResult(
                    task_id=task.task_id,
                    agent_id=None,
                    success=False,
                    reason="No suitable agents available - task queued"
                )
            else:
                result = DelegationResult(
                    task_id=task.task_id,
                    agent_id=None,
                    success=False,
                    reason="No suitable agents and pending queue full"
                )
            
            self.delegation_history.append(result)
            return result
        
        # Score and select best agent
        best_agent, score = self._select_best_agent(suitable_agents, task)
        
        # Assign task
        best_agent.active_tasks += 1
        self.task_to_agent[task.task_id] = best_agent.agent_id
        
        logger.info(
            f"Delegated task {task.task_id} to agent {best_agent.agent_id} "
            f"(score: {score:.3f}, load: {best_agent.active_tasks}/"
            f"{best_agent.max_concurrent_tasks})"
        )
        
        result = DelegationResult(
            task_id=task.task_id,
            agent_id=best_agent.agent_id,
            success=True,
            score=score,
            reason="Task successfully delegated"
        )
        
        self.delegation_history.append(result)
        return result
    
    def _find_suitable_agents(self, task: Task) -> List[AgentCapability]:
        """
        Find agents that can handle task
        
        Args:
            task: Task to delegate
            
        Returns:
            List of agents with required capabilities and capacity
        """
        suitable = []
        
        for agent in self.agents.values():
            # Check if agent has required capabilities
            if not agent.can_handle(task.required_capabilities):
                logger.debug(
                    f"Agent {agent.agent_id} lacks required capabilities: "
                    f"has {agent.capabilities}, needs {task.required_capabilities}"
                )
                continue
            
            # Check if agent has capacity
            if not agent.has_capacity():
                logger.debug(
                    f"Agent {agent.agent_id} at capacity: "
                    f"{agent.active_tasks}/{agent.max_concurrent_tasks}"
                )
                continue
            
            suitable.append(agent)
        
        return suitable
    
    def _select_best_agent(
        self,
        candidates: List[AgentCapability],
        task: Task
    ) -> Tuple[AgentCapability, float]:
        """
        Select best agent from candidates using weighted scoring
        
        Args:
            candidates: List of candidate agents
            task: Task to delegate
            
        Returns:
            Tuple of (best_agent, score)
        """
        scores = []
        
        for agent in candidates:
            # Load score (60% weight) - prefer less loaded agents
            utilization = agent.get_utilization()
            load_score = 1.0 - utilization
            
            # Specialization score (30% weight)
            specialization_score = agent.specialization_score.get(task.task_type, 0.5)
            
            # Success rate score (10% weight)
            success_rate = agent.success_rate.get(task.task_type, 1.0)
            
            # Priority multiplier (higher priority gets better agents)
            priority_multiplier = 1.0 + (task.priority.value * 0.15)
            
            # Combined weighted score
            total_score = (
                load_score * 0.6 +
                specialization_score * 0.3 +
                success_rate * 0.1
            ) * priority_multiplier
            
            scores.append((total_score, agent))
            
            logger.debug(
                f"Agent {agent.agent_id} score: {total_score:.3f} "
                f"(load: {load_score:.3f}, spec: {specialization_score:.3f}, "
                f"success: {success_rate:.3f})"
            )
        
        # Return agent with highest score
        scores.sort(key=lambda x: x[0], reverse=True)
        best_score, best_agent = scores[0]
        
        return best_agent, best_score
    
    async def mark_task_complete(
        self,
        task_id: str,
        agent_id: Optional[str] = None,
        success: bool = True,
        task_type: Optional[str] = None
    ):
        """
        Mark task as complete and free agent capacity
        
        Args:
            task_id: Task ID that completed
            agent_id: Optional agent ID (use stored mapping if not provided)
            success: Whether task completed successfully
            task_type: Task type (for success rate tracking)
        """
        # Get agent ID from mapping if not provided
        if agent_id is None:
            agent_id = self.task_to_agent.get(task_id)
        
        if agent_id is None:
            logger.warning(f"Task {task_id} not found in delegation mapping")
            return
        
        agent = self.agents.get(agent_id)
        if not agent:
            logger.warning(f"Agent {agent_id} not found")
            return
        
        # Update agent state
        agent.active_tasks = max(0, agent.active_tasks - 1)
        agent.last_task_completed = datetime.now()
        agent.total_tasks_completed += 1
        
        # Update success rate if task type provided
        if task_type and task_type in agent.success_rate:
            current_rate = agent.success_rate[task_type]
            # Update success rate with exponential moving average
            alpha = 0.1  # Smoothing factor
            new_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * current_rate
            agent.success_rate[task_type] = new_rate
        
        logger.info(
            f"Task completed by agent {agent_id} "
            f"(success: {success}, remaining: {agent.active_tasks})"
        )
        
        # Clean up mapping
        if task_id in self.task_to_agent:
            del self.task_to_agent[task_id]
        
        # Try to assign pending tasks
        await self._process_pending_tasks()
    
    async def _process_pending_tasks(self):
        """Attempt to assign pending tasks"""
        if not self.pending_tasks:
            return
        
        logger.info(f"Processing {len(self.pending_tasks)} pending tasks")
        
        tasks_to_retry = self.pending_tasks.copy()
        self.pending_tasks.clear()
        
        for task in tasks_to_retry:
            result = await self.delegate_task(task)
            if not result.success:
                # Still can't assign, keep pending
                self.pending_tasks.append(task)
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict]:
        """
        Get current status of an agent
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Agent status dictionary or None if not found
        """
        agent = self.agents.get(agent_id)
        if not agent:
            return None
        
        return {
            "agent_id": agent.agent_id,
            "active_tasks": agent.active_tasks,
            "max_tasks": agent.max_concurrent_tasks,
            "utilization": agent.get_utilization(),
            "capabilities": sorted(list(agent.capabilities)),
            "specialization": agent.specialization_score.copy(),
            "success_rates": agent.success_rate.copy(),
            "total_completed": agent.total_tasks_completed,
            "last_completed": agent.last_task_completed.isoformat() 
                            if agent.last_task_completed else None,
        }
    
    def get_collective_status(self) -> Dict:
        """
        Get status of entire agent collective
        
        Returns:
            Collective status with aggregated metrics
        """
        if not self.agents:
            return {
                "total_agents": 0,
                "total_capacity": 0,
                "active_tasks": 0,
                "pending_tasks": 0,
                "utilization": 0.0,
                "agents": {},
            }
        
        total_capacity = sum(a.max_concurrent_tasks for a in self.agents.values())
        total_active = sum(a.active_tasks for a in self.agents.values())
        
        agent_stats = {
            agent_id: {
                "utilization": agent.get_utilization(),
                "active": agent.active_tasks,
                "capacity": agent.max_concurrent_tasks,
                "capabilities": len(agent.capabilities),
            }
            for agent_id, agent in self.agents.items()
        }
        
        return {
            "total_agents": len(self.agents),
            "total_capacity": total_capacity,
            "active_tasks": total_active,
            "pending_tasks": len(self.pending_tasks),
            "utilization": total_active / total_capacity if total_capacity > 0 else 0.0,
            "agents": agent_stats,
        }
    
    def get_delegation_history(
        self,
        limit: Optional[int] = None,
        agent_id: Optional[str] = None,
        success_only: Optional[bool] = None
    ) -> List[DelegationResult]:
        """
        Get delegation history with optional filtering
        
        Args:
            limit: Maximum results to return
            agent_id: Filter by agent
            success_only: Filter by success status
            
        Returns:
            List of delegation results
        """
        results = self.delegation_history
        
        if agent_id:
            results = [r for r in results if r.agent_id == agent_id]
        
        if success_only is not None:
            results = [r for r in results if r.success == success_only]
        
        if limit:
            results = results[-limit:]
        
        return results
    
    def get_stats(self) -> Dict:
        """
        Get detailed delegation statistics
        
        Returns:
            Comprehensive statistics
        """
        history = self.delegation_history
        
        total_delegations = len(history)
        successful = sum(1 for r in history if r.success)
        failed = total_delegations - successful
        
        avg_score = 0.0
        if history:
            scores = [r.score for r in history if r.score is not None]
            if scores:
                avg_score = sum(scores) / len(scores)
        
        return {
            "total_delegations": total_delegations,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total_delegations if total_delegations > 0 else 0.0,
            "average_score": avg_score,
            "collective": self.get_collective_status(),
        }
