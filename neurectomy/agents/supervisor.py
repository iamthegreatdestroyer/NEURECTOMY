"""
Agent Supervisor for Health Monitoring and Auto-Recovery
Monitors agent heartbeats and automatically recovers failed agents
"""

import asyncio
from typing import Dict, Optional
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class AgentHealth:
    """Agent health information"""
    agent_id: str
    status: AgentStatus
    last_heartbeat: datetime
    failure_count: int = 0
    recovery_attempts: int = 0
    last_error: Optional[str] = None


class AgentSupervisor:
    """
    Supervises agent health and performs automatic recovery
    
    Features:
    - Heartbeat monitoring with configurable timeout
    - Automatic status transitions (healthy → degraded → failed)
    - Automatic restart after configurable failure threshold
    - Health tracking per agent
    - Recovery delay between attempts
    - Health summary reporting
    
    Example:
        >>> supervisor = AgentSupervisor(
        ...     heartbeat_timeout=30,
        ...     max_failures=3,
        ...     recovery_delay=5
        ... )
        >>> supervisor.register_agent("agent_01")
        >>> await supervisor.start_monitoring()
        >>> await supervisor.report_heartbeat("agent_01")
    """
    
    def __init__(
        self,
        heartbeat_timeout: int = 30,
        max_failures: int = 3,
        recovery_delay: int = 5,
        check_interval: int = 5,
    ):
        """
        Initialize supervisor
        
        Args:
            heartbeat_timeout: Heartbeat timeout in seconds (default: 30)
            max_failures: Maximum failures before giving up (default: 3)
            recovery_delay: Delay between recovery attempts in seconds (default: 5)
            check_interval: How often to check health in seconds (default: 5)
        """
        self.heartbeat_timeout = heartbeat_timeout
        self.max_failures = max_failures
        self.recovery_delay = recovery_delay
        self.check_interval = check_interval
        
        self.agents: Dict[str, AgentHealth] = {}
        self.monitoring_task: Optional[asyncio.Task] = None
        self.running = False
        
    async def start_monitoring(self):
        """
        Start the monitoring loop
        
        Raises:
            RuntimeError: If supervisor is already running
        """
        if self.running:
            logger.warning("Supervisor already running")
            return
            
        self.running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Agent supervisor started")
        
    async def stop_monitoring(self):
        """Stop the monitoring loop"""
        self.running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Agent supervisor stopped")
        
    def register_agent(self, agent_id: str) -> AgentHealth:
        """
        Register an agent for monitoring
        
        Args:
            agent_id: Unique agent identifier
            
        Returns:
            AgentHealth object for the registered agent
        """
        health = AgentHealth(
            agent_id=agent_id,
            status=AgentStatus.HEALTHY,
            last_heartbeat=datetime.now()
        )
        self.agents[agent_id] = health
        logger.info(f"Registered agent: {agent_id}")
        return health
        
    async def report_heartbeat(self, agent_id: str):
        """
        Agent reports heartbeat
        
        Args:
            agent_id: Unique agent identifier
        """
        if agent_id not in self.agents:
            self.register_agent(agent_id)
            
        health = self.agents[agent_id]
        health.last_heartbeat = datetime.now()
        
        # Reset failure count on successful heartbeat
        if health.status == AgentStatus.FAILED:
            health.status = AgentStatus.HEALTHY
            health.failure_count = 0
            health.last_error = None
            logger.info(f"Agent recovered: {agent_id}")
        elif health.status == AgentStatus.DEGRADED:
            health.status = AgentStatus.HEALTHY
            health.last_error = None
            logger.info(f"Agent recovered from degraded: {agent_id}")
            
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                await self._check_agents()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}", exc_info=True)
                
    async def _check_agents(self):
        """Check all agent health"""
        now = datetime.now()
        
        for agent_id, health in list(self.agents.items()):
            time_since_heartbeat = now - health.last_heartbeat
            timeout_seconds = self.heartbeat_timeout
            
            # Check if agent missed heartbeat
            if time_since_heartbeat.total_seconds() > timeout_seconds:
                if health.status == AgentStatus.HEALTHY:
                    logger.warning(
                        f"Agent missed heartbeat: {agent_id} "
                        f"({time_since_heartbeat.total_seconds():.1f}s ago)"
                    )
                    health.status = AgentStatus.DEGRADED
                    health.last_error = "missed_heartbeat"
                    
                elif health.status == AgentStatus.DEGRADED:
                    logger.error(f"Agent failed: {agent_id}")
                    health.status = AgentStatus.FAILED
                    health.failure_count += 1
                    health.last_error = "heartbeat_timeout"
                    
                    # Attempt recovery if under max failures
                    if health.failure_count <= self.max_failures:
                        await self._recover_agent(agent_id)
                    else:
                        logger.critical(
                            f"Agent {agent_id} exceeded max failures "
                            f"({self.max_failures}). Manual intervention required."
                        )
                        
    async def _recover_agent(self, agent_id: str):
        """
        Attempt to recover a failed agent
        
        Args:
            agent_id: Unique agent identifier
        """
        health = self.agents[agent_id]
        health.status = AgentStatus.RECOVERING
        health.recovery_attempts += 1
        
        logger.info(
            f"Attempting recovery: {agent_id} "
            f"(attempt {health.recovery_attempts}/{self.max_failures})"
        )
        
        try:
            # Wait before recovery attempt
            await asyncio.sleep(self.recovery_delay)
            
            # Restart agent (implementation depends on agent framework)
            await self._restart_agent(agent_id)
            
            health.status = AgentStatus.HEALTHY
            health.last_heartbeat = datetime.now()
            health.failure_count = 0
            health.last_error = None
            logger.info(f"Agent recovered successfully: {agent_id}")
            
        except Exception as e:
            logger.error(f"Recovery failed for {agent_id}: {e}")
            health.status = AgentStatus.FAILED
            health.last_error = str(e)
            
    async def _restart_agent(self, agent_id: str):
        """
        Restart an agent
        
        This is a placeholder - actual implementation depends on
        how agents are deployed (Docker, Kubernetes, local process, etc.)
        
        Args:
            agent_id: Unique agent identifier
        """
        # Placeholder implementation
        logger.info(f"Restarting agent: {agent_id}")
        
        # In production, this would:
        # - Stop the failed agent process/container
        # - Clean up resources (connections, files, etc.)
        # - Start a new instance
        # - Re-register with the collective
        
        # Example implementations:
        # 
        # For Kubernetes:
        # kubectl delete pod agent-{agent_id}
        # kubectl apply -f agent-{agent_id}.yaml
        #
        # For Docker:
        # docker restart agent-{agent_id}
        #
        # For local process:
        # subprocess.Popen(["python", f"agents/{agent_id}.py"])
        
        # For now, just simulate restart with a small delay
        await asyncio.sleep(0.5)
        
    def get_agent_health(self, agent_id: str) -> Optional[AgentHealth]:
        """
        Get health status for specific agent
        
        Args:
            agent_id: Unique agent identifier
            
        Returns:
            AgentHealth object or None if not found
        """
        return self.agents.get(agent_id)
        
    def get_all_health(self) -> Dict[str, AgentHealth]:
        """
        Get health status for all agents
        
        Returns:
            Dictionary mapping agent_id to AgentHealth
        """
        return self.agents.copy()
        
    def get_health_summary(self) -> Dict[str, int]:
        """
        Get summary of agent health across collective
        
        Returns:
            Dictionary with health counts and totals
        """
        summary = {
            "healthy": 0,
            "degraded": 0,
            "failed": 0,
            "recovering": 0,
            "total": len(self.agents)
        }
        
        for health in self.agents.values():
            summary[health.status.value] += 1
            
        return summary
    
    def get_failed_agents(self) -> Dict[str, AgentHealth]:
        """
        Get all failed agents
        
        Returns:
            Dictionary of failed agent IDs and their health info
        """
        return {
            agent_id: health
            for agent_id, health in self.agents.items()
            if health.status == AgentStatus.FAILED
        }
    
    def get_degraded_agents(self) -> Dict[str, AgentHealth]:
        """
        Get all degraded agents
        
        Returns:
            Dictionary of degraded agent IDs and their health info
        """
        return {
            agent_id: health
            for agent_id, health in self.agents.items()
            if health.status == AgentStatus.DEGRADED
        }
