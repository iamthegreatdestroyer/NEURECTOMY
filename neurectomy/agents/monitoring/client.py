"""
Elite Agent Collective - Metrics Collection & Update Client
============================================================

Provides client library for agents to update metrics during operation.
"""

from neurectomy.agents.monitoring.metrics import (
    EliteAgentMetrics, AgentTier, AgentSpecialization,
    AGENT_REGISTRY, get_agent_info, get_tier_name
)
from prometheus_client import REGISTRY
from typing import Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime
import time


@dataclass
class TaskMetrics:
    """Track metrics for a single task."""
    task_id: str
    task_type: str
    agent_id: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "pending"  # pending, completed, failed, timeout
    error_type: Optional[str] = None


class AgentMetricsClient:
    """
    Client for agents to report metrics during operation.
    
    Usage:
        client = AgentMetricsClient(agent_id="apex")
        
        # Track task
        task_id = client.start_task("design_system")
        try:
            result = agent.execute_task(...)
            client.complete_task(task_id)
        except TimeoutError:
            client.fail_task(task_id, error_type="timeout")
        
        # Update status
        client.update_status("healthy", availability=99.5)
        client.record_handoff(to_agent="cipher")
    """
    
    def __init__(self, agent_id: str, registry=None):
        """Initialize metrics client for agent."""
        self.agent_id = agent_id
        self.metrics = EliteAgentMetrics(registry=registry or REGISTRY)
        
        # Get agent info
        tier, specialization = get_agent_info(agent_id)
        self.tier = tier
        self.tier_number = tier.value
        self.specialization = specialization
        self.tier_name = get_tier_name(tier)
        
        # Agent display name (uppercase)
        self.agent_name = agent_id.upper()
        
        # Track active tasks
        self.active_tasks: Dict[str, TaskMetrics] = {}
        self.start_time = time.time()
        
        # Initialize status
        self._init_agent_metrics()
    
    def _init_agent_metrics(self):
        """Initialize agent metrics on startup."""
        # Set initial status (healthy)
        self.metrics.agent_status.labels(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            tier=self.tier_number,
            specialization=self.specialization.value,
        ).set(0)
        
        # Set availability (100% initially)
        self.metrics.agent_availability_percent.labels(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            tier=self.tier_number,
        ).set(100)
        
        # Set max capacity
        default_capacity = 15  # Default concurrent tasks
        self.metrics.agent_max_concurrent_capacity.labels(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            tier=self.tier_number,
        ).set(default_capacity)
        
        # Initialize rates to 0
        self.metrics.agent_utilization_ratio.labels(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            tier=self.tier_number,
        ).set(0)
        
        self.metrics.agent_active_tasks.labels(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            tier=self.tier_number,
        ).set(0)
    
    # ========================================================================
    # TASK MANAGEMENT
    # ========================================================================
    
    def start_task(self, task_type: str) -> str:
        """Record task start."""
        task_id = f"{self.agent_id}_{len(self.active_tasks)}_{int(time.time() * 1000)}"
        
        task = TaskMetrics(
            task_id=task_id,
            task_type=task_type,
            agent_id=self.agent_id,
            start_time=time.time(),
        )
        self.active_tasks[task_id] = task
        
        # Increment assigned counter
        self.metrics.tasks_assigned_total.labels(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            tier=self.tier_number,
            task_type=task_type,
        ).inc()
        
        # Update active task count
        self.metrics.agent_active_tasks.labels(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            tier=self.tier_number,
        ).set(len(self.active_tasks))
        
        return task_id
    
    def complete_task(self, task_id: str, success: bool = True):
        """Record task completion."""
        if task_id not in self.active_tasks:
            return
        
        task = self.active_tasks.pop(task_id)
        task.end_time = time.time()
        duration = task.end_time - task.start_time
        
        # Record duration
        self.metrics.task_duration_seconds.labels(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            tier=self.tier_number,
            task_type=task.task_type,
        ).observe(duration)
        
        if success:
            # Increment completed counter
            self.metrics.tasks_completed_total.labels(
                agent_id=self.agent_id,
                agent_name=self.agent_name,
                tier=self.tier_number,
                task_type=task.task_type,
            ).inc()
            task.status = "completed"
        else:
            # Increment failed counter
            self.metrics.tasks_failed_total.labels(
                agent_id=self.agent_id,
                agent_name=self.agent_name,
                tier=self.tier_number,
                task_type=task.task_type,
                error_type="unknown_error",
            ).inc()
            task.status = "failed"
        
        # Update active task count
        self.metrics.agent_active_tasks.labels(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            tier=self.tier_number,
        ).set(len(self.active_tasks))
    
    def fail_task(self, task_id: str, error_type: str = "unknown_error"):
        """Record task failure."""
        if task_id not in self.active_tasks:
            return
        
        task = self.active_tasks.pop(task_id)
        task.end_time = time.time()
        task.status = "failed"
        task.error_type = error_type
        duration = task.end_time - task.start_time
        
        # Record duration even on failure
        self.metrics.task_duration_seconds.labels(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            tier=self.tier_number,
            task_type=task.task_type,
        ).observe(duration)
        
        # Increment failed counter
        self.metrics.tasks_failed_total.labels(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            tier=self.tier_number,
            task_type=task.task_type,
            error_type=error_type,
        ).inc()
        
        # Update active task count
        self.metrics.agent_active_tasks.labels(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            tier=self.tier_number,
        ).set(len(self.active_tasks))
    
    # ========================================================================
    # STATUS & HEALTH
    # ========================================================================
    
    def update_status(self, status: str, availability: float = 100.0):
        """Update agent status (healthy=0, degraded=1, failed=2)."""
        status_code = {"healthy": 0, "degraded": 1, "failed": 2}.get(status, 0)
        
        self.metrics.agent_status.labels(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            tier=self.tier_number,
            specialization=self.specialization.value,
        ).set(status_code)
        
        self.metrics.agent_availability_percent.labels(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            tier=self.tier_number,
        ).set(availability)
    
    def record_recovery(self):
        """Record recovery event."""
        self.metrics.agent_recovery_events.labels(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            tier=self.tier_number,
        ).inc()
    
    def update_uptime(self):
        """Update current uptime."""
        uptime_seconds = time.time() - self.start_time
        self.metrics.agent_uptime_seconds.labels(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            tier=self.tier_number,
        ).set(uptime_seconds)
    
    # ========================================================================
    # UTILIZATION & CAPACITY
    # ========================================================================
    
    def update_utilization(
        self,
        active_tasks: int,
        max_capacity: int,
        queue_length: int = 0,
        idle_percent: float = 0.0
    ):
        """Update utilization metrics."""
        utilization = active_tasks / max_capacity if max_capacity > 0 else 0
        
        self.metrics.agent_utilization_ratio.labels(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            tier=self.tier_number,
        ).set(min(utilization, 1.0))
        
        self.metrics.agent_active_tasks.labels(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            tier=self.tier_number,
        ).set(active_tasks)
        
        self.metrics.agent_wait_queue_length.labels(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            tier=self.tier_number,
        ).set(queue_length)
        
        self.metrics.agent_idle_time_percent.labels(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            tier=self.tier_number,
        ).set(idle_percent)
    
    # ========================================================================
    # PERFORMANCE & QUALITY
    # ========================================================================
    
    def update_rates(
        self,
        success_rate: float,
        error_rate: float,
        timeout_rate: float,
        retry_rate: float
    ):
        """Update quality rates (all 0-100)."""
        self.metrics.agent_success_rate.labels(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            tier=self.tier_number,
        ).set(max(0, min(100, success_rate)))
        
        self.metrics.agent_error_rate.labels(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            tier=self.tier_number,
        ).set(max(0, min(100, error_rate)))
        
        self.metrics.agent_timeout_rate.labels(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            tier=self.tier_number,
        ).set(max(0, min(100, timeout_rate)))
        
        self.metrics.agent_retry_rate.labels(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            tier=self.tier_number,
        ).set(max(0, min(100, retry_rate)))
    
    # ========================================================================
    # COLLABORATION
    # ========================================================================
    
    def record_handoff(
        self,
        to_agent: str,
        success: bool = True
    ):
        """Record task handoff to another agent."""
        to_tier, _ = get_agent_info(to_agent)
        to_tier_number = to_tier.value
        
        self.metrics.agent_handoff_events.labels(
            from_agent=self.agent_id,
            to_agent=to_agent,
            from_tier=self.tier_number,
            to_tier=to_tier_number,
        ).inc()
    
    def record_collaboration(
        self,
        other_agent: str,
        collaboration_score: float
    ):
        """Record collaboration metrics with another agent."""
        other_tier, _ = get_agent_info(other_agent)
        other_tier_number = other_tier.value
        
        # Store collaboration score (0-1)
        self.metrics.agent_collaboration_score.labels(
            agent_a=self.agent_id,
            agent_b=other_agent,
            tier=min(self.tier_number, other_tier_number),
        ).set(max(0, min(1, collaboration_score)))
    
    def record_knowledge_sharing(
        self,
        to_agent: str,
        knowledge_type: str = "experience"
    ):
        """Record knowledge sharing event."""
        to_tier, _ = get_agent_info(to_agent)
        to_tier_number = to_tier.value
        
        self.metrics.knowledge_sharing_events.labels(
            from_agent=self.agent_id,
            to_agent=to_agent,
            from_tier=self.tier_number,
            to_tier=to_tier_number,
        ).inc()
    
    # ========================================================================
    # META-INTELLIGENCE
    # ========================================================================
    
    def update_learning_rate(self, learning_rate: float):
        """Update agent learning rate (0-1)."""
        self.metrics.agent_learning_rate.labels(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            tier=self.tier_number,
        ).set(max(0, min(1, learning_rate)))
    
    def record_breakthrough(self):
        """Record a breakthrough discovery."""
        self.metrics.breakthrough_discoveries_total.labels(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            tier=self.tier_number,
            specialization=self.specialization.value,
        ).inc()
    
    def update_memory_fitness(self, fitness_score: float):
        """Update memory fitness (0-1)."""
        self.metrics.mnemonic_memory_fitness.labels(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            tier=self.tier_number,
        ).set(max(0, min(1, fitness_score)))
    
    def update_retrieval_efficiency(
        self,
        retrieval_type: str,  # bloom_filter, lsh_index, hnsw_graph
        efficiency_percent: float
    ):
        """Update memory retrieval efficiency."""
        self.metrics.retrieval_efficiency.labels(
            retrieval_type=retrieval_type,
            tier=self.tier_number,
        ).set(max(0, min(100, efficiency_percent)))


# ============================================================================
# COLLECTIVE METRICS AGGREGATOR
# ============================================================================

class CollectiveMetricsAggregator:
    """
    Aggregates metrics across all agents and tiers.
    Called periodically (e.g., every 60 seconds) to compute collective metrics.
    """
    
    def __init__(self, metrics: EliteAgentMetrics):
        """Initialize aggregator."""
        self.metrics = metrics
    
    def aggregate_tier_metrics(self, tier: int):
        """Aggregate metrics for a specific tier."""
        # This would query all agents in tier and compute aggregates
        # Implementation depends on how agents report metrics
        pass
    
    def aggregate_collective_metrics(self):
        """Compute collective-level metrics."""
        # Aggregate across all tiers
        # Update:
        # - elite_collective_utilization_ratio
        # - elite_collective_active_tasks
        # - elite_collective_throughput_tasks_per_sec
        # - elite_collective_error_rate
        # - elite_collective_coordination_effectiveness
        pass
    
    def update_collective_intelligence_score(self, tier: int):
        """Update collective intelligence score for tier."""
        # Score = weighted blend of:
        # - Success rates (40%)
        # - Breakthrough discoveries (30%)
        # - Knowledge sharing (20%)
        # - Coordination effectiveness (10%)
        pass


if __name__ == "__main__":
    # Example usage
    client = AgentMetricsClient(agent_id="apex")
    
    # Simulate task execution
    task_id = client.start_task("design_system")
    time.sleep(0.1)  # Simulate work
    client.complete_task(task_id)
    
    # Update status
    client.update_status("healthy", availability=99.9)
    client.update_utilization(
        active_tasks=5,
        max_capacity=15,
        queue_length=2,
        idle_percent=40.0
    )
    client.update_rates(
        success_rate=95.0,
        error_rate=3.0,
        timeout_rate=2.0,
        retry_rate=1.0
    )
    
    # Record collaboration
    client.record_handoff(to_agent="cipher")
    client.record_collaboration(other_agent="cipher", collaboration_score=0.85)
    
    # Record learning
    client.update_learning_rate(0.05)
    client.update_memory_fitness(0.78)
    
    print("Metrics recorded successfully")
