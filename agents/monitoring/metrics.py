"""
Prometheus Metrics for Elite Agent Collective (40-Agent System)
Comprehensive monitoring of all agents and collective performance
"""

from prometheus_client import Counter, Histogram, Gauge, Info
from functools import wraps
import time
from typing import Callable, Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Individual Agent Metrics (Per-Agent Tracking)
# ============================================================================

agent_status = Gauge(
    'agents_status',
    'Agent operational status (0=healthy, 1=degraded, 2=failed)',
    ['agent_id', 'agent_name', 'tier'],
    help='Current agent health status'
)

agent_active_tasks = Gauge(
    'agents_active_tasks',
    'Number of active tasks for agent',
    ['agent_id', 'agent_name'],
    help='Current workload'
)

agent_availability_ratio = Gauge(
    'agents_availability_ratio',
    'Agent availability (0.0 to 1.0)',
    ['agent_id', 'agent_name'],
    help='Uptime ratio'
)


# ============================================================================
# Agent Task Metrics
# ============================================================================

agent_tasks_total = Counter(
    'agents_tasks_total',
    'Total tasks assigned to agent',
    ['agent_id', 'agent_name', 'task_type'],
    help='Cumulative task count'
)

agent_tasks_completed = Counter(
    'agents_tasks_completed',
    'Tasks completed successfully',
    ['agent_id', 'agent_name', 'task_type'],
    help='Successful task count'
)

agent_tasks_failed = Counter(
    'agents_tasks_failed',
    'Tasks that failed',
    ['agent_id', 'agent_name', 'error_type'],
    help='Failed task count'
)

agent_task_duration_seconds = Histogram(
    'agents_task_duration_seconds',
    'Task execution duration',
    ['agent_id', 'agent_name', 'task_type'],
    buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0),
    help='Task completion time'
)


# ============================================================================
# Agent Performance Metrics
# ============================================================================

agent_success_rate = Gauge(
    'agents_success_rate',
    'Agent success rate (0.0 to 1.0)',
    ['agent_id', 'agent_name'],
    help='Fraction of successful tasks'
)

agent_error_rate = Gauge(
    'agents_error_rate',
    'Agent error rate (0.0 to 1.0)',
    ['agent_id', 'agent_name'],
    help='Fraction of failed tasks'
)

agent_timeout_rate = Gauge(
    'agents_timeout_rate',
    'Agent timeout rate',
    ['agent_id', 'agent_name'],
    help='Fraction of timed-out tasks'
)

agent_average_task_duration = Gauge(
    'agents_average_task_duration_seconds',
    'Average task duration for agent',
    ['agent_id', 'agent_name'],
    help='Mean task execution time'
)

agent_p95_task_duration = Gauge(
    'agents_p95_task_duration_seconds',
    'P95 task duration for agent',
    ['agent_id', 'agent_name'],
    help='95th percentile task time'
)


# ============================================================================
# Agent Utilization Metrics
# ============================================================================

agent_utilization_ratio = Gauge(
    'agents_utilization_ratio',
    'Agent utilization (0.0 to 1.0)',
    ['agent_id', 'agent_name'],
    help='Active tasks / max capacity'
)

agent_max_capacity = Gauge(
    'agents_max_capacity',
    'Maximum concurrent tasks for agent',
    ['agent_id', 'agent_name'],
    help='Agent capacity'
)

agent_queue_length = Gauge(
    'agents_queue_length',
    'Tasks waiting for agent',
    ['agent_id', 'agent_name'],
    help='Queue depth'
)

agent_idle_time_seconds = Counter(
    'agents_idle_time_seconds',
    'Total idle time for agent',
    ['agent_id', 'agent_name'],
    help='Cumulative idle duration'
)


# ============================================================================
# Agent Specialization Metrics
# ============================================================================

agent_specialization = Gauge(
    'agents_specialization',
    'Agent specialization proficiency (0.0 to 1.0)',
    ['agent_id', 'agent_name', 'specialization'],
    help='Proficiency in specialization'
)

agent_task_type_distribution = Histogram(
    'agents_task_type_distribution',
    'Distribution of task types for agent',
    ['agent_id', 'agent_name', 'task_type'],
    buckets=(0.01, 0.05, 0.1, 0.2, 0.3, 0.5),
    help='Proportion of tasks by type'
)


# ============================================================================
# Agent Recovery and Resilience Metrics
# ============================================================================

agent_recovery_events_total = Counter(
    'agents_recovery_events_total',
    'Total recovery events',
    ['agent_id', 'agent_name', 'recovery_type'],
    help='Recovery operation count'
)

agent_restart_count = Counter(
    'agents_restart_count',
    'Agent restart count',
    ['agent_id', 'agent_name'],
    help='Number of restarts'
)

agent_mttr_seconds = Gauge(
    'agents_mttr_seconds',
    'Mean time to recovery',
    ['agent_id', 'agent_name'],
    help='Average recovery time'
)


# ============================================================================
# Collective (Aggregate) Metrics
# ============================================================================

collective_total_agents = Gauge(
    'agents_collective_total',
    'Total agents in collective',
    help='Number of active agents'
)

collective_healthy_agents = Gauge(
    'agents_collective_healthy',
    'Number of healthy agents',
    help='Operational agents'
)

collective_degraded_agents = Gauge(
    'agents_collective_degraded',
    'Number of degraded agents',
    help='Degraded agents'
)

collective_failed_agents = Gauge(
    'agents_collective_failed',
    'Number of failed agents',
    help='Failed agents'
)

collective_total_active_tasks = Gauge(
    'agents_collective_active_tasks',
    'Total active tasks across collective',
    help='Current total workload'
)

collective_utilization_ratio = Gauge(
    'agents_collective_utilization_ratio',
    'Collective utilization ratio',
    help='Overall resource utilization'
)

collective_throughput_tasks_per_second = Gauge(
    'agents_collective_throughput_tasks_per_second',
    'Collective throughput',
    help='Tasks/second across all agents'
)

collective_error_rate = Gauge(
    'agents_collective_error_rate',
    'Collective error rate',
    help='Fraction of failed tasks'
)

collective_success_rate = Gauge(
    'agents_collective_success_rate',
    'Collective success rate',
    help='Fraction of successful tasks'
)


# ============================================================================
# Tier-Specific Metrics
# ============================================================================

tier_health_score = Gauge(
    'agents_tier_health_score',
    'Health score for tier (0.0 to 1.0)',
    ['tier'],
    help='Aggregate tier health'
)

tier_utilization = Gauge(
    'agents_tier_utilization_ratio',
    'Tier utilization ratio',
    ['tier'],
    help='Average utilization in tier'
)

tier_task_count = Counter(
    'agents_tier_task_count',
    'Total tasks handled by tier',
    ['tier'],
    help='Cumulative tasks'
)

tier_error_rate = Gauge(
    'agents_tier_error_rate',
    'Tier error rate',
    ['tier'],
    help='Error rate for tier'
)


# ============================================================================
# Inter-Agent Collaboration Metrics
# ============================================================================

agent_collaboration_events_total = Counter(
    'agents_collaboration_events_total',
    'Inter-agent collaboration events',
    ['initiator_agent', 'target_agent'],
    help='Collaboration count'
)

agent_handoff_total = Counter(
    'agents_handoff_total',
    'Task handoffs between agents',
    ['source_agent', 'target_agent'],
    help='Handoff count'
)

agent_communication_latency_seconds = Histogram(
    'agents_communication_latency_seconds',
    'Communication latency between agents',
    buckets=(0.001, 0.01, 0.1, 1.0, 10.0),
    help='Inter-agent latency'
)


# ============================================================================
# Collective Intelligence Metrics
# ============================================================================

collective_intelligence_score = Gauge(
    'agents_collective_intelligence_score',
    'Collective intelligence metric',
    help='Emergent intelligence level'
)

collective_breakthrough_count = Counter(
    'agents_collective_breakthrough_count',
    'Breakthrough discoveries',
    help='Novel insights count'
)

agent_knowledge_sharing_events = Counter(
    'agents_knowledge_sharing_events',
    'Knowledge sharing events',
    ['source_tier', 'target_tier'],
    help='Cross-tier learning'
)


# ============================================================================
# System Information
# ============================================================================

system_info = Info(
    'agents_system',
    'Elite Agent Collective system information',
)

agent_info = Info(
    'agents_info',
    'Individual agent information',
)


# ============================================================================
# Decorators
# ============================================================================

def track_agent_task(agent_id: str, agent_name: str, task_type: str):
    """Decorator to track agent task execution"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                agent_tasks_total.labels(agent_id=agent_id, agent_name=agent_name, task_type=task_type).inc()
                result = await func(*args, **kwargs)
                agent_tasks_completed.labels(agent_id=agent_id, agent_name=agent_name, task_type=task_type).inc()
                return result
            except TimeoutError:
                status = "timeout"
                agent_tasks_failed.labels(agent_id=agent_id, agent_name=agent_name, error_type='timeout').inc()
                raise
            except Exception as e:
                status = "error"
                error_type = type(e).__name__
                agent_tasks_failed.labels(agent_id=agent_id, agent_name=agent_name, error_type=error_type).inc()
                logger.error(f"Agent {agent_name} task error: {e}")
                raise
            finally:
                duration = time.time() - start_time
                agent_task_duration_seconds.labels(
                    agent_id=agent_id,
                    agent_name=agent_name,
                    task_type=task_type
                ).observe(duration)
        
        return wrapper
    return decorator


# ============================================================================
# Helper Functions - Individual Agent
# ============================================================================

def update_agent_status(agent_id: str, agent_name: str, tier: str, status: int):
    """Update agent status (0=healthy, 1=degraded, 2=failed)"""
    agent_status.labels(agent_id=agent_id, agent_name=agent_name, tier=tier).set(status)


def update_agent_metrics(agent_id: str, agent_name: str, active_tasks: int, availability: float):
    """Update agent metrics"""
    agent_active_tasks.labels(agent_id=agent_id, agent_name=agent_name).set(active_tasks)
    agent_availability_ratio.labels(agent_id=agent_id, agent_name=agent_name).set(availability)


def update_agent_rates(agent_id: str, agent_name: str, success_rate: float, error_rate: float, timeout_rate: float):
    """Update agent performance rates"""
    agent_success_rate.labels(agent_id=agent_id, agent_name=agent_name).set(success_rate)
    agent_error_rate.labels(agent_id=agent_id, agent_name=agent_name).set(error_rate)
    agent_timeout_rate.labels(agent_id=agent_id, agent_name=agent_name).set(timeout_rate)


def update_agent_utilization(agent_id: str, agent_name: str, utilization: float, capacity: int, queue_length: int):
    """Update agent utilization metrics"""
    agent_utilization_ratio.labels(agent_id=agent_id, agent_name=agent_name).set(utilization)
    agent_max_capacity.labels(agent_id=agent_id, agent_name=agent_name).set(capacity)
    agent_queue_length.labels(agent_id=agent_id, agent_name=agent_name).set(queue_length)


def record_agent_recovery(agent_id: str, agent_name: str, recovery_type: str):
    """Record agent recovery event"""
    agent_recovery_events_total.labels(
        agent_id=agent_id,
        agent_name=agent_name,
        recovery_type=recovery_type
    ).inc()


# ============================================================================
# Helper Functions - Collective
# ============================================================================

def update_collective_health(total: int, healthy: int, degraded: int, failed: int):
    """Update collective health snapshot"""
    collective_total_agents.set(total)
    collective_healthy_agents.set(healthy)
    collective_degraded_agents.set(degraded)
    collective_failed_agents.set(failed)


def update_collective_metrics(active_tasks: int, utilization: float, throughput: float, error_rate: float):
    """Update collective performance metrics"""
    collective_total_active_tasks.set(active_tasks)
    collective_utilization_ratio.set(utilization)
    collective_throughput_tasks_per_second.set(throughput)
    collective_error_rate.set(error_rate)
    collective_success_rate.set(1.0 - error_rate)


def update_tier_health(tier: str, health_score: float, utilization: float, error_rate: float):
    """Update tier-level health metrics"""
    tier_health_score.labels(tier=tier).set(health_score)
    tier_utilization.labels(tier=tier).set(utilization)
    tier_error_rate.labels(tier=tier).set(error_rate)


# ============================================================================
# Helper Functions - Collaboration
# ============================================================================

def record_collaboration(initiator: str, target: str):
    """Record inter-agent collaboration"""
    agent_collaboration_events_total.labels(initiator_agent=initiator, target_agent=target).inc()


def record_task_handoff(source: str, target: str):
    """Record task handoff between agents"""
    agent_handoff_total.labels(source_agent=source, target_agent=target).inc()


def record_knowledge_sharing(source_tier: str, target_tier: str):
    """Record cross-tier knowledge sharing"""
    agent_knowledge_sharing_events.labels(source_tier=source_tier, target_tier=target_tier).inc()


# ============================================================================
# Helper Functions - Collective Intelligence
# ============================================================================

def update_collective_intelligence(score: float):
    """Update collective intelligence score"""
    collective_intelligence_score.set(score)


def record_breakthrough():
    """Record breakthrough discovery"""
    collective_breakthrough_count.inc()
