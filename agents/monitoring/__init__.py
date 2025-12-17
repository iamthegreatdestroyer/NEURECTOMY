"""
Elite Agent Collective Monitoring Module
40-Agent system with comprehensive health, performance, and collaboration tracking
"""

from .metrics import (
    # Individual agent
    agent_status,
    agent_active_tasks,
    agent_availability_ratio,
    
    # Task metrics
    agent_tasks_total,
    agent_tasks_completed,
    agent_tasks_failed,
    agent_task_duration_seconds,
    
    # Performance
    agent_success_rate,
    agent_error_rate,
    agent_timeout_rate,
    agent_average_task_duration,
    agent_p95_task_duration,
    
    # Utilization
    agent_utilization_ratio,
    agent_max_capacity,
    agent_queue_length,
    agent_idle_time_seconds,
    
    # Specialization
    agent_specialization,
    agent_task_type_distribution,
    
    # Resilience
    agent_recovery_events_total,
    agent_restart_count,
    agent_mttr_seconds,
    
    # Collective
    collective_total_agents,
    collective_healthy_agents,
    collective_degraded_agents,
    collective_failed_agents,
    collective_total_active_tasks,
    collective_utilization_ratio,
    collective_throughput_tasks_per_second,
    collective_error_rate,
    collective_success_rate,
    
    # Tier
    tier_health_score,
    tier_utilization,
    tier_task_count,
    tier_error_rate,
    
    # Collaboration
    agent_collaboration_events_total,
    agent_handoff_total,
    agent_communication_latency_seconds,
    
    # Collective intelligence
    collective_intelligence_score,
    collective_breakthrough_count,
    agent_knowledge_sharing_events,
    
    # System info
    system_info,
    agent_info,
    
    # Decorators
    track_agent_task,
    
    # Helpers
    update_agent_status,
    update_agent_metrics,
    update_agent_rates,
    update_agent_utilization,
    record_agent_recovery,
    update_collective_health,
    update_collective_metrics,
    update_tier_health,
    record_collaboration,
    record_task_handoff,
    record_knowledge_sharing,
    update_collective_intelligence,
    record_breakthrough,
)

__all__ = [
    "agent_status",
    "agent_active_tasks",
    "agent_availability_ratio",
    "agent_tasks_total",
    "agent_tasks_completed",
    "agent_tasks_failed",
    "agent_task_duration_seconds",
    "agent_success_rate",
    "agent_error_rate",
    "agent_timeout_rate",
    "agent_average_task_duration",
    "agent_p95_task_duration",
    "agent_utilization_ratio",
    "agent_max_capacity",
    "agent_queue_length",
    "agent_idle_time_seconds",
    "agent_specialization",
    "agent_task_type_distribution",
    "agent_recovery_events_total",
    "agent_restart_count",
    "agent_mttr_seconds",
    "collective_total_agents",
    "collective_healthy_agents",
    "collective_degraded_agents",
    "collective_failed_agents",
    "collective_total_active_tasks",
    "collective_utilization_ratio",
    "collective_throughput_tasks_per_second",
    "collective_error_rate",
    "collective_success_rate",
    "tier_health_score",
    "tier_utilization",
    "tier_task_count",
    "tier_error_rate",
    "agent_collaboration_events_total",
    "agent_handoff_total",
    "agent_communication_latency_seconds",
    "collective_intelligence_score",
    "collective_breakthrough_count",
    "agent_knowledge_sharing_events",
    "system_info",
    "agent_info",
    "track_agent_task",
    "update_agent_status",
    "update_agent_metrics",
    "update_agent_rates",
    "update_agent_utilization",
    "record_agent_recovery",
    "update_collective_health",
    "update_collective_metrics",
    "update_tier_health",
    "record_collaboration",
    "record_task_handoff",
    "record_knowledge_sharing",
    "update_collective_intelligence",
    "record_breakthrough",
]
