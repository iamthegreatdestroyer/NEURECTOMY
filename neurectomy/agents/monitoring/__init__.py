"""
Elite Agent Collective Monitoring System
========================================

Comprehensive metrics collection and monitoring for the 40-agent Elite Agent Collective.

Exports:
    - EliteAgentMetrics: Core metrics implementation (65+ metrics)
    - AgentMetricsClient: Client for agents to report metrics
    - PrometheusQueries: Pre-built query templates
    - AGENT_REGISTRY: Registry of all 40 agents
    - AgentTier, AgentSpecialization, AgentStatus: Enums
"""

from neurectomy.agents.monitoring.metrics import (
    # Classes
    EliteAgentMetrics,
    PrometheusQueries,
    OptimizationAnalyzer,
    OptimizationOpportunity,
    
    # Enums
    AgentTier,
    AgentSpecialization,
    AgentStatus,
    
    # Registry & Helpers
    AGENT_REGISTRY,
    get_agent_info,
    get_tier_name,
    
    # Configuration
    ALERT_RULES,
)

from neurectomy.agents.monitoring.client import (
    # Client classes
    AgentMetricsClient,
    CollectiveMetricsAggregator,
    TaskMetrics,
)

# Default metrics instance (singleton)
_default_metrics = None


def get_metrics(registry=None) -> EliteAgentMetrics:
    """
    Get or create default metrics instance.
    
    Args:
        registry: Optional Prometheus CollectorRegistry. Uses default if None.
    
    Returns:
        EliteAgentMetrics: Default metrics instance
    """
    global _default_metrics
    if _default_metrics is None:
        _default_metrics = EliteAgentMetrics(registry=registry)
    return _default_metrics


def get_client(agent_id: str) -> AgentMetricsClient:
    """
    Create metrics client for an agent.
    
    Args:
        agent_id: Agent identifier (e.g., "apex", "cipher")
    
    Returns:
        AgentMetricsClient: Initialized client for agent
    
    Raises:
        KeyError: If agent_id not found in registry
    """
    if agent_id not in AGENT_REGISTRY:
        raise KeyError(f"Unknown agent: {agent_id}")
    
    return AgentMetricsClient(agent_id=agent_id, registry=get_metrics().registry)


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Classes
    "EliteAgentMetrics",
    "AgentMetricsClient",
    "CollectiveMetricsAggregator",
    "PrometheusQueries",
    "OptimizationAnalyzer",
    "OptimizationOpportunity",
    "TaskMetrics",
    
    # Enums
    "AgentTier",
    "AgentSpecialization",
    "AgentStatus",
    
    # Data
    "AGENT_REGISTRY",
    "ALERT_RULES",
    
    # Functions
    "get_metrics",
    "get_client",
    "get_agent_info",
    "get_tier_name",
]


# ============================================================================
# VERSION & METADATA
# ============================================================================

__version__ = "1.0.0"
__description__ = "Elite Agent Collective Monitoring System"
__metrics_count__ = 65
__agent_count__ = 40
__tier_count__ = 8
__query_templates__ = 15
__alert_rules__ = 13
