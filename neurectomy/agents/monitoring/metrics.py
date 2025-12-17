"""
Elite Agent Collective - Comprehensive Metrics Design
=====================================================

Meta-level orchestration metrics for all 40 specialized agents.
Implements production-grade observability for:
  - Individual agent health and status
  - Task management and distribution
  - Utilization and capacity tracking
  - Performance and quality metrics
  - Collective-level aggregation
  - Cross-agent collaboration tracking
  - Meta-intelligence indicators
  - Per-tier performance analysis

ARCHITECTURE:
  - Prometheus metrics (time-series database)
  - Sub-linear retrieval (O(1) to O(log n))
  - Grafana dashboards (visualization)
  - AlertManager (automated alerting)
  - Per-tier aggregation
  - Agent specialization tracking

AGENTS (40 total):
  TIER 1 (5):   @APEX, @CIPHER, @ARCHITECT, @AXIOM, @VELOCITY
  TIER 2 (8):   @QUANTUM, @TENSOR, @FORTRESS, @NEURAL, @CRYPTO, 
                @FLUX, @PRISM, @SYNAPSE
  TIER 3-4 (3): @CORE, @HELIX, @VANGUARD, @ECLIPSE, @NEXUS, 
                @GENESIS, @OMNISCIENT
  TIER 5 (5):   @ATLAS, @FORGE, @SENTRY, @VERTEX, @STREAM
  TIER 6 (2):   @PHOTON, @LATTICE, @MORPH, @PHANTOM, @ORBIT
  TIER 7 (5):   @CANVAS, @LINGUA, @SCRIBE, @MENTOR, @BRIDGE
  TIER 8 (5):   @AEGIS, @LEDGER, @PULSE, @ARBITER, @ORACLE
"""

from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    CollectorRegistry, generate_latest,
    CONTENT_TYPE_LATEST, REGISTRY
)
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import time
from datetime import datetime, timedelta


# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

class AgentTier(Enum):
    """Agent specialization tier."""
    TIER_1_FOUNDATIONAL = 1
    TIER_2_SPECIALISTS = 2
    TIER_3_INNOVATORS = 3
    TIER_4_META = 4
    TIER_5_DOMAIN = 5
    TIER_6_EMERGING = 6
    TIER_7_HUMAN_CENTRIC = 7
    TIER_8_ENTERPRISE = 8


class AgentSpecialization(Enum):
    """Agent specialization category."""
    # Tier 1
    CS_ENGINEERING = "cs_engineering"          # @APEX
    CRYPTOGRAPHY = "cryptography"              # @CIPHER
    SYSTEMS_ARCHITECTURE = "systems_architecture"  # @ARCHITECT
    MATHEMATICS = "mathematics"                # @AXIOM
    PERFORMANCE = "performance"                # @VELOCITY
    
    # Tier 2
    QUANTUM_COMPUTING = "quantum_computing"    # @QUANTUM
    ML_DL = "ml_dl"                            # @TENSOR
    SECURITY = "security"                      # @FORTRESS
    AGI_RESEARCH = "agi_research"              # @NEURAL
    BLOCKCHAIN = "blockchain"                  # @CRYPTO
    DEVOPS = "devops"                          # @FLUX
    DATA_SCIENCE = "data_science"              # @PRISM
    INTEGRATION = "integration"                # @SYNAPSE
    
    # Tier 3-4
    LOW_LEVEL = "low_level"                    # @CORE
    BIOINFORMATICS = "bioinformatics"          # @HELIX
    RESEARCH = "research"                      # @VANGUARD
    TESTING = "testing"                        # @ECLIPSE
    SYNTHESIS = "synthesis"                    # @NEXUS
    INNOVATION = "innovation"                  # @GENESIS
    META_LEARNING = "meta_learning"            # @OMNISCIENT
    
    # Tier 5
    CLOUD = "cloud"                            # @ATLAS
    BUILD_SYSTEMS = "build_systems"            # @FORGE
    OBSERVABILITY = "observability"            # @SENTRY
    GRAPH_DB = "graph_db"                      # @VERTEX
    STREAM_PROCESSING = "stream_processing"    # @STREAM
    
    # Tier 6
    EDGE_IOT = "edge_iot"                      # @PHOTON
    CONSENSUS = "consensus"                    # @LATTICE
    MIGRATION = "migration"                    # @MORPH
    REVERSE_ENGINEERING = "reverse_engineering" # @PHANTOM
    EMBEDDED = "embedded"                      # @ORBIT
    
    # Tier 7
    UI_UX = "ui_ux"                            # @CANVAS
    NLP_LLM = "nlp_llm"                        # @LINGUA
    DOCUMENTATION = "documentation"            # @SCRIBE
    EDUCATION = "education"                    # @MENTOR
    CROSS_PLATFORM = "cross_platform"          # @BRIDGE
    
    # Tier 8
    COMPLIANCE = "compliance"                  # @AEGIS
    FINANCE = "finance"                        # @LEDGER
    HEALTHCARE = "healthcare"                  # @PULSE
    MERGE_RESOLUTION = "merge_resolution"      # @ARBITER
    PREDICTIVE = "predictive"                  # @ORACLE


# Agent registry: Map agent_id to tier and specialization
AGENT_REGISTRY = {
    # Tier 1 - Foundational
    "apex": (AgentTier.TIER_1_FOUNDATIONAL, AgentSpecialization.CS_ENGINEERING),
    "cipher": (AgentTier.TIER_1_FOUNDATIONAL, AgentSpecialization.CRYPTOGRAPHY),
    "architect": (AgentTier.TIER_1_FOUNDATIONAL, AgentSpecialization.SYSTEMS_ARCHITECTURE),
    "axiom": (AgentTier.TIER_1_FOUNDATIONAL, AgentSpecialization.MATHEMATICS),
    "velocity": (AgentTier.TIER_1_FOUNDATIONAL, AgentSpecialization.PERFORMANCE),
    
    # Tier 2 - Specialists
    "quantum": (AgentTier.TIER_2_SPECIALISTS, AgentSpecialization.QUANTUM_COMPUTING),
    "tensor": (AgentTier.TIER_2_SPECIALISTS, AgentSpecialization.ML_DL),
    "fortress": (AgentTier.TIER_2_SPECIALISTS, AgentSpecialization.SECURITY),
    "neural": (AgentTier.TIER_2_SPECIALISTS, AgentSpecialization.AGI_RESEARCH),
    "crypto": (AgentTier.TIER_2_SPECIALISTS, AgentSpecialization.BLOCKCHAIN),
    "flux": (AgentTier.TIER_2_SPECIALISTS, AgentSpecialization.DEVOPS),
    "prism": (AgentTier.TIER_2_SPECIALISTS, AgentSpecialization.DATA_SCIENCE),
    "synapse": (AgentTier.TIER_2_SPECIALISTS, AgentSpecialization.INTEGRATION),
    
    # Tier 3-4 - Innovators & Meta
    "core": (AgentTier.TIER_3_INNOVATORS, AgentSpecialization.LOW_LEVEL),
    "helix": (AgentTier.TIER_3_INNOVATORS, AgentSpecialization.BIOINFORMATICS),
    "vanguard": (AgentTier.TIER_3_INNOVATORS, AgentSpecialization.RESEARCH),
    "eclipse": (AgentTier.TIER_3_INNOVATORS, AgentSpecialization.TESTING),
    "nexus": (AgentTier.TIER_3_INNOVATORS, AgentSpecialization.SYNTHESIS),
    "genesis": (AgentTier.TIER_3_INNOVATORS, AgentSpecialization.INNOVATION),
    "omniscient": (AgentTier.TIER_4_META, AgentSpecialization.META_LEARNING),
    
    # Tier 5 - Domain
    "atlas": (AgentTier.TIER_5_DOMAIN, AgentSpecialization.CLOUD),
    "forge": (AgentTier.TIER_5_DOMAIN, AgentSpecialization.BUILD_SYSTEMS),
    "sentry": (AgentTier.TIER_5_DOMAIN, AgentSpecialization.OBSERVABILITY),
    "vertex": (AgentTier.TIER_5_DOMAIN, AgentSpecialization.GRAPH_DB),
    "stream": (AgentTier.TIER_5_DOMAIN, AgentSpecialization.STREAM_PROCESSING),
    
    # Tier 6 - Emerging
    "photon": (AgentTier.TIER_6_EMERGING, AgentSpecialization.EDGE_IOT),
    "lattice": (AgentTier.TIER_6_EMERGING, AgentSpecialization.CONSENSUS),
    "morph": (AgentTier.TIER_6_EMERGING, AgentSpecialization.MIGRATION),
    "phantom": (AgentTier.TIER_6_EMERGING, AgentSpecialization.REVERSE_ENGINEERING),
    "orbit": (AgentTier.TIER_6_EMERGING, AgentSpecialization.EMBEDDED),
    
    # Tier 7 - Human-Centric
    "canvas": (AgentTier.TIER_7_HUMAN_CENTRIC, AgentSpecialization.UI_UX),
    "lingua": (AgentTier.TIER_7_HUMAN_CENTRIC, AgentSpecialization.NLP_LLM),
    "scribe": (AgentTier.TIER_7_HUMAN_CENTRIC, AgentSpecialization.DOCUMENTATION),
    "mentor": (AgentTier.TIER_7_HUMAN_CENTRIC, AgentSpecialization.EDUCATION),
    "bridge": (AgentTier.TIER_7_HUMAN_CENTRIC, AgentSpecialization.CROSS_PLATFORM),
    
    # Tier 8 - Enterprise
    "aegis": (AgentTier.TIER_8_ENTERPRISE, AgentSpecialization.COMPLIANCE),
    "ledger": (AgentTier.TIER_8_ENTERPRISE, AgentSpecialization.FINANCE),
    "pulse": (AgentTier.TIER_8_ENTERPRISE, AgentSpecialization.HEALTHCARE),
    "arbiter": (AgentTier.TIER_8_ENTERPRISE, AgentSpecialization.MERGE_RESOLUTION),
    "oracle": (AgentTier.TIER_8_ENTERPRISE, AgentSpecialization.PREDICTIVE),
}


class AgentStatus(Enum):
    """Agent operational status."""
    HEALTHY = 0
    DEGRADED = 1
    FAILED = 2


# ============================================================================
# METRICS DEFINITIONS
# ============================================================================

class EliteAgentMetrics:
    """
    Comprehensive metrics collection for 40-agent Elite Agent Collective.
    
    Implements production-grade observability with sub-linear overhead.
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize metrics collection."""
        self.registry = registry or REGISTRY
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Set up all Prometheus metrics."""
        
        # ====================================================================
        # 1. AGENT HEALTH & STATUS METRICS
        # ====================================================================
        
        self.agent_status = Gauge(
            "elite_agent_status",
            "Agent operational status (0=healthy, 1=degraded, 2=failed)",
            labelnames=["agent_id", "agent_name", "tier", "specialization"],
            registry=self.registry
        )
        
        self.agent_availability_percent = Gauge(
            "elite_agent_availability_percent",
            "Agent availability percentage (0-100)",
            labelnames=["agent_id", "agent_name", "tier"],
            registry=self.registry
        )
        
        self.agent_recovery_events = Counter(
            "elite_agent_recovery_events_total",
            "Count of agent recovery events",
            labelnames=["agent_id", "agent_name", "tier"],
            registry=self.registry
        )
        
        self.agent_uptime_seconds = Gauge(
            "elite_agent_uptime_seconds",
            "Agent uptime in seconds",
            labelnames=["agent_id", "agent_name", "tier"],
            registry=self.registry
        )
        
        # ====================================================================
        # 2. TASK MANAGEMENT METRICS
        # ====================================================================
        
        self.tasks_assigned_total = Counter(
            "elite_tasks_assigned_total",
            "Total tasks assigned to agent",
            labelnames=["agent_id", "agent_name", "tier", "task_type"],
            registry=self.registry
        )
        
        self.tasks_completed_total = Counter(
            "elite_tasks_completed_total",
            "Total tasks completed successfully",
            labelnames=["agent_id", "agent_name", "tier", "task_type"],
            registry=self.registry
        )
        
        self.tasks_failed_total = Counter(
            "elite_tasks_failed_total",
            "Total tasks failed",
            labelnames=["agent_id", "agent_name", "tier", "task_type", "error_type"],
            registry=self.registry
        )
        
        self.task_duration_seconds = Histogram(
            "elite_task_duration_seconds",
            "Task execution duration",
            labelnames=["agent_id", "agent_name", "tier", "task_type"],
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0),
            registry=self.registry
        )
        
        # ====================================================================
        # 3. UTILIZATION & CAPACITY METRICS
        # ====================================================================
        
        self.agent_utilization_ratio = Gauge(
            "elite_agent_utilization_ratio",
            "Agent utilization ratio (0.0 to 1.0)",
            labelnames=["agent_id", "agent_name", "tier"],
            registry=self.registry
        )
        
        self.agent_active_tasks = Gauge(
            "elite_agent_active_tasks",
            "Current active tasks for agent",
            labelnames=["agent_id", "agent_name", "tier"],
            registry=self.registry
        )
        
        self.agent_max_concurrent_capacity = Gauge(
            "elite_agent_max_concurrent_capacity",
            "Maximum concurrent task capacity",
            labelnames=["agent_id", "agent_name", "tier"],
            registry=self.registry
        )
        
        self.agent_wait_queue_length = Gauge(
            "elite_agent_wait_queue_length",
            "Tasks waiting in queue",
            labelnames=["agent_id", "agent_name", "tier"],
            registry=self.registry
        )
        
        self.agent_idle_time_percent = Gauge(
            "elite_agent_idle_time_percent",
            "Percentage of time agent is idle",
            labelnames=["agent_id", "agent_name", "tier"],
            registry=self.registry
        )
        
        # ====================================================================
        # 4. PERFORMANCE & QUALITY METRICS
        # ====================================================================
        
        self.agent_success_rate = Gauge(
            "elite_agent_success_rate",
            "Success rate per agent (0-100)",
            labelnames=["agent_id", "agent_name", "tier"],
            registry=self.registry
        )
        
        self.agent_error_rate = Gauge(
            "elite_agent_error_rate",
            "Error rate per agent (0-100)",
            labelnames=["agent_id", "agent_name", "tier"],
            registry=self.registry
        )
        
        self.agent_timeout_rate = Gauge(
            "elite_agent_timeout_rate",
            "Task timeout rate per agent (0-100)",
            labelnames=["agent_id", "agent_name", "tier"],
            registry=self.registry
        )
        
        self.agent_retry_rate = Gauge(
            "elite_agent_retry_rate",
            "Retry necessity rate per agent (0-100)",
            labelnames=["agent_id", "agent_name", "tier"],
            registry=self.registry
        )
        
        # ====================================================================
        # 5. COLLECTIVE-LEVEL METRICS
        # ====================================================================
        
        self.collective_utilization_ratio = Gauge(
            "elite_collective_utilization_ratio",
            "Aggregate utilization across all agents",
            labelnames=["tier"],
            registry=self.registry
        )
        
        self.collective_active_tasks = Gauge(
            "elite_collective_active_tasks",
            "Total active tasks across collective",
            labelnames=["tier"],
            registry=self.registry
        )
        
        self.collective_throughput_tasks_per_sec = Gauge(
            "elite_collective_throughput_tasks_per_sec",
            "Collective throughput (tasks/second)",
            labelnames=["tier"],
            registry=self.registry
        )
        
        self.collective_error_rate = Gauge(
            "elite_collective_error_rate",
            "Collective error rate (0-100)",
            labelnames=["tier"],
            registry=self.registry
        )
        
        self.collective_coordination_effectiveness = Gauge(
            "elite_collective_coordination_effectiveness",
            "Coordination effectiveness score (0-100)",
            labelnames=["tier"],
            registry=self.registry
        )
        
        # ====================================================================
        # 6. CROSS-AGENT COLLABORATION METRICS
        # ====================================================================
        
        self.agent_handoff_events = Counter(
            "elite_agent_handoff_events_total",
            "Inter-agent task handoff events",
            labelnames=["from_agent", "to_agent", "from_tier", "to_tier"],
            registry=self.registry
        )
        
        self.agent_collaboration_score = Gauge(
            "elite_agent_collaboration_score",
            "Collaboration strength between agents",
            labelnames=["agent_a", "agent_b", "tier"],
            registry=self.registry
        )
        
        self.specialization_overlap = Gauge(
            "elite_specialization_overlap",
            "Specialization overlap ratio",
            labelnames=["agent_id", "agent_name", "specialization"],
            registry=self.registry
        )
        
        self.load_balance_efficiency = Gauge(
            "elite_load_balance_efficiency",
            "Load balancing efficiency (0-100)",
            labelnames=["tier"],
            registry=self.registry
        )
        
        # ====================================================================
        # 7. TIER-BASED PERFORMANCE METRICS
        # ====================================================================
        
        self.tier_utilization_ratio = Gauge(
            "elite_tier_utilization_ratio",
            "Utilization ratio per tier",
            labelnames=["tier", "tier_name"],
            registry=self.registry
        )
        
        self.tier_agent_count = Gauge(
            "elite_tier_agent_count",
            "Number of agents per tier",
            labelnames=["tier", "tier_name"],
            registry=self.registry
        )
        
        self.tier_total_tasks = Counter(
            "elite_tier_total_tasks",
            "Total tasks processed per tier",
            labelnames=["tier", "tier_name"],
            registry=self.registry
        )
        
        self.tier_success_rate = Gauge(
            "elite_tier_success_rate",
            "Success rate per tier (0-100)",
            labelnames=["tier", "tier_name"],
            registry=self.registry
        )
        
        self.tier_throughput = Gauge(
            "elite_tier_throughput_tasks_per_sec",
            "Throughput per tier (tasks/second)",
            labelnames=["tier", "tier_name"],
            registry=self.registry
        )
        
        # ====================================================================
        # 8. META-INTELLIGENCE METRICS
        # ====================================================================
        
        self.agent_learning_rate = Gauge(
            "elite_agent_learning_rate",
            "Agent improvement rate (0-1)",
            labelnames=["agent_id", "agent_name", "tier"],
            registry=self.registry
        )
        
        self.breakthrough_discoveries_total = Counter(
            "elite_breakthrough_discoveries_total",
            "Total breakthrough discoveries",
            labelnames=["agent_id", "agent_name", "tier", "specialization"],
            registry=self.registry
        )
        
        self.knowledge_sharing_events = Counter(
            "elite_knowledge_sharing_events_total",
            "Knowledge sharing events between agents",
            labelnames=["from_agent", "to_agent", "from_tier", "to_tier"],
            registry=self.registry
        )
        
        self.collective_intelligence_score = Gauge(
            "elite_collective_intelligence_score",
            "Overall collective intelligence indicator (0-100)",
            labelnames=["tier"],
            registry=self.registry
        )
        
        self.mnemonic_memory_fitness = Gauge(
            "elite_mnemonic_memory_fitness",
            "Average fitness of stored experiences",
            labelnames=["agent_id", "agent_name", "tier"],
            registry=self.registry
        )
        
        self.retrieval_efficiency = Gauge(
            "elite_retrieval_efficiency_percent",
            "Memory retrieval efficiency (0-100)",
            labelnames=["retrieval_type", "tier"],
            registry=self.registry
        )


# ============================================================================
# PROMETHEUS QUERIES TEMPLATES
# ============================================================================

class PrometheusQueries:
    """
    Pre-built Prometheus queries for common monitoring scenarios.
    """
    
    # ========================================================================
    # INDIVIDUAL AGENT DASHBOARDS
    # ========================================================================
    
    @staticmethod
    def agent_status_overview(agent_id: str) -> str:
        """Query for agent status overview."""
        return f"""
        # Agent Status Overview
        {agent_id}_status = elite_agent_status{{agent_id="{agent_id}"}}
        
        # Agent Availability
        {agent_id}_availability = elite_agent_availability_percent{{agent_id="{agent_id}"}}
        
        # Current Utilization
        {agent_id}_utilization = elite_agent_utilization_ratio{{agent_id="{agent_id}"}}
        
        # Active Tasks
        {agent_id}_active = elite_agent_active_tasks{{agent_id="{agent_id}"}}
        """
    
    @staticmethod
    def agent_performance_metrics(agent_id: str) -> str:
        """Query for agent performance metrics."""
        return f"""
        # Success Rate
        success_rate_{agent_id} = elite_agent_success_rate{{agent_id="{agent_id}"}}
        
        # Error Rate
        error_rate_{agent_id} = elite_agent_error_rate{{agent_id="{agent_id}"}}
        
        # Timeout Rate
        timeout_rate_{agent_id} = elite_agent_timeout_rate{{agent_id="{agent_id}"}}
        
        # Average Task Duration (p50, p95, p99)
        avg_duration_p50_{agent_id} = histogram_quantile(0.5, rate(
            elite_task_duration_seconds_bucket{{agent_id="{agent_id}"}}[5m]))
        avg_duration_p95_{agent_id} = histogram_quantile(0.95, rate(
            elite_task_duration_seconds_bucket{{agent_id="{agent_id}"}}[5m]))
        avg_duration_p99_{agent_id} = histogram_quantile(0.99, rate(
            elite_task_duration_seconds_bucket{{agent_id="{agent_id}"}}[5m]))
        """
    
    @staticmethod
    def agent_task_distribution(agent_id: str) -> str:
        """Query for agent task distribution."""
        return f"""
        # Tasks by Type
        tasks_by_type_{agent_id} = rate(
            elite_tasks_assigned_total{{agent_id="{agent_id}"}}[5m])
        
        # Completion Rate
        completion_rate_{agent_id} = rate(
            elite_tasks_completed_total{{agent_id="{agent_id}"}}[5m])
        
        # Failure Rate
        failure_rate_{agent_id} = rate(
            elite_tasks_failed_total{{agent_id="{agent_id}"}}[5m])
        """
    
    # ========================================================================
    # COLLECTIVE HEALTH DASHBOARD
    # ========================================================================
    
    @staticmethod
    def collective_health_overview() -> str:
        """Query for overall collective health."""
        return """
        # Overall Collective Utilization
        collective_utilization = avg(elite_agent_utilization_ratio)
        
        # Total Active Tasks (All Agents)
        total_active_tasks = sum(elite_agent_active_tasks)
        
        # Collective Success Rate
        collective_success = avg(elite_agent_success_rate)
        
        # Collective Error Rate
        collective_errors = avg(elite_agent_error_rate)
        
        # Agent Health Status Distribution
        healthy_agents = count(elite_agent_status == 0)
        degraded_agents = count(elite_agent_status == 1)
        failed_agents = count(elite_agent_status == 2)
        """
    
    @staticmethod
    def collective_throughput() -> str:
        """Query for collective throughput."""
        return """
        # Total Throughput (tasks/sec)
        collective_throughput = sum(rate(
            elite_tasks_completed_total[5m]))
        
        # Throughput by Tier
        tier_throughput = sum by (tier) (rate(
            elite_tasks_completed_total[5m]))
        
        # Peak Throughput (5-min window)
        peak_throughput_5m = max(sum(rate(
            elite_tasks_completed_total[1m])))
        """
    
    # ========================================================================
    # AGENT COMPARISON VIEWS
    # ========================================================================
    
    @staticmethod
    def agent_comparison_table() -> str:
        """Query for agent comparison table."""
        return """
        # Multi-Agent Comparison
        comparison_data = (
            on(agent_id) group_left()
            elite_agent_utilization_ratio,
            elite_agent_success_rate,
            elite_agent_error_rate,
            elite_agent_active_tasks,
            elite_agent_uptime_seconds
        )
        """
    
    @staticmethod
    def agent_efficiency_ranking() -> str:
        """Query for agent efficiency ranking."""
        return """
        # Efficiency Score (Success Rate - Error Rate)
        efficiency_score = (
            elite_agent_success_rate - elite_agent_error_rate
        ) / 100 * elite_agent_utilization_ratio
        
        # Rank by efficiency
        top_performers = topk(10, efficiency_score)
        """
    
    # ========================================================================
    # TIER PERFORMANCE ANALYSIS
    # ========================================================================
    
    @staticmethod
    def tier_performance_analysis(tier: str) -> str:
        """Query for tier-specific performance."""
        return f"""
        # Tier {tier} Utilization
        tier_{tier}_utilization = avg(
            elite_agent_utilization_ratio{{tier="{tier}"}})
        
        # Tier {tier} Throughput
        tier_{tier}_throughput = sum(rate(
            elite_tasks_completed_total{{tier="{tier}"}}[5m]))
        
        # Tier {tier} Success Rate
        tier_{tier}_success = avg(
            elite_agent_success_rate{{tier="{tier}"}})
        
        # Tier {tier} Load Balance
        tier_{tier}_load_balance = elite_load_balance_efficiency{{tier="{tier}"}}
        """
    
    @staticmethod
    def tier_comparison() -> str:
        """Query for tier comparison."""
        return """
        # All Tiers Side-by-Side
        tier_comparison = (
            on() group_left()
            elite_tier_utilization_ratio,
            elite_tier_success_rate,
            elite_tier_throughput
        )
        """
    
    # ========================================================================
    # SPECIALIZATION TRACKING
    # ========================================================================
    
    @staticmethod
    def specialization_performance(spec: str) -> str:
        """Query for specialization-specific performance."""
        return f"""
        # Performance for {spec} Specialization
        {spec}_agents = count(elite_agent_status{{specialization="{spec}"}})
        {spec}_utilization = avg(elite_agent_utilization_ratio{{specialization="{spec}"}})
        {spec}_success = avg(elite_agent_success_rate{{specialization="{spec}"}})
        {spec}_throughput = sum(rate(
            elite_tasks_completed_total{{specialization="{spec}"}}[5m]))
        """
    
    @staticmethod
    def specialization_overlap_analysis() -> str:
        """Query for specialization overlap."""
        return """
        # Specialization Overlap Scores
        overlap_matrix = elite_specialization_overlap
        
        # Overlapping Agents by Specialization
        overlap_by_spec = group by(specialization) (
            elite_specialization_overlap > 0.5)
        """
    
    # ========================================================================
    # COLLABORATION & HANDOFF TRACKING
    # ========================================================================
    
    @staticmethod
    def collaboration_network() -> str:
        """Query for collaboration network."""
        return """
        # Handoff Events Between Agents
        handoff_events = rate(elite_agent_handoff_events_total[5m])
        
        # Collaboration Scores
        collab_scores = elite_agent_collaboration_score
        
        # Most Collaborative Agents
        top_collaborators = topk(10, sum by(agent_a) (
            elite_agent_handoff_events_total))
        """
    
    @staticmethod
    def inter_tier_handoffs() -> str:
        """Query for inter-tier handoffs."""
        return """
        # Handoffs Between Tiers
        inter_tier_handoffs = sum by(from_tier, to_tier) (
            rate(elite_agent_handoff_events_total[5m]))
        
        # Handoff Success Rate (by tier pair)
        handoff_success = sum by(from_tier, to_tier) (
            rate(elite_agent_handoff_events_total[5m]))
        """
    
    # ========================================================================
    # META-INTELLIGENCE TRACKING
    # ========================================================================
    
    @staticmethod
    def learning_and_improvement() -> str:
        """Query for agent learning rates."""
        return """
        # Learning Rate by Agent
        learning_rates = elite_agent_learning_rate
        
        # Breakthrough Discoveries
        breakthroughs = increase(
            elite_breakthrough_discoveries_total[24h])
        
        # Knowledge Sharing Events (24h)
        knowledge_sharing_24h = increase(
            elite_knowledge_sharing_events_total[24h])
        """
    
    @staticmethod
    def mnemonic_memory_health() -> str:
        """Query for memory system health."""
        return """
        # Average Memory Fitness
        avg_memory_fitness = avg(elite_mnemonic_memory_fitness)
        
        # Retrieval Efficiency by Type
        retrieval_efficiency = elite_retrieval_efficiency_percent
        
        # Memory System Performance
        memory_health_score = (
            avg(elite_mnemonic_memory_fitness) + 
            avg(elite_retrieval_efficiency_percent)
        ) / 2
        """
    
    @staticmethod
    def collective_intelligence_score() -> str:
        """Query for overall collective intelligence."""
        return """
        # Collective Intelligence Score
        collective_intelligence = avg(
            elite_collective_intelligence_score)
        
        # Intelligence Trend (7-day)
        intelligence_trend = avg_over_time(
            elite_collective_intelligence_score[7d])
        
        # Tier Intelligence Comparison
        tier_intelligence = elite_collective_intelligence_score
        """


# ============================================================================
# ALERT RULES
# ============================================================================

ALERT_RULES = """
# Elite Agent Collective Alert Rules
# ===================================

groups:
  - name: elite_agent_health
    interval: 30s
    rules:
      # Agent Failure Alert
      - alert: AgentFailed
        expr: elite_agent_status == 2
        for: 2m
        labels:
          severity: critical
          component: agent
        annotations:
          summary: "Agent {{ $labels.agent_name }} ({{ $labels.agent_id }}) has failed"
          description: "Agent is not responding. Status: FAILED. Tier: {{ $labels.tier }}"
      
      # Agent Degradation Alert
      - alert: AgentDegraded
        expr: elite_agent_status == 1
        for: 5m
        labels:
          severity: warning
          component: agent
        annotations:
          summary: "Agent {{ $labels.agent_name }} is degraded"
          description: "Agent performance is degraded. Utilization: {{ $value }}"
      
      # Low Availability Alert
      - alert: LowAgentAvailability
        expr: elite_agent_availability_percent < 80
        for: 10m
        labels:
          severity: warning
          component: agent
        annotations:
          summary: "Agent {{ $labels.agent_name }} availability below 80%"
          description: "Current availability: {{ $value }}%"

  - name: elite_agent_performance
    interval: 30s
    rules:
      # High Error Rate Alert
      - alert: HighErrorRate
        expr: elite_agent_error_rate > 20
        for: 5m
        labels:
          severity: critical
          component: agent
        annotations:
          summary: "Agent {{ $labels.agent_name }} has high error rate"
          description: "Error rate: {{ $value }}%"
      
      # High Timeout Rate Alert
      - alert: HighTimeoutRate
        expr: elite_agent_timeout_rate > 10
        for: 5m
        labels:
          severity: warning
          component: agent
        annotations:
          summary: "Agent {{ $labels.agent_name }} has high timeout rate"
          description: "Timeout rate: {{ $value }}%"
      
      # Low Success Rate Alert
      - alert: LowSuccessRate
        expr: elite_agent_success_rate < 50
        for: 10m
        labels:
          severity: critical
          component: agent
        annotations:
          summary: "Agent {{ $labels.agent_name }} success rate below 50%"
          description: "Success rate: {{ $value }}%"

  - name: elite_agent_utilization
    interval: 30s
    rules:
      # Over-utilized Agent
      - alert: AgentOverUtilized
        expr: elite_agent_utilization_ratio > 0.95
        for: 5m
        labels:
          severity: warning
          component: capacity
        annotations:
          summary: "Agent {{ $labels.agent_name }} is over-utilized"
          description: "Utilization: {{ $value | humanizePercentage }}"
      
      # Queue Backlog Alert
      - alert: LargeQueueBacklog
        expr: elite_agent_wait_queue_length > 100
        for: 2m
        labels:
          severity: warning
          component: capacity
        annotations:
          summary: "Large queue backlog for agent {{ $labels.agent_name }}"
          description: "Queue length: {{ $value }} tasks"

  - name: elite_collective_health
    interval: 30s
    rules:
      # Collective Error Rate High
      - alert: CollectiveErrorRateHigh
        expr: elite_collective_error_rate > 15
        for: 5m
        labels:
          severity: critical
          component: collective
        annotations:
          summary: "Tier {{ $labels.tier }} collective error rate is high"
          description: "Error rate: {{ $value }}%"
      
      # Collective Utilization Imbalance
      - alert: UtilizationImbalance
        expr: |
          (max(elite_agent_utilization_ratio) - 
           min(elite_agent_utilization_ratio)) > 0.5
        for: 10m
        labels:
          severity: warning
          component: load_balance
        annotations:
          summary: "Significant utilization imbalance detected"
          description: "Utilization spread: {{ $value | humanizePercentage }}"
      
      # Low Coordination Effectiveness
      - alert: LowCoordinationEffectiveness
        expr: elite_collective_coordination_effectiveness < 60
        for: 15m
        labels:
          severity: warning
          component: coordination
        annotations:
          summary: "Tier {{ $labels.tier }} coordination effectiveness low"
          description: "Score: {{ $value }}%"

  - name: elite_meta_intelligence
    interval: 60s
    rules:
      # Breakthrough Discovery Drought
      - alert: BreakthroughDrought
        expr: increase(elite_breakthrough_discoveries_total[24h]) == 0
        for: 24h
        labels:
          severity: info
          component: meta_learning
        annotations:
          summary: "No breakthroughs discovered in past 24 hours"
          description: "Agent {{ $labels.agent_name }} may need optimization"
      
      # Low Memory Fitness
      - alert: LowMemoryFitness
        expr: elite_mnemonic_memory_fitness < 0.5
        for: 30m
        labels:
          severity: warning
          component: memory
        annotations:
          summary: "Agent {{ $labels.agent_name }} memory fitness is low"
          description: "Fitness score: {{ $value }}"
      
      # Poor Retrieval Efficiency
      - alert: PoorRetrievalEfficiency
        expr: elite_retrieval_efficiency_percent < 70
        for: 15m
        labels:
          severity: warning
          component: memory
        annotations:
          summary: "{{ $labels.retrieval_type }} retrieval efficiency is low"
          description: "Efficiency: {{ $value }}%"
"""


# ============================================================================
# OPTIMIZATION OPPORTUNITIES ANALYZER
# ============================================================================

@dataclass
class OptimizationOpportunity:
    """Represents an optimization opportunity."""
    title: str
    description: str
    affected_agents: List[str]
    potential_improvement: float  # 0.0 to 1.0
    priority: str  # "critical", "high", "medium", "low"
    recommendation: str


class OptimizationAnalyzer:
    """
    Analyzes metrics to identify optimization opportunities.
    """
    
    @staticmethod
    def analyze_utilization_imbalance(
        utilization_by_agent: Dict[str, float]
    ) -> Optional[OptimizationOpportunity]:
        """Detect and suggest fixes for utilization imbalance."""
        if not utilization_by_agent:
            return None
        
        values = list(utilization_by_agent.values())
        max_util = max(values)
        min_util = min(values)
        spread = max_util - min_util
        
        if spread > 0.4:
            return OptimizationOpportunity(
                title="Utilization Imbalance Detected",
                description=f"Large spread in agent utilization (max: {max_util:.2f}, min: {min_util:.2f})",
                affected_agents=[
                    agent for agent, util in utilization_by_agent.items()
                    if util > 0.8 or util < 0.2
                ],
                potential_improvement=spread / 2,
                priority="high",
                recommendation=(
                    "Implement dynamic task routing to balance load. "
                    "Route tasks away from over-utilized agents. "
                    "Consider task type specialization."
                )
            )
        return None
    
    @staticmethod
    def analyze_error_patterns(
        error_rates_by_agent: Dict[str, float]
    ) -> Optional[OptimizationOpportunity]:
        """Detect error patterns and suggest improvements."""
        high_error_agents = {
            agent: rate for agent, rate in error_rates_by_agent.items()
            if rate > 0.15
        }
        
        if high_error_agents:
            return OptimizationOpportunity(
                title="High Error Rate Detected",
                description=f"{len(high_error_agents)} agents with error rate > 15%",
                affected_agents=list(high_error_agents.keys()),
                potential_improvement=sum(high_error_agents.values()) / len(high_error_agents) / 2,
                priority="critical",
                recommendation=(
                    "Review error logs for root causes. "
                    "Consider circuit breaker pattern. "
                    "Implement retry with exponential backoff. "
                    "Add fallback mechanisms."
                )
            )
        return None
    
    @staticmethod
    def analyze_collaboration_gaps(
        handoff_matrix: Dict[Tuple[str, str], int]
    ) -> List[OptimizationOpportunity]:
        """Detect collaboration gaps between tiers."""
        opportunities = []
        
        # Identify sparse collaboration patterns
        total_handoffs = sum(handoff_matrix.values())
        average_handoff = total_handoffs / len(handoff_matrix) if handoff_matrix else 0
        
        sparse_pairs = {
            pair: count for pair, count in handoff_matrix.items()
            if count < average_handoff * 0.1
        }
        
        if sparse_pairs:
            opportunities.append(OptimizationOpportunity(
                title="Sparse Inter-Tier Collaboration",
                description=f"{len(sparse_pairs)} agent pairs have minimal handoffs",
                affected_agents=list(set(
                    agent for pair in sparse_pairs.keys() for agent in pair
                )),
                potential_improvement=0.15,
                priority="medium",
                recommendation=(
                    "Review specialization boundaries. "
                    "Consider cross-tier task routing. "
                    "Implement knowledge sharing mechanisms."
                )
            ))
        
        return opportunities


# ============================================================================
# METRICS EXPORTER
# ============================================================================

def export_metrics(registry: CollectorRegistry = REGISTRY) -> bytes:
    """Export metrics in Prometheus format."""
    return generate_latest(registry)


# ============================================================================
# HELPER FUNCTIONS FOR AGENT UPDATES
# ============================================================================

def get_agent_info(agent_id: str) -> Tuple[AgentTier, AgentSpecialization]:
    """Get tier and specialization for agent."""
    return AGENT_REGISTRY.get(
        agent_id,
        (AgentTier.TIER_1_FOUNDATIONAL, AgentSpecialization.CS_ENGINEERING)
    )


def get_tier_name(tier: AgentTier) -> str:
    """Get human-readable tier name."""
    names = {
        AgentTier.TIER_1_FOUNDATIONAL: "Tier 1 - Foundational",
        AgentTier.TIER_2_SPECIALISTS: "Tier 2 - Specialists",
        AgentTier.TIER_3_INNOVATORS: "Tier 3 - Innovators",
        AgentTier.TIER_4_META: "Tier 4 - Meta",
        AgentTier.TIER_5_DOMAIN: "Tier 5 - Domain",
        AgentTier.TIER_6_EMERGING: "Tier 6 - Emerging",
        AgentTier.TIER_7_HUMAN_CENTRIC: "Tier 7 - Human-Centric",
        AgentTier.TIER_8_ENTERPRISE: "Tier 8 - Enterprise",
    }
    return names.get(tier, "Unknown")


if __name__ == "__main__":
    # Example usage
    metrics = EliteAgentMetrics()
    
    # Print available metrics
    print("Elite Agent Collective Metrics Initialized")
    print(f"Total Agents: {len(AGENT_REGISTRY)}")
    print(f"Total Tiers: {len(AgentTier)}")
    print(f"Total Specializations: {len(AgentSpecialization)}")
