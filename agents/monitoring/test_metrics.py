"""
Comprehensive Test Suite for Elite Agent Collective Metrics
18A-6.3: Agent Health, Utilization, Collaboration, and Collective Intelligence Tracking
"""

import pytest
import time
import asyncio
from unittest.mock import Mock, patch, MagicMock
from prometheus_client import CollectorRegistry, generate_latest

from agents.monitoring.metrics import (
    # Individual Agent Metrics
    agent_status,
    agent_active_tasks,
    agent_availability_ratio,
    
    # Agent Task Metrics
    agent_tasks_total,
    agent_tasks_completed,
    agent_tasks_failed,
    agent_task_duration_seconds,
    
    # Agent Performance Metrics
    agent_success_rate,
    agent_error_rate,
    agent_timeout_rate,
    agent_average_task_duration,
    agent_p95_task_duration,
    
    # Agent Utilization Metrics
    agent_utilization_ratio,
    agent_max_capacity,
    agent_queue_length,
    agent_idle_time_seconds,
    
    # Agent Specialization Metrics
    agent_specialization,
    agent_task_type_distribution,
    
    # Agent Recovery and Resilience
    agent_recovery_events_total,
    agent_restart_count,
    agent_mttr_seconds,
    
    # Collective Metrics
    collective_total_agents,
    collective_healthy_agents,
    collective_degraded_agents,
    collective_failed_agents,
    collective_total_active_tasks,
    collective_utilization_ratio,
    collective_throughput_tasks_per_second,
    collective_error_rate,
    collective_success_rate,
    
    # Tier-Specific Metrics
    tier_health_score,
    tier_utilization,
    tier_task_count,
    tier_error_rate,
    
    # Inter-Agent Collaboration
    agent_collaboration_events_total,
    agent_handoff_total,
    agent_communication_latency_seconds,
    
    # Collective Intelligence
    collective_intelligence_score,
    collective_breakthrough_count,
    agent_knowledge_sharing_events,
    
    # Decorators and Helpers
    track_agent_task,
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


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def clean_registry():
    """Clean registry for isolated metric tests"""
    from prometheus_client import REGISTRY
    
    # Store original collectors
    original_collectors = list(REGISTRY._collector_to_names.keys())
    
    yield REGISTRY
    
    # Restore state
    collectors_to_remove = set(REGISTRY._collector_to_names.keys()) - set(original_collectors)
    for collector in collectors_to_remove:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


@pytest.fixture
def sample_agent():
    """Sample agent configuration"""
    return {
        'agent_id': 'APEX-001',
        'agent_name': 'APEX',
        'tier': 'TIER_1'
    }


@pytest.fixture
def all_agents(sample_agent):
    """All 40 Elite Agents"""
    tiers = {
        'TIER_1': ['APEX', 'CIPHER', 'ARCHITECT', 'AXIOM', 'VELOCITY'],
        'TIER_2': ['QUANTUM', 'TENSOR', 'FORTRESS', 'NEURAL', 'CRYPTO', 'FLUX', 'PRISM', 'SYNAPSE', 'CORE', 'HELIX', 'VANGUARD', 'ECLIPSE'],
        'TIER_3': ['NEXUS', 'GENESIS'],
        'TIER_4': ['OMNISCIENT'],
        'TIER_5': ['ATLAS', 'FORGE', 'SENTRY', 'VERTEX', 'STREAM'],
        'TIER_6': ['PHOTON', 'LATTICE', 'MORPH', 'PHANTOM', 'ORBIT'],
        'TIER_7': ['CANVAS', 'LINGUA', 'SCRIBE', 'MENTOR', 'BRIDGE'],
        'TIER_8': ['AEGIS', 'LEDGER', 'PULSE', 'ARBITER', 'ORACLE'],
    }
    
    agents = []
    agent_id = 1
    for tier, names in tiers.items():
        for name in names:
            agents.append({
                'agent_id': f'{name}-{agent_id:03d}',
                'agent_name': name,
                'tier': tier
            })
            agent_id += 1
    
    return agents


# ============================================================================
# TEST INDIVIDUAL AGENT HEALTH TRACKING
# ============================================================================

class TestIndividualAgentHealth:
    """Tests for individual agent health status tracking"""
    
    def test_agent_status_healthy(self, clean_registry, sample_agent):
        """Test agent status as healthy (0)"""
        update_agent_status(
            sample_agent['agent_id'],
            sample_agent['agent_name'],
            sample_agent['tier'],
            0  # Healthy
        )
        metrics = generate_latest(clean_registry)
        assert b'agents_status{' in metrics
    
    def test_agent_status_degraded(self, clean_registry, sample_agent):
        """Test agent status as degraded (1)"""
        update_agent_status(
            sample_agent['agent_id'],
            sample_agent['agent_name'],
            sample_agent['tier'],
            1  # Degraded
        )
        metrics = generate_latest(clean_registry)
        assert b'agents_status{' in metrics
    
    def test_agent_status_failed(self, clean_registry, sample_agent):
        """Test agent status as failed (2)"""
        update_agent_status(
            sample_agent['agent_id'],
            sample_agent['agent_name'],
            sample_agent['tier'],
            2  # Failed
        )
        metrics = generate_latest(clean_registry)
        assert b'agents_status{' in metrics
    
    def test_agent_availability_ratio(self, clean_registry, sample_agent):
        """Test agent availability ratio"""
        agent_availability_ratio.labels(
            agent_id=sample_agent['agent_id'],
            agent_name=sample_agent['agent_name']
        ).set(0.9999)
        metrics = generate_latest(clean_registry)
        assert b'agents_availability_ratio' in metrics
    
    def test_multiple_agents_health(self, clean_registry, all_agents):
        """Test health tracking for multiple agents"""
        for i, agent in enumerate(all_agents[:5]):  # Test 5 agents
            update_agent_status(
                agent['agent_id'],
                agent['agent_name'],
                agent['tier'],
                i % 3  # Vary health status
            )
        
        metrics = generate_latest(clean_registry)
        assert b'agents_status' in metrics


# ============================================================================
# TEST TASK METRICS AGGREGATION
# ============================================================================

class TestTaskMetricsAggregation:
    """Tests for agent task metrics aggregation"""
    
    def test_tasks_total_counter(self, clean_registry, sample_agent):
        """Test total tasks assigned"""
        agent_tasks_total.labels(
            agent_id=sample_agent['agent_id'],
            agent_name=sample_agent['agent_name'],
            task_type='analysis'
        ).inc()
        metrics = generate_latest(clean_registry)
        assert b'agents_tasks_total' in metrics
    
    def test_tasks_completed_counter(self, clean_registry, sample_agent):
        """Test completed tasks counter"""
        agent_tasks_completed.labels(
            agent_id=sample_agent['agent_id'],
            agent_name=sample_agent['agent_name'],
            task_type='analysis'
        ).inc(5)
        metrics = generate_latest(clean_registry)
        assert b'agents_tasks_completed' in metrics
    
    def test_tasks_failed_counter(self, clean_registry, sample_agent):
        """Test failed tasks counter"""
        agent_tasks_failed.labels(
            agent_id=sample_agent['agent_id'],
            agent_name=sample_agent['agent_name'],
            error_type='timeout'
        ).inc(2)
        metrics = generate_latest(clean_registry)
        assert b'agents_tasks_failed' in metrics
    
    def test_task_type_distribution(self, clean_registry, sample_agent):
        """Test task type distribution across agent"""
        task_types = ['analysis', 'optimization', 'verification', 'synthesis']
        for task_type in task_types:
            agent_tasks_total.labels(
                agent_id=sample_agent['agent_id'],
                agent_name=sample_agent['agent_name'],
                task_type=task_type
            ).inc(10)
        
        metrics = generate_latest(clean_registry)
        assert b'agents_tasks_total' in metrics
    
    def test_error_types_tracked_separately(self, clean_registry, sample_agent):
        """Test different error types are tracked"""
        error_types = ['timeout', 'oom', 'exception', 'constraint_violation']
        for error_type in error_types:
            agent_tasks_failed.labels(
                agent_id=sample_agent['agent_id'],
                agent_name=sample_agent['agent_name'],
                error_type=error_type
            ).inc()
        
        metrics = generate_latest(clean_registry)
        metrics_str = metrics.decode('utf-8')
        for error_type in error_types:
            assert f'error_type="{error_type}"' in metrics_str


# ============================================================================
# TEST UTILIZATION CALCULATION VERIFICATION
# ============================================================================

class TestUtilizationCalculation:
    """Tests for agent utilization calculation"""
    
    def test_utilization_ratio_calculation(self, clean_registry, sample_agent):
        """Test utilization ratio is calculated correctly"""
        active_tasks = 8
        max_capacity = 10
        expected_utilization = active_tasks / max_capacity
        
        update_agent_utilization(
            sample_agent['agent_id'],
            sample_agent['agent_name'],
            expected_utilization,
            max_capacity,
            2  # queue_length
        )
        
        metrics = generate_latest(clean_registry)
        assert b'agents_utilization_ratio' in metrics
    
    def test_utilization_range_validation(self, clean_registry, sample_agent):
        """Test utilization stays within 0.0-1.0 range"""
        for utilization in [0.0, 0.25, 0.50, 0.75, 1.0]:
            agent_utilization_ratio.labels(
                agent_id=sample_agent['agent_id'],
                agent_name=sample_agent['agent_name']
            ).set(utilization)
        
        metrics = generate_latest(clean_registry)
        assert b'agents_utilization_ratio' in metrics
    
    def test_queue_length_tracking(self, clean_registry, sample_agent):
        """Test queue length is tracked"""
        agent_queue_length.labels(
            agent_id=sample_agent['agent_id'],
            agent_name=sample_agent['agent_name']
        ).set(15)
        metrics = generate_latest(clean_registry)
        assert b'agents_queue_length' in metrics
    
    def test_max_capacity_tracking(self, clean_registry, sample_agent):
        """Test max capacity is tracked"""
        agent_max_capacity.labels(
            agent_id=sample_agent['agent_id'],
            agent_name=sample_agent['agent_name']
        ).set(50)
        metrics = generate_latest(clean_registry)
        assert b'agents_max_capacity' in metrics
    
    def test_active_tasks_gauge(self, clean_registry, sample_agent):
        """Test active tasks gauge"""
        agent_active_tasks.labels(
            agent_id=sample_agent['agent_id'],
            agent_name=sample_agent['agent_name']
        ).set(12)
        metrics = generate_latest(clean_registry)
        assert b'agents_active_tasks' in metrics


# ============================================================================
# TEST FAILURE RATE COMPUTATION
# ============================================================================

class TestFailureRateComputation:
    """Tests for agent failure rate calculation"""
    
    def test_error_rate_calculation(self, clean_registry, sample_agent):
        """Test error rate is calculated correctly"""
        success_rate = 0.95
        error_rate = 1.0 - success_rate
        
        update_agent_rates(
            sample_agent['agent_id'],
            sample_agent['agent_name'],
            success_rate,
            error_rate,
            0.01  # timeout_rate
        )
        
        metrics = generate_latest(clean_registry)
        assert b'agents_error_rate' in metrics
    
    def test_success_rate_metric(self, clean_registry, sample_agent):
        """Test success rate metric"""
        agent_success_rate.labels(
            agent_id=sample_agent['agent_id'],
            agent_name=sample_agent['agent_name']
        ).set(0.98)
        metrics = generate_latest(clean_registry)
        assert b'agents_success_rate' in metrics
    
    def test_timeout_rate_metric(self, clean_registry, sample_agent):
        """Test timeout rate metric"""
        agent_timeout_rate.labels(
            agent_id=sample_agent['agent_id'],
            agent_name=sample_agent['agent_name']
        ).set(0.02)
        metrics = generate_latest(clean_registry)
        assert b'agents_timeout_rate' in metrics
    
    def test_rate_consistency(self, clean_registry, sample_agent):
        """Test rates sum to 1.0 (success + error + timeout ≈ 1.0)"""
        success = 0.95
        error = 0.03
        timeout = 0.02
        
        update_agent_rates(
            sample_agent['agent_id'],
            sample_agent['agent_name'],
            success,
            error,
            timeout
        )
        
        # Rates should approximately sum to 1.0
        total = success + error + timeout
        assert abs(total - 1.0) < 0.01


# ============================================================================
# TEST RECOVERY EVENT TRACKING
# ============================================================================

class TestRecoveryEventTracking:
    """Tests for agent recovery and resilience tracking"""
    
    def test_recovery_event_counter(self, clean_registry, sample_agent):
        """Test recovery event is tracked"""
        record_agent_recovery(
            sample_agent['agent_id'],
            sample_agent['agent_name'],
            'circuit_breaker_reset'
        )
        metrics = generate_latest(clean_registry)
        assert b'agents_recovery_events_total' in metrics
    
    def test_multiple_recovery_types(self, clean_registry, sample_agent):
        """Test multiple recovery types"""
        recovery_types = ['circuit_breaker_reset', 'timeout_recovery', 'task_retry', 'failover']
        for recovery_type in recovery_types:
            agent_recovery_events_total.labels(
                agent_id=sample_agent['agent_id'],
                agent_name=sample_agent['agent_name'],
                recovery_type=recovery_type
            ).inc()
        
        metrics = generate_latest(clean_registry)
        assert b'agents_recovery_events_total' in metrics
    
    def test_restart_counter(self, clean_registry, sample_agent):
        """Test agent restart counter"""
        agent_restart_count.labels(
            agent_id=sample_agent['agent_id'],
            agent_name=sample_agent['agent_name']
        ).inc()
        metrics = generate_latest(clean_registry)
        assert b'agents_restart_count' in metrics
    
    def test_mttr_tracking(self, clean_registry, sample_agent):
        """Test Mean Time To Recovery tracking"""
        agent_mttr_seconds.labels(
            agent_id=sample_agent['agent_id'],
            agent_name=sample_agent['agent_name']
        ).set(5.23)
        metrics = generate_latest(clean_registry)
        assert b'agents_mttr_seconds' in metrics
    
    def test_idle_time_accumulation(self, clean_registry, sample_agent):
        """Test idle time accumulation"""
        agent_idle_time_seconds.labels(
            agent_id=sample_agent['agent_id'],
            agent_name=sample_agent['agent_name']
        ).inc(10.0)
        agent_idle_time_seconds.labels(
            agent_id=sample_agent['agent_id'],
            agent_name=sample_agent['agent_name']
        ).inc(5.0)
        metrics = generate_latest(clean_registry)
        assert b'agents_idle_time_seconds' in metrics


# ============================================================================
# TEST COLLECTIVE HEALTH AGGREGATION
# ============================================================================

class TestCollectiveHealthAggregation:
    """Tests for collective health metrics aggregation"""
    
    def test_collective_total_agents(self, clean_registry):
        """Test total agents in collective"""
        collective_total_agents.set(40)
        metrics = generate_latest(clean_registry)
        assert b'agents_collective_total' in metrics
    
    def test_collective_healthy_count(self, clean_registry):
        """Test healthy agents count"""
        collective_healthy_agents.set(38)
        metrics = generate_latest(clean_registry)
        assert b'agents_collective_healthy' in metrics
    
    def test_collective_degraded_count(self, clean_registry):
        """Test degraded agents count"""
        collective_degraded_agents.set(1)
        metrics = generate_latest(clean_registry)
        assert b'agents_collective_degraded' in metrics
    
    def test_collective_failed_count(self, clean_registry):
        """Test failed agents count"""
        collective_failed_agents.set(1)
        metrics = generate_latest(clean_registry)
        assert b'agents_collective_failed' in metrics
    
    def test_update_collective_health_helper(self, clean_registry):
        """Test update_collective_health helper function"""
        update_collective_health(40, 38, 1, 1)
        metrics = generate_latest(clean_registry)
        assert b'agents_collective' in metrics
    
    def test_collective_active_tasks(self, clean_registry):
        """Test collective active tasks"""
        collective_total_active_tasks.set(250)
        metrics = generate_latest(clean_registry)
        assert b'agents_collective_active_tasks' in metrics
    
    def test_collective_utilization_ratio(self, clean_registry):
        """Test collective utilization ratio"""
        collective_utilization_ratio.set(0.62)
        metrics = generate_latest(clean_registry)
        assert b'agents_collective_utilization_ratio' in metrics
    
    def test_collective_throughput(self, clean_registry):
        """Test collective throughput (tasks/second)"""
        collective_throughput_tasks_per_second.set(125.5)
        metrics = generate_latest(clean_registry)
        assert b'agents_collective_throughput_tasks_per_second' in metrics
    
    def test_collective_error_rate(self, clean_registry):
        """Test collective error rate"""
        collective_error_rate.set(0.02)
        metrics = generate_latest(clean_registry)
        assert b'agents_collective_error_rate' in metrics
    
    def test_collective_success_rate(self, clean_registry):
        """Test collective success rate"""
        collective_success_rate.set(0.98)
        metrics = generate_latest(clean_registry)
        assert b'agents_collective_success_rate' in metrics


# ============================================================================
# TEST TIER-LEVEL PERFORMANCE METRICS
# ============================================================================

class TestTierLevelMetrics:
    """Tests for tier-specific performance metrics"""
    
    def test_tier_health_score(self, clean_registry):
        """Test tier health score"""
        tier_health_score.labels(tier='TIER_1').set(0.95)
        metrics = generate_latest(clean_registry)
        assert b'agents_tier_health_score' in metrics
    
    def test_tier_utilization(self, clean_registry):
        """Test tier utilization metric"""
        tier_utilization.labels(tier='TIER_2').set(0.68)
        metrics = generate_latest(clean_registry)
        assert b'agents_tier_utilization_ratio' in metrics
    
    def test_tier_task_count(self, clean_registry):
        """Test tier cumulative task count"""
        tier_task_count.labels(tier='TIER_3').inc(500)
        metrics = generate_latest(clean_registry)
        assert b'agents_tier_task_count' in metrics
    
    def test_tier_error_rate(self, clean_registry):
        """Test tier error rate"""
        tier_error_rate.labels(tier='TIER_4').set(0.01)
        metrics = generate_latest(clean_registry)
        assert b'agents_tier_error_rate' in metrics
    
    def test_all_tiers_tracked(self, clean_registry):
        """Test all tiers are tracked"""
        tiers = ['TIER_1', 'TIER_2', 'TIER_3', 'TIER_4', 'TIER_5', 'TIER_6', 'TIER_7', 'TIER_8']
        for i, tier in enumerate(tiers):
            update_tier_health(tier, 0.90 + (i * 0.01), 0.50 + (i * 0.05), 0.02)
        
        metrics = generate_latest(clean_registry)
        assert b'agents_tier_health_score' in metrics


# ============================================================================
# TEST INTER-AGENT COLLABORATION TRACKING
# ============================================================================

class TestInterAgentCollaboration:
    """Tests for inter-agent collaboration metrics"""
    
    def test_collaboration_event_tracking(self, clean_registry, sample_agent):
        """Test collaboration event is tracked"""
        record_collaboration(sample_agent['agent_id'], 'CIPHER-001')
        metrics = generate_latest(clean_registry)
        assert b'agents_collaboration_events_total' in metrics
    
    def test_task_handoff_tracking(self, clean_registry, sample_agent):
        """Test task handoff between agents"""
        record_task_handoff(sample_agent['agent_id'], 'ARCHITECT-001')
        metrics = generate_latest(clean_registry)
        assert b'agents_handoff_total' in metrics
    
    def test_communication_latency_histogram(self, clean_registry):
        """Test communication latency histogram"""
        latencies = [0.001, 0.01, 0.05, 0.1, 1.0]
        for latency in latencies:
            agent_communication_latency_seconds.observe(latency)
        
        metrics = generate_latest(clean_registry)
        assert b'agents_communication_latency_seconds' in metrics
    
    def test_cross_tier_collaboration(self, clean_registry):
        """Test collaboration between tiers"""
        record_collaboration('APEX-001', 'TENSOR-001')
        record_collaboration('TENSOR-001', 'APEX-001')
        metrics = generate_latest(clean_registry)
        assert b'agents_collaboration_events_total' in metrics
    
    def test_knowledge_sharing_events(self, clean_registry):
        """Test knowledge sharing between tiers"""
        record_knowledge_sharing('TIER_1', 'TIER_2')
        record_knowledge_sharing('TIER_2', 'TIER_3')
        metrics = generate_latest(clean_registry)
        assert b'agents_knowledge_sharing_events' in metrics


# ============================================================================
# TEST AGENT SPECIALIZATION METRICS
# ============================================================================

class TestAgentSpecialization:
    """Tests for agent specialization tracking"""
    
    def test_specialization_proficiency(self, clean_registry, sample_agent):
        """Test specialization proficiency metric"""
        agent_specialization.labels(
            agent_id=sample_agent['agent_id'],
            agent_name=sample_agent['agent_name'],
            specialization='algorithm_design'
        ).set(0.95)
        metrics = generate_latest(clean_registry)
        assert b'agents_specialization' in metrics
    
    def test_multiple_specializations(self, clean_registry, sample_agent):
        """Test multiple specializations for agent"""
        specializations = ['algorithm_design', 'system_architecture', 'security_analysis', 'performance_optimization']
        for i, spec in enumerate(specializations):
            proficiency = 0.80 + (i * 0.05)
            agent_specialization.labels(
                agent_id=sample_agent['agent_id'],
                agent_name=sample_agent['agent_name'],
                specialization=spec
            ).set(proficiency)
        
        metrics = generate_latest(clean_registry)
        assert b'agents_specialization' in metrics
    
    def test_task_type_distribution_histogram(self, clean_registry, sample_agent):
        """Test task type distribution histogram"""
        task_types = ['analysis', 'optimization', 'verification', 'synthesis']
        for task_type in task_types:
            agent_task_type_distribution.labels(
                agent_id=sample_agent['agent_id'],
                agent_name=sample_agent['agent_name'],
                task_type=task_type
            ).observe(0.25)  # Equal distribution
        
        metrics = generate_latest(clean_registry)
        assert b'agents_task_type_distribution' in metrics


# ============================================================================
# TEST PERFORMANCE PERCENTILES
# ============================================================================

class TestPerformancePercentiles:
    """Tests for performance percentile metrics"""
    
    def test_task_duration_histogram(self, clean_registry, sample_agent):
        """Test task duration histogram"""
        durations = [0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0]
        for duration in durations:
            agent_task_duration_seconds.labels(
                agent_id=sample_agent['agent_id'],
                agent_name=sample_agent['agent_name'],
                task_type='analysis'
            ).observe(duration)
        
        metrics = generate_latest(clean_registry)
        assert b'agents_task_duration_seconds' in metrics
    
    def test_average_task_duration(self, clean_registry, sample_agent):
        """Test average task duration gauge"""
        agent_average_task_duration.labels(
            agent_id=sample_agent['agent_id'],
            agent_name=sample_agent['agent_name']
        ).set(5.42)
        metrics = generate_latest(clean_registry)
        assert b'agents_average_task_duration_seconds' in metrics
    
    def test_p95_task_duration(self, clean_registry, sample_agent):
        """Test P95 task duration gauge"""
        agent_p95_task_duration.labels(
            agent_id=sample_agent['agent_id'],
            agent_name=sample_agent['agent_name']
        ).set(15.32)
        metrics = generate_latest(clean_registry)
        assert b'agents_p95_task_duration_seconds' in metrics


# ============================================================================
# TEST BREAKTHROUGH DISCOVERY DETECTION
# ============================================================================

class TestBreakthroughDiscoveryDetection:
    """Tests for breakthrough discovery tracking"""
    
    def test_breakthrough_counter(self, clean_registry):
        """Test breakthrough discovery counter"""
        record_breakthrough()
        metrics = generate_latest(clean_registry)
        assert b'agents_collective_breakthrough_count' in metrics
    
    def test_multiple_breakthroughs(self, clean_registry):
        """Test multiple breakthrough detections"""
        for _ in range(5):
            record_breakthrough()
        
        metrics = generate_latest(clean_registry)
        assert b'agents_collective_breakthrough_count' in metrics
    
    def test_collective_intelligence_score(self, clean_registry):
        """Test collective intelligence score"""
        update_collective_intelligence(0.875)
        metrics = generate_latest(clean_registry)
        assert b'agents_collective_intelligence_score' in metrics
    
    def test_intelligence_score_range(self, clean_registry):
        """Test intelligence score stays in valid range"""
        for score in [0.0, 0.25, 0.50, 0.75, 1.0]:
            update_collective_intelligence(score)
        
        metrics = generate_latest(clean_registry)
        assert b'agents_collective_intelligence_score' in metrics


# ============================================================================
# TEST DECORATOR
# ============================================================================

class TestAgentTaskDecorator:
    """Tests for track_agent_task decorator"""
    
    @pytest.mark.asyncio
    async def test_decorator_success_path(self, clean_registry, sample_agent):
        """Test decorator tracks successful task"""
        @track_agent_task(sample_agent['agent_id'], sample_agent['agent_name'], 'analysis')
        async def analyze_data():
            await asyncio.sleep(0.001)
            return {'result': 'success'}
        
        result = await analyze_data()
        assert result['result'] == 'success'
        
        metrics = generate_latest(clean_registry)
        assert b'agents_tasks_total' in metrics
    
    @pytest.mark.asyncio
    async def test_decorator_error_tracking(self, clean_registry, sample_agent):
        """Test decorator tracks errors"""
        @track_agent_task(sample_agent['agent_id'], sample_agent['agent_name'], 'optimization')
        async def failing_task():
            await asyncio.sleep(0.001)
            raise ValueError("Task failed")
        
        with pytest.raises(ValueError):
            await failing_task()
        
        metrics = generate_latest(clean_registry)
        assert b'agents_tasks_failed' in metrics


# ============================================================================
# TEST CONCURRENT METRICS ACCESS
# ============================================================================

class TestConcurrentMetricsAccess:
    """Tests for concurrent metric access safety"""
    
    def test_concurrent_task_tracking(self, clean_registry, sample_agent):
        """Test concurrent task counter increments"""
        import threading
        
        def record_tasks():
            for _ in range(50):
                agent_tasks_total.labels(
                    agent_id=sample_agent['agent_id'],
                    agent_name=sample_agent['agent_name'],
                    task_type='analysis'
                ).inc()
        
        threads = [threading.Thread(target=record_tasks) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        metrics = generate_latest(clean_registry)
        assert b'agents_tasks_total' in metrics
    
    def test_concurrent_status_updates(self, clean_registry, all_agents):
        """Test concurrent status updates for multiple agents"""
        import threading
        
        def update_statuses():
            for agent in all_agents[:10]:
                agent_status.labels(
                    agent_id=agent['agent_id'],
                    agent_name=agent['agent_name'],
                    tier=agent['tier']
                ).set(0)
        
        threads = [threading.Thread(target=update_statuses) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        metrics = generate_latest(clean_registry)
        assert b'agents_status' in metrics


# ============================================================================
# TEST LABEL CARDINALITY VALIDATION
# ============================================================================

class TestLabelCardinality:
    """Tests for metric label cardinality"""
    
    def test_agent_labels_distinct(self, clean_registry, all_agents):
        """Test agent labels are distinct"""
        for agent in all_agents[:10]:
            agent_status.labels(
                agent_id=agent['agent_id'],
                agent_name=agent['agent_name'],
                tier=agent['tier']
            ).set(0)
        
        metrics = generate_latest(clean_registry)
        metrics_str = metrics.decode('utf-8')
        
        # Should have labels for each unique agent
        for agent in all_agents[:10]:
            assert agent['agent_name'].encode() in metrics
    
    def test_tier_labels_distinct(self, clean_registry):
        """Test tier labels are distinct"""
        tiers = ['TIER_1', 'TIER_2', 'TIER_3', 'TIER_4', 'TIER_5', 'TIER_6', 'TIER_7', 'TIER_8']
        for tier in tiers:
            tier_health_score.labels(tier=tier).set(0.90)
        
        metrics = generate_latest(clean_registry)
        for tier in tiers:
            assert f'tier="{tier}"'.encode() in metrics
    
    def test_reasonable_label_cardinality(self, clean_registry):
        """Test label cardinality remains reasonable"""
        # Each dimension should have limited values
        agent_ids = [f'APEX-{i:03d}' for i in range(5)]
        agent_names = ['APEX']
        task_types = ['analysis', 'optimization', 'verification', 'synthesis']
        
        count = 0
        for agent_id in agent_ids:
            for agent_name in agent_names:
                for task_type in task_types:
                    agent_tasks_total.labels(
                        agent_id=agent_id,
                        agent_name=agent_name,
                        task_type=task_type
                    ).inc()
                    count += 1
        
        # Should be manageable cardinality: 5 × 1 × 4 = 20 combinations
        assert count == len(agent_ids) * len(agent_names) * len(task_types)


# ============================================================================
# TEST INTEGRATION SCENARIOS
# ============================================================================

class TestIntegrationScenarios:
    """Integration tests combining multiple metrics"""
    
    def test_full_agent_lifecycle(self, clean_registry, sample_agent):
        """Test full agent lifecycle metrics"""
        # Agent starts
        update_agent_status(sample_agent['agent_id'], sample_agent['agent_name'], sample_agent['tier'], 0)
        
        # Agent receives tasks
        for _ in range(10):
            agent_tasks_total.labels(
                agent_id=sample_agent['agent_id'],
                agent_name=sample_agent['agent_name'],
                task_type='analysis'
            ).inc()
        
        # Agent completes tasks
        for _ in range(9):
            agent_tasks_completed.labels(
                agent_id=sample_agent['agent_id'],
                agent_name=sample_agent['agent_name'],
                task_type='analysis'
            ).inc()
        
        # One task fails
        agent_tasks_failed.labels(
            agent_id=sample_agent['agent_id'],
            agent_name=sample_agent['agent_name'],
            error_type='timeout'
        ).inc()
        
        # Update performance metrics
        update_agent_rates(sample_agent['agent_id'], sample_agent['agent_name'], 0.90, 0.08, 0.02)
        
        # Update utilization
        update_agent_utilization(sample_agent['agent_id'], sample_agent['agent_name'], 0.75, 50, 5)
        
        metrics = generate_latest(clean_registry)
        assert b'agents_status' in metrics
        assert b'agents_tasks_total' in metrics
        assert b'agents_tasks_completed' in metrics
        assert b'agents_tasks_failed' in metrics
    
    def test_collective_snapshot(self, clean_registry, all_agents):
        """Test collective health snapshot"""
        # Update individual agents
        for i, agent in enumerate(all_agents[:10]):
            status = i % 3  # Mix of health states
            update_agent_status(agent['agent_id'], agent['agent_name'], agent['tier'], status)
        
        # Update collective metrics
        healthy = 8
        degraded = 1
        failed = 1
        update_collective_health(10, healthy, degraded, failed)
        
        update_collective_metrics(
            active_tasks=250,
            utilization=0.62,
            throughput=125.5,
            error_rate=0.02
        )
        
        metrics = generate_latest(clean_registry)
        assert b'agents_collective_total' in metrics
        assert b'agents_collective_healthy' in metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
