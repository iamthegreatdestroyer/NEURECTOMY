"""
Tests for Agent Supervisor
"""

import asyncio
import pytest
from neurectomy.agents.supervisor import AgentSupervisor, AgentStatus, AgentHealth


@pytest.mark.asyncio
async def test_supervisor_initialization():
    """Test supervisor initialization"""
    supervisor = AgentSupervisor(heartbeat_timeout=30)
    assert supervisor.heartbeat_timeout == 30
    assert supervisor.running is False


@pytest.mark.asyncio
async def test_register_agent():
    """Test agent registration"""
    supervisor = AgentSupervisor()
    
    health = supervisor.register_agent("agent_01")
    
    assert health.agent_id == "agent_01"
    assert health.status == AgentStatus.HEALTHY
    assert health.failure_count == 0


@pytest.mark.asyncio
async def test_report_heartbeat():
    """Test heartbeat reporting"""
    supervisor = AgentSupervisor()
    supervisor.register_agent("agent_01")
    
    await supervisor.report_heartbeat("agent_01")
    
    health = supervisor.get_agent_health("agent_01")
    assert health.status == AgentStatus.HEALTHY


@pytest.mark.asyncio
async def test_auto_register_on_heartbeat():
    """Test agent auto-registration on first heartbeat"""
    supervisor = AgentSupervisor()
    
    await supervisor.report_heartbeat("agent_01")
    
    assert "agent_01" in supervisor.agents
    assert supervisor.get_agent_health("agent_01") is not None


@pytest.mark.asyncio
async def test_get_all_health():
    """Test get all agents health"""
    supervisor = AgentSupervisor()
    
    supervisor.register_agent("agent_01")
    supervisor.register_agent("agent_02")
    supervisor.register_agent("agent_03")
    
    health_dict = supervisor.get_all_health()
    
    assert len(health_dict) == 3
    assert "agent_01" in health_dict
    assert "agent_02" in health_dict
    assert "agent_03" in health_dict


@pytest.mark.asyncio
async def test_get_health_summary():
    """Test health summary"""
    supervisor = AgentSupervisor()
    
    supervisor.register_agent("agent_01")
    supervisor.register_agent("agent_02")
    supervisor.register_agent("agent_03")
    
    summary = supervisor.get_health_summary()
    
    assert summary["total"] == 3
    assert summary["healthy"] == 3
    assert summary["degraded"] == 0
    assert summary["failed"] == 0


@pytest.mark.asyncio
async def test_start_stop_monitoring():
    """Test start and stop monitoring"""
    supervisor = AgentSupervisor()
    
    assert supervisor.running is False
    
    await supervisor.start_monitoring()
    assert supervisor.running is True
    
    await supervisor.stop_monitoring()
    assert supervisor.running is False


@pytest.mark.asyncio
async def test_monitoring_loop_with_timeout():
    """Test monitoring loop detects timeouts"""
    supervisor = AgentSupervisor(
        heartbeat_timeout=1,
        check_interval=1,
    )
    
    supervisor.register_agent("agent_01")
    
    await supervisor.start_monitoring()
    
    # Wait for monitoring loop to detect timeout
    await asyncio.sleep(3)
    
    health = supervisor.get_agent_health("agent_01")
    assert health.status in [AgentStatus.DEGRADED, AgentStatus.FAILED]
    
    await supervisor.stop_monitoring()


@pytest.mark.asyncio
async def test_recovery_on_healthy_heartbeat():
    """Test that agent recovers when reporting heartbeat after failure"""
    supervisor = AgentSupervisor()
    
    agent_health = supervisor.register_agent("agent_01")
    agent_health.status = AgentStatus.FAILED
    agent_health.failure_count = 1
    
    await supervisor.report_heartbeat("agent_01")
    
    updated_health = supervisor.get_agent_health("agent_01")
    assert updated_health.status == AgentStatus.HEALTHY
    assert updated_health.failure_count == 0


@pytest.mark.asyncio
async def test_get_failed_agents():
    """Test getting list of failed agents"""
    supervisor = AgentSupervisor()
    
    supervisor.register_agent("agent_01")
    supervisor.register_agent("agent_02")
    supervisor.register_agent("agent_03")
    
    # Mark two as failed
    supervisor.agents["agent_01"].status = AgentStatus.FAILED
    supervisor.agents["agent_02"].status = AgentStatus.FAILED
    
    failed = supervisor.get_failed_agents()
    
    assert len(failed) == 2
    assert "agent_01" in failed
    assert "agent_02" in failed
    assert "agent_03" not in failed


@pytest.mark.asyncio
async def test_get_degraded_agents():
    """Test getting list of degraded agents"""
    supervisor = AgentSupervisor()
    
    supervisor.register_agent("agent_01")
    supervisor.register_agent("agent_02")
    supervisor.register_agent("agent_03")
    
    # Mark one as degraded
    supervisor.agents["agent_02"].status = AgentStatus.DEGRADED
    
    degraded = supervisor.get_degraded_agents()
    
    assert len(degraded) == 1
    assert "agent_02" in degraded


@pytest.mark.asyncio
async def test_agent_status_enum():
    """Test AgentStatus enum values"""
    assert AgentStatus.HEALTHY.value == "healthy"
    assert AgentStatus.DEGRADED.value == "degraded"
    assert AgentStatus.FAILED.value == "failed"
    assert AgentStatus.RECOVERING.value == "recovering"


def test_agent_health_dataclass():
    """Test AgentHealth dataclass"""
    from datetime import datetime
    
    now = datetime.now()
    health = AgentHealth(
        agent_id="test_agent",
        status=AgentStatus.HEALTHY,
        last_heartbeat=now,
        failure_count=0,
    )
    
    assert health.agent_id == "test_agent"
    assert health.status == AgentStatus.HEALTHY
    assert health.last_heartbeat == now
    assert health.failure_count == 0


@pytest.mark.asyncio
async def test_multiple_agents_health_tracking():
    """Test health tracking for multiple agents"""
    supervisor = AgentSupervisor()
    
    # Register 40 agents (simulating the Elite Agent Collective)
    for i in range(40):
        supervisor.register_agent(f"agent_{i:02d}")
    
    summary = supervisor.get_health_summary()
    
    assert summary["total"] == 40
    assert summary["healthy"] == 40
    
    # Mark some as degraded
    for i in range(5):
        supervisor.agents[f"agent_{i:02d}"].status = AgentStatus.DEGRADED
    
    summary = supervisor.get_health_summary()
    
    assert summary["healthy"] == 35
    assert summary["degraded"] == 5
