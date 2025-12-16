"""
Tests for Task Delegation System
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from neurectomy.agents.coordination.task_delegation import (
    TaskDelegator,
    AgentCapability,
    Task,
    TaskPriority,
    DelegationResult,
)


@pytest.fixture
def delegator():
    """Create task delegator for tests"""
    return TaskDelegator()


@pytest.fixture
def sample_agent():
    """Create sample agent"""
    return AgentCapability(
        agent_id="test_agent",
        capabilities={"inference", "text_generation"},
        max_concurrent_tasks=3,
        specialization_score={"inference": 1.0, "text_generation": 0.9},
    )


@pytest.fixture
def sample_task():
    """Create sample task"""
    return Task(
        task_id="task_001",
        task_type="inference",
        priority=TaskPriority.MEDIUM,
        payload={"prompt": "test"},
        required_capabilities={"inference"},
    )


class TestTaskPriority:
    """Test TaskPriority enum"""
    
    def test_priority_levels(self):
        """Test priority levels"""
        assert TaskPriority.LOW.value == 1
        assert TaskPriority.MEDIUM.value == 2
        assert TaskPriority.HIGH.value == 3
        assert TaskPriority.CRITICAL.value == 4


class TestAgentCapability:
    """Test AgentCapability"""
    
    def test_agent_creation(self):
        """Test agent creation"""
        agent = AgentCapability(
            agent_id="test",
            capabilities={"inference"},
        )
        
        assert agent.agent_id == "test"
        assert "inference" in agent.capabilities
        assert agent.max_concurrent_tasks == 5
        assert agent.active_tasks == 0
    
    def test_agent_utilization(self, sample_agent):
        """Test utilization calculation"""
        assert sample_agent.get_utilization() == 0.0
        
        sample_agent.active_tasks = 1
        assert sample_agent.get_utilization() == pytest.approx(1/3)
        
        sample_agent.active_tasks = 3
        assert sample_agent.get_utilization() == 1.0
    
    def test_agent_capacity(self, sample_agent):
        """Test capacity checking"""
        assert sample_agent.has_capacity()
        
        sample_agent.active_tasks = 3
        assert not sample_agent.has_capacity()
    
    def test_agent_can_handle(self, sample_agent):
        """Test capability matching"""
        assert sample_agent.can_handle({"inference"})
        assert sample_agent.can_handle({"inference", "text_generation"})
        assert not sample_agent.can_handle({"compression"})


class TestTask:
    """Test Task"""
    
    def test_task_creation(self):
        """Test task creation"""
        task = Task(
            task_id="task_1",
            task_type="inference",
            priority=TaskPriority.HIGH,
            payload={"data": "test"},
            required_capabilities={"inference"},
        )
        
        assert task.task_id == "task_1"
        assert task.task_type == "inference"
        assert task.priority == TaskPriority.HIGH
    
    def test_task_missing_id(self):
        """Test task requires ID"""
        with pytest.raises(ValueError):
            Task(
                task_id="",
                task_type="inference",
                priority=TaskPriority.MEDIUM,
                payload={},
                required_capabilities=set(),
            )
    
    def test_task_with_deadline(self):
        """Test task with deadline"""
        deadline = datetime.now() + timedelta(hours=1)
        task = Task(
            task_id="task_1",
            task_type="inference",
            priority=TaskPriority.CRITICAL,
            payload={},
            required_capabilities=set(),
            deadline=deadline,
        )
        
        assert task.deadline == deadline


class TestTaskDelegator:
    """Test TaskDelegator"""
    
    def test_delegator_creation(self):
        """Test delegator creation"""
        delegator = TaskDelegator()
        assert len(delegator.agents) == 0
        assert len(delegator.pending_tasks) == 0
    
    def test_agent_registration(self, delegator, sample_agent):
        """Test agent registration"""
        delegator.register_agent(sample_agent)
        
        assert sample_agent.agent_id in delegator.agents
        assert delegator.agents[sample_agent.agent_id] == sample_agent
    
    @pytest.mark.asyncio
    async def test_simple_delegation(self, delegator, sample_agent, sample_task):
        """Test simple task delegation"""
        delegator.register_agent(sample_agent)
        
        result = await delegator.delegate_task(sample_task)
        
        assert result.success
        assert result.agent_id == sample_agent.agent_id
        assert sample_agent.active_tasks == 1
    
    @pytest.mark.asyncio
    async def test_delegation_capability_mismatch(self, delegator, sample_agent):
        """Test delegation with capability mismatch"""
        delegator.register_agent(sample_agent)
        
        task = Task(
            task_id="task_1",
            task_type="compression",
            priority=TaskPriority.MEDIUM,
            payload={},
            required_capabilities={"compression"},
        )
        
        result = await delegator.delegate_task(task)
        
        assert not result.success
        assert result.agent_id is None
    
    @pytest.mark.asyncio
    async def test_delegation_capacity_exceeded(self, delegator, sample_agent, sample_task):
        """Test delegation when capacity exceeded"""
        sample_agent.max_concurrent_tasks = 1
        delegator.register_agent(sample_agent)
        
        # First task should succeed
        result1 = await delegator.delegate_task(sample_task)
        assert result1.success
        
        # Second task should fail
        task2 = Task(
            task_id="task_2",
            task_type="inference",
            priority=TaskPriority.MEDIUM,
            payload={},
            required_capabilities={"inference"},
        )
        result2 = await delegator.delegate_task(task2)
        assert not result2.success
    
    @pytest.mark.asyncio
    async def test_multiple_agents(self, delegator):
        """Test delegation with multiple agents"""
        agent1 = AgentCapability(
            agent_id="agent1",
            capabilities={"inference"},
            max_concurrent_tasks=1,
        )
        agent2 = AgentCapability(
            agent_id="agent2",
            capabilities={"inference"},
            max_concurrent_tasks=2,
        )
        
        delegator.register_agent(agent1)
        delegator.register_agent(agent2)
        
        task = Task(
            task_id="task_1",
            task_type="inference",
            priority=TaskPriority.MEDIUM,
            payload={},
            required_capabilities={"inference"},
        )
        
        result = await delegator.delegate_task(task)
        
        # Should prefer less loaded agent
        assert result.agent_id in ["agent1", "agent2"]
    
    @pytest.mark.asyncio
    async def test_mark_task_complete(self, delegator, sample_agent, sample_task):
        """Test marking task as complete"""
        delegator.register_agent(sample_agent)
        
        result = await delegator.delegate_task(sample_task)
        assert sample_agent.active_tasks == 1
        
        await delegator.mark_task_complete(sample_task.task_id, sample_agent.agent_id)
        assert sample_agent.active_tasks == 0
    
    @pytest.mark.asyncio
    async def test_pending_task_processing(self, delegator, sample_agent):
        """Test processing pending tasks"""
        sample_agent.max_concurrent_tasks = 1
        delegator.register_agent(sample_agent)
        
        # Create two tasks
        task1 = Task(
            task_id="task_1",
            task_type="inference",
            priority=TaskPriority.MEDIUM,
            payload={},
            required_capabilities={"inference"},
        )
        task2 = Task(
            task_id="task_2",
            task_type="inference",
            priority=TaskPriority.MEDIUM,
            payload={},
            required_capabilities={"inference"},
        )
        
        # Delegate first task (succeeds)
        result1 = await delegator.delegate_task(task1)
        assert result1.success
        
        # Delegate second task (fails, goes to pending)
        result2 = await delegator.delegate_task(task2)
        assert not result2.success
        assert len(delegator.pending_tasks) == 1
        
        # Complete first task
        await delegator.mark_task_complete(task1.task_id, sample_agent.agent_id)
        
        # Second task should be delegated
        assert len(delegator.pending_tasks) == 0
        assert sample_agent.active_tasks == 1
    
    def test_agent_status(self, delegator, sample_agent):
        """Test getting agent status"""
        delegator.register_agent(sample_agent)
        
        status = delegator.get_agent_status(sample_agent.agent_id)
        
        assert status["agent_id"] == sample_agent.agent_id
        assert status["active_tasks"] == 0
        assert status["max_tasks"] == 3
    
    def test_collective_status(self, delegator):
        """Test getting collective status"""
        agent1 = AgentCapability(
            agent_id="agent1",
            capabilities={"inference"},
            max_concurrent_tasks=3,
        )
        agent2 = AgentCapability(
            agent_id="agent2",
            capabilities={"compression"},
            max_concurrent_tasks=5,
        )
        
        delegator.register_agent(agent1)
        delegator.register_agent(agent2)
        
        status = delegator.get_collective_status()
        
        assert status["total_agents"] == 2
        assert status["total_capacity"] == 8
        assert status["active_tasks"] == 0
    
    def test_delegation_history(self, delegator):
        """Test delegation history tracking"""
        agent = AgentCapability(
            agent_id="agent1",
            capabilities={"inference"},
            max_concurrent_tasks=3,
        )
        delegator.register_agent(agent)
        
        result1 = DelegationResult(
            task_id="task_1",
            agent_id="agent1",
            success=True,
        )
        result2 = DelegationResult(
            task_id="task_2",
            agent_id=None,
            success=False,
        )
        
        delegator.delegation_history.append(result1)
        delegator.delegation_history.append(result2)
        
        history = delegator.get_delegation_history()
        assert len(history) == 2
        
        # Filter by success
        successful = delegator.get_delegation_history(success_only=True)
        assert len(successful) == 1
    
    def test_stats(self, delegator):
        """Test statistics"""
        agent = AgentCapability(
            agent_id="agent1",
            capabilities={"inference"},
            max_concurrent_tasks=3,
        )
        delegator.register_agent(agent)
        
        delegator.delegation_history.append(
            DelegationResult("task_1", "agent1", True, score=0.9)
        )
        delegator.delegation_history.append(
            DelegationResult("task_2", "agent1", True, score=0.85)
        )
        
        stats = delegator.get_stats()
        
        assert stats["total_delegations"] == 2
        assert stats["successful"] == 2
        assert stats["failed"] == 0
        assert stats["success_rate"] == 1.0
