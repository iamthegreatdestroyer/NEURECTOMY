"""
Tests for Workflow Engine
"""

import pytest
import asyncio
from datetime import datetime
from neurectomy.orchestration.workflow_engine import (
    WorkflowEngine,
    Workflow,
    Task,
    TaskStatus,
    WorkflowResult,
)


# Test fixtures and handlers
async def simple_handler(config: dict, workflow: Workflow):
    """Simple test handler"""
    await asyncio.sleep(0.1)
    return {"status": "success", "input": config}


async def error_handler(config: dict, workflow: Workflow):
    """Handler that raises an error"""
    raise ValueError("Test error")


async def slow_handler(config: dict, workflow: Workflow):
    """Slow handler for testing"""
    await asyncio.sleep(0.2)
    return {"status": "slow_success"}


@pytest.fixture
def engine():
    """Create workflow engine for tests"""
    engine = WorkflowEngine()
    engine.register_handler("test", simple_handler)
    engine.register_handler("error", error_handler)
    engine.register_handler("slow", slow_handler)
    return engine


@pytest.fixture
def simple_workflow():
    """Create simple single-task workflow"""
    return Workflow(
        workflow_id="test_workflow_1",
        name="Simple Test Workflow",
        tasks={
            "task_1": Task(
                task_id="task_1",
                name="First Task",
                task_type="test",
                config={"data": "test"},
                dependencies=[],
            )
        },
    )


@pytest.fixture
def dependent_workflow():
    """Create workflow with task dependencies"""
    return Workflow(
        workflow_id="test_workflow_2",
        name="Dependent Workflow",
        tasks={
            "task_1": Task(
                task_id="task_1",
                name="First Task",
                task_type="test",
                config={"data": "first"},
                dependencies=[],
            ),
            "task_2": Task(
                task_id="task_2",
                name="Second Task",
                task_type="test",
                config={"data": "second"},
                dependencies=["task_1"],
            ),
            "task_3": Task(
                task_id="task_3",
                name="Third Task",
                task_type="test",
                config={"data": "third"},
                dependencies=["task_2"],
            ),
        },
    )


class TestTaskStatus:
    """Test TaskStatus enum"""
    
    def test_status_values(self):
        """Test TaskStatus enum values"""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.SKIPPED.value == "skipped"


class TestTask:
    """Test Task dataclass"""
    
    def test_task_creation(self):
        """Test task creation"""
        task = Task(
            task_id="test_task",
            name="Test Task",
            task_type="test",
            config={"key": "value"},
        )
        
        assert task.task_id == "test_task"
        assert task.name == "Test Task"
        assert task.task_type == "test"
        assert task.status == TaskStatus.PENDING
        assert task.dependencies == []
    
    def test_task_with_dependencies(self):
        """Test task with dependencies"""
        task = Task(
            task_id="test_task",
            name="Test Task",
            task_type="test",
            config={},
            dependencies=["dep_1", "dep_2"],
        )
        
        assert task.dependencies == ["dep_1", "dep_2"]
    
    def test_task_missing_id(self):
        """Test task fails with missing ID"""
        with pytest.raises(ValueError):
            Task(task_id="", name="Test", task_type="test", config={})
    
    def test_task_missing_type(self):
        """Test task fails with missing type"""
        with pytest.raises(ValueError):
            Task(task_id="test", name="Test", task_type="", config={})
    
    def test_task_duration(self):
        """Test task duration calculation"""
        task = Task(
            task_id="test",
            name="Test",
            task_type="test",
            config={},
        )
        
        now = datetime.now()
        task.start_time = now
        task.end_time = now
        
        assert task.get_duration() == 0.0


class TestWorkflow:
    """Test Workflow dataclass"""
    
    def test_workflow_creation(self):
        """Test workflow creation"""
        task = Task(
            task_id="task_1",
            name="Task 1",
            task_type="test",
            config={},
        )
        
        workflow = Workflow(
            workflow_id="test_workflow",
            name="Test Workflow",
            tasks={"task_1": task},
        )
        
        assert workflow.workflow_id == "test_workflow"
        assert len(workflow.tasks) == 1
    
    def test_workflow_validation_empty(self):
        """Test workflow validation with no tasks"""
        workflow = Workflow(
            workflow_id="test",
            name="Empty Workflow",
            tasks={},
        )
        
        is_valid, error = workflow.validate()
        assert not is_valid
        assert "at least one task" in error
    
    def test_workflow_validation_unknown_dependency(self):
        """Test workflow validation with unknown dependency"""
        task = Task(
            task_id="task_1",
            name="Task 1",
            task_type="test",
            config={},
            dependencies=["unknown"],
        )
        
        workflow = Workflow(
            workflow_id="test",
            name="Bad Workflow",
            tasks={"task_1": task},
        )
        
        is_valid, error = workflow.validate()
        assert not is_valid
        assert "Unknown dependency" in error
    
    def test_workflow_validation_cycle(self):
        """Test workflow validation detects cycles"""
        task_1 = Task(
            task_id="task_1",
            name="Task 1",
            task_type="test",
            config={},
            dependencies=["task_2"],
        )
        
        task_2 = Task(
            task_id="task_2",
            name="Task 2",
            task_type="test",
            config={},
            dependencies=["task_1"],
        )
        
        workflow = Workflow(
            workflow_id="test",
            name="Cyclic Workflow",
            tasks={"task_1": task_1, "task_2": task_2},
        )
        
        is_valid, error = workflow.validate()
        assert not is_valid
        assert "cycles" in error
    
    def test_workflow_execution_order(self, dependent_workflow):
        """Test workflow execution order"""
        order = dependent_workflow.get_execution_order()
        
        assert order == ["task_1", "task_2", "task_3"]
    
    def test_workflow_task_levels(self, dependent_workflow):
        """Test task level calculation"""
        levels = dependent_workflow.get_task_levels()
        
        assert levels["task_1"] == 0
        assert levels["task_2"] == 1
        assert levels["task_3"] == 2


class TestWorkflowResult:
    """Test WorkflowResult"""
    
    def test_result_creation(self):
        """Test result creation"""
        result = WorkflowResult("test_workflow", "Test Workflow")
        
        assert result.workflow_id == "test_workflow"
        assert result.status == "unknown"
    
    def test_result_to_dict(self):
        """Test result conversion to dict"""
        result = WorkflowResult("test_workflow", "Test Workflow")
        result.status = "completed"
        
        result_dict = result.to_dict()
        
        assert result_dict["workflow_id"] == "test_workflow"
        assert result_dict["workflow_name"] == "Test Workflow"
        assert result_dict["status"] == "completed"


class TestWorkflowEngine:
    """Test WorkflowEngine"""
    
    def test_engine_creation(self):
        """Test engine creation"""
        engine = WorkflowEngine()
        assert engine.max_concurrent == 1
    
    def test_engine_handler_registration(self, engine):
        """Test handler registration"""
        assert "test" in engine.task_handlers
        assert "error" in engine.task_handlers
    
    @pytest.mark.asyncio
    async def test_simple_workflow_execution(self, engine, simple_workflow):
        """Test simple workflow execution"""
        result = await engine.execute_workflow(simple_workflow)
        
        assert result.status == "completed"
        assert result.tasks["task_1"]["status"] == "completed"
        assert result.tasks["task_1"]["result"]["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_dependent_workflow_execution(self, engine, dependent_workflow):
        """Test dependent workflow execution"""
        result = await engine.execute_workflow(dependent_workflow)
        
        assert result.status == "completed"
        assert result.tasks["task_1"]["status"] == "completed"
        assert result.tasks["task_2"]["status"] == "completed"
        assert result.tasks["task_3"]["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_task_failure_handling(self, engine):
        """Test handling of task failure"""
        workflow = Workflow(
            workflow_id="fail_test",
            name="Failure Test",
            tasks={
                "task_1": Task(
                    task_id="task_1",
                    name="Error Task",
                    task_type="error",
                    config={},
                ),
            },
        )
        
        with pytest.raises(Exception):
            await engine.execute_workflow(workflow)
    
    @pytest.mark.asyncio
    async def test_skipped_task_on_dependency_failure(self, engine):
        """Test task skipping when dependency fails"""
        workflow = Workflow(
            workflow_id="skip_test",
            name="Skip Test",
            tasks={
                "task_1": Task(
                    task_id="task_1",
                    name="Error Task",
                    task_type="error",
                    config={},
                    dependencies=[],
                ),
                "task_2": Task(
                    task_id="task_2",
                    name="Skip Task",
                    task_type="test",
                    config={},
                    dependencies=["task_1"],
                ),
            },
        )
        
        # Should raise due to error in task_1
        with pytest.raises(Exception):
            await engine.execute_workflow(workflow)
    
    @pytest.mark.asyncio
    async def test_workflow_metrics(self, engine, simple_workflow):
        """Test workflow metrics calculation"""
        result = await engine.execute_workflow(simple_workflow)
        metrics = engine.get_task_metrics(result)
        
        assert metrics["total_tasks"] == 1
        assert metrics["completed_tasks"] == 1
        assert metrics["failed_tasks"] == 0
        assert metrics["skipped_tasks"] == 0
    
    @pytest.mark.asyncio
    async def test_workflow_validation_error(self, engine):
        """Test workflow validation error handling"""
        workflow = Workflow(
            workflow_id="bad",
            name="Bad",
            tasks={
                "task": Task(
                    task_id="task",
                    name="Task",
                    task_type="test",
                    config={},
                    dependencies=["unknown"],
                )
            },
        )
        
        with pytest.raises(ValueError):
            await engine.execute_workflow(workflow)
