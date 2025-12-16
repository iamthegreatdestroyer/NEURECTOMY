"""
DAG-Based Workflow Engine
Orchestrates complex multi-step workflows with dependencies using directed acyclic graphs
"""

import asyncio
from typing import Dict, List, Any, Callable, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import networkx as nx
import logging

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    """Workflow task definition"""
    task_id: str
    name: str
    task_type: str
    config: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate task on creation"""
        if not self.task_id:
            raise ValueError("task_id is required")
        if not self.task_type:
            raise ValueError("task_type is required")
    
    def get_duration(self) -> Optional[float]:
        """Get task execution duration in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


@dataclass
class Workflow:
    """Workflow definition"""
    workflow_id: str
    name: str
    tasks: Dict[str, Task]
    
    def __post_init__(self):
        """Validate workflow on creation"""
        if not self.workflow_id:
            raise ValueError("workflow_id is required")
    
    def validate(self) -> Tuple[bool, Optional[str]]:
        """
        Validate workflow DAG
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.tasks:
            return False, "Workflow must have at least one task"
        
        # Build graph and check for unknown dependencies
        graph = nx.DiGraph()
        
        for task_id, task in self.tasks.items():
            graph.add_node(task_id)
            for dep in task.dependencies:
                if dep not in self.tasks:
                    return False, f"Unknown dependency: {dep} in task {task_id}"
                graph.add_edge(dep, task_id)
        
        # Check for cycles
        if not nx.is_directed_acyclic_graph(graph):
            try:
                cycles = list(nx.simple_cycles(graph))
                return False, f"Workflow contains cycles: {cycles}"
            except:
                return False, "Workflow contains cycles"
        
        return True, None
    
    def get_execution_order(self) -> List[str]:
        """Get tasks in topological execution order"""
        graph = nx.DiGraph()
        
        for task_id, task in self.tasks.items():
            graph.add_node(task_id)
            for dep in task.dependencies:
                graph.add_edge(dep, task_id)
        
        return list(nx.topological_sort(graph))
    
    def get_task_levels(self) -> Dict[str, int]:
        """Get depth level for each task (0 = no dependencies)"""
        levels = {}
        
        for task_id, task in self.tasks.items():
            if not task.dependencies:
                levels[task_id] = 0
            else:
                levels[task_id] = max(levels.get(dep, 0) for dep in task.dependencies) + 1
        
        return levels


class WorkflowResult:
    """Workflow execution result"""
    
    def __init__(self, workflow_id: str, workflow_name: str):
        self.workflow_id = workflow_id
        self.workflow_name = workflow_name
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.status: str = "unknown"
    
    def add_task_result(self, task_id: str, task: Task):
        """Add task execution result"""
        self.tasks[task_id] = {
            "status": task.status.value,
            "result": task.result,
            "error": task.error,
            "duration": task.get_duration(),
            "start_time": task.start_time.isoformat() if task.start_time else None,
            "end_time": task.end_time.isoformat() if task.end_time else None,
        }
    
    def get_total_duration(self) -> Optional[float]:
        """Get total workflow execution duration"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "status": self.status,
            "duration": self.get_total_duration(),
            "tasks": self.tasks,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }


class WorkflowEngine:
    """
    Executes workflows in topological order with dependency tracking
    
    Features:
    - DAG validation
    - Topological task ordering
    - Dependency tracking
    - Error handling and recovery
    - Execution statistics
    
    Example:
        >>> engine = WorkflowEngine()
        >>> engine.register_handler("fetch", fetch_handler)
        >>> engine.register_handler("compress", compress_handler)
        >>> result = await engine.execute_workflow(workflow)
    """
    
    def __init__(self, max_concurrent: int = 1):
        """
        Initialize workflow engine
        
        Args:
            max_concurrent: Maximum concurrent task executions (default: 1, sequential)
        """
        self.task_handlers: Dict[str, Callable] = {}
        self.max_concurrent = max_concurrent
    
    def register_handler(self, task_type: str, handler: Callable):
        """
        Register a task type handler
        
        Args:
            task_type: Type of task (e.g., "inference", "compression")
            handler: Async callable that handles the task
        """
        self.task_handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")
    
    async def execute_workflow(self, workflow: Workflow) -> WorkflowResult:
        """
        Execute workflow tasks in dependency order
        
        Args:
            workflow: Workflow to execute
            
        Returns:
            WorkflowResult with execution details
            
        Raises:
            ValueError: If workflow is invalid
        """
        logger.info(f"Starting workflow execution: {workflow.workflow_id}")
        
        # Validate workflow
        is_valid, error_msg = workflow.validate()
        if not is_valid:
            logger.error(f"Workflow validation failed: {error_msg}")
            raise ValueError(f"Workflow validation failed: {error_msg}")
        
        result = WorkflowResult(workflow.workflow_id, workflow.name)
        result.start_time = datetime.now()
        
        try:
            # Get execution order
            execution_order = workflow.get_execution_order()
            logger.info(f"Execution order: {execution_order}")
            
            # Execute tasks in order
            for task_id in execution_order:
                task = workflow.tasks[task_id]
                
                # Check if dependencies completed successfully
                if not self._dependencies_met(task, workflow):
                    logger.warning(f"Skipping task {task_id} - dependencies failed")
                    task.status = TaskStatus.SKIPPED
                    result.add_task_result(task_id, task)
                    continue
                
                await self._execute_task(task, workflow)
                result.add_task_result(task_id, task)
            
            # Determine overall status
            result.status = self._get_workflow_status(workflow)
            logger.info(f"Workflow completed with status: {result.status}")
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)
            result.status = "failed"
            raise
        
        finally:
            result.end_time = datetime.now()
        
        return result
    
    def _dependencies_met(self, task: Task, workflow: Workflow) -> bool:
        """
        Check if all task dependencies completed successfully
        
        Args:
            task: Task to check
            workflow: Workflow context
            
        Returns:
            True if all dependencies completed, False otherwise
        """
        for dep_id in task.dependencies:
            dep_task = workflow.tasks[dep_id]
            if dep_task.status != TaskStatus.COMPLETED:
                logger.debug(f"Dependency {dep_id} not completed (status: {dep_task.status})")
                return False
        return True
    
    async def _execute_task(self, task: Task, workflow: Workflow):
        """
        Execute a single task
        
        Args:
            task: Task to execute
            workflow: Workflow context
        """
        logger.info(f"Executing task: {task.task_id} ({task.task_type})")
        
        task.status = TaskStatus.RUNNING
        task.start_time = datetime.now()
        
        try:
            # Get handler
            handler = self.task_handlers.get(task.task_type)
            if not handler:
                raise ValueError(f"No handler for task type: {task.task_type}")
            
            logger.debug(f"Using handler for task type: {task.task_type}")
            
            # Execute handler with task config and workflow context
            if asyncio.iscoroutinefunction(handler):
                result = await handler(task.config, workflow)
            else:
                result = handler(task.config, workflow)
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            logger.info(f"Task completed: {task.task_id} ({task.get_duration():.2f}s)")
            
        except Exception as e:
            logger.error(f"Task failed: {task.task_id} - {e}", exc_info=True)
            task.status = TaskStatus.FAILED
            task.error = str(e)
        
        finally:
            task.end_time = datetime.now()
    
    def _get_workflow_status(self, workflow: Workflow) -> str:
        """
        Determine overall workflow status
        
        Args:
            workflow: Workflow to check
            
        Returns:
            Status string: "failed", "completed", "running", or "unknown"
        """
        statuses = [task.status for task in workflow.tasks.values()]
        
        if any(s == TaskStatus.FAILED for s in statuses):
            return "failed"
        elif all(s in [TaskStatus.COMPLETED, TaskStatus.SKIPPED] for s in statuses):
            return "completed"
        elif any(s == TaskStatus.RUNNING for s in statuses):
            return "running"
        else:
            return "unknown"
    
    def get_task_metrics(self, result: WorkflowResult) -> Dict[str, Any]:
        """
        Get metrics for workflow execution
        
        Args:
            result: Workflow execution result
            
        Returns:
            Dictionary with execution metrics
        """
        total_duration = result.get_total_duration()
        task_durations = [
            task_info["duration"]
            for task_info in result.tasks.values()
            if task_info["duration"] is not None
        ]
        
        metrics = {
            "total_duration": total_duration,
            "total_tasks": len(result.tasks),
            "completed_tasks": sum(1 for t in result.tasks.values() if t["status"] == "completed"),
            "failed_tasks": sum(1 for t in result.tasks.values() if t["status"] == "failed"),
            "skipped_tasks": sum(1 for t in result.tasks.values() if t["status"] == "skipped"),
        }
        
        if task_durations:
            metrics["avg_task_duration"] = sum(task_durations) / len(task_durations)
            metrics["max_task_duration"] = max(task_durations)
            metrics["min_task_duration"] = min(task_durations)
        
        return metrics
