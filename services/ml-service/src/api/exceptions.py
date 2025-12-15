"""Neurectomy Custom Exceptions"""

from typing import Optional, List


class NeurectomyError(Exception):
    """Base exception for all Neurectomy errors."""
    
    def __init__(self, message: str, error_code: str = "NEURECTOMY_ERROR", is_retryable: bool = False):
        super().__init__(message)
        self.error_code = error_code
        self.is_retryable = is_retryable


class AgentNotFoundError(NeurectomyError):
    """Raised when requested agent doesn't exist."""
    
    def __init__(self, role: str):
        super().__init__(f"Agent not found: {role}", "AGENT_NOT_FOUND")
        self.role = role


class AgentBusyError(NeurectomyError):
    """Raised when agent is busy with another task."""
    
    def __init__(self, agent_id: str, current_task: str):
        super().__init__(f"Agent {agent_id} busy with: {current_task}", "AGENT_BUSY", is_retryable=True)
        self.agent_id = agent_id
        self.current_task = current_task


class TaskExecutionError(NeurectomyError):
    """Raised when task execution fails."""
    
    def __init__(self, task_id: str, message: str, failed_agents: Optional[List[str]] = None):
        super().__init__(f"Task {task_id} failed: {message}", "TASK_EXECUTION_ERROR", is_retryable=True)
        self.task_id = task_id
        self.failed_agents = failed_agents or []


class PlanExecutionError(NeurectomyError):
    """Raised when plan execution fails."""
    
    def __init__(self, plan_id: str, failed_task: str, message: str):
        super().__init__(f"Plan {plan_id} failed at task {failed_task}: {message}", "PLAN_EXECUTION_ERROR")
        self.plan_id = plan_id
        self.failed_task = failed_task


class ContextBuildError(NeurectomyError):
    """Raised when context building fails."""
    
    def __init__(self, message: str, missing_files: Optional[List[str]] = None):
        super().__init__(message, "CONTEXT_BUILD_ERROR")
        self.missing_files = missing_files or []


class InferenceError(NeurectomyError):
    """Raised when inference fails."""
    
    def __init__(self, backend: str, message: str):
        super().__init__(f"Inference failed ({backend}): {message}", "INFERENCE_ERROR", is_retryable=True)
        self.backend = backend


class StorageError(NeurectomyError):
    """Raised when storage operations fail."""
    
    def __init__(self, operation: str, message: str):
        super().__init__(f"Storage {operation} failed: {message}", "STORAGE_ERROR", is_retryable=True)
        self.operation = operation


class ProjectError(NeurectomyError):
    """Raised when project operations fail."""
    
    def __init__(self, project: str, message: str):
        super().__init__(f"Project error ({project}): {message}", "PROJECT_ERROR")
        self.project = project


class CompressionError(NeurectomyError):
    """Raised when ΣLANG compression fails."""
    
    def __init__(self, message: str, original_size: int = 0):
        super().__init__(f"Compression failed: {message}", "COMPRESSION_ERROR", is_retryable=True)
        self.original_size = original_size
