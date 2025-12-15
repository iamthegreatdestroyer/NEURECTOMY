"""
Neurectomy Error Handling
=========================

Unified error handling across all components.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timezone
import traceback
import logging


class ErrorSeverity(Enum):
    """Error severity levels."""
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()
    FATAL = auto()


class ErrorCategory(Enum):
    """Error categories for routing."""
    INFERENCE = "inference"
    COMPRESSION = "compression"
    STORAGE = "storage"
    AGENT = "agent"
    ORCHESTRATION = "orchestration"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    RESOURCE = "resource"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


@dataclass
class NeurectomyError(Exception):
    """
    Base error class for all Neurectomy errors.
    
    Provides structured error information for debugging and recovery.
    """
    
    message: str
    category: ErrorCategory = ErrorCategory.UNKNOWN
    severity: ErrorSeverity = ErrorSeverity.ERROR
    
    # Context
    component: Optional[str] = None
    operation: Optional[str] = None
    task_id: Optional[str] = None
    agent_id: Optional[str] = None
    
    # Technical details
    original_exception: Optional[Exception] = None
    stack_trace: Optional[str] = None
    
    # Recovery hints
    recoverable: bool = True
    retry_recommended: bool = False
    fallback_available: bool = False
    
    # Metadata
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.original_exception and not self.stack_trace:
            self.stack_trace = traceback.format_exc()
    
    def __str__(self) -> str:
        parts = [f"[{self.category.value}] {self.message}"]
        if self.component:
            parts.append(f"Component: {self.component}")
        if self.operation:
            parts.append(f"Operation: {self.operation}")
        return " | ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize error for logging/transmission."""
        return {
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.name,
            "component": self.component,
            "operation": self.operation,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "recoverable": self.recoverable,
            "retry_recommended": self.retry_recommended,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
        }


# Specific Error Types

class InferenceError(NeurectomyError):
    """Errors during inference operations."""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.INFERENCE,
            component="ryot_llm",
            **kwargs
        )


class CompressionError(NeurectomyError):
    """Errors during compression/decompression."""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.COMPRESSION,
            component="sigmalang",
            **kwargs
        )


class StorageError(NeurectomyError):
    """Errors during storage operations."""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.STORAGE,
            component="sigmavault",
            **kwargs
        )


class AgentError(NeurectomyError):
    """Errors from agent operations."""
    def __init__(self, message: str, agent_id: str = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.AGENT,
            component="elite_collective",
            agent_id=agent_id,
            **kwargs
        )


class OrchestrationError(NeurectomyError):
    """Errors in task orchestration."""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.ORCHESTRATION,
            component="orchestrator",
            **kwargs
        )


class ConfigurationError(NeurectomyError):
    """Configuration-related errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.CONFIGURATION,
            recoverable=False,
            **kwargs
        )


class ResourceExhaustedError(NeurectomyError):
    """Resource exhaustion errors (memory, tokens, etc.)."""
    def __init__(self, message: str, resource_type: str = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.RESOURCE,
            retry_recommended=True,
            context={"resource_type": resource_type},
            **kwargs
        )


class ErrorHandler:
    """
    Central error handler for the Neurectomy system.
    
    Provides:
    - Error logging
    - Error recovery strategies
    - Error aggregation and reporting
    """
    
    def __init__(self):
        self._logger = logging.getLogger("neurectomy.errors")
        self._error_counts: Dict[ErrorCategory, int] = {
            cat: 0 for cat in ErrorCategory
        }
        self._recent_errors: list = []
        self._max_recent = 100
        
        # Recovery strategies
        self._recovery_strategies: Dict[ErrorCategory, callable] = {}
    
    def handle(self, error: NeurectomyError) -> bool:
        """
        Handle an error.
        
        Returns True if error was recovered, False otherwise.
        """
        # Log error
        self._log_error(error)
        
        # Track error
        self._track_error(error)
        
        # Attempt recovery
        if error.recoverable:
            return self._attempt_recovery(error)
        
        return False
    
    def register_recovery_strategy(
        self,
        category: ErrorCategory,
        strategy: callable,
    ) -> None:
        """Register a recovery strategy for an error category."""
        self._recovery_strategies[category] = strategy
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "counts_by_category": {
                cat.value: count
                for cat, count in self._error_counts.items()
            },
            "total_errors": sum(self._error_counts.values()),
            "recent_error_count": len(self._recent_errors),
        }
    
    def _log_error(self, error: NeurectomyError) -> None:
        """Log error with appropriate level."""
        log_method = {
            ErrorSeverity.DEBUG: self._logger.debug,
            ErrorSeverity.INFO: self._logger.info,
            ErrorSeverity.WARNING: self._logger.warning,
            ErrorSeverity.ERROR: self._logger.error,
            ErrorSeverity.CRITICAL: self._logger.critical,
            ErrorSeverity.FATAL: self._logger.critical,
        }.get(error.severity, self._logger.error)
        
        log_method(str(error), extra={"error_data": error.to_dict()})
    
    def _track_error(self, error: NeurectomyError) -> None:
        """Track error for statistics."""
        self._error_counts[error.category] += 1
        
        self._recent_errors.append(error.to_dict())
        if len(self._recent_errors) > self._max_recent:
            self._recent_errors.pop(0)
    
    def _attempt_recovery(self, error: NeurectomyError) -> bool:
        """Attempt to recover from error."""
        strategy = self._recovery_strategies.get(error.category)
        
        if strategy:
            try:
                return strategy(error)
            except Exception:
                return False
        
        return False


# Global error handler instance
_error_handler = ErrorHandler()


def get_error_handler() -> ErrorHandler:
    """Get the global error handler."""
    return _error_handler


def handle_error(error: NeurectomyError) -> bool:
    """Convenience function to handle an error."""
    return _error_handler.handle(error)
