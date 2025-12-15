"""
Neurectomy Logging System
=========================

Structured logging for all components.
"""

import logging
import json
import sys
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum


class LogLevel(Enum):
    """Log levels."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class LogConfig:
    """Logging configuration."""
    
    def __init__(
        self,
        level: LogLevel = LogLevel.INFO,
        console_enabled: bool = True,
        file_enabled: bool = True,
        file_path: str = "logs/neurectomy.log",
        json_format: bool = True,
        include_timestamp: bool = True,
        include_component: bool = True,
        max_file_size_mb: int = 100,
        backup_count: int = 5,
        log_performance: bool = True,
        performance_threshold_ms: float = 100.0,
    ):
        self.level = level
        self.console_enabled = console_enabled
        self.file_enabled = file_enabled
        self.file_path = file_path
        self.json_format = json_format
        self.include_timestamp = include_timestamp
        self.include_component = include_component
        self.max_file_size_mb = max_file_size_mb
        self.backup_count = backup_count
        self.log_performance = log_performance
        self.performance_threshold_ms = performance_threshold_ms


class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields
        if hasattr(record, "component"):
            log_data["component"] = record.component
        if hasattr(record, "task_id"):
            log_data["task_id"] = record.task_id
        if hasattr(record, "agent_id"):
            log_data["agent_id"] = record.agent_id
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
        if hasattr(record, "error_data"):
            log_data["error"] = record.error_data
        if hasattr(record, "metrics"):
            log_data["metrics"] = record.metrics
        
        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


class NeurectomyLogger:
    """
    Enhanced logger for Neurectomy components.
    
    Provides structured logging with component context.
    """
    
    def __init__(
        self,
        name: str,
        component: Optional[str] = None,
        config: Optional[LogConfig] = None,
    ):
        self.name = name
        self.component = component
        self.config = config or LogConfig()
        
        self._logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Configure the logger."""
        self._logger.setLevel(self.config.level.value)
        
        # Clear existing handlers
        self._logger.handlers.clear()
        
        # Console handler
        if self.config.console_enabled:
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(self.config.level.value)
            
            if self.config.json_format:
                console.setFormatter(StructuredFormatter())
            else:
                console.setFormatter(logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                ))
            
            self._logger.addHandler(console)
        
        # File handler
        if self.config.file_enabled:
            log_path = Path(self.config.file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            from logging.handlers import RotatingFileHandler
            
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count,
            )
            file_handler.setLevel(self.config.level.value)
            file_handler.setFormatter(StructuredFormatter())
            
            self._logger.addHandler(file_handler)
    
    def _log(
        self,
        level: int,
        message: str,
        **kwargs,
    ) -> None:
        """Internal log method with extra context."""
        extra = {"component": self.component}
        extra.update(kwargs)
        self._logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs) -> None:
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        self._log(logging.CRITICAL, message, **kwargs)
    
    def task_start(self, task_id: str, task_type: str, **kwargs) -> None:
        """Log task start."""
        self.info(f"Task started: {task_type}", task_id=task_id, **kwargs)
    
    def task_complete(
        self,
        task_id: str,
        duration_ms: float,
        **kwargs,
    ) -> None:
        """Log task completion."""
        self.info(
            f"Task completed in {duration_ms:.1f}ms",
            task_id=task_id,
            duration_ms=duration_ms,
            **kwargs,
        )
    
    def task_failed(self, task_id: str, error: str, **kwargs) -> None:
        """Log task failure."""
        self.error(f"Task failed: {error}", task_id=task_id, **kwargs)
    
    def performance(
        self,
        operation: str,
        duration_ms: float,
        **metrics,
    ) -> None:
        """Log performance metrics."""
        if not self.config.log_performance:
            return
        
        if duration_ms >= self.config.performance_threshold_ms:
            self.warning(
                f"Slow operation: {operation} ({duration_ms:.1f}ms)",
                duration_ms=duration_ms,
                metrics=metrics,
            )
        else:
            self.debug(
                f"Operation: {operation} ({duration_ms:.1f}ms)",
                duration_ms=duration_ms,
                metrics=metrics,
            )
    
    def agent_action(
        self,
        agent_id: str,
        action: str,
        **kwargs,
    ) -> None:
        """Log agent action."""
        self.info(f"Agent {agent_id}: {action}", agent_id=agent_id, **kwargs)


# Logger factory
_loggers: Dict[str, NeurectomyLogger] = {}


def get_logger(
    name: str,
    component: Optional[str] = None,
) -> NeurectomyLogger:
    """Get or create a logger."""
    key = f"{name}:{component or 'default'}"
    
    if key not in _loggers:
        _loggers[key] = NeurectomyLogger(name, component)
    
    return _loggers[key]
