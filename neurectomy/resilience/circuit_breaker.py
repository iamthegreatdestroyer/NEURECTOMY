"""
Circuit Breaker Pattern Implementation
Prevents cascading failures in distributed systems
"""

import asyncio
from typing import Callable, Optional, Any, Dict
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Rejecting requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 2          # Successes to close from half-open
    timeout_seconds: int = 60           # Time before attempting reset
    half_open_max_calls: int = 3        # Max calls in half-open state


class CircuitBreakerError(Exception):
    """Raised when circuit is open"""
    pass


class CircuitBreaker:
    """
    Circuit breaker for protecting services from cascading failures
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failures exceeded threshold, requests rejected
    - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0
        
        logger.info(f"Circuit breaker initialized: {name} (state={self.state.value})")
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        
        Args:
            func: Async function to execute
            *args, **kwargs: Function arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
        """
        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                logger.info(f"Circuit {self.name}: Transitioning to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                self.success_count = 0
            else:
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Service unavailable. Retry in {self._seconds_until_retry()}s"
                )
                
        # Limit calls in half-open state
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.config.half_open_max_calls:
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is HALF_OPEN. "
                    f"Max test calls ({self.config.half_open_max_calls}) reached"
                )
            self.half_open_calls += 1
            
        # Execute function
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
            
        except Exception as e:
            await self._on_failure()
            raise
            
    async def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            logger.info(
                f"Circuit {self.name}: Success in HALF_OPEN "
                f"({self.success_count}/{self.config.success_threshold})"
            )
            
            if self.success_count >= self.config.success_threshold:
                logger.info(f"Circuit {self.name}: Transitioning to CLOSED (recovered)")
                self.state = CircuitState.CLOSED
                self.success_count = 0
                self.half_open_calls = 0
                
    async def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            # Immediately re-open on failure in half-open
            logger.warning(
                f"Circuit {self.name}: Failure in HALF_OPEN, "
                f"transitioning back to OPEN"
            )
            self.state = CircuitState.OPEN
            self.success_count = 0
            self.half_open_calls = 0
            
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                logger.error(
                    f"Circuit {self.name}: Failure threshold reached "
                    f"({self.failure_count}/{self.config.failure_threshold}), "
                    f"transitioning to OPEN"
                )
                self.state = CircuitState.OPEN
                
    def _should_attempt_reset(self) -> bool:
        """Check if enough time passed to attempt reset"""
        if not self.last_failure_time:
            return False
            
        elapsed = datetime.now() - self.last_failure_time
        return elapsed.total_seconds() >= self.config.timeout_seconds
        
    def _seconds_until_retry(self) -> int:
        """Calculate seconds until retry attempt"""
        if not self.last_failure_time:
            return 0
            
        elapsed = datetime.now() - self.last_failure_time
        remaining = self.config.timeout_seconds - elapsed.total_seconds()
        return max(0, int(remaining))
        
    def get_state(self) -> Dict:
        """Get current circuit breaker state"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "seconds_until_retry": self._seconds_until_retry() if self.state == CircuitState.OPEN else 0
        }
        
    def reset(self):
        """Manually reset circuit breaker to closed state"""
        logger.info(f"Circuit {self.name}: Manual reset to CLOSED")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0


# Decorator for easy circuit breaker application
def with_circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None
):
    """
    Decorator to apply circuit breaker to async functions
    
    Example:
        @with_circuit_breaker("external_api")
        async def call_external_api():
            # Function implementation
            pass
    """
    circuit_breaker = CircuitBreaker(name, config)
    
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            return await circuit_breaker.call(func, *args, **kwargs)
        wrapper._circuit_breaker = circuit_breaker  # Expose for testing
        return wrapper
    return decorator
