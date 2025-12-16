"""
Retry Policy with Exponential Backoff
Handles transient failures gracefully
"""

import asyncio
from typing import Callable, Optional, Any, Type, Tuple
from dataclasses import dataclass
import logging
import random

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Retry policy configuration"""
    max_attempts: int = 3
    initial_delay: float = 1.0      # seconds
    max_delay: float = 60.0         # seconds
    exponential_base: float = 2.0
    jitter: bool = True             # Add randomness to prevent thundering herd
    retry_on: Tuple[Type[Exception], ...] = (Exception,)  # Exception types to retry


class RetryPolicy:
    """
    Exponential backoff retry with jitter
    
    Implements exponential backoff with optional jitter to prevent
    cascading retries (thundering herd problem).
    
    Formula: delay = min(initial_delay * (base ^ attempt), max_delay)
    With jitter: delay = delay * (0.5 + random() * 0.5)
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with retry
        
        Args:
            func: Async function to execute
            *args, **kwargs: Function arguments
        
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                result = await func(*args, **kwargs)
                
                if attempt > 0:
                    logger.info(f"Retry succeeded on attempt {attempt + 1}")
                    
                return result
                
            except self.config.retry_on as e:
                last_exception = e
                
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"All {self.config.max_attempts} attempts failed. "
                        f"Last error: {e}"
                    )
                    
        raise last_exception
        
    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate retry delay with exponential backoff and jitter
        
        Args:
            attempt: Current attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        # Exponential backoff: initial_delay * (base ^ attempt)
        delay = min(
            self.config.initial_delay * (self.config.exponential_base ** attempt),
            self.config.max_delay
        )
        
        if self.config.jitter:
            # Add jitter: randomize between 50% and 100% of calculated delay
            # This prevents thundering herd when many clients retry simultaneously
            delay = delay * (0.5 + random.random() * 0.5)
            
        return delay


def with_retry(config: Optional[RetryConfig] = None):
    """
    Decorator for automatic retry with exponential backoff
    
    Args:
        config: RetryConfig instance (optional)
        
    Example:
        @with_retry(RetryConfig(max_attempts=5))
        async def unstable_api_call():
            # May fail transiently
            pass
    """
    def decorator(func: Callable):
        policy = RetryPolicy(config)
        
        async def wrapper(*args, **kwargs):
            return await policy.execute(func, *args, **kwargs)
            
        wrapper._retry_policy = policy  # Expose for testing
        wrapper._original_func = func   # Store original function reference
        return wrapper
    return decorator
