"""
Fallback Manager for Graceful Degradation
Provides multiple fallback strategies when primary operations fail
"""

from typing import Callable, Optional, Any, List
from enum import Enum
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class FallbackType(Enum):
    """Types of fallback strategies"""
    CACHE = "cache"
    DEFAULT = "default"
    ALTERNATIVE_SERVICE = "alternative_service"
    DEGRADED_RESPONSE = "degraded_response"


@dataclass
class FallbackStrategy:
    """Configuration for a fallback strategy"""
    name: str
    fallback_type: FallbackType
    handler: Callable
    priority: int = 0  # Lower number = higher priority


class FallbackManager:
    """
    Manages fallback strategies for graceful degradation
    
    Implements a priority-based fallback chain that attempts
    multiple strategies when the primary function fails.
    
    Example:
        manager = FallbackManager()
        
        # Register fallback strategies
        manager.register_strategy(FallbackStrategy(
            name="cache",
            fallback_type=FallbackType.CACHE,
            handler=get_from_cache,
            priority=1
        ))
        
        manager.register_strategy(FallbackStrategy(
            name="default",
            fallback_type=FallbackType.DEFAULT,
            handler=lambda: "Default response",
            priority=2
        ))
        
        # Execute with fallback
        result = await manager.execute_with_fallback(
            primary_function,
            fallback_context={"key": "user_123"}
        )
    """
    
    def __init__(self):
        self.strategies: List[FallbackStrategy] = []
        
    def register_strategy(self, strategy: FallbackStrategy):
        """
        Register a fallback strategy
        
        Args:
            strategy: FallbackStrategy configuration
        """
        self.strategies.append(strategy)
        # Sort by priority (lower number first)
        self.strategies.sort(key=lambda s: s.priority)
        logger.info(f"Registered fallback strategy: {strategy.name} (priority={strategy.priority})")
        
    async def execute_with_fallback(
        self,
        primary_func: Callable,
        *args,
        fallback_context: Optional[dict] = None,
        **kwargs
    ) -> Any:
        """
        Execute primary function with automatic fallback
        
        Args:
            primary_func: Primary function to execute
            *args, **kwargs: Arguments for primary function
            fallback_context: Context data for fallback handlers
            
        Returns:
            Result from primary function or fallback
            
        Raises:
            Exception: If all attempts (primary + fallbacks) fail
        """
        fallback_context = fallback_context or {}
        
        # Try primary function
        try:
            result = await primary_func(*args, **kwargs)
            logger.debug("Primary function succeeded")
            return result
            
        except Exception as primary_error:
            logger.warning(f"Primary function failed: {primary_error}")
            
            # Try fallback strategies in priority order
            for strategy in self.strategies:
                try:
                    logger.info(f"Attempting fallback: {strategy.name}")
                    
                    result = await strategy.handler(fallback_context)
                    
                    logger.info(f"Fallback succeeded: {strategy.name}")
                    return result
                    
                except Exception as fallback_error:
                    logger.warning(
                        f"Fallback {strategy.name} failed: {fallback_error}"
                    )
                    continue
                    
            # All fallbacks failed
            logger.error("All fallback strategies failed")
            raise Exception(
                f"Primary function and all fallbacks failed. "
                f"Primary error: {primary_error}"
            )


# Predefined fallback handlers

async def cache_fallback(context: dict) -> Any:
    """
    Cache fallback handler
    
    Context should contain:
    - cache_key: Key to lookup
    - cache_client: Cache client instance
    
    Args:
        context: Fallback context dictionary
        
    Returns:
        Cached value
        
    Raises:
        ValueError: If required context missing or cache miss
    """
    cache_key = context.get("cache_key")
    cache_client = context.get("cache_client")
    
    if not cache_key or not cache_client:
        raise ValueError("Cache fallback requires cache_key and cache_client")
        
    cached_value = await cache_client.get(cache_key)
    
    if cached_value is None:
        raise ValueError(f"No cached value for key: {cache_key}")
        
    logger.info(f"Serving from cache: {cache_key}")
    return cached_value


async def default_value_fallback(context: dict) -> Any:
    """
    Default value fallback
    
    Context should contain:
    - default_value: Value to return
    
    Args:
        context: Fallback context dictionary
        
    Returns:
        Default value
        
    Raises:
        ValueError: If default_value not in context
    """
    default_value = context.get("default_value")
    
    if default_value is None:
        raise ValueError("Default value fallback requires default_value")
        
    logger.info("Serving default value")
    return default_value


async def alternative_service_fallback(context: dict) -> Any:
    """
    Alternative service fallback
    
    Context should contain:
    - alternative_func: Alternative function to call
    - args: Arguments for alternative function (optional)
    - kwargs: Keyword arguments for alternative function (optional)
    
    Args:
        context: Fallback context dictionary
        
    Returns:
        Result from alternative function
        
    Raises:
        ValueError: If alternative_func not in context
    """
    alternative_func = context.get("alternative_func")
    args = context.get("args", ())
    kwargs = context.get("kwargs", {})
    
    if not alternative_func:
        raise ValueError("Alternative service fallback requires alternative_func")
        
    logger.info("Calling alternative service")
    return await alternative_func(*args, **kwargs)


async def degraded_response_fallback(context: dict) -> Any:
    """
    Degraded response fallback
    
    Returns a degraded but still functional response.
    
    Context should contain:
    - degraded_response: Degraded response to return
    
    Args:
        context: Fallback context dictionary
        
    Returns:
        Degraded response
        
    Raises:
        ValueError: If degraded_response not in context
    """
    degraded_response = context.get("degraded_response")
    
    if degraded_response is None:
        raise ValueError("Degraded response fallback requires degraded_response")
        
    logger.info("Serving degraded response")
    return degraded_response


# Convenience decorator

def with_fallback(strategies: List[FallbackStrategy]):
    """
    Decorator to add fallback to async functions
    
    Args:
        strategies: List of FallbackStrategy instances
    
    Example:
        @with_fallback([
            FallbackStrategy("cache", FallbackType.CACHE, cache_handler, 1),
            FallbackStrategy("default", FallbackType.DEFAULT, default_handler, 2)
        ])
        async def get_data(key):
            # Primary implementation
            pass
    """
    def decorator(func: Callable):
        manager = FallbackManager()
        for strategy in strategies:
            manager.register_strategy(strategy)
            
        async def wrapper(*args, **kwargs):
            # Extract fallback context from kwargs if present
            fallback_context = kwargs.pop('fallback_context', {})
            return await manager.execute_with_fallback(
                func,
                *args,
                fallback_context=fallback_context,
                **kwargs
            )
            
        wrapper._fallback_manager = manager  # Expose for testing
        wrapper._original_func = func        # Store original function reference
        return wrapper
    return decorator
