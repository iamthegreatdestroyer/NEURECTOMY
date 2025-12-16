"""
Fallback Manager Tests
"""

import pytest
from neurectomy.resilience.fallback_manager import (
    FallbackManager,
    FallbackStrategy,
    FallbackType,
    cache_fallback,
    default_value_fallback,
    alternative_service_fallback,
    degraded_response_fallback,
    with_fallback
)


@pytest.mark.asyncio
async def test_fallback_manager_primary_success():
    """Test that primary function succeeds without fallback"""
    manager = FallbackManager()
    
    async def successful_primary():
        return "primary_result"
    
    result = await manager.execute_with_fallback(successful_primary)
    assert result == "primary_result"


@pytest.mark.asyncio
async def test_fallback_manager_no_strategies():
    """Test that exception raised when no strategies registered"""
    manager = FallbackManager()
    
    async def failing_primary():
        raise ValueError("Primary failed")
    
    with pytest.raises(Exception, match="Primary function and all fallbacks failed"):
        await manager.execute_with_fallback(failing_primary)


@pytest.mark.asyncio
async def test_fallback_to_single_strategy():
    """Test fallback to single strategy"""
    manager = FallbackManager()
    
    async def mock_fallback_handler(context):
        return "fallback_result"
    
    manager.register_strategy(FallbackStrategy(
        name="mock",
        fallback_type=FallbackType.DEFAULT,
        handler=mock_fallback_handler,
        priority=1
    ))
    
    async def failing_primary():
        raise ValueError("Primary failed")
    
    result = await manager.execute_with_fallback(failing_primary)
    assert result == "fallback_result"


@pytest.mark.asyncio
async def test_fallback_priority_order():
    """Test that fallbacks are tried in priority order"""
    manager = FallbackManager()
    
    call_order = []
    
    async def fallback_1(context):
        call_order.append(1)
        raise ValueError("Fallback 1 failed")
    
    async def fallback_2(context):
        call_order.append(2)
        return "fallback_2_result"
    
    async def fallback_3(context):
        call_order.append(3)
        return "fallback_3_result"
    
    manager.register_strategy(FallbackStrategy("f3", FallbackType.DEFAULT, fallback_3, priority=3))
    manager.register_strategy(FallbackStrategy("f1", FallbackType.DEFAULT, fallback_1, priority=1))
    manager.register_strategy(FallbackStrategy("f2", FallbackType.DEFAULT, fallback_2, priority=2))
    
    async def failing_primary():
        raise ValueError("Primary failed")
    
    result = await manager.execute_with_fallback(failing_primary)
    
    # Should try in priority order: 1, then 2
    assert call_order == [1, 2]
    assert result == "fallback_2_result"


@pytest.mark.asyncio
async def test_cache_fallback():
    """Test cache fallback handler"""
    class MockCache:
        async def get(self, key):
            if key == "cached_key":
                return "cached_value"
            return None
    
    context = {
        "cache_key": "cached_key",
        "cache_client": MockCache()
    }
    
    result = await cache_fallback(context)
    assert result == "cached_value"


@pytest.mark.asyncio
async def test_cache_fallback_missing_context():
    """Test cache fallback with missing context"""
    with pytest.raises(ValueError, match="requires cache_key and cache_client"):
        await cache_fallback({})


@pytest.mark.asyncio
async def test_cache_fallback_cache_miss():
    """Test cache fallback with cache miss"""
    class MockCache:
        async def get(self, key):
            return None
    
    context = {
        "cache_key": "missing_key",
        "cache_client": MockCache()
    }
    
    with pytest.raises(ValueError, match="No cached value"):
        await cache_fallback(context)


@pytest.mark.asyncio
async def test_default_value_fallback():
    """Test default value fallback"""
    context = {"default_value": "default_result"}
    
    result = await default_value_fallback(context)
    assert result == "default_result"


@pytest.mark.asyncio
async def test_default_value_fallback_missing():
    """Test default value fallback with missing value"""
    with pytest.raises(ValueError, match="requires default_value"):
        await default_value_fallback({})


@pytest.mark.asyncio
async def test_alternative_service_fallback():
    """Test alternative service fallback"""
    async def alternative_service(x, y):
        return x + y
    
    context = {
        "alternative_func": alternative_service,
        "args": (5, 3),
        "kwargs": {}
    }
    
    result = await alternative_service_fallback(context)
    assert result == 8


@pytest.mark.asyncio
async def test_alternative_service_fallback_missing():
    """Test alternative service fallback with missing function"""
    with pytest.raises(ValueError, match="requires alternative_func"):
        await alternative_service_fallback({})


@pytest.mark.asyncio
async def test_degraded_response_fallback():
    """Test degraded response fallback"""
    degraded_resp = {"status": "degraded", "data": []}
    context = {"degraded_response": degraded_resp}
    
    result = await degraded_response_fallback(context)
    assert result == degraded_resp


@pytest.mark.asyncio
async def test_degraded_response_fallback_missing():
    """Test degraded response fallback with missing response"""
    with pytest.raises(ValueError, match="requires degraded_response"):
        await degraded_response_fallback({})


@pytest.mark.asyncio
async def test_decorator_with_fallback():
    """Test @with_fallback decorator"""
    call_count = 0
    
    async def fallback_handler(context):
        return "fallback_result"
    
    @with_fallback([
        FallbackStrategy("fallback", FallbackType.DEFAULT, fallback_handler, 1)
    ])
    async def sometimes_failing():
        nonlocal call_count
        call_count += 1
        raise ValueError("Failed")
    
    result = await sometimes_failing()
    
    assert result == "fallback_result"
    assert call_count == 1


@pytest.mark.asyncio
async def test_decorator_with_fallback_context():
    """Test @with_fallback decorator with fallback_context"""
    async def context_aware_fallback(context):
        return f"fallback_{context.get('key')}"
    
    @with_fallback([
        FallbackStrategy("fallback", FallbackType.DEFAULT, context_aware_fallback, 1)
    ])
    async def failing_function():
        raise ValueError("Failed")
    
    result = await failing_function(fallback_context={"key": "test123"})
    
    assert result == "fallback_test123"


@pytest.mark.asyncio
async def test_decorator_exposes_manager():
    """Test that decorator exposes fallback manager"""
    async def mock_fallback(context):
        return "fallback"
    
    @with_fallback([
        FallbackStrategy("mock", FallbackType.DEFAULT, mock_fallback, 1)
    ])
    async def test_function():
        return "primary"
    
    assert hasattr(test_function, '_fallback_manager')
    assert isinstance(test_function._fallback_manager, FallbackManager)


@pytest.mark.asyncio
async def test_fallback_with_multiple_strategies():
    """Test fallback chain with multiple strategies"""
    manager = FallbackManager()
    
    # Register multiple fallbacks
    async def cache_handler(context):
        if context.get("has_cache"):
            return "cached"
        raise ValueError("No cache")
    
    async def default_handler(context):
        return "default"
    
    manager.register_strategy(FallbackStrategy("cache", FallbackType.CACHE, cache_handler, 1))
    manager.register_strategy(FallbackStrategy("default", FallbackType.DEFAULT, default_handler, 2))
    
    async def failing_primary():
        raise ValueError("Primary failed")
    
    # Try with no cache - should fall through to default
    result = await manager.execute_with_fallback(
        failing_primary,
        fallback_context={"has_cache": False}
    )
    
    assert result == "default"


@pytest.mark.asyncio
async def test_fallback_preserves_function_info():
    """Test that decorator preserves original function reference"""
    async def original_function():
        return "original"
    
    @with_fallback([])
    async def decorated():
        return await original_function()
    
    assert hasattr(decorated, '_original_func')


def test_fallback_strategy_dataclass():
    """Test FallbackStrategy dataclass"""
    async def handler(context):
        return "result"
    
    strategy = FallbackStrategy(
        name="test",
        fallback_type=FallbackType.CACHE,
        handler=handler,
        priority=5
    )
    
    assert strategy.name == "test"
    assert strategy.fallback_type == FallbackType.CACHE
    assert strategy.handler == handler
    assert strategy.priority == 5


def test_fallback_type_enum():
    """Test FallbackType enum values"""
    assert FallbackType.CACHE.value == "cache"
    assert FallbackType.DEFAULT.value == "default"
    assert FallbackType.ALTERNATIVE_SERVICE.value == "alternative_service"
    assert FallbackType.DEGRADED_RESPONSE.value == "degraded_response"
