"""
Retry Policy Tests
"""

import pytest
import asyncio
from neurectomy.resilience.retry_policies import (
    RetryPolicy,
    RetryConfig,
    with_retry
)


@pytest.mark.asyncio
async def test_retry_succeeds_on_first_attempt():
    """Test that retry succeeds immediately if no failure"""
    config = RetryConfig(max_attempts=3)
    policy = RetryPolicy(config)
    
    call_count = 0
    
    async def successful_function():
        nonlocal call_count
        call_count += 1
        return "success"
    
    result = await policy.execute(successful_function)
    
    assert result == "success"
    assert call_count == 1


@pytest.mark.asyncio
async def test_retry_succeeds_after_transient_failure():
    """Test that retry succeeds after transient failure"""
    config = RetryConfig(max_attempts=3, initial_delay=0.1)
    policy = RetryPolicy(config)
    
    call_count = 0
    
    async def sometimes_failing_function():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ConnectionError("Transient failure")
        return "success"
    
    result = await policy.execute(sometimes_failing_function)
    
    assert result == "success"
    assert call_count == 2


@pytest.mark.asyncio
async def test_retry_fails_after_max_attempts():
    """Test that retry raises exception after max attempts"""
    config = RetryConfig(max_attempts=3, initial_delay=0.01)
    policy = RetryPolicy(config)
    
    call_count = 0
    
    async def always_failing_function():
        nonlocal call_count
        call_count += 1
        raise ValueError("Persistent failure")
    
    with pytest.raises(ValueError, match="Persistent failure"):
        await policy.execute(always_failing_function)
    
    assert call_count == 3


@pytest.mark.asyncio
async def test_exponential_backoff_delay():
    """Test that delays increase exponentially"""
    config = RetryConfig(
        max_attempts=4,
        initial_delay=1.0,
        exponential_base=2.0,
        jitter=False  # Disable jitter for predictable testing
    )
    policy = RetryPolicy(config)
    
    # Calculate expected delays
    expected_delays = [
        1.0 * (2.0 ** 0),  # 1.0
        1.0 * (2.0 ** 1),  # 2.0
        1.0 * (2.0 ** 2),  # 4.0
    ]
    
    for i, expected_delay in enumerate(expected_delays):
        delay = policy._calculate_delay(i)
        assert delay == expected_delay


@pytest.mark.asyncio
async def test_max_delay_cap():
    """Test that max delay cap is enforced"""
    config = RetryConfig(
        max_attempts=10,
        initial_delay=1.0,
        exponential_base=2.0,
        max_delay=10.0,
        jitter=False
    )
    policy = RetryPolicy(config)
    
    # After enough attempts, delay should be capped at max_delay
    for attempt in range(10):
        delay = policy._calculate_delay(attempt)
        assert delay <= config.max_delay


@pytest.mark.asyncio
async def test_jitter_adds_randomness():
    """Test that jitter adds randomness to delays"""
    config = RetryConfig(
        max_attempts=3,
        initial_delay=10.0,
        exponential_base=2.0,
        jitter=True
    )
    policy = RetryPolicy(config)
    
    # Get multiple delay calculations for same attempt
    delays = []
    for _ in range(10):
        delay = policy._calculate_delay(1)
        delays.append(delay)
    
    # Delays should vary (with jitter enabled)
    unique_delays = len(set(delays))
    assert unique_delays > 1  # Should have different values


@pytest.mark.asyncio
async def test_jitter_range():
    """Test that jitter keeps delay within expected range"""
    config = RetryConfig(
        max_attempts=3,
        initial_delay=10.0,
        exponential_base=1.0,  # No exponential growth
        jitter=True
    )
    policy = RetryPolicy(config)
    
    # With jitter, delay should be between 50% and 100% of base
    expected_min = config.initial_delay * 0.5
    expected_max = config.initial_delay * 1.0
    
    for _ in range(100):
        delay = policy._calculate_delay(0)
        assert expected_min <= delay <= expected_max


@pytest.mark.asyncio
async def test_retry_specific_exception_types():
    """Test that retry only retries specified exception types"""
    config = RetryConfig(
        max_attempts=3,
        initial_delay=0.01,
        retry_on=(ConnectionError, TimeoutError)
    )
    policy = RetryPolicy(config)
    
    call_count = 0
    
    async def function_with_non_retriable_error():
        nonlocal call_count
        call_count += 1
        raise ValueError("Non-retriable error")
    
    # Should not retry ValueError
    with pytest.raises(ValueError):
        await policy.execute(function_with_non_retriable_error)
    
    assert call_count == 1  # Only one attempt


@pytest.mark.asyncio
async def test_decorator_application():
    """Test retry decorator application"""
    call_count = 0
    
    @with_retry(RetryConfig(max_attempts=3, initial_delay=0.01))
    async def sometimes_failing_function():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ConnectionError("Transient failure")
        return "success"
    
    result = await sometimes_failing_function()
    
    assert result == "success"
    assert call_count == 2


@pytest.mark.asyncio
async def test_decorator_exposes_policy():
    """Test that decorator exposes retry policy"""
    @with_retry(RetryConfig(max_attempts=3))
    async def test_function():
        return "success"
    
    # Should have _retry_policy attribute
    assert hasattr(test_function, '_retry_policy')
    assert isinstance(test_function._retry_policy, RetryPolicy)


@pytest.mark.asyncio
async def test_decorator_with_function_args():
    """Test retry decorator with function arguments"""
    call_count = 0
    
    @with_retry(RetryConfig(max_attempts=3, initial_delay=0.01))
    async def function_with_args(x, y, z=None):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ConnectionError("Transient failure")
        return x + y + (z or 0)
    
    result = await function_with_args(1, 2, z=3)
    
    assert result == 6
    assert call_count == 2


@pytest.mark.asyncio
async def test_retry_timing():
    """Test that actual retry delays follow expected pattern"""
    config = RetryConfig(
        max_attempts=3,
        initial_delay=0.1,
        exponential_base=2.0,
        jitter=False
    )
    policy = RetryPolicy(config)
    
    call_count = 0
    start_time = asyncio.get_event_loop().time()
    
    async def always_failing_function():
        nonlocal call_count
        call_count += 1
        raise ValueError("Failure")
    
    try:
        await policy.execute(always_failing_function)
    except ValueError:
        pass
    
    elapsed = asyncio.get_event_loop().time() - start_time
    
    # Should have waited at least: 0.1 + 0.2 = 0.3 seconds
    # (before giving up on 3rd attempt)
    assert elapsed >= 0.25  # Allow some margin
    assert call_count == 3


def test_retry_config_defaults():
    """Test RetryConfig default values"""
    config = RetryConfig()
    
    assert config.max_attempts == 3
    assert config.initial_delay == 1.0
    assert config.max_delay == 60.0
    assert config.exponential_base == 2.0
    assert config.jitter is True
    assert config.retry_on == (Exception,)
