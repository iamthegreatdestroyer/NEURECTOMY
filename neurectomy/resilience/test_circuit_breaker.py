"""
Circuit Breaker Tests
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from neurectomy.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitBreakerConfig,
    CircuitState
)


@pytest.mark.asyncio
async def test_circuit_breaker_initialized_closed():
    """Test that circuit breaker starts in CLOSED state"""
    circuit = CircuitBreaker("test_service")
    
    assert circuit.state == CircuitState.CLOSED
    assert circuit.failure_count == 0
    assert circuit.success_count == 0


@pytest.mark.asyncio
async def test_circuit_breaker_opens_on_failures():
    """Test that circuit opens after failure threshold"""
    config = CircuitBreakerConfig(failure_threshold=3)
    circuit = CircuitBreaker("test_service", config)
    
    async def failing_function():
        raise ValueError("Service error")
    
    # Trigger failures up to threshold
    for i in range(3):
        with pytest.raises(ValueError):
            await circuit.call(failing_function)
        assert circuit.failure_count == i + 1
    
    # Circuit should now be open
    assert circuit.state == CircuitState.OPEN
    
    # Further calls should raise CircuitBreakerError
    with pytest.raises(CircuitBreakerError):
        await circuit.call(failing_function)


@pytest.mark.asyncio
async def test_circuit_breaker_rejects_calls_when_open():
    """Test that circuit rejects calls when open"""
    config = CircuitBreakerConfig(failure_threshold=1)
    circuit = CircuitBreaker("test_service", config)
    
    async def failing_function():
        raise ValueError("Service error")
    
    # Open the circuit
    with pytest.raises(ValueError):
        await circuit.call(failing_function)
    
    assert circuit.state == CircuitState.OPEN
    
    # Circuit should reject multiple calls
    for _ in range(5):
        with pytest.raises(CircuitBreakerError):
            await circuit.call(failing_function)


@pytest.mark.asyncio
async def test_circuit_breaker_transitions_to_half_open():
    """Test that circuit transitions to HALF_OPEN after timeout"""
    config = CircuitBreakerConfig(
        failure_threshold=1,
        timeout_seconds=1
    )
    circuit = CircuitBreaker("test_service", config)
    
    async def failing_function():
        raise ValueError("Service error")
    
    # Open the circuit
    with pytest.raises(ValueError):
        await circuit.call(failing_function)
    
    assert circuit.state == CircuitState.OPEN
    
    # Wait for timeout
    await asyncio.sleep(1.1)
    
    # Next call should transition to HALF_OPEN (and fail)
    with pytest.raises(ValueError):
        async def recovering_function():
            return "success"
        await circuit.call(recovering_function)
    
    assert circuit.state == CircuitState.HALF_OPEN


@pytest.mark.asyncio
async def test_circuit_breaker_closes_after_recovery():
    """Test that circuit closes after successful recovery"""
    config = CircuitBreakerConfig(
        failure_threshold=1,
        success_threshold=2,
        timeout_seconds=1
    )
    circuit = CircuitBreaker("test_service", config)
    
    async def failing_function():
        raise ValueError("Service error")
    
    async def recovering_function():
        return "success"
    
    # Open the circuit
    with pytest.raises(ValueError):
        await circuit.call(failing_function)
    
    assert circuit.state == CircuitState.OPEN
    
    # Wait for timeout
    await asyncio.sleep(1.1)
    
    # Successful calls in HALF_OPEN
    result1 = await circuit.call(recovering_function)
    assert result1 == "success"
    assert circuit.state == CircuitState.HALF_OPEN
    
    result2 = await circuit.call(recovering_function)
    assert result2 == "success"
    assert circuit.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_circuit_breaker_reopens_on_failure_in_half_open():
    """Test that circuit re-opens on failure in HALF_OPEN state"""
    config = CircuitBreakerConfig(
        failure_threshold=1,
        success_threshold=2,
        timeout_seconds=1
    )
    circuit = CircuitBreaker("test_service", config)
    
    async def failing_function():
        raise ValueError("Service error")
    
    # Open the circuit
    with pytest.raises(ValueError):
        await circuit.call(failing_function)
    
    assert circuit.state == CircuitState.OPEN
    
    # Wait for timeout
    await asyncio.sleep(1.1)
    
    # Fail in HALF_OPEN
    with pytest.raises(ValueError):
        await circuit.call(failing_function)
    
    assert circuit.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_circuit_breaker_decorator():
    """Test circuit breaker as a decorator"""
    config = CircuitBreakerConfig(failure_threshold=2)
    
    call_count = 0
    
    @pytest.mark.asyncio
    async def test_func():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("Error")
        return "success"
    
    # Apply decorator manually (since we can't decorate inside test)
    from neurectomy.resilience.circuit_breaker import with_circuit_breaker
    decorated = with_circuit_breaker("test", config)(test_func)
    
    # First two calls fail and open circuit
    with pytest.raises(ValueError):
        await decorated()
    
    with pytest.raises(ValueError):
        await decorated()
    
    # Third call is rejected by circuit
    with pytest.raises(CircuitBreakerError):
        await decorated()
    
    # Circuit should have exposed _circuit_breaker
    assert hasattr(decorated, '_circuit_breaker')
    assert decorated._circuit_breaker.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_circuit_breaker_manual_reset():
    """Test manual reset of circuit breaker"""
    config = CircuitBreakerConfig(failure_threshold=1)
    circuit = CircuitBreaker("test_service", config)
    
    async def failing_function():
        raise ValueError("Service error")
    
    # Open the circuit
    with pytest.raises(ValueError):
        await circuit.call(failing_function)
    
    assert circuit.state == CircuitState.OPEN
    
    # Manual reset
    circuit.reset()
    
    assert circuit.state == CircuitState.CLOSED
    assert circuit.failure_count == 0


def test_circuit_breaker_get_state():
    """Test getting circuit breaker state"""
    config = CircuitBreakerConfig(failure_threshold=5)
    circuit = CircuitBreaker("test_service", config)
    
    state = circuit.get_state()
    
    assert state["name"] == "test_service"
    assert state["state"] == "closed"
    assert state["failure_count"] == 0
    assert state["success_count"] == 0
    assert state["last_failure"] is None


def test_circuit_breaker_seconds_until_retry():
    """Test seconds until retry calculation"""
    config = CircuitBreakerConfig(timeout_seconds=10)
    circuit = CircuitBreaker("test_service", config)
    
    # No failures yet
    assert circuit._seconds_until_retry() == 0
    
    # Simulate failure
    circuit.last_failure_time = datetime.now() - timedelta(seconds=3)
    
    # Should be approximately 7 seconds remaining
    remaining = circuit._seconds_until_retry()
    assert 6 <= remaining <= 7
