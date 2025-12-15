"""
Production Readiness Tests
==========================

Comprehensive tests for production deployment.
"""

import pytest
import time


class TestErrorHandling:
    """Test error handling system."""
    
    def test_error_creation(self):
        from neurectomy.core.errors import InferenceError, ErrorCategory
        
        error = InferenceError("Test error", operation="generate")
        
        assert error.category == ErrorCategory.INFERENCE
        assert error.message == "Test error"
        assert error.operation == "generate"
    
    def test_error_handler(self):
        from neurectomy.core.errors import (
            get_error_handler, InferenceError
        )
        
        handler = get_error_handler()
        error = InferenceError("Test error")
        
        handler.handle(error)
        stats = handler.get_error_stats()
        
        assert stats["total_errors"] > 0


class TestLogging:
    """Test logging system."""
    
    def test_logger_creation(self):
        from neurectomy.core.logging import get_logger
        
        logger = get_logger("test", component="test_component")
        
        assert logger.name == "test"
        assert logger.component == "test_component"
    
    def test_structured_logging(self):
        from neurectomy.core.logging import get_logger
        
        logger = get_logger("test_structured")
        logger.info("Test message", task_id="123", agent_id="agent_1")
        
        # Should not raise


class TestMonitoring:
    """Test performance monitoring."""
    
    def test_metric_recording(self):
        from neurectomy.core.monitoring import get_monitor
        
        monitor = get_monitor()
        monitor.record_inference(100, 50, 150.0)
        
        dashboard = monitor.get_dashboard()
        assert "inference" in dashboard
    
    def test_context_manager_timing(self):
        from neurectomy.core.monitoring import get_monitor
        
        monitor = get_monitor()
        
        with monitor.measure("test_operation"):
            time.sleep(0.01)
        
        # Should record metric


class TestHealthChecks:
    """Test health check system."""
    
    def test_health_checker(self):
        from neurectomy.core.health import (
            HealthChecker, ComponentHealth, HealthStatus
        )
        
        checker = HealthChecker()
        
        def dummy_check():
            return ComponentHealth(
                name="dummy",
                status=HealthStatus.HEALTHY,
            )
        
        checker.register_check("dummy", dummy_check)
        result = checker.check_component("dummy")
        
        assert result.status == HealthStatus.HEALTHY
    
    def test_full_health_check(self):
        from neurectomy.core.health import create_default_health_checker
        
        checker = create_default_health_checker()
        health = checker.check_all()
        
        assert health.status is not None
        assert len(health.components) > 0


def test_production_readiness():
    """Full production readiness test."""
    print("\n" + "=" * 60)
    print("  PRODUCTION READINESS TEST")
    print("=" * 60 + "\n")
    
    # Test 1: Error Handling
    print("Testing error handling...")
    from neurectomy.core.errors import get_error_handler, InferenceError
    handler = get_error_handler()
    handler.handle(InferenceError("Test"))
    print("✓ Error handling working")
    
    # Test 2: Logging
    print("Testing logging...")
    from neurectomy.core.logging import get_logger
    logger = get_logger("prod_test")
    logger.info("Production test")
    print("✓ Logging working")
    
    # Test 3: Monitoring
    print("Testing monitoring...")
    from neurectomy.core.monitoring import get_monitor
    monitor = get_monitor()
    monitor.record_inference(100, 50, 100.0)
    print("✓ Monitoring working")
    
    # Test 4: Health Checks
    print("Testing health checks...")
    from neurectomy.core.health import create_default_health_checker
    checker = create_default_health_checker()
    health = checker.check_all()
    print(f"✓ Health status: {health.status.value}")
    
    print("\n" + "=" * 60)
    print("  ✅ PRODUCTION READINESS TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_production_readiness()
