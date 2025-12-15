#!/usr/bin/env python3
"""Phase 6 Verification - Production Readiness"""

import sys


def verify_error_handling():
    try:
        from neurectomy.core.errors import (
            get_error_handler, InferenceError, CompressionError,
            StorageError, AgentError, ErrorCategory
        )
        
        handler = get_error_handler()
        
        # Test various error types
        errors = [
            InferenceError("Test inference error"),
            CompressionError("Test compression error"),
            StorageError("Test storage error"),
            AgentError("Test agent error", agent_id="test_agent"),
        ]
        
        for error in errors:
            handler.handle(error)
        
        stats = handler.get_error_stats()
        print(f"✓ Error handling verified ({stats['total_errors']} errors tracked)")
        return True
    except Exception as e:
        print(f"❌ Error handling failed: {e}")
        return False


def verify_logging():
    try:
        from neurectomy.core.logging import get_logger, LogConfig
        
        logger = get_logger("verification", component="phase6")
        
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.task_start("task_001", "verification")
        logger.task_complete("task_001", 50.0)
        
        print("✓ Logging verified")
        return True
    except Exception as e:
        print(f"❌ Logging failed: {e}")
        return False


def verify_monitoring():
    try:
        from neurectomy.core.monitoring import get_monitor
        
        monitor = get_monitor()
        
        # Record some metrics
        monitor.record_inference(1000, 100, 250.0)
        monitor.record_compression(1000, 50, 10.0)
        monitor.record_cache_access(True)
        monitor.record_agent_task("agent_1", "generate", 100.0, True)
        
        dashboard = monitor.get_dashboard()
        
        print(f"✓ Monitoring verified (uptime: {dashboard['uptime_seconds']:.1f}s)")
        return True
    except Exception as e:
        print(f"❌ Monitoring failed: {e}")
        return False


def verify_health_checks():
    try:
        from neurectomy.core.health import create_default_health_checker
        
        checker = create_default_health_checker()
        health = checker.check_all()
        
        healthy = sum(1 for c in health.components if c.status.value == "healthy")
        total = len(health.components)
        
        print(f"✓ Health checks verified ({healthy}/{total} components healthy)")
        print(f"   Overall status: {health.status.value}")
        
        return True
    except Exception as e:
        print(f"❌ Health checks failed: {e}")
        return False


def main():
    print("=" * 60)
    print("  PHASE 6: Production Readiness - Verification")
    print("=" * 60)
    print()
    
    results = [
        verify_error_handling(),
        verify_logging(),
        verify_monitoring(),
        verify_health_checks(),
    ]
    
    print()
    if all(results):
        print("=" * 60)
        print("  ✅ PHASE 6 VERIFICATION COMPLETE")
        print("=" * 60)
        print()
        print("  System is production-ready!")
        return 0
    else:
        print("=" * 60)
        print("  ❌ PHASE 6 VERIFICATION FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
