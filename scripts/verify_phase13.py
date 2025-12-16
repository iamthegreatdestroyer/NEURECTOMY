#!/usr/bin/env python3
"""Phase 13 Verification - Enterprise Features."""

import sys
from pathlib import Path


def verify_13a():
    """Verify Phase 13A: Multi-Tenancy."""
    print("\nPhase 13A: Multi-Tenancy")
    print("-" * 60)
    
    passed = 0
    failed = 0
    
    try:
        from neurectomy.tenants.models import Tenant, TenantTier, TenantManager
        t = Tenant(name="Test", tier=TenantTier.STARTER)
        assert t.is_within_quota
        assert t.get_remaining_quota() > 0
        print(f"  ✓ Tenant model: Created tenant {t.id[:8]}...")
        passed += 1
    except Exception as e:
        print(f"  ❌ Tenant model: {e}")
        failed += 1
    
    try:
        from neurectomy.tenants.models import TenantManager
        mgr = TenantManager()
        t = mgr.create_tenant("Test Corp", TenantTier.PROFESSIONAL)
        assert len(mgr.list_tenants()) == 1
        print(f"  ✓ Tenant manager: {len(mgr.list_tenants())} tenants")
        passed += 1
    except Exception as e:
        print(f"  ❌ Tenant manager: {e}")
        failed += 1
    
    try:
        from neurectomy.tenants.billing import UsageTracker
        tracker = UsageTracker()
        tracker.record("t1", 100, 50, 100.5, "/api/generate")
        usage = tracker.get_usage_today("t1")
        assert usage == 150
        print(f"  ✓ Usage tracking: {usage} tokens tracked")
        passed += 1
    except Exception as e:
        print(f"  ❌ Usage tracking: {e}")
        failed += 1
    
    return passed, failed


def verify_13b():
    """Verify Phase 13B: Observability."""
    print("\nPhase 13B: Observability")
    print("-" * 60)
    
    passed = 0
    failed = 0
    
    try:
        from neurectomy.observability.tracing import get_tracer, trace
        tracer = get_tracer()
        with tracer.start_span("test_operation") as span:
            span.set_attribute("test", "value")
        spans = tracer.get_spans()
        assert len(spans) > 0
        print(f"  ✓ Distributed tracing: {len(spans)} spans recorded")
        passed += 1
    except Exception as e:
        print(f"  ❌ Distributed tracing: {e}")
        failed += 1
    
    try:
        from neurectomy.observability.tracing import trace
        @trace("test_function")
        def test_func():
            return "result"
        result = test_func()
        assert result == "result"
        print(f"  ✓ Trace decorator: Function traced successfully")
        passed += 1
    except Exception as e:
        print(f"  ❌ Trace decorator: {e}")
        failed += 1
    
    try:
        from neurectomy.observability.metrics import metrics
        counter = metrics.get_counter("neurectomy_requests_total")
        counter.inc(endpoint="/api/test", status="200")
        assert counter.get_value(endpoint="/api/test", status="200") == 1
        print(f"  ✓ Metrics counter: Counter incremented")
        passed += 1
    except Exception as e:
        print(f"  ❌ Metrics counter: {e}")
        failed += 1
    
    try:
        from neurectomy.observability.metrics import metrics
        histogram = metrics.get_histogram("neurectomy_latency_seconds")
        histogram.observe(0.123, endpoint="/api/test")
        stats = histogram.get_stats(endpoint="/api/test")
        assert stats["count"] == 1
        print(f"  ✓ Metrics histogram: {stats['count']} observation(s)")
        passed += 1
    except Exception as e:
        print(f"  ❌ Metrics histogram: {e}")
        failed += 1
    
    return passed, failed


def verify_13c():
    """Verify Phase 13C: Security Hardening."""
    print("\nPhase 13C: Security Hardening")
    print("-" * 60)
    
    passed = 0
    failed = 0
    
    try:
        from neurectomy.security.auth import APIKeyManager
        mgr = APIKeyManager()
        key_id, full_key = mgr.create_key("tenant1", "test_key")
        api_key = mgr.validate_key(full_key)
        assert api_key is not None
        assert api_key.tenant_id == "tenant1"
        print(f"  ✓ API key authentication: Key created and validated")
        passed += 1
    except Exception as e:
        print(f"  ❌ API key authentication: {e}")
        failed += 1
    
    try:
        from neurectomy.security.auth import APIKeyManager
        mgr = APIKeyManager()
        key_id, full_key = mgr.create_key("tenant1", "test_key")
        mgr.revoke_key(key_id)
        api_key = mgr.validate_key(full_key)
        assert api_key is None
        print(f"  ✓ Key revocation: Key revoked successfully")
        passed += 1
    except Exception as e:
        print(f"  ❌ Key revocation: {e}")
        failed += 1
    
    try:
        from neurectomy.security.rate_limit import RateLimiter
        rl = RateLimiter(60)
        allowed1, _ = rl.check("user1")
        assert allowed1 is True
        print(f"  ✓ Rate limiting: Token bucket working")
        passed += 1
    except Exception as e:
        print(f"  ❌ Rate limiting: {e}")
        failed += 1
    
    try:
        from neurectomy.security.audit import AuditLogger, AuditAction, AuditEvent
        logger = AuditLogger()
        event = AuditEvent(
            action=AuditAction.AUTH_SUCCESS,
            tenant_id="tenant1",
        )
        logger.log(event)
        events = logger.get_events("tenant1")
        assert len(events) == 1
        print(f"  ✓ Audit logging: {len(events)} event(s) logged")
        passed += 1
    except Exception as e:
        print(f"  ❌ Audit logging: {e}")
        failed += 1
    
    return passed, failed


def main():
    """Run all verifications."""
    print("=" * 60)
    print("  PHASE 13: Enterprise Features - Verification")
    print("=" * 60)
    
    total_passed = 0
    total_failed = 0
    
    # 13A
    passed, failed = verify_13a()
    total_passed += passed
    total_failed += failed
    
    # 13B
    passed, failed = verify_13b()
    total_passed += passed
    total_failed += failed
    
    # 13C
    passed, failed = verify_13c()
    total_passed += passed
    total_failed += failed
    
    print()
    print("=" * 60)
    print(f"  Results: {total_passed} passed, {total_failed} failed")
    print("=" * 60)
    
    if total_failed == 0:
        print("  ✅ PHASE 13 VERIFICATION COMPLETE")
        print("=" * 60)
        return 0
    else:
        print("  ⚠️  Some verifications failed")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
