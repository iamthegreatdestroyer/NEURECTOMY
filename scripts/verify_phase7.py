#!/usr/bin/env python3
"""Phase 7 Verification - API & SDK"""

import sys


def verify_api_types():
    try:
        from neurectomy.api.types import (
            GenerationRequest, GenerationResponse,
            AgentTaskRequest, AgentTaskResponse,
            HealthResponse, MetricsResponse,
        )
        
        # Test request validation
        req = GenerationRequest(prompt="Test", max_tokens=100)
        assert req.prompt == "Test"
        assert req.max_tokens == 100
        
        print("✓ API types verified")
        return True
    except Exception as e:
        print(f"❌ API types failed: {e}")
        return False


def verify_sdk_client():
    try:
        from neurectomy.sdk import NeurectomyClient, NeurectomyConfig
        
        config = NeurectomyConfig(base_url="http://localhost:8000")
        client = NeurectomyClient(config)
        
        assert client.config.base_url == "http://localhost:8000"
        
        client.close()
        
        print("✓ SDK client verified")
        return True
    except Exception as e:
        print(f"❌ SDK client failed: {e}")
        return False


def verify_async_client():
    try:
        from neurectomy.sdk import AsyncNeurectomyClient
        
        # Just verify import works
        assert AsyncNeurectomyClient is not None
        
        print("✓ Async client verified")
        return True
    except Exception as e:
        print(f"❌ Async client failed: {e}")
        return False


def verify_api_app():
    try:
        from neurectomy.api.app import app
        
        assert app.title == "Neurectomy API"
        
        # Count routes
        routes = [r for r in app.routes if hasattr(r, 'path')]
        
        print(f"✓ API app verified ({len(routes)} routes)")
        return True
    except Exception as e:
        print(f"❌ API app failed: {e}")
        return False


def main():
    print("=" * 60)
    print("  PHASE 7: API & SDK - Verification")
    print("=" * 60)
    print()
    
    results = [
        verify_api_types(),
        verify_sdk_client(),
        verify_async_client(),
        verify_api_app(),
    ]
    
    print()
    if all(results):
        print("=" * 60)
        print("  ✅ PHASE 7 VERIFICATION COMPLETE")
        print("=" * 60)
        print()
        print("  API available at: http://localhost:8000")
        print("  Docs available at: http://localhost:8000/docs")
        return 0
    else:
        print("=" * 60)
        print("  ❌ PHASE 7 VERIFICATION FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
