#!/usr/bin/env python3
"""
Phase 4 Completion Verification
===============================
"""

import sys


def verify_orchestrator():
    """Verify orchestrator works."""
    try:
        from neurectomy import NeurectomyOrchestrator
        
        orchestrator = NeurectomyOrchestrator()
        health = orchestrator.health_check()
        
        assert health["orchestrator"] is True
        
        print("✓ Orchestrator verified")
        return True
    except Exception as e:
        print(f"❌ Orchestrator error: {e}")
        return False


def verify_agents():
    """Verify agent system works."""
    try:
        from neurectomy.agents import (
            AgentRegistry, InferenceAgent,
            SummarizationAgent, CodeAgent, ReasoningAgent,
            AgentConfig,
        )
        
        registry = AgentRegistry()
        
        # Register agents
        agents = [
            (InferenceAgent, "inf_test"),
            (SummarizationAgent, "sum_test"),
            (CodeAgent, "code_test"),
            (ReasoningAgent, "reason_test"),
        ]
        
        for agent_class, agent_id in agents:
            config = AgentConfig(agent_id=agent_id)
            registry.register(agent_class, config)
        
        assert len(registry.list_ids()) == 4
        
        print(f"✓ Agent registry verified ({len(registry.list_ids())} agents)")
        return True
    except Exception as e:
        print(f"❌ Agent error: {e}")
        return False


def verify_task_execution():
    """Verify task execution works."""
    try:
        from neurectomy import NeurectomyOrchestrator
        
        orchestrator = NeurectomyOrchestrator()
        result = orchestrator.generate("Hello!", max_tokens=5)
        
        assert result.task_id is not None
        
        print(f"✓ Task execution verified (status: {result.status.name})")
        return True
    except Exception as e:
        print(f"❌ Task execution error: {e}")
        return False


def verify_bridges():
    """Verify component bridges work."""
    try:
        from neurectomy.core.bridges import (
            InferenceBridge, CompressionBridge, StorageBridge
        )
        
        inference = InferenceBridge()
        compression = CompressionBridge()
        storage = StorageBridge()
        
        print("✓ Bridges verified:")
        print(f"   - Inference: {'ready' if inference.is_ready() else 'mock'}")
        print(f"   - Compression: {'ready' if compression.is_ready() else 'mock'}")
        print(f"   - Storage: {'ready' if storage.is_ready() else 'mock'}")
        return True
    except Exception as e:
        print(f"❌ Bridge error: {e}")
        return False


def main():
    """Run all verifications."""
    print("=" * 60)
    print("  PHASE 4: Neurectomy Integration - Verification")
    print("=" * 60)
    print()
    
    results = [
        verify_orchestrator(),
        verify_agents(),
        verify_task_execution(),
        verify_bridges(),
    ]
    
    print()
    if all(results):
        print("=" * 60)
        print("  ✅ PHASE 4 VERIFICATION COMPLETE")
        print("=" * 60)
        print()
        print("Ready for Phase 5: Elite Agent Collective")
        return 0
    else:
        print("=" * 60)
        print("  ❌ PHASE 4 VERIFICATION FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
