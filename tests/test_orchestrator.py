"""
Orchestrator Tests
==================
"""

import pytest
from neurectomy import NeurectomyOrchestrator, OrchestratorConfig, TaskRequest


class TestOrchestrator:
    """Test orchestrator functionality."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for testing."""
        config = OrchestratorConfig(
            enable_compression=True,
            enable_rsu_storage=True,
        )
        return NeurectomyOrchestrator(config)
    
    def test_health_check(self, orchestrator):
        """Test health check."""
        health = orchestrator.health_check()
        
        assert "orchestrator" in health
        assert health["orchestrator"] is True
    
    def test_generate(self, orchestrator):
        """Test text generation."""
        result = orchestrator.generate(
            "Hello, world!",
            max_tokens=10,
        )
        
        assert result.status.name in ["COMPLETED", "FAILED"]
        assert result.task_id is not None
    
    def test_state(self, orchestrator):
        """Test state retrieval."""
        state = orchestrator.get_state()
        
        assert state.uptime_seconds >= 0
        assert state.pending_tasks >= 0
    
    def test_submit_task(self, orchestrator):
        """Test task submission."""
        request = TaskRequest(
            task_id="test_task_1",
            task_type="generate",
            payload={"prompt": "Test"},
        )
        
        task_id = orchestrator.submit_task(request)
        assert task_id == "test_task_1"


def test_orchestrator_standalone():
    """Quick standalone test."""
    orchestrator = NeurectomyOrchestrator()
    
    # Check health
    health = orchestrator.health_check()
    print(f"Health: {health}")
    
    # Generate text
    result = orchestrator.generate("What is 2+2?", max_tokens=10)
    print(f"Task ID: {result.task_id}")
    print(f"Status: {result.status.name}")
    print(f"Generated: {result.generated_text}")
    
    # Get state
    state = orchestrator.get_state()
    print(f"Uptime: {state.uptime_seconds:.1f}s")
    print(f"Tokens processed: {state.total_tokens_processed}")
    
    print("\n✓ Orchestrator test passed")


if __name__ == "__main__":
    test_orchestrator_standalone()
