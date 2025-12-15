"""
Full Integration Test
=====================

End-to-end test of the complete Neurectomy system.
"""

import pytest


class TestFullIntegration:
    """Test complete system integration."""
    
    def test_orchestrator_with_collective(self):
        """Test orchestrator with Elite Collective."""
        from neurectomy import NeurectomyOrchestrator
        
        orchestrator = NeurectomyOrchestrator()
        health = orchestrator.health_check()
        
        assert health["orchestrator"] == True
    
    def test_elite_collective_init(self):
        """Test Elite Collective initializes correctly."""
        from neurectomy.elite import EliteCollective
        
        collective = EliteCollective()
        
        assert len(collective.list_agents()) == 40
        assert len(collective.list_teams()) == 5
    
    def test_generate_through_collective(self):
        """Test generation through Elite Collective."""
        from neurectomy.elite import EliteCollective
        from neurectomy.core.types import TaskRequest
        
        collective = EliteCollective()
        
        request = TaskRequest(
            task_id="integration_test_001",
            task_type="generate",
            payload={"prompt": "What is AI?"},
        )
        
        result = collective.execute(request)
        assert result.task_id == "integration_test_001"
    
    def test_all_teams_functional(self):
        """Test all teams respond to requests."""
        from neurectomy.elite import EliteCollective
        from neurectomy.core.types import TaskRequest, AgentCapability
        
        collective = EliteCollective()
        
        test_cases = [
            ("inference", [AgentCapability.INFERENCE]),
            ("compression", [AgentCapability.COMPRESSION]),
            ("storage", [AgentCapability.STORAGE]),
            ("analysis", [AgentCapability.ANALYSIS]),
            ("synthesis", [AgentCapability.SYNTHESIS]),
        ]
        
        for team_name, caps in test_cases:
            request = TaskRequest(
                task_id=f"test_{team_name}",
                task_type="test",
                payload={},
                required_capabilities=caps,
            )
            
            result = collective.execute(request)
            assert result.task_id == f"test_{team_name}"
    
    def test_cross_team_collaboration(self):
        """Test commanders can access each other."""
        from neurectomy.elite import EliteCollective
        
        collective = EliteCollective()
        
        inf_cmd = collective.get_team("inference")
        comp_cmd = collective.get_team("compression")
        
        # Verify collaboration is set up
        assert inf_cmd is not None
        assert comp_cmd is not None
        assert comp_cmd.agent_id in inf_cmd._collaborators
    
    def test_collective_stats(self):
        """Test statistics tracking."""
        from neurectomy.elite import EliteCollective
        from neurectomy.core.types import TaskRequest
        
        collective = EliteCollective()
        
        # Execute a task
        request = TaskRequest(
            task_id="stats_test",
            task_type="generate",
            payload={"prompt": "Test"},
        )
        collective.execute(request)
        
        stats = collective.get_stats()
        assert stats.total_tasks_completed >= 1
    
    def test_get_agent_by_id(self):
        """Test agent retrieval by ID."""
        from neurectomy.elite import EliteCollective
        
        collective = EliteCollective()
        
        # Get a known agent
        agent = collective.get_agent("inference_commander")
        assert agent is not None
        assert agent.agent_id == "inference_commander"
    
    def test_health_check(self):
        """Test health check returns expected structure."""
        from neurectomy.elite import EliteCollective
        
        collective = EliteCollective()
        
        health = collective.health_check()
        
        assert health["status"] == "healthy"
        assert health["total_agents"] == 40
        assert "teams" in health
        assert len(health["teams"]) == 5


def test_full_system():
    """Quick full system test."""
    from neurectomy import NeurectomyOrchestrator, create_elite_collective
    
    # Create orchestrator
    orchestrator = NeurectomyOrchestrator()
    print(f"Orchestrator: {orchestrator.health_check()}")
    
    # Create collective
    collective = create_elite_collective()
    print(f"Collective: {len(collective.list_agents())} agents")
    
    # Generate text
    result = orchestrator.generate("Hello!", max_tokens=5)
    print(f"Generation: {result.status.name}")
    
    print("\n✓ Full system test passed!")


if __name__ == "__main__":
    test_full_system()
