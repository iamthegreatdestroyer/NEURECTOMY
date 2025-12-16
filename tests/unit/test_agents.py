"""Unit tests for agents."""

import pytest


class TestAgents:
    
    @pytest.mark.unit
    def test_collective_initialization(self, collective):
        assert collective is not None
        agents = collective.list_agents()
        teams = collective.list_teams()
        assert len(agents) == 40
        assert len(teams) == 5
    
    @pytest.mark.unit
    def test_agent_retrieval(self, collective):
        agents = collective.list_agents()
        assert len(agents) > 0
        
        agent = collective.get_agent(agents[0])
        assert agent is not None
    
    @pytest.mark.unit
    def test_team_retrieval(self, collective):
        teams = collective.list_teams()
        
        for team_name in teams:
            team = collective.get_team(team_name)
            assert team is not None
    
    @pytest.mark.unit
    def test_task_routing(self, collective):
        try:
            from neurectomy.core.types import TaskRequest
            
            request = TaskRequest(
                task_id="test_001",
                task_type="generate",
                payload={"prompt": "Test"},
            )
            
            result = collective.execute(request)
            assert result is not None
        except (ImportError, TypeError):
            pytest.skip("TaskRequest not available")
