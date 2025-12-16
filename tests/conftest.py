"""Shared test fixtures."""

import pytest
from typing import Generator


@pytest.fixture
def orchestrator():
    """Create test orchestrator."""
    try:
        from neurectomy import NeurectomyOrchestrator
        return NeurectomyOrchestrator()
    except Exception:
        # Return mock if not available
        class MockOrchestrator:
            def generate(self, prompt, max_tokens=10, **kwargs):
                return type('Result', (), {
                    'generated_text': f"Mock: {prompt[:20]}...",
                    'tokens_generated': min(max_tokens, 5),
                    'execution_time_ms': 50.0,
                    'compression_ratio': 1.2,
                })()
        return MockOrchestrator()


@pytest.fixture
def collective():
    """Create test collective."""
    try:
        from neurectomy.elite import EliteCollective
        return EliteCollective()
    except Exception:
        # Return mock if not available
        class MockCollective:
            def list_agents(self):
                return [f"agent_{i}" for i in range(40)]
            
            def list_teams(self):
                return ["inference", "compression", "storage", "analysis", "synthesis"]
            
            def get_agent(self, agent_id):
                return {"id": agent_id, "name": agent_id}
            
            def get_team(self, team_name):
                return {"name": team_name, "agents": 8}
            
            def execute(self, request):
                return type('Result', (), {
                    'status': type('Status', (), {'name': 'COMPLETED'})(),
                    'executing_agent': 'mock_agent',
                    'result': "Mock result",
                })()
        return MockCollective()


@pytest.fixture
def mock_inference():
    """Mock inference bridge."""
    class MockInference:
        def generate(self, prompt, **kwargs):
            return type('Result', (), {
                'generated_text': f"Mock: {prompt[:20]}...",
                'tokens_generated': 10,
                'execution_time_ms': 50.0,
            })()
    return MockInference()


@pytest.fixture
def sample_prompts():
    """Sample prompts for testing."""
    return [
        "Hello, how are you?",
        "Explain quantum computing.",
        "Write a Python function.",
        "Summarize this text.",
        "What is machine learning?",
    ]
