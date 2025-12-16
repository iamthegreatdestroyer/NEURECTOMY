"""Chaos test runner."""

from typing import List, Dict, Any


class ChaosRunner:
    """Run chaos engineering scenarios."""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self._scenarios = []
    
    def add_scenario(self, scenario) -> None:
        """Add a scenario to run."""
        self._scenarios.append(scenario)
    
    def run_all(self) -> Dict[str, Dict[str, Any]]:
        """Run all scenarios and collect results."""
        results = {}
        
        for scenario in self._scenarios:
            print(f"Running chaos scenario: {scenario.name}")
            
            try:
                scenario.inject(self.orchestrator)
                result = scenario.verify_resilience(self.orchestrator)
                results[scenario.name] = result
            finally:
                scenario.restore(self.orchestrator)
        
        return results
