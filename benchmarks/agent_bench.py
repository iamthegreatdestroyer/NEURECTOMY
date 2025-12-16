"""
Agent Benchmarks
================

Benchmarks for Elite Agent Collective performance.
"""

from typing import Dict, Any, Optional
from .base import Benchmark, BenchmarkConfig


class AgentRoutingBenchmark(Benchmark):
    """Benchmark agent task routing latency."""
    
    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
    ):
        super().__init__(config)
        self._collective = None
    
    @property
    def name(self) -> str:
        return "agent_routing_latency"
    
    def setup(self) -> None:
        from neurectomy.elite import EliteCollective
        self._collective = EliteCollective()
    
    def run_iteration(self) -> Dict[str, Any]:
        import time
        from neurectomy.core.types import TaskRequest
        
        request = TaskRequest(
            task_id="bench_routing",
            task_type="generate",
            payload={"prompt": "Test"},
        )
        
        start = time.perf_counter()
        # Route to team (not execute)
        team_name = self._collective._route_to_team(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        return {
            "routing_time_ms": elapsed_ms,
            "team_selected": team_name or "none",
        }
    
    def teardown(self) -> None:
        self._collective = None


class AgentExecutionBenchmark(Benchmark):
    """Benchmark agent task execution."""
    
    def __init__(
        self,
        task_type: str = "generate",
        config: Optional[BenchmarkConfig] = None,
    ):
        super().__init__(config)
        self.task_type = task_type
        self._collective = None
    
    @property
    def name(self) -> str:
        return f"agent_execution_{self.task_type}"
    
    def setup(self) -> None:
        from neurectomy.elite import EliteCollective
        self._collective = EliteCollective()
    
    def run_iteration(self) -> Dict[str, Any]:
        import time
        from neurectomy.core.types import TaskRequest
        
        request = TaskRequest(
            task_id=f"bench_exec_{self.task_type}",
            task_type=self.task_type,
            payload={"prompt": "Benchmark test prompt", "max_tokens": 10},
        )
        
        start = time.perf_counter()
        result = self._collective.execute(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        return {
            "execution_time_ms": elapsed_ms,
            "status": result.status.name if result else "failed",
            "agent_used": result.executing_agent if result else "none",
        }
    
    def teardown(self) -> None:
        self._collective = None


class TeamThroughputBenchmark(Benchmark):
    """Benchmark team task throughput."""
    
    def __init__(
        self,
        team: str = "inference",
        num_tasks: int = 10,
        config: Optional[BenchmarkConfig] = None,
    ):
        super().__init__(config)
        self.team = team
        self.num_tasks = num_tasks
        self._collective = None
    
    @property
    def name(self) -> str:
        return f"team_throughput_{self.team}_{self.num_tasks}tasks"
    
    def setup(self) -> None:
        from neurectomy.elite import EliteCollective
        self._collective = EliteCollective()
    
    def run_iteration(self) -> Dict[str, Any]:
        import time
        from neurectomy.core.types import TaskRequest, AgentCapability
        
        # Map team to capability
        team_caps = {
            "inference": [AgentCapability.INFERENCE],
            "compression": [AgentCapability.COMPRESSION],
            "storage": [AgentCapability.STORAGE],
            "analysis": [AgentCapability.ANALYSIS],
            "synthesis": [AgentCapability.SYNTHESIS],
        }
        
        caps = team_caps.get(self.team, [])
        
        start = time.perf_counter()
        completed = 0
        
        for i in range(self.num_tasks):
            request = TaskRequest(
                task_id=f"bench_team_{i}",
                task_type="process",
                payload={"data": f"Task {i}"},
                required_capabilities=caps,
            )
            
            result = self._collective.execute(request)
            if result and result.status.name == "COMPLETED":
                completed += 1
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        tasks_per_sec = (self.num_tasks / elapsed_ms) * 1000 if elapsed_ms > 0 else 0
        
        return {
            "total_time_ms": elapsed_ms,
            "tasks_completed": completed,
            "tasks_per_second": tasks_per_sec,
        }
    
    def teardown(self) -> None:
        self._collective = None


class MultiAgentCollaborationBenchmark(Benchmark):
    """Benchmark multi-agent collaboration."""
    
    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
    ):
        super().__init__(config)
        self._collective = None
    
    @property
    def name(self) -> str:
        return "multi_agent_collaboration"
    
    def setup(self) -> None:
        from neurectomy.elite import EliteCollective
        self._collective = EliteCollective()
    
    def run_iteration(self) -> Dict[str, Any]:
        import time
        from neurectomy.core.types import TaskRequest, AgentCapability
        
        # Task requiring multiple capabilities
        request = TaskRequest(
            task_id="bench_collab",
            task_type="complex",
            payload={
                "prompt": "Analyze and summarize this text",
                "text": "Sample text for analysis" * 10,
            },
            required_capabilities=[
                AgentCapability.ANALYSIS,
                AgentCapability.SUMMARIZATION,
            ],
        )
        
        start = time.perf_counter()
        result = self._collective.execute(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        return {
            "collaboration_time_ms": elapsed_ms,
            "status": result.status.name if result else "failed",
        }
    
    def teardown(self) -> None:
        self._collective = None


def get_agent_benchmarks(config: Optional[BenchmarkConfig] = None) -> list:
    """Get all agent benchmarks."""
    return [
        AgentRoutingBenchmark(config=config),
        AgentExecutionBenchmark(task_type="generate", config=config),
        AgentExecutionBenchmark(task_type="summarize", config=config),
        AgentExecutionBenchmark(task_type="analyze", config=config),
        TeamThroughputBenchmark(team="inference", num_tasks=10, config=config),
        TeamThroughputBenchmark(team="analysis", num_tasks=10, config=config),
        MultiAgentCollaborationBenchmark(config=config),
    ]
