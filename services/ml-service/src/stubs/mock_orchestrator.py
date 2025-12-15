"""Mock Orchestrator Implementation for Testing"""

import time
import uuid
from typing import AsyncIterator, Dict, List, Optional

from ..api.types import (
    AgentMessage, AgentRole, ContextWindow, OrchestratorState,
    OrchestratorStatistics, TaskDefinition, TaskPlan, TaskResult, TaskStatus,
)
from ..api.interfaces import OrchestratorProtocol


class MockOrchestrator(OrchestratorProtocol):
    """Mock orchestrator for integration testing."""

    def __init__(self):
        self._state = OrchestratorState(
            is_running=True,
            available_agents=[role.name for role in AgentRole],
        )
        self._tasks_completed = 0
        self._tokens_used = 0

    async def execute(self, user_request: str, context: Optional[ContextWindow] = None) -> TaskResult:
        start_time = time.time()
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        self._tasks_completed += 1
        self._tokens_used += len(user_request.split())

        return TaskResult(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            output_message=f"[MockOrchestrator] Executed: {user_request[:100]}...",
            actual_time_ms=(time.time() - start_time) * 1000,
            agents_used=["APEX", "NEXUS"],
            tokens_consumed=len(user_request.split()),
            started_at=start_time,
            completed_at=time.time(),
        )

    async def execute_plan(self, plan: TaskPlan) -> List[TaskResult]:
        results = []
        for task in plan.tasks:
            result = await self.execute(task.description)
            result.task_id = task.task_id
            results.append(result)
        return results

    async def stream_execute(self, user_request: str, context: Optional[ContextWindow] = None) -> AsyncIterator[str]:
        words = f"[MockOrchestrator] Processing: {user_request}".split()
        for word in words:
            yield word + " "

    def create_plan(self, user_request: str, context: Optional[ContextWindow] = None) -> TaskPlan:
        return TaskPlan(
            plan_id=f"plan_{uuid.uuid4().hex[:8]}",
            title="Mock Plan",
            description=user_request,
            tasks=[
                TaskDefinition(
                    task_id=f"task_{uuid.uuid4().hex[:8]}",
                    title="Mock Task",
                    description=user_request,
                    assigned_agents=[AgentRole.APEX],
                    primary_agent=AgentRole.APEX,
                )
            ],
            created_at=time.time(),
        )

    def route_to_agent(self, message: AgentMessage) -> AgentRole:
        if "code" in message.content.lower() or "implement" in message.content.lower():
            return AgentRole.APEX
        if "design" in message.content.lower() or "architect" in message.content.lower():
            return AgentRole.ARCHITECT
        if "test" in message.content.lower():
            return AgentRole.ECLIPSE
        return AgentRole.NEXUS

    def get_state(self) -> OrchestratorState:
        return self._state

    def get_statistics(self) -> OrchestratorStatistics:
        return OrchestratorStatistics(
            total_tasks_completed=self._tasks_completed,
            total_tasks_failed=0,
            average_task_time_ms=100.0,
            agent_utilization={role.name: 0.5 for role in AgentRole},
            agent_success_rates={role.name: 1.0 for role in AgentRole},
            total_tokens_used=self._tokens_used,
            total_compression_savings=0,
            average_compression_ratio=1.0,
            started_at=time.time(),
            uptime_seconds=0.0,
        )
