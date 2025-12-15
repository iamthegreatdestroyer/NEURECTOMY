"""Mock Agent Implementation for Testing"""

import time
import uuid
from typing import AsyncIterator, List, Optional

from ..api.types import (
    AgentCapability, AgentMessage, AgentProfile, AgentResponse,
    AgentRole, Artifact, ContextWindow,
)
from ..api.interfaces import AgentProtocol


class MockAgent(AgentProtocol):
    """Mock agent for integration testing."""

    def __init__(self, role: AgentRole = AgentRole.APEX):
        self._agent_id = f"mock_{role.name.lower()}_{uuid.uuid4().hex[:8]}"
        self._role = role
        self._profile = AgentProfile(
            agent_id=self._agent_id,
            role=role,
            name=f"Mock {role.name}",
            description=f"Mock implementation of {role.name} agent",
            capabilities=[
                AgentCapability(
                    name="mock_capability",
                    description="Mock capability for testing",
                    input_types=["text"],
                    output_types=["text"],
                    estimated_time_ms=100,
                )
            ],
            avg_response_time_ms=100.0,
            success_rate=1.0,
            total_tasks_completed=0,
        )
        self._tasks_completed = 0

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def role(self) -> AgentRole:
        return self._role

    @property
    def profile(self) -> AgentProfile:
        return self._profile

    async def process(self, message: AgentMessage, context: ContextWindow) -> AgentResponse:
        start_time = time.time()
        self._tasks_completed += 1

        response_content = f"[MockAgent {self._role.name}] Processed: {message.content[:100]}..."

        return AgentResponse(
            response_id=f"resp_{uuid.uuid4().hex[:8]}",
            agent_id=self._agent_id,
            request_message_id=message.message_id,
            content=response_content,
            artifacts=[],
            success=True,
            processing_time_ms=(time.time() - start_time) * 1000,
            tokens_used=len(message.content.split()),
            compression_ratio=1.0,
        )

    async def stream_process(self, message: AgentMessage, context: ContextWindow) -> AsyncIterator[str]:
        words = f"[MockAgent {self._role.name}] Processing request...".split()
        for word in words:
            yield word + " "

    def can_handle(self, message: AgentMessage) -> bool:
        return True

    def get_capabilities(self) -> List[AgentCapability]:
        return self._profile.capabilities
