"""
Base Agent Implementation
=========================

Foundation for all Neurectomy agents.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from ..core.types import (
    TaskRequest, TaskResult, TaskStatus,
    AgentState, AgentCapability,
)
from ..core.bridges import InferenceBridge, CompressionBridge, StorageBridge


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    
    agent_id: Optional[str] = None
    agent_name: str = "BaseAgent"
    agent_type: str = "generic"
    
    # Capabilities
    capabilities: List[AgentCapability] = field(default_factory=list)
    
    # Resources
    max_context_tokens: int = 4096
    max_output_tokens: int = 2048
    
    # Behavior
    temperature: float = 0.7
    use_compression: bool = True
    use_caching: bool = True
    
    # System prompt
    system_prompt: str = ""


class BaseAgent(ABC):
    """
    Base class for all Neurectomy agents.
    
    Provides common functionality for specialized agents.
    """
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        inference: Optional[InferenceBridge] = None,
        compression: Optional[CompressionBridge] = None,
        storage: Optional[StorageBridge] = None,
    ):
        self.config = config or AgentConfig()
        
        # Generate agent ID if not provided
        if not self.config.agent_id:
            self.config.agent_id = f"agent_{uuid.uuid4().hex[:8]}"
        
        # Bridges (injected or created)
        self._inference = inference or InferenceBridge()
        self._compression = compression or CompressionBridge()
        self._storage = storage or StorageBridge()
        
        # State
        self._state = AgentState(
            agent_id=self.config.agent_id,
            agent_type=self.config.agent_type,
            capabilities=self.config.capabilities,
        )
        
        # Conversation history
        self._history: List[Dict[str, str]] = []
        self._context_tokens: int = 0
    
    @property
    def agent_id(self) -> str:
        """Get agent ID."""
        return self.config.agent_id
    
    @property
    def state(self) -> AgentState:
        """Get current agent state."""
        return self._state
    
    @abstractmethod
    def process(self, request: TaskRequest) -> TaskResult:
        """
        Process a task request.
        
        Must be implemented by subclasses.
        """
        pass
    
    def can_handle(self, request: TaskRequest) -> bool:
        """
        Check if agent can handle the request.
        
        Default implementation checks capabilities.
        """
        if not request.required_capabilities:
            return True
        
        return all(
            cap in self.config.capabilities
            for cap in request.required_capabilities
        )
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        include_history: bool = True,
    ) -> str:
        """
        Generate text with optional conversation history.
        """
        # Build full prompt
        full_prompt = self._build_prompt(prompt, include_history)
        
        # Generate
        text, metadata = self._inference.generate(
            prompt=full_prompt,
            max_tokens=max_tokens or self.config.max_output_tokens,
            temperature=self.config.temperature,
            use_compression=self.config.use_compression,
        )
        
        # Update history
        if text:
            self._history.append({"role": "user", "content": prompt})
            self._history.append({"role": "assistant", "content": text})
            self._update_context_tokens(metadata)
        
        return text
    
    def add_to_history(self, role: str, content: str) -> None:
        """Add message to conversation history."""
        self._history.append({"role": role, "content": content})
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history.clear()
        self._context_tokens = 0
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self._history.copy()
    
    def _build_prompt(self, user_prompt: str, include_history: bool) -> str:
        """Build full prompt with system prompt and history."""
        parts = []
        
        # System prompt
        if self.config.system_prompt:
            parts.append(f"System: {self.config.system_prompt}")
        
        # History
        if include_history and self._history:
            for msg in self._history[-10:]:  # Last 10 messages
                role = msg["role"].capitalize()
                parts.append(f"{role}: {msg['content']}")
        
        # Current prompt
        parts.append(f"User: {user_prompt}")
        parts.append("Assistant:")
        
        return "\n\n".join(parts)
    
    def _update_context_tokens(self, metadata: dict) -> None:
        """Update context token count."""
        tokens = metadata.get("tokens_processed", 0)
        self._context_tokens += tokens
        
        # Trim history if too long
        while self._context_tokens > self.config.max_context_tokens and self._history:
            removed = self._history.pop(0)
            # Estimate removed tokens
            self._context_tokens -= len(removed.get("content", "")) // 4
    
    def _create_success_result(
        self,
        request: TaskRequest,
        output: Any,
        generated_text: str = "",
    ) -> TaskResult:
        """Create successful task result."""
        return TaskResult(
            task_id=request.task_id,
            status=TaskStatus.COMPLETED,
            output=output,
            generated_text=generated_text,
            executing_agent=self.agent_id,
        )
    
    def _create_error_result(
        self,
        request: TaskRequest,
        error_message: str,
    ) -> TaskResult:
        """Create error task result."""
        return TaskResult(
            task_id=request.task_id,
            status=TaskStatus.FAILED,
            error_message=error_message,
            executing_agent=self.agent_id,
        )
