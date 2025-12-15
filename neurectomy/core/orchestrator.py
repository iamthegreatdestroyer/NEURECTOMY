"""
Neurectomy Core Orchestrator
============================

Central orchestration of all AI components.
"""

import time
import uuid
from typing import Optional, List, Dict, Any, Generator
from dataclasses import dataclass
from collections import deque
from datetime import datetime

from .types import (
    TaskRequest, TaskResult, TaskStatus, TaskPriority,
    AgentState, OrchestratorState, AgentCapability,
)
from .bridges import InferenceBridge, CompressionBridge, StorageBridge
from ..agents import AgentRegistry, BaseAgent


@dataclass
class OrchestratorConfig:
    """Orchestrator configuration."""
    
    # Concurrency
    max_concurrent_tasks: int = 10
    task_queue_size: int = 1000
    
    # Timeouts
    default_timeout_seconds: float = 60.0
    health_check_interval_seconds: float = 30.0
    
    # Features
    enable_compression: bool = True
    enable_rsu_storage: bool = True
    enable_caching: bool = True
    
    # Performance
    batch_size: int = 8
    prefetch_count: int = 4


class NeurectomyOrchestrator:
    """
    Central orchestrator for the Neurectomy system.
    
    Coordinates:
    - Ryot LLM inference
    - ΣLANG compression
    - ΣVAULT RSU storage
    - Agent management
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        
        # Initialize bridges
        self._inference = InferenceBridge()
        self._compression = CompressionBridge()
        self._storage = StorageBridge()
        
        # Task management
        self._task_queue: deque = deque(maxlen=self.config.task_queue_size)
        self._running_tasks: Dict[str, TaskRequest] = {}
        self._completed_tasks: Dict[str, TaskResult] = {}
        
        # Agent management
        self._agents: Dict[str, AgentState] = {}
        self._registry = AgentRegistry()
        
        # Register default agents
        self._register_default_agents()
        
        # Statistics
        self._start_time = time.time()
        self._total_tasks = 0
        self._total_tokens = 0
        self._compression_ratios: List[float] = []
        self._cache_hits = 0
        self._cache_misses = 0
    
    def submit_task(self, request: TaskRequest) -> str:
        """
        Submit task for execution.
        
        Returns task ID.
        """
        # Generate task ID if not provided
        if not request.task_id:
            request.task_id = f"task_{uuid.uuid4().hex[:12]}"
        
        # Add to queue
        self._task_queue.append(request)
        self._total_tasks += 1
        
        return request.task_id
    
    def execute_task(self, request: TaskRequest) -> TaskResult:
        """
        Execute a task synchronously.
        
        Routes to appropriate handler based on task type.
        """
        start_time = time.time()
        
        try:
            # Mark as running
            self._running_tasks[request.task_id] = request
            
            # Route to handler
            if request.task_type == "generate":
                result = self._handle_generate(request)
            elif request.task_type == "compress":
                result = self._handle_compress(request)
            elif request.task_type == "retrieve":
                result = self._handle_retrieve(request)
            elif request.task_type == "analyze":
                result = self._handle_analyze(request)
            else:
                result = TaskResult(
                    task_id=request.task_id,
                    status=TaskStatus.FAILED,
                    error_message=f"Unknown task type: {request.task_type}",
                )
            
            # Record metrics
            result.execution_time_ms = (time.time() - start_time) * 1000
            
        except Exception as e:
            result = TaskResult(
                task_id=request.task_id,
                status=TaskStatus.FAILED,
                error_message=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        
        finally:
            # Cleanup
            self._running_tasks.pop(request.task_id, None)
            self._completed_tasks[request.task_id] = result
        
        return result
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 1.0,
        conversation_id: Optional[str] = None,
    ) -> TaskResult:
        """
        Convenience method for text generation.
        """
        request = TaskRequest(
            task_id=f"gen_{uuid.uuid4().hex[:8]}",
            task_type="generate",
            payload={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            conversation_id=conversation_id,
            use_compression=self.config.enable_compression,
            use_rsu=self.config.enable_rsu_storage,
        )
        
        return self.execute_task(request)
    
    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 1.0,
    ) -> Generator[str, None, None]:
        """
        Stream text generation.
        """
        yield from self._inference.stream(prompt, max_tokens, temperature)
    
    def get_state(self) -> OrchestratorState:
        """Get current orchestrator state."""
        return OrchestratorState(
            inference_ready=self._inference.is_ready(),
            compression_ready=self._compression.is_ready(),
            storage_ready=self._storage.is_ready(),
            active_agents=sum(1 for a in self._agents.values() if a.is_active),
            total_agents=len(self._agents),
            pending_tasks=len(self._task_queue),
            running_tasks=len(self._running_tasks),
            completed_tasks=len(self._completed_tasks),
            total_tokens_processed=self._total_tokens,
            average_compression_ratio=self._get_avg_compression(),
            cache_hit_rate=self._get_cache_hit_rate(),
            uptime_seconds=time.time() - self._start_time,
        )
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all components."""
        return {
            "orchestrator": True,
            "inference": self._inference.is_ready(),
            "compression": self._compression.is_ready(),
            "storage": self._storage.is_ready(),
        }
    
    def _register_default_agents(self) -> None:
        """Register default specialized agents."""
        from ..agents import (
            InferenceAgent, SummarizationAgent,
            CodeAgent, ReasoningAgent, AgentConfig,
        )
        
        # Register agents
        agents = [
            (InferenceAgent, "inference_main"),
            (SummarizationAgent, "summarizer_main"),
            (CodeAgent, "coder_main"),
            (ReasoningAgent, "reasoner_main"),
        ]
        
        for agent_class, agent_id in agents:
            config = AgentConfig(agent_id=agent_id)
            self._registry.register(agent_class, config)
    
    def execute_with_agent(self, request: TaskRequest) -> TaskResult:
        """Execute task using registered agents."""
        agent = self._registry.find_for_task(request)
        
        if agent is None:
            return TaskResult(
                task_id=request.task_id,
                status=TaskStatus.FAILED,
                error_message="No suitable agent found",
            )
        
        return agent.process(request)
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent by ID."""
        return self._registry.get(agent_id)
    
    def list_agents(self) -> List[str]:
        """List all registered agent IDs."""
        return self._registry.list_ids()
    
    def _handle_generate(self, request: TaskRequest) -> TaskResult:
        """Handle text generation task."""
        prompt = request.payload.get("prompt", "")
        max_tokens = request.payload.get("max_tokens", request.max_tokens)
        temperature = request.payload.get("temperature", 1.0)
        
        # Try RSU warm-start if enabled
        rsu_ref = None
        cache_hit = False
        
        if request.use_rsu and request.conversation_id:
            # Would look up similar context here
            pass
        
        # Generate
        text, metadata = self._inference.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            use_compression=request.use_compression,
            conversation_id=request.conversation_id,
        )
        
        # Update statistics
        tokens_generated = metadata.get("tokens_generated", 0)
        tokens_processed = metadata.get("tokens_processed", 0)
        compression_ratio = metadata.get("compression_ratio", 1.0)
        
        self._total_tokens += tokens_generated + tokens_processed
        if compression_ratio > 1.0:
            self._compression_ratios.append(compression_ratio)
        
        if cache_hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1
        
        # Store RSU if enabled
        if request.use_rsu and text:
            rsu_ref = metadata.get("rsu_reference")
        
        return TaskResult(
            task_id=request.task_id,
            status=TaskStatus.COMPLETED if text else TaskStatus.FAILED,
            output=text,
            generated_text=text,
            tokens_processed=tokens_processed,
            tokens_generated=tokens_generated,
            compression_ratio=compression_ratio,
            rsu_reference=rsu_ref,
            cache_hit=cache_hit,
        )
    
    def _handle_compress(self, request: TaskRequest) -> TaskResult:
        """Handle compression task."""
        tokens = request.payload.get("tokens", [])
        
        data, metadata = self._compression.compress(
            tokens,
            request.conversation_id,
        )
        
        if "error" in metadata:
            return TaskResult(
                task_id=request.task_id,
                status=TaskStatus.FAILED,
                error_message=metadata["error"],
            )
        
        return TaskResult(
            task_id=request.task_id,
            status=TaskStatus.COMPLETED,
            output=data,
            compression_ratio=metadata.get("compression_ratio", 1.0),
        )
    
    def _handle_retrieve(self, request: TaskRequest) -> TaskResult:
        """Handle RSU retrieval task."""
        rsu_id = request.payload.get("rsu_id")
        
        if not rsu_id:
            return TaskResult(
                task_id=request.task_id,
                status=TaskStatus.FAILED,
                error_message="No RSU ID provided",
            )
        
        result = self._storage.retrieve_rsu(rsu_id)
        
        if result is None:
            return TaskResult(
                task_id=request.task_id,
                status=TaskStatus.FAILED,
                error_message=f"RSU not found: {rsu_id}",
            )
        
        glyph_data, kv_data = result
        
        return TaskResult(
            task_id=request.task_id,
            status=TaskStatus.COMPLETED,
            output={"glyph_data": glyph_data, "kv_cache": kv_data},
            rsu_reference=rsu_id,
        )
    
    def _handle_analyze(self, request: TaskRequest) -> TaskResult:
        """Handle analysis task."""
        # Placeholder for future analysis capabilities
        return TaskResult(
            task_id=request.task_id,
            status=TaskStatus.COMPLETED,
            output={"analysis": "Not implemented"},
        )
    
    def _get_avg_compression(self) -> float:
        """Get average compression ratio."""
        if not self._compression_ratios:
            return 1.0
        return sum(self._compression_ratios) / len(self._compression_ratios)
    
    def _get_cache_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self._cache_hits + self._cache_misses
        if total == 0:
            return 0.0
        return self._cache_hits / total
