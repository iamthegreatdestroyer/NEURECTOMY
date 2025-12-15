"""
Neurectomy Phase 0 Interface Contracts Usage Guide
===================================================

This guide demonstrates how to use Phase 0 Interface Contracts
to build components within the Neurectomy IDE ecosystem.
"""

# ==============================================================================
# EXAMPLE 1: IMPLEMENTING AN AGENT
# ==============================================================================

from src.api import AgentProtocol, AgentRole, AgentCapability, AgentMessage
from src.api import ContextWindow, AgentResponse, Artifact
import uuid


class APEXAgent:
    """
    Example implementation of AgentProtocol for APEX role.
    
    APEX is the "Elite Computer Science Engineering" agent.
    """
    
    @property
    def agent_id(self) -> str:
        return "agent-apex-001"
    
    @property
    def role(self) -> AgentRole:
        return AgentRole.APEX
    
    @property
    def profile(self):
        from src.api import AgentProfile
        return AgentProfile(
            agent_id=self.agent_id,
            role=self.role,
            name="APEX",
            description="Elite Computer Science Engineering",
            capabilities=[
                AgentCapability(
                    name="Code Generation",
                    description="Generate production-grade code",
                    input_types=["text", "context"],
                    output_types=["code", "artifact"]
                )
            ]
        )
    
    async def process(self, message: AgentMessage, context: ContextWindow) -> AgentResponse:
        """Process a message and return a response."""
        # Process logic here
        artifact = Artifact(
            artifact_id=str(uuid.uuid4()),
            artifact_type="code",
            name="generated_solution.py",
            content="# Generated code here",
            created_by=self.agent_id
        )
        
        return AgentResponse(
            response_id=str(uuid.uuid4()),
            agent_id=self.agent_id,
            request_message_id=message.message_id,
            content="Generated solution",
            artifacts=[artifact],
            success=True
        )
    
    async def stream_process(self, message: AgentMessage, context: ContextWindow):
        """Stream response tokens."""
        yield "Generating"
        yield " solution"
        yield "..."
    
    def can_handle(self, message: AgentMessage) -> bool:
        """Check if this agent can handle the message."""
        return "code" in message.intent.lower() or "implement" in message.intent.lower()
    
    def get_capabilities(self):
        """Get list of capabilities."""
        return self.profile.capabilities


# ==============================================================================
# EXAMPLE 2: IMPLEMENTING AN ORCHESTRATOR
# ==============================================================================

from src.api import OrchestratorProtocol, TaskPlan, TaskDefinition, TaskResult
from src.api import TaskStatus, OrchestratorState, OrchestratorStatistics


class Orchestrator:
    """
    Example implementation of OrchestratorProtocol.
    
    Routes tasks to appropriate agents and manages execution.
    """
    
    def __init__(self, agents: list):
        self.agents = {agent.role: agent for agent in agents}
        self.state = OrchestratorState()
        self.stats = OrchestratorStatistics()
    
    async def execute(self, user_request: str, context=None) -> TaskResult:
        """Execute a user request through the agent pipeline."""
        # 1. Create a plan
        plan = self.create_plan(user_request, context)
        
        # 2. Execute the plan
        results = await self.execute_plan(plan)
        
        # 3. Return the final result
        return results[-1] if results else TaskResult(
            task_id="failed",
            status=TaskStatus.FAILED
        )
    
    async def execute_plan(self, plan: TaskPlan) -> list:
        """Execute a pre-defined task plan."""
        results = []
        for task in plan.tasks:
            # Find agent for this task
            agent = self.route_to_agent(AgentMessage(
                message_id=task.task_id,
                sender="ORCHESTRATOR",
                recipients=["AGENT"],
                intent=task.title,
                content=task.description
            ))
            
            # Create context for task
            context = ContextWindow(
                context_id=str(uuid.uuid4()),
                scope=task.context_scope
            )
            
            # Execute the task
            message = AgentMessage(
                message_id=task.task_id,
                sender="ORCHESTRATOR",
                recipients=[agent.agent_id],
                intent=task.title,
                content=task.description
            )
            
            response = await agent.process(message, context)
            
            # Create result
            result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED if response.success else TaskStatus.FAILED,
                artifacts=response.artifacts,
                output_message=response.content
            )
            results.append(result)
        
        return results
    
    async def stream_execute(self, user_request: str, context=None):
        """Execute request with streaming output."""
        for token in ["Executing", " task", "..."]:
            yield token
    
    def create_plan(self, user_request: str, context=None) -> TaskPlan:
        """Create execution plan."""
        task = TaskDefinition(
            task_id=str(uuid.uuid4()),
            title=user_request,
            description=user_request,
            primary_agent=AgentRole.APEX
        )
        
        return TaskPlan(
            plan_id=str(uuid.uuid4()),
            title=user_request,
            description=user_request,
            tasks=[task],
            task_order=[task.task_id]
        )
    
    def route_to_agent(self, message: AgentMessage):
        """Route message to appropriate agent."""
        for agent in self.agents.values():
            if agent.can_handle(message):
                return agent
        return list(self.agents.values())[0]  # Default to first agent
    
    def get_state(self) -> OrchestratorState:
        """Get current state."""
        return self.state
    
    def get_statistics(self) -> OrchestratorStatistics:
        """Get statistics."""
        return self.stats


# ==============================================================================
# EXAMPLE 3: IMPLEMENTING INFERENCE BRIDGE
# ==============================================================================

from src.api import InferenceBridge, InferenceConfig, InferenceBackend


class RyotInferenceBridge:
    """
    Example implementation of InferenceBridge for Ryot LLM.
    """
    
    def __init__(self, config: InferenceConfig = None):
        self.config = config or InferenceConfig()
        self.backend = self.config.backend
    
    async def generate(self, prompt: str, config=None) -> str:
        """Generate completion."""
        cfg = config or self.config
        # Call Ryot LLM here
        return "Generated response from Ryot LLM"
    
    async def stream_generate(self, prompt: str, config=None):
        """Stream generation."""
        yield "Streaming"
        yield " response"
        yield " from Ryot"
    
    def get_backend_info(self) -> dict:
        """Get backend information."""
        return {
            "backend": self.backend.name,
            "model": self.config.ryot_model,
            "endpoint": self.config.ryot_endpoint
        }
    
    def is_available(self) -> bool:
        """Check if available."""
        return True  # Check actual connection
    
    def switch_backend(self, backend: InferenceBackend) -> bool:
        """Switch backend."""
        self.backend = backend
        self.config.backend = backend
        return True


# ==============================================================================
# EXAMPLE 4: CONTEXT MANAGEMENT
# ==============================================================================

from src.api import ContextManagerProtocol, ContextScope, CompressionLevel, ProjectContext


class ContextManager:
    """
    Example implementation of ContextManagerProtocol.
    """
    
    def __init__(self, inference_bridge=None):
        self.inference_bridge = inference_bridge
        self.cache = {}
    
    def build_context(self, scope: ContextScope, files=None, conversation_id=None) -> ContextWindow:
        """Build context window."""
        context = ContextWindow(
            context_id=str(uuid.uuid4()),
            scope=scope
        )
        
        if files:
            # Load files into context
            for file_path in files:
                context.files[file_path] = f"# Content of {file_path}"
        
        return context
    
    def compress_context(self, context: ContextWindow, level: CompressionLevel = CompressionLevel.BALANCED) -> ContextWindow:
        """Compress context using ΣLANG."""
        context.is_compressed = True
        context.compression_ratio = 0.6  # 60% of original size
        return context
    
    def decompress_context(self, context: ContextWindow) -> ContextWindow:
        """Decompress context."""
        context.is_compressed = False
        context.compression_ratio = 1.0
        return context
    
    def get_cached_context(self, context_id: str):
        """Get cached context."""
        return self.cache.get(context_id)
    
    def cache_context(self, context: ContextWindow) -> str:
        """Cache context."""
        self.cache[context.context_id] = context
        return context.context_id


# ==============================================================================
# EXAMPLE 5: USING THE FACTORY PATTERN
# ==============================================================================

from src.api import NeurectomyFactory, NeurectomyConfig


class NeurectomyFactoryImpl:
    """
    Example factory implementation.
    """
    
    def create_orchestrator(self, config=None):
        """Create orchestrator."""
        cfg = config or NeurectomyConfig()
        agents = self.create_agents()
        return Orchestrator(agents)
    
    def create_context_manager(self, inference_bridge=None):
        """Create context manager."""
        return ContextManager(inference_bridge)
    
    def create_inference_bridge(self, config=None):
        """Create inference bridge."""
        cfg = config or InferenceConfig()
        return RyotInferenceBridge(cfg)
    
    def create_storage_bridge(self, config=None):
        """Create storage bridge."""
        # Would implement StorageBridge here
        pass
    
    def create_agent_collective(self, inference_bridge):
        """Create agent collective."""
        # Would implement AgentCollective here
        pass
    
    def create_agents(self):
        """Helper to create all agents."""
        return [APEXAgent()]  # Add all 40 agents here


# ==============================================================================
# EXAMPLE 6: COMPLETE USAGE
# ==============================================================================

async def main():
    """
    Complete example of using Phase 0 Interface Contracts.
    """
    
    # Create factory
    factory = NeurectomyFactoryImpl()
    
    # Create all components
    config = NeurectomyConfig()
    inference = factory.create_inference_bridge(config.inference)
    context_manager = factory.create_context_manager(inference)
    orchestrator = factory.create_orchestrator(config)
    
    # Execute a task
    result = await orchestrator.execute("Implement a rate limiter")
    
    print(f"Task: {result.task_id}")
    print(f"Status: {result.status.name}")
    print(f"Output: {result.output_message}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
