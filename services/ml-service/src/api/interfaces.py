"""
Neurectomy Core Interface Protocols
===================================

Protocols for IDE orchestration and integration.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import (
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

from .types import (
    AgentCapability,
    AgentMessage,
    AgentProfile,
    AgentResponse,
    AgentRole,
    Artifact,
    CompressionLevel,
    ContextScope,
    ContextWindow,
    InferenceBackend,
    InferenceConfig,
    NeurectomyConfig,
    OrchestratorState,
    OrchestratorStatistics,
    ProjectContext,
    StorageConfig,
    TaskDefinition,
    TaskPlan,
    TaskResult,
    TaskStatus,
)


# =============================================================================
# AGENT PROTOCOL
# =============================================================================

@runtime_checkable
class AgentProtocol(Protocol):
    """
    Protocol for Elite Agent implementations.
    
    Each agent in the collective implements this interface.
    """
    
    @property
    @abstractmethod
    def agent_id(self) -> str:
        """Unique identifier for this agent."""
        ...
    
    @property
    @abstractmethod
    def role(self) -> AgentRole:
        """Agent's role in the collective."""
        ...
    
    @property
    @abstractmethod
    def profile(self) -> AgentProfile:
        """Agent's full profile."""
        ...
    
    @abstractmethod
    async def process(
        self,
        message: AgentMessage,
        context: ContextWindow,
    ) -> AgentResponse:
        """
        Process a message and produce a response.
        
        Args:
            message: Input message to process
            context: Current context window
            
        Returns:
            AgentResponse with results
        """
        ...
    
    @abstractmethod
    async def stream_process(
        self,
        message: AgentMessage,
        context: ContextWindow,
    ) -> AsyncIterator[str]:
        """
        Process message with streaming output.
        
        Args:
            message: Input message
            context: Current context
            
        Yields:
            Response tokens as they're generated
        """
        ...
    
    @abstractmethod
    def can_handle(
        self,
        message: AgentMessage,
    ) -> bool:
        """
        Check if this agent can handle the given message.
        
        Args:
            message: Message to evaluate
            
        Returns:
            True if agent can handle it
        """
        ...
    
    @abstractmethod
    def get_capabilities(self) -> List[AgentCapability]:
        """
        Get list of agent capabilities.
        
        Returns:
            List of capabilities
        """
        ...


# =============================================================================
# ORCHESTRATOR PROTOCOL
# =============================================================================

@runtime_checkable
class OrchestratorProtocol(Protocol):
    """
    Protocol for task orchestration across agents.
    
    The orchestrator routes work to appropriate agents and
    manages the execution pipeline.
    """
    
    @abstractmethod
    async def execute(
        self,
        user_request: str,
        context: Optional[ContextWindow] = None,
    ) -> TaskResult:
        """
        Execute a user request through the agent pipeline.
        
        Args:
            user_request: Natural language request from user
            context: Optional pre-built context
            
        Returns:
            TaskResult with execution outcome
        """
        ...
    
    @abstractmethod
    async def execute_plan(
        self,
        plan: TaskPlan,
    ) -> List[TaskResult]:
        """
        Execute a pre-defined task plan.
        
        Args:
            plan: Task plan to execute
            
        Returns:
            Results for each task in the plan
        """
        ...
    
    @abstractmethod
    async def stream_execute(
        self,
        user_request: str,
        context: Optional[ContextWindow] = None,
    ) -> AsyncIterator[str]:
        """
        Execute request with streaming output.
        
        Args:
            user_request: Natural language request
            context: Optional context
            
        Yields:
            Response tokens as generated
        """
        ...
    
    @abstractmethod
    def create_plan(
        self,
        user_request: str,
        context: Optional[ContextWindow] = None,
    ) -> TaskPlan:
        """
        Create execution plan without executing.
        
        Args:
            user_request: Request to plan for
            context: Optional context
            
        Returns:
            TaskPlan ready for execution
        """
        ...
    
    @abstractmethod
    def route_to_agent(
        self,
        message: AgentMessage,
    ) -> AgentRole:
        """
        Determine which agent should handle a message.
        
        Args:
            message: Message to route
            
        Returns:
            AgentRole best suited for the message
        """
        ...
    
    @abstractmethod
    def get_state(self) -> OrchestratorState:
        """
        Get current orchestrator state.
        
        Returns:
            Current OrchestratorState
        """
        ...
    
    @abstractmethod
    def get_statistics(self) -> OrchestratorStatistics:
        """
        Get orchestrator statistics.
        
        Returns:
            OrchestratorStatistics
        """
        ...


# =============================================================================
# CONTEXT MANAGER PROTOCOL
# =============================================================================

@runtime_checkable
class ContextManagerProtocol(Protocol):
    """
    Protocol for managing context across operations.
    
    Handles context building, compression via ΣLANG, and caching.
    """
    
    @abstractmethod
    def build_context(
        self,
        scope: ContextScope,
        files: Optional[List[str]] = None,
        conversation_id: Optional[str] = None,
    ) -> ContextWindow:
        """
        Build a context window for the given scope.
        
        Args:
            scope: Scope of context to build
            files: Specific files to include
            conversation_id: For conversation history
            
        Returns:
            Built ContextWindow
        """
        ...
    
    @abstractmethod
    def compress_context(
        self,
        context: ContextWindow,
        level: CompressionLevel = CompressionLevel.BALANCED,
    ) -> ContextWindow:
        """
        Compress context using ΣLANG.
        
        Args:
            context: Context to compress
            level: Compression aggressiveness
            
        Returns:
            Compressed ContextWindow
        """
        ...
    
    @abstractmethod
    def decompress_context(
        self,
        context: ContextWindow,
    ) -> ContextWindow:
        """
        Decompress ΣLANG-compressed context.
        
        Args:
            context: Compressed context
            
        Returns:
            Decompressed ContextWindow
        """
        ...
    
    @abstractmethod
    def get_cached_context(
        self,
        context_id: str,
    ) -> Optional[ContextWindow]:
        """
        Retrieve cached context by ID.
        
        Args:
            context_id: Context identifier
            
        Returns:
            Cached context or None
        """
        ...
    
    @abstractmethod
    def cache_context(
        self,
        context: ContextWindow,
    ) -> str:
        """
        Cache context for later retrieval.
        
        Args:
            context: Context to cache
            
        Returns:
            Context ID for retrieval
        """
        ...


# =============================================================================
# INFERENCE BRIDGE PROTOCOL
# =============================================================================

@runtime_checkable
class InferenceBridge(Protocol):
    """
    Protocol for connecting to inference backends.
    
    Abstracts Ryot LLM, Ollama, and cloud API access.
    """
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        config: Optional[InferenceConfig] = None,
    ) -> str:
        """
        Generate completion for prompt.
        
        Args:
            prompt: Input prompt
            config: Optional inference config
            
        Returns:
            Generated text
        """
        ...
    
    @abstractmethod
    async def stream_generate(
        self,
        prompt: str,
        config: Optional[InferenceConfig] = None,
    ) -> AsyncIterator[str]:
        """
        Stream generated tokens.
        
        Args:
            prompt: Input prompt
            config: Optional config
            
        Yields:
            Generated tokens
        """
        ...
    
    @abstractmethod
    def get_backend_info(self) -> Dict:
        """
        Get information about current backend.
        
        Returns:
            Backend information dict
        """
        ...
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if backend is available.
        
        Returns:
            True if ready
        """
        ...
    
    @abstractmethod
    def switch_backend(
        self,
        backend: InferenceBackend,
    ) -> bool:
        """
        Switch to different inference backend.
        
        Args:
            backend: Backend to switch to
            
        Returns:
            True if switch successful
        """
        ...


# =============================================================================
# STORAGE BRIDGE PROTOCOL
# =============================================================================

@runtime_checkable
class StorageBridge(Protocol):
    """
    Protocol for secure storage via ΣVAULT.
    """
    
    @abstractmethod
    def store_artifact(
        self,
        artifact: Artifact,
        encrypt: bool = True,
    ) -> str:
        """
        Store an artifact securely.
        
        Args:
            artifact: Artifact to store
            encrypt: Whether to encrypt
            
        Returns:
            Storage key
        """
        ...
    
    @abstractmethod
    def retrieve_artifact(
        self,
        artifact_id: str,
    ) -> Optional[Artifact]:
        """
        Retrieve stored artifact.
        
        Args:
            artifact_id: Artifact identifier
            
        Returns:
            Artifact or None
        """
        ...
    
    @abstractmethod
    def store_project(
        self,
        project: ProjectContext,
    ) -> bool:
        """
        Store project metadata.
        
        Args:
            project: Project context
            
        Returns:
            True if stored
        """
        ...
    
    @abstractmethod
    def lock_project(
        self,
        project_id: str,
        passphrase: Optional[str] = None,
    ) -> bool:
        """
        Lock project with additional security.
        
        Args:
            project_id: Project to lock
            passphrase: Optional extra passphrase
            
        Returns:
            True if locked
        """
        ...
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if storage is available.
        
        Returns:
            True if ready
        """
        ...


# =============================================================================
# PROJECT MANAGER PROTOCOL
# =============================================================================

@runtime_checkable
class ProjectManager(Protocol):
    """
    Protocol for managing projects in Neurectomy.
    """
    
    @abstractmethod
    def create_project(
        self,
        name: str,
        path: str,
        project_type: str,
    ) -> ProjectContext:
        """
        Create a new project.
        
        Args:
            name: Project name
            path: Filesystem path
            project_type: Type of project
            
        Returns:
            Created ProjectContext
        """
        ...
    
    @abstractmethod
    def open_project(
        self,
        path: str,
    ) -> ProjectContext:
        """
        Open existing project.
        
        Args:
            path: Project path
            
        Returns:
            ProjectContext
        """
        ...
    
    @abstractmethod
    def analyze_project(
        self,
        project: ProjectContext,
    ) -> Dict:
        """
        Analyze project structure and dependencies.
        
        Args:
            project: Project to analyze
            
        Returns:
            Analysis results
        """
        ...
    
    @abstractmethod
    def get_project_files(
        self,
        project: ProjectContext,
        pattern: Optional[str] = None,
    ) -> List[str]:
        """
        Get list of project files.
        
        Args:
            project: Project context
            pattern: Optional glob pattern
            
        Returns:
            List of file paths
        """
        ...


# =============================================================================
# AGENT COLLECTIVE PROTOCOL
# =============================================================================

@runtime_checkable
class AgentCollective(Protocol):
    """
    Protocol for managing the Elite Agent Collective.
    """
    
    @abstractmethod
    def get_agent(
        self,
        role: AgentRole,
    ) -> Optional[AgentProtocol]:
        """
        Get agent by role.
        
        Args:
            role: Agent role
            
        Returns:
            Agent instance or None
        """
        ...
    
    @abstractmethod
    def get_all_agents(self) -> List[AgentProtocol]:
        """
        Get all available agents.
        
        Returns:
            List of all agents
        """
        ...
    
    @abstractmethod
    def register_agent(
        self,
        agent: AgentProtocol,
    ) -> bool:
        """
        Register a new agent.
        
        Args:
            agent: Agent to register
            
        Returns:
            True if registered
        """
        ...
    
    @abstractmethod
    def get_agent_for_task(
        self,
        task: TaskDefinition,
    ) -> List[AgentProtocol]:
        """
        Get agents suitable for a task.
        
        Args:
            task: Task definition
            
        Returns:
            List of suitable agents
        """
        ...
    
    @abstractmethod
    def get_collective_statistics(self) -> Dict:
        """
        Get statistics about the collective.
        
        Returns:
            Statistics dictionary
        """
        ...


# =============================================================================
# NEURECTOMY FACTORY PROTOCOL
# =============================================================================

@runtime_checkable
class NeurectomyFactory(Protocol):
    """
    Factory for creating Neurectomy components.
    """
    
    @abstractmethod
    def create_orchestrator(
        self,
        config: Optional[NeurectomyConfig] = None,
    ) -> OrchestratorProtocol:
        """
        Create main orchestrator.
        
        Args:
            config: Optional configuration
            
        Returns:
            Configured orchestrator
        """
        ...
    
    @abstractmethod
    def create_context_manager(
        self,
        inference_bridge: Optional[InferenceBridge] = None,
    ) -> ContextManagerProtocol:
        """
        Create context manager.
        
        Args:
            inference_bridge: Optional inference bridge for compression
            
        Returns:
            Context manager
        """
        ...
    
    @abstractmethod
    def create_inference_bridge(
        self,
        config: Optional[InferenceConfig] = None,
    ) -> InferenceBridge:
        """
        Create inference bridge.
        
        Args:
            config: Inference configuration
            
        Returns:
            Configured inference bridge
        """
        ...
    
    @abstractmethod
    def create_storage_bridge(
        self,
        config: Optional[StorageConfig] = None,
    ) -> StorageBridge:
        """
        Create storage bridge.
        
        Args:
            config: Storage configuration
            
        Returns:
            Configured storage bridge
        """
        ...
    
    @abstractmethod
    def create_agent_collective(
        self,
        inference_bridge: InferenceBridge,
    ) -> AgentCollective:
        """
        Create agent collective.
        
        Args:
            inference_bridge: Bridge for agent inference
            
        Returns:
            Configured collective
        """
        ...
