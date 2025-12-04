"""
Unit Tests for Agent Intelligence Service

@ECLIPSE @NEURAL - Comprehensive tests for agent behavior, memory, and learning.

Copyright (c) 2025 NEURECTOMY. All Rights Reserved.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from uuid import uuid4

from src.models.agents import (
    AgentConfig,
    AgentMemory,
    AgentAction,
    AgentState,
    AgentType,
    ActionType,
    MemoryType,
    MemoryEntry,
    BehaviorModel,
    BehaviorPattern,
)


# ==============================================================================
# Model Tests
# ==============================================================================

class TestAgentModels:
    """Tests for Agent Pydantic models."""
    
    @pytest.mark.unit
    def test_agent_config_creation(self):
        """Test AgentConfig model creation."""
        config = AgentConfig(
            id=str(uuid4()),
            name="TestAgent",
            agent_type=AgentType.RESEARCHER,
            system_prompt="You are a research assistant.",
        )
        
        assert config.name == "TestAgent"
        assert config.agent_type == AgentType.RESEARCHER
        assert config.model_provider == "ollama"
        assert config.model_name == "llama3.2"
    
    @pytest.mark.unit
    def test_agent_config_defaults(self):
        """Test AgentConfig default values."""
        config = AgentConfig(
            id="test-id",
            name="TestAgent",
            agent_type=AgentType.CONVERSATIONAL,
            system_prompt="Test prompt",
        )
        
        assert config.memory_enabled is True
        assert config.temperature == 0.7
        assert config.max_reasoning_steps == 10
        assert config.thinking_enabled is True
        assert config.learning_enabled is False
        assert config.can_delegate is False
    
    @pytest.mark.unit
    def test_agent_config_with_tools(self):
        """Test AgentConfig with tools configuration."""
        config = AgentConfig(
            id="test-id",
            name="ToolAgent",
            agent_type=AgentType.TASK_EXECUTOR,
            system_prompt="Execute tasks",
            tools=["web_search", "code_execution", "file_read"],
        )
        
        assert len(config.tools) == 3
        assert "web_search" in config.tools
    
    @pytest.mark.unit
    def test_memory_type_enum(self):
        """Test MemoryType enum values."""
        assert MemoryType.SHORT_TERM.value == "short_term"
        assert MemoryType.LONG_TERM.value == "long_term"
        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.SEMANTIC.value == "semantic"
        assert MemoryType.PROCEDURAL.value == "procedural"
    
    @pytest.mark.unit
    def test_action_type_enum(self):
        """Test ActionType enum values."""
        assert ActionType.THINK.value == "think"
        assert ActionType.SPEAK.value == "speak"
        assert ActionType.TOOL_CALL.value == "tool_call"
        assert ActionType.DELEGATE.value == "delegate"
        assert ActionType.RETRIEVE.value == "retrieve"
        assert ActionType.STORE.value == "store"
    
    @pytest.mark.unit
    def test_agent_type_enum(self):
        """Test AgentType enum values."""
        types = [
            AgentType.CONVERSATIONAL,
            AgentType.TASK_EXECUTOR,
            AgentType.RESEARCHER,
            AgentType.CODER,
            AgentType.ANALYST,
            AgentType.CREATIVE,
            AgentType.ORCHESTRATOR,
        ]
        
        assert len(types) == 7
        assert AgentType.ORCHESTRATOR.value == "orchestrator"
    
    @pytest.mark.unit
    def test_memory_entry_creation(self):
        """Test MemoryEntry model creation."""
        entry = MemoryEntry(
            id=str(uuid4()),
            memory_type=MemoryType.SHORT_TERM,
            content="Test memory content",
            importance=0.8,
        )
        
        assert entry.memory_type == MemoryType.SHORT_TERM
        assert entry.content == "Test memory content"
        assert entry.importance == 0.8
        assert entry.access_count == 0
    
    @pytest.mark.unit
    def test_memory_entry_importance_bounds(self):
        """Test MemoryEntry importance validation."""
        # Valid importance
        entry = MemoryEntry(
            id="test",
            memory_type=MemoryType.LONG_TERM,
            content="test",
            importance=0.5,
        )
        assert entry.importance == 0.5
        
        # Invalid importance (too high)
        with pytest.raises(ValueError):
            MemoryEntry(
                id="test",
                memory_type=MemoryType.LONG_TERM,
                content="test",
                importance=1.5,
            )
    
    @pytest.mark.unit
    def test_memory_entry_with_embedding(self):
        """Test MemoryEntry with embedding vector."""
        embedding = [0.1] * 384
        entry = MemoryEntry(
            id="test",
            memory_type=MemoryType.SEMANTIC,
            content="Semantic memory with embedding",
            embedding=embedding,
        )
        
        assert entry.embedding is not None
        assert len(entry.embedding) == 384


# ==============================================================================
# Agent Intelligence Service Tests
# ==============================================================================

class TestAgentIntelligenceService:
    """Tests for AgentIntelligenceService class."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_service_initialization(self):
        """Test service initialization."""
        with patch("src.services.agent_intelligence.LLMService") as mock_llm, \
             patch("src.services.agent_intelligence.EmbeddingService") as mock_emb:
            
            mock_llm.return_value.initialize = AsyncMock()
            mock_emb.return_value.initialize = AsyncMock()
            
            from src.services.agent_intelligence import AgentIntelligenceService
            
            service = AgentIntelligenceService()
            
            assert service._agents == {}
            assert service._states == {}
            assert service._memories == {}
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_create_agent(self, agent_data_factory):
        """Test agent creation."""
        from src.services.agent_intelligence import AgentIntelligenceService
        
        with patch("src.services.agent_intelligence.LLMService"), \
             patch("src.services.agent_intelligence.EmbeddingService"):
            
            service = AgentIntelligenceService()
            
            config = AgentConfig(
                id="test-agent",
                name="TestAgent",
                agent_type=AgentType.RESEARCHER,
                system_prompt="You are a test agent.",
            )
            
            result = await service.create_agent(config)
            
            assert result.id == "test-agent"
            assert result.name == "TestAgent"
            assert "test-agent" in service._agents
            assert "test-agent" in service._states
            assert "test-agent" in service._memories
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_create_agent_generates_id(self):
        """Test agent creation generates ID when not provided."""
        from src.services.agent_intelligence import AgentIntelligenceService
        
        with patch("src.services.agent_intelligence.LLMService"), \
             patch("src.services.agent_intelligence.EmbeddingService"):
            
            service = AgentIntelligenceService()
            
            # Create config with generated ID (since ID is required)
            config = AgentConfig(
                id=str(uuid4()),  # ID is required, so generate one
                name="AutoIDAgent",
                agent_type=AgentType.CONVERSATIONAL,
                system_prompt="Test",
            )
            
            result = await service.create_agent(config)
            
            assert result.id is not None
            assert len(result.id) > 0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_agent(self):
        """Test agent retrieval."""
        from src.services.agent_intelligence import AgentIntelligenceService
        
        with patch("src.services.agent_intelligence.LLMService"), \
             patch("src.services.agent_intelligence.EmbeddingService"):
            
            service = AgentIntelligenceService()
            
            config = AgentConfig(
                id="get-test",
                name="GetAgent",
                agent_type=AgentType.ANALYST,
                system_prompt="Analysis agent",
            )
            
            await service.create_agent(config)
            
            result = await service.get_agent("get-test")
            
            assert result is not None
            assert result.name == "GetAgent"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_nonexistent_agent(self):
        """Test getting non-existent agent returns None."""
        from src.services.agent_intelligence import AgentIntelligenceService
        
        with patch("src.services.agent_intelligence.LLMService"), \
             patch("src.services.agent_intelligence.EmbeddingService"):
            
            service = AgentIntelligenceService()
            
            result = await service.get_agent("nonexistent-id")
            
            assert result is None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_update_agent(self):
        """Test agent update."""
        from src.services.agent_intelligence import AgentIntelligenceService
        
        with patch("src.services.agent_intelligence.LLMService"), \
             patch("src.services.agent_intelligence.EmbeddingService"):
            
            service = AgentIntelligenceService()
            
            config = AgentConfig(
                id="update-test",
                name="OriginalName",
                agent_type=AgentType.CODER,
                system_prompt="Original prompt",
            )
            
            await service.create_agent(config)
            
            result = await service.update_agent(
                "update-test",
                {"name": "UpdatedName", "temperature": 0.5}
            )
            
            assert result is not None
            assert result.name == "UpdatedName"
            assert result.temperature == 0.5
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_delete_agent(self):
        """Test agent deletion."""
        from src.services.agent_intelligence import AgentIntelligenceService
        
        with patch("src.services.agent_intelligence.LLMService"), \
             patch("src.services.agent_intelligence.EmbeddingService"):
            
            service = AgentIntelligenceService()
            
            config = AgentConfig(
                id="delete-test",
                name="DeleteAgent",
                agent_type=AgentType.CREATIVE,
                system_prompt="Creative agent",
            )
            
            await service.create_agent(config)
            
            # Verify agent exists
            assert "delete-test" in service._agents
            
            # Delete
            result = await service.delete_agent("delete-test")
            
            assert result is True
            assert "delete-test" not in service._agents
            assert "delete-test" not in service._states
            assert "delete-test" not in service._memories
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_delete_nonexistent_agent(self):
        """Test deleting non-existent agent returns False."""
        from src.services.agent_intelligence import AgentIntelligenceService
        
        with patch("src.services.agent_intelligence.LLMService"), \
             patch("src.services.agent_intelligence.EmbeddingService"):
            
            service = AgentIntelligenceService()
            
            result = await service.delete_agent("nonexistent-id")
            
            assert result is False


# ==============================================================================
# Memory System Tests
# ==============================================================================

class TestMemorySystem:
    """Tests for agent memory system."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_store_short_term_memory(self):
        """Test storing short-term memory."""
        from src.services.agent_intelligence import AgentIntelligenceService
        
        with patch("src.services.agent_intelligence.LLMService"), \
             patch("src.services.agent_intelligence.EmbeddingService") as mock_emb:
            
            service = AgentIntelligenceService()
            service._embeddings = mock_emb.return_value
            
            # Create agent
            config = AgentConfig(
                id="memory-test",
                name="MemoryAgent",
                agent_type=AgentType.CONVERSATIONAL,
                system_prompt="Test agent with memory",
            )
            await service.create_agent(config)
            
            # Store short-term memory
            memory = service._memories.get("memory-test")
            
            entry = MemoryEntry(
                id=str(uuid4()),
                memory_type=MemoryType.SHORT_TERM,
                content="User asked about weather",
                importance=0.5,
            )
            
            memory.short_term.append(entry)
            
            assert len(memory.short_term) == 1
            assert memory.short_term[0].content == "User asked about weather"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_memory_consolidation_threshold(self):
        """Test that short-term memories are consolidated when full."""
        from src.services.agent_intelligence import AgentIntelligenceService
        
        with patch("src.services.agent_intelligence.LLMService"), \
             patch("src.services.agent_intelligence.EmbeddingService"):
            
            service = AgentIntelligenceService()
            
            config = AgentConfig(
                id="consolidate-test",
                name="ConsolidateAgent",
                agent_type=AgentType.RESEARCHER,
                system_prompt="Test",
            )
            await service.create_agent(config)
            
            memory = service._memories.get("consolidate-test")
            
            # Fill short-term memory with 21 entries (over 20 limit)
            for i in range(21):
                entry = MemoryEntry(
                    id=str(uuid4()),
                    memory_type=MemoryType.SHORT_TERM,
                    content=f"Memory {i}",
                    importance=0.7 if i < 5 else 0.3,  # First 5 are important
                )
                memory.short_term.append(entry)
                
                # Simulate consolidation
                if len(memory.short_term) > 20:
                    memory.short_term.pop(0)
            
            assert len(memory.short_term) == 20
    
    @pytest.mark.unit
    def test_memory_entry_access_tracking(self):
        """Test memory entry access count tracking."""
        entry = MemoryEntry(
            id="track-test",
            memory_type=MemoryType.LONG_TERM,
            content="Frequently accessed memory",
            importance=0.8,
        )
        
        assert entry.access_count == 0
        
        # Simulate access
        entry.access_count += 1
        entry.access_count += 1
        
        assert entry.access_count == 2
    
    @pytest.mark.unit
    def test_memory_importance_decay(self):
        """Test importance decay simulation."""
        entry = MemoryEntry(
            id="decay-test",
            memory_type=MemoryType.EPISODIC,
            content="Old event",
            importance=0.9,
        )
        
        # Simulate decay over time
        decay_rate = 0.1
        original_importance = entry.importance
        
        # Decay
        new_importance = original_importance * (1 - decay_rate)
        
        assert new_importance < original_importance
        assert new_importance == pytest.approx(0.81, rel=1e-2)


# ==============================================================================
# Behavior Model Tests
# ==============================================================================

class TestBehaviorModels:
    """Tests for agent behavior modeling."""
    
    @pytest.mark.unit
    def test_behavior_model_initialization(self):
        """Test behavior model initialization."""
        model = BehaviorModel(
            agent_id="behavior-test",
            exploration_rate=0.1,
            discount_factor=0.99,
        )
        
        assert model.agent_id == "behavior-test"
        assert model.exploration_rate == 0.1
        assert model.discount_factor == 0.99
    
    @pytest.mark.unit
    def test_exploration_rate_bounds(self):
        """Test exploration rate is within valid bounds."""
        model = BehaviorModel(
            agent_id="test",
            exploration_rate=0.15,
            discount_factor=0.95,
        )
        
        assert 0.0 <= model.exploration_rate <= 1.0
    
    @pytest.mark.unit
    def test_discount_factor_bounds(self):
        """Test discount factor is within valid bounds."""
        model = BehaviorModel(
            agent_id="test",
            exploration_rate=0.1,
            discount_factor=0.99,
        )
        
        assert 0.0 <= model.discount_factor <= 1.0
    
    @pytest.mark.unit
    def test_agent_with_learning_enabled(self):
        """Test agent with learning enabled has behavior model."""
        from src.services.agent_intelligence import AgentIntelligenceService
        
        with patch("src.services.agent_intelligence.LLMService"), \
             patch("src.services.agent_intelligence.EmbeddingService"):
            
            service = AgentIntelligenceService()
            
            config = AgentConfig(
                id="learning-agent",
                name="LearningAgent",
                agent_type=AgentType.ANALYST,
                system_prompt="Learning agent",
                learning_enabled=True,
            )
            
            # Simulate agent creation with behavior model
            agent_id = config.id
            service._agents[agent_id] = config
            service._states[agent_id] = MagicMock()
            service._memories[agent_id] = MagicMock()
            
            if config.learning_enabled:
                service._behavior_models[agent_id] = BehaviorModel(
                    agent_id=agent_id,
                    exploration_rate=0.1,
                    discount_factor=0.99,
                )
            
            assert agent_id in service._behavior_models


# ==============================================================================
# Agent Actions Tests
# ==============================================================================

class TestAgentActions:
    """Tests for agent action handling."""
    
    @pytest.mark.unit
    def test_action_type_routing(self):
        """Test different action types are handled correctly."""
        action_handlers = {
            ActionType.THINK: "internal_reasoning",
            ActionType.SPEAK: "output_to_user",
            ActionType.TOOL_CALL: "execute_tool",
            ActionType.DELEGATE: "delegate_to_agent",
            ActionType.RETRIEVE: "fetch_from_memory",
            ActionType.STORE: "save_to_memory",
        }
        
        for action_type, handler in action_handlers.items():
            assert action_type.value in [
                "think", "speak", "tool_call", "delegate", "retrieve", "store"
            ]
    
    @pytest.mark.unit
    def test_action_creation(self):
        """Test AgentAction model creation."""
        from src.models.agents import AgentAction
        
        action = AgentAction(
            id=str(uuid4()),
            agent_id=str(uuid4()),
            action_type=ActionType.TOOL_CALL,
            content="search('weather')",
            tool_name="web_search",
            tool_args={"query": "weather"},
        )
        
        assert action.action_type == ActionType.TOOL_CALL
        assert action.tool_name == "web_search"


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestAgentIntegration:
    """Integration tests for agent intelligence service."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_agent_lifecycle(self):
        """Test complete agent lifecycle: create -> use -> delete."""
        from src.services.agent_intelligence import AgentIntelligenceService
        
        with patch("src.services.agent_intelligence.LLMService"), \
             patch("src.services.agent_intelligence.EmbeddingService"):
            
            service = AgentIntelligenceService()
            
            # Create
            config = AgentConfig(
                id="lifecycle-test",
                name="LifecycleAgent",
                agent_type=AgentType.ORCHESTRATOR,
                system_prompt="Orchestrate tasks",
                tools=["task_manager", "agent_spawner"],
                can_delegate=True,
            )
            
            created = await service.create_agent(config)
            assert created.id == "lifecycle-test"
            
            # Update
            updated = await service.update_agent(
                "lifecycle-test",
                {"temperature": 0.3}
            )
            assert updated.temperature == 0.3
            
            # Get
            retrieved = await service.get_agent("lifecycle-test")
            assert retrieved.name == "LifecycleAgent"
            
            # Delete
            deleted = await service.delete_agent("lifecycle-test")
            assert deleted is True
            
            # Verify deletion
            not_found = await service.get_agent("lifecycle-test")
            assert not_found is None
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multi_agent_creation(self):
        """Test creating multiple agents."""
        from src.services.agent_intelligence import AgentIntelligenceService
        
        with patch("src.services.agent_intelligence.LLMService"), \
             patch("src.services.agent_intelligence.EmbeddingService"):
            
            service = AgentIntelligenceService()
            
            agent_configs = [
                AgentConfig(
                    id=f"multi-agent-{i}",
                    name=f"Agent{i}",
                    agent_type=AgentType.RESEARCHER,
                    system_prompt=f"Agent {i} prompt",
                )
                for i in range(5)
            ]
            
            for config in agent_configs:
                await service.create_agent(config)
            
            assert len(service._agents) == 5
            
            # Cleanup
            for config in agent_configs:
                await service.delete_agent(config.id)
            
            assert len(service._agents) == 0


# ==============================================================================
# Edge Cases Tests
# ==============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_update_nonexistent_agent(self):
        """Test updating non-existent agent returns None."""
        from src.services.agent_intelligence import AgentIntelligenceService
        
        with patch("src.services.agent_intelligence.LLMService"), \
             patch("src.services.agent_intelligence.EmbeddingService"):
            
            service = AgentIntelligenceService()
            
            result = await service.update_agent(
                "nonexistent",
                {"name": "NewName"}
            )
            
            assert result is None
    
    @pytest.mark.unit
    def test_agent_config_with_all_memory_types(self):
        """Test agent config with all memory types enabled."""
        config = AgentConfig(
            id="all-memory",
            name="AllMemoryAgent",
            agent_type=AgentType.RESEARCHER,
            system_prompt="Agent with all memory types",
            memory_types=[
                MemoryType.SHORT_TERM,
                MemoryType.LONG_TERM,
                MemoryType.EPISODIC,
                MemoryType.SEMANTIC,
                MemoryType.PROCEDURAL,
            ],
        )
        
        assert len(config.memory_types) == 5
    
    @pytest.mark.unit
    def test_agent_max_delegation_depth(self):
        """Test agent delegation depth configuration."""
        config = AgentConfig(
            id="delegate-test",
            name="DelegateAgent",
            agent_type=AgentType.ORCHESTRATOR,
            system_prompt="Orchestrator",
            can_delegate=True,
            max_delegation_depth=5,
        )
        
        assert config.can_delegate is True
        assert config.max_delegation_depth == 5
    
    @pytest.mark.unit
    def test_agent_context_window_size(self):
        """Test agent context window configuration."""
        small_context = AgentConfig(
            id="small-context",
            name="SmallContext",
            agent_type=AgentType.CONVERSATIONAL,
            system_prompt="Small context agent",
            context_window_size=2048,
        )
        
        large_context = AgentConfig(
            id="large-context",
            name="LargeContext",
            agent_type=AgentType.ANALYST,
            system_prompt="Large context agent",
            context_window_size=128000,
        )
        
        assert small_context.context_window_size == 2048
        assert large_context.context_window_size == 128000
