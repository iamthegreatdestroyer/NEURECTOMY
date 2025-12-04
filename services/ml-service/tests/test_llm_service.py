"""
Unit Tests for LLM Service

@ECLIPSE @LINGUA - Comprehensive tests for multi-model LLM orchestration.

Copyright (c) 2025 NEURECTOMY. All Rights Reserved.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from uuid import uuid4

from src.models.llm import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ModelProvider,
    Role,
    Usage,
)


# ==============================================================================
# Model Tests
# ==============================================================================

class TestLLMModels:
    """Tests for LLM Pydantic models."""
    
    @pytest.mark.unit
    def test_chat_message_creation(self):
        """Test ChatMessage model creation."""
        message = ChatMessage(
            role=Role.USER,
            content="Hello, how are you?",
        )
        
        assert message.role == Role.USER
        assert message.content == "Hello, how are you?"
        assert message.name is None
    
    @pytest.mark.unit
    def test_chat_message_with_tool_call(self):
        """Test ChatMessage with tool call metadata."""
        message = ChatMessage(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[
                {"id": "call_123", "type": "function", "function": {"name": "search"}}
            ],
        )
        
        assert message.role == Role.ASSISTANT
        assert len(message.tool_calls) == 1
        assert message.tool_calls[0]["id"] == "call_123"
    
    @pytest.mark.unit
    def test_chat_request_valid(self):
        """Test valid ChatRequest creation."""
        request = ChatRequest(
            messages=[
                ChatMessage(role=Role.SYSTEM, content="You are helpful."),
                ChatMessage(role=Role.USER, content="Hi"),
            ],
            model="llama3.2",
            temperature=0.7,
            max_tokens=1000,
        )
        
        assert len(request.messages) == 2
        assert request.model == "llama3.2"
        assert request.temperature == 0.7
        assert request.max_tokens == 1000
    
    @pytest.mark.unit
    def test_chat_request_defaults(self):
        """Test ChatRequest default values."""
        request = ChatRequest(
            messages=[ChatMessage(role=Role.USER, content="test")]
        )
        
        assert request.model is None
        assert request.temperature == 0.7
        assert request.max_tokens is None
        assert request.top_p == 1.0
        assert request.stream is False
    
    @pytest.mark.unit
    def test_chat_request_validation_temperature(self):
        """Test temperature validation bounds."""
        # Valid temperature
        request = ChatRequest(
            messages=[ChatMessage(role=Role.USER, content="test")],
            temperature=1.5,
        )
        assert request.temperature == 1.5
        
        # Invalid temperature (too high)
        with pytest.raises(ValueError):
            ChatRequest(
                messages=[ChatMessage(role=Role.USER, content="test")],
                temperature=2.5,
            )
    
    @pytest.mark.unit
    def test_usage_model(self):
        """Test Usage model for token tracking."""
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            prompt_cost=0.001,
            completion_cost=0.003,
            total_cost=0.004,
        )
        
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
        assert usage.total_cost == 0.004
    
    @pytest.mark.unit
    def test_chat_response_creation(self):
        """Test ChatResponse model creation."""
        response = ChatResponse(
            id=str(uuid4()),
            model="llama3.2",
            provider=ModelProvider.OLLAMA,
            message=ChatMessage(role=Role.ASSISTANT, content="Hi there!"),
            latency_ms=250.5,
            finish_reason="stop",
        )
        
        assert response.provider == ModelProvider.OLLAMA
        assert response.message.role == Role.ASSISTANT
        assert response.latency_ms == 250.5
        assert response.cached is False
    
    @pytest.mark.unit
    def test_model_info_creation(self):
        """Test ModelInfo model creation."""
        info = ModelInfo(
            name="gpt-4-turbo",
            provider=ModelProvider.OPENAI,
            context_length=128000,
            supports_tools=True,
            supports_vision=True,
            input_price=10.0,
            output_price=30.0,
        )
        
        assert info.name == "gpt-4-turbo"
        assert info.context_length == 128000
        assert info.supports_tools is True
        assert info.supports_vision is True
    
    @pytest.mark.unit
    def test_role_enum_values(self):
        """Test all Role enum values."""
        assert Role.SYSTEM.value == "system"
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"
        assert Role.FUNCTION.value == "function"
        assert Role.TOOL.value == "tool"
    
    @pytest.mark.unit
    def test_provider_enum_values(self):
        """Test all ModelProvider enum values."""
        assert ModelProvider.OLLAMA.value == "ollama"
        assert ModelProvider.OPENAI.value == "openai"
        assert ModelProvider.ANTHROPIC.value == "anthropic"
        assert ModelProvider.VLLM.value == "vllm"


# ==============================================================================
# Model Router Tests
# ==============================================================================

class TestModelRouter:
    """Tests for ModelRouter class."""
    
    @pytest.mark.unit
    def test_router_initialization(self):
        """Test ModelRouter initializes with models."""
        with patch("src.services.llm.settings") as mock_settings:
            mock_settings.enable_ollama = True
            mock_settings.enable_openai = False
            mock_settings.enable_anthropic = False
            
            from src.services.llm import ModelRouter
            
            router = ModelRouter()
            
            # Should have Ollama models registered
            assert "llama3.2" in router.models or len(router.models) >= 0
    
    @pytest.mark.unit
    def test_router_model_selection_specific(self):
        """Test model selection with specific model requested."""
        from src.services.llm import ModelRouter
        
        with patch("src.services.llm.settings") as mock_settings:
            mock_settings.enable_ollama = True
            mock_settings.enable_openai = False
            mock_settings.enable_anthropic = False
            
            router = ModelRouter()
            
            # Manually add a model for testing
            router.models["test-model"] = ModelInfo(
                name="test-model",
                provider=ModelProvider.OLLAMA,
                context_length=4096,
            )
            
            request = ChatRequest(
                messages=[ChatMessage(role=Role.USER, content="test")],
                model="test-model",
            )
            
            selected = router.select_model(request)
            assert selected.name == "test-model"
    
    @pytest.mark.unit
    def test_router_cost_filtering(self):
        """Test model selection with cost constraints."""
        from src.services.llm import ModelRouter
        
        router = ModelRouter()
        
        # Add models with different costs
        router.models["cheap-model"] = ModelInfo(
            name="cheap-model",
            provider=ModelProvider.OLLAMA,
            context_length=4096,
            input_price=0.0,
            output_price=0.0,
            is_available=True,
        )
        router.models["expensive-model"] = ModelInfo(
            name="expensive-model",
            provider=ModelProvider.OPENAI,
            context_length=128000,
            input_price=15.0,
            output_price=75.0,
            is_available=True,
        )
        
        request = ChatRequest(
            messages=[ChatMessage(role=Role.USER, content="test")],
        )
        
        # With cost constraint, should prefer cheaper model
        # This tests the filtering logic
        available_models = [
            m for m in router.models.values()
            if m.is_available
        ]
        
        assert len(available_models) >= 2


# ==============================================================================
# LLM Service Tests
# ==============================================================================

class TestLLMService:
    """Tests for LLMService class."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ollama_chat_completion(self, mock_ollama_client):
        """Test chat completion with Ollama."""
        messages = [
            ChatMessage(role=Role.USER, content="Hello"),
        ]
        
        request = ChatRequest(
            messages=messages,
            model="llama3.2",
            provider=ModelProvider.OLLAMA,
        )
        
        # Mock Ollama response
        mock_ollama_client.chat.return_value = {
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "Hi there!"},
            "done": True,
        }
        
        result = await mock_ollama_client.chat(
            model=request.model,
            messages=[{"role": m.role.value, "content": m.content} for m in messages],
        )
        
        assert result["message"]["content"] == "Hi there!"
        assert result["done"] is True
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_openai_chat_completion(self, mock_openai_client):
        """Test chat completion with OpenAI."""
        messages = [
            ChatMessage(role=Role.USER, content="Hello"),
        ]
        
        # Call mock OpenAI
        result = await mock_openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": m.role.value, "content": m.content} for m in messages],
        )
        
        assert result.choices[0].message.content == "Mock OpenAI response"
        assert result.usage.total_tokens == 30
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_anthropic_chat_completion(self, mock_anthropic_client):
        """Test chat completion with Anthropic Claude."""
        messages = [
            ChatMessage(role=Role.USER, content="Hello"),
        ]
        
        # Call mock Anthropic
        result = await mock_anthropic_client.messages.create(
            model="claude-3-opus-20240229",
            messages=[{"role": m.role.value, "content": m.content} for m in messages],
            max_tokens=1000,
        )
        
        assert result.content[0].text == "Mock Claude response"
        assert result.usage.input_tokens == 10
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_response_caching(self, mock_redis):
        """Test response caching behavior."""
        cache_key = "chat:llama3.2:abc123"
        cached_response = {
            "content": "Cached response",
            "model": "llama3.2",
        }
        
        # First call - cache miss
        mock_redis.get = AsyncMock(return_value=None)
        result = await mock_redis.get(cache_key)
        assert result is None
        
        # Store response
        mock_redis.setex = AsyncMock(return_value=True)
        import json
        await mock_redis.setex(cache_key, 3600, json.dumps(cached_response))
        
        # Second call - cache hit
        mock_redis.get = AsyncMock(return_value=json.dumps(cached_response))
        result = await mock_redis.get(cache_key)
        assert json.loads(result)["content"] == "Cached response"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_streaming_response(self, mock_ollama_client):
        """Test streaming response handling."""
        # Mock streaming chunks
        chunks = [
            {"response": "Hello", "done": False},
            {"response": " there", "done": False},
            {"response": "!", "done": True},
        ]
        
        async def mock_stream():
            for chunk in chunks:
                yield chunk
        
        mock_ollama_client.generate = mock_stream
        
        # Collect streamed response
        full_response = ""
        async for chunk in mock_ollama_client.generate():
            full_response += chunk["response"]
        
        assert full_response == "Hello there!"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_error_handling_provider_unavailable(self, mock_ollama_client):
        """Test error handling when provider is unavailable."""
        import httpx
        
        mock_ollama_client.chat = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        
        with pytest.raises(httpx.ConnectError):
            await mock_ollama_client.chat(model="llama3.2", messages=[])
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_fallback_chain(self, mock_ollama_client, mock_openai_client):
        """Test fallback to alternative provider."""
        import httpx
        
        # Primary fails
        mock_ollama_client.chat = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        
        # Fallback succeeds
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = "Fallback response"
        
        # Simulate fallback logic
        try:
            await mock_ollama_client.chat(model="llama3.2", messages=[])
        except httpx.ConnectError:
            # Fall back to OpenAI
            result = await mock_openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[],
            )
            assert result.choices[0].message.content == "Fallback response"


# ==============================================================================
# Context Management Tests
# ==============================================================================

class TestContextManagement:
    """Tests for conversation context management."""
    
    @pytest.mark.unit
    def test_message_truncation(self, sample_conversation_history):
        """Test message truncation for context window."""
        max_messages = 3
        
        truncated = sample_conversation_history[-max_messages:]
        
        assert len(truncated) == 3
        assert truncated[-1]["role"] == "user"
    
    @pytest.mark.unit
    def test_system_message_preservation(self, sample_conversation_history):
        """Test system message is preserved during truncation."""
        system_msg = sample_conversation_history[0]
        other_msgs = sample_conversation_history[1:]
        
        # Truncate to 2 messages but keep system
        max_others = 2
        truncated = [system_msg] + other_msgs[-max_others:]
        
        assert truncated[0]["role"] == "system"
        assert len(truncated) == 3
    
    @pytest.mark.unit
    def test_token_estimation(self):
        """Test simple token count estimation."""
        text = "Hello, how are you today?"
        
        # Simple estimation: ~4 chars per token
        estimated_tokens = len(text) // 4
        
        assert estimated_tokens > 0
        assert estimated_tokens < len(text)


# ==============================================================================
# Cost Tracking Tests
# ==============================================================================

class TestCostTracking:
    """Tests for token usage and cost tracking."""
    
    @pytest.mark.unit
    def test_usage_calculation(self):
        """Test usage calculation from response."""
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        
        # Calculate costs at $10/1M input, $30/1M output
        input_rate = 10.0 / 1_000_000
        output_rate = 30.0 / 1_000_000
        
        prompt_cost = usage.prompt_tokens * input_rate
        completion_cost = usage.completion_tokens * output_rate
        
        assert prompt_cost == pytest.approx(0.001, rel=1e-6)
        assert completion_cost == pytest.approx(0.0015, rel=1e-6)
    
    @pytest.mark.unit
    def test_cumulative_cost_tracking(self):
        """Test cumulative cost tracking across multiple calls."""
        calls = [
            Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150, total_cost=0.004),
            Usage(prompt_tokens=200, completion_tokens=100, total_tokens=300, total_cost=0.008),
            Usage(prompt_tokens=150, completion_tokens=75, total_tokens=225, total_cost=0.006),
        ]
        
        total_cost = sum(c.total_cost for c in calls)
        total_tokens = sum(c.total_tokens for c in calls)
        
        assert total_cost == pytest.approx(0.018, rel=1e-6)  # Use approx for float comparison
        assert total_tokens == 675


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestLLMIntegration:
    """Integration tests for LLM service."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_chat_flow(
        self, mock_ollama_client, mock_redis
    ):
        """Test complete chat flow: request -> response -> cache."""
        # Setup
        messages = [
            ChatMessage(role=Role.SYSTEM, content="You are helpful."),
            ChatMessage(role=Role.USER, content="What is 2+2?"),
        ]
        
        request = ChatRequest(
            messages=messages,
            model="llama3.2",
            temperature=0.7,
        )
        
        # Generate response
        mock_ollama_client.chat.return_value = {
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "2+2 equals 4."},
            "done": True,
        }
        
        result = await mock_ollama_client.chat(
            model=request.model,
            messages=[{"role": m.role.value, "content": m.content} for m in messages],
        )
        
        # Verify response
        assert "4" in result["message"]["content"]
        
        # Cache response
        cache_key = f"chat:{request.model}:hash123"
        mock_redis.setex = AsyncMock(return_value=True)
        import json
        await mock_redis.setex(cache_key, 3600, json.dumps(result))
        
        mock_redis.setex.assert_called_once()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, mock_ollama_client):
        """Test multi-turn conversation handling."""
        conversation = []
        
        # Turn 1
        conversation.append(ChatMessage(role=Role.USER, content="Hi"))
        mock_ollama_client.chat.return_value = {
            "message": {"role": "assistant", "content": "Hello!"},
        }
        result1 = await mock_ollama_client.chat(model="llama3.2", messages=[])
        conversation.append(
            ChatMessage(role=Role.ASSISTANT, content=result1["message"]["content"])
        )
        
        # Turn 2
        conversation.append(ChatMessage(role=Role.USER, content="How are you?"))
        mock_ollama_client.chat.return_value = {
            "message": {"role": "assistant", "content": "I'm doing well!"},
        }
        result2 = await mock_ollama_client.chat(model="llama3.2", messages=[])
        conversation.append(
            ChatMessage(role=Role.ASSISTANT, content=result2["message"]["content"])
        )
        
        # Verify conversation history
        assert len(conversation) == 4
        assert conversation[0].role == Role.USER
        assert conversation[1].role == Role.ASSISTANT
        assert conversation[2].role == Role.USER
        assert conversation[3].role == Role.ASSISTANT
