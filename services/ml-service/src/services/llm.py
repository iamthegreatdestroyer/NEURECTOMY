"""
LLM Service with multi-model orchestration.

@LINGUA @OMNISCIENT - Multi-model LLM integration with intelligent routing.

Features:
- vLLM inference server integration
- Ollama local model support
- OpenAI/Anthropic cloud providers
- Intelligent model routing
- Response caching
- Cost optimization
- Fallback chains
"""

import asyncio
import hashlib
import time
from typing import Any, AsyncGenerator, Optional

import httpx
import structlog

from src.config import settings
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
from src.db.redis import get_redis, RedisClient

logger = structlog.get_logger()


class ModelRouter:
    """
    Intelligent model router for multi-model orchestration.
    
    @LINGUA @OMNISCIENT - Routes requests to optimal model based on:
    - Task complexity
    - Context length
    - Cost constraints
    - Availability
    - Performance requirements
    """
    
    def __init__(self):
        self.models: dict[str, ModelInfo] = {}
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Register available models."""
        # Ollama models (local)
        if settings.enable_ollama:
            self.models["llama3.2"] = ModelInfo(
                name="llama3.2",
                provider=ModelProvider.OLLAMA,
                context_length=128000,
                supports_tools=True,
                is_default=True,
                description="Llama 3.2 - Fast local inference",
            )
            self.models["codellama"] = ModelInfo(
                name="codellama",
                provider=ModelProvider.OLLAMA,
                context_length=16384,
                supports_tools=False,
                description="Code Llama - Specialized for code",
                tags=["code", "programming"],
            )
            self.models["mistral"] = ModelInfo(
                name="mistral",
                provider=ModelProvider.OLLAMA,
                context_length=32768,
                supports_tools=True,
                description="Mistral 7B - Efficient reasoning",
            )
        
        # OpenAI models
        if settings.enable_openai:
            self.models["gpt-4-turbo-preview"] = ModelInfo(
                name="gpt-4-turbo-preview",
                provider=ModelProvider.OPENAI,
                context_length=128000,
                supports_tools=True,
                supports_vision=True,
                input_price=10.0,
                output_price=30.0,
                description="GPT-4 Turbo - Most capable",
            )
            self.models["gpt-4o"] = ModelInfo(
                name="gpt-4o",
                provider=ModelProvider.OPENAI,
                context_length=128000,
                supports_tools=True,
                supports_vision=True,
                input_price=5.0,
                output_price=15.0,
                description="GPT-4o - Fast multimodal",
            )
        
        # Anthropic models
        if settings.enable_anthropic:
            self.models["claude-3-opus-20240229"] = ModelInfo(
                name="claude-3-opus-20240229",
                provider=ModelProvider.ANTHROPIC,
                context_length=200000,
                supports_tools=True,
                input_price=15.0,
                output_price=75.0,
                description="Claude 3 Opus - Most intelligent",
            )
            self.models["claude-3-sonnet-20240229"] = ModelInfo(
                name="claude-3-sonnet-20240229",
                provider=ModelProvider.ANTHROPIC,
                context_length=200000,
                supports_tools=True,
                input_price=3.0,
                output_price=15.0,
                description="Claude 3 Sonnet - Balanced",
            )
    
    def select_model(
        self,
        request: ChatRequest,
        task_type: Optional[str] = None,
        max_cost_per_1k_tokens: Optional[float] = None,
    ) -> ModelInfo:
        """
        Select optimal model for request.
        
        @OMNISCIENT - Intelligent model selection.
        """
        # If specific model requested, use it
        if request.model and request.model in self.models:
            return self.models[request.model]
        
        # Filter available models
        available = [m for m in self.models.values() if m.is_available]
        
        # Apply cost constraint
        if max_cost_per_1k_tokens is not None:
            available = [
                m for m in available
                if m.input_price is None or m.input_price <= max_cost_per_1k_tokens
            ]
        
        # Task-specific routing
        if task_type == "code":
            code_models = [m for m in available if "code" in m.tags]
            if code_models:
                available = code_models
        
        # Check context requirements
        total_context = sum(len(m.content) for m in request.messages)
        available = [m for m in available if m.context_length > total_context * 2]
        
        # Tool use requirement
        if request.tools:
            available = [m for m in available if m.supports_tools]
        
        # Return default or first available
        defaults = [m for m in available if m.is_default]
        if defaults:
            return defaults[0]
        
        return available[0] if available else list(self.models.values())[0]
    
    def get_fallback_chain(self, model: ModelInfo) -> list[ModelInfo]:
        """Get fallback models if primary fails."""
        fallbacks = []
        
        # Same provider first
        same_provider = [
            m for m in self.models.values()
            if m.provider == model.provider and m.name != model.name
        ]
        fallbacks.extend(same_provider)
        
        # Then other providers
        other_providers = [
            m for m in self.models.values()
            if m.provider != model.provider
        ]
        fallbacks.extend(other_providers)
        
        return fallbacks[:3]  # Max 3 fallbacks


class LLMService:
    """
    Unified LLM service for NEURECTOMY.
    
    @LINGUA @TENSOR - Production LLM integration with:
    - Multi-provider support (Ollama, OpenAI, Anthropic, vLLM)
    - Response caching
    - Streaming support
    - Automatic failover
    - Usage tracking
    """
    
    def __init__(self):
        self.router = ModelRouter()
        self._http_client: Optional[httpx.AsyncClient] = None
        self._redis: Optional[RedisClient] = None
    
    async def initialize(self) -> None:
        """Initialize HTTP client and connections."""
        self._http_client = httpx.AsyncClient(timeout=120.0)
        self._redis = await get_redis()
        logger.info("✅ LLM Service initialized")
    
    async def close(self) -> None:
        """Close connections."""
        if self._http_client:
            await self._http_client.aclose()
    
    @property
    def http_client(self) -> httpx.AsyncClient:
        if not self._http_client:
            raise RuntimeError("LLM Service not initialized")
        return self._http_client
    
    # =========================================================================
    # Chat Completion
    # =========================================================================
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Generate chat completion.
        
        @LINGUA - Multi-model chat with intelligent routing.
        """
        start_time = time.time()
        
        # Select model
        model_info = self.router.select_model(request)
        logger.info(f"Selected model: {model_info.name} ({model_info.provider})")
        
        # Check cache
        if self._redis and not request.stream:
            cache_key = self._compute_cache_key(request)
            cached = await self._redis.get_cached_llm_response(
                cache_key, model_info.name
            )
            if cached:
                return ChatResponse(
                    id=f"cached-{cache_key[:8]}",
                    model=model_info.name,
                    provider=model_info.provider,
                    message=ChatMessage(role=Role.ASSISTANT, content=cached),
                    latency_ms=(time.time() - start_time) * 1000,
                    cached=True,
                )
        
        # Route to provider
        try:
            response = await self._call_provider(request, model_info)
        except Exception as e:
            logger.error(f"Primary model failed: {e}")
            # Try fallbacks
            for fallback in self.router.get_fallback_chain(model_info):
                try:
                    logger.info(f"Trying fallback: {fallback.name}")
                    response = await self._call_provider(request, fallback)
                    break
                except Exception as fallback_error:
                    logger.error(f"Fallback {fallback.name} failed: {fallback_error}")
            else:
                raise RuntimeError("All models failed")
        
        # Cache response
        if self._redis and not request.stream:
            cache_key = self._compute_cache_key(request)
            await self._redis.cache_llm_response(
                cache_key,
                response.message.content,
                response.model,
            )
        
        response.latency_ms = (time.time() - start_time) * 1000
        return response
    
    async def chat_stream(
        self,
        request: ChatRequest,
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat completion.
        
        @LINGUA @VELOCITY - Streaming for real-time responses.
        """
        request.stream = True
        model_info = self.router.select_model(request)
        
        if model_info.provider == ModelProvider.OLLAMA:
            async for chunk in self._stream_ollama(request, model_info):
                yield chunk
        elif model_info.provider == ModelProvider.OPENAI:
            async for chunk in self._stream_openai(request, model_info):
                yield chunk
        elif model_info.provider == ModelProvider.ANTHROPIC:
            async for chunk in self._stream_anthropic(request, model_info):
                yield chunk
    
    # =========================================================================
    # Provider Implementations
    # =========================================================================
    
    async def _call_provider(
        self,
        request: ChatRequest,
        model_info: ModelInfo,
    ) -> ChatResponse:
        """Route to specific provider."""
        if model_info.provider == ModelProvider.OLLAMA:
            return await self._call_ollama(request, model_info)
        elif model_info.provider == ModelProvider.OPENAI:
            return await self._call_openai(request, model_info)
        elif model_info.provider == ModelProvider.ANTHROPIC:
            return await self._call_anthropic(request, model_info)
        else:
            raise ValueError(f"Unsupported provider: {model_info.provider}")
    
    async def _call_ollama(
        self,
        request: ChatRequest,
        model_info: ModelInfo,
    ) -> ChatResponse:
        """
        Call Ollama API.
        
        @LINGUA @FLUX - Local Ollama integration.
        """
        import uuid
        
        payload = {
            "model": model_info.name,
            "messages": [
                {"role": m.role.value, "content": m.content}
                for m in request.messages
            ],
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "top_p": request.top_p,
            },
        }
        
        if request.max_tokens:
            payload["options"]["num_predict"] = request.max_tokens
        
        response = await self.http_client.post(
            f"{settings.ollama_url}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        
        return ChatResponse(
            id=str(uuid.uuid4()),
            model=model_info.name,
            provider=ModelProvider.OLLAMA,
            message=ChatMessage(
                role=Role.ASSISTANT,
                content=data["message"]["content"],
            ),
            usage=Usage(
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            ),
            latency_ms=0,  # Will be set by caller
            finish_reason="stop",
        )
    
    async def _call_openai(
        self,
        request: ChatRequest,
        model_info: ModelInfo,
    ) -> ChatResponse:
        """
        Call OpenAI API.
        
        @LINGUA - OpenAI GPT integration.
        """
        import openai
        
        client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        
        response = await client.chat.completions.create(
            model=model_info.name,
            messages=[
                {"role": m.role.value, "content": m.content}
                for m in request.messages
            ],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            stop=request.stop,
            tools=request.tools,
        )
        
        choice = response.choices[0]
        
        return ChatResponse(
            id=response.id,
            model=model_info.name,
            provider=ModelProvider.OPENAI,
            message=ChatMessage(
                role=Role.ASSISTANT,
                content=choice.message.content or "",
                tool_calls=[tc.model_dump() for tc in choice.message.tool_calls or []],
            ),
            usage=Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                prompt_cost=(response.usage.prompt_tokens / 1_000_000) * model_info.input_price if model_info.input_price else None,
                completion_cost=(response.usage.completion_tokens / 1_000_000) * model_info.output_price if model_info.output_price else None,
            ),
            latency_ms=0,
            finish_reason=choice.finish_reason,
        )
    
    async def _call_anthropic(
        self,
        request: ChatRequest,
        model_info: ModelInfo,
    ) -> ChatResponse:
        """
        Call Anthropic API.
        
        @LINGUA - Claude integration.
        """
        import anthropic
        
        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        
        # Convert messages (Anthropic has different format)
        messages = []
        system_prompt = None
        
        for msg in request.messages:
            if msg.role == Role.SYSTEM:
                system_prompt = msg.content
            else:
                messages.append({
                    "role": "user" if msg.role == Role.USER else "assistant",
                    "content": msg.content,
                })
        
        response = await client.messages.create(
            model=model_info.name,
            messages=messages,
            system=system_prompt,
            max_tokens=request.max_tokens or 4096,
            temperature=request.temperature,
            top_p=request.top_p,
            stop_sequences=request.stop,
        )
        
        return ChatResponse(
            id=response.id,
            model=model_info.name,
            provider=ModelProvider.ANTHROPIC,
            message=ChatMessage(
                role=Role.ASSISTANT,
                content=response.content[0].text,
            ),
            usage=Usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                prompt_cost=(response.usage.input_tokens / 1_000_000) * model_info.input_price if model_info.input_price else None,
                completion_cost=(response.usage.output_tokens / 1_000_000) * model_info.output_price if model_info.output_price else None,
            ),
            latency_ms=0,
            finish_reason=response.stop_reason,
        )
    
    # =========================================================================
    # Streaming Implementations
    # =========================================================================
    
    async def _stream_ollama(
        self,
        request: ChatRequest,
        model_info: ModelInfo,
    ) -> AsyncGenerator[str, None]:
        """Stream from Ollama."""
        payload = {
            "model": model_info.name,
            "messages": [
                {"role": m.role.value, "content": m.content}
                for m in request.messages
            ],
            "stream": True,
            "options": {
                "temperature": request.temperature,
                "top_p": request.top_p,
            },
        }
        
        async with self.http_client.stream(
            "POST",
            f"{settings.ollama_url}/api/chat",
            json=payload,
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]
    
    async def _stream_openai(
        self,
        request: ChatRequest,
        model_info: ModelInfo,
    ) -> AsyncGenerator[str, None]:
        """Stream from OpenAI."""
        import openai
        
        client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        
        stream = await client.chat.completions.create(
            model=model_info.name,
            messages=[
                {"role": m.role.value, "content": m.content}
                for m in request.messages
            ],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=True,
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    async def _stream_anthropic(
        self,
        request: ChatRequest,
        model_info: ModelInfo,
    ) -> AsyncGenerator[str, None]:
        """Stream from Anthropic."""
        import anthropic
        
        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        
        messages = []
        system_prompt = None
        
        for msg in request.messages:
            if msg.role == Role.SYSTEM:
                system_prompt = msg.content
            else:
                messages.append({
                    "role": "user" if msg.role == Role.USER else "assistant",
                    "content": msg.content,
                })
        
        async with client.messages.stream(
            model=model_info.name,
            messages=messages,
            system=system_prompt,
            max_tokens=request.max_tokens or 4096,
            temperature=request.temperature,
        ) as stream:
            async for text in stream.text_stream:
                yield text
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def _compute_cache_key(self, request: ChatRequest) -> str:
        """Compute cache key from request."""
        content = "".join(m.content for m in request.messages)
        content += f"{request.temperature}{request.max_tokens}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get_available_models(self) -> list[ModelInfo]:
        """Get list of available models."""
        return [m for m in self.router.models.values() if m.is_available]
