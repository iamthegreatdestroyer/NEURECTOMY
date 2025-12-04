"""
LLM API Routes.

@LINGUA @SYNAPSE - LLM inference endpoints with streaming support.
"""

import asyncio
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.models.llm import (
    ChatRequest,
    ChatResponse,
    ModelProvider,
    ModelInfo,
)
from src.services.llm import LLMService

router = APIRouter()

# Service dependency
_llm_service: Optional[LLMService] = None


async def get_llm_service() -> LLMService:
    """Dependency to get LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
        await _llm_service.initialize()
    return _llm_service


class ModelsResponse(BaseModel):
    """Response with available models."""
    models: list[ModelInfo]


class StreamChunk(BaseModel):
    """Streaming response chunk."""
    content: str
    done: bool = False
    finish_reason: Optional[str] = None


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    llm: LLMService = Depends(get_llm_service),
) -> ChatResponse:
    """
    Chat completion endpoint.
    
    @LINGUA - Generates chat response using configured LLM.
    
    Supports multiple providers:
    - Ollama (local)
    - OpenAI
    - Anthropic
    
    Example:
        ```json
        {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"}
            ],
            "model": "llama3.2:latest",
            "temperature": 0.7
        }
        ```
    """
    try:
        return await llm.chat(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    llm: LLMService = Depends(get_llm_service),
) -> StreamingResponse:
    """
    Streaming chat completion endpoint.
    
    @LINGUA @STREAM - Real-time token streaming for better UX.
    
    Returns Server-Sent Events (SSE) stream.
    """
    request.stream = True
    
    async def generate() -> AsyncGenerator[str, None]:
        try:
            async for chunk in llm.chat_stream(request):
                yield f"data: {chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {{'error': '{str(e)}'}}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/models", response_model=ModelsResponse)
async def list_models(
    llm: LLMService = Depends(get_llm_service),
) -> ModelsResponse:
    """
    List available LLM models.
    
    Returns all models from all configured providers.
    """
    models = await llm.list_models()
    return ModelsResponse(models=models)


@router.get("/models/{provider}", response_model=ModelsResponse)
async def list_provider_models(
    provider: ModelProvider,
    llm: LLMService = Depends(get_llm_service),
) -> ModelsResponse:
    """
    List models from a specific provider.
    
    Args:
        provider: Model provider (ollama, openai, anthropic)
    """
    all_models = await llm.list_models()
    filtered = [m for m in all_models if m.provider == provider]
    return ModelsResponse(models=filtered)


class TokenCountRequest(BaseModel):
    """Request for token counting."""
    text: str
    model: str = "llama3.2:latest"


class TokenCountResponse(BaseModel):
    """Response with token count."""
    tokens: int
    model: str


@router.post("/tokens/count", response_model=TokenCountResponse)
async def count_tokens(
    request: TokenCountRequest,
    llm: LLMService = Depends(get_llm_service),
) -> TokenCountResponse:
    """
    Count tokens in text.
    
    Useful for estimating API costs and context limits.
    """
    # Simple approximation (4 chars per token on average)
    # In production, use tokenizer from tiktoken or model-specific
    token_count = len(request.text) // 4
    
    return TokenCountResponse(
        tokens=token_count,
        model=request.model,
    )


class GenerateRequest(BaseModel):
    """Simple generation request."""
    prompt: str
    model: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.7
    stop: list[str] = Field(default_factory=list)


class GenerateResponse(BaseModel):
    """Simple generation response."""
    text: str
    model: str
    tokens_used: int


@router.post("/generate", response_model=GenerateResponse)
async def generate(
    request: GenerateRequest,
    llm: LLMService = Depends(get_llm_service),
) -> GenerateResponse:
    """
    Simple text generation endpoint.
    
    Simpler interface for completion-style generation.
    """
    from src.models.llm import ChatMessage, ChatRequest as CR, Role
    
    chat_request = CR(
        messages=[ChatMessage(role=Role.USER, content=request.prompt)],
        model=request.model,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        stop=request.stop,
    )
    
    response = await llm.chat(chat_request)
    
    return GenerateResponse(
        text=response.message.content,
        model=response.model,
        tokens_used=response.usage.total_tokens if response.usage else 0,
    )
