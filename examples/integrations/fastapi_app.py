#!/usr/bin/env python3
"""Example FastAPI integration."""

from fastapi import FastAPI
from pydantic import BaseModel
from neurectomy import NeurectomyOrchestrator

app = FastAPI(
    title="Neurectomy AI App",
    description="Example FastAPI integration with Neurectomy",
    version="1.0.0",
)

# Initialize orchestrator
orchestrator = NeurectomyOrchestrator()


class GenerateRequest(BaseModel):
    """Request model for generation."""

    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7


class GenerateResponse(BaseModel):
    """Response model for generation."""

    text: str
    tokens: int
    latency_ms: float


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text from prompt."""
    result = orchestrator.generate(
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    return GenerateResponse(
        text=result.generated_text,
        tokens=result.tokens_generated,
        latency_ms=result.execution_time_ms,
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "neurectomy-api"}


@app.get("/info")
async def info():
    """Get service information."""
    return {
        "name": "Neurectomy AI Service",
        "version": "1.0.0",
        "capabilities": {
            "generation": True,
            "streaming": False,  # Requires different endpoint
            "agents": 40,
            "teams": 5,
        },
    }


if __name__ == "__main__":
    import uvicorn

    print("Starting Neurectomy API server...")
    print("Visit http://localhost:8080/docs for interactive API docs")

    uvicorn.run(app, host="0.0.0.0", port=8080)
