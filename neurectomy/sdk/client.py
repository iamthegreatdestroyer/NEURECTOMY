"""
Neurectomy Python SDK
=====================

Client library for interacting with Neurectomy API.
"""

from typing import Optional, Dict, Any, Generator, List
from dataclasses import dataclass
import httpx
import json


@dataclass
class NeurectomyConfig:
    """SDK configuration."""
    
    base_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 3


class NeurectomyClient:
    """
    Python client for Neurectomy API.
    
    Usage:
        client = NeurectomyClient()
        response = client.generate("Hello, world!")
        print(response.text)
    """
    
    def __init__(self, config: Optional[NeurectomyConfig] = None):
        self.config = config or NeurectomyConfig()
        
        self._client = httpx.Client(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            headers=self._build_headers(),
        )
    
    def _build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers
    
    # =========================================================================
    # GENERATION
    # =========================================================================
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters
        
        Returns:
            Generation response
        """
        response = self._client.post(
            "/v1/generate",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs,
            },
        )
        response.raise_for_status()
        return response.json()
    
    def stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """
        Stream text generation.
        
        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Yields:
            Generated text chunks
        """
        with self._client.stream(
            "POST",
            "/v1/generate/stream",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    if "text" in data:
                        yield data["text"]
                    if data.get("done"):
                        break
    
    # =========================================================================
    # AGENTS
    # =========================================================================
    
    def list_agents(self) -> Dict[str, Any]:
        """List all available agents."""
        response = self._client.get("/v1/agents")
        response.raise_for_status()
        return response.json()
    
    def get_agent(self, agent_id: str) -> Dict[str, Any]:
        """Get agent details."""
        response = self._client.get(f"/v1/agents/{agent_id}")
        response.raise_for_status()
        return response.json()
    
    def execute_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        agent_id: Optional[str] = None,
        team: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a task using the Elite Collective.
        
        Args:
            task_type: Type of task (generate, summarize, analyze, code, translate)
            payload: Task-specific payload
            agent_id: Specific agent to use
            team: Specific team to use
        
        Returns:
            Task result
        """
        response = self._client.post(
            "/v1/agents/task",
            json={
                "task_type": task_type,
                "payload": payload,
                "agent_id": agent_id,
                "team": team,
            },
        )
        response.raise_for_status()
        return response.json()
    
    # =========================================================================
    # CONVERSATIONS
    # =========================================================================
    
    def chat(
        self,
        conversation_id: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
    ) -> Dict[str, Any]:
        """
        Send message in conversation context.
        
        Args:
            conversation_id: Conversation ID
            messages: List of {"role": "user/assistant", "content": "..."}
            max_tokens: Maximum response tokens
        
        Returns:
            Response with assistant message
        """
        response = self._client.post(
            f"/v1/conversations/{conversation_id}/message",
            json={
                "conversation_id": conversation_id,
                "messages": messages,
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        return response.json()
    
    # =========================================================================
    # HEALTH & MONITORING
    # =========================================================================
    
    def health(self) -> Dict[str, Any]:
        """Get system health status."""
        response = self._client.get("/health")
        response.raise_for_status()
        return response.json()
    
    def metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        response = self._client.get("/metrics")
        response.raise_for_status()
        return response.json()
    
    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        try:
            health = self.health()
            return health.get("status") == "healthy"
        except Exception:
            return False
    
    # =========================================================================
    # CONTEXT MANAGER
    # =========================================================================
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self._client.close()
    
    def close(self):
        """Close the client."""
        self._client.close()


class AsyncNeurectomyClient:
    """
    Async Python client for Neurectomy API.
    
    Usage:
        async with AsyncNeurectomyClient() as client:
            response = await client.generate("Hello!")
    """
    
    def __init__(self, config: Optional[NeurectomyConfig] = None):
        self.config = config or NeurectomyConfig()
        
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            headers=self._build_headers(),
        )
    
    def _build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate text asynchronously."""
        response = await self._client.post(
            "/v1/generate",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs,
            },
        )
        response.raise_for_status()
        return response.json()
    
    async def health(self) -> Dict[str, Any]:
        """Get system health asynchronously."""
        response = await self._client.get("/health")
        response.raise_for_status()
        return response.json()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        await self._client.aclose()
    
    async def close(self):
        """Close the async client."""
        await self._client.aclose()
