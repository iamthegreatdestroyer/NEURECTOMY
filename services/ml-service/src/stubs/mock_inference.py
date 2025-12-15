"""Mock Inference Bridge for Testing"""

import time
from typing import AsyncIterator, Dict, Optional

from ..api.types import InferenceBackend, InferenceConfig
from ..api.interfaces import InferenceBridge


class MockInferenceBridge(InferenceBridge):
    """Mock inference bridge for integration testing."""

    def __init__(self, backend: InferenceBackend = InferenceBackend.MOCK):
        self._backend = backend
        self._available = True
        self._request_count = 0

    async def generate(self, prompt: str, config: Optional[InferenceConfig] = None) -> str:
        self._request_count += 1
        config = config or InferenceConfig()
        
        # Simulate processing time
        time.sleep(0.01)
        
        return f"[MockInference] Response to: {prompt[:100]}... (max_tokens={config.max_tokens})"

    async def stream_generate(self, prompt: str, config: Optional[InferenceConfig] = None) -> AsyncIterator[str]:
        words = f"[MockInference] Streaming response to: {prompt[:50]}".split()
        for word in words:
            yield word + " "

    def get_backend_info(self) -> Dict:
        return {
            "backend": self._backend.name,
            "model": "mock-model",
            "context_window": 4096,
            "tokens_per_second": 20.0,
            "request_count": self._request_count,
        }

    def is_available(self) -> bool:
        return self._available

    def switch_backend(self, backend: InferenceBackend) -> bool:
        self._backend = backend
        return True
