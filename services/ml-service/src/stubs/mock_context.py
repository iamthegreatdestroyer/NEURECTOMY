"""Mock Context Manager for Testing"""

import uuid
from typing import Dict, List, Optional

from ..api.types import CompressionLevel, ContextScope, ContextWindow
from ..api.interfaces import ContextManagerProtocol


class MockContextManager(ContextManagerProtocol):
    """Mock context manager for integration testing."""

    def __init__(self):
        self._cache: Dict[str, ContextWindow] = {}

    def build_context(
        self,
        scope: ContextScope,
        files: Optional[List[str]] = None,
        conversation_id: Optional[str] = None,
    ) -> ContextWindow:
        context_id = f"ctx_{uuid.uuid4().hex[:8]}"
        
        file_contents = {}
        if files:
            for f in files:
                file_contents[f] = f"[Mock content for {f}]"

        return ContextWindow(
            context_id=context_id,
            scope=scope,
            files=file_contents,
            conversation_history=[],
            is_compressed=False,
            compression_ratio=1.0,
            total_tokens=sum(len(c.split()) for c in file_contents.values()),
            effective_tokens=sum(len(c.split()) for c in file_contents.values()),
        )

    def compress_context(
        self,
        context: ContextWindow,
        level: CompressionLevel = CompressionLevel.BALANCED,
    ) -> ContextWindow:
        compression_ratios = {
            CompressionLevel.NONE: 1.0,
            CompressionLevel.LIGHT: 5.0,
            CompressionLevel.BALANCED: 15.0,
            CompressionLevel.AGGRESSIVE: 30.0,
        }
        
        ratio = compression_ratios.get(level, 15.0)
        
        return ContextWindow(
            context_id=context.context_id,
            scope=context.scope,
            files=context.files,
            conversation_history=context.conversation_history,
            is_compressed=True,
            compression_ratio=ratio,
            sigma_encoded=b"[mock_sigma_encoded]",
            total_tokens=context.total_tokens,
            effective_tokens=int(context.total_tokens / ratio),
        )

    def decompress_context(self, context: ContextWindow) -> ContextWindow:
        return ContextWindow(
            context_id=context.context_id,
            scope=context.scope,
            files=context.files,
            conversation_history=context.conversation_history,
            is_compressed=False,
            compression_ratio=1.0,
            total_tokens=context.total_tokens,
            effective_tokens=context.total_tokens,
        )

    def get_cached_context(self, context_id: str) -> Optional[ContextWindow]:
        return self._cache.get(context_id)

    def cache_context(self, context: ContextWindow) -> str:
        self._cache[context.context_id] = context
        return context.context_id
