"""
Component Bridges
=================

Bridges connecting Neurectomy to Ryot LLM, ΣLANG, and ΣVAULT.
"""

from typing import Optional, Tuple, List, Generator

from .types import TaskRequest, TaskResult, TaskStatus


class InferenceBridge:
    """
    Bridge to Ryot LLM inference engine.
    """
    
    def __init__(self):
        self._engine = None
        self._ready = False
        self._init_engine()
    
    def _init_engine(self) -> None:
        """Initialize Ryot LLM engine."""
        try:
            # Try real engine first
            from ryot_llm.core.engine import RyotEngine
            # Would need model path - use mock for now
            raise ImportError("Use mock for initial setup")
        except ImportError:
            # Use mock
            try:
                from ryot_llm.stubs import MockInferenceEngine
                self._engine = MockInferenceEngine()
                self._ready = True
            except ImportError:
                self._ready = False
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 1.0,
        use_compression: bool = True,
        conversation_id: Optional[str] = None,
    ) -> Tuple[str, dict]:
        """
        Generate text using Ryot LLM.
        
        Returns (generated_text, metadata)
        """
        if not self._ready or self._engine is None:
            return "", {"error": "Engine not ready"}
        
        try:
            from ryot_llm.api.types import GenerationConfig
            
            config = GenerationConfig(
                max_tokens=max_tokens,
                temperature=temperature,
                use_sigma_compression=use_compression,
                conversation_id=conversation_id,
            )
            
            result = self._engine.generate(prompt, config)
            
            return result.generated_text, {
                "tokens_generated": result.completion_tokens,
                "tokens_processed": result.prompt_tokens,
                "compression_ratio": result.compression_ratio,
                "rsu_reference": result.rsu_reference,
            }
        except Exception as e:
            return "", {"error": str(e)}
    
    def stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 1.0,
    ) -> Generator[str, None, None]:
        """Stream generation token by token."""
        if not self._ready or self._engine is None:
            return
        
        try:
            from ryot_llm.api.types import GenerationConfig
            
            config = GenerationConfig(
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            for chunk in self._engine.stream(prompt, config):
                yield chunk.text
        except Exception:
            return
    
    def is_ready(self) -> bool:
        """Check if inference is ready."""
        return self._ready
    
    def get_model_info(self) -> dict:
        """Get model information."""
        if not self._ready or self._engine is None:
            return {}
        
        info = self._engine.get_model_info()
        return {
            "model_name": info.model_name,
            "vocab_size": info.vocab_size,
            "context_window": info.context_window,
        }


class CompressionBridge:
    """
    Bridge to ΣLANG compression.
    """
    
    def __init__(self):
        self._engine = None
        self._ready = False
        self._init_engine()
    
    def _init_engine(self) -> None:
        """Initialize ΣLANG compression."""
        try:
            from sigmalang.adapters import SigmaCompressionAdapter
            self._engine = SigmaCompressionAdapter()
            self._ready = True
        except ImportError:
            self._ready = False
    
    def compress(
        self,
        tokens: List[int],
        conversation_id: Optional[str] = None,
    ) -> Tuple[bytes, dict]:
        """
        Compress tokens using ΣLANG.
        
        Returns (compressed_data, metadata)
        """
        if not self._ready or self._engine is None:
            return b"", {"error": "Compression not ready"}
        
        try:
            from sigmalang.adapters import RyotTokenSequence
            
            token_seq = RyotTokenSequence.from_list(tokens)
            encoded = self._engine.encode(token_seq, conversation_id)
            
            return encoded.glyph_sequence, {
                "original_tokens": len(tokens),
                "compressed_glyphs": encoded.compressed_glyph_count,
                "compression_ratio": encoded.compression_ratio,
                "semantic_hash": encoded.semantic_hash,
            }
        except Exception as e:
            return b"", {"error": str(e)}
    
    def decompress(self, data: bytes) -> Tuple[List[int], dict]:
        """Decompress back to tokens."""
        if not self._ready or self._engine is None:
            return [], {"error": "Compression not ready"}
        
        try:
            from sigmalang.adapters import RyotSigmaEncodedContext
            
            # Create context from bytes
            context = RyotSigmaEncodedContext(
                glyph_sequence=data,
                original_token_count=0,
                compressed_glyph_count=len(data) // 2,
            )
            
            decoded = self._engine.decode(context)
            return decoded.to_list(), {}
        except Exception as e:
            return [], {"error": str(e)}
    
    def get_compression_ratio(self) -> float:
        """Get current compression ratio."""
        if self._engine:
            return self._engine.get_compression_ratio()
        return 1.0
    
    def is_ready(self) -> bool:
        """Check if compression is ready."""
        return self._ready


class StorageBridge:
    """
    Bridge to ΣVAULT storage.
    """
    
    def __init__(self):
        self._storage = None
        self._ready = False
        self._init_storage()
    
    def _init_storage(self) -> None:
        """Initialize ΣVAULT storage."""
        try:
            from sigmavault.rsu import RSUStorage
            self._storage = RSUStorage()
            self._ready = True
        except ImportError:
            self._ready = False
    
    def store_rsu(
        self,
        glyph_data: bytes,
        semantic_hash: int,
        token_count: int,
        kv_cache: Optional[bytes] = None,
        conversation_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Store RSU in vault.
        
        Returns RSU ID or None.
        """
        if not self._ready or self._storage is None:
            return None
        
        try:
            entry = self._storage.store(
                glyph_data=glyph_data,
                semantic_hash=semantic_hash,
                original_token_count=token_count,
                kv_cache_data=kv_cache,
                conversation_id=conversation_id,
            )
            return entry.rsu_id
        except Exception:
            return None
    
    def retrieve_rsu(self, rsu_id: str) -> Optional[Tuple[bytes, Optional[bytes]]]:
        """
        Retrieve RSU from vault.
        
        Returns (glyph_data, kv_cache) or None.
        """
        if not self._ready or self._storage is None:
            return None
        
        try:
            stored = self._storage.retrieve(rsu_id)
            if stored:
                return stored.glyph_data, stored.kv_cache_data
        except Exception:
            pass
        
        return None
    
    def find_similar(
        self,
        semantic_hash: int,
        threshold: float = 0.85,
    ) -> List[str]:
        """Find similar RSUs by semantic hash."""
        if not self._ready or self._storage is None:
            return []
        
        try:
            matches = self._storage.find_similar(semantic_hash, threshold)
            return [m.rsu_id for m in matches]
        except Exception:
            return []
    
    def is_ready(self) -> bool:
        """Check if storage is ready."""
        return self._ready
    
    def get_statistics(self) -> dict:
        """Get storage statistics."""
        if self._storage:
            return self._storage.get_statistics()
        return {}
