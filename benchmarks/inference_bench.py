"""
Inference Benchmarks
====================

Benchmarks for Ryot LLM inference performance.
"""

from typing import Dict, Any, Optional
from .base import Benchmark, BenchmarkConfig


class TokenGenerationBenchmark(Benchmark):
    """Benchmark token generation speed."""
    
    def __init__(
        self,
        prompt: str = "The quick brown fox",
        max_tokens: int = 100,
        config: Optional[BenchmarkConfig] = None,
    ):
        super().__init__(config)
        self.prompt = prompt
        self.max_tokens = max_tokens
        self._orchestrator = None
    
    @property
    def name(self) -> str:
        return f"token_generation_{self.max_tokens}tok"
    
    def setup(self) -> None:
        from neurectomy import NeurectomyOrchestrator
        self._orchestrator = NeurectomyOrchestrator()
    
    def run_iteration(self) -> Dict[str, Any]:
        result = self._orchestrator.generate(
            self.prompt,
            max_tokens=self.max_tokens,
        )
        
        return {
            "tokens_generated": result.tokens_generated,
            "tokens_per_second": result.tokens_generated / (result.execution_time_ms / 1000) if result.execution_time_ms > 0 else 0,
        }
    
    def teardown(self) -> None:
        self._orchestrator = None


class FirstTokenLatencyBenchmark(Benchmark):
    """Benchmark time to first token (TTFT)."""
    
    def __init__(
        self,
        prompt: str = "Hello",
        config: Optional[BenchmarkConfig] = None,
    ):
        super().__init__(config)
        self.prompt = prompt
        self._orchestrator = None
    
    @property
    def name(self) -> str:
        return "first_token_latency"
    
    def setup(self) -> None:
        from neurectomy import NeurectomyOrchestrator
        self._orchestrator = NeurectomyOrchestrator()
    
    def run_iteration(self) -> Dict[str, Any]:
        import time
        
        start = time.perf_counter()
        first_token = None
        
        for chunk in self._orchestrator.stream_generate(self.prompt, max_tokens=10):
            if first_token is None:
                first_token = time.perf_counter()
                ttft_ms = (first_token - start) * 1000
                break
        
        return {"ttft_ms": ttft_ms if first_token else 0}
    
    def teardown(self) -> None:
        self._orchestrator = None


class BatchInferenceBenchmark(Benchmark):
    """Benchmark batch inference throughput."""
    
    def __init__(
        self,
        prompts: Optional[list] = None,
        batch_size: int = 8,
        max_tokens: int = 50,
        config: Optional[BenchmarkConfig] = None,
    ):
        super().__init__(config)
        self.prompts = prompts or [f"Prompt {i}: Tell me about" for i in range(batch_size)]
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self._orchestrator = None
    
    @property
    def name(self) -> str:
        return f"batch_inference_{self.batch_size}x{self.max_tokens}"
    
    def setup(self) -> None:
        from neurectomy import NeurectomyOrchestrator
        self._orchestrator = NeurectomyOrchestrator()
    
    def run_iteration(self) -> Dict[str, Any]:
        total_tokens = 0
        
        for prompt in self.prompts[:self.batch_size]:
            result = self._orchestrator.generate(prompt, max_tokens=self.max_tokens)
            total_tokens += result.tokens_generated
        
        return {
            "total_tokens": total_tokens,
            "prompts_processed": self.batch_size,
        }
    
    def teardown(self) -> None:
        self._orchestrator = None


class ContextLengthBenchmark(Benchmark):
    """Benchmark performance across different context lengths."""
    
    def __init__(
        self,
        context_length: int = 1000,
        max_tokens: int = 50,
        config: Optional[BenchmarkConfig] = None,
    ):
        super().__init__(config)
        self.context_length = context_length
        self.max_tokens = max_tokens
        self._prompt = None
        self._orchestrator = None
    
    @property
    def name(self) -> str:
        return f"context_length_{self.context_length}"
    
    def setup(self) -> None:
        from neurectomy import NeurectomyOrchestrator
        self._orchestrator = NeurectomyOrchestrator()
        
        # Generate prompt of approximately context_length tokens
        base = "The quick brown fox jumps over the lazy dog. "
        repeats = max(1, self.context_length // 10)
        self._prompt = base * repeats
    
    def run_iteration(self) -> Dict[str, Any]:
        result = self._orchestrator.generate(
            self._prompt,
            max_tokens=self.max_tokens,
        )
        
        return {
            "input_tokens": result.tokens_processed,
            "output_tokens": result.tokens_generated,
        }
    
    def teardown(self) -> None:
        self._orchestrator = None
        self._prompt = None


def get_inference_benchmarks(config: Optional[BenchmarkConfig] = None) -> list:
    """Get all inference benchmarks."""
    return [
        TokenGenerationBenchmark(max_tokens=50, config=config),
        TokenGenerationBenchmark(max_tokens=100, config=config),
        TokenGenerationBenchmark(max_tokens=256, config=config),
        FirstTokenLatencyBenchmark(config=config),
        BatchInferenceBenchmark(batch_size=4, config=config),
        BatchInferenceBenchmark(batch_size=8, config=config),
        ContextLengthBenchmark(context_length=500, config=config),
        ContextLengthBenchmark(context_length=2000, config=config),
        ContextLengthBenchmark(context_length=4000, config=config),
    ]
