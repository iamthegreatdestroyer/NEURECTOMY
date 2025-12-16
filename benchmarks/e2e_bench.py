"""
End-to-End Benchmarks
=====================

Full pipeline benchmarks for the complete system.
"""

from typing import Dict, Any, Optional
from .base import Benchmark, BenchmarkConfig


class FullPipelineBenchmark(Benchmark):
    """Benchmark complete request pipeline."""
    
    def __init__(
        self,
        prompt: str = "Explain the theory of relativity in simple terms.",
        max_tokens: int = 100,
        config: Optional[BenchmarkConfig] = None,
    ):
        super().__init__(config)
        self.prompt = prompt
        self.max_tokens = max_tokens
        self._orchestrator = None
    
    @property
    def name(self) -> str:
        return f"full_pipeline_{self.max_tokens}tok"
    
    def setup(self) -> None:
        from neurectomy import NeurectomyOrchestrator
        self._orchestrator = NeurectomyOrchestrator()
    
    def run_iteration(self) -> Dict[str, Any]:
        result = self._orchestrator.generate(
            self.prompt,
            max_tokens=self.max_tokens,
        )
        
        return {
            "total_latency_ms": result.execution_time_ms,
            "tokens_generated": result.tokens_generated,
            "compression_ratio": result.compression_ratio,
            "cache_hit": result.cache_hit if hasattr(result, 'cache_hit') else False,
        }
    
    def teardown(self) -> None:
        self._orchestrator = None


class CacheHitRateBenchmark(Benchmark):
    """Benchmark cache hit rate with repeated queries."""
    
    def __init__(
        self,
        num_unique_prompts: int = 10,
        repetitions: int = 5,
        config: Optional[BenchmarkConfig] = None,
    ):
        super().__init__(config)
        self.num_unique_prompts = num_unique_prompts
        self.repetitions = repetitions
        self._orchestrator = None
        self._prompts = None
    
    @property
    def name(self) -> str:
        return f"cache_hit_rate_{self.num_unique_prompts}x{self.repetitions}"
    
    def setup(self) -> None:
        from neurectomy import NeurectomyOrchestrator
        self._orchestrator = NeurectomyOrchestrator()
        
        self._prompts = [
            f"Unique prompt number {i} for cache testing"
            for i in range(self.num_unique_prompts)
        ]
    
    def run_iteration(self) -> Dict[str, Any]:
        cache_hits = 0
        cache_misses = 0
        total_requests = 0
        
        # Run each prompt multiple times
        for _ in range(self.repetitions):
            for prompt in self._prompts:
                result = self._orchestrator.generate(prompt, max_tokens=20)
                total_requests += 1
                
                if hasattr(result, 'cache_hit') and result.cache_hit:
                    cache_hits += 1
                else:
                    cache_misses += 1
        
        hit_rate = cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "total_requests": total_requests,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "hit_rate": hit_rate,
        }
    
    def teardown(self) -> None:
        self._orchestrator = None


class ConversationBenchmark(Benchmark):
    """Benchmark multi-turn conversation performance."""
    
    def __init__(
        self,
        num_turns: int = 5,
        config: Optional[BenchmarkConfig] = None,
    ):
        super().__init__(config)
        self.num_turns = num_turns
        self._orchestrator = None
    
    @property
    def name(self) -> str:
        return f"conversation_{self.num_turns}turns"
    
    def setup(self) -> None:
        from neurectomy import NeurectomyOrchestrator
        self._orchestrator = NeurectomyOrchestrator()
    
    def run_iteration(self) -> Dict[str, Any]:
        import time
        import uuid
        
        conversation_id = f"bench_conv_{uuid.uuid4().hex[:8]}"
        
        turns = [
            "Hello, how are you?",
            "Tell me about yourself.",
            "What can you help me with?",
            "Can you write code?",
            "Thank you for your help!",
        ]
        
        start = time.perf_counter()
        total_tokens = 0
        
        for i in range(min(self.num_turns, len(turns))):
            result = self._orchestrator.generate(
                turns[i],
                max_tokens=50,
                conversation_id=conversation_id,
            )
            total_tokens += result.tokens_generated
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        return {
            "conversation_time_ms": elapsed_ms,
            "turns_completed": self.num_turns,
            "total_tokens": total_tokens,
            "avg_turn_time_ms": elapsed_ms / self.num_turns,
        }
    
    def teardown(self) -> None:
        self._orchestrator = None


class ConcurrentRequestsBenchmark(Benchmark):
    """Benchmark concurrent request handling."""
    
    def __init__(
        self,
        num_concurrent: int = 5,
        config: Optional[BenchmarkConfig] = None,
    ):
        super().__init__(config)
        self.num_concurrent = num_concurrent
        self._orchestrator = None
    
    @property
    def name(self) -> str:
        return f"concurrent_requests_{self.num_concurrent}"
    
    def setup(self) -> None:
        from neurectomy import NeurectomyOrchestrator
        self._orchestrator = NeurectomyOrchestrator()
    
    def run_iteration(self) -> Dict[str, Any]:
        import time
        import concurrent.futures
        
        prompts = [
            f"Concurrent request {i}: Tell me something interesting"
            for i in range(self.num_concurrent)
        ]
        
        def process_prompt(prompt):
            return self._orchestrator.generate(prompt, max_tokens=30)
        
        start = time.perf_counter()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_concurrent) as executor:
            results = list(executor.map(process_prompt, prompts))
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        successful = sum(1 for r in results if r and r.tokens_generated > 0)
        total_tokens = sum(r.tokens_generated for r in results if r)
        
        return {
            "total_time_ms": elapsed_ms,
            "concurrent_requests": self.num_concurrent,
            "successful_requests": successful,
            "total_tokens": total_tokens,
            "requests_per_second": (self.num_concurrent / elapsed_ms) * 1000 if elapsed_ms > 0 else 0,
        }
    
    def teardown(self) -> None:
        self._orchestrator = None


def get_e2e_benchmarks(config: Optional[BenchmarkConfig] = None) -> list:
    """Get all end-to-end benchmarks."""
    return [
        FullPipelineBenchmark(max_tokens=50, config=config),
        FullPipelineBenchmark(max_tokens=100, config=config),
        FullPipelineBenchmark(max_tokens=256, config=config),
        CacheHitRateBenchmark(num_unique_prompts=10, repetitions=3, config=config),
        ConversationBenchmark(num_turns=3, config=config),
        ConversationBenchmark(num_turns=5, config=config),
        ConcurrentRequestsBenchmark(num_concurrent=3, config=config),
        ConcurrentRequestsBenchmark(num_concurrent=5, config=config),
    ]
