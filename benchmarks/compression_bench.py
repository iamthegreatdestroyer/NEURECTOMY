"""
Compression Benchmarks
======================

Benchmarks for ΣLANG compression performance.
"""

from typing import Dict, Any, Optional
from .base import Benchmark, BenchmarkConfig


class CompressionRatioBenchmark(Benchmark):
    """Benchmark compression ratio."""
    
    def __init__(
        self,
        text_size: int = 1000,
        compression_level: int = 2,
        config: Optional[BenchmarkConfig] = None,
    ):
        super().__init__(config)
        self.text_size = text_size
        self.compression_level = compression_level
        self._text = None
        self._compressor = None
    
    @property
    def name(self) -> str:
        return f"compression_ratio_{self.text_size}chars_level{self.compression_level}"
    
    def setup(self) -> None:
        from neurectomy.core.bridges import CompressionBridge
        self._compressor = CompressionBridge()
        
        # Generate test text
        base = "This is a sample text for compression benchmarking. It contains various words and patterns that are typical in natural language processing tasks. "
        repeats = max(1, self.text_size // len(base))
        self._text = base * repeats
    
    def run_iteration(self) -> Dict[str, Any]:
        original_size = len(self._text)
        
        compressed = self._compressor.compress(self._text)
        compressed_size = len(compressed) if compressed else original_size
        
        ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        return {
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": ratio,
        }
    
    def teardown(self) -> None:
        self._compressor = None
        self._text = None


class CompressionThroughputBenchmark(Benchmark):
    """Benchmark compression throughput (chars/sec)."""
    
    def __init__(
        self,
        text_size: int = 10000,
        config: Optional[BenchmarkConfig] = None,
    ):
        super().__init__(config)
        self.text_size = text_size
        self._text = None
        self._compressor = None
    
    @property
    def name(self) -> str:
        return f"compression_throughput_{self.text_size}chars"
    
    def setup(self) -> None:
        from neurectomy.core.bridges import CompressionBridge
        self._compressor = CompressionBridge()
        
        base = "Sample text for throughput testing with various patterns and repetitions. "
        repeats = max(1, self.text_size // len(base))
        self._text = base * repeats
    
    def run_iteration(self) -> Dict[str, Any]:
        import time
        
        start = time.perf_counter()
        compressed = self._compressor.compress(self._text)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        chars_per_sec = (len(self._text) / elapsed_ms) * 1000 if elapsed_ms > 0 else 0
        
        return {
            "chars_processed": len(self._text),
            "chars_per_second": chars_per_sec,
            "compression_time_ms": elapsed_ms,
        }
    
    def teardown(self) -> None:
        self._compressor = None
        self._text = None


class DecompressionBenchmark(Benchmark):
    """Benchmark decompression speed."""
    
    def __init__(
        self,
        text_size: int = 5000,
        config: Optional[BenchmarkConfig] = None,
    ):
        super().__init__(config)
        self.text_size = text_size
        self._compressed = None
        self._original_size = 0
        self._compressor = None
    
    @property
    def name(self) -> str:
        return f"decompression_{self.text_size}chars"
    
    def setup(self) -> None:
        from neurectomy.core.bridges import CompressionBridge
        self._compressor = CompressionBridge()
        
        base = "Decompression benchmark text with patterns. "
        repeats = max(1, self.text_size // len(base))
        text = base * repeats
        self._original_size = len(text)
        
        self._compressed = self._compressor.compress(text)
    
    def run_iteration(self) -> Dict[str, Any]:
        import time
        
        start = time.perf_counter()
        decompressed = self._compressor.decompress(self._compressed)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        return {
            "decompression_time_ms": elapsed_ms,
            "output_size": len(decompressed) if decompressed else 0,
        }
    
    def teardown(self) -> None:
        self._compressor = None
        self._compressed = None


class RoundTripBenchmark(Benchmark):
    """Benchmark full compression + decompression cycle."""
    
    def __init__(
        self,
        text_size: int = 5000,
        config: Optional[BenchmarkConfig] = None,
    ):
        super().__init__(config)
        self.text_size = text_size
        self._text = None
        self._compressor = None
    
    @property
    def name(self) -> str:
        return f"compression_roundtrip_{self.text_size}chars"
    
    def setup(self) -> None:
        from neurectomy.core.bridges import CompressionBridge
        self._compressor = CompressionBridge()
        
        base = "Round trip compression benchmark text. "
        repeats = max(1, self.text_size // len(base))
        self._text = base * repeats
    
    def run_iteration(self) -> Dict[str, Any]:
        import time
        
        # Compress
        start_compress = time.perf_counter()
        compressed = self._compressor.compress(self._text)
        compress_ms = (time.perf_counter() - start_compress) * 1000
        
        # Decompress
        start_decompress = time.perf_counter()
        decompressed = self._compressor.decompress(compressed)
        decompress_ms = (time.perf_counter() - start_decompress) * 1000
        
        # Verify
        is_valid = decompressed == self._text if decompressed else False
        
        return {
            "compress_time_ms": compress_ms,
            "decompress_time_ms": decompress_ms,
            "total_time_ms": compress_ms + decompress_ms,
            "is_valid": is_valid,
        }
    
    def teardown(self) -> None:
        self._compressor = None
        self._text = None


def get_compression_benchmarks(config: Optional[BenchmarkConfig] = None) -> list:
    """Get all compression benchmarks."""
    return [
        CompressionRatioBenchmark(text_size=1000, config=config),
        CompressionRatioBenchmark(text_size=5000, config=config),
        CompressionRatioBenchmark(text_size=10000, config=config),
        CompressionThroughputBenchmark(text_size=5000, config=config),
        CompressionThroughputBenchmark(text_size=20000, config=config),
        DecompressionBenchmark(text_size=5000, config=config),
        RoundTripBenchmark(text_size=5000, config=config),
    ]
