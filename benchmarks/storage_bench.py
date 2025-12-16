"""
Storage Benchmarks
==================

Benchmarks for ΣVAULT storage performance.
"""

from typing import Dict, Any, Optional
import uuid
from .base import Benchmark, BenchmarkConfig


class RSUWriteBenchmark(Benchmark):
    """Benchmark RSU write performance."""
    
    def __init__(
        self,
        data_size: int = 1000,
        config: Optional[BenchmarkConfig] = None,
    ):
        super().__init__(config)
        self.data_size = data_size
        self._storage = None
        self._test_data = None
    
    @property
    def name(self) -> str:
        return f"rsu_write_{self.data_size}bytes"
    
    def setup(self) -> None:
        from neurectomy.core.bridges import StorageBridge
        self._storage = StorageBridge()
        self._test_data = "x" * self.data_size
    
    def run_iteration(self) -> Dict[str, Any]:
        import time
        
        rsu_id = f"bench_{uuid.uuid4().hex[:8]}"
        
        start = time.perf_counter()
        result = self._storage.store_rsu(
            rsu_id=rsu_id,
            data=self._test_data,
            metadata={"benchmark": True},
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        bytes_per_sec = (self.data_size / elapsed_ms) * 1000 if elapsed_ms > 0 else 0
        
        return {
            "write_time_ms": elapsed_ms,
            "bytes_written": self.data_size,
            "bytes_per_second": bytes_per_sec,
        }
    
    def teardown(self) -> None:
        self._storage = None


class RSUReadBenchmark(Benchmark):
    """Benchmark RSU read performance."""
    
    def __init__(
        self,
        data_size: int = 1000,
        config: Optional[BenchmarkConfig] = None,
    ):
        super().__init__(config)
        self.data_size = data_size
        self._storage = None
        self._rsu_id = None
    
    @property
    def name(self) -> str:
        return f"rsu_read_{self.data_size}bytes"
    
    def setup(self) -> None:
        from neurectomy.core.bridges import StorageBridge
        self._storage = StorageBridge()
        
        # Pre-write data to read
        self._rsu_id = f"bench_read_{uuid.uuid4().hex[:8]}"
        test_data = "y" * self.data_size
        self._storage.store_rsu(
            rsu_id=self._rsu_id,
            data=test_data,
            metadata={"benchmark": True},
        )
    
    def run_iteration(self) -> Dict[str, Any]:
        import time
        
        start = time.perf_counter()
        result = self._storage.retrieve_rsu(self._rsu_id)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        bytes_read = len(result) if result else 0
        bytes_per_sec = (bytes_read / elapsed_ms) * 1000 if elapsed_ms > 0 else 0
        
        return {
            "read_time_ms": elapsed_ms,
            "bytes_read": bytes_read,
            "bytes_per_second": bytes_per_sec,
        }
    
    def teardown(self) -> None:
        self._storage = None


class SemanticSearchBenchmark(Benchmark):
    """Benchmark semantic similarity search."""
    
    def __init__(
        self,
        num_entries: int = 100,
        config: Optional[BenchmarkConfig] = None,
    ):
        super().__init__(config)
        self.num_entries = num_entries
        self._storage = None
    
    @property
    def name(self) -> str:
        return f"semantic_search_{self.num_entries}entries"
    
    def setup(self) -> None:
        from neurectomy.core.bridges import StorageBridge
        self._storage = StorageBridge()
        
        # Pre-populate with test entries
        for i in range(self.num_entries):
            self._storage.store_rsu(
                rsu_id=f"search_bench_{i}",
                data=f"Test data entry {i} with unique content",
                metadata={"index": i},
            )
    
    def run_iteration(self) -> Dict[str, Any]:
        import time
        
        query = "Test data entry with content"
        
        start = time.perf_counter()
        results = self._storage.find_similar(query, max_results=10)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        return {
            "search_time_ms": elapsed_ms,
            "results_found": len(results) if results else 0,
        }
    
    def teardown(self) -> None:
        self._storage = None


class ManifoldNavigationBenchmark(Benchmark):
    """Benchmark 8D manifold coordinate operations."""
    
    def __init__(
        self,
        operations: int = 100,
        config: Optional[BenchmarkConfig] = None,
    ):
        super().__init__(config)
        self.operations = operations
        self._storage = None
    
    @property
    def name(self) -> str:
        return f"manifold_navigation_{self.operations}ops"
    
    def setup(self) -> None:
        from neurectomy.core.bridges import StorageBridge
        self._storage = StorageBridge()
    
    def run_iteration(self) -> Dict[str, Any]:
        import time
        import random
        
        start = time.perf_counter()
        
        for _ in range(self.operations):
            # Simulate coordinate computation
            coords = tuple(random.random() for _ in range(8))
            # Would call actual manifold navigation
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        ops_per_sec = (self.operations / elapsed_ms) * 1000 if elapsed_ms > 0 else 0
        
        return {
            "navigation_time_ms": elapsed_ms,
            "operations": self.operations,
            "ops_per_second": ops_per_sec,
        }
    
    def teardown(self) -> None:
        self._storage = None


def get_storage_benchmarks(config: Optional[BenchmarkConfig] = None) -> list:
    """Get all storage benchmarks."""
    return [
        RSUWriteBenchmark(data_size=1000, config=config),
        RSUWriteBenchmark(data_size=10000, config=config),
        RSUWriteBenchmark(data_size=100000, config=config),
        RSUReadBenchmark(data_size=1000, config=config),
        RSUReadBenchmark(data_size=10000, config=config),
        SemanticSearchBenchmark(num_entries=100, config=config),
        SemanticSearchBenchmark(num_entries=1000, config=config),
        ManifoldNavigationBenchmark(operations=100, config=config),
    ]
