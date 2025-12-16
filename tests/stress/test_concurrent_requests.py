"""Stress tests for concurrent requests."""

import pytest
import concurrent.futures
import time


class TestConcurrentRequests:
    
    @pytest.mark.stress
    @pytest.mark.slow
    def test_concurrent_generation(self, orchestrator):
        """Test handling multiple concurrent requests."""
        num_requests = 10
        prompts = [f"Concurrent request {i}" for i in range(num_requests)]
        
        def generate(prompt):
            return orchestrator.generate(prompt, max_tokens=20)
        
        start = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            results = list(ex.map(generate, prompts))
        
        elapsed = time.time() - start
        
        assert len(results) == num_requests
        assert all(r is not None for r in results)
        print(f"Completed {num_requests} requests in {elapsed:.2f}s")
    
    @pytest.mark.stress
    @pytest.mark.slow
    def test_sustained_load(self, orchestrator):
        """Test sustained load over time."""
        duration_seconds = 5
        requests_completed = 0
        
        start = time.time()
        
        while time.time() - start < duration_seconds:
            result = orchestrator.generate("Load test", max_tokens=5)
            if result:
                requests_completed += 1
        
        rps = requests_completed / duration_seconds
        print(f"Sustained {rps:.2f} requests/second")
        
        assert requests_completed > 0
