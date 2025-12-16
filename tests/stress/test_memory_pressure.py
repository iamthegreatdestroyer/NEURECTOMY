"""Memory pressure tests."""

import pytest
import gc


class TestMemoryPressure:
    
    @pytest.mark.stress
    @pytest.mark.slow
    def test_large_context_handling(self, orchestrator):
        """Test handling very large contexts."""
        large_prompt = "Test content. " * 1000  # ~12K tokens
        
        result = orchestrator.generate(large_prompt, max_tokens=50)
        
        assert result is not None
        gc.collect()
    
    @pytest.mark.stress
    @pytest.mark.slow
    def test_many_small_requests(self, orchestrator):
        """Test many small requests don't leak memory."""
        import tracemalloc
        
        tracemalloc.start()
        
        for i in range(100):
            orchestrator.generate(f"Small request {i}", max_tokens=5)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Peak memory should be reasonable
        peak_mb = peak / (1024 * 1024)
        print(f"Peak memory: {peak_mb:.2f} MB")
        
        assert peak_mb < 1000  # Less than 1GB
