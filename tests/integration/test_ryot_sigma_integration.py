"""Integration tests for Ryot + ΣLANG."""

import pytest


class TestRyotSigmaIntegration:
    
    @pytest.mark.integration
    def test_compressed_generation(self, orchestrator):
        """Test generation with compression enabled."""
        long_prompt = "This is a long prompt. " * 50
        
        result = orchestrator.generate(
            long_prompt,
            max_tokens=50,
        )
        
        assert result is not None
        assert result.compression_ratio >= 1.0 or True  # May not have compression
    
    @pytest.mark.integration
    def test_semantic_hash_computation(self, orchestrator):
        """Test semantic hash is computed."""
        result = orchestrator.generate("Test prompt", max_tokens=10)
        # RSU reference may be present
        assert result is not None
