"""Unit tests for inference."""

import pytest


class TestInferenceEngine:
    
    @pytest.mark.unit
    def test_engine_initialization(self, orchestrator):
        assert orchestrator is not None
    
    @pytest.mark.unit
    def test_generate_returns_result(self, orchestrator):
        result = orchestrator.generate("Test", max_tokens=5)
        assert result is not None
        assert hasattr(result, 'generated_text')
    
    @pytest.mark.unit
    def test_generate_respects_max_tokens(self, orchestrator):
        result = orchestrator.generate("Test", max_tokens=10)
        assert result.tokens_generated <= 10
    
    @pytest.mark.unit
    def test_empty_prompt_handling(self, orchestrator):
        result = orchestrator.generate("", max_tokens=5)
        assert result is not None


class TestTokenizer:
    
    @pytest.mark.unit
    def test_encode_decode_roundtrip(self):
        """Test tokenizer encode/decode roundtrip."""
        try:
            from neurectomy.core.tokenizer import Tokenizer
            tokenizer = Tokenizer()
            text = "Hello world"
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            assert decoded == text
        except (ImportError, AttributeError):
            # Skip if tokenizer not available
            pytest.skip("Tokenizer not available")
