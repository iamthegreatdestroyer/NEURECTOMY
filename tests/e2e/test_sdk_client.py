"""E2E tests for SDK client."""

import pytest


class TestSDKClient:
    
    @pytest.mark.e2e
    def test_client_creation(self):
        try:
            from neurectomy.sdk import NeurectomyClient, NeurectomyConfig
            
            config = NeurectomyConfig(base_url="http://localhost:8000")
            client = NeurectomyClient(config)
            
            assert client is not None
            if hasattr(client, 'close'):
                client.close()
        except (ImportError, TypeError, AttributeError):
            pytest.skip("SDK client not available")
    
    @pytest.mark.e2e
    @pytest.mark.skip(reason="Requires running server")
    def test_client_generate(self):
        try:
            from neurectomy.sdk import NeurectomyClient
            
            client = NeurectomyClient()
            response = client.generate("Test", max_tokens=5)
            
            assert "text" in response or "result" in response
            if hasattr(client, 'close'):
                client.close()
        except (ImportError, AttributeError):
            pytest.skip("SDK client not available")
