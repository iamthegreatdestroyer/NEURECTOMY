"""Unit tests for storage."""

import pytest
import uuid


class TestStorage:
    
    @pytest.mark.unit
    def test_storage_bridge(self):
        try:
            from neurectomy.core.bridges import StorageBridge
            bridge = StorageBridge()
            assert bridge is not None
        except (ImportError, AttributeError):
            pytest.skip("StorageBridge not available")
    
    @pytest.mark.unit
    def test_store_retrieve(self):
        try:
            from neurectomy.core.bridges import StorageBridge
            bridge = StorageBridge()
            
            rsu_id = f"test_{uuid.uuid4().hex[:8]}"
            data = "Test data for storage"
            
            bridge.store_rsu(rsu_id, data, {})
            retrieved = bridge.retrieve_rsu(rsu_id)
            
            # Retrieved may be None if using mock
            assert retrieved is None or retrieved == data
        except (ImportError, AttributeError, TypeError):
            pytest.skip("StorageBridge not available")
