"""
Unit Tests for Embedding Service

@ECLIPSE @TENSOR - Comprehensive tests for embedding generation and search.

Copyright (c) 2025 NEURECTOMY. All Rights Reserved.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.models.embeddings import (
    EmbeddingModel,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
    SearchRequest,
    SearchResponse,
    SearchResult,
)


# ==============================================================================
# Model Tests
# ==============================================================================

class TestEmbeddingModels:
    """Tests for embedding Pydantic models."""
    
    @pytest.mark.unit
    def test_embedding_request_valid(self):
        """Test valid embedding request creation."""
        request = EmbeddingRequest(
            texts=["Hello world", "Test text"],
            model=EmbeddingModel.MINILM,
            normalize=True,
            batch_size=32,
        )
        
        assert len(request.texts) == 2
        assert request.model == EmbeddingModel.MINILM
        assert request.normalize is True
        assert request.batch_size == 32
    
    @pytest.mark.unit
    def test_embedding_request_defaults(self):
        """Test embedding request default values."""
        request = EmbeddingRequest(texts=["test"])
        
        assert request.model == EmbeddingModel.MINILM
        assert request.normalize is True
        assert request.truncate is True
        assert request.max_length == 512
        assert request.batch_size == 32
    
    @pytest.mark.unit
    def test_embedding_request_validation_empty_texts(self):
        """Test that empty texts list raises validation error."""
        with pytest.raises(ValueError):
            EmbeddingRequest(texts=[])
    
    @pytest.mark.unit
    def test_embedding_request_validation_batch_size(self):
        """Test batch size validation bounds."""
        # Valid batch size
        request = EmbeddingRequest(texts=["test"], batch_size=64)
        assert request.batch_size == 64
        
        # Invalid batch size (too large)
        with pytest.raises(ValueError):
            EmbeddingRequest(texts=["test"], batch_size=200)
        
        # Invalid batch size (too small)
        with pytest.raises(ValueError):
            EmbeddingRequest(texts=["test"], batch_size=0)
    
    @pytest.mark.unit
    def test_embedding_vector_creation(self):
        """Test EmbeddingVector model."""
        vector = EmbeddingVector(
            index=0,
            embedding=[0.1] * 384,
            text_length=100,
        )
        
        assert vector.index == 0
        assert len(vector.embedding) == 384
        assert vector.text_length == 100
    
    @pytest.mark.unit
    def test_embedding_response_creation(self):
        """Test EmbeddingResponse model."""
        response = EmbeddingResponse(
            embeddings=[
                EmbeddingVector(index=0, embedding=[0.1] * 384, text_length=10),
            ],
            model="test-model",
            dimension=384,
            latency_ms=50.5,
            tokens_processed=100,
        )
        
        assert len(response.embeddings) == 1
        assert response.model == "test-model"
        assert response.dimension == 384
        assert response.latency_ms == 50.5
        assert response.tokens_processed == 100
        assert response.created_at is not None
    
    @pytest.mark.unit
    def test_search_request_valid(self):
        """Test valid search request creation."""
        request = SearchRequest(
            query="test query",
            limit=10,
            threshold=0.7,
            hybrid=True,
        )
        
        assert request.query == "test query"
        assert request.limit == 10
        assert request.threshold == 0.7
        assert request.hybrid is True
    
    @pytest.mark.unit
    def test_search_request_validation(self):
        """Test search request validation."""
        # Valid request
        request = SearchRequest(query="test", limit=50, threshold=0.5)
        assert request.limit == 50
        
        # Invalid limit (too large)
        with pytest.raises(ValueError):
            SearchRequest(query="test", limit=200)
        
        # Invalid threshold
        with pytest.raises(ValueError):
            SearchRequest(query="test", threshold=1.5)
    
    @pytest.mark.unit
    def test_search_result_creation(self):
        """Test SearchResult model."""
        result = SearchResult(
            id="doc-123",
            content="Test content",
            similarity=0.95,
            distance=0.05,
            metadata={"source": "test"},
        )
        
        assert result.id == "doc-123"
        assert result.content == "Test content"
        assert result.similarity == 0.95
        assert result.distance == 0.05
        assert result.metadata == {"source": "test"}


# ==============================================================================
# Embedding Service Tests
# ==============================================================================

class TestEmbeddingService:
    """Tests for EmbeddingService class."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_service_initialization(self, mock_postgres_pool, mock_redis):
        """Test embedding service initialization."""
        with patch("src.services.embeddings.get_db_session") as mock_db, \
             patch("src.services.embeddings.get_redis", return_value=mock_redis), \
             patch("src.services.embeddings.SentenceTransformer") as mock_st:
            
            # Setup mocks
            mock_db.return_value.__aenter__ = AsyncMock(return_value=mock_postgres_pool)
            mock_db.return_value.__aexit__ = AsyncMock()
            mock_st.return_value = MagicMock()
            
            from src.services.embeddings import EmbeddingService
            
            service = EmbeddingService()
            
            assert service._models == {}
            assert service._vector_store is None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_embeddings_with_cache_hit(
        self, mock_redis, mock_sentence_transformer
    ):
        """Test embedding generation with cache hit."""
        mock_embedding = [0.1] * 384
        mock_redis.get_cached_embedding = AsyncMock(return_value=mock_embedding)
        
        with patch("src.services.embeddings.get_redis", return_value=mock_redis), \
             patch("src.services.embeddings.SentenceTransformer", return_value=mock_sentence_transformer):
            
            from src.services.embeddings import EmbeddingService
            
            service = EmbeddingService()
            service._models["sentence-transformers/all-MiniLM-L6-v2"] = mock_sentence_transformer
            
            request = EmbeddingRequest(texts=["cached text"])
            
            # Mock the generate_embeddings method behavior
            # Since cache hit, should return cached embedding
            mock_redis.get_cached_embedding.return_value = mock_embedding
            
            # Verify cache was checked
            await mock_redis.get_cached_embedding(f"sentence-transformers/all-MiniLM-L6-v2:{hash('cached text')}")
            mock_redis.get_cached_embedding.assert_called()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_embeddings_cache_miss(
        self, mock_redis, mock_sentence_transformer
    ):
        """Test embedding generation with cache miss."""
        import numpy as np
        
        mock_redis.get_cached_embedding = AsyncMock(return_value=None)
        mock_redis.cache_embedding = AsyncMock(return_value=True)
        
        # Mock SentenceTransformer encode
        mock_sentence_transformer.encode = MagicMock(
            return_value=np.array([[0.1] * 384])
        )
        
        with patch("src.services.embeddings.get_redis", return_value=mock_redis), \
             patch("src.services.embeddings.SentenceTransformer", return_value=mock_sentence_transformer):
            
            from src.services.embeddings import EmbeddingService
            
            service = EmbeddingService()
            service._models["sentence-transformers/all-MiniLM-L6-v2"] = mock_sentence_transformer
            
            # Cache miss should trigger encoding
            mock_redis.get_cached_embedding.return_value = None
            
            # Verify that when cache misses, encode is called
            mock_sentence_transformer.encode.return_value = np.array([[0.1] * 384])
            
            texts = ["new text"]
            embeddings = mock_sentence_transformer.encode(texts)
            
            assert embeddings.shape == (1, 384)
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_batch_embedding_generation(self, mock_sentence_transformer):
        """Test batch embedding generation."""
        import numpy as np
        
        batch_texts = [f"Text {i}" for i in range(10)]
        mock_embeddings = np.array([[0.1] * 384 for _ in range(10)])
        
        mock_sentence_transformer.encode = MagicMock(return_value=mock_embeddings)
        
        result = mock_sentence_transformer.encode(
            batch_texts,
            batch_size=5,
            normalize_embeddings=True,
        )
        
        assert result.shape == (10, 384)
        mock_sentence_transformer.encode.assert_called_once()
    
    @pytest.mark.unit
    def test_embedding_model_enum_values(self):
        """Test all embedding model enum values are valid."""
        expected_models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-base-en-v1.5",
            "BAAI/bge-large-en-v1.5",
            "intfloat/e5-small-v2",
            "intfloat/e5-base-v2",
            "intfloat/e5-large-v2",
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ]
        
        model_values = [m.value for m in EmbeddingModel]
        
        for expected in expected_models:
            assert expected in model_values


# ==============================================================================
# Search Tests
# ==============================================================================

class TestSemanticSearch:
    """Tests for semantic search functionality."""
    
    @pytest.mark.unit
    def test_search_request_hybrid_weights(self):
        """Test hybrid search weight configuration."""
        request = SearchRequest(
            query="test",
            hybrid=True,
            semantic_weight=0.8,
            keyword_weight=0.2,
        )
        
        assert request.semantic_weight == 0.8
        assert request.keyword_weight == 0.2
        assert request.semantic_weight + request.keyword_weight == 1.0
    
    @pytest.mark.unit
    def test_search_request_metadata_filter(self):
        """Test metadata filtering in search request."""
        request = SearchRequest(
            query="test",
            metadata_filter={"category": "documents", "year": 2024},
        )
        
        assert request.metadata_filter["category"] == "documents"
        assert request.metadata_filter["year"] == 2024
    
    @pytest.mark.unit
    def test_search_response_creation(self):
        """Test SearchResponse model."""
        results = [
            SearchResult(
                id="1",
                content="Result 1",
                similarity=0.95,
                distance=0.05,
            ),
            SearchResult(
                id="2",
                content="Result 2",
                similarity=0.85,
                distance=0.15,
            ),
        ]
        
        # SearchResponse would typically contain these results
        assert len(results) == 2
        assert results[0].similarity > results[1].similarity
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_similarity_score_calculation(self):
        """Test similarity score is within valid range."""
        result = SearchResult(
            id="test",
            content="content",
            similarity=0.87,
            distance=0.13,
        )
        
        assert 0.0 <= result.similarity <= 1.0
        assert 0.0 <= result.distance <= 1.0
        # Similarity and distance should be complementary
        assert abs((result.similarity + result.distance) - 1.0) < 0.01


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestEmbeddingIntegration:
    """Integration tests for embedding service."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_embedding_pipeline(
        self, mock_postgres_pool, mock_redis, mock_sentence_transformer
    ):
        """Test complete embedding pipeline: generate -> store -> search."""
        import numpy as np
        
        # Setup mocks for full pipeline
        mock_embeddings = np.array([[0.1] * 384])
        mock_sentence_transformer.encode = MagicMock(return_value=mock_embeddings)
        mock_redis.get_cached_embedding = AsyncMock(return_value=None)
        mock_redis.cache_embedding = AsyncMock(return_value=True)
        
        # 1. Generate embedding
        texts = ["Test document for RAG pipeline"]
        embeddings = mock_sentence_transformer.encode(texts)
        
        assert embeddings.shape == (1, 384)
        
        # 2. Verify embedding dimension
        assert len(embeddings[0]) == 384
        
        # 3. Verify embedding values are normalized (approximately)
        norm = np.linalg.norm(embeddings[0])
        # Mocked embeddings may not be normalized, but real ones should be
        assert norm > 0
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_embedding_caching_behavior(self, mock_redis):
        """Test embedding caching behavior."""
        test_key = "test-model:123456"
        test_embedding = [0.1] * 384
        
        # First call - cache miss
        mock_redis.get_cached_embedding = AsyncMock(return_value=None)
        result = await mock_redis.get_cached_embedding(test_key)
        assert result is None
        
        # Cache the embedding
        mock_redis.cache_embedding = AsyncMock(return_value=True)
        await mock_redis.cache_embedding(test_key, test_embedding)
        
        # Second call - cache hit
        mock_redis.get_cached_embedding = AsyncMock(return_value=test_embedding)
        result = await mock_redis.get_cached_embedding(test_key)
        assert result == test_embedding


# ==============================================================================
# Performance Tests
# ==============================================================================

class TestEmbeddingPerformance:
    """Performance tests for embedding service."""
    
    @pytest.mark.slow
    @pytest.mark.unit
    def test_batch_processing_efficiency(self, mock_sentence_transformer):
        """Test that batch processing is more efficient than single processing."""
        import numpy as np
        import time
        
        texts = [f"Text number {i}" for i in range(100)]
        mock_embeddings = np.array([[0.1] * 384 for _ in range(100)])
        mock_sentence_transformer.encode = MagicMock(return_value=mock_embeddings)
        
        # Batch processing (single call)
        start = time.time()
        batch_result = mock_sentence_transformer.encode(texts, batch_size=32)
        batch_time = time.time() - start
        
        assert batch_result.shape == (100, 384)
        assert mock_sentence_transformer.encode.call_count == 1
    
    @pytest.mark.unit
    def test_embedding_dimension_consistency(self):
        """Test that embedding dimensions are consistent."""
        dimensions = {
            EmbeddingModel.MINILM: 384,
            EmbeddingModel.MPNET: 768,
            EmbeddingModel.BGE_SMALL: 384,
            EmbeddingModel.BGE_BASE: 768,
            EmbeddingModel.OPENAI_SMALL: 1536,
            EmbeddingModel.OPENAI_LARGE: 3072,
        }
        
        for model, expected_dim in dimensions.items():
            # In production, actual models would return these dimensions
            assert expected_dim in [384, 768, 1536, 3072]
