"""
Embedding Service for vector generation and similarity search.

@TENSOR @STREAM - Real-time embedding generation pipeline.

Features:
- Multiple embedding model support
- Batch processing for efficiency
- Automatic model loading/caching
- Integration with pgvector
"""

import asyncio
import time
from typing import Optional

import numpy as np
import structlog
import torch
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.models.embeddings import (
    EmbeddingModel,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from src.db.vector import VectorStore
from src.db.postgres import get_db_session
from src.db.redis import get_redis

logger = structlog.get_logger()


class EmbeddingService:
    """
    High-performance embedding service.
    
    @TENSOR @VELOCITY - Optimized embedding generation with:
    - GPU acceleration
    - Model caching
    - Batch processing
    - Redis caching for frequently used texts
    """
    
    def __init__(self):
        self._models: dict[str, SentenceTransformer] = {}
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._vector_store: Optional[VectorStore] = None
        
        logger.info(f"Embedding service using device: {self._device}")
    
    async def initialize(self) -> None:
        """Initialize default embedding model and vector store."""
        # Load default model
        await self._load_model(settings.embedding_model)
        
        # Initialize vector store
        self._vector_store = VectorStore(
            table_name="embeddings",
            dimension=settings.embedding_dim,
        )
        
        async with get_db_session() as session:
            await self._vector_store.initialize(session)
        
        logger.info("✅ Embedding service initialized")
    
    async def _load_model(self, model_name: str) -> SentenceTransformer:
        """
        Load and cache embedding model.
        
        @TENSOR - Model loading with GPU optimization.
        """
        if model_name not in self._models:
            logger.info(f"Loading embedding model: {model_name}")
            
            # Run in thread pool to not block event loop
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                None,
                lambda: SentenceTransformer(model_name, device=self._device),
            )
            
            # Optimize for inference
            model.eval()
            if self._device == "cuda":
                model.half()  # FP16 for faster inference
            
            self._models[model_name] = model
            logger.info(f"✅ Model loaded: {model_name}")
        
        return self._models[model_name]
    
    async def generate_embeddings(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse:
        """
        Generate embeddings for input texts.
        
        @TENSOR @STREAM - Batch embedding generation.
        """
        start_time = time.time()
        
        model_name = request.model.value if request.model else settings.embedding_model
        model = await self._load_model(model_name)
        
        # Check Redis cache for existing embeddings
        redis = await get_redis()
        embeddings = []
        texts_to_embed = []
        text_indices = []
        
        for i, text in enumerate(request.texts):
            cache_key = f"{model_name}:{hash(text)}"
            cached = await redis.get_cached_embedding(cache_key)
            
            if cached:
                embeddings.append(EmbeddingVector(
                    index=i,
                    embedding=cached,
                    text_length=len(text),
                ))
            else:
                texts_to_embed.append(text)
                text_indices.append(i)
        
        # Generate embeddings for uncached texts
        if texts_to_embed:
            loop = asyncio.get_event_loop()
            new_embeddings = await loop.run_in_executor(
                None,
                lambda: model.encode(
                    texts_to_embed,
                    batch_size=request.batch_size,
                    normalize_embeddings=request.normalize,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                ),
            )
            
            # Cache and add to results
            for idx, (text_idx, text) in enumerate(zip(text_indices, texts_to_embed)):
                emb_list = new_embeddings[idx].tolist()
                
                # Cache embedding
                cache_key = f"{model_name}:{hash(text)}"
                await redis.cache_embedding(cache_key, emb_list)
                
                embeddings.append(EmbeddingVector(
                    index=text_idx,
                    embedding=emb_list,
                    text_length=len(text),
                ))
        
        # Sort by original index
        embeddings.sort(key=lambda x: x.index)
        
        # Calculate tokens (approximate)
        total_tokens = sum(len(text.split()) for text in request.texts)
        
        return EmbeddingResponse(
            embeddings=embeddings,
            model=model_name,
            dimension=len(embeddings[0].embedding) if embeddings else 0,
            latency_ms=(time.time() - start_time) * 1000,
            tokens_processed=total_tokens,
        )
    
    async def search(self, request: SearchRequest) -> SearchResponse:
        """
        Perform semantic similarity search.
        
        @LINGUA @VELOCITY - Fast semantic search with optional hybrid mode.
        """
        start_time = time.time()
        
        # Generate query embedding
        model = await self._load_model(settings.embedding_model)
        
        loop = asyncio.get_event_loop()
        query_embedding = await loop.run_in_executor(
            None,
            lambda: model.encode(
                request.query,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).tolist(),
        )
        
        # Perform search
        async with get_db_session() as session:
            if request.hybrid:
                results = await self._vector_store.hybrid_search(
                    session=session,
                    query_text=request.query,
                    query_embedding=query_embedding,
                    limit=request.limit,
                    semantic_weight=request.semantic_weight,
                    keyword_weight=request.keyword_weight,
                )
                search_type = "hybrid"
            else:
                results = await self._vector_store.search(
                    session=session,
                    query_embedding=query_embedding,
                    limit=request.limit,
                    threshold=request.threshold,
                    metadata_filter=request.metadata_filter,
                )
                search_type = "semantic"
        
        # Convert to response format
        search_results = [
            SearchResult(
                id=r.id,
                content=r.content,
                similarity=r.similarity,
                distance=r.distance,
                metadata=r.metadata if request.include_metadata else None,
            )
            for r in results
        ]
        
        return SearchResponse(
            results=search_results,
            query=request.query,
            total_results=len(search_results),
            collection=request.collection,
            search_type=search_type,
            latency_ms=(time.time() - start_time) * 1000,
        )
    
    async def index_document(
        self,
        content: str,
        metadata: Optional[dict] = None,
        collection: Optional[str] = None,
    ) -> str:
        """
        Index a document with its embedding.
        
        @TENSOR @VERTEX - Document indexing for RAG.
        """
        # Generate embedding
        model = await self._load_model(settings.embedding_model)
        
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: model.encode(
                content,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).tolist(),
        )
        
        # Store in vector database
        async with get_db_session() as session:
            doc_id = await self._vector_store.insert(
                session=session,
                content=content,
                embedding=embedding,
                metadata=metadata,
            )
        
        logger.info(f"Indexed document: {doc_id}")
        return doc_id
    
    async def batch_index(
        self,
        documents: list[tuple[str, dict]],
    ) -> list[str]:
        """
        Batch index multiple documents.
        
        @VELOCITY - Optimized batch indexing.
        """
        model = await self._load_model(settings.embedding_model)
        
        contents = [doc[0] for doc in documents]
        metadata_list = [doc[1] for doc in documents]
        
        # Generate all embeddings in batch
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(
                contents,
                batch_size=32,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ),
        )
        
        # Store all in database
        async with get_db_session() as session:
            docs_with_embeddings = [
                (content, emb.tolist(), meta)
                for content, emb, meta in zip(contents, embeddings, metadata_list)
            ]
            doc_ids = await self._vector_store.batch_insert(
                session=session,
                documents=docs_with_embeddings,
            )
        
        logger.info(f"Batch indexed {len(doc_ids)} documents")
        return doc_ids
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the index."""
        async with get_db_session() as session:
            return await self._vector_store.delete(session, doc_id)
    
    def get_model_info(self, model_name: str) -> dict:
        """Get information about an embedding model."""
        model_configs = {
            EmbeddingModel.MINILM.value: {"dimension": 384, "max_length": 256},
            EmbeddingModel.MPNET.value: {"dimension": 768, "max_length": 384},
            EmbeddingModel.BGE_SMALL.value: {"dimension": 384, "max_length": 512},
            EmbeddingModel.BGE_BASE.value: {"dimension": 768, "max_length": 512},
            EmbeddingModel.BGE_LARGE.value: {"dimension": 1024, "max_length": 512},
            EmbeddingModel.E5_SMALL.value: {"dimension": 384, "max_length": 512},
            EmbeddingModel.E5_BASE.value: {"dimension": 768, "max_length": 512},
            EmbeddingModel.E5_LARGE.value: {"dimension": 1024, "max_length": 512},
        }
        return model_configs.get(model_name, {"dimension": 384, "max_length": 512})
