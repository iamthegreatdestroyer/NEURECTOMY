"""
Embeddings API Routes.

@TENSOR @SYNAPSE - Embedding generation and vector search endpoints.
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from src.models.embeddings import (
    EmbeddingRequest,
    EmbeddingResponse,
    SearchRequest,
    SearchResponse,
    EmbeddingModel,
)
from src.services.embeddings import EmbeddingService

router = APIRouter()

# Service dependency
_embedding_service: Optional[EmbeddingService] = None


async def get_embedding_service() -> EmbeddingService:
    """Dependency to get embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
        await _embedding_service.initialize()
    return _embedding_service


class ModelsResponse(BaseModel):
    """Available embedding models."""
    models: list[dict]


class IndexRequest(BaseModel):
    """Request to index a document."""
    content: str
    metadata: dict = Field(default_factory=dict)
    namespace: str = "default"


class IndexResponse(BaseModel):
    """Response after indexing."""
    id: str
    indexed: bool


class BatchIndexRequest(BaseModel):
    """Batch index request."""
    documents: list[dict]  # Each with 'content' and optional 'metadata'
    namespace: str = "default"


class BatchIndexResponse(BaseModel):
    """Batch index response."""
    indexed: int
    ids: list[str]


class DeleteRequest(BaseModel):
    """Delete documents request."""
    ids: Optional[list[str]] = None
    namespace: str = "default"
    filter: Optional[dict] = None


class DeleteResponse(BaseModel):
    """Delete response."""
    deleted: int


@router.post("/generate", response_model=EmbeddingResponse)
async def generate_embeddings(
    request: EmbeddingRequest,
    embeddings: EmbeddingService = Depends(get_embedding_service),
) -> EmbeddingResponse:
    """
    Generate embeddings for text.
    
    @TENSOR - Converts text to vector embeddings.
    
    Supports batch processing for multiple texts.
    
    Example:
        ```json
        {
            "texts": ["Hello world", "How are you?"],
            "model": "all-MiniLM-L6-v2"
        }
        ```
    """
    try:
        return await embeddings.generate_embeddings(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    embeddings: EmbeddingService = Depends(get_embedding_service),
) -> SearchResponse:
    """
    Search for similar documents.
    
    @TENSOR @VERTEX - Vector similarity search with optional filtering.
    
    Supports:
    - Semantic search via query text
    - Metadata filtering
    - Score thresholding
    
    Example:
        ```json
        {
            "query": "machine learning algorithms",
            "limit": 10,
            "score_threshold": 0.7,
            "metadata_filter": {"category": "ml"}
        }
        ```
    """
    try:
        return await embeddings.search(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index", response_model=IndexResponse)
async def index_document(
    request: IndexRequest,
    embeddings: EmbeddingService = Depends(get_embedding_service),
) -> IndexResponse:
    """
    Index a document for search.
    
    @TENSOR - Generates embedding and stores in vector database.
    """
    try:
        doc_id = await embeddings.index_document(
            content=request.content,
            metadata=request.metadata,
            namespace=request.namespace,
        )
        return IndexResponse(id=doc_id, indexed=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index/batch", response_model=BatchIndexResponse)
async def batch_index(
    request: BatchIndexRequest,
    background_tasks: BackgroundTasks,
    embeddings: EmbeddingService = Depends(get_embedding_service),
) -> BatchIndexResponse:
    """
    Batch index multiple documents.
    
    @TENSOR @VELOCITY - Efficient batch processing for large document sets.
    
    For very large batches, processing continues in background.
    """
    try:
        ids = await embeddings.batch_index(
            documents=request.documents,
            namespace=request.namespace,
        )
        return BatchIndexResponse(indexed=len(ids), ids=ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents", response_model=DeleteResponse)
async def delete_documents(
    request: DeleteRequest,
    embeddings: EmbeddingService = Depends(get_embedding_service),
) -> DeleteResponse:
    """
    Delete documents from index.
    
    Can delete by IDs or by metadata filter.
    """
    try:
        deleted = await embeddings.delete(
            ids=request.ids,
            namespace=request.namespace,
            filter=request.filter,
        )
        return DeleteResponse(deleted=deleted)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    """
    List available embedding models.
    
    Returns all supported embedding models.
    """
    # Map models to their embedding dimensions
    dimension_map = {
        EmbeddingModel.MINILM: 384,
        EmbeddingModel.MPNET: 768,
        EmbeddingModel.BGE_SMALL: 384,
        EmbeddingModel.BGE_BASE: 768,
        EmbeddingModel.BGE_LARGE: 1024,
        EmbeddingModel.E5_SMALL: 384,
        EmbeddingModel.E5_BASE: 768,
        EmbeddingModel.E5_LARGE: 1024,
        EmbeddingModel.OPENAI_SMALL: 1536,
        EmbeddingModel.OPENAI_LARGE: 3072,
        EmbeddingModel.OPENAI_ADA: 1536,
    }
    
    models = [
        {
            "name": model.value,
            "dimensions": dimension_map.get(model, 384),
            "max_tokens": 512 if "mini" in model.value.lower() else 8192,
        }
        for model in EmbeddingModel
    ]
    return ModelsResponse(models=models)


class SimilarityRequest(BaseModel):
    """Compute similarity between texts."""
    text1: str
    text2: str
    model: EmbeddingModel = EmbeddingModel.MINILM


class SimilarityResponse(BaseModel):
    """Similarity score response."""
    similarity: float
    model: str


@router.post("/similarity", response_model=SimilarityResponse)
async def compute_similarity(
    request: SimilarityRequest,
    embeddings: EmbeddingService = Depends(get_embedding_service),
) -> SimilarityResponse:
    """
    Compute cosine similarity between two texts.
    
    @TENSOR - Useful for duplicate detection and text comparison.
    """
    import numpy as np
    
    # Generate embeddings for both texts
    response = await embeddings.generate_embeddings(
        EmbeddingRequest(
            texts=[request.text1, request.text2],
            model=request.model,
        )
    )
    
    # Compute cosine similarity
    emb1 = np.array(response.embeddings[0].embedding)
    emb2 = np.array(response.embeddings[1].embedding)
    
    similarity = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    
    return SimilarityResponse(
        similarity=similarity,
        model=request.model.value,
    )
