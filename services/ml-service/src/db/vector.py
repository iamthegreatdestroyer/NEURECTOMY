"""
Vector store abstraction for pgvector operations.

@VERTEX @PRISM - pgvector Configuration & Optimization for embeddings.
"""

from typing import Any, Optional
from dataclasses import dataclass
from enum import Enum

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np
import structlog

from src.config import settings

logger = structlog.get_logger()


class IndexType(Enum):
    """Vector index types supported by pgvector."""
    IVFFLAT = "ivfflat"
    HNSW = "hnsw"


class DistanceMetric(Enum):
    """Distance metrics for vector similarity."""
    L2 = "vector_l2_ops"           # Euclidean distance
    COSINE = "vector_cosine_ops"   # Cosine similarity
    IP = "vector_ip_ops"           # Inner product


@dataclass
class SearchResult:
    """Result from vector similarity search."""
    id: str
    content: str
    metadata: dict
    distance: float
    similarity: float


class VectorStore:
    """
    High-performance vector store using pgvector.
    
    Optimized for:
    - High-dimensional embeddings (768-4096 dimensions)
    - HNSW indexing for fast approximate nearest neighbor search
    - Hybrid search combining semantic + keyword matching
    
    @VERTEX @VELOCITY - Performance-optimized vector operations.
    """
    
    def __init__(
        self,
        table_name: str = "embeddings",
        dimension: int = 384,
        index_type: IndexType = IndexType.HNSW,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
    ):
        self.table_name = table_name
        self.dimension = dimension
        self.index_type = index_type
        self.distance_metric = distance_metric
    
    async def initialize(self, session: AsyncSession) -> None:
        """
        Initialize vector store table and indexes.
        
        Creates optimized HNSW index for fast similarity search.
        """
        # Create table if not exists
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            content TEXT NOT NULL,
            embedding vector({self.dimension}) NOT NULL,
            metadata JSONB DEFAULT '{{}}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            
            -- Full-text search column
            content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
        );
        """
        await session.execute(text(create_table_sql))
        
        # Create HNSW index for vector similarity
        if self.index_type == IndexType.HNSW:
            index_sql = f"""
            CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_hnsw_idx
            ON {self.table_name} 
            USING hnsw (embedding {self.distance_metric.value})
            WITH (m = 16, ef_construction = 64);
            """
        else:
            index_sql = f"""
            CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_ivfflat_idx
            ON {self.table_name}
            USING ivfflat (embedding {self.distance_metric.value})
            WITH (lists = 100);
            """
        
        await session.execute(text(index_sql))
        
        # Create GIN index for full-text search
        gin_sql = f"""
        CREATE INDEX IF NOT EXISTS {self.table_name}_content_gin_idx
        ON {self.table_name} USING GIN (content_tsv);
        """
        await session.execute(text(gin_sql))
        
        # Create index on metadata for filtering
        metadata_sql = f"""
        CREATE INDEX IF NOT EXISTS {self.table_name}_metadata_gin_idx
        ON {self.table_name} USING GIN (metadata);
        """
        await session.execute(text(metadata_sql))
        
        await session.commit()
        logger.info(f"✅ Vector store '{self.table_name}' initialized")
    
    async def insert(
        self,
        session: AsyncSession,
        content: str,
        embedding: list[float],
        metadata: Optional[dict] = None,
    ) -> str:
        """Insert a document with its embedding."""
        import json
        
        sql = f"""
        INSERT INTO {self.table_name} (content, embedding, metadata)
        VALUES (:content, :embedding, :metadata)
        RETURNING id;
        """
        
        result = await session.execute(
            text(sql),
            {
                "content": content,
                "embedding": f"[{','.join(map(str, embedding))}]",
                "metadata": json.dumps(metadata or {}),
            },
        )
        
        row = result.fetchone()
        return str(row[0])
    
    async def batch_insert(
        self,
        session: AsyncSession,
        documents: list[tuple[str, list[float], dict]],
    ) -> list[str]:
        """
        Batch insert documents with embeddings.
        
        @VELOCITY - Optimized for bulk operations.
        """
        import json
        
        ids = []
        for content, embedding, metadata in documents:
            doc_id = await self.insert(session, content, embedding, metadata)
            ids.append(doc_id)
        
        await session.commit()
        return ids
    
    async def search(
        self,
        session: AsyncSession,
        query_embedding: list[float],
        limit: int = 10,
        threshold: float = 0.7,
        metadata_filter: Optional[dict] = None,
    ) -> list[SearchResult]:
        """
        Semantic similarity search using vector embeddings.
        
        @LINGUA @VELOCITY - Optimized semantic search with filtering.
        """
        import json
        
        # Build metadata filter clause
        filter_clause = ""
        params: dict[str, Any] = {
            "embedding": f"[{','.join(map(str, query_embedding))}]",
            "limit": limit,
        }
        
        if metadata_filter:
            filter_clause = "AND metadata @> :metadata_filter"
            params["metadata_filter"] = json.dumps(metadata_filter)
        
        # Use cosine distance for similarity
        sql = f"""
        SELECT 
            id,
            content,
            metadata,
            embedding <=> :embedding::vector AS distance,
            1 - (embedding <=> :embedding::vector) AS similarity
        FROM {self.table_name}
        WHERE 1 - (embedding <=> :embedding::vector) >= :threshold
        {filter_clause}
        ORDER BY embedding <=> :embedding::vector
        LIMIT :limit;
        """
        
        params["threshold"] = threshold
        
        result = await session.execute(text(sql), params)
        rows = result.fetchall()
        
        return [
            SearchResult(
                id=str(row[0]),
                content=row[1],
                metadata=row[2],
                distance=row[3],
                similarity=row[4],
            )
            for row in rows
        ]
    
    async def hybrid_search(
        self,
        session: AsyncSession,
        query_text: str,
        query_embedding: list[float],
        limit: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> list[SearchResult]:
        """
        Hybrid search combining semantic similarity and keyword matching.
        
        @LINGUA @TENSOR - RAG-optimized retrieval with hybrid search.
        """
        sql = f"""
        WITH semantic_results AS (
            SELECT 
                id,
                content,
                metadata,
                1 - (embedding <=> :embedding::vector) AS semantic_score
            FROM {self.table_name}
            ORDER BY embedding <=> :embedding::vector
            LIMIT :limit * 2
        ),
        keyword_results AS (
            SELECT 
                id,
                content,
                metadata,
                ts_rank(content_tsv, plainto_tsquery('english', :query)) AS keyword_score
            FROM {self.table_name}
            WHERE content_tsv @@ plainto_tsquery('english', :query)
            ORDER BY ts_rank(content_tsv, plainto_tsquery('english', :query)) DESC
            LIMIT :limit * 2
        )
        SELECT DISTINCT
            COALESCE(s.id, k.id) AS id,
            COALESCE(s.content, k.content) AS content,
            COALESCE(s.metadata, k.metadata) AS metadata,
            COALESCE(s.semantic_score, 0) * :semantic_weight + 
            COALESCE(k.keyword_score, 0) * :keyword_weight AS combined_score
        FROM semantic_results s
        FULL OUTER JOIN keyword_results k ON s.id = k.id
        ORDER BY combined_score DESC
        LIMIT :limit;
        """
        
        result = await session.execute(
            text(sql),
            {
                "embedding": f"[{','.join(map(str, query_embedding))}]",
                "query": query_text,
                "limit": limit,
                "semantic_weight": semantic_weight,
                "keyword_weight": keyword_weight,
            },
        )
        rows = result.fetchall()
        
        return [
            SearchResult(
                id=str(row[0]),
                content=row[1],
                metadata=row[2],
                distance=1 - row[3],
                similarity=row[3],
            )
            for row in rows
        ]
    
    async def delete(
        self,
        session: AsyncSession,
        doc_id: str,
    ) -> bool:
        """Delete a document by ID."""
        sql = f"DELETE FROM {self.table_name} WHERE id = :id RETURNING id;"
        result = await session.execute(text(sql), {"id": doc_id})
        deleted = result.fetchone()
        await session.commit()
        return deleted is not None
    
    async def update_embedding(
        self,
        session: AsyncSession,
        doc_id: str,
        new_embedding: list[float],
    ) -> bool:
        """Update embedding for an existing document."""
        sql = f"""
        UPDATE {self.table_name}
        SET embedding = :embedding, updated_at = NOW()
        WHERE id = :id
        RETURNING id;
        """
        result = await session.execute(
            text(sql),
            {
                "id": doc_id,
                "embedding": f"[{','.join(map(str, new_embedding))}]",
            },
        )
        updated = result.fetchone()
        await session.commit()
        return updated is not None
