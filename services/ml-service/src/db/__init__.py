"""
Database layer for ML Service.

Provides connections to:
- PostgreSQL (with pgvector for embeddings)
- TimescaleDB (for time-series metrics)
- Redis (for caching and queues)
- Neo4j (for agent knowledge graphs)
"""

from src.db.postgres import init_db, get_db_session, AsyncSessionLocal
from src.db.redis import get_redis, RedisClient
from src.db.vector import VectorStore

__all__ = [
    "init_db",
    "get_db_session", 
    "AsyncSessionLocal",
    "get_redis",
    "RedisClient",
    "VectorStore",
]
