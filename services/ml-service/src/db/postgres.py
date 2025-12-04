"""
PostgreSQL database connection with pgvector support.

@TENSOR @VERTEX - Vector database configuration for embeddings storage.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy import text
import structlog

from src.config import settings

logger = structlog.get_logger()

# Create async engine
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

# Session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Base class for ORM models
Base = declarative_base()


async def init_db() -> None:
    """
    Initialize database connections and extensions.
    
    Ensures pgvector extension is enabled for embedding storage.
    """
    async with engine.begin() as conn:
        # Enable pgvector extension
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        logger.info("✅ pgvector extension enabled")
        
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
        logger.info("✅ Database tables created")


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for database sessions.
    
    Usage:
        async with get_db_session() as session:
            result = await session.execute(query)
    """
    session = AsyncSessionLocal()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def close_db() -> None:
    """Close database engine."""
    await engine.dispose()
    logger.info("Database connections closed")
