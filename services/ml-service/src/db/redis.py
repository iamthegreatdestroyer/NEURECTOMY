"""
Redis client for caching and message queues.

@VELOCITY @STREAM - High-performance caching for ML inference.
"""

from typing import Any, Optional
import json

import redis.asyncio as redis
import structlog

from src.config import settings

logger = structlog.get_logger()


class RedisClient:
    """
    Async Redis client with ML-specific utilities.
    
    Features:
    - Model response caching
    - Embedding cache with TTL
    - Rate limiting for API calls
    - Distributed locks for training jobs
    """
    
    def __init__(self):
        self._client: Optional[redis.Redis] = None
    
    async def connect(self) -> None:
        """Establish Redis connection."""
        self._client = await redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        await self._client.ping()
        logger.info("✅ Redis connected")
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
    
    @property
    def client(self) -> redis.Redis:
        """Get Redis client instance."""
        if not self._client:
            raise RuntimeError("Redis not connected. Call connect() first.")
        return self._client
    
    # =========================================================================
    # Caching Methods
    # =========================================================================
    
    async def cache_embedding(
        self,
        key: str,
        embedding: list[float],
        ttl: int = 86400,  # 24 hours
    ) -> None:
        """Cache an embedding vector."""
        await self.client.setex(
            f"emb:{key}",
            ttl,
            json.dumps(embedding),
        )
    
    async def get_cached_embedding(self, key: str) -> Optional[list[float]]:
        """Retrieve cached embedding."""
        data = await self.client.get(f"emb:{key}")
        return json.loads(data) if data else None
    
    async def cache_llm_response(
        self,
        prompt_hash: str,
        response: str,
        model: str,
        ttl: int = 3600,  # 1 hour
    ) -> None:
        """Cache LLM response for deduplication."""
        await self.client.setex(
            f"llm:{model}:{prompt_hash}",
            ttl,
            response,
        )
    
    async def get_cached_llm_response(
        self,
        prompt_hash: str,
        model: str,
    ) -> Optional[str]:
        """Retrieve cached LLM response."""
        return await self.client.get(f"llm:{model}:{prompt_hash}")
    
    # =========================================================================
    # Rate Limiting
    # =========================================================================
    
    async def check_rate_limit(
        self,
        key: str,
        max_requests: int,
        window_seconds: int,
    ) -> tuple[bool, int]:
        """
        Check and increment rate limit counter.
        
        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        pipe = self.client.pipeline()
        pipe.incr(f"rate:{key}")
        pipe.expire(f"rate:{key}", window_seconds)
        results = await pipe.execute()
        
        current_count = results[0]
        remaining = max(0, max_requests - current_count)
        is_allowed = current_count <= max_requests
        
        return is_allowed, remaining
    
    # =========================================================================
    # Distributed Locks
    # =========================================================================
    
    async def acquire_lock(
        self,
        lock_name: str,
        ttl: int = 300,
    ) -> bool:
        """Acquire a distributed lock for training jobs."""
        return await self.client.set(
            f"lock:{lock_name}",
            "locked",
            nx=True,
            ex=ttl,
        )
    
    async def release_lock(self, lock_name: str) -> None:
        """Release a distributed lock."""
        await self.client.delete(f"lock:{lock_name}")
    
    # =========================================================================
    # Queue Operations (for training jobs)
    # =========================================================================
    
    async def enqueue_job(self, queue: str, job_data: dict) -> None:
        """Add a job to the training queue."""
        await self.client.rpush(f"queue:{queue}", json.dumps(job_data))
    
    async def dequeue_job(self, queue: str, timeout: int = 0) -> Optional[dict]:
        """Pop a job from the training queue."""
        result = await self.client.blpop(f"queue:{queue}", timeout=timeout)
        if result:
            return json.loads(result[1])
        return None


# Global Redis client instance
_redis_client: Optional[RedisClient] = None


async def get_redis() -> RedisClient:
    """Get or create Redis client."""
    global _redis_client
    if _redis_client is None:
        _redis_client = RedisClient()
        await _redis_client.connect()
    return _redis_client
