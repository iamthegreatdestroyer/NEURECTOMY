"""Rate limiting."""

import time
from collections import defaultdict
from typing import Tuple, Dict


class TokenBucket:
    """Token bucket algorithm for rate limiting."""
    
    def __init__(self, rate: float, capacity: int):
        """Initialize token bucket.
        
        Args:
            rate: Tokens per second
            capacity: Maximum tokens
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last_update = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens.
        
        Returns:
            True if tokens were available, False otherwise
        """
        now = time.time()
        elapsed = now - self.last_update
        
        # Refill tokens
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_update = now
        
        # Try to consume
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def get_available_tokens(self) -> float:
        """Get current available tokens."""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        return self.tokens


class RateLimiter:
    """Rate limiter using token bucket."""
    
    def __init__(self, requests_per_minute: int = 60):
        """Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute
        """
        self.rate = requests_per_minute / 60.0  # Convert to requests per second
        self._buckets: Dict[str, TokenBucket] = {}
    
    def check(self, key: str, tokens: int = 1) -> Tuple[bool, float]:
        """Check if request is allowed.
        
        Args:
            key: Identifier for rate limit bucket
            tokens: Number of tokens to consume
            
        Returns:
            (allowed: bool, retry_after: float in seconds)
        """
        if key not in self._buckets:
            self._buckets[key] = TokenBucket(self.rate, 10)
        
        bucket = self._buckets[key]
        allowed = bucket.consume(tokens)
        
        if allowed:
            return True, 0.0
        else:
            # Calculate retry after
            retry_after = (tokens - bucket.get_available_tokens()) / self.rate
            return False, retry_after
    
    def reset(self, key: str) -> None:
        """Reset rate limiter for a key."""
        if key in self._buckets:
            del self._buckets[key]
    
    def get_status(self, key: str) -> Dict[str, float]:
        """Get rate limiter status."""
        if key not in self._buckets:
            self._buckets[key] = TokenBucket(self.rate, 10)
        
        bucket = self._buckets[key]
        return {
            "available_tokens": bucket.get_available_tokens(),
            "capacity": bucket.capacity,
            "rate": bucket.rate,
        }


# Global rate limiters for different endpoints
rate_limiters = {
    "default": RateLimiter(60),
    "strict": RateLimiter(10),
    "generous": RateLimiter(1000),
}
