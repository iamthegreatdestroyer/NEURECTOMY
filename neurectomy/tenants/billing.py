"""Usage tracking and billing."""

from dataclasses import dataclass, field
from datetime import datetime, date
from collections import defaultdict
from typing import Dict, List


@dataclass
class UsageRecord:
    """Single usage record."""
    tenant_id: str
    timestamp: datetime
    tokens_in: int
    tokens_out: int
    latency_ms: float
    endpoint: str = ""


class UsageTracker:
    """Tracks usage for billing."""
    
    def __init__(self):
        self._records: List[UsageRecord] = []
        self._daily: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    def record(
        self,
        tenant_id: str,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float,
        endpoint: str = "",
    ) -> None:
        """Record a usage event."""
        record = UsageRecord(
            tenant_id=tenant_id,
            timestamp=datetime.utcnow(),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            endpoint=endpoint,
        )
        self._records.append(record)
        
        # Track daily usage
        today = date.today().isoformat()
        self._daily[tenant_id][today] += tokens_in + tokens_out
    
    def get_usage(self, tenant_id: str) -> Dict[str, int]:
        """Get daily usage summary."""
        return dict(self._daily.get(tenant_id, {}))
    
    def get_usage_today(self, tenant_id: str) -> int:
        """Get total tokens used today."""
        today = date.today().isoformat()
        return self._daily.get(tenant_id, {}).get(today, 0)
    
    def get_daily_breakdown(self, tenant_id: str) -> Dict[str, Dict[str, int]]:
        """Get detailed daily breakdown."""
        breakdown = defaultdict(lambda: {"tokens": 0, "requests": 0})
        
        for record in self._records:
            if record.tenant_id == tenant_id:
                day = record.timestamp.date().isoformat()
                breakdown[day]["tokens"] += record.tokens_in + record.tokens_out
                breakdown[day]["requests"] += 1
        
        return dict(breakdown)
    
    def get_records(self, tenant_id: str) -> List[UsageRecord]:
        """Get all usage records for a tenant."""
        return [r for r in self._records if r.tenant_id == tenant_id]
    
    def estimate_cost(self, tenant_id: str, cost_per_1k_tokens: float = 0.01) -> float:
        """Estimate cost based on usage."""
        total_tokens = self.get_usage_today(tenant_id)
        return (total_tokens / 1000) * cost_per_1k_tokens


# Global usage tracker
usage_tracker = UsageTracker()
