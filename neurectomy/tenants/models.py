"""Multi-tenancy models."""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid


class TenantTier(Enum):
    """Tenant service tiers."""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


@dataclass
class TenantQuota:
    """Quota limits for a tenant."""
    max_requests_per_day: int = 1000
    max_tokens_per_request: int = 4096
    max_concurrent_requests: int = 5
    max_storage_mb: int = 1000


@dataclass
class Tenant:
    """Tenant model for multi-tenancy."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    tier: TenantTier = TenantTier.FREE
    quota: TenantQuota = field(default_factory=TenantQuota)
    requests_today: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_within_quota(self) -> bool:
        """Check if tenant is within daily request quota."""
        return self.requests_today < self.quota.max_requests_per_day
    
    def record_usage(self, tokens: int = 1) -> None:
        """Record usage for this tenant."""
        self.requests_today += 1
    
    def reset_daily_quota(self) -> None:
        """Reset daily request counter."""
        self.requests_today = 0
    
    def get_remaining_quota(self) -> int:
        """Get remaining requests for today."""
        return max(0, self.quota.max_requests_per_day - self.requests_today)


class TenantManager:
    """Manages multiple tenants."""
    
    def __init__(self):
        self._tenants: Dict[str, Tenant] = {}
    
    def create_tenant(self, name: str, tier: TenantTier = TenantTier.FREE) -> Tenant:
        """Create a new tenant."""
        tenant = Tenant(name=name, tier=tier)
        
        # Set tier-specific quotas
        if tier == TenantTier.STARTER:
            tenant.quota = TenantQuota(
                max_requests_per_day=10000,
                max_tokens_per_request=8192,
                max_concurrent_requests=10,
                max_storage_mb=5000,
            )
        elif tier == TenantTier.PROFESSIONAL:
            tenant.quota = TenantQuota(
                max_requests_per_day=100000,
                max_tokens_per_request=32768,
                max_concurrent_requests=50,
                max_storage_mb=50000,
            )
        elif tier == TenantTier.ENTERPRISE:
            tenant.quota = TenantQuota(
                max_requests_per_day=1000000,
                max_tokens_per_request=131072,
                max_concurrent_requests=500,
                max_storage_mb=1000000,
            )
        
        self._tenants[tenant.id] = tenant
        return tenant
    
    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get a tenant by ID."""
        return self._tenants.get(tenant_id)
    
    def list_tenants(self) -> List[Tenant]:
        """List all tenants."""
        return list(self._tenants.values())
    
    def delete_tenant(self, tenant_id: str) -> bool:
        """Delete a tenant."""
        if tenant_id in self._tenants:
            del self._tenants[tenant_id]
            return True
        return False
    
    def upgrade_tenant(self, tenant_id: str, new_tier: TenantTier) -> Optional[Tenant]:
        """Upgrade a tenant to a new tier."""
        tenant = self._tenants.get(tenant_id)
        if tenant:
            tenant.tier = new_tier
            # Re-apply tier-specific quotas
            return self.create_tenant(tenant.name, new_tier)
        return None
