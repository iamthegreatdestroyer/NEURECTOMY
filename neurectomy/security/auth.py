"""Authentication system."""

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import secrets
import hmac
from typing import Dict, Optional, Tuple


@dataclass
class APIKey:
    """API key model."""
    key_id: str
    key_hash: str
    tenant_id: str
    name: str
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None


class APIKeyManager:
    """Manages API keys for authentication."""
    
    def __init__(self):
        self._keys: Dict[str, APIKey] = {}
    
    def create_key(self, tenant_id: str, name: str) -> Tuple[str, str]:
        """Create a new API key.
        
        Returns:
            (key_id, full_key) where full_key should be stored securely
        """
        key_id = f"nk_{secrets.token_hex(8)}"
        raw_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        self._keys[key_id] = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            tenant_id=tenant_id,
            name=name,
        )
        
        # Return key_id and full key (only shown once)
        return key_id, f"{key_id}.{raw_key}"
    
    def validate_key(self, full_key: str) -> Optional[APIKey]:
        """Validate an API key.
        
        Returns:
            APIKey object if valid, None otherwise
        """
        if not full_key or "." not in full_key:
            return None
        
        try:
            key_id, raw_key = full_key.split(".", 1)
        except ValueError:
            return None
        
        api_key = self._keys.get(key_id)
        if not api_key or not api_key.is_active:
            return None
        
        # Timing-safe comparison
        expected_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        if not hmac.compare_digest(expected_hash, api_key.key_hash):
            return None
        
        # Update last used
        api_key.last_used = datetime.utcnow()
        return api_key
    
    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        if key_id in self._keys:
            self._keys[key_id].is_active = False
            return True
        return False
    
    def delete_key(self, key_id: str) -> bool:
        """Delete an API key."""
        if key_id in self._keys:
            del self._keys[key_id]
            return True
        return False
    
    def list_keys(self, tenant_id: str) -> list:
        """List all keys for a tenant."""
        return [
            key for key in self._keys.values()
            if key.tenant_id == tenant_id
        ]
    
    def get_key_info(self, key_id: str) -> Optional[APIKey]:
        """Get key information (without the secret)."""
        return self._keys.get(key_id)


# Global API key manager
api_key_manager = APIKeyManager()
