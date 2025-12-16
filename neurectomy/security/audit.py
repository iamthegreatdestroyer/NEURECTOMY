"""Security audit logging."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
from typing import Dict, Any, Optional, List


class AuditAction(Enum):
    """Audit action types."""
    AUTH_SUCCESS = "auth.success"
    AUTH_FAILURE = "auth.failure"
    API_REQUEST = "api.request"
    KEY_CREATED = "key.created"
    KEY_REVOKED = "key.revoked"
    QUOTA_EXCEEDED = "quota.exceeded"
    CONFIG_CHANGED = "config.changed"
    DATA_ACCESSED = "data.accessed"
    DATA_MODIFIED = "data.modified"
    SECURITY_ALERT = "security.alert"


@dataclass
class AuditEvent:
    """Single audit event."""
    action: AuditAction
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action": self.action.value,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "ip_address": self.ip_address,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class AuditLogger:
    """Logger for security audit events."""
    
    def __init__(self, log_file: str = "audit.log", memory_limit: int = 10000):
        """Initialize audit logger.
        
        Args:
            log_file: Path to audit log file
            memory_limit: Maximum events to keep in memory
        """
        self.log_file = Path(log_file)
        self.memory_limit = memory_limit
        self._events: List[AuditEvent] = []
    
    def log(self, event: AuditEvent) -> None:
        """Log an audit event."""
        # Add to in-memory buffer
        self._events.append(event)
        
        # Keep memory limit
        if len(self._events) > self.memory_limit:
            self._events = self._events[-self.memory_limit:]
        
        # Write to file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "a") as f:
            f.write(event.to_json() + "\n")
    
    def log_auth(
        self,
        success: bool,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log authentication event."""
        self.log(AuditEvent(
            action=AuditAction.AUTH_SUCCESS if success else AuditAction.AUTH_FAILURE,
            tenant_id=tenant_id,
            user_id=user_id,
            ip_address=ip_address,
            details=details or {},
        ))
    
    def log_api_request(
        self,
        tenant_id: str,
        endpoint: str,
        method: str,
        ip_address: Optional[str] = None,
        status_code: int = 200,
    ) -> None:
        """Log API request."""
        self.log(AuditEvent(
            action=AuditAction.API_REQUEST,
            tenant_id=tenant_id,
            ip_address=ip_address,
            details={
                "endpoint": endpoint,
                "method": method,
                "status_code": status_code,
            },
        ))
    
    def log_quota_exceeded(
        self,
        tenant_id: str,
        quota_type: str,
        limit: int,
        current: int,
    ) -> None:
        """Log quota exceeded event."""
        self.log(AuditEvent(
            action=AuditAction.QUOTA_EXCEEDED,
            tenant_id=tenant_id,
            details={
                "quota_type": quota_type,
                "limit": limit,
                "current": current,
            },
        ))
    
    def get_events(self, tenant_id: Optional[str] = None) -> List[AuditEvent]:
        """Get audit events."""
        if tenant_id:
            return [e for e in self._events if e.tenant_id == tenant_id]
        return self._events.copy()
    
    def get_recent_events(self, limit: int = 100, tenant_id: Optional[str] = None) -> List[AuditEvent]:
        """Get recent audit events."""
        events = self.get_events(tenant_id)
        return events[-limit:]


# Global audit logger
audit_logger = AuditLogger()
