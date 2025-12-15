"""
Health Check System
===================

Comprehensive health monitoring for all components.
"""

from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a single component."""
    
    name: str
    status: HealthStatus
    message: Optional[str] = None
    latency_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    checked_at: datetime = None
    
    def __post_init__(self):
        if self.checked_at is None:
            self.checked_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "details": self.details,
            "checked_at": self.checked_at.isoformat(),
        }


@dataclass
class SystemHealth:
    """Overall system health."""
    
    status: HealthStatus
    components: List[ComponentHealth]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "components": [c.to_dict() for c in self.components],
        }


class HealthChecker:
    """
    Comprehensive health checker for the Neurectomy system.
    """
    
    def __init__(self):
        self._checks: Dict[str, Callable] = {}
        self._last_results: Dict[str, ComponentHealth] = {}
    
    def register_check(
        self,
        name: str,
        check_func: Callable[[], ComponentHealth],
    ) -> None:
        """Register a health check function."""
        self._checks[name] = check_func
    
    def check_component(self, name: str) -> ComponentHealth:
        """Run health check for a specific component."""
        if name not in self._checks:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                message="No health check registered",
            )
        
        try:
            import time
            start = time.perf_counter()
            result = self._checks[name]()
            result.latency_ms = (time.perf_counter() - start) * 1000
            self._last_results[name] = result
            return result
        except Exception as e:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
            )
    
    def check_all(self) -> SystemHealth:
        """Run all health checks."""
        components = []
        
        for name in self._checks:
            result = self.check_component(name)
            components.append(result)
        
        # Determine overall status
        if all(c.status == HealthStatus.HEALTHY for c in components):
            overall = HealthStatus.HEALTHY
        elif any(c.status == HealthStatus.UNHEALTHY for c in components):
            overall = HealthStatus.UNHEALTHY
        elif any(c.status == HealthStatus.DEGRADED for c in components):
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.UNKNOWN
        
        return SystemHealth(status=overall, components=components)
    
    def get_last_results(self) -> Dict[str, ComponentHealth]:
        """Get last health check results."""
        return self._last_results.copy()


def create_default_health_checker() -> HealthChecker:
    """Create health checker with default checks."""
    checker = HealthChecker()
    
    # Inference health check
    def check_inference() -> ComponentHealth:
        try:
            from ..core.bridges import InferenceBridge
            bridge = InferenceBridge()
            
            if bridge.is_ready():
                return ComponentHealth(
                    name="inference",
                    status=HealthStatus.HEALTHY,
                    message="Inference engine ready",
                    details=bridge.get_model_info(),
                )
            else:
                return ComponentHealth(
                    name="inference",
                    status=HealthStatus.DEGRADED,
                    message="Inference engine using mock",
                )
        except Exception as e:
            return ComponentHealth(
                name="inference",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )
    
    # Compression health check
    def check_compression() -> ComponentHealth:
        try:
            from ..core.bridges import CompressionBridge
            bridge = CompressionBridge()
            
            if bridge.is_ready():
                return ComponentHealth(
                    name="compression",
                    status=HealthStatus.HEALTHY,
                    message="ΣLANG compression ready",
                    details={"ratio": bridge.get_compression_ratio()},
                )
            else:
                return ComponentHealth(
                    name="compression",
                    status=HealthStatus.DEGRADED,
                    message="Compression not available",
                )
        except Exception as e:
            return ComponentHealth(
                name="compression",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )
    
    # Storage health check
    def check_storage() -> ComponentHealth:
        try:
            from ..core.bridges import StorageBridge
            bridge = StorageBridge()
            
            if bridge.is_ready():
                return ComponentHealth(
                    name="storage",
                    status=HealthStatus.HEALTHY,
                    message="ΣVAULT storage ready",
                    details=bridge.get_statistics(),
                )
            else:
                return ComponentHealth(
                    name="storage",
                    status=HealthStatus.DEGRADED,
                    message="Storage not available",
                )
        except Exception as e:
            return ComponentHealth(
                name="storage",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )
    
    # Elite Collective health check
    def check_collective() -> ComponentHealth:
        try:
            from ..elite import EliteCollective
            collective = EliteCollective()
            
            agent_count = len(collective.list_agents())
            team_count = len(collective.list_teams())
            
            if agent_count == 40 and team_count == 5:
                return ComponentHealth(
                    name="elite_collective",
                    status=HealthStatus.HEALTHY,
                    message=f"{agent_count} agents in {team_count} teams",
                    details={"agents": agent_count, "teams": team_count},
                )
            else:
                return ComponentHealth(
                    name="elite_collective",
                    status=HealthStatus.DEGRADED,
                    message=f"Only {agent_count} agents available",
                    details={"agents": agent_count, "teams": team_count},
                )
        except Exception as e:
            return ComponentHealth(
                name="elite_collective",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )
    
    # Register all checks
    checker.register_check("inference", check_inference)
    checker.register_check("compression", check_compression)
    checker.register_check("storage", check_storage)
    checker.register_check("elite_collective", check_collective)
    
    return checker
