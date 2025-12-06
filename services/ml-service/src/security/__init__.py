"""
Security Module for ML Service

@CIPHER @FORTRESS - Authentication, authorization, and security middleware.

Copyright (c) 2025 NEURECTOMY. All Rights Reserved.

Features:
- JWT-based authentication
- API key management
- Rate limiting (Redis-backed for distributed deployments)
- Request validation
- Audit logging
- Security headers
- Argon2id password hashing (OWASP 2024 compliant)
"""

import hashlib
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Set
from enum import Enum
from functools import wraps
import structlog
from collections import defaultdict
import os

from pydantic import BaseModel, Field, EmailStr
from fastapi import HTTPException, Request, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

# @CIPHER - Prefer Argon2id over bcrypt (OWASP 2024 recommendation)
try:
    from argon2 import PasswordHasher, exceptions as argon2_exceptions
    from argon2.profiles import RFC_9106_LOW_MEMORY
    ARGON2_AVAILABLE = True
except ImportError:
    ARGON2_AVAILABLE = False

try:
    from passlib.context import CryptContext
    PASSLIB_AVAILABLE = True
except ImportError:
    PASSLIB_AVAILABLE = False

# @FORTRESS - Redis for distributed token revocation and rate limiting
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


logger = structlog.get_logger()


# ==============================================================================
# Security Configuration
# ==============================================================================

class SecurityConfig(BaseModel):
    """
    Security configuration.
    
    @CIPHER - OWASP 2024 compliant settings:
    - Password min length: 12 characters (up from 8)
    - Argon2id parameters: 64MB memory, 3 iterations, 4 threads
    - Unique character requirement: 6 minimum
    """
    secret_key: str = Field(default_factory=lambda: os.environ.get("JWT_SECRET", secrets.token_urlsafe(64)))
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 15  # Reduced from 30 for security
    refresh_token_expire_days: int = 1     # Reduced from 7 for security
    api_key_prefix: str = "neurectomy_"
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60
    # @CIPHER - Password policy aligned with Rust service
    password_min_length: int = 12          # Increased from 8
    password_max_length: int = 128
    password_min_unique_chars: int = 6     # New requirement
    password_require_special: bool = True  # New requirement
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 15
    # Redis configuration for distributed deployments
    redis_url: str = Field(default_factory=lambda: os.environ.get("REDIS_URL", "redis://localhost:6379"))


# Default config
SECURITY_CONFIG = SecurityConfig()


# ==============================================================================
# Security Event Logger (SIEM-Ready)
# ==============================================================================

class SecurityEventType(str, Enum):
    """
    Security event types for SIEM integration.
    
    @SENTRY - Categories aligned with MITRE ATT&CK framework and
    common SIEM solutions (Splunk, Elasticsearch, Azure Sentinel).
    """
    # Authentication events
    AUTH_LOGIN_SUCCESS = "auth.login.success"
    AUTH_LOGIN_FAILURE = "auth.login.failure"
    AUTH_LOGOUT = "auth.logout"
    AUTH_TOKEN_ISSUED = "auth.token.issued"
    AUTH_TOKEN_REFRESHED = "auth.token.refreshed"
    AUTH_TOKEN_REVOKED = "auth.token.revoked"
    AUTH_TOKEN_INVALID = "auth.token.invalid"
    AUTH_TOKEN_EXPIRED = "auth.token.expired"
    
    # Authorization events
    AUTHZ_ACCESS_DENIED = "authz.access.denied"
    AUTHZ_PERMISSION_DENIED = "authz.permission.denied"
    AUTHZ_ROLE_CHANGED = "authz.role.changed"
    
    # Account events
    ACCOUNT_CREATED = "account.created"
    ACCOUNT_UPDATED = "account.updated"
    ACCOUNT_DELETED = "account.deleted"
    ACCOUNT_LOCKED = "account.locked"
    ACCOUNT_UNLOCKED = "account.unlocked"
    ACCOUNT_PASSWORD_CHANGED = "account.password.changed"
    ACCOUNT_PASSWORD_RESET = "account.password.reset"
    
    # API Key events
    API_KEY_CREATED = "api_key.created"
    API_KEY_REVOKED = "api_key.revoked"
    API_KEY_USED = "api_key.used"
    API_KEY_INVALID = "api_key.invalid"
    
    # Rate limiting events
    RATE_LIMIT_EXCEEDED = "rate_limit.exceeded"
    RATE_LIMIT_WARNING = "rate_limit.warning"
    
    # Security threat events
    THREAT_SQL_INJECTION = "threat.sql_injection"
    THREAT_XSS = "threat.xss"
    THREAT_PATH_TRAVERSAL = "threat.path_traversal"
    THREAT_COMMAND_INJECTION = "threat.command_injection"
    THREAT_BRUTE_FORCE = "threat.brute_force"
    THREAT_SUSPICIOUS_IP = "threat.suspicious_ip"
    
    # Session events
    SESSION_CREATED = "session.created"
    SESSION_DESTROYED = "session.destroyed"
    SESSION_HIJACK_ATTEMPT = "session.hijack_attempt"
    
    # Configuration events
    CONFIG_CHANGED = "config.changed"
    CONFIG_SECURITY_SETTING = "config.security_setting"
    
    # Audit events
    AUDIT_DATA_ACCESS = "audit.data.access"
    AUDIT_DATA_EXPORT = "audit.data.export"
    AUDIT_ADMIN_ACTION = "audit.admin.action"


class SecurityEventSeverity(str, Enum):
    """SIEM-compatible severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SecurityEvent(BaseModel):
    """
    SIEM-ready security event format.
    
    Compatible with:
    - Common Event Format (CEF)
    - Elastic Common Schema (ECS)
    - Azure Sentinel
    - Splunk CIM
    """
    # Event identification
    event_id: str = Field(default_factory=lambda: secrets.token_urlsafe(16))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_type: SecurityEventType
    severity: SecurityEventSeverity
    
    # Source information
    source_ip: Optional[str] = None
    source_port: Optional[int] = None
    user_agent: Optional[str] = None
    
    # User context
    user_id: Optional[str] = None
    username: Optional[str] = None
    session_id: Optional[str] = None
    
    # Request context
    request_id: Optional[str] = None
    request_method: Optional[str] = None
    request_path: Optional[str] = None
    
    # Event details
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    
    # Outcome
    outcome: str = "unknown"  # success, failure, unknown
    
    # Classification
    category: str = "security"
    action: Optional[str] = None
    
    class Config:
        use_enum_values = True


class SecurityEventLogger:
    """
    SIEM-ready security event logger.
    
    @SENTRY - Provides structured security event logging compatible with
    enterprise SIEM solutions. Events can be forwarded to:
    - Elasticsearch/OpenSearch
    - Splunk
    - Azure Sentinel
    - AWS CloudWatch
    - Datadog
    
    Usage:
        security_logger = SecurityEventLogger()
        security_logger.log_auth_failure(
            request=request,
            username="user@example.com",
            reason="Invalid password"
        )
    """
    
    def __init__(self):
        self._logger = structlog.get_logger("security.events")
    
    def _extract_request_context(self, request: Optional[Request]) -> Dict[str, Any]:
        """Extract security-relevant context from a request."""
        if not request:
            return {}
        
        # Get client IP (handle proxies)
        client_ip = request.client.host if request.client else None
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        
        return {
            "source_ip": client_ip,
            "user_agent": request.headers.get("User-Agent"),
            "request_id": getattr(request.state, "request_id", None),
            "request_method": request.method,
            "request_path": str(request.url.path),
        }
    
    def log_event(
        self,
        event_type: SecurityEventType,
        severity: SecurityEventSeverity,
        message: str,
        request: Optional[Request] = None,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        outcome: str = "unknown",
        **details
    ) -> SecurityEvent:
        """
        Log a security event.
        
        Args:
            event_type: Type of security event
            severity: Severity level
            message: Human-readable message
            request: Optional FastAPI request for context
            user_id: Optional user identifier
            username: Optional username
            outcome: Event outcome (success, failure, unknown)
            **details: Additional event details
        
        Returns:
            The logged SecurityEvent
        """
        request_context = self._extract_request_context(request)
        
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            message=message,
            user_id=user_id,
            username=username,
            outcome=outcome,
            details=details,
            **request_context
        )
        
        # Log using structlog with all event data
        log_method = getattr(self._logger, severity.value, self._logger.info)
        log_method(
            message,
            event_id=event.event_id,
            event_type=event.event_type,
            severity=event.severity,
            source_ip=event.source_ip,
            user_id=event.user_id,
            username=event.username,
            request_id=event.request_id,
            outcome=event.outcome,
            **event.details
        )
        
        return event
    
    # Convenience methods for common security events
    
    def log_auth_success(
        self,
        request: Optional[Request] = None,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        **details
    ) -> SecurityEvent:
        """Log successful authentication."""
        return self.log_event(
            SecurityEventType.AUTH_LOGIN_SUCCESS,
            SecurityEventSeverity.INFO,
            f"Successful authentication for {username or user_id}",
            request=request,
            user_id=user_id,
            username=username,
            outcome="success",
            **details
        )
    
    def log_auth_failure(
        self,
        request: Optional[Request] = None,
        username: Optional[str] = None,
        reason: str = "Unknown",
        **details
    ) -> SecurityEvent:
        """Log failed authentication."""
        return self.log_event(
            SecurityEventType.AUTH_LOGIN_FAILURE,
            SecurityEventSeverity.WARNING,
            f"Authentication failed for {username}: {reason}",
            request=request,
            username=username,
            outcome="failure",
            failure_reason=reason,
            **details
        )
    
    def log_token_invalid(
        self,
        request: Optional[Request] = None,
        reason: str = "Invalid token",
        **details
    ) -> SecurityEvent:
        """Log invalid token usage."""
        return self.log_event(
            SecurityEventType.AUTH_TOKEN_INVALID,
            SecurityEventSeverity.WARNING,
            f"Invalid token: {reason}",
            request=request,
            outcome="failure",
            failure_reason=reason,
            **details
        )
    
    def log_rate_limit_exceeded(
        self,
        request: Optional[Request] = None,
        identifier: str = "unknown",
        limit: int = 0,
        window_seconds: int = 0,
        **details
    ) -> SecurityEvent:
        """Log rate limit exceeded."""
        return self.log_event(
            SecurityEventType.RATE_LIMIT_EXCEEDED,
            SecurityEventSeverity.WARNING,
            f"Rate limit exceeded for {identifier}",
            request=request,
            outcome="failure",
            identifier=identifier,
            limit=limit,
            window_seconds=window_seconds,
            **details
        )
    
    def log_threat_detected(
        self,
        threat_type: SecurityEventType,
        request: Optional[Request] = None,
        payload: Optional[str] = None,
        **details
    ) -> SecurityEvent:
        """Log detected security threat."""
        severity = SecurityEventSeverity.CRITICAL if threat_type in [
            SecurityEventType.THREAT_SQL_INJECTION,
            SecurityEventType.THREAT_COMMAND_INJECTION,
        ] else SecurityEventSeverity.ERROR
        
        return self.log_event(
            threat_type,
            severity,
            f"Security threat detected: {threat_type.value}",
            request=request,
            outcome="blocked",
            payload_preview=payload[:100] if payload else None,
            **details
        )
    
    def log_brute_force_detected(
        self,
        request: Optional[Request] = None,
        identifier: str = "unknown",
        attempts: int = 0,
        **details
    ) -> SecurityEvent:
        """Log brute force attack detection."""
        return self.log_event(
            SecurityEventType.THREAT_BRUTE_FORCE,
            SecurityEventSeverity.CRITICAL,
            f"Brute force attack detected from {identifier}",
            request=request,
            outcome="blocked",
            identifier=identifier,
            attempt_count=attempts,
            **details
        )
    
    def log_account_locked(
        self,
        request: Optional[Request] = None,
        username: Optional[str] = None,
        reason: str = "Too many failed attempts",
        lock_duration_minutes: int = 15,
        **details
    ) -> SecurityEvent:
        """Log account lockout."""
        return self.log_event(
            SecurityEventType.ACCOUNT_LOCKED,
            SecurityEventSeverity.WARNING,
            f"Account locked: {username}",
            request=request,
            username=username,
            outcome="success",
            lock_reason=reason,
            lock_duration_minutes=lock_duration_minutes,
            **details
        )
    
    def log_api_key_event(
        self,
        event_type: SecurityEventType,
        key_id: str,
        request: Optional[Request] = None,
        user_id: Optional[str] = None,
        **details
    ) -> SecurityEvent:
        """Log API key events."""
        severity = SecurityEventSeverity.INFO if event_type in [
            SecurityEventType.API_KEY_CREATED,
            SecurityEventType.API_KEY_USED,
        ] else SecurityEventSeverity.WARNING
        
        return self.log_event(
            event_type,
            severity,
            f"API key event: {event_type.value}",
            request=request,
            user_id=user_id,
            outcome="success" if event_type != SecurityEventType.API_KEY_INVALID else "failure",
            key_id=key_id,
            **details
        )
    
    def log_access_denied(
        self,
        request: Optional[Request] = None,
        user_id: Optional[str] = None,
        resource: str = "unknown",
        required_permission: str = "unknown",
        **details
    ) -> SecurityEvent:
        """Log access denied events."""
        return self.log_event(
            SecurityEventType.AUTHZ_ACCESS_DENIED,
            SecurityEventSeverity.WARNING,
            f"Access denied to {resource}",
            request=request,
            user_id=user_id,
            outcome="failure",
            resource=resource,
            required_permission=required_permission,
            **details
        )


# Global security event logger instance
security_events = SecurityEventLogger()


# ==============================================================================
# Banned Passwords List
# ==============================================================================

# @CIPHER - Common passwords to reject (subset of SecLists 10k-most-common)
BANNED_PASSWORDS: Set[str] = {
    "password", "123456", "12345678", "qwerty", "abc123", "monkey", "1234567",
    "letmein", "trustno1", "dragon", "baseball", "iloveyou", "master", "sunshine",
    "ashley", "bailey", "passw0rd", "shadow", "123123", "654321", "superman",
    "qazwsx", "michael", "football", "password1", "password123", "welcome",
    "welcome1", "admin", "admin123", "root", "toor", "pass", "test", "guest",
    "master123", "changeme", "123456789", "12345", "1234567890", "0987654321",
    "password!", "p@ssw0rd", "p@ssword", "pa$$word", "neurectomy", "secret",
}


# ==============================================================================
# Password Handling
# ==============================================================================

class PasswordHandler:
    """
    Secure password handling with Argon2id (OWASP 2024 recommended).
    
    @CIPHER - Security priorities:
    1. Argon2id (preferred) - OWASP 2024 recommendation
    2. bcrypt (fallback) - Still acceptable
    3. REJECT if neither available - No insecure fallback
    
    Argon2id Parameters (OWASP 2024):
    - Memory: 64 MB (65536 KB)
    - Iterations: 3
    - Parallelism: 4 threads
    - Hash length: 32 bytes
    - Salt length: 16 bytes
    """
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SECURITY_CONFIG
        self._hasher_type: str = "none"
        self._argon2_hasher: Optional[PasswordHasher] = None
        self._bcrypt_context: Optional[CryptContext] = None
        
        # Try Argon2id first (preferred)
        if ARGON2_AVAILABLE:
            try:
                self._argon2_hasher = PasswordHasher(
                    time_cost=3,           # Iterations
                    memory_cost=65536,     # 64 MB
                    parallelism=4,         # Threads
                    hash_len=32,           # Hash output length
                    salt_len=16,           # Salt length
                )
                # Test that it works
                test_hash = self._argon2_hasher.hash("test_password")
                self._argon2_hasher.verify(test_hash, "test_password")
                self._hasher_type = "argon2id"
                logger.info("Password hashing: Argon2id initialized (OWASP 2024 compliant)")
            except Exception as e:
                logger.warning(f"Argon2id initialization failed: {e}")
                self._argon2_hasher = None
        
        # Fallback to bcrypt if Argon2id unavailable
        if self._hasher_type == "none" and PASSLIB_AVAILABLE:
            try:
                self._bcrypt_context = CryptContext(
                    schemes=["bcrypt"],
                    deprecated="auto",
                    bcrypt__rounds=12  # Increased from default 10
                )
                self._bcrypt_context.hash("test")
                self._hasher_type = "bcrypt"
                logger.warning("Password hashing: Using bcrypt fallback (Argon2id preferred)")
            except Exception as e:
                logger.warning(f"bcrypt initialization failed: {e}")
                self._bcrypt_context = None
        
        # @CIPHER - CRITICAL: No insecure fallback (removed SHA256)
        if self._hasher_type == "none":
            logger.error(
                "SECURITY CRITICAL: No secure password hasher available! "
                "Install argon2-cffi (recommended) or passlib[bcrypt]. "
                "SHA256 fallback has been removed for security reasons."
            )
    
    def hash_password(self, password: str) -> str:
        """
        Hash a password securely.
        
        @CIPHER - Only secure algorithms allowed:
        - Argon2id (preferred)
        - bcrypt (acceptable fallback)
        
        Raises:
            RuntimeError: If no secure hasher is available
        """
        if self._hasher_type == "argon2id" and self._argon2_hasher:
            return self._argon2_hasher.hash(password)
        
        if self._hasher_type == "bcrypt" and self._bcrypt_context:
            return self._bcrypt_context.hash(password)
        
        # @CIPHER - CRITICAL: Reject if no secure hasher
        raise RuntimeError(
            "No secure password hasher available. "
            "Install argon2-cffi: pip install argon2-cffi"
        )
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash.
        
        Supports:
        - Argon2id hashes ($argon2id$...)
        - bcrypt hashes ($2b$...)
        - Legacy SHA256 hashes (sha256:...) - VERIFY ONLY, will prompt rehash
        """
        try:
            # Argon2id verification
            if hashed_password.startswith("$argon2"):
                if self._argon2_hasher:
                    try:
                        self._argon2_hasher.verify(hashed_password, plain_password)
                        return True
                    except argon2_exceptions.VerifyMismatchError:
                        return False
                    except argon2_exceptions.InvalidHash:
                        logger.error("Corrupted Argon2 hash detected - possible tampering")
                        return False
                return False
            
            # bcrypt verification
            if hashed_password.startswith("$2"):
                if self._bcrypt_context:
                    try:
                        return self._bcrypt_context.verify(plain_password, hashed_password)
                    except Exception:
                        return False
                return False
            
            # Legacy SHA256 verification (read-only, for migration)
            if hashed_password.startswith("sha256:"):
                logger.warning(
                    "Legacy SHA256 password hash detected - this should be migrated to Argon2id"
                )
                parts = hashed_password.split(":")
                if len(parts) == 3:
                    salt, stored_hash = parts[1], parts[2]
                    computed_hash = hashlib.sha256(f"{salt}{plain_password}".encode()).hexdigest()
                    return secrets.compare_digest(computed_hash, stored_hash)
                return False
            
            return False
            
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    def needs_rehash(self, hashed_password: str) -> bool:
        """
        Check if password hash needs upgrading.
        
        Returns True if:
        - Using legacy SHA256 (must migrate)
        - Using bcrypt when Argon2id is available
        - Argon2id parameters have changed
        """
        # Legacy SHA256 always needs rehash
        if hashed_password.startswith("sha256:"):
            return True
        
        # bcrypt should be upgraded to Argon2id if available
        if hashed_password.startswith("$2") and self._hasher_type == "argon2id":
            return True
        
        # Check if Argon2id parameters changed
        if hashed_password.startswith("$argon2") and self._argon2_hasher:
            try:
                return self._argon2_hasher.check_needs_rehash(hashed_password)
            except Exception:
                return False
        
        return False
    
    def validate_password_strength(self, password: str) -> List[str]:
        """
        Validate password meets OWASP 2024 security requirements.
        
        @CIPHER - Unified policy (matching Rust service):
        - Minimum 12 characters
        - Maximum 128 characters
        - At least 1 uppercase letter
        - At least 1 lowercase letter
        - At least 1 digit
        - At least 1 special character
        - At least 6 unique characters
        - Not in banned password list
        """
        errors = []
        config = self.config
        
        # Length checks
        if len(password) < config.password_min_length:
            errors.append(f"Password must be at least {config.password_min_length} characters")
        
        if len(password) > config.password_max_length:
            errors.append(f"Password must not exceed {config.password_max_length} characters")
        
        # Character class checks
        if not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")
        
        special_chars = "!@#$%^&*()_+-=[]{}|;:'\",.<>?/\\`~"
        if config.password_require_special and not any(c in special_chars for c in password):
            errors.append("Password must contain at least one special character")
        
        # Unique character check
        unique_chars = len(set(password))
        if unique_chars < config.password_min_unique_chars:
            errors.append(f"Password must contain at least {config.password_min_unique_chars} unique characters")
        
        # Banned password check
        if password.lower() in BANNED_PASSWORDS:
            errors.append("This password is too common and not allowed")
        
        return errors


# ==============================================================================
# JWT Token Management
# ==============================================================================

class TokenType(str, Enum):
    """Types of tokens."""
    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"


class TokenPayload(BaseModel):
    """JWT token payload."""
    sub: str  # Subject (user ID)
    type: TokenType
    exp: datetime
    iat: datetime = Field(default_factory=datetime.utcnow)
    scopes: List[str] = Field(default_factory=list)
    jti: str = Field(default_factory=lambda: secrets.token_urlsafe(16))  # JWT ID


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int  # seconds


class TokenManager:
    """
    JWT token creation and validation with distributed token revocation.
    
    @CIPHER @FORTRESS - Security features:
    - Redis-backed token revocation for distributed deployments
    - Automatic TTL cleanup of revoked tokens
    - Fallback to in-memory for development
    - Algorithm restriction to prevent confusion attacks
    """
    
    # @CIPHER - Allowed algorithms (prevent algorithm confusion attacks)
    ALLOWED_ALGORITHMS = {"HS256", "HS384", "HS512", "RS256", "RS384", "RS512", "ES256", "EdDSA"}
    
    def __init__(self, config: SecurityConfig = None, redis_client: Optional[Any] = None):
        self.config = config or SECURITY_CONFIG
        self._redis: Optional[Any] = redis_client
        self._revoked_tokens: Set[str] = set()  # Fallback for non-distributed
        self._revocation_prefix = "token:revoked:"
        
        # Validate algorithm
        if self.config.algorithm not in self.ALLOWED_ALGORITHMS:
            raise ValueError(f"Algorithm {self.config.algorithm} not in allowed list: {self.ALLOWED_ALGORITHMS}")
    
    async def initialize_redis(self, redis_url: str = None) -> bool:
        """
        Initialize Redis connection for distributed token revocation.
        
        @FORTRESS - Best practice: Always use Redis in production for:
        - Multi-instance token revocation
        - Persistence across restarts
        - Distributed rate limiting
        """
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available - using in-memory token store (not suitable for production)")
            return False
        
        try:
            url = redis_url or self.config.redis_url
            self._redis = redis.from_url(url, decode_responses=True)
            await self._redis.ping()
            logger.info("Token manager connected to Redis for distributed revocation")
            return True
        except Exception as e:
            logger.error(f"Redis connection failed: {e} - falling back to in-memory store")
            self._redis = None
            return False
    
    async def _is_token_revoked(self, jti: str) -> bool:
        """Check if token is revoked (Redis or in-memory)."""
        if self._redis:
            try:
                return await self._redis.exists(f"{self._revocation_prefix}{jti}")
            except Exception as e:
                logger.error(f"Redis revocation check failed: {e}")
                # Fall through to in-memory check
        
        return jti in self._revoked_tokens
    
    async def _revoke_token_internal(self, jti: str, exp_timestamp: int) -> None:
        """Add token to revocation store with TTL."""
        if self._redis:
            try:
                # Set TTL based on token expiration
                ttl = max(0, exp_timestamp - int(time.time()))
                if ttl > 0:
                    await self._redis.setex(
                        f"{self._revocation_prefix}{jti}",
                        ttl,
                        "1"
                    )
                    return
            except Exception as e:
                logger.error(f"Redis revocation failed: {e}")
        
        # Fallback to in-memory
        self._revoked_tokens.add(jti)
    
    def create_access_token(
        self,
        user_id: str,
        scopes: List[str] = None
    ) -> str:
        """Create a new access token."""
        if not JWT_AVAILABLE:
            raise RuntimeError("PyJWT not installed")
        
        expires = datetime.utcnow() + timedelta(minutes=self.config.access_token_expire_minutes)
        
        payload = TokenPayload(
            sub=user_id,
            type=TokenType.ACCESS,
            exp=expires,
            scopes=scopes or []
        )
        
        return jwt.encode(
            payload.model_dump(),
            self.config.secret_key,
            algorithm=self.config.algorithm
        )
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create a new refresh token."""
        if not JWT_AVAILABLE:
            raise RuntimeError("PyJWT not installed")
        
        expires = datetime.utcnow() + timedelta(days=self.config.refresh_token_expire_days)
        
        payload = TokenPayload(
            sub=user_id,
            type=TokenType.REFRESH,
            exp=expires
        )
        
        return jwt.encode(
            payload.model_dump(),
            self.config.secret_key,
            algorithm=self.config.algorithm
        )
    
    def create_token_pair(
        self,
        user_id: str,
        scopes: List[str] = None
    ) -> TokenResponse:
        """Create both access and refresh tokens."""
        access_token = self.create_access_token(user_id, scopes)
        refresh_token = self.create_refresh_token(user_id)
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.config.access_token_expire_minutes * 60
        )
    
    async def decode_token_async(self, token: str) -> TokenPayload:
        """Decode and validate a token (async version with Redis revocation check)."""
        if not JWT_AVAILABLE:
            raise RuntimeError("PyJWT not installed")
        
        try:
            # @CIPHER - Explicit algorithm restriction
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],  # Only allow configured algorithm
                options={
                    "require": ["exp", "sub", "jti"],  # Required claims
                    "verify_exp": True,
                    "verify_iat": True,
                }
            )
            
            token_data = TokenPayload(**payload)
            
            # Check if token is revoked (async Redis check)
            if await self._is_token_revoked(token_data.jti):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )
            
            return token_data
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"  # @CIPHER - Don't expose specific error details
            )
    
    def decode_token(self, token: str) -> TokenPayload:
        """
        Decode and validate a token (sync version).
        
        Note: Use decode_token_async() in production for Redis support.
        """
        if not JWT_AVAILABLE:
            raise RuntimeError("PyJWT not installed")
        
        try:
            # @CIPHER - Explicit algorithm restriction
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
                options={
                    "require": ["exp", "sub", "jti"],
                    "verify_exp": True,
                }
            )
            
            token_data = TokenPayload(**payload)
            
            # Check if token is revoked (in-memory only for sync)
            if token_data.jti in self._revoked_tokens:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )
            
            return token_data
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    async def revoke_token(self, token: str) -> None:
        """Revoke a token (add to revocation store with TTL)."""
        try:
            # Decode without revocation check (it's being revoked)
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
                options={"verify_exp": False}  # Allow revoking expired tokens
            )
            token_data = TokenPayload(**payload)
            
            # Get expiration timestamp for TTL
            exp_timestamp = int(token_data.exp.timestamp()) if token_data.exp else int(time.time()) + 86400
            
            await self._revoke_token_internal(token_data.jti, exp_timestamp)
            logger.info("Token revoked", jti=token_data.jti)
            
        except jwt.InvalidTokenError:
            pass  # Token already invalid
    
    def revoke_token_sync(self, token: str) -> None:
        """Revoke a token synchronously (in-memory only)."""
        try:
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
                options={"verify_exp": False}
            )
            token_data = TokenPayload(**payload)
            self._revoked_tokens.add(token_data.jti)
            logger.info("Token revoked (sync)", jti=token_data.jti)
        except jwt.InvalidTokenError:
            pass
    
    async def refresh_access_token(self, refresh_token: str) -> TokenResponse:
        """Generate new access token from refresh token."""
        payload = await self.decode_token_async(refresh_token)
        
        if payload.type != TokenType.REFRESH:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        # Create new access token
        access_token = self.create_access_token(payload.sub, payload.scopes)
        
        return TokenResponse(
            access_token=access_token,
            expires_in=self.config.access_token_expire_minutes * 60
        )


# ==============================================================================
# API Key Management
# ==============================================================================

class APIKey(BaseModel):
    """API Key model."""
    key_id: str
    key_hash: str  # We store hash, not the actual key
    name: str
    user_id: str
    scopes: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    is_active: bool = True


class APIKeyManager:
    """
    API Key generation and validation with Argon2id hashing.
    
    @CIPHER: Uses Argon2id instead of SHA256 for cryptographically secure
    API key storage. Keys are hashed with unique salts and cannot be
    reversed even if the database is compromised.
    """
    
    # Argon2id parameters for API keys (lighter than passwords for faster validation)
    API_KEY_MEMORY_COST = 32768  # 32MB - lower than password for faster validation
    API_KEY_TIME_COST = 1  # Single iteration
    API_KEY_PARALLELISM = 1
    API_KEY_HASH_LENGTH = 32
    
    def __init__(self, config: SecurityConfig = SECURITY_CONFIG):
        self.config = config
        self._keys: Dict[str, APIKey] = {}  # In production, use database
        self._password_handler = PasswordHandler()
        
        # Initialize Argon2 hasher for API keys
        try:
            from argon2 import PasswordHasher, Type
            self._hasher = PasswordHasher(
                memory_cost=self.API_KEY_MEMORY_COST,
                time_cost=self.API_KEY_TIME_COST,
                parallelism=self.API_KEY_PARALLELISM,
                hash_len=self.API_KEY_HASH_LENGTH,
                type=Type.ID  # Argon2id
            )
            self._use_argon2 = True
        except ImportError:
            logger.warning("argon2-cffi not available, API key hashing disabled")
            self._hasher = None
            self._use_argon2 = False
    
    def _hash_api_key(self, raw_key: str) -> str:
        """
        Hash an API key using Argon2id.
        
        Args:
            raw_key: The raw API key to hash
        
        Returns:
            Argon2id hash string
        """
        if self._use_argon2 and self._hasher:
            return self._hasher.hash(raw_key)
        else:
            # Fallback: use PBKDF2-HMAC-SHA256 with high iterations
            import hashlib
            import os
            salt = os.urandom(16)
            key = hashlib.pbkdf2_hmac(
                'sha256',
                raw_key.encode(),
                salt,
                iterations=100000,
                dklen=32
            )
            return f"pbkdf2:{salt.hex()}:{key.hex()}"
    
    def _verify_api_key(self, raw_key: str, key_hash: str) -> bool:
        """
        Verify an API key against its hash.
        
        Args:
            raw_key: The raw API key to verify
            key_hash: The stored hash to compare against
        
        Returns:
            True if the key matches, False otherwise
        """
        try:
            if self._use_argon2 and self._hasher and key_hash.startswith("$argon2"):
                self._hasher.verify(key_hash, raw_key)
                return True
            elif key_hash.startswith("pbkdf2:"):
                # Handle PBKDF2 fallback
                parts = key_hash.split(":")
                if len(parts) != 3:
                    return False
                salt = bytes.fromhex(parts[1])
                stored_key = bytes.fromhex(parts[2])
                import hashlib
                computed_key = hashlib.pbkdf2_hmac(
                    'sha256',
                    raw_key.encode(),
                    salt,
                    iterations=100000,
                    dklen=32
                )
                return secrets.compare_digest(computed_key, stored_key)
            else:
                # Legacy SHA256 hash migration support
                legacy_hash = hashlib.sha256(raw_key.encode()).hexdigest()
                return secrets.compare_digest(key_hash, legacy_hash)
        except Exception:
            return False
    
    def _needs_rehash(self, key_hash: str) -> bool:
        """Check if a key hash needs to be upgraded to Argon2id."""
        if self._use_argon2 and self._hasher:
            # Rehash if using old format or if params have changed
            if not key_hash.startswith("$argon2"):
                return True
            try:
                return self._hasher.check_needs_rehash(key_hash)
            except Exception:
                return True
        return False
    
    def generate_api_key(
        self,
        name: str,
        user_id: str,
        scopes: List[str] = None,
        expires_days: Optional[int] = None
    ) -> tuple[str, APIKey]:
        """
        Generate a new API key with Argon2id hashing.
        
        Returns: (raw_key, api_key_model)
        """
        # Generate key with prefix
        raw_key = f"{self.config.api_key_prefix}{secrets.token_urlsafe(32)}"
        key_id = secrets.token_urlsafe(8)
        
        # Hash the key using Argon2id
        key_hash = self._hash_api_key(raw_key)
        
        expires_at = None
        if expires_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)
        
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            user_id=user_id,
            scopes=scopes or [],
            expires_at=expires_at
        )
        
        self._keys[key_id] = api_key
        
        logger.info("API key generated", key_id=key_id, name=name, user_id=user_id)
        
        # Return raw key (only time it's visible) and model
        return raw_key, api_key
    
    def validate_api_key(self, raw_key: str) -> Optional[APIKey]:
        """
        Validate an API key and return associated data.
        
        Supports automatic migration from legacy SHA256 hashes to Argon2id.
        """
        if not raw_key.startswith(self.config.api_key_prefix):
            return None
        
        for api_key in self._keys.values():
            if self._verify_api_key(raw_key, api_key.key_hash):
                # Check if active
                if not api_key.is_active:
                    return None
                
                # Check expiration
                if api_key.expires_at and datetime.utcnow() > api_key.expires_at:
                    return None
                
                # Rehash if needed (transparent migration to Argon2id)
                if self._needs_rehash(api_key.key_hash):
                    api_key.key_hash = self._hash_api_key(raw_key)
                    logger.info("API key hash upgraded to Argon2id", key_id=api_key.key_id)
                
                # Update last used
                api_key.last_used_at = datetime.utcnow()
                
                return api_key
        
        return None
    
    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        if key_id in self._keys:
            self._keys[key_id].is_active = False
            logger.info("API key revoked", key_id=key_id)
            return True
        return False
    
    def list_api_keys(self, user_id: str) -> List[APIKey]:
        """List all API keys for a user."""
        return [
            key for key in self._keys.values()
            if key.user_id == user_id
        ]


# ==============================================================================
# Rate Limiting
# ==============================================================================

class RateLimitResult(BaseModel):
    """Rate limit check result."""
    allowed: bool
    remaining: int
    reset_at: datetime
    limit: int
    retry_after: Optional[int] = None  # Seconds until next request allowed


class RateLimiter:
    """
    Distributed sliding window rate limiter with Redis backend.
    
    @FORTRESS - Features:
    - Redis-backed for distributed deployments
    - Sliding window (more accurate than fixed window)
    - Per-user and per-endpoint rate limiting
    - Graceful fallback to in-memory for development
    - Exponential backoff recommendations
    
    Algorithm: Sliding Window Log
    - Stores timestamps of requests in sorted set
    - Removes expired timestamps on each check
    - O(log n) for each operation
    """
    
    def __init__(
        self,
        requests_per_window: int = None,
        window_seconds: int = None,
        redis_client: Optional[Any] = None,
        config: SecurityConfig = None
    ):
        self.config = config or SECURITY_CONFIG
        self.requests_per_window = requests_per_window or self.config.rate_limit_requests
        self.window_seconds = window_seconds or self.config.rate_limit_window_seconds
        self._redis: Optional[Any] = redis_client
        self._prefix = "ratelimit:"
        
        # Fallback in-memory storage (development only)
        self._buckets: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"tokens": self.requests_per_window, "last_update": time.time()}
        )
        self._request_logs: Dict[str, List[float]] = defaultdict(list)
    
    async def initialize_redis(self, redis_url: str = None) -> bool:
        """Initialize Redis connection for distributed rate limiting."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available - using in-memory rate limiter (not suitable for production)")
            return False
        
        try:
            url = redis_url or self.config.redis_url
            self._redis = redis.from_url(url, decode_responses=True)
            await self._redis.ping()
            logger.info("Rate limiter connected to Redis for distributed limiting")
            return True
        except Exception as e:
            logger.error(f"Redis connection failed: {e} - falling back to in-memory limiter")
            self._redis = None
            return False
    
    def _get_bucket_key(
        self,
        identifier: str,
        endpoint: Optional[str] = None
    ) -> str:
        """Generate bucket key."""
        base = f"{self._prefix}{identifier}"
        if endpoint:
            # Normalize endpoint for consistent keys
            endpoint_hash = hashlib.md5(endpoint.encode()).hexdigest()[:8]
            return f"{base}:{endpoint_hash}"
        return base
    
    async def check_async(
        self,
        identifier: str,
        endpoint: Optional[str] = None,
        cost: int = 1
    ) -> RateLimitResult:
        """
        Check if request is allowed (async with Redis support).
        
        @FORTRESS - Sliding window algorithm:
        1. Remove timestamps older than window
        2. Count remaining timestamps
        3. Allow if count < limit
        4. Add current timestamp if allowed
        """
        key = self._get_bucket_key(identifier, endpoint)
        now = time.time()
        window_start = now - self.window_seconds
        
        if self._redis:
            try:
                # Use Redis pipeline for atomicity
                pipe = self._redis.pipeline()
                
                # Remove expired entries
                pipe.zremrangebyscore(key, "-inf", window_start)
                
                # Count current entries
                pipe.zcard(key)
                
                # Execute
                results = await pipe.execute()
                current_count = results[1]
                
                if current_count < self.requests_per_window:
                    # Add new request with current timestamp as score
                    await self._redis.zadd(key, {f"{now}:{identifier}": now})
                    await self._redis.expire(key, self.window_seconds + 10)  # TTL with buffer
                    
                    return RateLimitResult(
                        allowed=True,
                        remaining=self.requests_per_window - current_count - 1,
                        reset_at=datetime.utcnow() + timedelta(seconds=self.window_seconds),
                        limit=self.requests_per_window
                    )
                else:
                    # Get oldest timestamp to calculate retry_after
                    oldest = await self._redis.zrange(key, 0, 0, withscores=True)
                    retry_after = int(oldest[0][1] - window_start) + 1 if oldest else self.window_seconds
                    
                    return RateLimitResult(
                        allowed=False,
                        remaining=0,
                        reset_at=datetime.utcnow() + timedelta(seconds=retry_after),
                        limit=self.requests_per_window,
                        retry_after=retry_after
                    )
                    
            except Exception as e:
                logger.error(f"Redis rate limit check failed: {e}")
                # Fall through to in-memory
        
        # In-memory fallback (sliding window log)
        request_log = self._request_logs[key]
        
        # Remove expired entries
        request_log = [ts for ts in request_log if ts > window_start]
        self._request_logs[key] = request_log
        
        if len(request_log) < self.requests_per_window:
            request_log.append(now)
            return RateLimitResult(
                allowed=True,
                remaining=self.requests_per_window - len(request_log),
                reset_at=datetime.utcnow() + timedelta(seconds=self.window_seconds),
                limit=self.requests_per_window
            )
        else:
            retry_after = int(request_log[0] - window_start) + 1 if request_log else self.window_seconds
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=datetime.utcnow() + timedelta(seconds=retry_after),
                limit=self.requests_per_window,
                retry_after=retry_after
            )
    
    def check(
        self,
        identifier: str,
        endpoint: Optional[str] = None,
        cost: int = 1
    ) -> RateLimitResult:
        """
        Check if request is allowed (sync version, in-memory only).
        
        Use check_async() in production for Redis support.
        """
        key = self._get_bucket_key(identifier, endpoint)
        now = time.time()
        window_start = now - self.window_seconds
        
        # In-memory sliding window log
        request_log = self._request_logs[key]
        
        # Remove expired entries
        request_log = [ts for ts in request_log if ts > window_start]
        self._request_logs[key] = request_log
        
        if len(request_log) < self.requests_per_window:
            request_log.append(now)
            return RateLimitResult(
                allowed=True,
                remaining=self.requests_per_window - len(request_log),
                reset_at=datetime.utcnow() + timedelta(seconds=self.window_seconds),
                limit=self.requests_per_window
            )
        else:
            retry_after = int(request_log[0] - window_start) + 1 if request_log else self.window_seconds
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=datetime.utcnow() + timedelta(seconds=retry_after),
                limit=self.requests_per_window,
                retry_after=retry_after
            )
    
    async def get_remaining_async(self, identifier: str, endpoint: Optional[str] = None) -> int:
        """Get remaining requests for identifier (async)."""
        key = self._get_bucket_key(identifier, endpoint)
        now = time.time()
        window_start = now - self.window_seconds
        
        if self._redis:
            try:
                # Remove expired and count
                await self._redis.zremrangebyscore(key, "-inf", window_start)
                count = await self._redis.zcard(key)
                return max(0, self.requests_per_window - count)
            except Exception:
                pass
        
        # In-memory fallback
        request_log = self._request_logs.get(key, [])
        request_log = [ts for ts in request_log if ts > window_start]
        return max(0, self.requests_per_window - len(request_log))
    
    def get_remaining(self, identifier: str, endpoint: Optional[str] = None) -> int:
        """Get remaining requests for identifier (sync)."""
        key = self._get_bucket_key(identifier, endpoint)
        now = time.time()
        window_start = now - self.window_seconds
        
        request_log = self._request_logs.get(key, [])
        request_log = [ts for ts in request_log if ts > window_start]
        return max(0, self.requests_per_window - len(request_log))


# ==============================================================================
# Audit Logging
# ==============================================================================

class AuditEventType(str, Enum):
    """Types of audit events."""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    TOKEN_REFRESH = "token_refresh"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    PERMISSION_DENIED = "permission_denied"
    RATE_LIMITED = "rate_limited"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    ADMIN_ACTION = "admin_action"


class AuditLogEntry(BaseModel):
    """Audit log entry."""
    id: str = Field(default_factory=lambda: secrets.token_urlsafe(16))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_type: AuditEventType
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    status_code: Optional[int] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class AuditLogger:
    """Security audit logger."""
    
    def __init__(self, max_entries: int = 10000):
        self._entries: List[AuditLogEntry] = []
        self._max_entries = max_entries
    
    def log(
        self,
        event_type: AuditEventType,
        user_id: Optional[str] = None,
        request: Optional[Request] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> AuditLogEntry:
        """Log an audit event."""
        entry = AuditLogEntry(
            event_type=event_type,
            user_id=user_id,
            details=details or {}
        )
        
        if request:
            entry.ip_address = request.client.host if request.client else None
            entry.user_agent = request.headers.get("user-agent")
            entry.endpoint = str(request.url.path)
            entry.method = request.method
        
        self._entries.append(entry)
        
        # Trim old entries
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]
        
        # Also log to structured logger
        logger.info(
            "audit_event",
            event_type=event_type.value,
            user_id=user_id,
            ip_address=entry.ip_address,
            endpoint=entry.endpoint
        )
        
        return entry
    
    def get_entries(
        self,
        user_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditLogEntry]:
        """Query audit log entries."""
        entries = self._entries
        
        if user_id:
            entries = [e for e in entries if e.user_id == user_id]
        
        if event_type:
            entries = [e for e in entries if e.event_type == event_type]
        
        if since:
            entries = [e for e in entries if e.timestamp >= since]
        
        return entries[-limit:]


# ==============================================================================
# FastAPI Security Dependencies
# ==============================================================================

# Initialize components
token_manager = TokenManager()
api_key_manager = APIKeyManager()
rate_limiter = RateLimiter()
audit_logger = AuditLogger()
password_handler = PasswordHandler()

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    api_key: Optional[str] = Depends(api_key_header),
    request: Request = None
) -> Dict[str, Any]:
    """
    Get current user from JWT token or API key.
    
    Returns user info dict with user_id and scopes.
    """
    user_info = None
    
    # Try JWT token first
    if credentials:
        try:
            payload = token_manager.decode_token(credentials.credentials)
            user_info = {
                "user_id": payload.sub,
                "scopes": payload.scopes,
                "auth_type": "jwt"
            }
        except HTTPException:
            pass  # Try API key next
    
    # Try API key
    if not user_info and api_key:
        validated_key = api_key_manager.validate_api_key(api_key)
        if validated_key:
            user_info = {
                "user_id": validated_key.user_id,
                "scopes": validated_key.scopes,
                "auth_type": "api_key",
                "key_id": validated_key.key_id
            }
    
    if not user_info:
        audit_logger.log(
            AuditEventType.PERMISSION_DENIED,
            request=request,
            details={"reason": "No valid authentication"}
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return user_info


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    api_key: Optional[str] = Depends(api_key_header)
) -> Optional[Dict[str, Any]]:
    """Get current user if authenticated, None otherwise."""
    try:
        return await get_current_user(credentials, api_key)
    except HTTPException:
        return None


def require_scopes(*required_scopes: str):
    """Dependency factory for scope-based authorization."""
    async def check_scopes(
        user: Dict[str, Any] = Depends(get_current_user),
        request: Request = None
    ) -> Dict[str, Any]:
        user_scopes = set(user.get("scopes", []))
        
        # Admin has all scopes
        if "admin" in user_scopes:
            return user
        
        missing = set(required_scopes) - user_scopes
        if missing:
            audit_logger.log(
                AuditEventType.PERMISSION_DENIED,
                user_id=user.get("user_id"),
                request=request,
                details={"required_scopes": list(required_scopes), "missing": list(missing)}
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required scopes: {', '.join(missing)}"
            )
        
        return user
    
    return check_scopes


async def rate_limit_check(
    request: Request,
    user: Optional[Dict[str, Any]] = Depends(get_optional_user)
) -> None:
    """Rate limiting dependency."""
    # Use user ID if authenticated, otherwise IP
    if user:
        identifier = user.get("user_id", "anonymous")
    else:
        identifier = request.client.host if request.client else "unknown"
    
    result = rate_limiter.check(identifier, str(request.url.path))
    
    if not result.allowed:
        audit_logger.log(
            AuditEventType.RATE_LIMITED,
            user_id=user.get("user_id") if user else None,
            request=request
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={
                "X-RateLimit-Limit": str(result.limit),
                "X-RateLimit-Remaining": str(result.remaining),
                "X-RateLimit-Reset": result.reset_at.isoformat()
            }
        )


# ==============================================================================
# Security Middleware
# ==============================================================================

class SecurityHeaders:
    """Security headers to add to responses."""
    
    HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
    }


def add_security_headers(response):
    """Add security headers to response."""
    for header, value in SecurityHeaders.HEADERS.items():
        response.headers[header] = value
    return response


# ==============================================================================
# Input Validation & Sanitization
# ==============================================================================

class InputSanitizer:
    """
    Comprehensive input sanitization utilities.
    
    @CIPHER: Enhanced with recursive sanitization, expanded XSS patterns,
    content-type validation, and path traversal protection.
    """
    
    # SQL Injection patterns - comprehensive detection
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|EXEC|EXECUTE)\b)",
        r"(\b(CREATE|TRUNCATE|GRANT|REVOKE|COMMIT|ROLLBACK)\b)",
        r"(\b(DECLARE|CURSOR|FETCH|OPEN|CLOSE)\b)",
        r"(--|#|/\*|\*/)",
        r"('|\"|`)",
        r"(\bOR\b\s*\d+\s*=\s*\d+)",  # OR 1=1 patterns
        r"(\bAND\b\s*\d+\s*=\s*\d+)",  # AND 1=1 patterns
        r"(;\s*(DROP|DELETE|UPDATE|INSERT))",  # Statement chaining
        r"(\bWAITFOR\b\s+\bDELAY\b)",  # Time-based injection
        r"(\bBENCHMARK\b\s*\()",  # MySQL benchmark
        r"(\bSLEEP\b\s*\()",  # Time-based
    ]
    
    # XSS patterns - comprehensive coverage
    XSS_PATTERNS = [
        r"<script[^>]*>",
        r"</script>",
        r"javascript\s*:",
        r"vbscript\s*:",
        r"on\w+\s*=",  # Event handlers
        r"<\s*iframe",
        r"<\s*frame",
        r"<\s*embed",
        r"<\s*object",
        r"<\s*applet",
        r"<\s*meta",
        r"<\s*link",
        r"<\s*style",
        r"<\s*base",
        r"<\s*form",
        r"<\s*input",
        r"<\s*body[^>]*onload",
        r"<\s*img[^>]*onerror",
        r"<\s*svg[^>]*onload",
        r"expression\s*\(",  # CSS expression
        r"url\s*\(\s*[\"']?\s*data:",  # Data URLs in CSS
        r"@import",  # CSS import
        r"<\s*!\[CDATA\[",  # CDATA sections
        r"&\{",  # HTML entity exploitation
        r"\\x[0-9a-fA-F]{2}",  # Hex encoding
        r"\\u[0-9a-fA-F]{4}",  # Unicode encoding
        r"%3[cC]",  # URL-encoded <
        r"%3[eE]",  # URL-encoded >
        r"&#x?[0-9a-fA-F]+;?",  # HTML entities (numeric)
    ]
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.\\",
        r"%2[eE]%2[eE]",  # URL-encoded ..
        r"%252[eE]%252[eE]",  # Double-encoded ..
        r"/etc/passwd",
        r"/etc/shadow",
        r"C:\\Windows",
        r"\\\\[^\\]+\\",  # UNC paths
    ]
    
    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$]",
        r"\$\(",
        r"`[^`]*`",
        r"\|\|",
        r"&&",
        r"\$\{",
        r">\s*/dev/",
        r"<\s*/dev/",
    ]
    
    # Allowed content types
    ALLOWED_CONTENT_TYPES = {
        "application/json",
        "application/xml",
        "text/plain",
        "text/html",
        "multipart/form-data",
        "application/x-www-form-urlencoded",
    }
    
    # HTML entities for escaping
    HTML_ESCAPE_MAP = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#x27;",
        "/": "&#x2F;",
        "`": "&#x60;",
        "=": "&#x3D;",
    }
    
    @classmethod
    def sanitize_string(
        cls,
        value: str,
        max_length: int = 1000,
        escape_html: bool = True,
        strip_dangerous: bool = True
    ) -> str:
        """
        Sanitize a string input with comprehensive protection.
        
        Args:
            value: Input string to sanitize
            max_length: Maximum allowed length
            escape_html: Whether to escape HTML characters
            strip_dangerous: Whether to remove dangerous patterns
        
        Returns:
            Sanitized string
        """
        if not value:
            return value
        
        # Truncate
        value = value[:max_length]
        
        # Remove null bytes and other control characters
        value = "".join(
            char for char in value
            if ord(char) >= 32 or char in "\t\n\r"
        )
        
        # Normalize Unicode to prevent bypasses
        import unicodedata
        value = unicodedata.normalize("NFKC", value)
        
        # Strip dangerous patterns if requested
        if strip_dangerous:
            import re
            for pattern in cls.XSS_PATTERNS:
                value = re.sub(pattern, "", value, flags=re.IGNORECASE)
        
        # Escape HTML if requested
        if escape_html:
            for char, entity in cls.HTML_ESCAPE_MAP.items():
                value = value.replace(char, entity)
        
        # Strip leading/trailing whitespace
        value = value.strip()
        
        return value
    
    @classmethod
    def sanitize_object(
        cls,
        obj: Any,
        max_depth: int = 10,
        max_string_length: int = 1000,
        _current_depth: int = 0
    ) -> Any:
        """
        Recursively sanitize an object (dict, list, or primitive).
        
        Args:
            obj: Object to sanitize
            max_depth: Maximum recursion depth
            max_string_length: Maximum string length
            _current_depth: Current recursion depth (internal)
        
        Returns:
            Sanitized object
        """
        if _current_depth >= max_depth:
            return None  # Prevent infinite recursion
        
        if obj is None:
            return None
        
        if isinstance(obj, str):
            return cls.sanitize_string(obj, max_length=max_string_length)
        
        if isinstance(obj, (int, float, bool)):
            return obj
        
        if isinstance(obj, dict):
            return {
                cls.sanitize_string(str(k), max_length=256, escape_html=False): 
                cls.sanitize_object(
                    v,
                    max_depth=max_depth,
                    max_string_length=max_string_length,
                    _current_depth=_current_depth + 1
                )
                for k, v in obj.items()
            }
        
        if isinstance(obj, (list, tuple)):
            return [
                cls.sanitize_object(
                    item,
                    max_depth=max_depth,
                    max_string_length=max_string_length,
                    _current_depth=_current_depth + 1
                )
                for item in obj
            ]
        
        # For other types, convert to string and sanitize
        return cls.sanitize_string(str(obj), max_length=max_string_length)
    
    @classmethod
    def detect_sql_injection(cls, value: str) -> bool:
        """
        Detect potential SQL injection patterns.
        
        Args:
            value: Input string to check
        
        Returns:
            True if SQL injection detected, False otherwise
        """
        if not value:
            return False
        
        import re
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        return False
    
    @classmethod
    def detect_xss(cls, value: str) -> bool:
        """
        Detect potential XSS patterns.
        
        Args:
            value: Input string to check
        
        Returns:
            True if XSS detected, False otherwise
        """
        if not value:
            return False
        
        import re
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        return False
    
    @classmethod
    def detect_path_traversal(cls, value: str) -> bool:
        """
        Detect path traversal attempts.
        
        Args:
            value: Input string to check
        
        Returns:
            True if path traversal detected, False otherwise
        """
        if not value:
            return False
        
        import re
        for pattern in cls.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        return False
    
    @classmethod
    def detect_command_injection(cls, value: str) -> bool:
        """
        Detect command injection attempts.
        
        Args:
            value: Input string to check
        
        Returns:
            True if command injection detected, False otherwise
        """
        if not value:
            return False
        
        import re
        for pattern in cls.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, value):
                return True
        return False
    
    @classmethod
    def detect_all_threats(cls, value: str) -> Dict[str, bool]:
        """
        Run all threat detection checks on a value.
        
        Args:
            value: Input string to check
        
        Returns:
            Dictionary with threat detection results
        """
        return {
            "sql_injection": cls.detect_sql_injection(value),
            "xss": cls.detect_xss(value),
            "path_traversal": cls.detect_path_traversal(value),
            "command_injection": cls.detect_command_injection(value),
        }
    
    @classmethod
    def validate_content_type(cls, content_type: str) -> bool:
        """
        Validate content type against allowlist.
        
        Args:
            content_type: Content-Type header value
        
        Returns:
            True if content type is allowed, False otherwise
        """
        if not content_type:
            return False
        
        # Extract base content type (ignore charset, etc.)
        base_type = content_type.split(";")[0].strip().lower()
        return base_type in cls.ALLOWED_CONTENT_TYPES
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format with comprehensive checks."""
        if not email or len(email) > 254:
            return False
        
        import re
        # RFC 5322 compliant pattern
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            return False
        
        # Additional checks
        local_part, domain = email.rsplit("@", 1)
        
        # Local part length check (max 64)
        if len(local_part) > 64:
            return False
        
        # Domain length check (max 253)
        if len(domain) > 253:
            return False
        
        # No consecutive dots
        if ".." in email:
            return False
        
        return True
    
    @staticmethod
    def validate_uuid(value: str) -> bool:
        """Validate UUID format."""
        if not value:
            return False
        
        import re
        pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
        return bool(re.match(pattern, value, re.IGNORECASE))
    
    @staticmethod
    def sanitize_filename(filename: str, max_length: int = 255) -> str:
        """
        Sanitize a filename for safe storage.
        
        Args:
            filename: Original filename
            max_length: Maximum filename length
        
        Returns:
            Safe filename
        """
        if not filename:
            return "unnamed"
        
        import re
        
        # Remove path components
        filename = filename.replace("\\", "/").split("/")[-1]
        
        # Remove null bytes
        filename = filename.replace("\x00", "")
        
        # Remove dangerous characters
        filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", filename)
        
        # Remove leading/trailing dots and spaces
        filename = filename.strip(". ")
        
        # Truncate
        if len(filename) > max_length:
            name, ext = (filename.rsplit(".", 1) + [""])[:2]
            if ext:
                max_name_len = max_length - len(ext) - 1
                filename = f"{name[:max_name_len]}.{ext}"
            else:
                filename = filename[:max_length]
        
        return filename or "unnamed"
    
    @classmethod
    def sanitize_url(cls, url: str) -> Optional[str]:
        """
        Sanitize and validate a URL.
        
        Args:
            url: URL to sanitize
        
        Returns:
            Sanitized URL or None if invalid
        """
        if not url:
            return None
        
        from urllib.parse import urlparse, urlunparse
        
        try:
            parsed = urlparse(url)
            
            # Only allow http and https schemes
            if parsed.scheme not in ("http", "https"):
                return None
            
            # Ensure netloc exists
            if not parsed.netloc:
                return None
            
            # Check for dangerous patterns
            if cls.detect_xss(url) or cls.detect_command_injection(url):
                return None
            
            # Reconstruct URL safely
            return urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                ""  # Remove fragment
            ))
        except Exception:
            return None


# ==============================================================================
# Login Attempt Tracking
# ==============================================================================

class LoginAttemptTracker:
    """Track failed login attempts for brute force protection."""
    
    def __init__(
        self,
        max_attempts: int = SECURITY_CONFIG.max_failed_attempts,
        lockout_minutes: int = SECURITY_CONFIG.lockout_duration_minutes
    ):
        self.max_attempts = max_attempts
        self.lockout_minutes = lockout_minutes
        self._attempts: Dict[str, List[datetime]] = defaultdict(list)
        self._lockouts: Dict[str, datetime] = {}
    
    def record_failure(self, identifier: str) -> None:
        """Record a failed login attempt."""
        now = datetime.utcnow()
        self._attempts[identifier].append(now)
        
        # Clean old attempts
        cutoff = now - timedelta(minutes=self.lockout_minutes)
        self._attempts[identifier] = [
            t for t in self._attempts[identifier]
            if t > cutoff
        ]
        
        # Check if should lockout
        if len(self._attempts[identifier]) >= self.max_attempts:
            self._lockouts[identifier] = now + timedelta(minutes=self.lockout_minutes)
            logger.warning("Account locked due to failed attempts", identifier=identifier)
    
    def record_success(self, identifier: str) -> None:
        """Record successful login, clear attempts."""
        self._attempts[identifier] = []
        if identifier in self._lockouts:
            del self._lockouts[identifier]
    
    def is_locked(self, identifier: str) -> bool:
        """Check if identifier is locked out."""
        if identifier in self._lockouts:
            if datetime.utcnow() < self._lockouts[identifier]:
                return True
            else:
                del self._lockouts[identifier]
        return False
    
    def get_lockout_remaining(self, identifier: str) -> Optional[int]:
        """Get remaining lockout time in seconds."""
        if identifier in self._lockouts:
            remaining = (self._lockouts[identifier] - datetime.utcnow()).total_seconds()
            if remaining > 0:
                return int(remaining)
        return None


# Initialize tracker
login_tracker = LoginAttemptTracker()


# ==============================================================================
# Convenience Functions
# ==============================================================================

def create_user_tokens(user_id: str, scopes: List[str] = None) -> TokenResponse:
    """Create tokens for a user."""
    return token_manager.create_token_pair(user_id, scopes)


def verify_user_password(plain_password: str, hashed_password: str) -> bool:
    """Verify user password."""
    return password_handler.verify_password(plain_password, hashed_password)


def hash_user_password(password: str) -> str:
    """Hash a password for storage."""
    return password_handler.hash_password(password)


def generate_api_key_for_user(
    name: str,
    user_id: str,
    scopes: List[str] = None
) -> tuple[str, APIKey]:
    """Generate API key for user."""
    return api_key_manager.generate_api_key(name, user_id, scopes)
