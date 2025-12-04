"""
Security Module for ML Service

@CIPHER @FORTRESS - Authentication, authorization, and security middleware.

Copyright (c) 2025 NEURECTOMY. All Rights Reserved.

Features:
- JWT-based authentication
- API key management
- Rate limiting
- Request validation
- Audit logging
- Security headers
"""

import hashlib
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from functools import wraps
import structlog
from collections import defaultdict

from pydantic import BaseModel, Field, EmailStr
from fastapi import HTTPException, Request, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

try:
    from passlib.context import CryptContext
    PASSLIB_AVAILABLE = True
except ImportError:
    PASSLIB_AVAILABLE = False


logger = structlog.get_logger()


# ==============================================================================
# Security Configuration
# ==============================================================================

class SecurityConfig(BaseModel):
    """Security configuration."""
    secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    api_key_prefix: str = "neurectomy_"
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60
    password_min_length: int = 8
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 15


# Default config
SECURITY_CONFIG = SecurityConfig()


# ==============================================================================
# Password Handling
# ==============================================================================

class PasswordHandler:
    """Secure password handling with bcrypt."""
    
    def __init__(self):
        self._bcrypt_available = False
        if PASSLIB_AVAILABLE:
            try:
                self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
                # Test that bcrypt actually works (compatibility check for Python 3.13)
                self.pwd_context.hash("test")
                self._bcrypt_available = True
            except (ValueError, Exception) as e:
                logger.warning(f"bcrypt not fully compatible: {e}, using fallback")
                self.pwd_context = None
        else:
            self.pwd_context = None
            logger.warning("passlib not available, using fallback hashing")
    
    def hash_password(self, password: str) -> str:
        """Hash a password securely."""
        if self._bcrypt_available and self.pwd_context:
            try:
                return self.pwd_context.hash(password)
            except ValueError:
                # Fallback if bcrypt fails
                pass
        # Fallback: SHA256 with salt (not recommended for production)
        salt = secrets.token_hex(16)
        hashed = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
        return f"sha256:{salt}:{hashed}"
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        if self._bcrypt_available and self.pwd_context and not hashed_password.startswith("sha256:"):
            try:
                return self.pwd_context.verify(plain_password, hashed_password)
            except ValueError:
                return False
        # Fallback verification
        if hashed_password.startswith("sha256:"):
            parts = hashed_password.split(":")
            if len(parts) == 3:
                salt, stored_hash = parts[1], parts[2]
                computed_hash = hashlib.sha256(f"{salt}{plain_password}".encode()).hexdigest()
                return secrets.compare_digest(computed_hash, stored_hash)
        return False
    
    def validate_password_strength(self, password: str) -> List[str]:
        """Validate password meets security requirements."""
        errors = []
        
        if len(password) < SECURITY_CONFIG.password_min_length:
            errors.append(f"Password must be at least {SECURITY_CONFIG.password_min_length} characters")
        
        if not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")
        
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("Password must contain at least one special character")
        
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
    """JWT token creation and validation."""
    
    def __init__(self, config: SecurityConfig = SECURITY_CONFIG):
        self.config = config
        self._revoked_tokens: set = set()  # In production, use Redis
    
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
    
    def decode_token(self, token: str) -> TokenPayload:
        """Decode and validate a token."""
        if not JWT_AVAILABLE:
            raise RuntimeError("PyJWT not installed")
        
        try:
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm]
            )
            
            token_data = TokenPayload(**payload)
            
            # Check if token is revoked
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
                detail=f"Invalid token: {str(e)}"
            )
    
    def revoke_token(self, token: str) -> None:
        """Revoke a token (add to blacklist)."""
        try:
            payload = self.decode_token(token)
            self._revoked_tokens.add(payload.jti)
            logger.info("Token revoked", jti=payload.jti)
        except HTTPException:
            pass  # Token already invalid
    
    def refresh_access_token(self, refresh_token: str) -> TokenResponse:
        """Generate new access token from refresh token."""
        payload = self.decode_token(refresh_token)
        
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
    """API Key generation and validation."""
    
    def __init__(self, config: SecurityConfig = SECURITY_CONFIG):
        self.config = config
        self._keys: Dict[str, APIKey] = {}  # In production, use database
        self._password_handler = PasswordHandler()
    
    def generate_api_key(
        self,
        name: str,
        user_id: str,
        scopes: List[str] = None,
        expires_days: Optional[int] = None
    ) -> tuple[str, APIKey]:
        """
        Generate a new API key.
        
        Returns: (raw_key, api_key_model)
        """
        # Generate key with prefix
        raw_key = f"{self.config.api_key_prefix}{secrets.token_urlsafe(32)}"
        key_id = secrets.token_urlsafe(8)
        
        # Hash the key for storage
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
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
        """Validate an API key and return associated data."""
        if not raw_key.startswith(self.config.api_key_prefix):
            return None
        
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        for api_key in self._keys.values():
            if secrets.compare_digest(api_key.key_hash, key_hash):
                # Check if active
                if not api_key.is_active:
                    return None
                
                # Check expiration
                if api_key.expires_at and datetime.utcnow() > api_key.expires_at:
                    return None
                
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


class RateLimiter:
    """
    Token bucket rate limiter.
    
    Supports per-user and per-endpoint rate limiting.
    """
    
    def __init__(
        self,
        requests_per_window: int = SECURITY_CONFIG.rate_limit_requests,
        window_seconds: int = SECURITY_CONFIG.rate_limit_window_seconds
    ):
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self._buckets: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"tokens": requests_per_window, "last_update": time.time()}
        )
    
    def _get_bucket_key(
        self,
        identifier: str,
        endpoint: Optional[str] = None
    ) -> str:
        """Generate bucket key."""
        if endpoint:
            return f"{identifier}:{endpoint}"
        return identifier
    
    def check(
        self,
        identifier: str,
        endpoint: Optional[str] = None,
        cost: int = 1
    ) -> RateLimitResult:
        """
        Check if request is allowed.
        
        Args:
            identifier: User ID, IP address, or API key
            endpoint: Optional endpoint for per-endpoint limits
            cost: Number of tokens this request costs
        """
        key = self._get_bucket_key(identifier, endpoint)
        bucket = self._buckets[key]
        
        now = time.time()
        elapsed = now - bucket["last_update"]
        
        # Refill tokens based on time elapsed
        refill = (elapsed / self.window_seconds) * self.requests_per_window
        bucket["tokens"] = min(self.requests_per_window, bucket["tokens"] + refill)
        bucket["last_update"] = now
        
        # Check if request is allowed
        if bucket["tokens"] >= cost:
            bucket["tokens"] -= cost
            return RateLimitResult(
                allowed=True,
                remaining=int(bucket["tokens"]),
                reset_at=datetime.utcnow() + timedelta(seconds=self.window_seconds),
                limit=self.requests_per_window
            )
        else:
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=datetime.utcnow() + timedelta(seconds=self.window_seconds),
                limit=self.requests_per_window
            )
    
    def get_remaining(self, identifier: str, endpoint: Optional[str] = None) -> int:
        """Get remaining requests for identifier."""
        key = self._get_bucket_key(identifier, endpoint)
        bucket = self._buckets.get(key)
        if bucket:
            return int(bucket["tokens"])
        return self.requests_per_window


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
    """Input sanitization utilities."""
    
    # Dangerous patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER)\b)",
        r"(--|#|;)",
        r"('|\"|`)",
    ]
    
    XSS_PATTERNS = [
        r"<script[^>]*>",
        r"javascript:",
        r"on\w+\s*=",
    ]
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """Sanitize a string input."""
        if not value:
            return value
        
        # Truncate
        value = value[:max_length]
        
        # Remove null bytes
        value = value.replace("\x00", "")
        
        # Strip leading/trailing whitespace
        value = value.strip()
        
        return value
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_uuid(value: str) -> bool:
        """Validate UUID format."""
        import re
        pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        return bool(re.match(pattern, value, re.IGNORECASE))


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
