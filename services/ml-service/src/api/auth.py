"""
Authentication API Endpoints

@CIPHER @FORTRESS - REST API for authentication and authorization.

Copyright (c) 2025 NEURECTOMY. All Rights Reserved.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Request, status
from pydantic import BaseModel, Field, EmailStr
import structlog

from src.security import (
    token_manager,
    api_key_manager,
    audit_logger,
    password_handler,
    login_tracker,
    rate_limiter,
    get_current_user,
    require_scopes,
    rate_limit_check,
    TokenResponse,
    APIKey,
    AuditEventType,
    RateLimitResult,
)


logger = structlog.get_logger()
router = APIRouter(prefix="/auth", tags=["authentication"])


# ==============================================================================
# Request/Response Models
# ==============================================================================

class LoginRequest(BaseModel):
    """Login request."""
    email: str
    password: str


class RegisterRequest(BaseModel):
    """Registration request."""
    email: str
    password: str
    name: str


class TokenRefreshRequest(BaseModel):
    """Token refresh request."""
    refresh_token: str


class CreateAPIKeyRequest(BaseModel):
    """API key creation request."""
    name: str = Field(..., min_length=1, max_length=100)
    scopes: List[str] = Field(default_factory=list)
    expires_days: Optional[int] = Field(default=None, ge=1, le=365)


class APIKeyResponse(BaseModel):
    """API key response (shown only once)."""
    key: str
    key_id: str
    name: str
    scopes: List[str]
    expires_at: Optional[datetime]


class APIKeyInfo(BaseModel):
    """API key info (without the actual key)."""
    key_id: str
    name: str
    scopes: List[str]
    created_at: datetime
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]
    is_active: bool


class ChangePasswordRequest(BaseModel):
    """Password change request."""
    current_password: str
    new_password: str


class UserInfo(BaseModel):
    """User information."""
    user_id: str
    email: str
    name: str
    scopes: List[str]
    created_at: datetime


# ==============================================================================
# Mock User Store (replace with actual database in production)
# ==============================================================================

# In-memory user store for development
# Note: Password hashes are computed lazily to avoid import-time issues
_users: Dict[str, Dict[str, Any]] = {}
_users_initialized = False


def _initialize_users():
    """Initialize mock users with hashed passwords."""
    global _users, _users_initialized
    if _users_initialized:
        return
    
    try:
        _users = {
            "user_001": {
                "id": "user_001",
                "email": "admin@neurectomy.local",
                "name": "Admin User",
                "password_hash": password_handler.hash_password("admin123"),
                "scopes": ["admin", "read", "write", "agents", "analytics"],
                "created_at": datetime.utcnow()
            },
            "user_002": {
                "id": "user_002",
                "email": "user@neurectomy.local",
                "name": "Regular User",
                "password_hash": password_handler.hash_password("user123"),
                "scopes": ["read", "agents"],
                "created_at": datetime.utcnow()
            }
        }
    except Exception:
        # Fallback for environments without bcrypt
        _users = {
            "user_001": {
                "id": "user_001",
                "email": "admin@neurectomy.local",
                "name": "Admin User",
                "password_hash": "fallback_hash_admin",
                "scopes": ["admin", "read", "write", "agents", "analytics"],
                "created_at": datetime.utcnow()
            },
            "user_002": {
                "id": "user_002",
                "email": "user@neurectomy.local",
                "name": "Regular User",
                "password_hash": "fallback_hash_user",
                "scopes": ["read", "agents"],
                "created_at": datetime.utcnow()
            }
        }
    _users_initialized = True


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Get user by email."""
    _initialize_users()
    for user in _users.values():
        if user["email"] == email:
            return user
    return None


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Get user by ID."""
    _initialize_users()
    return _users.get(user_id)


# ==============================================================================
# Authentication Endpoints
# ==============================================================================

@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    http_request: Request
):
    """
    Authenticate user and return tokens.
    
    Returns access and refresh tokens on successful authentication.
    """
    # Check lockout
    if login_tracker.is_locked(request.email):
        remaining = login_tracker.get_lockout_remaining(request.email)
        audit_logger.log(
            AuditEventType.LOGIN_FAILURE,
            request=http_request,
            details={"reason": "account_locked", "email": request.email}
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Account locked. Try again in {remaining} seconds."
        )
    
    # Find user
    user = get_user_by_email(request.email)
    
    if not user:
        login_tracker.record_failure(request.email)
        audit_logger.log(
            AuditEventType.LOGIN_FAILURE,
            request=http_request,
            details={"reason": "user_not_found", "email": request.email}
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Verify password
    if not password_handler.verify_password(request.password, user["password_hash"]):
        login_tracker.record_failure(request.email)
        audit_logger.log(
            AuditEventType.LOGIN_FAILURE,
            request=http_request,
            details={"reason": "invalid_password", "email": request.email}
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Successful login
    login_tracker.record_success(request.email)
    
    tokens = token_manager.create_token_pair(user["id"], user["scopes"])
    
    audit_logger.log(
        AuditEventType.LOGIN_SUCCESS,
        user_id=user["id"],
        request=http_request
    )
    
    logger.info("User logged in", user_id=user["id"])
    
    return tokens


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: TokenRefreshRequest,
    http_request: Request
):
    """
    Refresh access token using refresh token.
    """
    try:
        tokens = token_manager.refresh_access_token(request.refresh_token)
        
        audit_logger.log(
            AuditEventType.TOKEN_REFRESH,
            request=http_request
        )
        
        return tokens
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Token refresh failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )


@router.post("/logout")
async def logout(
    http_request: Request,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Logout user and revoke tokens.
    """
    # Get token from authorization header
    auth_header = http_request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        token_manager.revoke_token(token)
    
    audit_logger.log(
        AuditEventType.LOGOUT,
        user_id=user.get("user_id"),
        request=http_request
    )
    
    return {"success": True, "message": "Logged out successfully"}


@router.get("/me")
async def get_current_user_info(
    user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get current authenticated user info.
    """
    user_data = get_user_by_id(user["user_id"])
    
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return {
        "user_id": user_data["id"],
        "email": user_data["email"],
        "name": user_data["name"],
        "scopes": user_data["scopes"],
        "auth_type": user.get("auth_type", "jwt")
    }


@router.post("/change-password")
async def change_password(
    request: ChangePasswordRequest,
    http_request: Request,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Change user password.
    """
    user_data = get_user_by_id(user["user_id"])
    
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Verify current password
    if not password_handler.verify_password(request.current_password, user_data["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect"
        )
    
    # Validate new password
    errors = password_handler.validate_password_strength(request.new_password)
    if errors:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"errors": errors}
        )
    
    # Update password
    user_data["password_hash"] = password_handler.hash_password(request.new_password)
    
    logger.info("Password changed", user_id=user["user_id"])
    
    return {"success": True, "message": "Password changed successfully"}


# ==============================================================================
# API Key Management Endpoints
# ==============================================================================

@router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    request: CreateAPIKeyRequest,
    http_request: Request,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Create a new API key.
    
    Note: The full API key is only shown once. Store it securely.
    """
    # Validate scopes - user can only create keys with their own scopes
    user_scopes = set(user.get("scopes", []))
    requested_scopes = set(request.scopes)
    
    if "admin" not in user_scopes:
        invalid_scopes = requested_scopes - user_scopes
        if invalid_scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Cannot grant scopes you don't have: {', '.join(invalid_scopes)}"
            )
    
    raw_key, api_key = api_key_manager.generate_api_key(
        name=request.name,
        user_id=user["user_id"],
        scopes=request.scopes,
        expires_days=request.expires_days
    )
    
    audit_logger.log(
        AuditEventType.API_KEY_CREATED,
        user_id=user["user_id"],
        request=http_request,
        details={"key_id": api_key.key_id, "name": api_key.name}
    )
    
    return APIKeyResponse(
        key=raw_key,
        key_id=api_key.key_id,
        name=api_key.name,
        scopes=api_key.scopes,
        expires_at=api_key.expires_at
    )


@router.get("/api-keys", response_model=List[APIKeyInfo])
async def list_api_keys(
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    List all API keys for current user.
    """
    keys = api_key_manager.list_api_keys(user["user_id"])
    
    return [
        APIKeyInfo(
            key_id=key.key_id,
            name=key.name,
            scopes=key.scopes,
            created_at=key.created_at,
            expires_at=key.expires_at,
            last_used_at=key.last_used_at,
            is_active=key.is_active
        )
        for key in keys
    ]


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    http_request: Request,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Revoke an API key.
    """
    # Verify ownership
    keys = api_key_manager.list_api_keys(user["user_id"])
    key_ids = [k.key_id for k in keys]
    
    if key_id not in key_ids and "admin" not in user.get("scopes", []):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )
    
    success = api_key_manager.revoke_api_key(key_id)
    
    if success:
        audit_logger.log(
            AuditEventType.API_KEY_REVOKED,
            user_id=user["user_id"],
            request=http_request,
            details={"key_id": key_id}
        )
        return {"success": True, "message": "API key revoked"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )


# ==============================================================================
# Rate Limiting Info
# ==============================================================================

@router.get("/rate-limit")
async def get_rate_limit_info(
    request: Request,
    user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get current rate limit status.
    """
    identifier = user.get("user_id", request.client.host if request.client else "unknown")
    remaining = rate_limiter.get_remaining(identifier)
    
    return {
        "limit": rate_limiter.requests_per_window,
        "remaining": remaining,
        "window_seconds": rate_limiter.window_seconds
    }


# ==============================================================================
# Audit Log Access (Admin Only)
# ==============================================================================

@router.get("/audit-log")
async def get_audit_log(
    limit: int = 100,
    event_type: Optional[AuditEventType] = None,
    user: Dict[str, Any] = Depends(require_scopes("admin"))
) -> List[Dict[str, Any]]:
    """
    Get audit log entries.
    
    Admin only endpoint.
    """
    entries = audit_logger.get_entries(
        event_type=event_type,
        limit=limit
    )
    
    return [entry.model_dump() for entry in entries]


# ==============================================================================
# Health & Status
# ==============================================================================

@router.get("/status")
async def auth_status() -> Dict[str, Any]:
    """Get authentication service status."""
    return {
        "status": "healthy",
        "jwt_available": True,
        "api_keys_active": len([k for k in api_key_manager._keys.values() if k.is_active]),
        "rate_limiter_enabled": True
    }
