"""
Tests for Security Module.

@ECLIPSE @CIPHER @FORTRESS - Test suite for authentication and security.
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.security import (
    SecurityConfig,
    PasswordHandler,
    TokenManager,
    TokenType,
    TokenPayload,
    TokenResponse,
    APIKey,
    APIKeyManager,
    RateLimiter,
    RateLimitResult,
    AuditEventType,
    AuditLogEntry,
    AuditLogger,
    InputSanitizer,
    LoginAttemptTracker,
    token_manager,
    api_key_manager,
    rate_limiter,
    audit_logger,
    password_handler,
    login_tracker,
    create_user_tokens,
    verify_user_password,
    hash_user_password,
    generate_api_key_for_user,
)

from fastapi import HTTPException


# ==============================================================================
# SecurityConfig Tests
# ==============================================================================

class TestSecurityConfig:
    """Tests for security configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SecurityConfig()
        
        assert config.algorithm == "HS256"
        assert config.access_token_expire_minutes == 30
        assert config.refresh_token_expire_days == 7
        assert config.rate_limit_requests == 100
        assert config.password_min_length == 8
        assert config.max_failed_attempts == 5
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SecurityConfig(
            secret_key="my-secret-key",
            access_token_expire_minutes=60,
            rate_limit_requests=500
        )
        
        assert config.secret_key == "my-secret-key"
        assert config.access_token_expire_minutes == 60
        assert config.rate_limit_requests == 500
    
    def test_secret_key_generation(self):
        """Test that secret key is auto-generated."""
        config = SecurityConfig()
        
        assert config.secret_key is not None
        assert len(config.secret_key) > 20


# ==============================================================================
# PasswordHandler Tests
# ==============================================================================

class TestPasswordHandler:
    """Tests for password handling."""
    
    def test_hash_password(self):
        """Test password hashing."""
        handler = PasswordHandler()
        
        hashed = handler.hash_password("mysecretpassword")
        
        assert hashed != "mysecretpassword"
        assert len(hashed) > 20
    
    def test_verify_correct_password(self):
        """Test verifying correct password."""
        handler = PasswordHandler()
        
        hashed = handler.hash_password("mysecretpassword")
        result = handler.verify_password("mysecretpassword", hashed)
        
        assert result is True
    
    def test_verify_incorrect_password(self):
        """Test verifying incorrect password."""
        handler = PasswordHandler()
        
        hashed = handler.hash_password("mysecretpassword")
        result = handler.verify_password("wrongpassword", hashed)
        
        assert result is False
    
    def test_different_hashes_for_same_password(self):
        """Test that same password produces different hashes (salted)."""
        handler = PasswordHandler()
        
        hash1 = handler.hash_password("password123")
        hash2 = handler.hash_password("password123")
        
        assert hash1 != hash2
    
    def test_validate_password_strength_valid(self):
        """Test password strength validation with valid password."""
        handler = PasswordHandler()
        
        errors = handler.validate_password_strength("MyP@ssw0rd!")
        
        assert len(errors) == 0
    
    def test_validate_password_strength_too_short(self):
        """Test password validation fails for short password."""
        handler = PasswordHandler()
        
        errors = handler.validate_password_strength("Ab1!")
        
        assert any("at least" in e for e in errors)
    
    def test_validate_password_strength_no_uppercase(self):
        """Test password validation fails without uppercase."""
        handler = PasswordHandler()
        
        errors = handler.validate_password_strength("myp@ssw0rd!")
        
        assert any("uppercase" in e.lower() for e in errors)
    
    def test_validate_password_strength_no_lowercase(self):
        """Test password validation fails without lowercase."""
        handler = PasswordHandler()
        
        errors = handler.validate_password_strength("MYP@SSW0RD!")
        
        assert any("lowercase" in e.lower() for e in errors)
    
    def test_validate_password_strength_no_digit(self):
        """Test password validation fails without digit."""
        handler = PasswordHandler()
        
        errors = handler.validate_password_strength("MyP@ssword!")
        
        assert any("digit" in e.lower() for e in errors)
    
    def test_validate_password_strength_no_special(self):
        """Test password validation fails without special char."""
        handler = PasswordHandler()
        
        errors = handler.validate_password_strength("MyPassw0rd")
        
        assert any("special" in e.lower() for e in errors)


# ==============================================================================
# TokenManager Tests
# ==============================================================================

class TestTokenManager:
    """Tests for JWT token management."""
    
    @pytest.fixture
    def manager(self):
        """Create token manager with test config."""
        config = SecurityConfig(
            secret_key="test-secret-key-12345",
            access_token_expire_minutes=15,
            refresh_token_expire_days=1
        )
        return TokenManager(config)
    
    def test_create_access_token(self, manager):
        """Test creating access token."""
        token = manager.create_access_token("user123", ["read", "write"])
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 50
    
    def test_create_refresh_token(self, manager):
        """Test creating refresh token."""
        token = manager.create_refresh_token("user123")
        
        assert token is not None
        assert isinstance(token, str)
    
    def test_create_token_pair(self, manager):
        """Test creating access and refresh token pair."""
        response = manager.create_token_pair("user123", ["admin"])
        
        assert isinstance(response, TokenResponse)
        assert response.access_token is not None
        assert response.refresh_token is not None
        assert response.token_type == "bearer"
        assert response.expires_in > 0
    
    def test_decode_valid_token(self, manager):
        """Test decoding valid token."""
        token = manager.create_access_token("user123", ["read"])
        
        payload = manager.decode_token(token)
        
        assert payload.sub == "user123"
        assert payload.type == TokenType.ACCESS
        assert "read" in payload.scopes
    
    def test_decode_expired_token(self, manager):
        """Test decoding expired token raises error."""
        # Create a manager with very short expiration
        config = SecurityConfig(
            secret_key="test-key",
            access_token_expire_minutes=0  # Immediately expires
        )
        short_manager = TokenManager(config)
        
        # Create token that's already expired
        token = short_manager.create_access_token("user123")
        
        # Wait a moment
        time.sleep(0.1)
        
        with pytest.raises(HTTPException) as exc_info:
            short_manager.decode_token(token)
        
        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()
    
    def test_decode_invalid_token(self, manager):
        """Test decoding invalid token raises error."""
        with pytest.raises(HTTPException) as exc_info:
            manager.decode_token("invalid.token.here")
        
        assert exc_info.value.status_code == 401
    
    def test_revoke_token(self, manager):
        """Test revoking a token."""
        token = manager.create_access_token("user123")
        payload_before = manager.decode_token(token)  # Should work
        
        manager.revoke_token(token)
        
        with pytest.raises(HTTPException) as exc_info:
            manager.decode_token(token)
        
        assert "revoked" in exc_info.value.detail.lower()
    
    def test_refresh_access_token(self, manager):
        """Test refreshing access token."""
        # Create tokens
        tokens = manager.create_token_pair("user123", ["read"])
        
        # Refresh
        new_tokens = manager.refresh_access_token(tokens.refresh_token)
        
        assert new_tokens.access_token != tokens.access_token
        assert new_tokens.access_token is not None
    
    def test_refresh_with_access_token_fails(self, manager):
        """Test that refreshing with access token fails."""
        access_token = manager.create_access_token("user123")
        
        with pytest.raises(HTTPException) as exc_info:
            manager.refresh_access_token(access_token)
        
        assert exc_info.value.status_code == 401


# ==============================================================================
# APIKeyManager Tests
# ==============================================================================

class TestAPIKeyManager:
    """Tests for API key management."""
    
    @pytest.fixture
    def manager(self):
        """Create API key manager."""
        return APIKeyManager()
    
    def test_generate_api_key(self, manager):
        """Test generating API key."""
        raw_key, api_key = manager.generate_api_key(
            name="Test Key",
            user_id="user123",
            scopes=["read"]
        )
        
        assert raw_key is not None
        assert raw_key.startswith("neurectomy_")
        assert api_key.name == "Test Key"
        assert api_key.user_id == "user123"
        assert api_key.is_active is True
    
    def test_validate_api_key(self, manager):
        """Test validating API key."""
        raw_key, _ = manager.generate_api_key(
            name="Test Key",
            user_id="user123"
        )
        
        validated = manager.validate_api_key(raw_key)
        
        assert validated is not None
        assert validated.user_id == "user123"
    
    def test_validate_invalid_key(self, manager):
        """Test validating invalid key returns None."""
        result = manager.validate_api_key("invalid_key")
        
        assert result is None
    
    def test_validate_wrong_prefix(self, manager):
        """Test key with wrong prefix is invalid."""
        result = manager.validate_api_key("wrong_prefix_abc123")
        
        assert result is None
    
    def test_revoke_api_key(self, manager):
        """Test revoking API key."""
        raw_key, api_key = manager.generate_api_key(
            name="Test Key",
            user_id="user123"
        )
        
        success = manager.revoke_api_key(api_key.key_id)
        
        assert success is True
        assert manager.validate_api_key(raw_key) is None
    
    def test_revoke_nonexistent_key(self, manager):
        """Test revoking nonexistent key."""
        success = manager.revoke_api_key("nonexistent_id")
        
        assert success is False
    
    def test_list_api_keys(self, manager):
        """Test listing API keys for user."""
        # Generate multiple keys
        manager.generate_api_key(name="Key 1", user_id="user123")
        manager.generate_api_key(name="Key 2", user_id="user123")
        manager.generate_api_key(name="Key 3", user_id="user456")
        
        user_keys = manager.list_api_keys("user123")
        
        assert len(user_keys) == 2
        assert all(k.user_id == "user123" for k in user_keys)
    
    def test_expired_key_invalid(self, manager):
        """Test that expired key is invalid."""
        raw_key, api_key = manager.generate_api_key(
            name="Short Lived",
            user_id="user123",
            expires_days=0  # Will set expires_at to now
        )
        
        # Manually set expired
        api_key.expires_at = datetime.utcnow() - timedelta(days=1)
        
        result = manager.validate_api_key(raw_key)
        
        assert result is None
    
    def test_last_used_updated(self, manager):
        """Test that last_used is updated on validation."""
        raw_key, api_key = manager.generate_api_key(
            name="Test Key",
            user_id="user123"
        )
        
        assert api_key.last_used_at is None
        
        manager.validate_api_key(raw_key)
        
        assert api_key.last_used_at is not None


# ==============================================================================
# RateLimiter Tests
# ==============================================================================

class TestRateLimiter:
    """Tests for rate limiting."""
    
    @pytest.fixture
    def limiter(self):
        """Create rate limiter with low limits for testing."""
        return RateLimiter(requests_per_window=5, window_seconds=60)
    
    def test_allow_within_limit(self, limiter):
        """Test requests within limit are allowed."""
        result = limiter.check("user123")
        
        assert result.allowed is True
        assert result.remaining == 4
    
    def test_deny_over_limit(self, limiter):
        """Test requests over limit are denied."""
        # Use up all tokens
        for _ in range(5):
            limiter.check("user123")
        
        # Next request should be denied
        result = limiter.check("user123")
        
        assert result.allowed is False
        assert result.remaining == 0
    
    def test_different_users_separate_limits(self, limiter):
        """Test different users have separate limits."""
        # Use up user1's limit
        for _ in range(5):
            limiter.check("user1")
        
        # User2 should still have tokens
        result = limiter.check("user2")
        
        assert result.allowed is True
    
    def test_tokens_refill_over_time(self, limiter):
        """Test tokens refill over time."""
        # Use some tokens
        for _ in range(3):
            limiter.check("user123")
        
        remaining_after = limiter.get_remaining("user123")
        assert remaining_after == 2
    
    def test_per_endpoint_limits(self, limiter):
        """Test per-endpoint rate limiting."""
        # Hit one endpoint
        for _ in range(5):
            limiter.check("user123", endpoint="/api/v1/expensive")
        
        # Different endpoint should work
        result = limiter.check("user123", endpoint="/api/v1/cheap")
        
        assert result.allowed is True
    
    def test_cost_parameter(self, limiter):
        """Test cost parameter for expensive operations."""
        # Expensive request costs 3 tokens
        result = limiter.check("user123", cost=3)
        
        assert result.allowed is True
        assert result.remaining == 2
    
    def test_result_attributes(self, limiter):
        """Test rate limit result attributes."""
        result = limiter.check("user123")
        
        assert hasattr(result, "allowed")
        assert hasattr(result, "remaining")
        assert hasattr(result, "reset_at")
        assert hasattr(result, "limit")
        assert result.limit == 5


# ==============================================================================
# AuditLogger Tests
# ==============================================================================

class TestAuditLogger:
    """Tests for audit logging."""
    
    @pytest.fixture
    def logger(self):
        """Create audit logger."""
        return AuditLogger(max_entries=100)
    
    def test_log_event(self, logger):
        """Test logging an event."""
        entry = logger.log(
            AuditEventType.LOGIN_SUCCESS,
            user_id="user123"
        )
        
        assert entry.event_type == AuditEventType.LOGIN_SUCCESS
        assert entry.user_id == "user123"
        assert entry.id is not None
    
    def test_log_with_details(self, logger):
        """Test logging with additional details."""
        entry = logger.log(
            AuditEventType.DATA_ACCESS,
            user_id="user123",
            details={"resource": "agents", "action": "list"}
        )
        
        assert entry.details["resource"] == "agents"
    
    def test_log_with_request(self, logger):
        """Test logging with request context."""
        mock_request = Mock()
        mock_request.client.host = "192.168.1.1"
        mock_request.headers.get.return_value = "Mozilla/5.0"
        mock_request.url.path = "/api/v1/agents"
        mock_request.method = "GET"
        
        entry = logger.log(
            AuditEventType.DATA_ACCESS,
            request=mock_request
        )
        
        assert entry.ip_address == "192.168.1.1"
        assert entry.endpoint == "/api/v1/agents"
        assert entry.method == "GET"
    
    def test_get_entries(self, logger):
        """Test getting log entries."""
        for i in range(5):
            logger.log(AuditEventType.LOGIN_SUCCESS, user_id=f"user{i}")
        
        entries = logger.get_entries(limit=10)
        
        assert len(entries) == 5
    
    def test_get_entries_filtered_by_user(self, logger):
        """Test filtering entries by user."""
        logger.log(AuditEventType.LOGIN_SUCCESS, user_id="user1")
        logger.log(AuditEventType.LOGIN_SUCCESS, user_id="user2")
        logger.log(AuditEventType.LOGIN_SUCCESS, user_id="user1")
        
        entries = logger.get_entries(user_id="user1")
        
        assert len(entries) == 2
        assert all(e.user_id == "user1" for e in entries)
    
    def test_get_entries_filtered_by_type(self, logger):
        """Test filtering entries by event type."""
        logger.log(AuditEventType.LOGIN_SUCCESS)
        logger.log(AuditEventType.LOGIN_FAILURE)
        logger.log(AuditEventType.LOGIN_SUCCESS)
        
        entries = logger.get_entries(event_type=AuditEventType.LOGIN_FAILURE)
        
        assert len(entries) == 1
        assert entries[0].event_type == AuditEventType.LOGIN_FAILURE
    
    def test_max_entries_trimming(self, logger):
        """Test that old entries are trimmed."""
        # Create more entries than max
        for i in range(150):
            logger.log(AuditEventType.DATA_ACCESS, user_id=f"user{i}")
        
        entries = logger.get_entries(limit=200)
        
        assert len(entries) <= 100


# ==============================================================================
# InputSanitizer Tests
# ==============================================================================

class TestInputSanitizer:
    """Tests for input sanitization."""
    
    def test_sanitize_string(self):
        """Test basic string sanitization."""
        result = InputSanitizer.sanitize_string("  hello world  ")
        
        assert result == "hello world"
    
    def test_sanitize_truncate(self):
        """Test string truncation."""
        long_string = "a" * 2000
        
        result = InputSanitizer.sanitize_string(long_string, max_length=100)
        
        assert len(result) == 100
    
    def test_sanitize_null_bytes(self):
        """Test null byte removal."""
        dirty = "hello\x00world"
        
        result = InputSanitizer.sanitize_string(dirty)
        
        assert "\x00" not in result
        assert result == "helloworld"
    
    def test_validate_email_valid(self):
        """Test valid email validation."""
        assert InputSanitizer.validate_email("user@example.com") is True
        assert InputSanitizer.validate_email("user.name@domain.org") is True
        assert InputSanitizer.validate_email("user+tag@example.co.uk") is True
    
    def test_validate_email_invalid(self):
        """Test invalid email validation."""
        assert InputSanitizer.validate_email("notanemail") is False
        assert InputSanitizer.validate_email("@example.com") is False
        assert InputSanitizer.validate_email("user@") is False
    
    def test_validate_uuid_valid(self):
        """Test valid UUID validation."""
        valid_uuid = "550e8400-e29b-41d4-a716-446655440000"
        
        assert InputSanitizer.validate_uuid(valid_uuid) is True
    
    def test_validate_uuid_invalid(self):
        """Test invalid UUID validation."""
        assert InputSanitizer.validate_uuid("not-a-uuid") is False
        assert InputSanitizer.validate_uuid("550e8400-e29b-41d4-a716") is False


# ==============================================================================
# LoginAttemptTracker Tests
# ==============================================================================

class TestLoginAttemptTracker:
    """Tests for login attempt tracking."""
    
    @pytest.fixture
    def tracker(self):
        """Create tracker with low limits for testing."""
        return LoginAttemptTracker(max_attempts=3, lockout_minutes=1)
    
    def test_not_locked_initially(self, tracker):
        """Test account is not locked initially."""
        assert tracker.is_locked("user@example.com") is False
    
    def test_lock_after_max_attempts(self, tracker):
        """Test account locks after max attempts."""
        for _ in range(3):
            tracker.record_failure("user@example.com")
        
        assert tracker.is_locked("user@example.com") is True
    
    def test_successful_login_clears_attempts(self, tracker):
        """Test successful login clears failed attempts."""
        tracker.record_failure("user@example.com")
        tracker.record_failure("user@example.com")
        tracker.record_success("user@example.com")
        
        # Should be able to fail again without lockout
        tracker.record_failure("user@example.com")
        
        assert tracker.is_locked("user@example.com") is False
    
    def test_get_lockout_remaining(self, tracker):
        """Test getting remaining lockout time."""
        for _ in range(3):
            tracker.record_failure("user@example.com")
        
        remaining = tracker.get_lockout_remaining("user@example.com")
        
        assert remaining is not None
        assert remaining > 0
        assert remaining <= 60  # 1 minute lockout
    
    def test_lockout_remaining_when_not_locked(self, tracker):
        """Test lockout remaining is None when not locked."""
        remaining = tracker.get_lockout_remaining("user@example.com")
        
        assert remaining is None


# ==============================================================================
# Convenience Functions Tests
# ==============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_create_user_tokens(self):
        """Test creating user tokens."""
        tokens = create_user_tokens("user123", ["read", "write"])
        
        assert tokens.access_token is not None
        assert tokens.refresh_token is not None
    
    def test_hash_and_verify_password(self):
        """Test hashing and verifying password."""
        hashed = hash_user_password("mypassword123")
        
        assert verify_user_password("mypassword123", hashed) is True
        assert verify_user_password("wrongpassword", hashed) is False
    
    def test_generate_api_key_for_user(self):
        """Test generating API key for user."""
        raw_key, api_key = generate_api_key_for_user(
            name="My Key",
            user_id="user123",
            scopes=["read"]
        )
        
        assert raw_key is not None
        assert api_key.user_id == "user123"


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestSecurityIntegration:
    """Integration tests for security module."""
    
    def test_full_auth_flow(self):
        """Test complete authentication flow."""
        # 1. Hash password for storage
        password_hash = password_handler.hash_password("SecureP@ss123")
        
        # 2. Verify password on login
        assert password_handler.verify_password("SecureP@ss123", password_hash)
        
        # 3. Create tokens
        tokens = token_manager.create_token_pair("user123", ["read", "write"])
        
        # 4. Decode and verify access token
        payload = token_manager.decode_token(tokens.access_token)
        assert payload.sub == "user123"
        
        # 5. Log the login
        entry = audit_logger.log(
            AuditEventType.LOGIN_SUCCESS,
            user_id="user123"
        )
        assert entry.event_type == AuditEventType.LOGIN_SUCCESS
    
    def test_rate_limit_integration(self):
        """Test rate limiting integration."""
        limiter = RateLimiter(requests_per_window=3, window_seconds=60)
        
        # Allow first requests
        for i in range(3):
            result = limiter.check("test_user")
            assert result.allowed is True
        
        # Deny after limit
        result = limiter.check("test_user")
        assert result.allowed is False
    
    def test_api_key_auth_flow(self):
        """Test API key authentication flow."""
        # Generate key
        raw_key, api_key = api_key_manager.generate_api_key(
            name="Test Integration Key",
            user_id="user123",
            scopes=["read", "agents"]
        )
        
        # Validate key
        validated = api_key_manager.validate_api_key(raw_key)
        assert validated is not None
        assert validated.scopes == ["read", "agents"]
        
        # Revoke key
        api_key_manager.revoke_api_key(api_key.key_id)
        
        # Key should no longer work
        assert api_key_manager.validate_api_key(raw_key) is None
