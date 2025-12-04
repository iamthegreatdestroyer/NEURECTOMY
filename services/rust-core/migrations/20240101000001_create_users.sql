-- NEURECTOMY: Users Table Migration
-- @VERTEX Database Schema Design
-- PostgreSQL with UUID v7 for time-ordered IDs

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Custom types for user roles and status
CREATE TYPE user_role AS ENUM (
    'admin',
    'developer',
    'researcher',
    'viewer',
    'system'
);

CREATE TYPE user_status AS ENUM (
    'active',
    'inactive',
    'suspended',
    'pending_verification',
    'deleted'
);

CREATE TYPE auth_provider AS ENUM (
    'local',
    'github',
    'google',
    'microsoft',
    'okta',
    'saml'
);

-- Users table - core identity
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Identity
    email VARCHAR(255) NOT NULL UNIQUE,
    username VARCHAR(100) NOT NULL UNIQUE,
    display_name VARCHAR(255),
    avatar_url TEXT,
    
    -- Authentication
    password_hash VARCHAR(255), -- Argon2id hash, NULL for SSO users
    auth_provider auth_provider NOT NULL DEFAULT 'local',
    external_id VARCHAR(255), -- ID from external auth provider
    email_verified BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Authorization
    role user_role NOT NULL DEFAULT 'developer',
    permissions JSONB NOT NULL DEFAULT '[]'::jsonb,
    
    -- Status
    status user_status NOT NULL DEFAULT 'pending_verification',
    
    -- Security
    mfa_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    mfa_secret VARCHAR(255),
    failed_login_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until TIMESTAMPTZ,
    last_login_at TIMESTAMPTZ,
    last_login_ip INET,
    
    -- Preferences
    preferences JSONB NOT NULL DEFAULT '{
        "theme": "system",
        "language": "en",
        "notifications": {
            "email": true,
            "push": true,
            "training_complete": true,
            "agent_errors": true
        },
        "editor": {
            "fontSize": 14,
            "tabSize": 2,
            "wordWrap": true
        }
    }'::jsonb,
    
    -- Quotas & Limits
    quota JSONB NOT NULL DEFAULT '{
        "max_agents": 50,
        "max_containers": 10,
        "max_training_hours": 100,
        "max_storage_gb": 50,
        "used_agents": 0,
        "used_containers": 0,
        "used_training_hours": 0,
        "used_storage_gb": 0
    }'::jsonb,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMPTZ
);

-- User sessions table
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Session info
    token_hash VARCHAR(255) NOT NULL UNIQUE,
    refresh_token_hash VARCHAR(255),
    
    -- Device/Client info
    device_id VARCHAR(255),
    device_name VARCHAR(255),
    user_agent TEXT,
    ip_address INET,
    
    -- Security
    is_revoked BOOLEAN NOT NULL DEFAULT FALSE,
    revoked_at TIMESTAMPTZ,
    revoked_reason VARCHAR(255),
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    last_active_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Refresh tokens for token rotation
CREATE TABLE refresh_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id UUID NOT NULL REFERENCES user_sessions(id) ON DELETE CASCADE,
    
    token_hash VARCHAR(255) NOT NULL UNIQUE,
    token_family UUID NOT NULL, -- For detecting token reuse attacks
    
    is_revoked BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL
);

-- API Keys for programmatic access
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    name VARCHAR(255) NOT NULL,
    description TEXT,
    key_hash VARCHAR(255) NOT NULL UNIQUE,
    key_prefix VARCHAR(12) NOT NULL, -- First 12 chars for identification (neurec_xxx)
    
    -- Permissions
    permissions JSONB NOT NULL DEFAULT '["read"]'::jsonb,
    scopes JSONB NOT NULL DEFAULT '["agents:read", "containers:read"]'::jsonb,
    
    -- Rate limiting
    rate_limit INTEGER NOT NULL DEFAULT 1000, -- requests per hour
    
    -- Security
    allowed_ips JSONB DEFAULT NULL, -- NULL = all IPs allowed
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    last_used_at TIMESTAMPTZ,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ
);

-- Indexes for users
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_status ON users(status);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_auth_provider ON users(auth_provider);
CREATE INDEX idx_users_created_at ON users(created_at DESC);

-- Indexes for sessions
CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_token_hash ON user_sessions(token_hash);
CREATE INDEX idx_user_sessions_expires_at ON user_sessions(expires_at);
CREATE INDEX idx_user_sessions_is_revoked ON user_sessions(is_revoked);

-- Indexes for refresh tokens
CREATE INDEX idx_refresh_tokens_user_id ON refresh_tokens(user_id);
CREATE INDEX idx_refresh_tokens_session_id ON refresh_tokens(session_id);
CREATE INDEX idx_refresh_tokens_token_family ON refresh_tokens(token_family);
CREATE INDEX idx_refresh_tokens_expires_at ON refresh_tokens(expires_at);

-- Indexes for API keys
CREATE INDEX idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX idx_api_keys_key_prefix ON api_keys(key_prefix);
CREATE INDEX idx_api_keys_is_active ON api_keys(is_active);

-- Trigger for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Comments for documentation
COMMENT ON TABLE users IS 'Core user accounts for NEURECTOMY platform';
COMMENT ON TABLE user_sessions IS 'Active user sessions with device tracking';
COMMENT ON TABLE refresh_tokens IS 'JWT refresh tokens with rotation support';
COMMENT ON TABLE api_keys IS 'API keys for programmatic access';
