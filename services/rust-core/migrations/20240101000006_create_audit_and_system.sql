-- NEURECTOMY: Audit Logs & System Tables Migration
-- @VERTEX Database Schema Design
-- Comprehensive audit logging, notifications, and system settings

-- Audit action type enum
CREATE TYPE audit_action AS ENUM (
    'create',
    'read',
    'update',
    'delete',
    'login',
    'logout',
    'export',
    'import',
    'share',
    'revoke',
    'start',
    'stop',
    'execute',
    'other'
);

-- Audit resource type enum
CREATE TYPE audit_resource AS ENUM (
    'user',
    'agent',
    'container',
    'conversation',
    'training_job',
    'dataset',
    'api_key',
    'session',
    'system',
    'settings'
);

-- Notification type enum
CREATE TYPE notification_type AS ENUM (
    'info',
    'success',
    'warning',
    'error',
    'training_complete',
    'training_failed',
    'agent_error',
    'security_alert',
    'system_update',
    'quota_warning'
);

-- Audit logs table
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Who
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    user_email VARCHAR(255), -- Denormalized for when user is deleted
    session_id UUID,
    api_key_id UUID REFERENCES api_keys(id) ON DELETE SET NULL,
    
    -- What
    action audit_action NOT NULL,
    resource_type audit_resource NOT NULL,
    resource_id UUID,
    resource_name VARCHAR(255),
    
    -- Details
    description TEXT NOT NULL,
    old_values JSONB DEFAULT NULL,
    new_values JSONB DEFAULT NULL,
    changes JSONB DEFAULT NULL, -- Diff of old and new
    
    -- Context
    ip_address INET,
    user_agent TEXT,
    request_id VARCHAR(255),
    
    -- Outcome
    success BOOLEAN NOT NULL DEFAULT TRUE,
    error_message TEXT,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
) PARTITION BY RANGE (created_at);

-- Create partitions for audit logs (monthly)
CREATE TABLE audit_logs_current PARTITION OF audit_logs
    FOR VALUES FROM (DATE_TRUNC('month', CURRENT_DATE)) 
    TO (DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month');

CREATE TABLE audit_logs_archive PARTITION OF audit_logs
    FOR VALUES FROM (MINVALUE) 
    TO (DATE_TRUNC('month', CURRENT_DATE));

-- Notifications table
CREATE TABLE notifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Content
    type notification_type NOT NULL,
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    
    -- Link to related resource
    resource_type audit_resource,
    resource_id UUID,
    action_url TEXT,
    
    -- Status
    is_read BOOLEAN NOT NULL DEFAULT FALSE,
    read_at TIMESTAMPTZ,
    is_archived BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Delivery
    email_sent BOOLEAN NOT NULL DEFAULT FALSE,
    push_sent BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- System settings (key-value store)
CREATE TABLE system_settings (
    key VARCHAR(255) PRIMARY KEY,
    value JSONB NOT NULL,
    description TEXT,
    
    -- Validation
    value_type VARCHAR(50) NOT NULL DEFAULT 'string', -- 'string', 'number', 'boolean', 'json'
    validation_schema JSONB DEFAULT NULL,
    
    -- Access
    is_public BOOLEAN NOT NULL DEFAULT FALSE,
    requires_restart BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_by UUID REFERENCES users(id)
);

-- Feature flags
CREATE TABLE feature_flags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Identity
    key VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- State
    is_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Targeting
    target_type VARCHAR(50) NOT NULL DEFAULT 'all', -- 'all', 'percentage', 'users', 'roles'
    target_config JSONB NOT NULL DEFAULT '{}'::jsonb,
    
    -- Rollout
    rollout_percentage INTEGER NOT NULL DEFAULT 100 CHECK (rollout_percentage >= 0 AND rollout_percentage <= 100),
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    enabled_at TIMESTAMPTZ,
    disabled_at TIMESTAMPTZ
);

-- Scheduled jobs
CREATE TABLE scheduled_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Identity
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    
    -- Schedule (cron format)
    schedule VARCHAR(100) NOT NULL,
    timezone VARCHAR(100) NOT NULL DEFAULT 'UTC',
    
    -- Job config
    job_type VARCHAR(100) NOT NULL,
    job_config JSONB NOT NULL DEFAULT '{}'::jsonb,
    
    -- Status
    is_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    last_run_at TIMESTAMPTZ,
    last_run_status VARCHAR(50),
    last_run_error TEXT,
    next_run_at TIMESTAMPTZ,
    
    -- Concurrency
    max_retries INTEGER NOT NULL DEFAULT 3,
    retry_delay_seconds INTEGER NOT NULL DEFAULT 60,
    timeout_seconds INTEGER NOT NULL DEFAULT 300,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Job history
CREATE TABLE scheduled_job_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES scheduled_jobs(id) ON DELETE CASCADE,
    
    -- Execution
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    duration_ms INTEGER,
    
    -- Status
    status VARCHAR(50) NOT NULL DEFAULT 'running',
    error_message TEXT,
    
    -- Output
    output JSONB DEFAULT NULL
);

-- Indexes for audit logs
CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);
CREATE INDEX idx_audit_logs_resource ON audit_logs(resource_type, resource_id);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at DESC);
CREATE INDEX idx_audit_logs_success ON audit_logs(success);

-- Indexes for notifications
CREATE INDEX idx_notifications_user_id ON notifications(user_id);
CREATE INDEX idx_notifications_type ON notifications(type);
CREATE INDEX idx_notifications_is_read ON notifications(is_read);
CREATE INDEX idx_notifications_created_at ON notifications(created_at DESC);

-- Indexes for system settings
CREATE INDEX idx_system_settings_is_public ON system_settings(is_public);

-- Indexes for feature flags
CREATE INDEX idx_feature_flags_is_enabled ON feature_flags(is_enabled);

-- Indexes for scheduled jobs
CREATE INDEX idx_scheduled_jobs_is_enabled ON scheduled_jobs(is_enabled);
CREATE INDEX idx_scheduled_jobs_next_run ON scheduled_jobs(next_run_at);

-- Indexes for job history
CREATE INDEX idx_scheduled_job_history_job_id ON scheduled_job_history(job_id);
CREATE INDEX idx_scheduled_job_history_started_at ON scheduled_job_history(started_at DESC);

-- Triggers
CREATE TRIGGER update_system_settings_updated_at
    BEFORE UPDATE ON system_settings
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_feature_flags_updated_at
    BEFORE UPDATE ON feature_flags
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_scheduled_jobs_updated_at
    BEFORE UPDATE ON scheduled_jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Insert default system settings
INSERT INTO system_settings (key, value, description, value_type, is_public) VALUES
    ('app.name', '"NEURECTOMY"', 'Application name', 'string', true),
    ('app.version', '"0.1.0"', 'Application version', 'string', true),
    ('app.maintenance_mode', 'false', 'Enable maintenance mode', 'boolean', false),
    ('auth.session_timeout_minutes', '1440', 'Session timeout in minutes', 'number', false),
    ('auth.max_failed_logins', '5', 'Max failed login attempts before lockout', 'number', false),
    ('auth.lockout_duration_minutes', '30', 'Account lockout duration', 'number', false),
    ('agents.max_per_user', '50', 'Maximum agents per user', 'number', false),
    ('containers.max_per_user', '10', 'Maximum containers per user', 'number', false),
    ('training.max_concurrent_jobs', '2', 'Max concurrent training jobs', 'number', false),
    ('storage.max_per_user_gb', '50', 'Maximum storage per user in GB', 'number', false);

-- Insert default feature flags
INSERT INTO feature_flags (key, name, description, is_enabled) VALUES
    ('dark_mode', 'Dark Mode', 'Enable dark mode UI', true),
    ('beta_features', 'Beta Features', 'Enable beta features for testing', false),
    ('ai_code_review', 'AI Code Review', 'Enable AI-powered code review', true),
    ('vector_search', 'Vector Search', 'Enable semantic vector search', true),
    ('realtime_collab', 'Real-time Collaboration', 'Enable real-time collaboration features', false);

-- Comments
COMMENT ON TABLE audit_logs IS 'Comprehensive audit trail for all actions';
COMMENT ON TABLE notifications IS 'User notifications';
COMMENT ON TABLE system_settings IS 'Global system configuration';
COMMENT ON TABLE feature_flags IS 'Feature flag management';
COMMENT ON TABLE scheduled_jobs IS 'Scheduled background jobs';
COMMENT ON TABLE scheduled_job_history IS 'History of scheduled job executions';
