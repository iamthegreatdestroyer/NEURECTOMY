-- NEURECTOMY: Containers Table Migration
-- @VERTEX Database Schema Design
-- Container management for isolated agent execution environments

-- Container status enum
CREATE TYPE container_status AS ENUM (
    'pending',
    'creating',
    'starting',
    'running',
    'paused',
    'stopping',
    'stopped',
    'restarting',
    'failed',
    'terminated',
    'deleted'
);

-- Container type enum
CREATE TYPE container_type AS ENUM (
    'agent_runtime',
    'training_env',
    'evaluation_env',
    'sandbox',
    'custom'
);

-- Containers table
CREATE TABLE containers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    agent_id UUID REFERENCES agents(id) ON DELETE SET NULL,
    
    -- Identity
    name VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- Type
    container_type container_type NOT NULL DEFAULT 'agent_runtime',
    
    -- Docker/Container Info
    docker_id VARCHAR(100), -- Docker container ID
    image VARCHAR(500) NOT NULL DEFAULT 'neurectomy/agent-runtime:latest',
    image_digest VARCHAR(255),
    
    -- Resource Limits
    resources JSONB NOT NULL DEFAULT '{
        "cpu_limit": "2",
        "memory_limit": "4Gi",
        "gpu_request": "0",
        "ephemeral_storage": "10Gi",
        "network_bandwidth": "100Mbps"
    }'::jsonb,
    
    -- Resource Usage (updated periodically)
    resource_usage JSONB NOT NULL DEFAULT '{
        "cpu_percent": 0,
        "memory_bytes": 0,
        "memory_percent": 0,
        "disk_read_bytes": 0,
        "disk_write_bytes": 0,
        "network_rx_bytes": 0,
        "network_tx_bytes": 0
    }'::jsonb,
    
    -- Environment
    environment JSONB NOT NULL DEFAULT '{}'::jsonb, -- Environment variables
    secrets JSONB DEFAULT NULL, -- Encrypted secrets
    volumes JSONB NOT NULL DEFAULT '[]'::jsonb, -- Volume mounts
    
    -- Network
    ports JSONB NOT NULL DEFAULT '[]'::jsonb, -- Port mappings
    network_mode VARCHAR(100) DEFAULT 'bridge',
    ip_address INET,
    
    -- Status
    status container_status NOT NULL DEFAULT 'pending',
    status_message TEXT,
    exit_code INTEGER,
    
    -- Health
    health_status VARCHAR(50) DEFAULT 'unknown',
    health_check_config JSONB DEFAULT '{
        "enabled": true,
        "interval_seconds": 30,
        "timeout_seconds": 5,
        "retries": 3,
        "start_period_seconds": 60
    }'::jsonb,
    last_health_check_at TIMESTAMPTZ,
    
    -- Metrics
    metrics JSONB NOT NULL DEFAULT '{
        "uptime_seconds": 0,
        "restart_count": 0,
        "total_requests": 0,
        "avg_response_time_ms": 0
    }'::jsonb,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    stopped_at TIMESTAMPTZ,
    deleted_at TIMESTAMPTZ
);

-- Container logs (append-only for performance)
CREATE TABLE container_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    container_id UUID NOT NULL REFERENCES containers(id) ON DELETE CASCADE,
    
    -- Log entry
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    stream VARCHAR(10) NOT NULL DEFAULT 'stdout', -- 'stdout', 'stderr'
    level VARCHAR(20) NOT NULL DEFAULT 'info', -- 'debug', 'info', 'warn', 'error'
    message TEXT NOT NULL,
    
    -- Context
    context JSONB DEFAULT NULL,
    
    -- Partitioning support
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
) PARTITION BY RANGE (created_at);

-- Create partitions for container logs (last 30 days + current)
CREATE TABLE container_logs_current PARTITION OF container_logs
    FOR VALUES FROM (CURRENT_DATE) TO (CURRENT_DATE + INTERVAL '1 day');

CREATE TABLE container_logs_archive PARTITION OF container_logs
    FOR VALUES FROM (MINVALUE) TO (CURRENT_DATE);

-- Container events for audit
CREATE TABLE container_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    container_id UUID NOT NULL REFERENCES containers(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id),
    
    -- Event
    event_type VARCHAR(100) NOT NULL, -- 'create', 'start', 'stop', 'restart', 'delete', etc.
    event_data JSONB DEFAULT NULL,
    
    -- Source
    source VARCHAR(100) NOT NULL DEFAULT 'user', -- 'user', 'system', 'scheduler', 'health_check'
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Container templates for quick setup
CREATE TABLE container_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Identity
    name VARCHAR(255) NOT NULL UNIQUE,
    display_name VARCHAR(255) NOT NULL,
    description TEXT,
    icon VARCHAR(100),
    
    -- Template config
    container_type container_type NOT NULL,
    image VARCHAR(500) NOT NULL,
    default_resources JSONB NOT NULL,
    default_environment JSONB NOT NULL DEFAULT '{}'::jsonb,
    required_volumes JSONB NOT NULL DEFAULT '[]'::jsonb,
    
    -- Visibility
    is_public BOOLEAN NOT NULL DEFAULT TRUE,
    created_by UUID REFERENCES users(id),
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for containers
CREATE INDEX idx_containers_user_id ON containers(user_id);
CREATE INDEX idx_containers_agent_id ON containers(agent_id);
CREATE INDEX idx_containers_status ON containers(status);
CREATE INDEX idx_containers_type ON containers(container_type);
CREATE INDEX idx_containers_created_at ON containers(created_at DESC);
CREATE INDEX idx_containers_docker_id ON containers(docker_id);

-- Indexes for container logs
CREATE INDEX idx_container_logs_container_id ON container_logs(container_id);
CREATE INDEX idx_container_logs_timestamp ON container_logs(timestamp DESC);
CREATE INDEX idx_container_logs_level ON container_logs(level);

-- Indexes for container events
CREATE INDEX idx_container_events_container_id ON container_events(container_id);
CREATE INDEX idx_container_events_type ON container_events(event_type);
CREATE INDEX idx_container_events_created_at ON container_events(created_at DESC);

-- Triggers
CREATE TRIGGER update_containers_updated_at
    BEFORE UPDATE ON containers
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_container_templates_updated_at
    BEFORE UPDATE ON container_templates
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Comments
COMMENT ON TABLE containers IS 'Docker containers for agent execution';
COMMENT ON TABLE container_logs IS 'Partitioned container log storage';
COMMENT ON TABLE container_events IS 'Container lifecycle events for audit';
COMMENT ON TABLE container_templates IS 'Reusable container configurations';
