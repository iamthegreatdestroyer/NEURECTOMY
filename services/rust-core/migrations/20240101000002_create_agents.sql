-- NEURECTOMY: Agents Table Migration
-- @VERTEX Database Schema Design
-- Core agent entities and configurations

-- Agent status enum
CREATE TYPE agent_status AS ENUM (
    'draft',
    'configuring',
    'training',
    'trained',
    'deploying',
    'active',
    'paused',
    'failed',
    'archived',
    'deleted'
);

-- Agent type enum
CREATE TYPE agent_type AS ENUM (
    'autonomous',
    'assistant',
    'specialist',
    'orchestrator',
    'retrieval',
    'code_generator',
    'custom'
);

-- Model provider enum
CREATE TYPE model_provider AS ENUM (
    'openai',
    'anthropic',
    'ollama',
    'huggingface',
    'azure_openai',
    'cohere',
    'mistral',
    'groq',
    'custom'
);

-- Agents table
CREATE TABLE agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Identity
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(255) NOT NULL,
    description TEXT,
    avatar_url TEXT,
    tags JSONB NOT NULL DEFAULT '[]'::jsonb,
    
    -- Type & Classification
    agent_type agent_type NOT NULL DEFAULT 'assistant',
    category VARCHAR(100),
    
    -- Model Configuration
    model_provider model_provider NOT NULL DEFAULT 'ollama',
    model_name VARCHAR(255) NOT NULL DEFAULT 'llama3.2',
    model_config JSONB NOT NULL DEFAULT '{
        "temperature": 0.7,
        "max_tokens": 4096,
        "top_p": 0.95,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop_sequences": []
    }'::jsonb,
    
    -- System Prompt & Behavior
    system_prompt TEXT NOT NULL DEFAULT 'You are a helpful AI assistant.',
    persona JSONB DEFAULT NULL, -- Extended persona configuration
    
    -- Capabilities
    capabilities JSONB NOT NULL DEFAULT '{
        "web_search": false,
        "code_execution": false,
        "file_access": false,
        "image_generation": false,
        "voice_synthesis": false,
        "tool_use": true,
        "memory": true,
        "rag": false
    }'::jsonb,
    
    -- Tools & Functions
    tools JSONB NOT NULL DEFAULT '[]'::jsonb,
    mcp_servers JSONB NOT NULL DEFAULT '[]'::jsonb,
    
    -- Memory & Context
    memory_config JSONB NOT NULL DEFAULT '{
        "type": "sliding_window",
        "max_messages": 50,
        "max_tokens": 8000,
        "summarization": true,
        "vector_memory": false
    }'::jsonb,
    
    -- RAG Configuration (if enabled)
    rag_config JSONB DEFAULT NULL,
    
    -- Status
    status agent_status NOT NULL DEFAULT 'draft',
    status_message TEXT,
    
    -- Metrics
    metrics JSONB NOT NULL DEFAULT '{
        "total_conversations": 0,
        "total_messages": 0,
        "total_tokens": 0,
        "avg_response_time_ms": 0,
        "success_rate": 0,
        "user_rating": 0
    }'::jsonb,
    
    -- Version Control
    version INTEGER NOT NULL DEFAULT 1,
    parent_version_id UUID REFERENCES agents(id),
    is_published BOOLEAN NOT NULL DEFAULT FALSE,
    published_at TIMESTAMPTZ,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_active_at TIMESTAMPTZ,
    deleted_at TIMESTAMPTZ,
    
    -- Constraints
    CONSTRAINT unique_user_agent_slug UNIQUE (user_id, slug)
);

-- Agent versions for history
CREATE TABLE agent_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    
    version INTEGER NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- Snapshot of agent config at this version
    config_snapshot JSONB NOT NULL,
    
    -- Change metadata
    change_summary TEXT,
    changed_by UUID REFERENCES users(id),
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT unique_agent_version UNIQUE (agent_id, version)
);

-- Agent tools registry
CREATE TABLE agent_tools (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Identity
    name VARCHAR(255) NOT NULL UNIQUE,
    display_name VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    
    -- Type & Category
    tool_type VARCHAR(100) NOT NULL, -- 'builtin', 'mcp', 'custom', 'api'
    category VARCHAR(100),
    
    -- Schema
    input_schema JSONB NOT NULL,
    output_schema JSONB,
    
    -- Implementation
    implementation JSONB NOT NULL, -- How to execute the tool
    
    -- Security
    requires_approval BOOLEAN NOT NULL DEFAULT FALSE,
    required_permissions JSONB NOT NULL DEFAULT '[]'::jsonb,
    
    -- Status
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Agent-Tool relationships (many-to-many)
CREATE TABLE agent_tool_assignments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    tool_id UUID NOT NULL REFERENCES agent_tools(id) ON DELETE CASCADE,
    
    -- Configuration overrides
    config_override JSONB DEFAULT NULL,
    is_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT unique_agent_tool UNIQUE (agent_id, tool_id)
);

-- Agent knowledge bases (for RAG)
CREATE TABLE agent_knowledge_bases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    
    name VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- Source type
    source_type VARCHAR(100) NOT NULL, -- 'file', 'url', 'github', 'notion', 'confluence'
    source_config JSONB NOT NULL,
    
    -- Processing status
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    processing_progress JSONB DEFAULT NULL,
    
    -- Vector store info
    vector_store VARCHAR(100) NOT NULL DEFAULT 'pinecone',
    vector_index_name VARCHAR(255),
    document_count INTEGER NOT NULL DEFAULT 0,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_synced_at TIMESTAMPTZ
);

-- Indexes for agents
CREATE INDEX idx_agents_user_id ON agents(user_id);
CREATE INDEX idx_agents_status ON agents(status);
CREATE INDEX idx_agents_type ON agents(agent_type);
CREATE INDEX idx_agents_model_provider ON agents(model_provider);
CREATE INDEX idx_agents_is_published ON agents(is_published);
CREATE INDEX idx_agents_created_at ON agents(created_at DESC);
CREATE INDEX idx_agents_tags ON agents USING GIN(tags);

-- Indexes for agent versions
CREATE INDEX idx_agent_versions_agent_id ON agent_versions(agent_id);
CREATE INDEX idx_agent_versions_version ON agent_versions(version DESC);

-- Indexes for agent tools
CREATE INDEX idx_agent_tools_type ON agent_tools(tool_type);
CREATE INDEX idx_agent_tools_category ON agent_tools(category);
CREATE INDEX idx_agent_tools_is_active ON agent_tools(is_active);

-- Indexes for agent-tool assignments
CREATE INDEX idx_agent_tool_assignments_agent_id ON agent_tool_assignments(agent_id);
CREATE INDEX idx_agent_tool_assignments_tool_id ON agent_tool_assignments(tool_id);

-- Indexes for knowledge bases
CREATE INDEX idx_agent_knowledge_bases_agent_id ON agent_knowledge_bases(agent_id);
CREATE INDEX idx_agent_knowledge_bases_status ON agent_knowledge_bases(status);

-- Triggers
CREATE TRIGGER update_agents_updated_at
    BEFORE UPDATE ON agents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agent_tools_updated_at
    BEFORE UPDATE ON agent_tools
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agent_knowledge_bases_updated_at
    BEFORE UPDATE ON agent_knowledge_bases
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Comments
COMMENT ON TABLE agents IS 'AI agents created and managed by users';
COMMENT ON TABLE agent_versions IS 'Version history for agents';
COMMENT ON TABLE agent_tools IS 'Registry of available tools for agents';
COMMENT ON TABLE agent_tool_assignments IS 'Tool assignments to specific agents';
COMMENT ON TABLE agent_knowledge_bases IS 'RAG knowledge bases for agents';
