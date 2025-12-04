-- NEURECTOMY: Conversations & Messages Tables Migration
-- @VERTEX Database Schema Design
-- Chat history, conversations, and message storage

-- Conversation status enum
CREATE TYPE conversation_status AS ENUM (
    'active',
    'archived',
    'deleted'
);

-- Message role enum
CREATE TYPE message_role AS ENUM (
    'system',
    'user',
    'assistant',
    'tool',
    'function'
);

-- Conversations table
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    
    -- Identity
    title VARCHAR(500),
    summary TEXT,
    tags JSONB NOT NULL DEFAULT '[]'::jsonb,
    
    -- Status
    status conversation_status NOT NULL DEFAULT 'active',
    
    -- Context
    context JSONB DEFAULT NULL, -- Additional context passed to agent
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    
    -- Statistics
    message_count INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_message_at TIMESTAMPTZ,
    archived_at TIMESTAMPTZ,
    deleted_at TIMESTAMPTZ
);

-- Messages table
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    
    -- Message content
    role message_role NOT NULL,
    content TEXT NOT NULL,
    
    -- For tool/function calls
    tool_calls JSONB DEFAULT NULL,
    tool_call_id VARCHAR(255),
    function_name VARCHAR(255),
    
    -- Token counts
    token_count INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    
    -- Model info
    model VARCHAR(255),
    model_provider model_provider,
    
    -- Performance
    latency_ms INTEGER,
    
    -- Rating/Feedback
    user_rating INTEGER CHECK (user_rating >= 1 AND user_rating <= 5),
    user_feedback TEXT,
    
    -- Metadata
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    
    -- Position in conversation
    position INTEGER NOT NULL,
    
    -- Edit tracking
    is_edited BOOLEAN NOT NULL DEFAULT FALSE,
    original_content TEXT,
    edited_at TIMESTAMPTZ,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Message attachments (files, images)
CREATE TABLE message_attachments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id UUID NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    
    -- File info
    filename VARCHAR(500) NOT NULL,
    mime_type VARCHAR(255) NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    
    -- Storage
    storage_path TEXT NOT NULL,
    storage_provider VARCHAR(100) NOT NULL DEFAULT 'local', -- 'local', 's3', 'azure'
    
    -- Metadata
    metadata JSONB DEFAULT NULL,
    
    -- For images
    width INTEGER,
    height INTEGER,
    thumbnail_path TEXT,
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Conversation shares (for collaboration)
CREATE TABLE conversation_shares (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    
    -- Share type
    share_type VARCHAR(50) NOT NULL DEFAULT 'link', -- 'link', 'user', 'team'
    
    -- For user shares
    shared_with_user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    
    -- For link shares
    share_token VARCHAR(255) UNIQUE,
    is_public BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Permissions
    can_edit BOOLEAN NOT NULL DEFAULT FALSE,
    can_delete BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Expiration
    expires_at TIMESTAMPTZ,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by UUID NOT NULL REFERENCES users(id)
);

-- Conversation templates (pre-defined conversations)
CREATE TABLE conversation_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Identity
    name VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    tags JSONB NOT NULL DEFAULT '[]'::jsonb,
    
    -- Template content
    initial_messages JSONB NOT NULL DEFAULT '[]'::jsonb,
    system_prompt_override TEXT,
    context JSONB DEFAULT NULL,
    
    -- Usage
    use_count INTEGER NOT NULL DEFAULT 0,
    
    -- Visibility
    is_public BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for conversations
CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_conversations_agent_id ON conversations(agent_id);
CREATE INDEX idx_conversations_status ON conversations(status);
CREATE INDEX idx_conversations_created_at ON conversations(created_at DESC);
CREATE INDEX idx_conversations_last_message_at ON conversations(last_message_at DESC);
CREATE INDEX idx_conversations_tags ON conversations USING GIN(tags);

-- Indexes for messages
CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX idx_messages_role ON messages(role);
CREATE INDEX idx_messages_position ON messages(conversation_id, position);
CREATE INDEX idx_messages_created_at ON messages(created_at DESC);

-- Indexes for attachments
CREATE INDEX idx_message_attachments_message_id ON message_attachments(message_id);

-- Indexes for shares
CREATE INDEX idx_conversation_shares_conversation_id ON conversation_shares(conversation_id);
CREATE INDEX idx_conversation_shares_shared_with ON conversation_shares(shared_with_user_id);
CREATE INDEX idx_conversation_shares_token ON conversation_shares(share_token);

-- Indexes for templates
CREATE INDEX idx_conversation_templates_user_id ON conversation_templates(user_id);
CREATE INDEX idx_conversation_templates_is_public ON conversation_templates(is_public);

-- Triggers
CREATE TRIGGER update_conversations_updated_at
    BEFORE UPDATE ON conversations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_conversation_templates_updated_at
    BEFORE UPDATE ON conversation_templates
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function to update conversation stats when messages added
CREATE OR REPLACE FUNCTION update_conversation_stats()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE conversations
    SET 
        message_count = message_count + 1,
        total_tokens = total_tokens + COALESCE(NEW.token_count, 0),
        last_message_at = NEW.created_at,
        updated_at = NOW()
    WHERE id = NEW.conversation_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_conversation_stats
    AFTER INSERT ON messages
    FOR EACH ROW
    EXECUTE FUNCTION update_conversation_stats();

-- Comments
COMMENT ON TABLE conversations IS 'Chat conversations between users and agents';
COMMENT ON TABLE messages IS 'Individual messages in conversations';
COMMENT ON TABLE message_attachments IS 'File attachments for messages';
COMMENT ON TABLE conversation_shares IS 'Conversation sharing permissions';
COMMENT ON TABLE conversation_templates IS 'Pre-defined conversation starters';
