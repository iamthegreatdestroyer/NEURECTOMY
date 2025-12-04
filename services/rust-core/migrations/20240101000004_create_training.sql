-- NEURECTOMY: Training & Evaluation Tables Migration
-- @VERTEX Database Schema Design
-- Training jobs, datasets, and evaluation metrics

-- Training status enum
CREATE TYPE training_status AS ENUM (
    'queued',
    'preparing',
    'downloading_data',
    'preprocessing',
    'training',
    'validating',
    'saving',
    'completed',
    'failed',
    'cancelled',
    'paused'
);

-- Training method enum
CREATE TYPE training_method AS ENUM (
    'fine_tuning',
    'lora',
    'qlora',
    'full_fine_tuning',
    'rlhf',
    'dpo',
    'sft',
    'custom'
);

-- Dataset type enum
CREATE TYPE dataset_type AS ENUM (
    'instruction',
    'conversation',
    'completion',
    'preference',
    'classification',
    'custom'
);

-- Training jobs table
CREATE TABLE training_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    agent_id UUID REFERENCES agents(id) ON DELETE SET NULL,
    
    -- Identity
    name VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- Training Configuration
    method training_method NOT NULL DEFAULT 'lora',
    base_model VARCHAR(255) NOT NULL,
    base_model_provider model_provider NOT NULL,
    
    -- Hyperparameters
    hyperparameters JSONB NOT NULL DEFAULT '{
        "learning_rate": 0.0002,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "epochs": 3,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "lr_scheduler_type": "cosine",
        "optimizer": "adamw_8bit",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
    }'::jsonb,
    
    -- Dataset
    dataset_config JSONB NOT NULL DEFAULT '{
        "train_split": 0.9,
        "eval_split": 0.1,
        "max_seq_length": 2048,
        "shuffle": true
    }'::jsonb,
    
    -- Compute Resources
    resources JSONB NOT NULL DEFAULT '{
        "gpu_type": "A100",
        "gpu_count": 1,
        "cpu_cores": 8,
        "memory_gb": 32
    }'::jsonb,
    
    -- Status
    status training_status NOT NULL DEFAULT 'queued',
    status_message TEXT,
    
    -- Progress
    progress JSONB NOT NULL DEFAULT '{
        "current_epoch": 0,
        "total_epochs": 0,
        "current_step": 0,
        "total_steps": 0,
        "train_loss": null,
        "eval_loss": null,
        "eta_seconds": null
    }'::jsonb,
    
    -- Results
    results JSONB DEFAULT NULL,
    output_model_path TEXT,
    
    -- Metrics History
    metrics_history JSONB NOT NULL DEFAULT '[]'::jsonb,
    
    -- Execution
    container_id UUID REFERENCES containers(id),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    estimated_duration_seconds INTEGER,
    actual_duration_seconds INTEGER,
    
    -- Cost tracking
    cost_estimate JSONB DEFAULT NULL,
    actual_cost JSONB DEFAULT NULL,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    cancelled_at TIMESTAMPTZ
);

-- Datasets table
CREATE TABLE datasets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Identity
    name VARCHAR(255) NOT NULL,
    description TEXT,
    tags JSONB NOT NULL DEFAULT '[]'::jsonb,
    
    -- Type & Format
    dataset_type dataset_type NOT NULL,
    format VARCHAR(50) NOT NULL DEFAULT 'jsonl', -- 'jsonl', 'csv', 'parquet', 'huggingface'
    
    -- Source
    source_type VARCHAR(100) NOT NULL DEFAULT 'upload', -- 'upload', 'url', 'huggingface', 'generated'
    source_config JSONB DEFAULT NULL,
    
    -- Storage
    storage_path TEXT,
    file_size_bytes BIGINT,
    checksum VARCHAR(64),
    
    -- Statistics
    stats JSONB NOT NULL DEFAULT '{
        "total_examples": 0,
        "train_examples": 0,
        "eval_examples": 0,
        "avg_input_length": 0,
        "avg_output_length": 0,
        "total_tokens": 0
    }'::jsonb,
    
    -- Schema
    schema JSONB DEFAULT NULL,
    columns JSONB DEFAULT NULL,
    
    -- Processing
    processing_status VARCHAR(50) NOT NULL DEFAULT 'pending',
    processing_error TEXT,
    
    -- Visibility
    is_public BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at TIMESTAMPTZ
);

-- Training job datasets (many-to-many)
CREATE TABLE training_job_datasets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    training_job_id UUID NOT NULL REFERENCES training_jobs(id) ON DELETE CASCADE,
    dataset_id UUID NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    
    role VARCHAR(50) NOT NULL DEFAULT 'train', -- 'train', 'eval', 'test'
    sample_count INTEGER,
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT unique_job_dataset_role UNIQUE (training_job_id, dataset_id, role)
);

-- Evaluation runs
CREATE TABLE evaluation_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    agent_id UUID REFERENCES agents(id) ON DELETE SET NULL,
    training_job_id UUID REFERENCES training_jobs(id) ON DELETE SET NULL,
    
    -- Identity
    name VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- Configuration
    eval_config JSONB NOT NULL DEFAULT '{
        "metrics": ["accuracy", "f1", "bleu", "rouge"],
        "sample_size": 100,
        "temperature": 0.1
    }'::jsonb,
    
    -- Test dataset
    dataset_id UUID REFERENCES datasets(id),
    
    -- Status
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    status_message TEXT,
    
    -- Results
    results JSONB DEFAULT NULL,
    detailed_results JSONB DEFAULT NULL,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

-- Training checkpoints
CREATE TABLE training_checkpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    training_job_id UUID NOT NULL REFERENCES training_jobs(id) ON DELETE CASCADE,
    
    -- Checkpoint info
    step INTEGER NOT NULL,
    epoch FLOAT NOT NULL,
    
    -- Storage
    checkpoint_path TEXT NOT NULL,
    file_size_bytes BIGINT,
    
    -- Metrics at checkpoint
    metrics JSONB NOT NULL,
    
    -- Is this the best checkpoint?
    is_best BOOLEAN NOT NULL DEFAULT FALSE,
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for training jobs
CREATE INDEX idx_training_jobs_user_id ON training_jobs(user_id);
CREATE INDEX idx_training_jobs_agent_id ON training_jobs(agent_id);
CREATE INDEX idx_training_jobs_status ON training_jobs(status);
CREATE INDEX idx_training_jobs_method ON training_jobs(method);
CREATE INDEX idx_training_jobs_created_at ON training_jobs(created_at DESC);

-- Indexes for datasets
CREATE INDEX idx_datasets_user_id ON datasets(user_id);
CREATE INDEX idx_datasets_type ON datasets(dataset_type);
CREATE INDEX idx_datasets_is_public ON datasets(is_public);
CREATE INDEX idx_datasets_tags ON datasets USING GIN(tags);

-- Indexes for job datasets
CREATE INDEX idx_training_job_datasets_job ON training_job_datasets(training_job_id);
CREATE INDEX idx_training_job_datasets_dataset ON training_job_datasets(dataset_id);

-- Indexes for evaluation runs
CREATE INDEX idx_evaluation_runs_user_id ON evaluation_runs(user_id);
CREATE INDEX idx_evaluation_runs_agent_id ON evaluation_runs(agent_id);
CREATE INDEX idx_evaluation_runs_training_job_id ON evaluation_runs(training_job_id);

-- Indexes for checkpoints
CREATE INDEX idx_training_checkpoints_job ON training_checkpoints(training_job_id);
CREATE INDEX idx_training_checkpoints_step ON training_checkpoints(step);

-- Triggers
CREATE TRIGGER update_training_jobs_updated_at
    BEFORE UPDATE ON training_jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_datasets_updated_at
    BEFORE UPDATE ON datasets
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Comments
COMMENT ON TABLE training_jobs IS 'ML training jobs for agent fine-tuning';
COMMENT ON TABLE datasets IS 'Training and evaluation datasets';
COMMENT ON TABLE training_job_datasets IS 'Dataset assignments to training jobs';
COMMENT ON TABLE evaluation_runs IS 'Agent evaluation runs';
COMMENT ON TABLE training_checkpoints IS 'Training checkpoint storage';
