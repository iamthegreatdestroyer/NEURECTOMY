-- NEURECTOMY Database Initialization Script
-- Creates all required databases and extensions for Phase 2

-- Create MLflow database
CREATE DATABASE mlflow;

-- Create Optuna database
CREATE DATABASE optuna;

-- Connect to main database and enable extensions
\c neurectomy

-- Enable vector extension for pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable pg_trgm for text search
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Connect to MLflow database and set up
\c mlflow

-- Enable UUID extension for MLflow
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Connect to Optuna database and set up
\c optuna

-- Enable UUID extension for Optuna
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE mlflow TO neurectomy;
GRANT ALL PRIVILEGES ON DATABASE optuna TO neurectomy;
