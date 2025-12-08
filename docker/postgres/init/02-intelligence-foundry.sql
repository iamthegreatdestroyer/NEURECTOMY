-- Intelligence Foundry Database Initialization
-- Creates databases and users for MLflow and Optuna

-- Create MLflow database and user
CREATE DATABASE mlflow;
CREATE USER mlflow WITH PASSWORD 'mlflow';
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;

-- Create Optuna database and user
CREATE DATABASE optuna;
CREATE USER optuna WITH PASSWORD 'optuna';
GRANT ALL PRIVILEGES ON DATABASE optuna TO optuna;

-- Connect to mlflow database and grant schema privileges
\c mlflow
GRANT ALL ON SCHEMA public TO mlflow;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO mlflow;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO mlflow;

-- Connect to optuna database and grant schema privileges
\c optuna
GRANT ALL ON SCHEMA public TO optuna;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO optuna;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO optuna;

-- Return to default database
\c postgres
