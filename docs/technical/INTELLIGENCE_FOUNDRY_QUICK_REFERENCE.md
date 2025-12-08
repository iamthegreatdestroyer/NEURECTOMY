# Intelligence Foundry - Quick Reference Guide

## Service URLs

| Service          | URL                        | Credentials           | Purpose                       |
| ---------------- | -------------------------- | --------------------- | ----------------------------- |
| ML Service API   | http://localhost:8002      | -                     | Intelligence Foundry REST API |
| ML Service Docs  | http://localhost:8002/docs | -                     | Swagger UI                    |
| MLflow UI        | http://localhost:5000      | -                     | Experiment tracking           |
| MinIO Console    | http://localhost:9001      | minioadmin/minioadmin | S3 storage management         |
| Optuna Dashboard | http://localhost:8085      | -                     | Hyperparameter optimization   |

## Docker Commands

### Start Services

```bash
# Start all Intelligence Foundry services
docker-compose up -d minio mlflow ml-service optuna-dashboard

# Start specific service
docker-compose up -d ml-service
```

### Stop Services

```bash
# Stop all
docker-compose down

# Stop specific service
docker-compose stop ml-service
```

### View Status

```bash
# Check all services
docker-compose ps

# Check specific services
docker-compose ps minio mlflow ml-service optuna-dashboard
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f ml-service

# Last 50 lines
docker-compose logs --tail=50 mlflow
```

### Restart Services

```bash
# Restart all
docker-compose restart

# Restart specific
docker-compose restart ml-service
```

### Rebuild Images

```bash
# Rebuild MLflow
docker-compose build mlflow

# Rebuild ml-service
docker-compose build ml-service

# Rebuild and restart
docker-compose up -d --build ml-service
```

## Health Checks

### ml-service

```bash
curl http://localhost:8002/health
```

**Expected Response:**

```json
{
  "service": "intelligence-foundry-ml",
  "status": "healthy",
  "timestamp": "2025-12-07T19:41:51.992358",
  "websocket_connections": 0,
  "mlflow": "healthy",
  "optuna": "healthy"
}
```

### MLflow

```bash
curl http://localhost:5000/health
```

### MinIO

```bash
curl http://localhost:9000/minio/health/live
```

## Common Tasks

### Create MLflow Experiment

```bash
curl -X POST http://localhost:8002/api/mlflow/experiments/create \
  -H "Content-Type: application/json" \
  -d '{"name": "my-experiment", "artifact_location": "s3://mlflow-artifacts/my-experiment"}'
```

### List MLflow Experiments

```bash
curl http://localhost:8002/api/mlflow/experiments/list
```

### Create Optuna Study

```bash
curl -X POST http://localhost:8002/api/optuna/studies/create \
  -H "Content-Type: application/json" \
  -d '{"study_name": "my-study", "direction": "minimize"}'
```

### List Optuna Studies

```bash
curl http://localhost:8002/api/optuna/studies/list
```

### WebSocket Connection (JavaScript)

```javascript
const ws = new WebSocket("ws://localhost:8002/ws/intelligence-foundry");

ws.onopen = () => {
  console.log("Connected to Intelligence Foundry");
  ws.send(JSON.stringify({ type: "ping" }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log("Received:", data);
};
```

## Troubleshooting

### Service Won't Start

```bash
# Check dependencies
docker-compose ps postgres redis minio

# Check logs
docker-compose logs <service-name>

# Restart with fresh build
docker-compose down
docker-compose build <service-name>
docker-compose up -d <service-name>
```

### MLflow Unhealthy

```bash
# Check PostgreSQL connection
docker exec neurectomy-postgres psql -U mlflow -d mlflow -c "SELECT 1;"

# Check MinIO connection
docker exec neurectomy-mlflow curl http://minio:9000/minio/health/live

# View logs
docker-compose logs mlflow --tail=50
```

### Port Already in Use

```powershell
# Check what's using the port (Windows)
netstat -ano | findstr :8002

# Or use Docker
docker ps --format "table {{.Names}}\t{{.Ports}}" | Select-String "8002"
```

### Database Connection Issues

```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Connect to PostgreSQL
docker exec -it neurectomy-postgres psql -U neurectomy

# List databases
\l

# Connect to mlflow database
\c mlflow

# List tables
\dt
```

### Clear All Data (Nuclear Option)

```bash
# WARNING: This deletes all data!
docker-compose down -v
docker-compose up -d
```

## Environment Variables

### ml-service (.env)

```env
# Core
SERVICE_NAME=intelligence-foundry-ml
SERVICE_VERSION=1.0.0
HOST=0.0.0.0
PORT=8000
DEBUG=false

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_ARTIFACT_ROOT=s3://mlflow-artifacts/
MLFLOW_BACKEND_STORE_URI=postgresql://mlflow:mlflow@postgres:5432/mlflow

# Optuna
OPTUNA_STORAGE=postgresql://optuna:optuna@postgres:5432/optuna

# S3
S3_ENDPOINT_URL=http://minio:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin

# Training
DEFAULT_BATCH_SIZE=32
DEFAULT_EPOCHS=10
DEFAULT_LEARNING_RATE=0.001

# WebSocket
WS_HEARTBEAT_INTERVAL=30
WS_MAX_CONNECTIONS=100

# CORS
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

## Database Access

### PostgreSQL

```bash
# Connect as main user
docker exec -it neurectomy-postgres psql -U neurectomy

# Connect as mlflow user
docker exec -it neurectomy-postgres psql -U mlflow -d mlflow

# Connect as optuna user
docker exec -it neurectomy-postgres psql -U optuna -d optuna

# List all databases
docker exec neurectomy-postgres psql -U neurectomy -c "\l"

# List all users
docker exec neurectomy-postgres psql -U neurectomy -c "\du"
```

### Redis

```bash
# Connect to Redis
docker exec -it neurectomy-redis redis-cli

# Check connection
PING

# List all keys
KEYS *

# Get key value
GET <key>
```

## MinIO (S3) Operations

### mc Client (Inside Container)

```bash
# List buckets
docker exec neurectomy-minio mc ls myminio/

# List objects in bucket
docker exec neurectomy-minio mc ls myminio/mlflow-artifacts/

# Download object
docker exec neurectomy-minio mc cp myminio/mlflow-artifacts/path/to/file /tmp/

# Upload object
docker exec neurectomy-minio mc cp /path/to/file myminio/mlflow-artifacts/
```

### AWS CLI (From Host)

```bash
# Configure AWS CLI for MinIO
aws configure --profile minio
# Access Key: minioadmin
# Secret Key: minioadmin
# Region: us-east-1

# List buckets
aws --profile minio --endpoint-url http://localhost:9000 s3 ls

# List objects
aws --profile minio --endpoint-url http://localhost:9000 s3 ls s3://mlflow-artifacts/

# Download file
aws --profile minio --endpoint-url http://localhost:9000 s3 cp s3://mlflow-artifacts/file.txt ./
```

## Monitoring

### Check Container Resources

```bash
# All containers
docker stats

# Specific containers
docker stats neurectomy-minio neurectomy-mlflow neurectomy-ml-service
```

### Check Disk Usage

```bash
# All volumes
docker system df -v

# Specific volumes
docker volume inspect neurectomy-minio-data
docker volume inspect neurectomy-ml-service-cache
```

### Export Logs

```bash
# Export all logs
docker-compose logs > intelligence-foundry-logs.txt

# Export specific service logs
docker-compose logs ml-service > ml-service-logs.txt
```

## Backup & Restore

### Backup PostgreSQL

```bash
# Backup mlflow database
docker exec neurectomy-postgres pg_dump -U mlflow mlflow > mlflow-backup.sql

# Backup optuna database
docker exec neurectomy-postgres pg_dump -U optuna optuna > optuna-backup.sql

# Backup all databases
docker exec neurectomy-postgres pg_dumpall -U neurectomy > full-backup.sql
```

### Restore PostgreSQL

```bash
# Restore mlflow database
docker exec -i neurectomy-postgres psql -U mlflow mlflow < mlflow-backup.sql

# Restore optuna database
docker exec -i neurectomy-postgres psql -U optuna optuna < optuna-backup.sql
```

### Backup MinIO

```bash
# Backup entire bucket
docker exec neurectomy-minio mc mirror myminio/mlflow-artifacts /backup/mlflow-artifacts/

# Or use mc client from host
mc mirror minio/mlflow-artifacts ./backup/mlflow-artifacts/
```

## API Quick Reference

### MLflow Endpoints

```bash
# Create experiment
POST /api/mlflow/experiments/create
{"name": "string", "artifact_location": "string"}

# Get experiment
GET /api/mlflow/experiments/{experiment_id}

# List experiments
GET /api/mlflow/experiments/list

# Create run
POST /api/mlflow/runs/create
{"experiment_id": "string", "run_name": "string"}

# Log metric
POST /api/mlflow/runs/{run_id}/log-metric
{"key": "string", "value": 0.0, "timestamp": 0, "step": 0}

# Log parameter
POST /api/mlflow/runs/{run_id}/log-parameter
{"key": "string", "value": "string"}
```

### Optuna Endpoints

```bash
# Create study
POST /api/optuna/studies/create
{"study_name": "string", "direction": "minimize"}

# Get study
GET /api/optuna/studies/{study_name}

# List studies
GET /api/optuna/studies/list

# Create trial
POST /api/optuna/studies/{study_name}/trials/create
{"params": {"param1": 0.5}, "value": 0.85}

# Get best trial
GET /api/optuna/studies/{study_name}/best-trial

# Get best parameters
GET /api/optuna/studies/{study_name}/best-params
```

## Performance Tips

1. **Use Volume Caching:**
   - ml_service_cache stores pip/torch downloads
   - Saves ~500MB and 5+ minutes on rebuilds

2. **Hot Reload Development:**
   - Volume mount `./services/ml-service:/app`
   - Code changes reflect immediately (no rebuild)

3. **Parallel Builds:**

   ```bash
   docker-compose build --parallel
   ```

4. **Prune Unused Resources:**

   ```bash
   docker system prune -a --volumes
   ```

5. **Optimize Images:**
   - Use multi-stage builds
   - Minimize layer count
   - Use .dockerignore

## Security Checklist

- [ ] Change MinIO credentials from default
- [ ] Use strong PostgreSQL passwords
- [ ] Enable TLS/SSL for external access
- [ ] Implement API authentication
- [ ] Use Docker secrets for sensitive data
- [ ] Regularly update base images
- [ ] Scan images for vulnerabilities (Trivy)
- [ ] Implement network policies
- [ ] Enable audit logging
- [ ] Set up automated backups

---

**Last Updated:** December 7, 2025  
**Version:** 1.0.0  
**Status:** ✅ Operational
