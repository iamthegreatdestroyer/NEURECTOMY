# Intelligence Foundry Integration Tests

This directory contains comprehensive integration tests for the Intelligence Foundry ML service.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                          # Shared fixtures and configuration
├── integration/
│   ├── __init__.py
│   ├── test_health.py                   # Health check tests
│   ├── test_mlflow_experiments.py       # MLflow experiment tests
│   ├── test_mlflow_runs.py              # MLflow run tests
│   ├── test_mlflow_artifacts.py         # MLflow artifact tests
│   ├── test_optuna_studies.py           # Optuna study tests
│   ├── test_optuna_optimization.py      # Optuna optimization tests
│   ├── test_websocket.py                # WebSocket streaming tests
│   └── test_end_to_end.py               # End-to-end workflow tests
```

## Prerequisites

### 1. Docker Services Running

All Intelligence Foundry services must be operational:

```bash
# Start services
docker-compose up -d minio mlflow ml-service optuna-dashboard

# Verify health
curl http://localhost:8002/health
```

### 2. Install Test Dependencies

```bash
# From services/ml-service directory
pip install -r requirements-test.txt
```

## Running Tests

### Run All Tests

```bash
# From services/ml-service directory
pytest

# With verbose output
pytest -v

# With coverage report
pytest --cov
```

### Run Specific Test Files

```bash
# Health checks only
pytest tests/integration/test_health.py

# MLflow tests
pytest tests/integration/test_mlflow_experiments.py
pytest tests/integration/test_mlflow_runs.py
pytest tests/integration/test_mlflow_artifacts.py

# Optuna tests
pytest tests/integration/test_optuna_studies.py

# WebSocket tests
pytest tests/integration/test_websocket.py
```

### Run Tests by Marker

```bash
# Run only MLflow tests
pytest -m mlflow

# Run only Optuna tests
pytest -m optuna

# Run integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

### Parallel Execution

```bash
# Run tests in parallel (4 workers)
pytest -n 4
```

## Test Categories

### 1. Health Checks (`test_health.py`)

- Main service health endpoint
- MLflow connectivity
- MinIO connectivity
- Optuna connectivity
- Subsystem health verification

**Expected Results:**

- All services return healthy status
- HTTP 200 responses
- Correct JSON response structure

### 2. MLflow Experiments (`test_mlflow_experiments.py`)

- Experiment creation with tags
- Experiment listing
- Experiment retrieval by ID and name
- Experiment search with filters
- Duplicate experiment handling

**Expected Results:**

- Experiments created in MLflow
- Searchable by tags and metadata
- Proper error handling for duplicates

### 3. MLflow Runs (`test_mlflow_runs.py`)

- Run creation within experiments
- Metric logging (single and batch)
- Parameter logging
- Tag logging
- Run status updates
- Run search with filters

**Expected Results:**

- Runs track metrics, params, tags correctly
- Search filters work properly
- Batch operations succeed

### 4. MLflow Artifacts (`test_mlflow_artifacts.py`)

- Artifact upload to S3 (MinIO)
- Artifact download from S3
- Artifact listing
- Model registration
- Directory structure preservation

**Expected Results:**

- Artifacts stored in MinIO S3 buckets
- Download matches upload
- Model registry integration works

### 5. Optuna Studies (`test_optuna_studies.py`)

- Study creation with direction
- Study listing
- Study retrieval
- Trial creation (manual)
- Trial listing
- Best trial retrieval
- Best parameters retrieval

**Expected Results:**

- Studies stored in PostgreSQL
- Trials tracked correctly
- Best trial selection works

## Test Coverage Goals

- **Overall Coverage:** ≥80%
- **Critical Paths:** ≥95%
  - Health checks
  - Experiment/run creation
  - Artifact storage
  - Study/trial management

## Common Issues & Solutions

### Issue: Services Not Healthy

**Symptom:** Tests fail with connection errors

**Solution:**

```bash
# Check service status
docker-compose ps

# Restart unhealthy services
docker-compose restart mlflow ml-service

# Check logs
docker-compose logs ml-service --tail=50
```

### Issue: Port Already in Use

**Symptom:** `Bind for 0.0.0.0:8002 failed`

**Solution:**

```bash
# Find process using port
netstat -ano | findstr :8002

# Change port in docker-compose.yml if needed
```

### Issue: PostgreSQL Connection Failed

**Symptom:** `password authentication failed`

**Solution:**

```bash
# Verify database users exist
docker exec neurectomy-postgres psql -U neurectomy -c "\du"

# Recreate users if needed (see DOCKER_DEPLOYMENT_VERIFICATION.md)
```

### Issue: MinIO Access Denied

**Symptom:** `Access Denied` when uploading artifacts

**Solution:**

```bash
# Check MinIO credentials in .env
# Should be: minioadmin/minioadmin

# Verify buckets exist
docker exec neurectomy-minio mc ls myminio/
```

## Performance Benchmarks

Expected performance for integration tests:

| Test Category         | Expected Duration | Target  |
| --------------------- | ----------------- | ------- |
| Health checks         | < 5s              | < 2s    |
| Experiment creation   | < 10s             | < 5s    |
| Run operations        | < 15s             | < 10s   |
| Artifact upload (1MB) | < 20s             | < 15s   |
| Study creation        | < 10s             | < 5s    |
| Trial operations      | < 15s             | < 10s   |
| WebSocket connection  | < 5s              | < 2s    |
| Full test suite       | < 5 min           | < 3 min |

## Continuous Integration

### GitHub Actions Configuration

```yaml
name: Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Start Docker services
        run: |
          docker-compose up -d minio mlflow ml-service
          sleep 30

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          cd services/ml-service
          pip install -r requirements.txt
          pip install -r requirements-test.txt

      - name: Run tests
        run: |
          cd services/ml-service
          pytest -v --cov --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./services/ml-service/coverage.xml
```

## Next Steps

After integration tests pass:

1. **Performance Testing** - Benchmark API endpoints under load
2. **Load Testing** - Test with 100+ concurrent users
3. **Stress Testing** - Find breaking points
4. **Chaos Engineering** - Test failure scenarios
5. **Security Testing** - Vulnerability scanning
6. **End-to-End Testing** - Full user workflows

## Documentation

- [Docker Deployment Guide](../../docs/technical/DOCKER_DEPLOYMENT_COMPLETE.md)
- [API Documentation](http://localhost:8002/docs)
- [Quick Reference](../../docs/technical/INTELLIGENCE_FOUNDRY_QUICK_REFERENCE.md)
