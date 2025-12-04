# MLflow with PostgreSQL support
FROM ghcr.io/mlflow/mlflow:v2.17.0

# Install psycopg2 for PostgreSQL support and curl for health checks
RUN pip install --no-cache-dir psycopg2-binary && \
    apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Expose MLflow port
EXPOSE 5000
