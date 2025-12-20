# ============================================================================
# NEURECTOMY ML Service - Production Dockerfile (Intelligence Foundry)
# Multi-stage build for optimized container size
# ============================================================================

FROM python:3.11-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r neurectomy && useradd -r -g neurectomy neurectomy

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config.py .
COPY mlflow_server.py .
COPY optuna_service.py .
COPY training_engine.py .
COPY intelligence_foundry_main.py .
COPY main.py .
COPY src/ ./src/

# Set ownership
RUN chown -R neurectomy:neurectomy /app

# Switch to non-root user
USER neurectomy

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    PORT=16081

# Expose port
EXPOSE 16081

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:16081/health || exit 1

# Run Intelligence Foundry application
CMD ["uvicorn", "intelligence_foundry_main:app", "--host", "0.0.0.0", "--port", "16081"]
