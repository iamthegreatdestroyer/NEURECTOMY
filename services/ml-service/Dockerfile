# ============================================================================
# NEURECTOMY ML Service - Production Dockerfile
# Multi-stage build for optimized container size
# ============================================================================

# Stage 1: Build dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.7.1

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Export dependencies to requirements.txt (without dev dependencies)
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes --without dev

# Stage 2: Runtime
FROM python:3.11-slim as runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r neurectomy && useradd -r -g neurectomy neurectomy

# Copy requirements and install
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
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
    PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
