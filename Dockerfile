# Simplified Dockerfile for Railway
FROM python:3.11-slim

# Set environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SURRAPI_PORT=8000 \
    SURRAPI_DEVICE=cpu \
    SURRAPI_LOG_LEVEL=INFO

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies - use CPU-only torch to reduce size
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/
COPY scripts/ ./scripts/

# Generate demo weights
RUN python scripts/generate_demo_weights.py

# Create non-root user
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Expose port
EXPOSE 8000

# Start server - use PORT env var provided by Railway
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
