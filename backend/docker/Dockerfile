FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY requirements.txt .
COPY backend/ backend/
COPY scripts/ scripts/
COPY scripts/ingest_data.py .
COPY gmm_model.pkl .  # May not exist yet, but that's ok

# Install Python dependencies
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache -r requirements.txt

# Create data directory for models
RUN mkdir -p /app/data

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run FastAPI app
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

