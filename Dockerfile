# ─────────────────────────────────────────────────────────────────────────────
# Multi-Stage Dockerfile for Telecom Vision API
# Stage 1 (builder): installs Python deps into a clean venv
# Stage 2 (runtime): copies only the venv + app code — no build tools in prod
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System libs required to compile OpenCV / EasyOCR / PyMuPDF wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libgl1 \
        libglx-mesa0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime system libs (no compiler needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libglx-mesa0 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY app/ ./app/

# Storage directories (mounted as volumes in production)
# We also create .EasyOCR so the non-root 'appuser' can download AI models at runtime
RUN mkdir -p /app/storage/uploads /app/storage/outputs /app/model_weights /app/.EasyOCR \
    && chown -R appuser:appuser /app

USER appuser

# FastAPI/Uvicorn defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000


# Uvicorn with 4 workers; tune --workers based on CPU count
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
