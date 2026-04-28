# ──────────────────────────────────────────────────────────────────────────────
# Dockerfile — Nakaseke NCD-AI Hypertension Screener
#
# Multi-stage build:
#   Stage 1 (builder) – installs all Python dependencies into a virtual env.
#   Stage 2 (runtime) – copies only the venv + application code into a slim
#                       final image, keeping the image under 1 GB.
#
# Usage:
#   docker build -t nakaseke-ncd-ai .
#   docker run -p 5000:5000 nakaseke-ncd-ai
# ──────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system build tools needed for some Python packages (scipy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first so Docker can cache the pip layer
COPY requirements.txt .

# Build a virtual environment to isolate dependencies
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Paul Sentongo <nankyamaggie8@gmail.com>"
LABEL description="Nakaseke NCD-AI — Hypertension Risk Screener"
LABEL version="1.0.0"

# Runtime system library for OpenMP (LightGBM / XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual env from builder
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /home/appuser/app

# Copy application source
COPY --chown=appuser:appuser . .

# Switch to non-root
USER appuser

# Expose the Flask port
EXPOSE 5000

# Health check: Docker will restart the container if /health fails
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"

# Use gunicorn for production (falls back to Flask dev server if not installed)
CMD ["python", "-m", "gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "2", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "app.app:app"]
