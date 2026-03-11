# Multi-stage Dockerfile for LEDiff service
# Stage 1: Install dependencies
# Stage 2: Runtime

# --- Stage 1: Build environment ---
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenexr-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY service/backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install LEDiff custom diffusers
COPY vendor/LEDiff/ /app/vendor/LEDiff/
RUN pip install --no-cache-dir -e /app/vendor/LEDiff/

# --- Stage 2: Runtime ---
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenexr-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/vendor/LEDiff /app/vendor/LEDiff

# Copy application code
COPY service/backend/app/ /app/app/
COPY service/frontend/static/ /app/frontend/static/

# Weights must be mounted or copied at runtime
# Expected at /app/weights/highlight/ and/or /app/weights/shadow/

ENV LEDIFF_WEIGHTS_DIR=/app/weights
ENV JOB_TTL_HOURS=24
ENV MAX_MEGAPIXELS=16

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
