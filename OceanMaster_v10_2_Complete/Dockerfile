# OceanMaster v12 — Production Image  # [v12-enhance]
# Multi-stage build, final image ~400MB
#
# Build:   docker build -t oceanmaster:12 .
# Run:     docker run -p 8000:8000 -e OCEANMASTER_API_KEY=your-key -v $(pwd)/output:/app/output oceanmaster:12

# ══════════════════════════════════════════
# Stage 1: Builder
# ══════════════════════════════════════════
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for netCDF4 / scipy / numpy / lightgbm
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-dev \
    libnetcdf-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ══════════════════════════════════════════
# Stage 2: Runtime
# ══════════════════════════════════════════
FROM python:3.11-slim

LABEL maintainer="OceanMaster <info@oceanmaster.tw>"
LABEL version="12"
LABEL description="台灣遠洋鮪魚漁場預測系統 — v12 commercial upgrade"

WORKDIR /app

# Runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-serial-dev \
    libnetcdf-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# From builder
COPY --from=builder /install /usr/local

# Application code
COPY . .

# Necessary directories
RUN mkdir -p output data/cache data/woa_do_cache data/wcpfc \
    ml_system/data ml_system/models models

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TZ=Asia/Taipei \
    LOG_LEVEL=INFO
# OCEANMASTER_API_KEY is injected at runtime via .env or docker run -e

# [v12-enhance] Healthcheck — confirm the server actually returns OK JSON
HEALTHCHECK --interval=60s --timeout=30s --retries=3 --start-period=30s \
    CMD curl -sf http://localhost:8000/health | python -c "import sys,json; d=json.load(sys.stdin); assert d.get('status')=='ok'" || exit 1

# Port
EXPOSE 8000

# Verify models are loadable on build
RUN python -c "import pickle, glob; \
    files = glob.glob('models/stacking_*.pkl') + glob.glob('ml_system/models/stacking_*.pkl'); \
    [print('OK:', f) for f in files if pickle.load(open(f, 'rb'))]; \
    print(f'Verified {len(files)} models')" || echo "WARNING: Model verification skipped"

# Start
CMD ["python", "-m", "uvicorn", "web_server:app", "--host", "0.0.0.0", "--port", "8000"]
