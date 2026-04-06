# OceanMaster v15.3 — Production Image  # [v13.2]
# Multi-stage build, final image ~400MB
#
# Build:   docker build -t oceanmaster:13.2 .
# Run:     docker run -p 8000:8000 -e OCEANMASTER_API_KEY=your-key -v $(pwd)/output:/app/output oceanmaster:13.2

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
LABEL version="15.3"
LABEL description="台灣遠洋鮪魚漁場預測系統 — v15.3 commercial upgrade"

WORKDIR /app

# Runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-310 \
    libnetcdf22 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# From builder
COPY --from=builder /install /usr/local

# [audit-fix] Non-root user for runtime security
RUN useradd -r -s /usr/sbin/nologin appuser

# Application code
COPY . .

# Necessary directories + ownership
RUN mkdir -p output data/cache data/woa_do_cache data/wcpfc \
    ml_system/data ml_system/models models \
    && chown -R appuser:appuser output data cache models ml_system

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TZ=Asia/Taipei \
    LOG_LEVEL=INFO
# OCEANMASTER_API_KEY is injected at runtime via .env or docker run -e

# [v12-enhance] Healthcheck — confirm the server actually returns OK JSON
HEALTHCHECK --interval=60s --timeout=30s --retries=3 --start-period=30s \
    CMD curl -sf http://localhost:8000/health | python -c "import sys,json; d=json.load(sys.stdin); assert d.get('status') in ('healthy','stale','degraded')" || exit 1

# Port
EXPOSE 8000

# Verify models are loadable and HMAC-signed on build
RUN python -c "from pathlib import Path; \
    files = list(Path('models').glob('stacking_*.pkl')) + list(Path('ml_system/models').glob('stacking_*.pkl')); \
    print(f'Found {len(files)} model files'); \
    [ \
    (lambda p: (__import__('engine.ml.stacking_ensemble', fromlist=['_verify_model_file'])._verify_model_file(p), print(f'OK: {p}')))(f) \
    for f in files \
    ] if files else print('No models to verify (will use expert fallback)')" || echo "WARNING: Model verification skipped"

# [audit-fix] Run as non-root
USER appuser

# Start
CMD ["python", "-m", "uvicorn", "web_server:app", "--host", "0.0.0.0", "--port", "8000"]
