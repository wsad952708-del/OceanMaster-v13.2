"""
OceanMaster — Deployment Package (FastAPI)
============================================
Production-ready API server for serving ML predictions.

Endpoints:
  POST /predict         — Single location prediction
  POST /predict/grid    — Grid prediction for a region
  GET  /health          — Health check
  GET  /model/info      — Model metadata and performance
  GET  /species         — List supported species

Run:
  python deployment_package.py             # Starts server on port 8000
  python deployment_package.py --test      # Run API tests without server

Docker:
  docker build -t oceanmaster-ml .
  docker run -p 8000:8000 oceanmaster-ml
"""

import json
import logging
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Dict, List

log = logging.getLogger("OceanMaster.API")

# ─── Check FastAPI availability ────────────────
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    log.warning("FastAPI not installed. API server unavailable. pip install fastapi uvicorn")


# ═══════════════════════════════════════════════════
#  Request/Response Models
# ═══════════════════════════════════════════════════

if HAS_FASTAPI:

    class PredictRequest(BaseModel):
        latitude: float = Field(..., ge=-90, le=90, description="Latitude")
        longitude: float = Field(..., ge=-180, le=180, description="Longitude")
        date: str = Field(..., description="Date (YYYY-MM-DD)")
        species: str = Field(default="yellowfin", description="Target species")
        ocean_features: Optional[Dict[str, float]] = Field(
            default=None,
            description="Optional ocean features (sst, chl, ssh, do, etc.)"
        )

    class GridPredictRequest(BaseModel):
        lat_min: float = Field(..., ge=-90, le=90)
        lat_max: float = Field(..., ge=-90, le=90)
        lon_min: float = Field(..., ge=-180, le=180)
        lon_max: float = Field(..., ge=-180, le=180)
        date: str
        species: str = "yellowfin"
        resolution: float = Field(default=0.5, ge=0.1, le=5.0)

    class PredictResponse(BaseModel):
        latitude: float
        longitude: float
        date: str
        species: str
        cpue_prediction: float
        cpue_ml: float
        cpue_science: float
        phi: float
        hsi: float
        confidence: float
        confidence_level: str
        recommendation: str

    class HealthResponse(BaseModel):
        status: str
        model_loaded: bool
        species_available: List[str]
        version: str
        timestamp: str


# ═══════════════════════════════════════════════════
#  API Application
# ═══════════════════════════════════════════════════

def create_app(model_dir: str = "ml_system/models") -> "FastAPI":
    """Create and configure the FastAPI application."""
    if not HAS_FASTAPI:
        raise RuntimeError("FastAPI not installed.")

    app = FastAPI(
        title="OceanMaster ML Prediction API",
        description=(
            "Fishing hotspot prediction combining Deutsch 2015 Metabolic Index, "
            "SEAPODYM habitat models, and trained ML ensemble."
        ),
        version="2.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Load predictors
    predictors = {}
    for species in ["yellowfin", "bigeye", "albacore", "skipjack"]:
        from integrated_predictor import IntegratedPredictor
        pred = IntegratedPredictor(species=species, fusion_strategy="ml_constrained")
        pred.load_ml_model(model_dir)
        predictors[species] = pred

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest):
        """Predict CPUE at a single location."""
        if request.species not in predictors:
            raise HTTPException(400, f"Unknown species: {request.species}")

        predictor = predictors[request.species]

        # Use provided features or simulate
        if request.ocean_features:
            features = request.ocean_features
        else:
            from historical_data_collector import OceanEnvironmentSimulator
            env = OceanEnvironmentSimulator(seed=hash(
                f"{request.latitude}{request.longitude}{request.date}"
            ) % 2**31)
            dt = datetime.strptime(request.date, "%Y-%m-%d")
            features = {
                "sst": env.simulate_sst(request.latitude, request.longitude, dt.month),
                "chl": env.simulate_chlorophyll(request.latitude, request.longitude, dt.month),
                "ssh": env.simulate_ssh(request.latitude, request.longitude, dt.month),
                "do": env.simulate_do(
                    request.latitude, request.longitude, dt.month,
                    env.simulate_sst(request.latitude, request.longitude, dt.month)
                ),
                "current_speed": env.simulate_current_speed(request.latitude, request.longitude),
                "front_strength": env.simulate_front_strength(request.latitude, request.longitude),
                "eddy_strength": env.simulate_eddy_strength(request.latitude, request.longitude),
            }

        result = predictor.predict(
            request.latitude, request.longitude, request.date, features
        )

        return PredictResponse(
            latitude=result.lat,
            longitude=result.lon,
            date=result.date,
            species=result.species,
            cpue_prediction=round(result.cpue_final, 2),
            cpue_ml=round(result.cpue_ml, 2),
            cpue_science=round(result.cpue_science, 2),
            phi=round(result.phi, 3),
            hsi=round(result.hsi, 4),
            confidence=round(result.confidence, 3),
            confidence_level=result.confidence_level,
            recommendation=result.recommendation,
        )

    @app.post("/predict/grid")
    async def predict_grid(request: GridPredictRequest):
        """Predict CPUE across a spatial grid."""
        if request.species not in predictors:
            raise HTTPException(400, f"Unknown species: {request.species}")

        predictor = predictors[request.species]
        df = predictor.predict_grid(
            lat_range=(request.lat_min, request.lat_max),
            lon_range=(request.lon_min, request.lon_max),
            date=request.date,
            resolution=request.resolution,
        )
        return {"predictions": df.to_dict(orient="records"), "count": len(df)}

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check."""
        available = [sp for sp, p in predictors.items() if p.ml_available]
        return HealthResponse(
            status="healthy",
            model_loaded=len(available) > 0,
            species_available=available if available else list(predictors.keys()),
            version="2.0.0",
            timestamp=datetime.utcnow().isoformat(),
        )

    @app.get("/model/info")
    async def model_info():
        """Return model metadata."""
        info = {}
        model_path = Path(model_dir)
        for species in predictors:
            metrics_file = model_path / f"training_metrics_{species}.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    info[species] = json.load(f)
            else:
                info[species] = {
                    "status": "no_trained_model",
                    "mode": "science_only",
                }
        return info

    @app.get("/species")
    async def list_species():
        """List supported species."""
        return {
            "species": list(predictors.keys()),
            "default": "yellowfin",
        }

    return app


# ═══════════════════════════════════════════════════
#  Dockerfile Generator
# ═══════════════════════════════════════════════════

def generate_dockerfile(output_dir: str = "."):
    """Generate a Dockerfile for deployment."""
    dockerfile = """FROM python:3.11-slim

WORKDIR /app

COPY requirements_ml.txt .
RUN pip install --no-cache-dir -r requirements_ml.txt

COPY ml_system/ ./ml_system/
COPY engine/ ./engine/
COPY config.py .

EXPOSE 8000

CMD ["uvicorn", "ml_system.deployment_package:create_app", "--host", "0.0.0.0", "--port", "8000", "--factory"]
"""
    path = Path(output_dir) / "Dockerfile.ml"
    path.write_text(dockerfile)
    log.info(f"Dockerfile written to {path}")

    # Requirements
    reqs = """scikit-learn>=1.2
numpy>=1.24
pandas>=2.0
scipy>=1.10
joblib>=1.2
fastapi>=0.100
uvicorn>=0.23
pydantic>=2.0
"""
    req_path = Path(output_dir) / "requirements_ml.txt"
    req_path.write_text(reqs)
    log.info(f"Requirements written to {req_path}")


# ═══════════════════════════════════════════════════
#  Quick Test (no server needed)
# ═══════════════════════════════════════════════════

def run_api_test():
    """Test the prediction logic without starting the server."""
    print("=" * 60)
    print("API Logic Test (no server)")
    print("=" * 60)

    from integrated_predictor import IntegratedPredictor

    predictor = IntegratedPredictor(species="yellowfin", fusion_strategy="ml_constrained")
    predictor.load_ml_model("ml_system/models")

    test_cases = [
        {"lat": 25.0, "lon": 135.0, "date": "2026-06-15", "label": "Kuroshio"},
        {"lat": 10.0, "lon": 155.0, "date": "2026-06-15", "label": "Tropical"},
        {"lat": 5.0,  "lon": 170.0, "date": "2026-01-15", "label": "Equatorial Winter"},
        {"lat": 30.0, "lon": 145.0, "date": "2026-09-01", "label": "Subtropical Fall"},
    ]

    from historical_data_collector import OceanEnvironmentSimulator
    env = OceanEnvironmentSimulator(seed=99)

    for tc in test_cases:
        dt = datetime.strptime(tc["date"], "%Y-%m-%d")
        features = {
            "sst": env.simulate_sst(tc["lat"], tc["lon"], dt.month),
            "chl": env.simulate_chlorophyll(tc["lat"], tc["lon"], dt.month),
            "ssh": env.simulate_ssh(tc["lat"], tc["lon"], dt.month),
            "do": env.simulate_do(tc["lat"], tc["lon"], dt.month,
                                  env.simulate_sst(tc["lat"], tc["lon"], dt.month)),
            "current_speed": env.simulate_current_speed(tc["lat"], tc["lon"]),
            "front_strength": env.simulate_front_strength(tc["lat"], tc["lon"]),
            "eddy_strength": env.simulate_eddy_strength(tc["lat"], tc["lon"]),
        }

        result = predictor.predict(tc["lat"], tc["lon"], tc["date"], features)
        print(f"\n  {tc['label']:25s} → {result.cpue_final:6.1f} kg/day "
              f"[{result.recommendation.upper():8s}] "
              f"conf={result.confidence:.2f} Φ={result.phi:.1f} HSI={result.hsi:.3f}")

    print("\nAll tests passed.")


# ═══════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    if "--test" in sys.argv:
        run_api_test()
    elif "--dockerfile" in sys.argv:
        generate_dockerfile()
    else:
        if not HAS_FASTAPI:
            print("FastAPI not installed. Running test mode instead.")
            run_api_test()
        else:
            import uvicorn
            app = create_app()
            uvicorn.run(app, host="0.0.0.0", port=8000)
