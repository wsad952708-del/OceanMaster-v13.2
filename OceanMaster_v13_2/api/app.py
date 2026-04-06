"""
OceanMaster REST API — FastAPI Application
============================================
Production-grade API for OceanMaster prediction system.

Endpoints:
  GET  /health           — System status
  POST /predict          — Run full prediction pipeline
  GET  /hotspots         — Latest hotspots (GeoJSON)
  GET  /hotspots/{sp}    — Filter by species
  GET  /species          — Supported species list
  POST /retrain          — Check data collection status (auto-finetune disabled)

Run:
  uvicorn api.app:app --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import glob
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# [v15.3-refactor] 共用安全模組
from engine.shared_security import (
    rate_check as _rate_check,
    sanitize_for_json as _sanitize_for_json,
    verify_session_token as _verify_session_token_shared,
)

from fastapi import FastAPI, HTTPException, Request, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    HealthResponse, PredictRequest, PredictResponse, HotspotItem,
    GeoJSONResponse, GeoJSONFeature,
    SpeciesListResponse, SpeciesInfo,
    RetrainRequest, RetrainResponse,
)

log = logging.getLogger("OceanMaster.API")

# Import pipeline (same approach as web_server.py)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from main_v10_3 import OceanMasterPipeline, VERSION

_VALID_MODES = {"offshore", "nearshore", "inshore"}


# ─── App Setup ───
_is_production = os.getenv("OCEANMASTER_ENV") == "production"
app = FastAPI(
    title="OceanMaster API",
    description="AI-powered fishing ground prediction system",
    version=VERSION,
    # [v13.2-P2] Disable docs in production to prevent schema/version leakage
    docs_url=None if _is_production else "/docs",
    redoc_url=None if _is_production else "/redoc",
)

# [v13.2-P1] CORS — origins controlled via env var, restrict in production
# [audit-fix] 預設限制為 localhost，生產環境透過 CORS_ORIGINS 環境變數設定
_cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

# ─── Auth ───
API_KEY = os.getenv("OCEANMASTER_API_KEY", "")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# [v13.2-P0] Production guard: refuse to start without API key
if not API_KEY and os.getenv("OCEANMASTER_ENV") == "production":
    raise SystemExit("FATAL: OCEANMASTER_API_KEY must be set in production")

_start_time = time.time()
_last_prediction: Optional[str] = None

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
HOTSPOT_DIR = OUTPUT_DIR


def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key or session token if OCEANMASTER_API_KEY is set."""
    if not API_KEY:
        # [v13.2-P0] dev-mode only — production blocked at startup
        if os.getenv("OCEANMASTER_ENV") == "production":
            raise HTTPException(status_code=503, detail="Server misconfigured")
        return True  # dev mode only
    # [v15.3-refactor] 也接受 session token（與 web_server 一致）
    _SESSION_SECRET = os.environ.get("SESSION_SECRET", "")
    if api_key == API_KEY:
        return True
    if _SESSION_SECRET and _verify_session_token_shared(api_key, _SESSION_SECRET):
        return True
    raise HTTPException(status_code=401, detail="Invalid API key")


# [v15.3-refactor] Rate limiter & NaN sanitizer — 使用 shared_security 模組
# _rate_check, _sanitize_for_json 已從 engine.shared_security import


# ─── Concurrency Guard ───
# [v13.2-P3] WARNING: asyncio.Semaphore is per-process.
# With uvicorn --workers N (N>1), each worker has its own semaphore,
# so N concurrent pipelines can run. Keep workers=1 or use an external lock.
_predict_sem = asyncio.Semaphore(1)  # at most 1 concurrent /predict per worker


# ─── GET /health ───
@app.get("/health", response_model=HealthResponse)
async def health():
    """System health check."""
    # Check which models are loaded
    models_loaded = {}
    for sp in ["yellowfin", "bigeye", "albacore", "skipjack"]:
        model_path = PROJECT_ROOT / "models" / f"stacking_{sp}.pkl"
        models_loaded[sp] = model_path.exists()

    # Data cache age
    cache_dir = PROJECT_ROOT / "data" / "cache"
    cache_age = None
    if cache_dir.exists():
        npz_files = list(cache_dir.glob("*.npz"))
        if npz_files:
            newest = max(f.stat().st_mtime for f in npz_files)
            cache_age = round((time.time() - newest) / 3600, 1)

    return HealthResponse(
        status="healthy",
        version=VERSION,
        uptime_seconds=round(time.time() - _start_time, 1),
        last_prediction=_last_prediction,
        models_loaded=models_loaded,
        data_cache_age_hours=cache_age,
    )


# ─── POST /predict ───
_DEFAULT_LAT = (5, 35)
_DEFAULT_LON = (120, 175)
_DEFAULT_VESSEL = (22.61, 120.28)  # 高雄港

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, request: Request, _=Depends(verify_api_key)):
    """Run full prediction pipeline (direct async call, no subprocess)."""
    global _last_prediction

    # [v13.2-P1] Rate limit check
    ip = request.client.host if request.client else "unknown"
    if not _rate_check(ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # [v13.2-sec] Mode whitelist validation
    if req.mode not in _VALID_MODES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid mode '{req.mode}'. Must be one of: {sorted(_VALID_MODES)}"
        )

    # [v13.2-P1] Concurrency guard — only 1 pipeline at a time
    if _predict_sem.locked():
        raise HTTPException(status_code=503, detail="Prediction already running, please retry later")

    lat_range = tuple(req.lat_range) if req.lat_range and len(req.lat_range) == 2 else _DEFAULT_LAT
    lon_range = tuple(req.lon_range) if req.lon_range and len(req.lon_range) == 2 else _DEFAULT_LON

    async with _predict_sem:
        t0 = time.time()
        try:
            pipeline = OceanMasterPipeline(
                lat_range=lat_range,
                lon_range=lon_range,
                vessel_pos=_DEFAULT_VESSEL,
                output_dir=str(OUTPUT_DIR),
            )
            results = await asyncio.wait_for(pipeline.run_full(), timeout=900)
            elapsed = round(time.time() - t0, 1)

            # Load hotspots from pipeline output + sanitize NaN/Inf
            hotspots = _sanitize_for_json(_load_latest_hotspots(req.species))
            _last_prediction = datetime.now().isoformat()

            return PredictResponse(
                status="ok",
                elapsed_seconds=elapsed,
                n_hotspots=len(hotspots),
                hotspots=hotspots,
                timestamp=_last_prediction,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Pipeline timed out (>15 min)")
        except HTTPException:
            raise
        except Exception as e:
            log.error(f"Pipeline error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error")


# ─── GET /hotspots ───
@app.get("/hotspots", response_model=GeoJSONResponse)
async def get_hotspots(species: Optional[str] = None):
    """Get latest hotspots as GeoJSON."""
    hotspots = _load_latest_hotspots([species] if species else None)
    
    features = []
    for h in hotspots:
        features.append(GeoJSONFeature(
            geometry={"type": "Point", "coordinates": [h.lon, h.lat]},
            properties={
                "rank": h.rank,
                "species": h.species,
                "species_zh": h.species_zh,
                "score": h.score,
                "hsi_pct": h.hsi_pct,
                "sst": h.sst,
                "chl": h.chl,
                "depth_m": h.depth_m,
                "cpue_index": h.cpue_index,
                "hook_depth_optimal": h.hook_depth_optimal,
                "distance_nm": h.distance_nm,
                "eez": h.eez,
            }
        ))
    
    return GeoJSONResponse(features=features)


# ─── GET /hotspots/{species} ───
@app.get("/hotspots/{species}", response_model=GeoJSONResponse)
async def get_hotspots_by_species(species: str):
    """Get hotspots filtered by species."""
    return await get_hotspots(species=species)


# ─── GET /species ───
@app.get("/species", response_model=SpeciesListResponse)
async def list_species():
    """List all supported species with parameters."""
    # [v13.2-P1] sys.path already set at module level (L45)
    from engine.species_params import SPECIES
    
    items = []
    for key, params in SPECIES.items():
        items.append(SpeciesInfo(
            key=key,
            name_zh=params.get("name_zh", ""),
            name_en=params.get("name_en", ""),
            scientific=params.get("scientific", ""),
            Topt_C=params.get("Topt_C", 25.0),
            T_min=params.get("T_min", 10.0),
            T_max=params.get("T_max", 35.0),
            gear_type=params.get("gear_type", "longline"),
        ))
    
    return SpeciesListResponse(count=len(items), species=items)


# ─── POST /retrain ───
@app.post("/retrain", response_model=RetrainResponse, dependencies=[Depends(verify_api_key)])
async def retrain(req: RetrainRequest):
    """Check incremental data collection status (auto-finetune disabled)."""
    # [v13.2-P1] sys.path already set at module level (L45)
    from engine.incremental_learner import maybe_finetune, _count_unique_runs
    
    n_runs = _count_unique_runs()
    
    if req.force or n_runs >= 30:
        finetuned = maybe_finetune(model_dir=str(PROJECT_ROOT / "models"))
        return RetrainResponse(
            status="ok",
            message="Fine-tuning completed" if finetuned else "Fine-tuning skipped (insufficient data)",
            n_runs=n_runs,
            finetuned=finetuned,
        )
    else:
        return RetrainResponse(
            status="pending",
            message=f"Need {30 - n_runs} more runs before auto fine-tune (force=true to override)",
            n_runs=n_runs,
            finetuned=False,
        )


# ─── Helpers ───
def _load_latest_hotspots(species_filter=None) -> list:
    """Load the most recent hotspots JSON."""
    patterns = [
        str(OUTPUT_DIR / "hotspots*.json"),
        str(OUTPUT_DIR / "OceanMaster*.json"),
    ]
    
    all_files = []
    for p in patterns:
        all_files.extend(glob.glob(p))
    
    if not all_files:
        # Try GeoJSON
        geojson_files = glob.glob(str(OUTPUT_DIR / "*.geojson"))
        if geojson_files:
            latest = max(geojson_files, key=os.path.getmtime)
            return _parse_geojson(latest, species_filter)
        return []
    
    latest = max(all_files, key=os.path.getmtime)
    
    try:
        with open(latest, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return []
    
    items = raw if isinstance(raw, list) else raw.get("hotspots", [])
    
    result = []
    sp_zh_map = {
        "yellowfin": "黃鰭鮪", "bigeye": "大目鮪",
        "skipjack": "正鰹", "albacore": "長鰭鮪",
        "japanese_flying_squid": "日本魷", "pacific_saury": "秋刀魚",
        "mahi_mahi": "鬼頭刀", "blue_marlin": "旗魚",
        "mackerel_scad": "竹筴魚",
    }
    
    for h in items:
        sp = h.get("species", "")
        if species_filter and sp not in species_filter:
            continue
        
        result.append(HotspotItem(
            rank=h.get("rank", 0),
            lat=h.get("lat", 0),
            lon=h.get("lon", 0),
            species=sp,
            species_zh=sp_zh_map.get(sp, sp),
            score=round(h.get("score", 0), 4),
            hsi_pct=int(round(h.get("score", 0) * 100)),
            sst=round(h.get("sst", 0), 1),
            chl=round(h.get("chl", 0), 4),
            depth_m=round(h.get("depth_m", 0), 0),
            mld_m=round(h.get("mld_m", 0), 0),
            z20_m=round(h.get("z20_m", 0), 0),
            cpue_index=round(h.get("cpue_index", 0), 3),
            cpue_ci_low=round(h.get("cpue_ci_low", 0), 3),
            cpue_ci_high=round(h.get("cpue_ci_high", 0), 3),
            hook_depth_optimal=h.get("hook_depth_optimal", 0),
            distance_nm=round(h.get("distance_nm", 0), 0),
            eez=h.get("eez", ""),
        ))
    
    return result


def _parse_geojson(filepath, species_filter=None) -> list:
    """Parse GeoJSON file into hotspot items."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    
    result = []
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        coords = feat.get("geometry", {}).get("coordinates", [0, 0])
        sp = props.get("species", "")
        if species_filter and sp not in species_filter:
            continue
        
        result.append(HotspotItem(
            rank=props.get("rank", 0),
            lat=coords[1] if len(coords) > 1 else 0,
            lon=coords[0],
            species=sp,
            score=round(props.get("score", 0), 4),
            hsi_pct=int(round(props.get("score", 0) * 100)),
            sst=round(props.get("sst", 0), 1),
        ))
    
    return result
