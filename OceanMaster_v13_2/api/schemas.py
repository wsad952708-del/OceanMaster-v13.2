"""
OceanMaster REST API — Pydantic Schemas
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str  # [audit-fix] 由呼叫端填入 VERSION，不硬寫
    uptime_seconds: float
    last_prediction: Optional[str] = None
    models_loaded: Dict[str, bool] = {}
    data_cache_age_hours: Optional[float] = None


class PredictRequest(BaseModel):
    mode: str = Field("offshore", description="offshore / inshore")
    species: Optional[List[str]] = Field(
        None, description="Filter species, e.g. ['yellowfin','bigeye']. None = all"
    )
    lat_range: Optional[List[float]] = Field(
        None, description="[lat_min, lat_max], e.g. [5, 35]"
    )
    lon_range: Optional[List[float]] = Field(
        None, description="[lon_min, lon_max], e.g. [120, 175]"
    )
    use_cache: bool = Field(True, description="Use cached data if available")


class HotspotItem(BaseModel):
    rank: int
    lat: float
    lon: float
    species: str
    species_zh: str = ""
    score: float
    hsi_pct: int
    sst: float = 0
    chl: float = 0
    depth_m: float = 0
    mld_m: float = 0
    z20_m: float = 0
    cpue_index: float = 0
    cpue_ci_low: float = 0
    cpue_ci_high: float = 0
    hook_depth_optimal: int = 0
    distance_nm: float = 0
    eez: str = ""


class PredictResponse(BaseModel):
    status: str
    elapsed_seconds: float
    n_hotspots: int
    hotspots: List[HotspotItem]
    data_quality: Optional[Dict] = None
    timestamp: str


class GeoJSONFeature(BaseModel):
    type: str = "Feature"
    geometry: Dict
    properties: Dict


class GeoJSONResponse(BaseModel):
    type: str = "FeatureCollection"
    features: List[GeoJSONFeature]


class SpeciesInfo(BaseModel):
    key: str
    name_zh: str
    name_en: str
    scientific: str
    Topt_C: float
    T_min: float
    T_max: float
    gear_type: str = "longline"


class SpeciesListResponse(BaseModel):
    count: int
    species: List[SpeciesInfo]


class RetrainRequest(BaseModel):
    force: bool = Field(False, description="Force retrain even if below threshold")


class RetrainResponse(BaseModel):
    status: str
    message: str
    n_runs: int = 0
    finetuned: bool = False
