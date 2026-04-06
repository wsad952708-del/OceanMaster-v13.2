"""
OceanMaster v10.3 — ML全魚種 + 精度增強版
==========================================
基於 v10.2 全面修正版，新增:

  [A-1] ML 4魚種全整合 (yellowfin/bigeye/skipjack/albacore)
  [A-2] 備用衛星數據源 (多鏡像 ERDDAP 回退)
  [A-3] ENSO/ONI 即時指數 (NOAA CPC → 內建回退)
  [B-1] 月相漁獲因子 (新月+25%, 滿月-15%)
  [B-2] 歷史漁場先驗權重 (已知高產區加分)
  [B-3] 深度溫度適合度 (大目鮪/長鰭鮪)
  [B-4] ENSO 空間修正 (El Niño魚群東移/La Niña西聚)

使用:
  python main_v10_3.py --run-now
  python main_v10_3.py --api
"""

import asyncio
import argparse
import logging
import sys
import time
import json
import warnings
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import os

# ─── CMEMS 帳號設定 (Copernicus Marine Service) ───
if not os.environ.get("CMEMS_USER"):
    os.environ["CMEMS_USER"] = "wsad952708@gmail.com"
    os.environ["CMEMS_PASS"] = "Aa123456789."

# 抑制常見 warning
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice")
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# ML
try:
    import joblib
    ML_LIBS_OK = True
except ImportError:
    ML_LIBS_OK = False

# ── v8 核心演算法 ──
from engine.algorithms import (
    detect_sst_fronts,     # (sst, lat, lon) → Dict
    compute_boa_gradient,     # (chl, lat, lon) → Dict
    compute_ftle,          # (u, v, lat, lon) → Dict
    compute_thermocline,   # (temp_3d, depths, lat, lon) → Dict
    calculate_eke,         # (ssh, lat, lon) → Dict (v10.4 geostrophic EKE)
)
from engine.hsi_models import compute_all_hsi, get_moon_phase
from engine.ai_fusion import fuse_and_rank, assess_safety
from engine.kml_generator import generate_kml

# ── v9 ── (optional, may need httpx)
try:
    from engine.cmems_ssh import EddyDetector
except ImportError:
    EddyDetector = None
try:
    from engine.eez.eez_checker import EEZChecker
except ImportError:
    EEZChecker = None
try:
    from engine.navigation.route_planner import FuelOptimalRouter
except ImportError:
    FuelOptimalRouter = None
try:
    from engine.geojson_output import generate_geojson, generate_satellite_email_payload
except ImportError:
    generate_geojson = None
    generate_satellite_email_payload = None

# ── v10 生態模組 ──
from engine.dissolved_oxygen import DissolvedOxygenAnalyzer
# PrimaryProductionEngine removed — unified on ForageEngine (same VGPM, less duplication)
from engine.ocean_physics import BathymetryAnalyzer, OceanPhysicsEngine

# ── v10.4 ML 數據模組 ──
try:
    from engine.gebco_features import GEBCOFeatures
    HAS_GEBCO = True
except ImportError:
    HAS_GEBCO = False

# ── v10.2 真實數據擷取 (零 np.random) ──
from engine.data_fetcher_v2 import OceanDataFetcher

# ── v10.3 安全防護與氣象特徵 ──
try:
    from engine.weather_fetcher import WeatherFetcher
    from engine.typhoon_tracker import TyphoonTracker
    from engine.safety_checker import (
        is_safe_for_fishing, SafetyLevel,
        assess_grid_safety, filter_hotspots_by_safety,
    )
    from engine.wave_fetcher import WaveFetcher
    from engine.pressure_features import compute_pressure_features
    from engine.tchp_calculator import compute_tchp, tchp_alert_message
    from engine.typhoon_ml_features import TyphoonFeatureEngineer
    from engine.rainfall_fetcher import RainfallFetcher
    from engine.mld_forecast import MLDForecaster
    SAFETY_OK = True
except ImportError as _safety_err:
    SAFETY_OK = False
    log.warning(f"Safety/Weather modules not available: {_safety_err}")

# ── v10.4 商業實戰化模組 ──
try:
    from engine.salinity_fetcher import SalinityFetcher
    SALINITY_OK = True
except ImportError:
    SALINITY_OK = False
    log.warning("SalinityFetcher not available")

try:
    from compliance.regulation_checker import RegulationChecker, filter_hotspots
    LEGAL_OK = True
except ImportError:
    LEGAL_OK = False
    log.warning("RegulationChecker not available")

# ── v10.2 商業核心 [P0-1]: 這次真的呼叫了 ──
from engine.commercial_core_v2 import (
    MetabolicIndexEngine, SEAPODYMHabitatEngine,
    EddyEdgeDetector, FrontPersistenceTracker,
    MultiDepthFeatureExtractor, CommercialGradeHSI,
    METABOLIC_TRAITS,
)

# ── v10.2 精度增強 [P0-2]: 這次真的呼叫了 ──
from engine.accuracy_booster import (
    SSTAnomalyEngine, ChlAnomalyEngine,
    TemporalLagEngine, ENSOAdjuster,
    CurrentConvergenceDetector, ExplainableHSI,
    UltimateAccuracyBooster,
)

from config import OUTPUT, SPECIES_PARAMS

# ── v10.3 SHAP 可解釋性 ──
try:
    from engine.shap_explainer import SHAPExplainer, get_shap_explainer
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

# ── v10.3 增強數據源 ──
try:
    from engine.enhanced_data_sources import (
        fetch_enso_oni, enso_species_modifier,
        compute_lunar_fishing_factor,
        compute_historical_prior,
        compute_depth_temperature_index,
        apply_all_enhancements,
    )
    ENHANCED_OK = True
except ImportError:
    ENHANCED_OK = False

# ── v10.4 GreenFish Lite 模組 ──
try:
    from engine.thermocline_fetcher import ThermoclineFetcher
    from engine.eddy_detector import EddyDetector as GFEddyDetector
    from engine.forage_engine import ForageEngine
    from engine.dvm_model import DVMModel
    from engine.greenfish_hsi import GreenFishLiteHSI
    from engine.vessel_lights import VesselLightValidator
    GREENFISH_OK = True
except ImportError as _gf_err:
    GREENFISH_OK = False
    log_gf_err = str(_gf_err)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("OceanMaster")

VERSION = "10.3"


class OceanMasterPipeline:
    """
    v10.3 管線 — ML全魚種 + 精度增強

    Step 1: 真實數據抓取 + ENSO/ONI
    Step 2: 物理演算法 (鋒面/FTLE/溫躍層)
    Step 3: 生態分析 (DO/VGPM/營養鏈)
    Step 4: v8 基礎 HSI + 月相因子
    Step 5: 商業核心 HSI (Phi + SEAPODYM)     ← [P0-1]
    Step 6: 精度增強 (異常/ENSO/匯聚/SHAP)    ← [P0-2] + ONI即時
    Step 6.5: ML Ensemble CPUE預測 (4魚種)    ← [A-1] NEW
    Step 6.8: B級增強 (歷史漁場/深度/月相)    ← [B-1~4] NEW
    Step 7: 融合排名 + EEZ
    Step 8: 航線
    Step 9: 輸出
    """

    def __init__(
        self,
        lat_range=(5, 35),
        lon_range=(120, 175),
        species=None,
        output_dir="output",
        vessel_pos=None,
        flag_state="TWN",
        target_date=None,
    ):
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.species = species or [
            "skipjack", "yellowfin", "bigeye", "albacore",
            "squid_todarodes", "squid_ommastrephes",
        ]
        self.output_dir = output_dir
        self.vessel_pos = vessel_pos or (25.13, 121.74)
        self.flag_state = flag_state

        # ── v10.3 歷史日期分析模式 ──
        if target_date:
            if isinstance(target_date, str):
                self.target_date = datetime.strptime(target_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            else:
                self.target_date = target_date
            log.info(f"  📅 歷史模式: {self.target_date.strftime('%Y-%m-%d')}")
        else:
            self.target_date = None

        self.fetcher = OceanDataFetcher(lat_range, lon_range, target_date=self.target_date)
        self.do_analyzer = DissolvedOxygenAnalyzer()
        # pp_engine removed — ForageEngine provides same VGPM+forage functionality
        self.physics_engine = OceanPhysicsEngine()
        # EEZChecker uses classmethods, no instance needed
        self.route_planner = FuelOptimalRouter()

        # [P0-1/P0-2] 商業核心: 初始化 + 後面真的呼叫
        self.commercial_hsi = CommercialGradeHSI()
        self.accuracy_booster = UltimateAccuracyBooster()
        self.explainer = ExplainableHSI()
        self.gebco = GEBCOFeatures() if HAS_GEBCO else None

        # [v10.4] GreenFish Lite 引擎初始化
        if GREENFISH_OK:
            self.thermocline = ThermoclineFetcher()
            self.gf_eddy = GFEddyDetector()
            self.forage_engine = ForageEngine()
            self.dvm_model = DVMModel()
            self.greenfish_hsi = GreenFishLiteHSI()
            self.vessel_validator = VesselLightValidator()
            log.info("  ✅ GreenFish Lite: 6 modules loaded")
        else:
            log.info(f"  ⚠️ GreenFish Lite unavailable: {log_gf_err if 'log_gf_err' in dir() else 'import error'}")

        # ── v10.3 安全防護引擎初始化 ──
        if SAFETY_OK:
            self.weather_fetcher = WeatherFetcher()
            self.typhoon_tracker = TyphoonTracker()
            self.wave_fetcher = WaveFetcher()
            self.rainfall_fetcher = RainfallFetcher()
            self.mld_forecaster = MLDForecaster()
            self.typhoon_features = TyphoonFeatureEngineer()
            self.typhoon_features.load_ibtracs()  # 載入歷史颱風資料
            log.info("  ✅ Safety/Weather: 6 modules loaded")
        else:
            self.weather_fetcher = None
            log.info("  ⚠️ Safety/Weather modules unavailable")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # ── [Phase 9] ML 44-feature 模型載入 (FishingStackingModel) ──
        self.ml44_models = {}  # 44-feature models (Phase 9)
        self.ml44_available = False
        try:
            from engine.ml.stacking_ensemble import FishingStackingModel
            ml_species = ["yellowfin", "bigeye", "skipjack", "albacore"]
            for sp in ml_species:
                fsm = FishingStackingModel(species=sp, model_dir="models")
                if fsm.safe_model_load():
                    self.ml44_models[sp] = fsm
                    self.ml44_available = True
            if self.ml44_models:
                log.info(f"  ✅ ML 44-feature models loaded: {list(self.ml44_models.keys())}")
            else:
                log.info("  ⚠️ No 44-feature ML models found")
        except Exception as e:
            log.warning(f"  ML 44-feature init failed: {e}")

        # ── [A-1] ML 模型載入 — Legacy 12-feature (fallback) ──
        self.ml_models = {}
        self.ml_scalers = {}  # GFW-trained scalers from stacking pkl
        self.ml_pipelines = {}
        self.ml_available = False
        if ML_LIBS_OK:
            ml_species = ["yellowfin", "bigeye", "skipjack", "albacore"]
            for sp in ml_species:
                if sp in self.ml44_models:
                    continue  # Skip — already have 44-feature model
                model_path = Path(f"ml_system/models/stacking_{sp}.pkl")
                pipe_path = Path(f"ml_system/models/feature_pipeline_{sp}.pkl")
                if model_path.exists():
                    try:
                        loaded = joblib.load(model_path)
                        # GFW-trained models are saved as dict {model, scaler, species}
                        if isinstance(loaded, dict) and "model" in loaded:
                            self.ml_models[sp] = loaded["model"]
                            self.ml_scalers[sp] = loaded.get("scaler")
                            log.info(f"  ML model {sp} (12-feat): unwrapped dict → {type(loaded['model']).__name__}")
                        else:
                            self.ml_models[sp] = loaded
                        if pipe_path.exists():
                            self.ml_pipelines[sp] = joblib.load(pipe_path)
                        self.ml_available = True
                    except Exception as e:
                        log.warning(f"  ML model {sp} load failed: {e}")
            if self.ml_models:
                log.info(f"  ✅ ML 12-feature fallback loaded: {list(self.ml_models.keys())}")
                # 註冊 ML 模型到 SHAP 解釋器
                if SHAP_OK:
                    self._shap_explainer = get_shap_explainer()
                    _GFW_FEATURES = [
                        "sst", "chl", "ssh", "do", "current_speed",
                        "front_strength", "eddy_strength", "phi",
                        "bathy_depth", "bathy_slope", "dist_seamount", "dist_shelf_break",
                    ]
                    for sp, model in self.ml_models.items():
                        feat_names = _GFW_FEATURES
                        if sp in self.ml_pipelines and hasattr(self.ml_pipelines[sp], 'feature_names_'):
                            feat_names = list(self.ml_pipelines[sp].feature_names_)
                        self._shap_explainer.register_model(sp, model, feat_names)
                else:
                    self._shap_explainer = None
            else:
                if not self.ml44_available:
                    log.info("  ⚠️ No ML models found (science-only mode)")
                self._shap_explainer = None

        log.info("=" * 65)
        log.info(f"  OceanMaster v{VERSION}")
        log.info(f"  Range: {lat_range[0]}-{lat_range[1]}N, {lon_range[0]}-{lon_range[1]}E")
        log.info(f"  Species: {len(self.species)} | Vessel: {self.vessel_pos}")
        log.info(f"  Tech: 24 (13 commercial + 6 GreenFish + 5 enhanced)")
        _ml44_str = f"ML-44: ✅ ({len(self.ml44_models)} species)" if self.ml44_available else "ML-44: ❌"
        _ml12_str = f"ML-12: ✅ ({len(self.ml_models)} species)" if self.ml_available else "ML-12: ❌"
        log.info(f"  ML Ensemble: {_ml44_str} | {_ml12_str}")
        log.info(f"  GreenFish Lite: {'✅ ACTIVE' if GREENFISH_OK else '❌ NOT LOADED'}")
        log.info(f"  Enhanced: {'✅' if ENHANCED_OK else '❌'} (ENSO/Lunar/HistPrior/Depth)")
        log.info(f"  Safety/Weather: {'✅' if SAFETY_OK else '❌'} (Typhoon/Wave/Pressure/TCHP)")
        log.info("=" * 65)

    async def run_full(self) -> dict:
        t0 = time.time()
        now = self.target_date or datetime.now(timezone.utc)
        month = now.month
        doy = now.timetuple().tm_yday
        results = {"version": VERSION, "timestamp": now.isoformat()}
        if self.target_date:
            results["analysis_mode"] = "historical"
            results["target_date"] = self.target_date.strftime("%Y-%m-%d")

        # ═══════════════════════════════════════════
        # Step 1: 真實數據 [P1-1: 零 np.random]
        # ═══════════════════════════════════════════
        log.info("\n  Step 1/9: Data fetch (real APIs)")
        data = await self.fetcher.fetch_all()

        # [A-3] 取得 ENSO ONI 指數
        self._oni = 0.0
        if ENHANCED_OK:
            try:
                import httpx
                async with httpx.AsyncClient(timeout=15) as ec:
                    self._oni = await fetch_enso_oni(ec)
            except Exception:
                self._oni = await fetch_enso_oni(None)

        lats = data["lats"]
        lons = data["lons"]
        sst = data["sst"]
        chl = data["chl"]
        u_current = data["u_current"]
        v_current = data["v_current"]
        wind_u = data["wind_u"]
        wind_v = data["wind_v"]
        salinity = data["salinity"]
        do_surface = data.get("do_surface")
        temp_3d = data.get("temp_3d")
        ssh = data.get("ssh")

        ny, nx = len(lats), len(lons)
        log.info(f"  Grid: {ny}x{nx} = {ny*nx} pts")

        # ═══════════════════════════════════════════
        # Step 1.5: 氣象安全數據擷取 [v10.3]
        # ═══════════════════════════════════════════
        weather_data = None
        typhoon_alerts = []
        wave_data = None
        pressure_data = None
        tchp_result = None

        if SAFETY_OK and self.weather_fetcher:
            log.info("\n  Step 1.5/9: Weather & Safety data fetch")
            try:
                lat_range = (float(lats.min()), float(lats.max()))
                lon_range = (float(lons.min()), float(lons.max()))

                # 並行擷取: 氣象 + 颱風 + 波高
                weather_task = self.weather_fetcher.fetch_marine_weather(
                    lat_range, lon_range, ny=ny, nx=nx
                )
                typhoon_task = self.typhoon_tracker.fetch_active_typhoons()

                import asyncio as _aio
                weather_data, typhoon_alerts = await _aio.gather(
                    weather_task, typhoon_task, return_exceptions=True
                )
                if isinstance(weather_data, Exception):
                    log.warning(f"  Weather fetch error: {weather_data}")
                    weather_data = None
                if isinstance(typhoon_alerts, Exception):
                    log.warning(f"  Typhoon fetch error: {typhoon_alerts}")
                    typhoon_alerts = []

                # 波高 (WW3 → Open-Meteo → 風速估算 → 氣候態)
                wind_speed = None
                if weather_data and "wind_speed" in weather_data:
                    wind_speed = weather_data["wind_speed"]
                elif wind_u is not None and wind_v is not None:
                    wind_speed = np.sqrt(wind_u**2 + wind_v**2)

                wave_data = await self.wave_fetcher.fetch_wave_height(
                    lat_range, lon_range, lats, lons,
                    wind_speed=wind_speed,
                )

                # 氣壓場特徵
                if weather_data and "pressure_msl" in weather_data:
                    pressure_data = compute_pressure_features(
                        weather_data["pressure_msl"], lats, lons
                    )

                # TCHP (基於現有 HYCOM temp_3d)
                if temp_3d is not None and temp_3d.ndim == 3:
                    std_depths = np.array([0, 10, 20, 30, 50, 75, 100,
                                           125, 150, 200, 250, 300, 400, 500])[:temp_3d.shape[0]]
                    tchp_result = compute_tchp(temp_3d, std_depths, lats, lons)
                    alert_msg = tchp_alert_message(tchp_result)
                    if alert_msg:
                        log.warning(f"  {alert_msg}")

                # 安全網格評估
                wh_grid = wave_data["wave_height"] if wave_data else None
                ws_grid = wind_speed
                p_grid = weather_data["pressure_msl"] if weather_data else None
                self._safety_grid = assess_grid_safety(
                    lats, lons,
                    typhoon_alerts=typhoon_alerts,
                    wave_height=wh_grid,
                    wind_speed=ws_grid,
                    pressure_msl=p_grid,
                )

                # 颱風 ML 特徵
                self._typhoon_ml = self.typhoon_features.compute_features_grid(
                    lats, lons, now
                )

                # [v10.4] SSS 鹽度特徵
                self._sss_data = None
                if SALINITY_OK:
                    try:
                        sss_fetcher = SalinityFetcher()
                        self._sss_data = await sss_fetcher.fetch_sss_features(
                            lat_range, lon_range, lats, lons,
                            fallback_salinity=salinity,
                        )
                    except Exception as e:
                        log.warning(f"  SSS fetch error: {e}")

                n_typhoons = len(typhoon_alerts) if isinstance(typhoon_alerts, list) else 0
                wave_src = wave_data.get('source', '?') if wave_data else '?'
                sss_src = self._sss_data.get('source', '?') if self._sss_data else '?'
                log.info(f"  ✅ Safety: {n_typhoons} typhoons, wave={wave_src}, sss={sss_src}")

            except Exception as e:
                log.warning(f"  Step 1.5 error (non-fatal): {e}")
                self._safety_grid = None
                self._typhoon_ml = None
                self._sss_data = None
        else:
            self._safety_grid = None
            self._typhoon_ml = None
            self._sss_data = None

        # ── [v10.4] CHL Gap-filling ── NaN > 30% 時用空間插值填補
        chl_nan_ratio = np.sum(np.isnan(chl)) / max(chl.size, 1)
        if chl_nan_ratio > 0.30:
            log.warning(f"  CHL NaN ratio = {chl_nan_ratio:.1%} — applying spatial gap-fill")
            try:
                from scipy.ndimage import generic_filter
                def _nanmean_filter(x):
                    v = x[~np.isnan(x)]
                    return np.nanmean(v) if len(v) > 0 else np.nan

                # 多次迭代填補，由近及遠
                filled = chl.copy()
                for radius in [3, 5, 9]:
                    still_nan = np.isnan(filled)
                    if not np.any(still_nan):
                        break
                    smoothed = generic_filter(filled, _nanmean_filter, size=radius, mode='constant', cval=np.nan)
                    filled = np.where(still_nan, smoothed, filled)

                # 仍有 NaN 的用全域中位數
                remaining_nan = np.isnan(filled)
                if np.any(remaining_nan):
                    global_median = np.nanmedian(filled)
                    if np.isfinite(global_median):
                        filled[remaining_nan] = global_median
                    else:
                        filled[remaining_nan] = 0.3  # 開洋合理預設值

                new_nan_ratio = np.sum(np.isnan(filled)) / max(filled.size, 1)
                log.info(f"  CHL gap-fill: {chl_nan_ratio:.1%} → {new_nan_ratio:.1%} NaN")
                chl = filled
            except ImportError:
                log.warning("  scipy not available, skipping CHL gap-fill")

        # ── [商用] SST Gap-filling ── NaN > 50% 時用空間插值填補
        sst_nan_ratio = np.sum(np.isnan(sst)) / max(sst.size, 1)
        if sst_nan_ratio > 0.30:
            log.warning(f"  SST NaN ratio = {sst_nan_ratio:.1%} — applying spatial gap-fill")
            try:
                from scipy.ndimage import generic_filter
                def _nanmean_sst(x):
                    v = x[~np.isnan(x)]
                    return np.nanmean(v) if len(v) > 0 else np.nan

                filled_sst = sst.copy()
                for radius in [3, 5, 9, 15]:
                    still_nan = np.isnan(filled_sst)
                    if not np.any(still_nan):
                        break
                    smoothed = generic_filter(filled_sst, _nanmean_sst, size=radius, mode='constant', cval=np.nan)
                    filled_sst = np.where(still_nan, smoothed, filled_sst)

                remaining_nan = np.isnan(filled_sst)
                if np.any(remaining_nan):
                    global_median = np.nanmedian(filled_sst)
                    if np.isfinite(global_median):
                        filled_sst[remaining_nan] = global_median
                    else:
                        filled_sst[remaining_nan] = 25.0  # 西太平洋合理預設

                new_sst_nan = np.sum(np.isnan(filled_sst)) / max(filled_sst.size, 1)
                log.info(f"  SST gap-fill: {sst_nan_ratio:.1%} → {new_sst_nan:.1%} NaN")
                sst = filled_sst
            except ImportError:
                log.warning("  scipy not available, skipping SST gap-fill")

        # ═══════════════════════════════════════════
        # Step 2: 物理演算法
        # ── [v10.4] 資料品質追蹤 ──
        data_sources = data.get("data_sources", {})
        currents_source = data_sources.get("CURRENTS", "unknown")
        currents_is_clim = "CLIM" in str(currents_source).upper()
        if currents_is_clim:
            log.warning("  ⚠️ 海流為氣候態 → EKE/渦旋偵測將受限")

        # ═══════════════════════════════════════════
        log.info("\n  Step 2/9: Physical algorithms")

        # [P0-3] 修正: 傳入 (sst, lats, lons) → 回傳 Dict
        sst_front_result = detect_sst_fronts(sst, lats, lons)
        front_strength = sst_front_result["front_strength"]

        # [P0-3] 修正: 傳入 (chl, lats, lons)
        chl_front_result = compute_boa_gradient(chl, lats, lons)

        # [P0-3] 修正: 回傳 Dict，取 ["ftle"]
        ftle_result = compute_ftle(u_current, v_current, lats, lons)
        ftle_field = ftle_result["ftle"]

        # [P0-3] 修正: (temp_3d, depths, lats, lons)
        thermocline_result = None
        if temp_3d is not None:
            depths_arr = np.array([0, 50, 100, 200, 300, 500])[:temp_3d.shape[0]]
            thermocline_result = compute_thermocline(temp_3d, depths_arr, lats, lons)

        log.info(f"  SST fronts: {np.nansum(front_strength > 0.3):.0f} strong pts")
        log.info(f"  FTLE mean: {np.nanmean(ftle_field):.4f}")

        # v10 物理量
        physics = self.physics_engine.analyze_full(
            u_current, v_current, salinity,
            wind_u, wind_v, lats,
            species_list=self.species[:4],
        )

        # 海底地形 (GEBCO enriched)
        bathy = BathymetryAnalyzer.generate_bathymetry(lats, lons)
        depth_si = {}
        for sp in self.species:
            depth_si[sp] = BathymetryAnalyzer.compute_depth_suitability(bathy, sp)

        # GEBCO 進階特徵 (供 ML FeatureEngineer 使用)
        gebco_feats = {}
        if self.gebco is not None:
            try:
                gebco_feats = self.gebco.compute_features(lats, lons, bathy=bathy)
                log.info(f"  GEBCO: depth=[{np.nanmin(bathy):.0f},{np.nanmax(bathy):.0f}]m, "
                         f"slope avg={np.nanmean(gebco_feats.get('slope', 0)):.2f}°")
            except Exception as e:
                log.debug(f"  GEBCO features: {e}")

        # ═══════════════════════════════════════════
        # Step 2.5: GreenFish Lite — 物理層升級
        # ═══════════════════════════════════════════
        gf_z20 = gf_mld = gf_eddy_result = gf_forage = gf_feeding = None
        if GREENFISH_OK:
            log.info("\n  Step 2.5: GreenFish Lite (Z20/MLD/Eddy/Forage/DVM)")

            # ── Z20 + MLD: GLORYS12 → HYCOM → 氣候態 ──
            glorys_result = self.thermocline.fetch_glorys12(lats, lons, month)
            if glorys_result is not None:
                gf_z20 = glorys_result["z20"]
                gf_mld = glorys_result["mld"]
                log.info(f"  ✅ Z20/MLD from {glorys_result.get('source', 'GLORYS12')}")
            elif temp_3d is not None and temp_3d.ndim == 3 and temp_3d.shape[0] >= 3:
                # 若 HYCOM 解析度太低 (< 目標格點的 25%)，用氣候態更可靠
                hycom_coverage = (temp_3d.shape[1] * temp_3d.shape[2]) / max(ny * nx, 1)
                if hycom_coverage < 0.01:
                    log.warning(f"  HYCOM 3D coverage too low ({hycom_coverage:.1%}) → climatology")
                    gf_z20 = ThermoclineFetcher.climatology_z20(lats, lons, month)
                    gf_mld = ThermoclineFetcher.climatology_mld(lats, lons, month)
                else:
                    tc_result = self.thermocline.compute_from_temp3d(temp_3d, lats, lons)
                    gf_z20 = tc_result["z20"]
                    gf_mld = tc_result["mld"]
                    log.info(f"  Z20: {np.nanmean(gf_z20):.0f}m | MLD: {np.nanmean(gf_mld):.0f}m")
            else:
                gf_z20 = ThermoclineFetcher.climatology_z20(lats, lons, month)
                gf_mld = ThermoclineFetcher.climatology_mld(lats, lons, month)
                log.info(f"  Z20/MLD: climatology (no 3D data)")

            # ── 渦旋偵測 ──
            if ssh is not None and u_current is not None and not currents_is_clim:
                gf_eddy_result = self.gf_eddy.detect(ssh, u_current, v_current, lats, lons)
                log.info(f"  Eddies: {np.sum(gf_eddy_result['eddy_core'])} core pts")
            elif currents_is_clim:
                gf_eddy_result = self.gf_eddy._empty_result(ny, nx)
                log.warning(f"  Eddies: skipped (climatology currents → EKE≡0)")
            else:
                gf_eddy_result = self.gf_eddy._empty_result(ny, nx)

            # ── 餌料場推算 ──
            gf_forage = self.forage_engine.compute(
                chl, sst, lats,
                z20=gf_z20, mld=gf_mld, month=month,
            )
            log.info(f"  NPP: {np.nanmean(gf_forage['npp']):.0f} mgC/m²/day")

            # ── DVM 覓食指數 (每個魚種) ──
            gf_feeding = {}
            for sp in self.species:
                sp_key = sp.split("_")[0] if "squid" not in sp else sp  # normalize
                if sp_key in ("squid_todarodes", "squid_ommastrephes"):
                    gf_feeding[sp] = gf_forage["forage_total"]  # 魷魚用總餌場
                else:
                    try:
                        gf_feeding[sp] = self.dvm_model.compute_feeding_index(
                            sp, gf_forage["forage_surface"], gf_forage["forage_deep"],
                            gf_z20, gf_mld, sst, hour_utc=now.hour,
                        )
                    except Exception as e:
                        log.warning(f"  DVM {sp}: {e}")
                        gf_feeding[sp] = gf_forage["forage_total"].copy()

        # ═══════════════════════════════════════════
        # Step 3: 生態分析
        # ═══════════════════════════════════════════
        log.info("\n  Step 3/9: Ecology (DO + ForageEngine)")

        do_result = self.do_analyzer.analyze(lats, lons, month=month, sst=sst)

        # 使用 ForageEngine (已在 Step 2.5 計算) 取代 PrimaryProductionEngine
        # 建立 pp_result 兼容結構給後續使用
        if gf_forage is not None and gf_feeding is not None:
            pp_result = {
                "npp": gf_forage["npp"],
                "forage_index": {sp: gf_feeding.get(sp, gf_forage["forage_total"])
                                 for sp in self.species[:4]},
            }
            log.info(f"  Forage: from GreenFish ForageEngine (unified)")
        else:
            # GreenFish 不可用時，用簡化估算
            from engine.forage_engine import ForageEngine
            _fe = ForageEngine()
            _forage = _fe.compute(chl, sst, lats, month=month)
            pp_result = {
                "npp": _forage["npp"],
                "forage_index": {sp: _forage["forage_total"] for sp in self.species[:4]},
            }
            log.info(f"  Forage: standalone ForageEngine (no GreenFish)")

        # ═══════════════════════════════════════════
        # Step 4: v8 基礎 HSI
        # ═══════════════════════════════════════════
        log.info("\n  Step 4/9: Base HSI (v8)")

        # [P0-7] 修正: get_moon_phase(year, month, day)
        moon = get_moon_phase(now.year, now.month, now.day)

        # [B-1] 月相漁獲因子
        self._lunar = {"fishing_factor": 1.0, "species_factors": {}}
        if ENHANCED_OK:
            self._lunar = compute_lunar_fishing_factor(now.year, now.month, now.day)

        # [P0-5] 修正: 傳 Dict 而非 kwargs
        ocean_features = {
            "sst": sst, "chl": chl, "ssh": ssh,
            "front_strength": front_strength,
            "ftle": ftle_field,
            "thermocline_depth": thermocline_result["thermocline_depth"] if thermocline_result else None,
            "d20_depth": thermocline_result["d20_depth"] if thermocline_result else None,
            "viirs_lights": data.get("viirs_lights"),
            "lat": lats, "lon": lons,
            # v10.4 enriched features (for ML FeatureEngineer)
            "u": u_current, "v": v_current,
            "mld": gf_mld,
            "moon_phase": moon,
            "eddy_core": gf_eddy_result["eddy_core"] if gf_eddy_result else None,
            "bathy_depth": gebco_feats.get("depth"),
            "bathy_slope": gebco_feats.get("slope"),
            "dist_to_seamount": gebco_feats.get("dist_to_seamount"),
            "dist_to_shelf_break": gebco_feats.get("dist_to_shelf_break"),
        }
        hsi_results = compute_all_hsi(ocean_features, self.species, moon)

        # ═══════════════════════════════════════════
        # Step 5: 商業核心 HSI  ← [P0-1] 死代碼復活
        # ═══════════════════════════════════════════
        log.info("\n  Step 5/9: Commercial core HSI (13 techs)")

        ow_param = self._compute_okubo_weiss(u_current, v_current)

        for sp in self.species:
            if sp not in hsi_results or "hsi" not in hsi_results[sp]:
                continue
            v8_hsi = hsi_results[sp]["hsi"]

            # [P0-4] 修正: dict 取代 locals()
            feat = {
                "do_si": do_result["do_si"].get(sp, np.ones_like(v8_hsi) * 0.5),
                "forage": pp_result["forage_index"].get(sp, np.ones_like(v8_hsi) * 0.5),
                "d_si": depth_si.get(sp, np.ones_like(v8_hsi) * 0.5),
                "eke_si": physics.get("eke_si", np.ones_like(v8_hsi) * 0.5),
                "sal_si": physics.get("salinity_si", {}).get(sp, np.ones_like(v8_hsi) * 0.5),
                "upwell": physics.get("upwelling_index", np.zeros_like(v8_hsi)),
            }
            for key in feat:
                arr = feat[key]
                if arr.shape != v8_hsi.shape:
                    try:
                        feat[key] = np.broadcast_to(arr, v8_hsi.shape).copy()
                    except ValueError:
                        feat[key] = np.ones_like(v8_hsi) * 0.5

            # DO field for Phi engine
            do_field = do_result.get("do_surface", np.ones_like(sst) * 5.0)
            if do_field.shape != sst.shape:
                do_field = np.ones_like(sst) * 5.0

            forage_for_comm = feat["forage"]
            if forage_for_comm.shape != sst.shape:
                forage_for_comm = np.ones_like(sst) * 0.5

            # ──────────────────────────────────────
            # [P0-1] 真的呼叫 CommercialGradeHSI
            # ──────────────────────────────────────
            comm_result = self.commercial_hsi.compute_ultimate_hsi(
                sst=sst, chl=chl,
                do_surface=do_field,
                forage_index=forage_for_comm,
                front_strength=front_strength,
                species=sp,
                ow=ow_param,
                sla=ssh,
                eke_si=feat["eke_si"] if feat["eke_si"].shape == sst.shape else None,
                depth_si=feat["d_si"] if feat["d_si"].shape == sst.shape else None,
                salinity=salinity,
                temp_3d=temp_3d,
                do_3d=None,
            )

            comm_hsi = comm_result["hsi"]

            # [P1-2] 融合: 權重和 = 1.0 (NaN 安全)
            v8_safe = np.nan_to_num(v8_hsi, nan=0.0)  # v8 NaN → 0 避免污染加權平均
            raw_blend = 0.35 * v8_safe + 0.65 * comm_hsi
            # 若 v8 全為 NaN，記錄但不影響計算 (已用 v8_safe)
            nan_count = np.sum(np.isnan(v8_hsi))
            if nan_count > 0:
                log.warning(f"  {sp}: v8 produced NaN → using comm_hsi only ({nan_count} pts)")
            blended = np.clip(raw_blend, 0.0, 1.0).astype(np.float32)

            # ── [v10.4] GreenFish Lite HSI 增強 ──
            if GREENFISH_OK and gf_z20 is not None:
                try:
                    gf_result = self.greenfish_hsi.compute(
                        species=sp,
                        sst=sst,
                        phi_viability=comm_result["phi_viability"],
                        feeding_index=gf_feeding.get(sp) if gf_feeding else None,
                        z20=gf_z20,
                        mld=gf_mld,
                        eddy_edge=gf_eddy_result["eddy_edge"] if gf_eddy_result else None,
                        eke=gf_eddy_result["eke"] if gf_eddy_result else None,
                        front_strength=front_strength,
                        chl=chl,
                    )
                    gf_hsi = gf_result["hsi"] / 100.0  # 0-100 → 0-1
                    # 三層融合: v8(15%) + commercial(35%) + GreenFish(50%)
                    blended = np.clip(
                        0.15 * v8_safe + 0.35 * comm_hsi + 0.50 * gf_hsi,
                        0.0, 1.0
                    ).astype(np.float32)
                    # 保存 GreenFish 各子指數
                    hsi_results[sp]["gf_components"] = gf_result["components"]
                    hsi_results[sp]["gf_hsi"] = gf_hsi
                    log.info(f"  {sp}: GreenFish HSI={np.nanmean(gf_hsi):.3f} (integrated)")
                except Exception as e:
                    log.warning(f"  {sp}: GreenFish HSI failed ({e}), using comm blend")

            hsi_results[sp]["hsi"] = blended
            hsi_results[sp]["phi"] = comm_result["phi"]
            hsi_results[sp]["phi_viability"] = comm_result["phi_viability"]
            hsi_results[sp]["h_thermal"] = comm_result["h_thermal"]
            hsi_results[sp]["h_feeding"] = comm_result["h_feeding"]
            hsi_results[sp]["front_persistence"] = comm_result["front_persistence"]
            # 渦旋邊緣: 只在 GreenFish HSI 成功時才用新版
            gf_hsi_ok = "gf_hsi" in hsi_results.get(sp, {})
            hsi_results[sp]["eddy_edge"] = gf_eddy_result["eddy_edge"] if (gf_hsi_ok and gf_eddy_result is not None) else comm_result["eddy_edge"]
            hsi_results[sp]["do_si"] = feat["do_si"]
            hsi_results[sp]["forage_index"] = feat["forage"]
            hsi_results[sp]["depth_si"] = feat["d_si"]
            hsi_results[sp]["npp"] = gf_forage["npp"] if (GREENFISH_OK and gf_forage is not None) else pp_result["npp"]
            hsi_results[sp]["confidence"] = comm_result["confidence"]
            # GreenFish 額外數據
            if GREENFISH_OK:
                hsi_results[sp]["z20"] = gf_z20
                hsi_results[sp]["mld"] = gf_mld
                hsi_results[sp]["eke"] = gf_eddy_result["eke"] if gf_eddy_result else None
                hsi_results[sp]["feeding_index"] = gf_feeding.get(sp) if gf_feeding else None

            log.info(
                f"  {sp}: v8={np.nanmean(v8_hsi):.3f} "
                f"comm={np.nanmean(comm_hsi):.3f} "
                f"blend={np.nanmean(blended):.3f} "
                f"Phi={np.nanmean(comm_result['phi']):.2f}"
            )

        # ═══════════════════════════════════════════
        # Step 6: 精度增強  ← [P0-2] 死代碼復活
        # ═══════════════════════════════════════════
        log.info("\n  Step 6/9: Accuracy boost (anomaly/ENSO/SHAP)")

        # NaN 安全: 確保 currents 不是 NaN (氣候態回退可能產生 NaN)
        u_safe = np.where(np.isfinite(u_current), u_current, 0.0).astype(np.float32)
        v_safe = np.where(np.isfinite(v_current), v_current, 0.0).astype(np.float32)

        # [v10.3.1] 保存 Step 5 的有效 HSI 作為安全備份
        step5_hsi_backup = {}
        for sp in self.species:
            if sp in hsi_results and "hsi" in hsi_results.get(sp, {}):
                step5_hsi_backup[sp] = hsi_results[sp]["hsi"].copy()

        for sp in self.species:
            if sp not in hsi_results or "hsi" not in hsi_results[sp]:
                continue
            base_hsi = np.nan_to_num(hsi_results[sp]["hsi"].copy(), nan=0.0, posinf=1.0, neginf=0.0)

            # ──────────────────────────────────────
            # [P0-2] 真的呼叫 UltimateAccuracyBooster
            # ──────────────────────────────────────
            try:
                enhanced = self.accuracy_booster.enhance_hsi(
                    base_hsi=base_hsi,
                    sst=sst, chl=chl,
                    u_current=u_safe, v_current=v_safe,
                    lats=lats, lons=lons,
                    month=month, species=sp,
                    oni=self._oni,  # [A-3] 即時 ENSO ONI
                )
                enh_hsi = np.asarray(enhanced["enhanced_hsi"], dtype=np.float64)
                # NaN 防護: 若增強後產生 NaN, 回退到 base
                nan_pct = np.sum(~np.isfinite(enh_hsi)) / max(enh_hsi.size, 1)
                if nan_pct > 0.5:
                    log.warning(f"  {sp}: accuracy_booster produced {nan_pct:.0%} NaN → fallback to base HSI")
                    enh_hsi = np.where(np.isfinite(enh_hsi), enh_hsi, base_hsi)
                hsi_results[sp]["hsi"] = np.clip(
                    np.nan_to_num(enh_hsi, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0
                ).astype(np.float32)
                hsi_results[sp]["sst_anomaly"] = enhanced.get("sst_anomaly", np.zeros_like(base_hsi))
                hsi_results[sp]["chl_bloom"] = enhanced.get("chl_bloom", np.zeros_like(base_hsi))
                hsi_results[sp]["convergence"] = enhanced.get("convergence", np.zeros_like(base_hsi))
            except Exception as e:
                log.warning(f"  {sp}: accuracy_booster failed ({e}) → keeping base HSI")
                hsi_results[sp]["hsi"] = np.nan_to_num(base_hsi, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

            # [v10.3.1] 最終安全閘: 若仍然全 NaN，回退到 Step 5 備份
            final_arr = hsi_results[sp]["hsi"]
            if not np.any(np.isfinite(final_arr)) or np.nanmax(final_arr) == 0.0:
                if sp in step5_hsi_backup:
                    backup = np.nan_to_num(step5_hsi_backup[sp], nan=0.0).astype(np.float32)
                    if np.nanmax(backup) > 0:
                        hsi_results[sp]["hsi"] = backup
                        log.warning(f"  {sp}: Step 6 produced all-zero/NaN → restored Step 5 HSI (max={np.max(backup):.3f})")

            log.info(f"  {sp}: final={np.nanmean(hsi_results[sp]['hsi']):.3f}")

        # ═══════════════════════════════════════════
        # Step 6.5: ML Ensemble CPUE預測
        # [Phase 9] 44-feature FishingStackingModel → fallback 12-feature
        # ═══════════════════════════════════════════
        _ml_any = self.ml44_available or self.ml_available
        if _ml_any:
            log.info("\n  Step 6.5: ML Ensemble prediction")

            # 確保所有 ML 魚種都有 hsi_results 條目
            _all_ml_species = set(list(self.ml44_models.keys()) + list(self.ml_models.keys()))
            for sp in _all_ml_species:
                if sp not in hsi_results or "hsi" not in hsi_results.get(sp, {}):
                    log.info(f"  {sp}: creating ML-only placeholder (no base HSI)")
                    hsi_results[sp] = {
                        "hsi": np.full((ny, nx), 0.1, dtype=np.float32),
                        "phi": np.full((ny, nx), 5.0, dtype=np.float32),
                        "confidence": np.ones((ny, nx), dtype=np.float32),
                    }

            current_speed = np.sqrt(u_safe**2 + v_safe**2)

            for sp in set(list(self.species) + list(_all_ml_species)):
                if sp not in hsi_results or "hsi" not in hsi_results[sp]:
                    continue

                science_hsi = hsi_results[sp]["hsi"]
                phi_grid = hsi_results[sp].get("phi", np.ones_like(sst) * 3.0)

                # ── [Phase 9] Try 44-feature model first ──
                if sp in self.ml44_models:
                    try:
                        fsm = self.ml44_models[sp]
                        # predict() uses FeatureEngineer internally → 44 features
                        cpue_grid = fsm.predict(ocean_features)
                        cpue_grid = np.clip(cpue_grid, 0, 1)  # already [0, 1]

                        # Hybrid fusion: 40% science + 60% ML
                        fused_hsi = np.clip(
                            0.4 * np.nan_to_num(science_hsi, nan=0.0) + 0.6 * cpue_grid,
                            0.0, 1.0
                        ).astype(np.float32)

                        phi_mask = phi_grid < 2.0
                        fused_hsi[phi_mask] *= 0.5

                        hsi_results[sp]["hsi"] = fused_hsi
                        hsi_results[sp]["ml_cpue"] = cpue_grid
                        hsi_results[sp]["ml_confidence"] = cpue_grid
                        hsi_results[sp]["ml_type"] = "44-feature"

                        log.info(
                            f"  {sp}: ML-44 CPUE mean={np.nanmean(cpue_grid):.3f}, "
                            f"fused HSI={np.nanmean(fused_hsi):.3f}"
                        )
                        continue  # Done — skip 12-feature fallback
                    except Exception as e:
                        log.warning(f"  {sp}: ML-44 prediction failed ({e}), trying 12-feature fallback")

                # ── Legacy 12-feature fallback ──
                if sp not in self.ml_models:
                    continue

                try:
                    n_points = ny * nx

                    def safe_flat(arr, default=0.0):
                        if arr is None:
                            return np.full(n_points, default)
                        try:
                            a = np.broadcast_to(arr, (ny, nx)).copy()
                            return a.ravel()
                        except Exception:
                            return np.full(n_points, default)

                    sst_f = safe_flat(sst, 25.0)
                    chl_f = safe_flat(chl, 0.3)
                    ssh_f = safe_flat(ssh, 0.0)
                    do_f = safe_flat(do_surface, 5.0)
                    cs_f = safe_flat(current_speed, 0.3)
                    front_s = safe_flat(
                        sst_front_result.get("front_strength") if sst_front_result else None, 0.0
                    )
                    eke_grid = gf_eddy_result["eke"] if gf_eddy_result is not None else None
                    eddy_s = safe_flat(eke_grid, 0.01)
                    phi_f = safe_flat(phi_grid, 3.0)
                    bathy_d = safe_flat(ocean_features.get("bathy_depth"), -2000.0)
                    bathy_s = safe_flat(ocean_features.get("bathy_slope"), 1.0)
                    bathy_sm = safe_flat(ocean_features.get("dist_to_seamount"), 200.0)
                    bathy_sb = safe_flat(ocean_features.get("dist_to_shelf_break"), 100.0)

                    X_grid = np.column_stack([
                        sst_f, chl_f, ssh_f, do_f, cs_f,
                        front_s, eddy_s, phi_f,
                        bathy_d, bathy_s, bathy_sm, bathy_sb,
                    ])
                    X_grid = np.nan_to_num(X_grid, nan=0.0, posinf=0.0, neginf=0.0)

                    if sp in self.ml_scalers and self.ml_scalers[sp] is not None:
                        try:
                            X_grid = self.ml_scalers[sp].transform(X_grid)
                        except Exception as e:
                            log.warning(f"  {sp}: scaler transform failed ({e}), using raw features")
                    elif sp in self.ml_pipelines:
                        pipe = self.ml_pipelines[sp]
                        if isinstance(pipe, dict) and "scaler" in pipe:
                            try:
                                X_grid = pipe["scaler"].transform(X_grid)
                            except Exception as e:
                                log.warning(f"  {sp}: pipeline scaler failed ({e})")

                    cpue_pred = self.ml_models[sp].predict(X_grid)
                    cpue_grid = np.maximum(cpue_pred.reshape(ny, nx), 0.0)

                    medians = {"yellowfin": 45, "bigeye": 35, "skipjack": 100, "albacore": 32}
                    scales = {"yellowfin": 20, "bigeye": 12, "skipjack": 35, "albacore": 10}
                    med = medians.get(sp, 45)
                    scl = scales.get(sp, 20)
                    cpue_norm = 1.0 / (1.0 + np.exp(-(cpue_grid - med) / scl))
                    cpue_norm = np.nan_to_num(cpue_norm, nan=0.5, posinf=1.0, neginf=0.0)

                    fused_hsi = np.clip(
                        0.4 * np.nan_to_num(science_hsi, nan=0.0) + 0.6 * cpue_norm,
                        0.0, 1.0
                    ).astype(np.float32)

                    phi_mask = phi_grid < 2.0
                    fused_hsi[phi_mask] *= 0.5

                    hsi_results[sp]["hsi"] = fused_hsi
                    hsi_results[sp]["ml_cpue"] = cpue_grid
                    hsi_results[sp]["ml_confidence"] = cpue_norm
                    hsi_results[sp]["ml_type"] = "12-feature"

                    log.info(
                        f"  {sp}: ML-12 CPUE mean={np.nanmean(cpue_grid):.1f} kg/day, "
                        f"fused HSI={np.nanmean(fused_hsi):.3f}"
                    )
                except Exception as e:
                    log.warning(f"  ⚠️ ML prediction failed for {sp}: {e}")
                    log.warning("  Falling back to science-only HSI")
        else:
            log.info("\n  Step 6.5: Skipped (ML model not loaded)")

        # ═══════════════════════════════════════════
        # Step 6.8: B級精度增強 (ENSO/歷史/深度/月相)
        # ═══════════════════════════════════════════
        if ENHANCED_OK:
            log.info("\n  Step 6.8: B-level enhancements (ENSO/Lunar/HistPrior/Depth)")
            all_sp = set(list(self.species) + list(self.ml_models.keys()))
            for sp in all_sp:
                if sp not in hsi_results or "hsi" not in hsi_results[sp]:
                    continue
                lunar_f = self._lunar.get("species_factors", {}).get(sp, self._lunar.get("fishing_factor", 1.0))
                try:
                    enh = apply_all_enhancements(
                        hsi=hsi_results[sp]["hsi"],
                        species=sp,
                        lats=lats, lons=lons,
                        sst=sst, month=month,
                        oni=self._oni,
                        lunar_factor=lunar_f,
                    )
                    enh_hsi = enh["enhanced_hsi"]
                    # NaN 防護
                    enh_hsi = np.nan_to_num(enh_hsi, nan=0.0)
                    hsi_results[sp]["hsi"] = np.clip(enh_hsi, 0.0, 1.0).astype(np.float32)
                    hsi_results[sp]["enso_modifier"] = enh.get("enso_modifier", np.zeros_like(enh_hsi))
                    hsi_results[sp]["historical_prior"] = enh.get("historical_prior", np.zeros_like(enh_hsi))
                    hsi_results[sp]["depth_index"] = enh.get("depth_index", np.zeros_like(enh_hsi))
                except Exception as e:
                    log.warning(f"  {sp}: B-level enhancement failed ({e})")
        else:
            log.info("\n  Step 6.8: Skipped (enhanced_data_sources not available)")

        # ═══════════════════════════════════════════
        # Step 7: 融合排名 + EEZ
        # ═══════════════════════════════════════════
        log.info("\n  Step 7/9: Rank + EEZ")

        # [v10.3.1] 最終 NaN 清除 — 確保進入排名的 HSI 不含 NaN
        for sp in list(hsi_results.keys()):
            if "hsi" in hsi_results[sp]:
                arr = hsi_results[sp]["hsi"]
                if not np.all(np.isfinite(arr)):
                    cleaned = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
                    hsi_results[sp]["hsi"] = np.clip(cleaned, 0.0, 1.0)
                    n_fixed = np.sum(~np.isfinite(arr))
                    log.warning(f"  [NaN guard] {sp}: fixed {n_fixed} non-finite values before ranking")

        # [P0-6] 修正: fuse_and_rank(hsi, ocean_features_dict, lat, lon, top_n)
        fusion_features = {
            "front_strength": front_strength,
            "ftle": ftle_field,
            "viirs_lights": data.get("viirs_lights"),
            "bathy": bathy,  # [v10.5] 傳遞水深數據供物種-水深篩選
        }

        # [v10.3] 將低壓漁場加分加入 fusion
        if pressure_data and "low_pressure_bonus" in pressure_data:
            fusion_features["low_pressure_bonus"] = pressure_data["low_pressure_bonus"]

        fusion_result = fuse_and_rank(
            hsi_results, fusion_features,
            lats, lons,
            top_n=OUTPUT.get("top_n_hotspots", 30),
        )
        ranked = fusion_result.get("combined_hotspots", [])

        # [v10.3] 安全過濾 — 移除 AVOID 區域的 hotspots
        if SAFETY_OK and isinstance(typhoon_alerts, list):
            wh_grid = wave_data["wave_height"] if wave_data else None
            ranked = filter_hotspots_by_safety(
                ranked,
                typhoon_alerts=typhoon_alerts,
                wave_height_grid=wh_grid,
                lats=lats, lons=lons,
            )

        # [v10.4] 法規掃雷 — 剔除 MPA/RFMO 禁漁區座標
        if LEGAL_OK:
            n_before = len(ranked)
            ranked = filter_hotspots(ranked, remove_illegal=True)
            n_removed = n_before - len(ranked)
            if n_removed > 0:
                log.warning(f"  ⚖️ Legal filter: {n_removed}/{n_before} hotspots in restricted zones")

        # EEZ + 生態資訊
        for h in ranked:
            eez_result = EEZChecker.check_point(h["lat"], h["lon"])
            h["eez"] = eez_result.get("eez_name", "公海")
            h["eez_high_seas"] = eez_result.get("is_high_seas", True)
            li = np.argmin(np.abs(lats - h["lat"]))
            lj = np.argmin(np.abs(lons - h["lon"]))
            sp = h.get("species", "skipjack")

            if li < pp_result["npp"].shape[0] and lj < pp_result["npp"].shape[1]:
                h["npp"] = float(pp_result["npp"][li, lj])
            if do_surface is not None and li < do_surface.shape[0] and lj < do_surface.shape[1]:
                h["do_surface"] = float(do_surface[li, lj])
            if li < bathy.shape[0] and lj < bathy.shape[1]:
                h["depth_m"] = float(-bathy[li, lj])
            # [商用] SST — sea_conditions API 需要此欄位
            if sst is not None and li < sst.shape[0] and lj < sst.shape[1]:
                v = float(sst[li, lj])
                if np.isfinite(v):
                    h["sst"] = round(v, 2)

            # Phi
            if sp in hsi_results and "phi" in hsi_results[sp]:
                pg = hsi_results[sp]["phi"]
                if li < pg.shape[0] and lj < pg.shape[1]:
                    h["phi"] = float(pg[li, lj])

            # [v10.4] GreenFish Lite 附加資訊
            if GREENFISH_OK and sp in hsi_results:
                sd = hsi_results[sp]
                if "z20" in sd and sd["z20"] is not None and li < sd["z20"].shape[0] and lj < sd["z20"].shape[1]:
                    h["z20_m"] = float(sd["z20"][li, lj])
                if "mld" in sd and sd["mld"] is not None and li < sd["mld"].shape[0] and lj < sd["mld"].shape[1]:
                    h["mld_m"] = float(sd["mld"][li, lj])
                if "eke" in sd and sd["eke"] is not None and li < sd["eke"].shape[0] and lj < sd["eke"].shape[1]:
                    h["eke"] = float(sd["eke"][li, lj])
                if "feeding_index" in sd and sd["feeding_index"] is not None and li < sd["feeding_index"].shape[0] and lj < sd["feeding_index"].shape[1]:
                    h["feeding_index"] = float(sd["feeding_index"][li, lj])
                if "gf_hsi" in sd and sd["gf_hsi"] is not None and li < sd["gf_hsi"].shape[0] and lj < sd["gf_hsi"].shape[1]:
                    h["greenfish_hsi"] = round(float(sd["gf_hsi"][li, lj]) * 100, 1)

            # SHAP — 真實 SHAP 或 weight-based fallback
            if sp in hsi_results:
                sd = hsi_results[sp]
                fv = {}
                for fk in ["h_thermal", "phi_viability", "h_feeding", "front_persistence", "eddy_edge"]:
                    if fk in sd and sd[fk] is not None:
                        a = sd[fk]
                        if li < a.shape[0] and lj < a.shape[1]:
                            fv[fk] = float(a[li, lj])

                # 嘗試真正的 SHAP 解釋
                shap_done = False
                if SHAP_OK and self._shap_explainer is not None and sp in self._shap_explainer._explainers:
                    try:
                        # 收集 ML 特徵用於 SHAP
                        ml_features = dict(fv)
                        if li < sst.shape[0] and lj < sst.shape[1]:
                            ml_features["sst"] = float(sst[li, lj])
                        if do_surface is not None and li < do_surface.shape[0] and lj < do_surface.shape[1]:
                            ml_features["do_surface"] = float(do_surface[li, lj])
                        if li < pp_result["npp"].shape[0] and lj < pp_result["npp"].shape[1]:
                            ml_features["npp"] = float(pp_result["npp"][li, lj])
                        shap_result = self._shap_explainer.explain_point(ml_features, sp)
                        h["explain"] = shap_result["explanation_text"]
                        h["shap_values"] = shap_result["raw_shap_values"]
                        h["shap_method"] = shap_result["method"]
                        shap_done = True
                    except Exception as e:
                        log.debug(f"  SHAP for {sp} at ({h['lat']},{h['lon']}): {e}")

                if not shap_done and fv:
                    contribs = self.explainer.compute_factor_contributions(fv, h["score"])
                    sp_zh = SPECIES_PARAMS.get(sp, {}).get("name_zh", sp)
                    h["explain"] = self.explainer.generate_explanation_text(contribs, sp_zh, h["score"])

            # Distance
            dlat = h["lat"] - self.vessel_pos[0]
            dlon = h["lon"] - self.vessel_pos[1]
            dist = np.sqrt(dlat**2 + (dlon * np.cos(np.radians(self.vessel_pos[0])))**2) * 60
            h["distance_nm"] = round(float(dist), 1)
            h["travel_hours"] = round(dist / 10, 1)

            # [A-1] ML CPUE (v10.3) — 永遠儲存確保 API schema 一致
            h["ml_cpue_kg_day"] = 0.0
            h["ml_confidence"] = 0.0
            if self.ml_available and sp in hsi_results and "ml_cpue" in hsi_results[sp]:
                cpue_g = hsi_results[sp]["ml_cpue"]
                conf_g = hsi_results[sp]["ml_confidence"]
                if li < cpue_g.shape[0] and lj < cpue_g.shape[1]:
                    h["ml_cpue_kg_day"] = round(float(cpue_g[li, lj]), 1)
                    h["ml_confidence"] = round(float(conf_g[li, lj]), 3)

            # [B-1] 月相
            if ENHANCED_OK:
                h["lunar_phase"] = self._lunar.get("phase_name", "")
                h["lunar_factor"] = self._lunar.get("species_factors", {}).get(sp, 1.0)

        # [v10.4] VIIRS 漁船燈光驗證
        if GREENFISH_OK:
            viirs_data = data.get("viirs_lights", np.array([]).reshape(0, 3))
            if viirs_data is not None and len(viirs_data) > 0:
                validation = self.vessel_validator.validate(ranked, viirs_data)
                ranked = validation["spots_with_validation"]
                results["vessel_validation"] = {
                    "total_vessels": validation["total_vessels"],
                    "validated_spots": validation["validated_spots"],
                    "validation_score": validation["validation_score"],
                }

        results["hotspots"] = ranked
        log.info(f"  Ranked: {len(ranked)} hotspots")

        # ═══════════════════════════════════════════
        # Step 8: Route
        # ═══════════════════════════════════════════
        log.info("\n  Step 8/9: Route planning")
        results["route"] = None
        if ranked:
            try:
                from engine.navigation.route_planner import RoutePoint
                best = ranked[0]
                start_pt = RoutePoint(self.vessel_pos[0], self.vessel_pos[1], "vessel")
                dest_pt = RoutePoint(best["lat"], best["lon"], f"#{best.get('rank',1)} {best.get('species','')}")
                ocean_currents_dict = None
                if u_current is not None and v_current is not None:
                    ocean_currents_dict = {
                        "u": u_current, "v": v_current,
                        "lat": lats, "lon": lons,
                    }
                results["route"] = self.route_planner.plan_route(
                    start_pt, dest_pt, ocean_currents=ocean_currents_dict,
                )
            except Exception as e:
                log.warning(f"  Route: {e}")

        # ═══════════════════════════════════════════
        # Step 9: Output
        # ═══════════════════════════════════════════
        log.info("\n  Step 9/9: Output")
        ts = now.strftime("%Y%m%d_%H%M")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        try:
            kp = f"{self.output_dir}/OceanMaster_v103_{ts}.kml"
            generate_kml(ranked, lats, lons,
                         vessel_lat=self.vessel_pos[0],
                         vessel_lon=self.vessel_pos[1],
                         output_path=kp)
            results["kml_path"] = kp
        except Exception as e:
            log.warning(f"  KML: {e}")

        # ── HTML Interactive Map ──
        try:
            from engine.html_map_generator import generate_html_map
            html_path = f"{self.output_dir}/OceanMaster_v103_{ts}_Map.html"
            generate_html_map(
                ranked, html_path,
                vessel_pos=self.vessel_pos,
                timestamp=now.isoformat(),
                area={
                    "lat_min": self.lat_range[0], "lat_max": self.lat_range[1],
                    "lon_min": self.lon_range[0], "lon_max": self.lon_range[1],
                },
            )
            results["html_path"] = html_path
        except Exception as e:
            log.warning(f"  HTML Map: {e}")

        try:
            gp = f"{self.output_dir}/OceanMaster_v103_{ts}.geojson"
            fusion_dict = {"combined_hotspots": ranked}
            gj_path = generate_geojson(
                fusion_dict,
                route_result=results.get("route"),
                output_dir=self.output_dir,
            )
            results["geojson_path"] = gj_path
        except Exception as e:
            log.warning(f"  GeoJSON: {e}")

        try:
            hp = f"{self.output_dir}/hotspots_{ts}.json"
            with open(hp, 'w', encoding='utf-8') as f:
                json.dump(ranked, f, ensure_ascii=False, indent=2, default=str)
            # [商用] 同時寫非時間戳版 — lifespan 冷啟動優先讀此檔
            hp_latest = f"{self.output_dir}/hotspots.json"
            with open(hp_latest, 'w', encoding='utf-8') as f:
                json.dump(ranked, f, ensure_ascii=False, indent=2, default=str)
            results["hotspot_json"] = hp
        except Exception as e:
            log.warning(f"  JSON: {e}")

        elapsed = time.time() - t0
        results["elapsed_seconds"] = elapsed
        results["data_sources"] = data.get("data_sources", {})
        # [商用] 計算風場/洋流統計 — sea_conditions API 需要
        _ws, _wd, _cs, _cd = None, None, None, None
        try:
            if wind_u is not None and wind_v is not None:
                _ws_grid = np.sqrt(wind_u**2 + wind_v**2)
                _ws = round(float(np.nanmean(_ws_grid)), 1)
                _wd = round(float(np.nanmean(np.degrees(np.arctan2(-wind_u, -wind_v)) % 360)), 0)
            if u_current is not None and v_current is not None:
                _cs_grid = np.sqrt(u_current**2 + v_current**2)
                _cs = round(float(np.nanmean(_cs_grid)), 2)
                _cd = round(float(np.nanmean(np.degrees(np.arctan2(v_current, u_current)) % 360)), 0)
        except Exception:
            pass

        results["analysis_summary"] = {
            "version": VERSION,
            "elapsed": f"{elapsed:.1f}s",
            "hotspots": len(ranked),
            "commercial_core": True,
            "accuracy_booster": True,
            "greenfish_lite": GREENFISH_OK,
            "ml_ensemble": self.ml_available,
            "ml_species": list(self.ml_models.keys()) if self.ml_available else [],
            "enhanced": ENHANCED_OK,
            "enso_oni": self._oni,
            "lunar": self._lunar.get("phase_name", "N/A"),
            "data_sources": data.get("data_sources", {}),
            "wind_speed": _ws,
            "wind_dir": _wd,
            "current_speed": _cs,
            "current_dir": _cd,
        }

        # [商用] 持久化 analysis_summary — lifespan 冷啟動讀取
        try:
            summary_path = f"{self.output_dir}/analysis_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(results["analysis_summary"], f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            log.warning(f"  Summary persist: {e}")

        ml_tag = f"ML: {'✅ ' + str(len(self.ml_models)) + ' species' if self.ml_available else '❌'}"
        gf_tag = f"GreenFish: {'✅' if GREENFISH_OK else '❌'}"
        enh_tag = f"Enhanced: {'✅' if ENHANCED_OK else '❌'}"
        log.info(f"\n{'='*65}")
        log.info(f"  v{VERSION} done: {elapsed:.1f}s, {len(ranked)} hotspots")
        log.info(f"  Commercial: ACTIVE | GreenFish Lite: {'ACTIVE' if GREENFISH_OK else 'OFF'} | Booster: ACTIVE")
        log.info(f"  {ml_tag} | {gf_tag} | {enh_tag} | ONI: {self._oni:+.2f}")
        log.info(f"{'='*65}")
        return results

    @staticmethod
    def _compute_okubo_weiss(u, v):
        du_dy, du_dx = np.gradient(u)
        dv_dy, dv_dx = np.gradient(v)
        sn = du_dx - dv_dy
        ss = dv_dx + du_dy
        omega = dv_dx - du_dy
        return (sn**2 + ss**2 - omega**2).astype(np.float32)


# ═══════════════════════════════════════════════════
# FastAPI
# ═══════════════════════════════════════════════════

def create_app():
    try:
        from fastapi import FastAPI, Query
        from fastapi.responses import JSONResponse
    except ImportError:
        log.error("pip install fastapi uvicorn")
        return None

    app = FastAPI(title=f"OceanMaster v{VERSION}", version=VERSION)

    @app.get("/")
    async def root():
        return {"system": f"OceanMaster v{VERSION}", "status": "ready",
                "commercial_core": True, "accuracy_booster": True}

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": VERSION}

    @app.get("/analyze")
    async def analyze(
        lat_min: float = Query(5), lat_max: float = Query(35),
        lon_min: float = Query(120), lon_max: float = Query(175),
        vessel_lat: float = Query(25.13), vessel_lon: float = Query(121.74),
        date: str = Query(None, description="歷史日期 YYYY-MM-DD"),
    ):
        p = OceanMasterPipeline(
            lat_range=(lat_min, lat_max), lon_range=(lon_min, lon_max),
            vessel_pos=(vessel_lat, vessel_lon),
            target_date=date)
        r = await p.run_full()
        clean = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
        return JSONResponse(content=clean)

    return app


# ═══════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=f"OceanMaster v{VERSION} (ML全魚種+增強)")
    parser.add_argument("--run-now", action="store_true")
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--lat-min", type=float, default=5)
    parser.add_argument("--lat-max", type=float, default=35)
    parser.add_argument("--lon-min", type=float, default=120)
    parser.add_argument("--lon-max", type=float, default=175)
    parser.add_argument("--vessel-lat", type=float, default=25.13)
    parser.add_argument("--vessel-lon", type=float, default=121.74)
    parser.add_argument("--date", type=str, default=None,
                        help="歷史日期 YYYY-MM-DD (預設=今天)")
    parser.add_argument("--output", default="output")
    args = parser.parse_args()

    if args.api:
        app = create_app()
        if app:
            import uvicorn
            uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        p = OceanMasterPipeline(
            lat_range=(args.lat_min, args.lat_max),
            lon_range=(args.lon_min, args.lon_max),
            vessel_pos=(args.vessel_lat, args.vessel_lon),
            output_dir=args.output,
            target_date=args.date)
        r = asyncio.run(p.run_full())
        print(f"\nDone: {r.get('elapsed_seconds',0):.1f}s, "
              f"{len(r.get('hotspots',[]))} hotspots")


if __name__ == "__main__":
    main()
