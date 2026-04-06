"""
OceanMaster v13.2 — 台灣雙模式 (近海/遠洋) + 黑潮引擎
=======================================================
基於 v12.0，新增:

  [v13-1] 雙模式切換 (Dual-Scale): 近海 20-27N/118-125E + 遠洋 5-35N/120-175E
  [v13-2] 黑潮引擎 (Kuroshio): 軸線估計/入侵指數/邊緣渦旋/物種加成
  [v13-3] 4 新物種: 鬼頭刀/旗魚/竹筴魚/秋刀魚
  [v13-4] 漁業歷史數據接口 (CatchDataInterface)
  [v13-5] CLI 預設高雄港 (22.61, 120.28)

使用:
  python main_v10_3.py --run-now --mode nearshore
  python main_v10_3.py --run-now --mode offshore
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

# [v13.2-R7] scikit-learn version compatibility shim
# sklearn ≥1.6 renamed force_all_finite → ensure_all_finite in check_array().
# Models pickled with older sklearn trigger TypeError when loaded by newer sklearn.
try:
    import sklearn.utils.validation as _skval
    _orig_check_array = _skval.check_array
    def _compat_check_array(*args, **kwargs):
        if "force_all_finite" in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return _orig_check_array(*args, **kwargs)
    _skval.check_array = _compat_check_array
    logging.getLogger("OceanMaster").debug("sklearn compat shim applied (force_all_finite → ensure_all_finite)")
except Exception as e:
    log.debug(f"[降級] main_v10_3.py: {e}")


# ═══════════════════════════════════════════════════════
# [v13.2-P1] Pipeline Error Classification
#   CriticalDataError   → 停止管線 (e.g. SST 完全無資料)
#   DegradedDataWarning → 降級繼續 + 標記 data_quality
#   OptionalFeatureError→ 靜默跳過 (e.g. VIIRS/GFW/Himawari)
# ═══════════════════════════════════════════════════════
class CriticalDataError(Exception):
    """Fatal: pipeline cannot produce meaningful output. Abort."""
    pass

class DegradedDataWarning(Exception):
    """Non-fatal: core data degraded. Continue but flag data_quality."""
    pass

class OptionalFeatureError(Exception):
    """Non-fatal: optional enhancement unavailable. Silently skip."""
    pass

# [v13.2-fix] 自動載入 .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv 未安裝時忽略

# ─── CMEMS 帳號設定 (Copernicus Marine Service) ───
# 請在 .env 中設定 CMEMS_USER 和 CMEMS_PASS
if not os.environ.get("CMEMS_USER"):
    logging.getLogger("OceanMaster").warning(
        "CMEMS_USER/CMEMS_PASS 未設定 — CMEMS 數據源將無法使用。"
        " 請在 .env 中設定，或 export CMEMS_USER=xxx CMEMS_PASS=xxx"
    )

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
    compute_chl_gradient,     # (chl, lat, lon) → Dict [v15.3-fix: renamed from compute_boa_gradient to avoid shadowing]
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

# [v13.2-fix] logging 初始化提前 — 避免 import 錯誤時 NameError
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("OceanMaster")

# ── v10.3 安全防護與氣象特徵 ──
try:
    from engine.weather_fetcher import WeatherFetcher
    from engine.typhoon_tracker import TyphoonTracker
    from engine.typhoon_golden_zone import TyphoonGoldenZone
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
    log.info("RegulationChecker not available")

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

from config import OUTPUT, SPECIES_PARAMS, OPERATION_PRESETS, KAOHSIUNG_PORT

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
    from engine.micronekton_model import MicronektonModel
    GREENFISH_OK = True
except ImportError as _gf_err:
    GREENFISH_OK = False
    _log_gf_err = str(_gf_err)

# ── v13 黑潮引擎 ──
try:
    from engine.kuroshio_engine import KuroshioEngine
    KUROSHIO_OK = True
except ImportError:
    KUROSHIO_OK = False

# ── v13 漁業歷史數據接口 ──
try:
    from engine.catch_data_interface import CatchDataInterface
    CATCH_DATA_OK = True
except ImportError:
    CATCH_DATA_OK = False

# ── v13 HSI 動態權重 ──
try:
    from engine.hsi_dynamic_weights import compute_dynamic_weights, apply_weighted_hsi
    HSI_WEIGHTS_OK = True
except ImportError:
    HSI_WEIGHTS_OK = False

# ── v13 ENSO 校準 ──
try:
    from engine.enso_calibrator import ENSOCalibrator
    ENSO_OK = True
except ImportError:
    ENSO_OK = False

# ── v13 路線規劃 ──
try:
    from engine.route_planner_v2 import compute_route_cost, compute_all_routes
    ROUTE_OK = True
except ImportError:
    ROUTE_OK = False

# [v13.2-fix] logging 已在上方初始化 (L97)
# 此處僅保留 VERSION 定義

VERSION = "13.2"


# ── [v13.2-P2] 陸地遮罩 — 使用共享模組 ──
from engine.land_mask import KNOWN_LAND_BBOXES, near_known_land, DEFAULT_BUFFER_DEG

# Backward-compat aliases used by _is_ocean()
_KNOWN_LAND_BBOXES = KNOWN_LAND_BBOXES
_LAND_BUFFER_DEG = DEFAULT_BUFFER_DEG


def _near_known_land(lat: float, lon: float) -> bool:
    """[v13.2] Delegates to engine.land_mask.near_known_land()."""
    return near_known_land(lat, lon, buffer_deg=DEFAULT_BUFFER_DEG)



def _is_ocean(lat: float, lon: float, bathy: np.ndarray,
              lats: np.ndarray, lons: np.ndarray,
              min_depth: float = -200.0) -> bool:
    """[v13.2] 商業級陸地過濾 — 三層防護。

    Layer 1: bathy 深度門檻 (預設 -200m，排除淺海/珊瑚礁/大陸棚邊緣)
    Layer 2: 已知陸地 bbox + 28km buffer 幾何檢查
    Layer 3: 當 bathy 不可靠時 (None/空)，用幾何檢查兜底

    商業遠洋延繩釣船不會在 -200m 以淺的水域作業。
    """
    # Layer 3: 無 bathy 資料時退化為幾何檢查
    if bathy is None or bathy.size == 0:
        return not _near_known_land(lat, lon)

    li = int(np.argmin(np.abs(lats - lat)))
    lj = int(np.argmin(np.abs(lons - lon)))
    if li >= bathy.shape[0] or lj >= bathy.shape[1]:
        return not _near_known_land(lat, lon)

    depth_val = float(bathy[li, lj])

    # Layer 1: bathy 深度 — 商業漁場至少 -200m
    if depth_val > min_depth:
        return False

    # Layer 2: 即使 bathy 顯示足夠深，如果在已知陸地 bbox 內且 bathy 不夠深 (-500m)
    # 代表 ETOPO 解析度不足，可能是島間插值假象
    if _near_known_land(lat, lon) and depth_val > -500.0:
        return False

    return True


class OceanMasterPipeline:
    """
    v12.0 管線 — ML全魚種 + 精度增強

    Step 1: 真實數據抓取 + ENSO/ONI
    Step 2: 物理演算法 (鋒面/FTLE/溫躍層)
    Step 3: 生態分析 (DO/VGPM/營養鏈)
    Step 4: v8 基礎 HSI + 月相因子
    Step 5: 商業核心 HSI (Phi + SEAPODYM)     ← [P0-1]
    Step 6: 精度增強 (異常/ENSO/匯聚/SHAP)    ← [P0-2] + ONI即時
    Step 6.5: ML Ensemble CPUE預測 (4魚種)    ← [A-1] NEW
    Step 6.8: B級增強 (歷史漁場/深度/月相)    ← [B-1~4] NEW
    Step 7: 融合排名 + EEZ + 陸地過濾
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
        mode=None,  # [v13] "nearshore" | "offshore" | None
    ):
        # [v13] 雙模式切換
        if mode and mode in OPERATION_PRESETS:
            preset = OPERATION_PRESETS[mode]
            lat_range = preset["lat_range"]
            lon_range = preset["lon_range"]
            species = species or preset["species"]
            vessel_pos = vessel_pos or preset["vessel_pos"]
            log.info(f"  🌊 Mode: {preset['name']} ({preset['description']})")
        self.mode = mode

        self.lat_range = lat_range
        self.lon_range = lon_range
        self.species = species or [
            "skipjack", "yellowfin", "bigeye", "albacore",
            "mahi_mahi", "blue_marlin", "mackerel_scad",
        ]
        self.output_dir = output_dir
        self.vessel_pos = vessel_pos or KAOHSIUNG_PORT  # [v13] 高雄港
        self.flag_state = flag_state

        # ── v12.0 歷史日期分析模式 ──
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
            log.info(f"  ⚠️ GreenFish Lite unavailable: {_log_gf_err if '_log_gf_err' in vars() else 'import error'}")

        # ── v13 黑潮引擎 ──
        self.kuroshio = KuroshioEngine() if KUROSHIO_OK else None
        if self.kuroshio:
            log.info("  ✅ Kuroshio Engine: loaded")

        # ── v13 漁業歷史數據接口 ──
        self.catch_data = CatchDataInterface() if CATCH_DATA_OK else None
        if self.catch_data:
            log.info(f"  ✅ CatchData: {self.catch_data.record_count} records")

        # ── v12.0 安全防護引擎初始化 ──
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
            # [v13.2-P0] Import HMAC verification
            from engine.ml.stacking_ensemble import _verify_model_file
            ml_species = ["yellowfin", "bigeye", "skipjack", "albacore"]
            for sp in ml_species:
                # [v13.2] ML-12 loads independently as true fallback for ML-44
                model_path = Path(f"models/ml12_stacking_{sp}.pkl")
                pipe_path = Path(f"models/ml12_feature_pipeline_{sp}.pkl")
                if model_path.exists():
                    try:
                        # [v13.2-P0] Verify signature before loading
                        _verify_model_file(model_path)
                        loaded = joblib.load(model_path)
                        # GFW-trained models are saved as dict {model, scaler, species}
                        if isinstance(loaded, dict) and "model" in loaded:
                            self.ml_models[sp] = loaded["model"]
                            self.ml_scalers[sp] = loaded.get("scaler")
                            log.info(f"  ML model {sp} (12-feat): unwrapped dict → {type(loaded['model']).__name__}")
                        else:
                            self.ml_models[sp] = loaded
                        if pipe_path.exists():
                            _verify_model_file(pipe_path)
                            self.ml_pipelines[sp] = joblib.load(pipe_path)
                        self.ml_available = True
                    except RuntimeError as e:
                        log.error(f"  [P0] ML model {sp} signature verification failed: {e}")
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

    @staticmethod
    def _hours_ago(fetch_time, now):
        """Calculate hours since fetch_time relative to now."""
        if fetch_time is None:
            return None
        try:
            if isinstance(fetch_time, str):
                ft = datetime.fromisoformat(fetch_time.replace("Z", "+00:00"))
            else:
                ft = fetch_time
            return round((now - ft).total_seconds() / 3600, 1)
        except Exception:
            return None

    @staticmethod
    def _nanmean_gapfill(arr, radii, fallback_val):
        """[v13.2-fix] Normalized Convolution gap-fill — uniform_filter (50-100x faster than generic_filter)"""
        from scipy.ndimage import uniform_filter
        filled = arr.copy()
        for r in radii:
            still_nan = np.isnan(filled)
            if not np.any(still_nan):
                break
            tmp = np.where(still_nan, 0.0, filled)
            mask = (~still_nan).astype(np.float64)
            sum_f = uniform_filter(tmp, size=r, mode='constant', cval=0.0)
            cnt_f = uniform_filter(mask, size=r, mode='constant', cval=0.0)
            cnt_f = np.maximum(cnt_f, 1e-10)
            mean_f = sum_f / cnt_f
            fill_mask = still_nan & (cnt_f > 1e-10 / (r * r))
            filled = np.where(fill_mask, mean_f, filled)
        remaining = np.isnan(filled)
        if np.any(remaining):
            gm = np.nanmedian(filled)
            filled[remaining] = gm if np.isfinite(gm) else fallback_val
        return filled

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

        # [v13.2-P1] CMEMS 10-day forecast
        forecast_data = data.get("forecast", {})
        if forecast_data:
            log.info(f"  📅 CMEMS Forecast: {len(forecast_data)} days available")
            results["forecast_days"] = len(forecast_data)
            results["forecast"] = {
                str(day): {
                    "sst_mean": float(np.nanmean(fc["sst"])) if "sst" in fc else None,
                    "has_currents": "u" in fc and "v" in fc,
                    "has_salinity": "salinity" in fc,
                }
                for day, fc in forecast_data.items()
            }

        # [v13.2-P1] Himawari 每小時 SST 狀態
        sst_hourly = data.get("sst_hourly")
        sst_hourly_src = data.get("sst_hourly_source")
        if sst_hourly is not None:
            log.info(f"  🛰️ SST source: 每小時 Himawari ({sst_hourly_src}), "
                     f"mean={float(np.nanmean(sst_hourly)):.1f}°C")
            results["sst_source"] = sst_hourly_src
        else:
            log.info("  🛰️ SST source: 每日 OISST (Himawari unavailable)")
            results["sst_source"] = "OISST-daily"

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
        wind_speed = data.get("wind_speed")  # [v16] grid of wind speed m/s
        bottom_temp = data.get("bottom_temp")  # [v16] bottom temp from HYCOM 3D
        cmems_nppv = data.get("nppv")   # [v16.0] CMEMS BGC satellite NPP (mgC/m³/day)
        cmems_phyc = data.get("phyc")   # [v16.0] CMEMS BGC phytoplankton carbon (mmol/m³)

        ny, nx = len(lats), len(lons)
        log.info(f"  Grid: {ny}x{nx} = {ny*nx} pts")

        # [v13.2-P1] Critical data validation — SST is required for all HSI
        if sst is None or not np.any(np.isfinite(sst)):
            raise CriticalDataError(
                "SST data is completely missing or all NaN. "
                "All HSI calculations would be meaningless. Aborting pipeline."
            )

        # [v13.2-P1] Degraded data tracking
        _degraded_fields = []
        if chl is None or not np.any(np.isfinite(chl)):
            _degraded_fields.append("CHL")
            log.warning("  ⚠️ CHL data missing — forage/VGPM calculations will be impaired")
        if ssh is None or not np.any(np.isfinite(ssh)):
            # [v13.2-R7] Attempt WOA climatology SSH fallback before degrading
            try:
                from engine.data_fetcher_v2 import WOAClimatology
                _woa = WOAClimatology()
                _ssh_clim = _woa.ssh(lats, lons, datetime.now().month) if hasattr(_woa, 'ssh') else None
                if _ssh_clim is not None and np.any(np.isfinite(_ssh_clim)):
                    ssh = _ssh_clim
                    data["ssh"] = ssh
                    log.info("  SSH: using WOA climatology fallback (CMEMS unavailable)")
                else:
                    raise ValueError("WOA SSH not available")
            except Exception:
                _degraded_fields.append("SSH")
                log.warning("  ⚠️ SSH data missing — EKE/eddy detection will be impaired")
        if do_surface is None or not np.any(np.isfinite(do_surface)):
            _degraded_fields.append("DO")
            log.warning("  ⚠️ DO data missing — metabolic index will use defaults")
        if _degraded_fields:
            results["degraded_fields"] = _degraded_fields
            results["data_quality_warning"] = f"Degraded: {', '.join(_degraded_fields)}"

        # ═══════════════════════════════════════════
        # Step 1.5: 氣象安全數據擷取 [v12.0]
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

                import asyncio as _aio  # noqa: already imported; alias kept for gather
                weather_data, typhoon_alerts = await asyncio.gather(
                    weather_task, typhoon_task, return_exceptions=True
                )
                if isinstance(weather_data, Exception):
                    log.warning(f"  Weather fetch error: {weather_data}")
                    weather_data = None
                if isinstance(typhoon_alerts, Exception):
                    log.warning(f"  Typhoon fetch error: {typhoon_alerts}")
                    typhoon_alerts = []

                # 波高 (WW3 → Open-Meteo → 風速估算 → 氣候態)
                _wave_wind_speed = None
                if weather_data and "wind_speed" in weather_data:
                    _wave_wind_speed = weather_data["wind_speed"]
                elif wind_u is not None and wind_v is not None:
                    _wave_wind_speed = np.sqrt(wind_u**2 + wind_v**2)

                wave_data = await self.wave_fetcher.fetch_wave_height(
                    lat_range, lon_range, lats, lons,
                    wind_speed=_wave_wind_speed,
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
                        self._sss_data = await asyncio.wait_for(
                            sss_fetcher.fetch_sss_features(
                                lat_range, lon_range, lats, lons,
                                fallback_salinity=salinity,
                            ),
                            timeout=20  # 最多等 20 秒
                        )
                    except asyncio.TimeoutError:
                        log.warning("  SSS timeout — using HYCOM salinity fallback")
                        self._sss_data = {"source": "HYCOM-fallback", "sss": salinity}
                    except Exception as e:
                        log.warning(f"  SSS fetch error: {e}")

                n_typhoons = len(typhoon_alerts) if isinstance(typhoon_alerts, list) else 0
                wave_src = wave_data.get('source', '?') if wave_data else '?'
                sss_src = self._sss_data.get('source', '?') if self._sss_data else '?'
                log.info(f"  ✅ Safety: {n_typhoons} typhoons, wave={wave_src}, sss={sss_src}")

            except Exception as e:
                log.warning(f"  Step 1.5 skipped (OptionalFeatureError): {e}")
                self._safety_grid = None
                self._typhoon_ml = None
                self._sss_data = None
        else:
            self._safety_grid = None
            self._typhoon_ml = None
            self._sss_data = None

        # ─── [v17-P1] 升級 salinity: 優先用 CMEMS SSS，次用 HYCOM，最後 climatology ───
        _sal_source = "data_fetcher"
        if self._sss_data is not None and self._sss_data.get("sss") is not None:
            _sss_grid = self._sss_data["sss"]
            if isinstance(_sss_grid, np.ndarray) and _sss_grid.shape == salinity.shape:
                salinity = _sss_grid
                _sal_source = self._sss_data.get("source", "CMEMS-SSS")
            elif isinstance(_sss_grid, np.ndarray):
                # shape mismatch — try to resize
                try:
                    from scipy.ndimage import zoom
                    zoom_y = salinity.shape[0] / _sss_grid.shape[0]
                    zoom_x = salinity.shape[1] / _sss_grid.shape[1]
                    salinity = zoom(_sss_grid, (zoom_y, zoom_x), order=1).astype(np.float32)
                    _sal_source = self._sss_data.get("source", "CMEMS-SSS") + "-resized"
                except Exception:
                    pass  # keep original salinity
        log.info(f"  🧂 Salinity: source={_sal_source}, "
                 f"range=[{np.nanmin(salinity):.2f}, {np.nanmax(salinity):.2f}] psu, "
                 f"mean={np.nanmean(salinity):.2f} psu")

        # [v13.2-fix] B-1: 共用 gap-fill — uniform_filter 取代 generic_filter (快 50-100x)

        # ── CHL Gap-filling ── NaN > 30% 時填補
        chl_nan_ratio = np.sum(np.isnan(chl)) / max(chl.size, 1)
        if chl_nan_ratio > 0.30:
            log.warning(f"  CHL NaN ratio = {chl_nan_ratio:.1%} — applying spatial gap-fill")
            try:
                chl = self._nanmean_gapfill(chl, radii=[3, 5, 9], fallback_val=0.3)
                new_nan = np.sum(np.isnan(chl)) / max(chl.size, 1)
                log.info(f"  CHL gap-fill: {chl_nan_ratio:.1%} → {new_nan:.1%} NaN")
            except ImportError:
                log.warning("  scipy not available, skipping CHL gap-fill")

        # ── SST Gap-filling ── NaN > 30% 時填補
        sst_nan_ratio = np.sum(np.isnan(sst)) / max(sst.size, 1)
        if sst_nan_ratio > 0.30:
            log.warning(f"  SST NaN ratio = {sst_nan_ratio:.1%} — applying spatial gap-fill")
            try:
                sst = self._nanmean_gapfill(sst, radii=[3, 5, 9, 15], fallback_val=25.0)
                new_nan = np.sum(np.isnan(sst)) / max(sst.size, 1)
                log.info(f"  SST gap-fill: {sst_nan_ratio:.1%} → {new_nan:.1%} NaN")
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

        # [v13.2] 每個演算法獨立 try/except，任一失敗不終止管線
        ny, nx = sst.shape

        # SST 鋒面
        try:
            sst_front_result = detect_sst_fronts(sst, lats, lons)
        except Exception as e:
            log.warning(f"  ⚠️ SST front detection degraded: {e}")
            _degraded_fields.append("SST_FRONTS")
            sst_front_result = {
                "front_strength": np.zeros((ny, nx), dtype=np.float32),
                "gradient_magnitude": np.zeros((ny, nx), dtype=np.float32),
                "gradient_raw": np.zeros((ny, nx), dtype=np.float32),
                "front_mask": np.zeros((ny, nx), dtype=bool),
                "front_direction": np.zeros((ny, nx), dtype=np.float32),
            }
        front_strength = sst_front_result["front_strength"]

        # Chl-a 鋒面
        try:
            chl_front_result = compute_chl_gradient(chl, lats, lons)
        except Exception as e:
            log.warning(f"  ⚠️ CHL front detection degraded: {e}")
            _degraded_fields.append("CHL_FRONTS")
            chl_front_result = {
                "chl_gradient": np.zeros((ny, nx), dtype=np.float32),
                "chl_front_mask": np.zeros((ny, nx), dtype=bool),
            }

        # ─── [v16] ② CHL 空間梯度 → 融合鋒面強度 ───
        # Direct gradient computation from CHL (Sobel-like)
        chl_safe = np.nan_to_num(chl, nan=0.0)
        chl_gy, chl_gx = np.gradient(chl_safe)
        chl_grad_mag = np.sqrt(chl_gy**2 + chl_gx**2)
        chl_p95 = max(float(np.nanpercentile(chl_grad_mag[chl_grad_mag > 0], 95)) if np.any(chl_grad_mag > 0) else 0.001, 0.001)
        chl_front_norm = np.clip(chl_grad_mag / chl_p95, 0, 1).astype(np.float32)
        # Fuse SST + CHL fronts (equally weighted)
        sst_front_norm = np.clip(front_strength / max(float(np.nanmax(front_strength)), 0.001), 0, 1)
        front_strength = np.clip(0.5 * sst_front_norm + 0.5 * chl_front_norm, 0, 1).astype(np.float32)
        log.info(f"  🌊 Combined front: SST+CHL fused, mean={np.nanmean(front_strength):.4f}, "
                 f"CHL grad p95={chl_p95:.6f}")

        # FTLE
        try:
            ftle_result = compute_ftle(u_current, v_current, lats, lons)
        except Exception as e:
            log.error(f"  ❌ compute_ftle failed: {e}")
            ftle_result = {
                "ftle": np.zeros((ny, nx), dtype=np.float32),
                "ridges": np.zeros((ny, nx), dtype=bool),
            }
        ftle_field = ftle_result["ftle"]

        # 溫躍層
        thermocline_result = None
        if temp_3d is not None:
            try:
                depths_arr = np.array([0, 50, 100, 200, 300, 500])[:temp_3d.shape[0]]
                thermocline_result = compute_thermocline(temp_3d, depths_arr, lats, lons)
            except Exception as e:
                log.error(f"  ❌ compute_thermocline failed: {e}")

        log.info(f"  SST fronts: {np.nansum(front_strength > 0.3):.0f} strong pts")
        log.info(f"  FTLE mean: {np.nanmean(ftle_field):.4f}")

        # ─── [v16] ① SSH 異常值 → 渦旋偵測 ───
        # NRT SSH minus climatology = anomaly (positive=warm eddy, negative=upwelling)
        try:
            from engine.data_fetcher_v2 import WOAClimatology
            ssh_clim = WOAClimatology.ssh_climatology(lats, lons, month) if hasattr(WOAClimatology, 'ssh_climatology') else np.zeros_like(ssh)
            ssh_anomaly = np.nan_to_num(ssh, nan=0.0) - ssh_clim
            ssh_anom_abs = np.abs(ssh_anomaly)
            ssh_anom_p95 = max(float(np.nanpercentile(ssh_anom_abs[ssh_anom_abs > 0], 95)) if np.any(ssh_anom_abs > 0) else 0.01, 0.01)
            ssh_anom_si = np.clip(ssh_anom_abs / ssh_anom_p95, 0, 1).astype(np.float32)
            log.info(f"  🌀 SSH anomaly: mean={np.nanmean(ssh_anomaly):.4f}m, |anom| p95={ssh_anom_p95:.4f}m")
        except Exception as e:
            log.warning(f"  SSH anomaly failed ({e}), using zeros")
            ssh_anomaly = np.zeros((ny, nx), dtype=np.float32)
            ssh_anom_si = np.zeros((ny, nx), dtype=np.float32)


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
        # Step 2.3: 黑潮特徵 [v13]
        # ═══════════════════════════════════════════
        kuroshio_result = None
        if self.kuroshio is not None:
            log.info("\n  Step 2.3: Kuroshio Current features")
            try:
                kuroshio_result = self.kuroshio.analyze_full(
                    sst, lats, lons, u_current, v_current,
                    adt=data.get("ssh"), month=month
                )
            except Exception as e:
                log.warning(f"  Kuroshio analysis failed: {e}")
        else:
            log.info("  Step 2.3: Kuroshio skipped (module not loaded)")

        # ═══════════════════════════════════════════
        # Step 2.5: GreenFish Lite — 物理層升級
        # ═══════════════════════════════════════════
        gf_z20 = gf_mld = gf_eddy_result = gf_forage = gf_feeding = None
        if GREENFISH_OK:
            log.info("\n  Step 2.5: GreenFish Lite (Z20/MLD/Eddy/Forage/DVM)")

        # ── Z20 + MLD + T100: GLORYS12 → HYCOM → 氣候態 ──
            t100_field = None
            gradient_strength_field = None
            gradient_depth_field = None
            delta_t_field = None

            glorys_result = self.thermocline.fetch_glorys12(lats, lons, month)
            if glorys_result is not None:
                gf_z20 = glorys_result["z20"]
                gf_mld = glorys_result["mld"]
                t100_field = glorys_result.get("t100")
                gradient_strength_field = glorys_result.get("gradient_strength")
                gradient_depth_field = glorys_result.get("gradient_depth")
                # [海鷹] 計算 ΔT = SST - T100
                if t100_field is not None and sst is not None:
                    try:
                        if sst.shape == t100_field.shape:
                            delta_t_field = (sst - t100_field).astype(np.float32)
                        else:
                            _ny = min(sst.shape[0], t100_field.shape[0])
                            _nx = min(sst.shape[1], t100_field.shape[1])
                            delta_t_field = np.full_like(t100_field, np.nan)
                            delta_t_field[:_ny, :_nx] = (sst[:_ny, :_nx] - t100_field[:_ny, :_nx])
                    except Exception as e:
                        log.debug(f"[降級] main_v10_3.py: {e}")
                log.info(f"  ✅ Z20/MLD/T100 from {glorys_result.get('source', 'GLORYS12')}")
                if t100_field is not None:
                    log.info(f"  🌡️ T100: avg={np.nanmean(t100_field):.1f}°C")
                if delta_t_field is not None:
                    log.info(f"  ΔT(SST-T100): avg={np.nanmean(delta_t_field):.1f}°C")
            elif temp_3d is not None and temp_3d.ndim == 3 and temp_3d.shape[0] >= 3:
                # 若 HYCOM 解析度太低 (< 目標格點的 25%)，用氣候態更可靠
                hycom_coverage = (temp_3d.shape[1] * temp_3d.shape[2]) / max(ny * nx, 1)
                if hycom_coverage < 0.01:
                    log.warning(f"  HYCOM 3D coverage too low ({hycom_coverage:.1%}) → climatology")
                    gf_z20 = ThermoclineFetcher.climatology_z20(lats, lons, month)
                    gf_mld = ThermoclineFetcher.climatology_mld(lats, lons, month)
                else:
                    tc_result = self.thermocline.compute_from_temp3d(
                        temp_3d, lats, lons, sst_grid=sst
                    )
                    gf_z20 = tc_result["z20"]
                    gf_mld = tc_result["mld"]
                    t100_field = tc_result.get("t100")
                    gradient_strength_field = tc_result.get("gradient_strength")
                    gradient_depth_field = tc_result.get("gradient_depth")
                    delta_t_field = tc_result.get("delta_t_surface_100")
                    log.info(f"  Z20: {np.nanmean(gf_z20):.0f}m | MLD: {np.nanmean(gf_mld):.0f}m")
                    if t100_field is not None:
                        log.info(f"  🌡️ T100: avg={np.nanmean(t100_field):.1f}°C")
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

        # [v13.2] DO: CMEMS BGC 優先 → WOA analyzer fallback
        if do_surface is not None and np.any(np.isfinite(do_surface)) and np.nanmean(do_surface) > 0.1:
            log.info(f"  ✅ DO: using CMEMS BGC (mean={np.nanmean(do_surface):.2f} ml/L)")
            do_result = {
                "do_surface": do_surface,
                "do_si": {},  # will be filled per-species below
            }
            # 為每個魚種計算 DO 適宜度
            for sp in self.species:
                sp_params = SPECIES_PARAMS.get(sp, {})
                do_min = sp_params.get("do_min", 2.0)
                do_pref = sp_params.get("do_preferred", 4.0)
                k_do = 2.0 / max(do_pref - do_min, 0.5)
                do_si_sp = 1.0 / (1.0 + np.exp(-k_do * (do_surface - (do_min + do_pref) / 2)))
                do_result["do_si"][sp] = do_si_sp.astype(np.float32)
        else:
            log.info("  DO: CMEMS unavailable, using WOA analyzer")
            do_result = self.do_analyzer.analyze(lats, lons, month=month, sst=sst)
            # 如果 WOA 也失敗，使用 CMEMS BGC 或氣候態 fallback
            if do_result.get("do_surface") is None or np.nanmean(do_result.get("do_surface", 0)) < 0.01:
                if do_surface is not None:
                    do_result["do_surface"] = do_surface
                    log.warning("  WOA failed, using CMEMS BGC do_surface as fallback")
                else:
                    do_result["do_surface"] = np.ones((ny, nx), dtype=np.float32) * 5.0
                    log.warning("  WOA+CMEMS both failed, using default DO=5.0 ml/L")

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

        # [v16.0] CMEMS nppv 衛星數據優先 — 真實觀測取代 VGPM 公式估算
        # nppv 單位: mgC/m³/day (體積), 需 ×Zeu 轉為 mgC/m²/day (面積)
        _npp_source = "VGPM-formula"
        if cmems_nppv is not None and np.any(np.isfinite(cmems_nppv)):
            _nppv_valid = np.sum(np.isfinite(cmems_nppv) & (cmems_nppv > 0))
            if _nppv_valid > 0.1 * cmems_nppv.size:
                # 轉換: nppv(mgC/m³/day) × Zeu(m) ≈ NPP(mgC/m²/day)
                # 使用已計算的 euphotic depth (或簡單估算)
                if gf_forage is not None and "z_euphotic" in gf_forage:
                    _zeu = gf_forage["z_euphotic"]
                elif chl is not None:
                    _zeu = np.clip(568.2 * np.power(np.clip(chl, 0.01, 30), -0.746), 5, 250)
                else:
                    _zeu = np.full_like(cmems_nppv, 100.0)
                # 確保尺寸匹配
                if cmems_nppv.shape == _zeu.shape:
                    _sat_npp = np.nan_to_num(cmems_nppv, nan=0) * _zeu
                else:
                    _min_y = min(cmems_nppv.shape[0], _zeu.shape[0])
                    _min_x = min(cmems_nppv.shape[1], _zeu.shape[1])
                    _sat_npp = np.zeros_like(pp_result["npp"])
                    _sat_npp[:_min_y, :_min_x] = (
                        np.nan_to_num(cmems_nppv[:_min_y, :_min_x], nan=0) *
                        _zeu[:_min_y, :_min_x]
                    )
                _sat_npp = np.clip(_sat_npp, 0, 2500).astype(np.float32)
                # 用衛星數據覆蓋 VGPM 公式值
                pp_result["npp"] = _sat_npp
                _npp_source = "CMEMS-BGC-satellite"
                log.info(f"  ✅ NPP: CMEMS 衛星數據 (mean={np.nanmean(_sat_npp):.0f} mgC/m²/d, "
                         f"valid={_nppv_valid}/{cmems_nppv.size})")
            else:
                log.info(f"  NPP: CMEMS nppv coverage too low ({_nppv_valid}/{cmems_nppv.size}), using VGPM")
        else:
            log.info(f"  NPP: CMEMS nppv unavailable, using VGPM formula")
        pp_result["npp_source"] = _npp_source

        # [v13.2-P1] NPZ 生態動力模型
        npz_result = None
        try:
            from engine.npz_model import NPZModel
            npz = NPZModel()
            npz_result = npz.run(sst, chl, n_days=15)
            results["npz_phytoplankton_mean"] = float(np.nanmean(npz_result["phytoplankton"]))
            results["npz_zooplankton_mean"] = float(np.nanmean(npz_result["zooplankton"]))
            log.info(f"  🧬 NPZ ecology: P={results['npz_phytoplankton_mean']:.3f}, "
                     f"Z={results['npz_zooplankton_mean']:.3f}")
        except Exception as e:
            log.debug(f"  NPZ model skipped (OptionalFeatureError): {e}")

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
            "ssh_anomaly": ssh_anomaly,  # [v16] NRT SSH anomaly
            "chl_front_norm": chl_front_norm,  # [v16] CHL gradient front
            "front_strength": front_strength,  # [v16] now fused SST+CHL
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
            "bathy_roughness": gebco_feats.get("roughness"),  # [v15.0] V3.0 #1: seabed ruggedness
            "dist_to_seamount": gebco_feats.get("dist_to_seamount"),
            "dist_to_shelf_break": gebco_feats.get("dist_to_shelf_break"),
            # [海鷹/蒼鷺] 核心新特徵
            "t100": t100_field if GREENFISH_OK else None,
            "gradient_strength": gradient_strength_field if GREENFISH_OK else None,
            "gradient_depth": gradient_depth_field if GREENFISH_OK else None,
            "delta_t_surface_100": delta_t_field if GREENFISH_OK else None,
            "chl_lag15d": data.get("chl_lag15d"),  # [蒼鷺] 15天CHL時滯
        }
        hsi_results = compute_all_hsi(ocean_features, self.species, moon)

        # ═══════════════════════════════════════════
        # Step 5: 商業核心 HSI  ← [P0-1] 死代碼復活
        # ═══════════════════════════════════════════
        log.info("\n  Step 5/9: Commercial core HSI (13 techs)")

        ow_param = self._compute_okubo_weiss(u_current, v_current, lats, lons)

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
            do_field = do_result.get("do_surface", np.ones_like(sst) * 4.0)
            if do_field.shape != sst.shape:
                do_field = np.ones_like(sst) * 4.0
                log.warning(f"  {sp}: DO field shape mismatch — fallback to 4.0 μmol/kg, OMZ may be underestimated")

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
                eke_si=ssh_anom_si if ssh_anom_si is not None and ssh_anom_si.shape == sst.shape else (feat["eke_si"] if feat["eke_si"].shape == sst.shape else None),
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
                    # [v17] 競品整合: 用 WCPFC 數據學習的最優權重 (GreenFish 核心技術)
                    try:
                        from engine.blend_optimizer import load_optimal_weights
                        _bw = load_optimal_weights(sp, weight_type="hsi")
                        _w_v8 = _bw.get("w_v8", 0.15)
                        _w_comm = _bw.get("w_comm", 0.35)
                        _w_gf = _bw.get("w_gf", 0.50)
                    except Exception:
                        _w_v8, _w_comm, _w_gf = 0.15, 0.35, 0.50
                        log.warning(f"  {sp}: blend_optimizer load failed — using unvalidated fallback weights: v8=0.15, comm=0.35, gf=0.50")
                    blended = np.clip(
                        _w_v8 * v8_safe + _w_comm * comm_hsi + _w_gf * gf_hsi,
                        0.0, 1.0
                    ).astype(np.float32)
                    # 保存 GreenFish 各子指數
                    hsi_results[sp]["gf_components"] = gf_result["components"]
                    hsi_results[sp]["gf_hsi"] = gf_hsi
                    log.info(f"  {sp}: GreenFish HSI={np.nanmean(gf_hsi):.3f} (integrated)")
                except Exception as e:
                    log.warning(f"  {sp}: GreenFish HSI failed ({e}), using comm blend")

            hsi_results[sp]["hsi"] = blended

            # [v17-P5-fix] 風場懲罰已移至 ai_fusion.py (含漸進衰減)，此處不再重複

            # ─── [v17] ① 底溫加成: 擴展到所有深層物種 (海鷹AI 20+變量啟發) ───
            if bottom_temp is not None and bottom_temp.shape == blended.shape:
                # [v17] 物種專屬底溫偏好 (擴展自 INCOIS/海鷹 研究)
                _bt_peaks = {
                    "bigeye": (10.0, 3.0),     # 偏好 8-12°C
                    "albacore": (10.0, 3.0),   # 偏好 8-12°C
                    "yellowfin": (15.0, 4.0),  # 偏好 12-18°C (溫躍層上方)
                    "skipjack": (18.0, 5.0),   # 偏好 14-22°C (表層)
                }
                if sp in _bt_peaks:
                    bt_center, bt_sigma = _bt_peaks[sp]
                    bt_pref = np.exp(-((bottom_temp - bt_center)**2) / (2 * bt_sigma**2))
                    bt_bonus = 0.05 * bt_pref  # up to +5% HSI
                    blended = np.clip(blended + bt_bonus, 0.0, 1.0).astype(np.float32)
                    hsi_results[sp]["hsi"] = blended
                    log.info(f"  {sp}: 🌊 Bottom temp bonus: peak={bt_center}°C, "
                             f"mean bt={np.nanmean(bottom_temp):.1f}°C, bonus={np.nanmean(bt_bonus):.4f}")
                hsi_results[sp]["bottom_temp"] = bottom_temp

            # ─── [v17] 風場漁場位移預測 (INCOIS PFZ 核心技術) ───
            if wind_speed is not None and blended is not None:
                try:
                    wind_u = data.get("wind_u")
                    wind_v = data.get("wind_v")
                    if wind_u is not None and wind_v is not None:
                        # 預測 24h 後表層物質位移方向 (km)
                        # 風場驅動漂移 ≈ 3% 風速 × 86400s / 1000 (Ekman transport 近似)
                        drift_u = wind_u * 0.03 * 86400 / 1000  # km/day
                        drift_v = wind_v * 0.03 * 86400 / 1000
                        drift_mag = np.sqrt(drift_u**2 + drift_v**2)
                        # 匯聚區: 漂移方向一致 → 物質堆積 → 適合捕魚
                        # 用梯度散度的負值作為匯聚指標
                        if drift_mag.shape == blended.shape:
                            wind_convergence = -(
                                np.gradient(drift_u, axis=1) + np.gradient(drift_v, axis=0)
                            )
                            wc_norm = np.clip(wind_convergence / max(np.nanmax(np.abs(wind_convergence)), 1e-6), -1, 1)
                            wc_bonus = 0.03 * np.clip(wc_norm, 0, 1)  # 只獎勵匯聚 (負散度)
                            blended = np.clip(blended + wc_bonus, 0.0, 1.0).astype(np.float32)
                            hsi_results[sp]["hsi"] = blended
                            n_conv = int(np.sum(wc_bonus > 0.005))
                            if n_conv > 0:
                                log.info(f"  {sp}: 💨 Wind convergence bonus: {n_conv} cells, "
                                         f"mean bonus={np.nanmean(wc_bonus):.4f}")
                except Exception as e:
                    log.debug(f"  {sp}: wind displacement prediction skipped: {e}")

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
                        # [v13.2-P1] run_in_executor to avoid blocking event loop
                        loop = asyncio.get_running_loop()
                        cpue_grid = await loop.run_in_executor(None, fsm.predict, ocean_features)
                        cpue_grid = np.clip(cpue_grid, 0, 1)  # already [0, 1]

                        # [v15.3-audit] Hybrid fusion: 80% science + 20% ML
                        # Spatial CV R² 為負 → ML 泛化不足，物理 HSI 主導
                        fused_hsi = np.clip(
                            0.8 * np.nan_to_num(science_hsi, nan=0.0) + 0.2 * cpue_grid,
                            0.0, 1.0
                        ).astype(np.float32)

                        phi_mask = phi_grid < 2.0
                        fused_hsi[phi_mask] *= 0.5

                        hsi_results[sp]["hsi"] = fused_hsi
                        hsi_results[sp]["ml_cpue"] = cpue_grid
                        # [v17-P4] ml_confidence removed — use ml_cpue_norm instead
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

                    cpue_pred = await loop.run_in_executor(
                        None, self.ml_models[sp].predict, X_grid
                    )
                    cpue_grid = np.maximum(cpue_pred.reshape(ny, nx), 0.0)

                    # [v15.3-fix] 使用 config.CPUE_NORMALIZATION 統一來源
                    from config import CPUE_NORMALIZATION
                    _cpue_cfg = CPUE_NORMALIZATION.get(sp, {"median": 45, "scale": 20})
                    med = _cpue_cfg["median"]
                    scl = _cpue_cfg["scale"]
                    cpue_norm = 1.0 / (1.0 + np.exp(-(cpue_grid - med) / scl))
                    cpue_norm = np.nan_to_num(cpue_norm, nan=0.5, posinf=1.0, neginf=0.0)

                    # [v17] 競品整合: 用數據學習的ML融合權重
                    try:
                        from engine.blend_optimizer import load_optimal_weights
                        _mw = load_optimal_weights(sp, weight_type="ml")
                        _w_sci = _mw.get("w_science", 0.80)
                        _w_ml = _mw.get("w_ml", 0.20)
                    except Exception:
                        _w_sci, _w_ml = 0.80, 0.20
                    fused_hsi = np.clip(
                        _w_sci * np.nan_to_num(science_hsi, nan=0.0) + _w_ml * cpue_norm,
                        0.0, 1.0
                    ).astype(np.float32)

                    phi_mask = phi_grid < 2.0
                    fused_hsi[phi_mask] *= 0.5

                    hsi_results[sp]["hsi"] = fused_hsi
                    hsi_results[sp]["ml_cpue"] = cpue_grid
                    hsi_results[sp]["ml_cpue_norm"] = cpue_norm   # [v17] 正名: sigmoid(CPUE) 非信心值
                    # [v17-P4] ml_confidence removed — use ml_cpue_norm instead
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
                        temp_3d=temp_3d,
                        temp_3d_depths=[0, 50, 100, 200, 300, 500] if temp_3d is not None else None,
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
            "wind_speed": wind_speed,  # [v17-P5] 風速 → 鋒面位移加分
        }

        # [v10.3] 將低壓漁場加分加入 fusion
        if pressure_data and "low_pressure_bonus" in pressure_data:
            fusion_features["low_pressure_bonus"] = pressure_data["low_pressure_bonus"]

        # [v17] 競品整合: 將鋒面持久性 + 匯聚帶加入 fusion (INCOIS PFZ + Aker)
        # 使用第一個可用物種的數據 (所有物種共用同一海洋場)
        for _sp_data in hsi_results.values():
            if "front_persistence" in _sp_data and _sp_data["front_persistence"] is not None:
                fusion_features["front_persistence"] = _sp_data["front_persistence"]
                break
        for _sp_data in hsi_results.values():
            if "convergence" in _sp_data and _sp_data["convergence"] is not None:
                fusion_features["convergence_index"] = _sp_data["convergence"]
                break

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

        # [v13.2] 商業級陸地過濾 — 三層防護 (bathy -200m + 幾何 bbox + buffer)
        n_before_land = len(ranked)
        ranked = [h for h in ranked
                  if _is_ocean(h["lat"], h["lon"], bathy, lats, lons)]
        n_land_removed = n_before_land - len(ranked)
        if n_land_removed > 0:
            log.warning(f"  🏔️ Land filter: removed {n_land_removed}/{n_before_land} "
                        f"hotspots (bathy > -200m or near land bbox)")

        # ─── [v15-fix] R2: 季節性過濾 ───
        # Species active months (outside = 90% score penalty)
        _ACTIVE_MONTHS = {
            "pacific_saury": [8, 9, 10, 11, 12],
            "japanese_flying_squid": [6, 7, 8, 9, 10, 11],
        }
        for h in ranked:
            sp = h.get("species", "")
            active = _ACTIVE_MONTHS.get(sp)
            if active and now.month not in active:
                h["score"] = h["score"] * 0.1
                h["season_warning"] = True
            else:
                h["season_warning"] = False

        # Re-sort after season penalty
        ranked.sort(key=lambda h: h.get("score", 0), reverse=True)
        ranked = ranked[:OUTPUT.get("top_n_hotspots", 20)]

        # ─── [v13.5] 降尺度: 利用高解析度衛星數據精修熱點座標 ───
        try:
            from engine.hotspot_downscaler import downscale_hotspots
            ranked = downscale_hotspots(
                ranked, sst=sst, chl=chl,
                ssh=data.get("ssh"),
                lats=lats, lons=lons,
                search_radius_deg=0.5,
            )
        except Exception as e:
            log.warning(f"  降尺度失敗 (不影響結果): {e}")
            for h in ranked:
                h["precision_grade"] = "C"
                h["precision_label"] = "原始網格 (~0.5°)"

        # ─── [v13.5] 戰術特徵: 月相 + 渦旋強度 ───
        _tactical_moon = None
        _tactical_eddy = None
        try:
            from engine.tactical_features import lunar_phase_index, compute_eddy_index
            _tactical_moon = lunar_phase_index()
            log.info(f"  🌙 月相: {_tactical_moon['name']} (亮度 {_tactical_moon['fullness']:.2f})")
            log.info(f"     {_tactical_moon['hook_depth_advice']}")
            if data.get("ssh") is not None:
                _tactical_eddy = compute_eddy_index(data["ssh"], lats, lons)
        except Exception as e:
            log.warning(f"  戰術特徵計算失敗: {e}")

        # ─── [v16] 誠實顯示: score_display = 真實 HSI score ───
        if ranked:
            # [v13.5] 計算數據時效性 (含實際抓取時間戳)
            _NRT_SOURCES = {"MUR", "OSTIA", "CMEMS-NRT", "CMEMS"}
            _REANALYSIS = {"HYCOM", "WOA"}
            _sst_src = data.get("sst_source", "")
            _chl_src = data.get("chl_source", "")
            _sst_time = data.get("sst_fetch_time") or data.get("sst_time")
            _chl_time = data.get("chl_fetch_time") or data.get("chl_time")
            _ssh_time = data.get("ssh_fetch_time") or data.get("ssh_time")

            _sst_age_h = self._hours_ago(_sst_time, now)
            _chl_age_h = self._hours_ago(_chl_time, now)
            _ssh_age_h = self._hours_ago(_ssh_time, now)

            for i, h in enumerate(ranked):
                raw = float(h.get("score", 0))
                if "raw_hsi" not in h:
                    h["raw_hsi"] = round(np.clip(raw, 0.01, 1.0), 2)

                # [v13.5] 增強版數據新鮮度 — 含時間戳和年齡
                _sst_label = ("NRT" if any(k in _sst_src for k in _NRT_SOURCES)
                              else ("再分析" if any(k in _sst_src for k in _REANALYSIS)
                                    else "歷史均值"))
                _chl_label = "NRT" if _chl_src and _chl_src != "climatology" else "歷史均值"

                h["data_freshness"] = {
                    "sst": _sst_label,
                    "sst_age_hours": _sst_age_h,
                    "sst_source": _sst_src or "unknown",
                    "chl": _chl_label,
                    "chl_age_hours": _chl_age_h,
                    "ssh": "NRT" if data.get("ssh") is not None else "無",
                    "ssh_age_hours": _ssh_age_h,
                    "currents": "NRT" if data.get("u_current") is not None else "公式估算",
                    "bottom_temp": "NRT" if data.get("bottom_temp") is not None else "公式估算",
                    "wind": "NRT" if wind_speed is not None else "無",
                    "mld": "NRT" if gf_mld is not None else "歷史均值",
                }

                # [v13.5] 時效性警語 — 任一關鍵數據超過 48 小時
                stale_features = []
                if _sst_age_h is not None and _sst_age_h > 48:
                    stale_features.append(f"SST ({_sst_age_h:.0f}h前)")
                if _chl_age_h is not None and _chl_age_h > 48:
                    stale_features.append(f"葉綠素 ({_chl_age_h:.0f}h前)")
                if _ssh_age_h is not None and _ssh_age_h > 48:
                    stale_features.append(f"SSH ({_ssh_age_h:.0f}h前)")
                if stale_features:
                    h["staleness_warning"] = (
                        f"⚠️ 此預測基於 {', '.join(stale_features)} 的數據，"
                        f"建議現場確認鋒面位置"
                    )

                # [v13.5] 戰術: 月相 + 渦旋
                if _tactical_moon:
                    h["lunar_phase"] = _tactical_moon["name"]
                    h["lunar_fullness"] = _tactical_moon["fullness"]
                    h["lunar_day"] = _tactical_moon["lunar_day"]
                    h["hook_depth_advice"] = _tactical_moon["hook_depth_advice"]
                    h["lunar_fishing_impact"] = _tactical_moon["fishing_impact"]

                if _tactical_eddy is not None:
                    li = np.argmin(np.abs(lats - h["lat"]))
                    lj = np.argmin(np.abs(lons - h["lon"]))
                    ei = _tactical_eddy["eddy_intensity"]
                    et = _tactical_eddy["eddy_type"]
                    ci = _tactical_eddy["cyclonic_index"]
                    if li < ei.shape[0] and lj < ei.shape[1]:
                        h["eddy_intensity"] = round(float(ei[li, lj]), 3)
                        _et = int(et[li, lj])
                        h["eddy_type_code"] = _et
                        h["eddy_type_label"] = (
                            "暖渦 (下沉流)" if _et == 1
                            else "冷渦 (上升流)" if _et == -1
                            else "中性"
                        )
                        h["cyclonic_index"] = round(float(ci[li, lj]), 3)

            scores_raw = [h["score"] for h in ranked]
            scores_disp = [round(np.clip(h["score"], 0.01, 0.95), 2) for h in ranked]
            raw_hsis = [h.get("raw_hsi", 0) for h in ranked]
            log.info(f"  📊 HSI: [{min(scores_raw):.4f}–{max(scores_raw):.4f}] "
                     f"display [{min(scores_disp):.2f}–{max(scores_disp):.2f}] "
                     f"raw [{min(raw_hsis):.3f}–{max(raw_hsis):.3f}]")

        # [v15.3-fix] 預建構 loop 內用到的物件 — 避免每個 hotspot 重複初始化
        try:
            from engine.ocean_diagnostics import FishingTimeOptimizer
            _fto = FishingTimeOptimizer()
        except Exception:
            _fto = None
        try:
            from engine.env_cpue_estimator import EnvironmentalCPUEEstimator
            _cpue_est = EnvironmentalCPUEEstimator()
        except Exception:
            _cpue_est = None
        try:
            from engine.catch_composition import CatchCompositionForecaster
            _ccf = CatchCompositionForecaster()
        except Exception:
            _ccf = None

        # [v15.3-perf] MicronektonModel — loop 外預計算完整 2D grid，避免每個 hotspot 重算
        _mn_result_cache = None
        _mn_cache_model = None
        try:
            _npp_grid = pp_result.get("npp")
            if GREENFISH_OK and sst is not None and _npp_grid is not None:
                _mn_cache_model = MicronektonModel()
                _depth_grid = bathy if bathy is not None else np.full_like(sst, 3000)
                _mld_grid = gf_mld if gf_mld is not None else np.full_like(sst, 50)
                _z20_grid = gf_z20 if gf_z20 is not None else np.full_like(sst, 150)
                _chl_grid = chl if chl is not None else np.full_like(sst, 0.3)
                _moon_ill = self._lunar.get("illumination", 0.5)
                _hour_utc = now.hour
                _mn_result_cache = _mn_cache_model.compute(
                    sst, _npp_grid, _mld_grid, _z20_grid, _chl_grid,
                    _depth_grid, moon_illum=_moon_ill, is_night=(_hour_utc < 6 or _hour_utc > 18),
                )
        except Exception as _mn_pre_err:
            log.debug(f"  Micronekton precompute skip: {_mn_pre_err}")

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
            # [v13.2] DO: 從 do_result 讀取（已合併 CMEMS + WOA）
            _do_field = do_result.get("do_surface")
            if _do_field is not None and li < _do_field.shape[0] and lj < _do_field.shape[1]:
                do_val = float(_do_field[li, lj])
                h["do_surface"] = do_val
                h["do_ml"] = round(do_val, 2)  # UI 用 do_ml 欄位
            elif do_surface is not None and li < do_surface.shape[0] and lj < do_surface.shape[1]:
                do_val = float(do_surface[li, lj])
                h["do_surface"] = do_val
                h["do_ml"] = round(do_val, 2)
            if li < bathy.shape[0] and lj < bathy.shape[1]:
                h["depth_m"] = float(-bathy[li, lj])
            # [v13.2] Z20/MLD 直接從 gf_z20/gf_mld 寫入（不依賴 GreenFish HSI 成功）
            if gf_z20 is not None and li < gf_z20.shape[0] and lj < gf_z20.shape[1]:
                h.setdefault("z20_m", float(gf_z20[li, lj]))
            if gf_mld is not None and li < gf_mld.shape[0] and lj < gf_mld.shape[1]:
                h.setdefault("mld_m", float(gf_mld[li, lj]))
            # [商用] SST — sea_conditions API 需要此欄位
            if sst is not None and li < sst.shape[0] and lj < sst.shape[1]:
                v = float(sst[li, lj])
                if np.isfinite(v):
                    h["sst"] = round(v, 2)

            # CHL
            if chl is not None and li < chl.shape[0] and lj < chl.shape[1]:
                cv = float(chl[li, lj])
                if np.isfinite(cv):
                    h["chl"] = round(cv, 4)

            # [海鷹/蒼鷺] T100, gradient, delta_t, chl_lag15d 注入 hotspot
            if GREENFISH_OK:
                if t100_field is not None and li < t100_field.shape[0] and lj < t100_field.shape[1]:
                    tv = float(t100_field[li, lj])
                    if np.isfinite(tv):
                        h["t100"] = round(tv, 1)
                if gradient_strength_field is not None and li < gradient_strength_field.shape[0] and lj < gradient_strength_field.shape[1]:
                    gv = float(gradient_strength_field[li, lj])
                    if np.isfinite(gv):
                        h["gradient_strength"] = round(gv, 3)
                if delta_t_field is not None and li < delta_t_field.shape[0] and lj < delta_t_field.shape[1]:
                    dv = float(delta_t_field[li, lj])
                    if np.isfinite(dv):
                        h["delta_t_surface_100"] = round(dv, 1)
            _chl_lag = data.get("chl_lag15d")
            if _chl_lag is not None and li < _chl_lag.shape[0] and lj < _chl_lag.shape[1]:
                clv = float(_chl_lag[li, lj])
                if np.isfinite(clv):
                    h["chl_lag15d"] = round(clv, 4)

            # Front persistence / strength
            if front_strength is not None and li < front_strength.shape[0] and lj < front_strength.shape[1]:
                fv = float(front_strength[li, lj])
                if np.isfinite(fv):
                    h["front_persistence"] = round(fv, 3)
            if sp in hsi_results and "front_persistence" in hsi_results[sp]:
                _fp = hsi_results[sp]["front_persistence"]
                if _fp is not None and li < _fp.shape[0] and lj < _fp.shape[1]:
                    fpv = float(_fp[li, lj])
                    if np.isfinite(fpv) and fpv > h.get("front_persistence", 0):
                        h["front_persistence"] = round(fpv, 3)

            # Eddy edge
            if sp in hsi_results and "eddy_edge" in hsi_results[sp]:
                _ee = hsi_results[sp]["eddy_edge"]
                if _ee is not None and li < _ee.shape[0] and lj < _ee.shape[1]:
                    eev = float(_ee[li, lj])
                    if np.isfinite(eev):
                        h["eddy_edge"] = round(eev, 3)

            # [v16] SSH anomaly + CHL front enrichment
            if ssh_anomaly is not None and li < ssh_anomaly.shape[0] and lj < ssh_anomaly.shape[1]:
                anom_val = float(ssh_anomaly[li, lj])
                h["ssh_anomaly"] = round(anom_val, 4)
                if anom_val > 0.05:
                    h["eddy_type"] = "暖渦"
                elif anom_val < -0.05:
                    h["eddy_type"] = "上升流"
                else:
                    h["eddy_type"] = "中性"
            if chl_front_norm is not None and li < chl_front_norm.shape[0] and lj < chl_front_norm.shape[1]:
                h["chl_front_strength"] = round(float(chl_front_norm[li, lj]), 3)

            # [v16] Wind speed + bottom temp enrichment
            if wind_speed is not None and li < wind_speed.shape[0] and lj < wind_speed.shape[1]:
                ws_val = float(wind_speed[li, lj])
                h["wind_speed_ms"] = round(ws_val, 1)
                h["wind_warning"] = ws_val > 15.0
            if bottom_temp is not None and li < bottom_temp.shape[0] and lj < bottom_temp.shape[1]:
                h["bottom_temp_c"] = round(float(bottom_temp[li, lj]), 1)

            # Convergence
            if sp in hsi_results and "convergence" in hsi_results[sp]:
                _cv = hsi_results[sp]["convergence"]
                if _cv is not None and li < _cv.shape[0] and lj < _cv.shape[1]:
                    cvv = float(_cv[li, lj])
                    if np.isfinite(cvv):
                        h["convergence"] = round(cvv, 3)

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
                    h["feeding_index"] = min(0.95, float(sd["feeding_index"][li, lj]))
                if "gf_hsi" in sd and sd["gf_hsi"] is not None and li < sd["gf_hsi"].shape[0] and lj < sd["gf_hsi"].shape[1]:
                    h["greenfish_hsi"] = round(float(sd["gf_hsi"][li, lj]) * 100, 1)

            # ═══ [v13.2] 生態鏈指標 — 寫入 DVM/OMZ/浮游動物/月相 ═══
            # DVM 深度 (基於物種參數)
            try:
                from engine.species_params import DVM_PARAMS
                dvm_info = DVM_PARAMS.get(sp, {})
                day_r = dvm_info.get("day_depth_range", (50, 200))
                night_r = dvm_info.get("night_depth_range", (0, 50))
                import math
                hour_utc = datetime.now(timezone.utc).hour
                day_w = 0.5 * (1 + math.cos(2 * math.pi * (hour_utc - 12) / 24))
                h["dvm_depth_m"] = round(day_w * sum(day_r)/2 + (1-day_w) * sum(night_r)/2, 1)
            except Exception:
                h["dvm_depth_m"] = 100.0

            # 月相 + 光照 — [v13.9] 只在 tactical 模組未設定時才用 fallback
            if "lunar_phase" not in h:
                h["lunar_phase"] = self._lunar.get("phase_name", "未知")
            if "lunar_fullness" not in h:
                h["lunar_illumination"] = round(self._lunar.get("illumination", 0.0), 3)

            # [v16.0] 浮遊動物代理 — 優先用 CMEMS phyc 衛星數據
            _zoo_set = False
            if cmems_phyc is not None and li < cmems_phyc.shape[0] and lj < cmems_phyc.shape[1]:
                _phyc_val = float(cmems_phyc[li, lj])
                if np.isfinite(_phyc_val) and _phyc_val > 0:
                    # phyc (mmolC/m³) → zoo proxy: Lindeman 10-15% transfer efficiency
                    # Typical phyc: 0.1-50 mmolC/m³, saturate at ~20 mmolC/m³
                    h["zoo_proxy"] = round(min(0.95, _phyc_val / 20.0), 3)
                    _zoo_set = True
            if not _zoo_set:
                _npp_val = h.get("npp", 0)
                if _npp_val > 0:
                    # NPP → 浮游動物: Ikeda & Motoda (1978) 簡化
                    # max NPP ~ 2500, 合理轉換
                    h["zoo_proxy"] = round(min(0.95, _npp_val / 2500.0), 3)
                elif chl is not None and li < chl.shape[0] and lj < chl.shape[1]:
                    _chl_val = float(chl[li, lj])
                    h["zoo_proxy"] = round(min(0.95, max(0, _chl_val / 2.0)), 3)
                else:
                    h["zoo_proxy"] = 0.0

            # OMZ 效應 (基於 DO 溶氧)
            _do_val = h.get("do_surface", 0)
            if _do_val > 0:
                # DO < 2 ml/L = 嚴重 OMZ 壓縮，DO > 5 = 無 OMZ
                if _do_val < 2.0:
                    h["omz_net_effect"] = round(0.8 - 0.3 * _do_val, 3)  # 壓縮效應
                elif _do_val < 3.5:
                    h["omz_net_effect"] = round(0.3 * (3.5 - _do_val) / 1.5, 3)  # 邊緣增益
                else:
                    h["omz_net_effect"] = 0.0
            else:
                h["omz_net_effect"] = 0.0

            # 覓食百分比 (從 pp_result)
            if sp in pp_result.get("forage_index", {}):
                _fi = pp_result["forage_index"][sp]
                if li < _fi.shape[0] and lj < _fi.shape[1]:
                    h["forage_pct"] = round(float(_fi[li, lj]) * 100, 1)

            # [v15.3-perf] Micronekton — 使用 loop 外的預計算結果
            if _mn_result_cache is not None:
                try:
                    _mn_cache_model.enrich_hotspot(h, _mn_result_cache, lats, lons)
                except Exception as _mn_err:
                    log.debug(f"  Micronekton enrichment skip: {_mn_err}")

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

            # [v17-P4] ML CPUE (v10.3) — 永遠儲存確保 API schema 一致
            h["ml_cpue_kg_day"] = 0.0
            h["ml_confidence"] = 0.0  # [v17-P4] deprecated key, kept for API compat
            h["ml_cpue_norm"] = 0.0  # [v16-cal] 誠實名稱: sigmoid(CPUE)
            h["ml_confidence_note"] = "sigmoid 正規化 CPUE (0-1)，非預測信心值"
            if self.ml_available and sp in hsi_results and "ml_cpue" in hsi_results[sp]:
                cpue_g = hsi_results[sp]["ml_cpue"]
                norm_g = hsi_results[sp].get("ml_cpue_norm", cpue_g)  # [v17-P4] use ml_cpue_norm
                if li < cpue_g.shape[0] and lj < cpue_g.shape[1]:
                    h["ml_cpue_kg_day"] = round(float(cpue_g[li, lj]), 1)
                    norm_val = round(float(norm_g[li, lj]), 3)
                    h["ml_confidence"] = norm_val  # [v17-P4] deprecated, kept for API compat
                    h["ml_cpue_norm"] = norm_val

            # ═══ [v16.0] HTML 欄位補齊 — 確保地圖顯示完整 ═══
            # (1) confidence / do / hook_depth_label — 名稱映射
            h["confidence"] = h.get("ml_cpue_norm", h.get("ml_confidence", 0.0))
            h["do"] = h.get("do_surface", 0.0)
            h["hook_depth_label"] = h.get("hook_depth_advice", "依物種習性調整")

            # (2) fishing_method — 漁法建議 (依物種)
            _fishing_methods = {
                "skipjack": "圍網 / 竿釣",
                "yellowfin": "延繩釣 / 圍網",
                "bigeye": "深水延繩釣",
                "albacore": "拖釣 / 延繩釣",
                "swordfish": "深水延繩釣 / 劍旗魚專用",
                "squid_todarodes": "魷釣機",
                "squid_ommastrephes": "魷釣機",
            }
            h["fishing_method"] = _fishing_methods.get(sp, "延繩釣")

            # (3) best_fishing_time — 最佳捕撈時段
            try:
                _lunar_ill = self._lunar.get("illumination", 0.5)
                _ft_result = _fto.compute(sp, lunar_illumination=_lunar_ill, utc_offset=8)
                _windows = _ft_result.get("best_hours_local", [])
                if _windows:
                    h["best_fishing_time"] = ", ".join(
                        f"{s:02d}:00-{e:02d}:59" for s, e in _windows
                    )
                else:
                    h["best_fishing_time"] = "全日可作業"
            except Exception:
                h["best_fishing_time"] = "依傳統經驗"

            # (4) cpue_index / cpue_ci_low / cpue_ci_high / cpue_confidence
            try:
                _sst_val = h.get("sst", 25.0)
                _chl_val = float(chl[li, lj]) if (chl is not None and li < chl.shape[0] and lj < chl.shape[1]) else 0.3
                _cpue_r = _cpue_est.compute(
                    sst=_sst_val, chl=_chl_val, species=sp, month=month,
                    do_surface=h.get("do_surface"),
                    hsi_score=h.get("score", 0.5),
                    wind_speed=h.get("wind_speed_ms"),
                )
                h["cpue_index"] = _cpue_r["cpue_estimate"]
                h["cpue_ci_low"] = _cpue_r["cpue_range"][0]
                h["cpue_ci_high"] = _cpue_r["cpue_range"][1]
                h["cpue_confidence"] = _cpue_r["confidence"]
            except Exception:
                h["cpue_index"] = round(h.get("score", 0.5) * 50, 1)
                h["cpue_ci_low"] = round(h["cpue_index"] * 0.5, 1)
                h["cpue_ci_high"] = round(h["cpue_index"] * 1.5, 1)
                h["cpue_confidence"] = "low"

            # (5) gfw_validation — GFW 漁船活動驗證
            _gfw_grid = data.get("gfw_fishing_hours")
            if _gfw_grid is not None and li < _gfw_grid.shape[0] and lj < _gfw_grid.shape[1]:
                _gfw_h = float(_gfw_grid[li, lj])
                if _gfw_h > 100:
                    h["gfw_validation"] = f"GFW 高活動 ({_gfw_h:.0f}h/30d) ✅"
                elif _gfw_h > 10:
                    h["gfw_validation"] = f"GFW 中活動 ({_gfw_h:.0f}h/30d)"
                elif _gfw_h > 0:
                    h["gfw_validation"] = f"GFW 低活動 ({_gfw_h:.1f}h/30d)"
                else:
                    h["gfw_validation"] = "GFW 無近期漁船記錄"
            else:
                h["gfw_validation"] = "GFW 數據不可用"

            # (6) primary_species_prob — 物種概率
            try:
                _cc_result = _ccf.forecast(
                    sst=h.get("sst", 25), chl=_chl_val,
                    lat=h["lat"], month=month,
                )
                _sp_prob = _cc_result.get("probabilities", {}).get(sp, 0)
                h["primary_species_prob"] = f"{sp}: {_sp_prob:.0f}%"
            except Exception:
                h["primary_species_prob"] = ""

            # [B-1] 月相 — [v13.9] 只在 tactical 模組未設定時才用 fallback
            if ENHANCED_OK:
                if "lunar_phase" not in h:
                    h["lunar_phase"] = self._lunar.get("phase_name", "")
                h.setdefault("lunar_factor", self._lunar.get("species_factors", {}).get(sp, 1.0))

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

        # [v15.4] 颱風後黃金漁場加成
        if SAFETY_OK and typhoon_alerts:
            try:
                _tz = TyphoonGoldenZone()
                _golden = _tz.compute_golden_zones(
                    typhoon_alerts, sst, lats, lons,
                )
                if _golden["n_active_golden"] > 0:
                    ranked = _tz.enrich_hotspots(ranked, _golden, lats, lons)
                    results["typhoon_golden_zones"] = _golden["golden_zones"]
                    log.info(f"  🌀→🐟 Golden zones: {_golden['n_active_golden']} active")
            except Exception as _gz_err:
                log.debug(f"  Golden zone skip: {_gz_err}")

        results["hotspots"] = ranked
        log.info(f"  Ranked: {len(ranked)} hotspots")

        # [v13.9] Debug: 印出每個熱點的完整欄位字典
        for _di, _dh in enumerate(ranked[:3]):
            _critical_fields = {
                k: _dh.get(k) for k in [
                    "lat", "lon", "score", "species", "sst", "do_surface", "npp",
                    "depth_m", "phi", "z20_m", "mld_m", "eke", "feeding_index",
                    "greenfish_hsi", "lunar_phase", "hook_depth_advice",
                    "eddy_type_label", "precision_grade", "data_freshness",
                    "shap_values", "distance_nm", "dvm_depth_m",
                ]
            }
            log.info(f"  🔍 Hotspot #{_di+1} fields: {_critical_fields}")

        # ═══════════════════════════════════════════
        # [v15.3-fix] Step 7.5: ENSO 校準 — 移到 Fleet Ops 之前
        # 確保 CPUE estimator 和 route cost 使用 ENSO 校準後的分數
        # ═══════════════════════════════════════════
        if ENSO_OK and ranked:
            log.info("\n  Step 7.5: ENSO calibration")
            try:
                enso = ENSOCalibrator()
                results["enso"] = enso.summary()
                for h in ranked:
                    sp = h.get("species", "yellowfin")
                    mod = enso.get_species_modifier(sp)
                    h["enso_hsi_modifier"] = mod["hsi_modifier"]
                    h["enso_lat_shift"] = mod["lat_shift"]
                    h["score"] = max(0.0, min(0.95, h["score"] + mod["hsi_modifier"]))
                log.info(f"  ENSO: {enso.phase}, ONI={enso.oni:.2f}")
            except Exception as e:
                log.warning(f"  ENSO calibration: {e}")

        # [v17-P5-fix] score_display 在 ENSO 校準後賦值，確保含 ENSO modifier
        for h in ranked:
            h["score_display"] = round(np.clip(h["score"], 0.01, 0.95), 2)

        # ═══════════════════════════════════════════
        # [v14.0] Fleet Operations: 海流剪切 + TSP 巡航路線
        # ═══════════════════════════════════════════
        try:
            import signal
            from engine.fleet_operations import (
                compute_current_shear, enrich_hotspots_with_shear,
                compute_optimal_route,
            )

            # 1. 海流剪切預警
            _u_surf = data.get("u_current")
            _v_surf = data.get("v_current")
            _u_deep = data.get("u_current_deep") or data.get("u_300m")
            _v_deep = data.get("v_current_deep") or data.get("v_300m")
            shear_data = compute_current_shear(
                _u_surf, _v_surf, _u_deep, _v_deep, lats, lons
            )
            if shear_data is not None:
                ranked = enrich_hotspots_with_shear(ranked, shear_data, lats, lons)
                results["current_shear"] = {
                    "max_shear": float(shear_data["shear_magnitude"].max()),
                    "warn_cells": int(np.sum(shear_data["risk_level"] == 1)),
                    "danger_cells": int(np.sum(shear_data["risk_level"] == 2)),
                }

            # 2. TSP 巡航路線
            route_result = compute_optimal_route(
                ranked,
                vessel_lat=self.vessel_pos[0],
                vessel_lon=self.vessel_pos[1],
                max_stops=7,
            )
            results["optimized_route"] = route_result

        except Exception as e:
            log.warning(f"  Fleet Ops 失敗 (不影響主功能): {e}")
            results.setdefault("optimized_route", {"route": [], "total_distance_nm": 0})

        # ═══════════════════════════════════════════
        # Step 7.8: 路線成本 [v13]
        # ═══════════════════════════════════════════
        if ROUTE_OK and ranked:
            log.info("\n  Step 7.8: Route cost calculation")
            try:
                route_costs = compute_all_routes(
                    [{"lat": h["lat"], "lon": h["lon"]} for h in ranked],
                    port=tuple(self.vessel_pos),
                )
                for i, rc in enumerate(route_costs):
                    # match by lat/lon
                    for h in ranked:
                        if abs(h["lat"] - rc["hotspot"]["lat"]) < 0.01 and \
                           abs(h["lon"] - rc["hotspot"]["lon"]) < 0.01:
                            h["route_cost_usd"] = rc["total_cost_usd"]
                            h["distance_km"] = rc["distance_km"]
                            h["fuel_ton"] = rc["fuel_total_ton"]
                            h["transit_days"] = rc["transit_days"]
                            h["bearing"] = rc["bearing_compass"]
                            break
                results["route_costs"] = route_costs
                if route_costs:
                    log.info(f"  Routes: cheapest ${route_costs[0]['total_cost_usd']:.0f} "
                             f"({route_costs[0]['distance_km']:.0f}km)")
            except Exception as e:
                log.warning(f"  Route cost: {e}")

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
        # Step 8.5: P0 modules (GreenFish/AZTI/NOAA emulation)
        # ═══════════════════════════════════════════
        log.info("\n  Step 8.5: Advanced analytics (GreenFish/AZTI concept)")

        # --- 8-Day Forecast HSI ---
        try:
            from engine.forecast_hsi import generate_8day_forecast
            forecast_input = data.get("forecast", {})
            if forecast_input:
                fc_result = generate_8day_forecast(
                    forecast_data=forecast_input,
                    lats=lats, lons=lons,
                    species_list=list(hsi_results.keys()),
                    current_hsi_results=hsi_results,
                )
                results["forecast_hsi"] = fc_result
            else:
                log.info("  📅 Forecast: no CMEMS forecast data (cached?)")
        except Exception as e:
            log.warning(f"  Forecast HSI: {e}")

        # --- [v16] ③ 台灣漁市場行情 → CPUE 校正 ---
        market_calibration = None
        try:
            from engine.fishery_market import fetch_fishery_market_data, calibrate_cpue_baseline
            from engine.cpue_estimator import _BASELINE_CPUE
            market_data = fetch_fishery_market_data()
            if market_data.get("_meta", {}).get("records", 0) > 0:
                market_calibration, cal_log = calibrate_cpue_baseline(
                    _BASELINE_CPUE, market_data, now.month,
                )
                results["fishery_market"] = market_data.get("_meta", {})
                results["cpue_calibration"] = cal_log
            else:
                log.info("  🐟 漁市行情: no data, using original CPUE baseline")
        except Exception as e:
            log.warning(f"  Fishery Market: {e}")

        # --- CPUE Estimator + Bootstrap ---
        try:
            from engine.cpue_estimator import enrich_hotspots_with_cpue
            if ranked:
                enrich_hotspots_with_cpue(
                    ranked, month=now.month, oni=self._oni,
                    market_calibration=market_calibration,
                )
        except Exception as e:
            log.warning(f"  CPUE Estimator: {e}")

        # --- Species Probability + Conformal Prediction ---
        try:
            from engine.species_probability import enrich_hotspots_with_species_prob
            if ranked:
                enrich_hotspots_with_species_prob(
                    ranked, hsi_results=hsi_results,
                    lats=lats, lons=lons, month=now.month,
                )
        except Exception as e:
            log.warning(f"  Species Probability: {e}")

        # --- GFW Fishing Effort Heatmap (P1) ---
        try:
            from engine.gfw_heatmap import compute_gfw_heatmap, bayesian_hsi_fusion, enrich_hotspots_with_gfw
            gfw_density = compute_gfw_heatmap(lats, lons, month=now.month)
            results["gfw_density"] = gfw_density
            if ranked:
                enrich_hotspots_with_gfw(ranked, gfw_density, lats, lons)
        except Exception as e:
            log.warning(f"  GFW Heatmap: {e}")

        # --- Argo Float T/S Profiles (P2) ---
        argo_mld = gf_mld
        argo_z20 = gf_z20
        try:
            from engine.argo_profiles import fetch_argo_profiles, build_argo_mld_grid
            argo_profs = await fetch_argo_profiles(
                lat_range=self.lat_range, lon_range=self.lon_range,
                days_back=30, max_profiles=50,
            )
            if argo_profs:
                argo_mld, argo_z20 = build_argo_mld_grid(
                    argo_profs, lats, lons, satellite_mld=gf_mld,
                )
                results["argo_profiles"] = len(argo_profs)
        except Exception as e:
            log.warning(f"  Argo Profiles: {e}")

        # --- Hook Depth Recommendation (P1+P2 enhanced) ---
        try:
            from engine.hook_depth import enrich_hotspots_with_hook_depth
            if ranked:
                enrich_hotspots_with_hook_depth(
                    ranked,
                    mld_grid=argo_mld, z20_grid=argo_z20,
                    lats=lats, lons=lons, month=now.month,
                )
        except Exception as e:
            log.warning(f"  Hook Depth: {e}")

        # --- Temporal Forecast / LSTM (P1) ---
        try:
            from engine.temporal_forecast import compute_trend_signals
            forecast_input = data.get("forecast", {})
            if forecast_input and sst is not None:
                trend = compute_trend_signals(forecast_input, sst, lats, lons)
                results["temporal_trend"] = trend
                log.info(f"  📈 Trend: {trend.get('signal', 'N/A')}")
        except Exception as e:
            log.warning(f"  Temporal Forecast: {e}")

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
                forecast_result=results.get("forecast_hsi"),
                gfw_density=results.get("gfw_density"),
                lats=lats,
                lons=lons,
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

        # --- P2: Incremental Learning (log + finetune check) ---
        try:
            from engine.incremental_learner import log_run_results, maybe_finetune
            n_runs = log_run_results(ranked)
            if n_runs >= 30:
                maybe_finetune(model_dir=str(Path("models")))
        except Exception as e:
            log.warning(f"  Incremental Learner: {e}")

        elapsed = time.time() - t0
        results["elapsed_seconds"] = elapsed
        results["data_sources"] = data.get("data_sources", {})
        # [商用] 計算風場/洋流統計 — sea_conditions API 需要
        _ws, _wd, _cs, _cd = None, None, None, None
        try:
            if wind_u is not None and wind_v is not None:
                _ws_grid = np.sqrt(wind_u**2 + wind_v**2)
                _ws = round(float(np.nanmean(_ws_grid)), 1)
                # [v13.2-fix] A-6: 圓形平均 (circular mean) — 避免 0°/360° 邊界錯誤
                _wd_rad = np.arctan2(-wind_u, -wind_v)
                _wd = round(float(np.degrees(np.arctan2(
                    np.nanmean(np.sin(_wd_rad)), np.nanmean(np.cos(_wd_rad))
                )) % 360), 0)
            if u_current is not None and v_current is not None:
                _cs_grid = np.sqrt(u_current**2 + v_current**2)
                _cs = round(float(np.nanmean(_cs_grid)), 2)
                _cd_rad = np.arctan2(v_current, u_current)
                _cd = round(float(np.degrees(np.arctan2(
                    np.nanmean(np.sin(_cd_rad)), np.nanmean(np.cos(_cd_rad))
                )) % 360), 0)
        except Exception as e:
            log.debug(f"[降級] main_v10_3.py: {e}")

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
    def _compute_okubo_weiss(u, v, lats=None, lons=None):
        """[v13.2-fix] Okubo-Weiss with physical scale (dx/dy in degrees → m)."""
        ny, nx = u.shape
        # 使用經緯度間距計算真實距離 (m)
        if lats is not None and len(lats) > 1:
            dy = np.abs(lats[1] - lats[0]) * 111320.0  # 1° lat ≈ 111.32 km
        else:
            dy = 0.25 * 111320.0  # 預設 0.25° 解析度
        if lons is not None and len(lons) > 1:
            mean_lat = np.mean(lats) if lats is not None else 20.0
            dx = np.abs(lons[1] - lons[0]) * 111320.0 * np.cos(np.radians(mean_lat))
        else:
            dx = 0.25 * 111320.0 * np.cos(np.radians(20.0))
        du_dy, du_dx = np.gradient(u, dy, dx)
        dv_dy, dv_dx = np.gradient(v, dy, dx)
        sn = du_dx - dv_dy
        ss = dv_dx + du_dy
        omega = dv_dx - du_dy
        return (sn**2 + ss**2 - omega**2).astype(np.float32)


# ═══════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=f"OceanMaster v{VERSION} (台灣雙模式+黑潮)")
    parser.add_argument("--run-now", action="store_true")
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--mode", type=str, default=None,
                        choices=["nearshore", "offshore"],
                        help="近海模式(台灣周邊) / 遠洋模式(西太平洋)")
    parser.add_argument("--lat-min", type=float, default=None)
    parser.add_argument("--lat-max", type=float, default=None)
    parser.add_argument("--lon-min", type=float, default=None)
    parser.add_argument("--lon-max", type=float, default=None)
    parser.add_argument("--vessel-lat", type=float, default=22.61)  # [v13] 高雄港
    parser.add_argument("--vessel-lon", type=float, default=120.28)
    parser.add_argument("--date", type=str, default=None,
                        help="歷史日期 YYYY-MM-DD (預設=今天)")
    parser.add_argument("--output", default="output")
    args = parser.parse_args()

    if args.api:
        # [v13.2-fix] A-10: 直接用 web_server.py 的完整 app (含認證/dashboard/CORS)
        import uvicorn
        uvicorn.run("web_server:app", host="0.0.0.0", port=8000)
    else:
        # [v13] mode 優先, 否則用手動 lat/lon (預設近海)
        if args.mode:
            p = OceanMasterPipeline(
                vessel_pos=(args.vessel_lat, args.vessel_lon),
                output_dir=args.output,
                target_date=args.date,
                mode=args.mode,
            )
        else:
            lat_range = (
                args.lat_min if args.lat_min is not None else 20,
                args.lat_max if args.lat_max is not None else 27,
            )
            lon_range = (
                args.lon_min if args.lon_min is not None else 118,
                args.lon_max if args.lon_max is not None else 125,
            )
            p = OceanMasterPipeline(
                lat_range=lat_range,
                lon_range=lon_range,
                vessel_pos=(args.vessel_lat, args.vessel_lon),
                output_dir=args.output,
                target_date=args.date,
            )
        r = asyncio.run(p.run_full())
        print(f"\nDone: {r.get('elapsed_seconds',0):.1f}s, "
              f"{len(r.get('hotspots',[]))} hotspots")


if __name__ == "__main__":
    main()
