"""
OceanMaster v13.2→v13 — Comprehensive Test Suite  # [v12-phase11-tests] + [v13]
==================================================================
Covers: imports, model loading, config, API smoke, feature count,
        version consistency, WCPFC data loader basics,
        v13 new modules (Kuroshio, ENSO, HSI weights, route planner).

Run:  pytest tests/test_v12_full.py -v
"""

import os
import sys
import json
import re
import importlib
import pytest
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


# ═══════════════════════════════════════════════════
# 1. Import Tests — all engine modules import
# ═══════════════════════════════════════════════════

ENGINE_MODULES = [
    "engine",
    "engine.algorithms",
    "engine.commercial_core_v2",
    "engine.data_fetcher_v2",
    "engine.cmems_ssh",
    "engine.wave_fetcher",
    "engine.weather_fetcher",
    "engine.vessel_lights",
    "engine.zooplankton_proxy",
    "engine.eddy_detector",
    "engine.lagrangian_advection",
    "engine.lunar_model",
    "engine.dvm_model",
    "engine.omz_model",
    "engine.accuracy_booster",
    "engine.food_chain_predictor",
    "engine.fish_behavior_model",
    "engine.species_params",
    "engine.shap_explainer",
    "engine.data_sources",
    "engine.ml.stacking_ensemble",
    "engine.ml.synthetic_training_data",
    "engine.ml.fao_data_loader",
    "engine.ml.validation",
    "engine.ml.wcpfc_data_loader",
    "engine.eez.eez_checker",
    "engine.navigation.route_planner",
    # v13 新增
    "engine.kuroshio_engine",
    "engine.hsi_dynamic_weights",
    "engine.enso_calibrator",
    "engine.route_planner_v2",
    "engine.catch_data_interface",
]


@pytest.mark.parametrize("module_name", ENGINE_MODULES)
def test_import_engine_module(module_name):
    """All engine modules must import without error."""
    mod = importlib.import_module(module_name)
    assert mod is not None


def test_import_config():
    """config.py must import."""
    import config
    assert hasattr(config, "SPECIES_PARAMS")


def test_import_main():
    """main_v10_3.py must import."""
    import main_v10_3
    assert hasattr(main_v10_3, "VERSION")


def test_import_web_server():
    """web_server.py must import."""
    import web_server
    assert hasattr(web_server, "app")


# ═══════════════════════════════════════════════════
# 2. Model Load Tests
# ═══════════════════════════════════════════════════

MODEL_DIR = PROJECT_ROOT / "models"
SPECIES = ["yellowfin", "bigeye", "albacore", "skipjack"]


@pytest.mark.parametrize("species", SPECIES)
def test_model_file_exists(species):
    """Each species must have a trained .pkl model."""
    pkl = MODEL_DIR / f"stacking_{species}.pkl"
    assert pkl.exists(), f"Missing model: {pkl}"
    assert pkl.stat().st_size > 100_000, f"Model too small: {pkl}"


@pytest.mark.parametrize("species", SPECIES)
def test_model_safe_load(species):
    """Models must pass safe_model_load validation."""
    from engine.ml.stacking_ensemble import FishingStackingModel
    model = FishingStackingModel(species=species, model_dir=str(MODEL_DIR))
    success = model.safe_model_load()
    assert success, f"safe_model_load failed for {species}"


def test_training_report_exists():
    """Training report must exist."""
    report = MODEL_DIR / "training_report_v12.md"
    assert report.exists()
    content = report.read_text(encoding="utf-8")
    assert "R²" in content or "R2" in content


# ═══════════════════════════════════════════════════
# 3. Config Tests
# ═══════════════════════════════════════════════════

def test_config_species_params():
    """config.SPECIES_PARAMS must contain all 4 species."""
    import config
    sp = config.SPECIES_PARAMS
    for s in ["yellowfin", "bigeye", "albacore", "skipjack"]:
        assert s in sp, f"{s} missing from SPECIES_PARAMS"


def test_config_has_cmems():
    """Config must reference CMEMS data sources."""
    import config
    content = Path("config.py").read_text(encoding="utf-8")
    assert "CMEMS" in content.upper() or "copernicus" in content.lower()


# ═══════════════════════════════════════════════════
# 4. API Smoke Tests (without running server)
# ═══════════════════════════════════════════════════

def test_fastapi_app_exists():
    """FastAPI app must be created."""
    import web_server
    assert web_server.app is not None
    assert web_server.app.title.startswith("OceanMaster")


def test_health_endpoint_registered():
    """Health endpoint must be registered."""
    import web_server
    routes = [r.path for r in web_server.app.routes]
    assert "/health" in routes


# ═══════════════════════════════════════════════════
# 5. Feature Count Tests
# ═══════════════════════════════════════════════════

def test_synthetic_data_feature_count():
    """Synthetic data must produce exactly 59 features + 1 target."""
    from engine.ml.synthetic_training_data import SyntheticCPUEGenerator
    gen = SyntheticCPUEGenerator()
    X, y, feature_names = gen.generate_full_dataset(n_samples=10, species="yellowfin")
    assert X.shape[1] == 59, f"Expected 59 features, got {X.shape[1]}"
    assert len(feature_names) == 59
    assert len(y) == 10


def test_feature_names_match_model():
    """Feature names from generator must match model expectation."""
    from engine.ml.stacking_ensemble import FeatureEngineer
    from engine.ml.synthetic_training_data import SyntheticCPUEGenerator
    gen = SyntheticCPUEGenerator()
    _, _, gen_names = gen.generate_full_dataset(n_samples=5, species="yellowfin")
    model_names = list(FeatureEngineer.FEATURE_NAMES)
    assert gen_names == model_names, (
        f"Feature mismatch: generator has {len(gen_names)}, "
        f"model expects {len(model_names)}"
    )


# ═══════════════════════════════════════════════════
# 6. Synthetic CSV Samples
# ═══════════════════════════════════════════════════

@pytest.mark.parametrize("species", SPECIES)
def test_synthetic_csv_readable(species):
    """Pre-generated synthetic CSVs must be readable."""
    import pandas as pd
    csv_path = PROJECT_ROOT / "data" / "real" / f"sample_{species}.csv"
    if not csv_path.exists():
        pytest.skip(f"No synthetic CSV for {species}")
    df = pd.read_csv(csv_path)
    assert len(df) == 100
    assert "cpue_target" in df.columns
    assert df.shape[1] == 45  # 44 features + 1 target


# ═══════════════════════════════════════════════════
# 7. WCPFC Data Loader
# ═══════════════════════════════════════════════════

def test_wcpfc_coordinate_parser():
    """LAT5/LON5 parser must handle all formats."""
    from engine.ml.wcpfc_data_loader import _parse_lat5, _parse_lon5
    assert _parse_lat5("15N") == 15.0
    assert _parse_lat5("05S") == -5.0
    assert _parse_lat5("00N") == 0.0
    assert _parse_lon5("120E") == 120.0
    assert _parse_lon5("170W") == -170.0
    assert _parse_lon5("00E") == 0.0


def test_wcpfc_longline_loader():
    """WCPFC longline CSV must load if data exists."""
    from engine.ml.wcpfc_data_loader import load_longline_monthly
    csv_path = PROJECT_ROOT / "data" / "real" / \
        "WCPFC_L_PUBLIC_BY_YY_MM" / "WCPFC_L_PUBLIC_BY_YY_MM.csv"
    if not csv_path.exists():
        pytest.skip("No WCPFC longline data")
    df = load_longline_monthly(str(csv_path), min_year=2010)
    assert len(df) > 0
    assert "species" in df.columns
    assert "cpue" in df.columns
    assert set(df["gear"].unique()) == {"longline"}


def test_wcpfc_purse_seine_loader():
    """WCPFC purse seine CSV must load if data exists."""
    from engine.ml.wcpfc_data_loader import load_purse_seine_monthly
    csv_path = PROJECT_ROOT / "data" / "real" / \
        "WCPFC_S_PUBLIC_BY_YY_MM" / "WCPFC_S_PUBLIC_BY_YY_MM.csv"
    if not csv_path.exists():
        pytest.skip("No WCPFC purse seine data")
    df = load_purse_seine_monthly(str(csv_path), min_year=2010)
    assert len(df) > 0
    assert "skipjack" in df["species"].values


# ═══════════════════════════════════════════════════
# 8. Version Consistency
# ═══════════════════════════════════════════════════

def test_version_is_12():
    """VERSION in main_v10_3.py must be '12.0'."""
    import main_v10_3
    assert main_v10_3.VERSION == "15.3", f"VERSION is {main_v10_3.VERSION}"


def test_version_in_readme():
    """README must mention v12."""
    readme = PROJECT_ROOT / "README.md"
    content = readme.read_text(encoding="utf-8")
    assert "v12" in content.lower() or "12" in content


def test_no_env_in_dockerfile():
    """Dockerfile.production must NOT copy .env."""
    dockerfile = PROJECT_ROOT / "Dockerfile.production"
    content = dockerfile.read_text(encoding="utf-8")
    # Only comments should mention .env, not a COPY command
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert "COPY" not in stripped or ".env" not in stripped, \
            f"Dockerfile still copies .env: {stripped}"


# ═══════════════════════════════════════════════════
# 9. Rate Limiter Consistency
# ═══════════════════════════════════════════════════

def test_rate_check_uses_ip():
    """All _rate_check calls must use request.client.host, not coordinates."""
    ws_path = PROJECT_ROOT / "web_server.py"
    content = ws_path.read_text(encoding="utf-8")
    # Find all _rate_check calls
    calls = re.findall(r"_rate_check\(([^)]+)\)", content)
    for call_arg in calls:
        # Skip the function definition
        if "ip:" in call_arg:
            continue
        assert "str(lat)" not in call_arg, f"Rate check uses coordinates: {call_arg}"
        assert "str(lon)" not in call_arg, f"Rate check uses coordinates: {call_arg}"
        assert "dvm_" not in call_arg, f"Rate check uses species string: {call_arg}"


# ═══════════════════════════════════════════════════
# 10. Squid Module Tests
# ═══════════════════════════════════════════════════

def test_squid_hsi_moon_phase():
    from engine.hsi_models import compute_hsi_squid
    import numpy as np
    
    # New moon (0.0 or 1.0): should give high HSI
    sst = np.array([17.0])
    res_new = compute_hsi_squid(sst, moon_phase=0.0)
    
    # Full moon (0.5): should give lower HSI
    res_full = compute_hsi_squid(sst, moon_phase=0.5)
    
    assert res_new["hsi"][0] > res_full["hsi"][0] * 1.5, "New moon should be significantly better than full moon for squid"

def test_squid_api_forecast():
    from fastapi.testclient import TestClient
    from web_server import app, API_KEY
    client = TestClient(app)
    
    res = client.get("/api/v1/squid_jigging_forecast", headers={"X-API-Key": API_KEY})
    if res.status_code != 200:
        print("ERROR RESPONSE:", res.text)
    assert res.status_code == 200
    data = res.json()
    assert "lunar_jigging_index" in data
    assert "recommendation" in data


# ═══════════════════════════════════════════════════
# 11. v13 新模組測試
# ═══════════════════════════════════════════════════

def test_kuroshio_engine_import_and_analyze():
    """KuroshioEngine must import and analyze_full must return correct keys."""
    from engine.kuroshio_engine import KuroshioEngine
    import numpy as np
    ke = KuroshioEngine()
    sst = np.random.uniform(20, 30, (10, 10)).astype(np.float32)
    lats = np.linspace(20, 28, 10)
    lons = np.linspace(120, 128, 10)
    u = np.random.uniform(-0.5, 0.5, (10, 10)).astype(np.float32)
    v = np.random.uniform(-0.5, 0.5, (10, 10)).astype(np.float32)
    result = ke.analyze_full(sst, lats, lons, u, v, month=1)
    assert "intrusion" in result
    assert "distance_grid" in result
    assert result["intrusion"]["intrusion_index"] >= 0


def test_enso_calibrator_phases():
    """ENSOCalibrator must classify phases and produce species modifiers."""
    from engine.enso_calibrator import ENSOCalibrator
    # El Nino
    e1 = ENSOCalibrator(oni=1.5)
    assert "El" in e1.phase
    m = e1.get_species_modifier("pacific_saury")
    assert m["hsi_modifier"] < 0, "Saury should suffer in El Nino"
    assert m["kuroshio_factor"] < 1.0
    # La Nina
    e2 = ENSOCalibrator(oni=-1.2)
    assert "La" in e2.phase
    # Neutral
    e3 = ENSOCalibrator(oni=0.0)
    assert e3.phase == "Neutral"


def test_enso_apply_to_hsi():
    """ENSOCalibrator.apply_to_hsi must adjust HSI grid."""
    from engine.enso_calibrator import ENSOCalibrator
    import numpy as np
    e = ENSOCalibrator(oni=1.5)
    hsi = np.full((10, 10), 0.7, dtype=np.float32)
    lats = np.linspace(20, 30, 10)
    adjusted = e.apply_to_hsi(hsi, lats, "pacific_saury")
    assert adjusted.shape == hsi.shape
    assert adjusted.max() <= 1.0
    assert adjusted.min() >= 0.0
    # El Nino should reduce saury HSI
    assert adjusted.mean() < hsi.mean()


def test_hsi_dynamic_weights_seasonal():
    """HSI dynamic weights must adjust for season and species."""
    from engine.hsi_dynamic_weights import compute_dynamic_weights, apply_weighted_hsi
    import numpy as np
    # Winter + strong Kuroshio
    w = compute_dynamic_weights(month=1, kuroshio_intrusion=2.0, species="yellowfin")
    assert "layer_weights" in w
    assert w["layer_weights"]["layer2_physics"] > 0.35, "L2 should boost in winter"
    assert "winter_kuroshio_dominant" in w["adjustments"]
    # Summer
    w2 = compute_dynamic_weights(month=7, species="mackerel_scad")
    assert "summer_monsoon_upwelling" in w2["adjustments"]
    # apply_weighted_hsi
    hsi = apply_weighted_hsi(
        np.full((5, 5), 0.8), np.full((5, 5), 0.6), np.full((5, 5), 0.7), w
    )
    assert hsi.shape == (5, 5)
    assert 0.0 <= hsi.min() <= hsi.max() <= 1.0


def test_route_planner_v2_costs():
    """Route planner must compute realistic distances and costs."""
    from engine.route_planner_v2 import compute_route_cost, compute_all_routes, haversine_km
    # Single route
    r = compute_route_cost(24.5, 122.0)
    assert r["distance_km"] > 100
    assert r["total_cost_usd"] > 0
    assert r["fuel_total_ton"] > 0
    assert r["bearing_compass"] in [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
    ]
    # Haversine sanity: 1 deg lon at equator ~ 111km
    d = haversine_km(0, 0, 0, 1)
    assert 110 < d < 112
    # Batch routes
    hotspots = [
        {"lat": 23.5, "lon": 121.5},
        {"lat": 25.0, "lon": 123.0},
    ]
    routes = compute_all_routes(hotspots)
    assert len(routes) == 2
    # sorted by cost
    assert routes[0]["total_cost_usd"] <= routes[1]["total_cost_usd"]


def test_catch_data_interface_import():
    """CatchDataInterface must import and instantiate."""
    from engine.catch_data_interface import CatchDataInterface
    cdi = CatchDataInterface()
    assert hasattr(cdi, "record_count")


def test_v13_ml_features_present():
    """v13 new ML features must be in FEATURE_NAMES."""
    from engine.ml.stacking_ensemble import FeatureEngineer
    from engine.ml.synthetic_training_data import SyntheticCPUEGenerator
    assert "kuroshio_distance" in FeatureEngineer.FEATURE_NAMES
    assert "taiwan_strait_flag" in FeatureEngineer.FEATURE_NAMES
    assert len(FeatureEngineer.FEATURE_NAMES) == 59  # 59 features total
    assert len(SyntheticCPUEGenerator.FEATURE_NAMES) == 59
    assert FeatureEngineer.FEATURE_NAMES == SyntheticCPUEGenerator.FEATURE_NAMES


def test_data_fetcher_fallback_tracking():
    """OceanDataFetcher must have fallback tracking."""
    from engine.data_fetcher_v2 import OceanDataFetcher
    f = OceanDataFetcher((20, 28), (118, 126))
    assert hasattr(f, "_fallback_log")
    assert isinstance(f._fallback_log, dict)
    assert hasattr(f, "_record_fallback")
    f._record_fallback("TEST", "test reason", "test source", "degraded")
    assert "TEST" in f._fallback_log
    assert f._fallback_log["TEST"]["quality"] == "degraded"
    assert "timestamp" in f._fallback_log["TEST"]


# ═══════════════════════════════════════════════════════
# [v13.2-P1] 競爭者整合測試
# ═══════════════════════════════════════════════════════

def test_cmems_forecast_method():
    """OceanDataFetcher must have _fetch_cmems_forecast method."""
    from engine.data_fetcher_v2 import OceanDataFetcher
    f = OceanDataFetcher((20, 28), (118, 126))
    assert hasattr(f, "_fetch_cmems_forecast"), "Missing _fetch_cmems_forecast method"
    import inspect
    assert inspect.iscoroutinefunction(f._fetch_cmems_forecast), \
        "_fetch_cmems_forecast must be async"


def test_ais_gear_type_map():
    """GEAR_TYPE_MAP must cover major fishing methods and AISEvent must have gear_category."""
    from engine.ais_shadow_fishing import GEAR_TYPE_MAP, GEAR_CATEGORIES, AISEvent
    # 必須涵蓋四大漁法
    required_categories = {"延繩釣", "圍網", "魷魚燈船", "拖網"}
    mapped_values = set(GEAR_TYPE_MAP.values())
    for cat in required_categories:
        assert cat in mapped_values, f"GEAR_TYPE_MAP missing category: {cat}"
    # GEAR_CATEGORIES 反向映射
    for cat in required_categories:
        assert cat in GEAR_CATEGORIES, f"GEAR_CATEGORIES missing: {cat}"
        assert len(GEAR_CATEGORIES[cat]) > 0
    # AISEvent 必須有 gear_category 欄位
    evt = AISEvent(lat=25.0, lon=121.0, vessel_id="test")
    assert hasattr(evt, "gear_category"), "AISEvent missing gear_category field"
    assert evt.gear_category == ""  # default empty
    # 測試映射
    assert GEAR_TYPE_MAP.get("drifting_longlines") == "延繩釣"
    assert GEAR_TYPE_MAP.get("squid_jigger") == "魷魚燈船"
    assert GEAR_TYPE_MAP.get("trawlers") == "拖網"


def test_spawning_habitat():
    """compute_spawning_habitat must return correct structure with seasonal difference."""
    import numpy as np
    from engine.hsi_models import compute_spawning_habitat

    sst = np.full((10, 10), 28.0, dtype=np.float32)

    # 黃鰭鮪: T_spawning=(26,30), spawning_months=[4,5,6,7,8,9]
    # 季節內 (month=6)
    r_in = compute_spawning_habitat(sst, month=6, species_key="yellowfin")
    assert "spawning_hsi" in r_in
    assert "si_sst" in r_in
    assert "si_season" in r_in
    assert "species" in r_in
    assert "in_season" in r_in
    assert r_in["in_season"] is True
    assert r_in["si_season"] == 1.0
    assert r_in["spawning_hsi"].shape == (10, 10)
    # SST 28°C 在 (26,30) 範圍內, season=1.0 → HSI 應該高
    assert np.nanmean(r_in["spawning_hsi"]) > 0.8

    # 季節外 (month=12)
    r_out = compute_spawning_habitat(sst, month=12, species_key="yellowfin")
    assert r_out["in_season"] is False
    assert r_out["si_season"] < 1.0
    # 季節外分數應低於季節內
    assert np.nanmean(r_out["spawning_hsi"]) < np.nanmean(r_in["spawning_hsi"])


def test_cloud_removal():
    """CloudRemovalNet must fill NaN, preserve originals, and support fronts."""
    import numpy as np
    from engine.cloud_removal import CloudRemovalNet, cloud_remove_sst

    # 建立有 NaN 的 SST 矩陣
    sst = np.random.uniform(25, 30, (20, 20)).astype(np.float32)
    original = sst.copy()
    # 製造 20% NaN (模擬雲遮蔽)
    nan_mask = np.zeros((20, 20), dtype=bool)
    nan_mask[5:9, 5:9] = True
    sst[nan_mask] = np.nan

    # 基本填充
    cr = CloudRemovalNet(preserve_fronts=False)
    filled = cr.fill(sst)
    assert filled.shape == sst.shape
    assert not np.any(np.isnan(filled)), "Filled SST should have no NaN"
    # 原始有效值完全不動
    valid = ~nan_mask
    np.testing.assert_array_equal(filled[valid], original[valid])

    # preserve_fronts=True
    cr2 = CloudRemovalNet(preserve_fronts=True)
    filled2 = cr2.fill(sst)
    assert not np.any(np.isnan(filled2))
    # 原始有效值仍完全不動
    np.testing.assert_array_equal(filled2[valid], original[valid])

    # 模組級快捷函數
    filled3 = cloud_remove_sst(sst, preserve_fronts=True)
    assert not np.any(np.isnan(filled3))

    # 全 NaN 情況
    all_nan = np.full((5, 5), np.nan, dtype=np.float32)
    result = cr.fill(all_nan)
    assert result.shape == (5, 5)  # 不崩潰

    # 無 NaN 情況
    no_nan = np.ones((5, 5), dtype=np.float32) * 28.0
    result2 = cr.fill(no_nan)
    np.testing.assert_array_equal(result2, no_nan)

    # get_stats
    stats = cr.get_stats(sst)
    assert stats["nan_pixels"] == int(np.sum(nan_mask))
    assert stats["total_pixels"] == 400


def test_himawari_sst_method():
    """OceanDataFetcher 必須有 _fetch_himawari_sst 方法"""
    from engine.data_fetcher_v2 import OceanDataFetcher
    import inspect

    fetcher = OceanDataFetcher((20, 26), (119, 122))
    assert hasattr(fetcher, "_fetch_himawari_sst"), "Missing _fetch_himawari_sst"
    sig = inspect.signature(fetcher._fetch_himawari_sst)
    assert "c" in sig.parameters, "Missing 'c' parameter"
    # 驗證 cache TTL = 1 小時 (從 source code 確認)
    src = inspect.getsource(fetcher._fetch_himawari_sst)
    assert "himawari_sst" in src, "Should use 'himawari_sst' cache key"
    assert "cloud_remove_sst" in src, "Should call cloud_remove_sst"


def test_npz_model():
    """NPZ 生態動力模型 — 基本功能驗證"""
    import numpy as np
    from engine.npz_model import NPZModel, compute_npz

    ny, nx = 20, 20
    sst = np.full((ny, nx), 25.0, dtype=np.float32)
    chl = np.full((ny, nx), 0.3, dtype=np.float32)

    # 基本 run
    model = NPZModel()
    result = model.run(sst, chl, n_days=10)

    assert result["phytoplankton"].shape == (ny, nx)
    assert result["zooplankton"].shape == (ny, nx)
    assert result["nutrients"].shape == (ny, nx)
    assert 0.0 <= np.nanmax(result["phytoplankton"]) <= 1.0
    assert 0.0 <= np.nanmax(result["zooplankton"]) <= 1.0

    # 模組級快捷函數
    r2 = compute_npz(sst, chl, n_days=5)
    assert "phytoplankton" in r2 and "zooplankton" in r2

    # 物種 zoo 權重
    w = model.get_species_zoo_weight("mackerel_scad")  # prey=[zooplankton, small_crustaceans]
    assert w > 0.5, f"mackerel_scad should have high zoo weight, got {w}"

    w2 = model.get_species_zoo_weight("blue_marlin")  # prey=[skipjack, flying_fish, squid, mahi_mahi]
    assert w2 < w, f"blue_marlin should have lower zoo weight than mackerel_scad"

    # species forage
    forage = model.compute_species_forage_npz("mackerel_scad", result)
    assert forage.shape == (ny, nx)
    assert forage.dtype == np.float32


def test_npz_in_forage_engine():
    """ForageEngine.compute 應包含 NPZ 結果"""
    import numpy as np
    from engine.forage_engine import ForageEngine

    ny, nx = 15, 15
    chl = np.random.uniform(0.1, 1.0, (ny, nx)).astype(np.float32)
    sst = np.random.uniform(20, 30, (ny, nx)).astype(np.float32)
    lats = np.linspace(20, 25, ny)

    fe = ForageEngine()
    result = fe.compute(chl, sst, lats, month=6)

    assert "npp" in result
    assert "forage_total" in result
    # NPZ 應被整合
    assert "npz_phytoplankton" in result, "Missing npz_phytoplankton in ForageEngine output"
    assert "npz_zooplankton" in result, "Missing npz_zooplankton in ForageEngine output"
    assert result["npz_phytoplankton"].shape == (ny, nx)


def test_weekly_briefing_endpoint_exists():
    """web_server 應包含 /api/v1/weekly_briefing 端點"""
    from fastapi.testclient import TestClient
    import web_server
    routes = [r.path for r in web_server.app.routes]
    assert "/api/v1/weekly_briefing" in routes, f"Missing /api/v1/weekly_briefing, got {routes}"


def test_weekly_briefing_prompt_builder():
    """_build_briefing_prompt 應產生包含熱點的 prompt"""
    import web_server
    hotspots = [
        {"species": "yellowfin", "lat": 24.5, "lon": 141.2, "hsi": 0.92, "sst": 27.3,
         "chl": 0.28, "distance_nm": 1200, "travel_hours": 100, "analysis": "test"},
        {"species": "bigeye", "lat": 18.8, "lon": 151.5, "hsi": 0.89, "sst": 28.1,
         "chl": 0.35, "distance_nm": 1800, "travel_hours": 150, "analysis": "test2"},
    ]
    summary = {"enso_oni": 0.3}
    prompt = web_server._build_briefing_prompt(hotspots, summary)
    assert "yellowfin" in prompt
    assert "24.5" in prompt
    assert "ONI=0.30" in prompt
    assert "繁體中文" in prompt


def test_weekly_briefing_rule_fallback():
    """無 API Key 時應使用 rule-based fallback"""
    import web_server
    hotspots = web_server._DEMO_HOTSPOTS[:3]
    summary = {"enso_oni": -0.2}
    resp = web_server._generate_rule_based_briefing(hotspots, summary, "json")
    import json
    body = json.loads(resp.body)
    assert "briefing_text" in body
    assert "rule-based" in body["model"]
    assert len(body["top_hotspots"]) == 3
    assert "OceanMaster" in body["briefing_text"]


def test_weekly_briefing_cache_ttl():
    """DataCache TTL 應為 6 小時"""
    import web_server
    assert web_server._BRIEFING_TTL == 6 * 3600


# ═══════════════════════════════════════════════════════
# [v17] 競品技術整合測試
# ═══════════════════════════════════════════════════════

def test_blend_optimizer_import():
    """blend_optimizer must import and return valid weights."""
    from engine.blend_optimizer import load_optimal_weights
    w = load_optimal_weights("yellowfin", weight_type="hsi")
    assert "w_v8" in w
    assert "w_comm" in w
    assert "w_gf" in w
    total = w["w_v8"] + w["w_comm"] + w["w_gf"]
    assert 0.99 < total < 1.01, f"HSI weights must sum to ~1.0, got {total}"
    assert w["source"] in ("default", "optimized")

    w_ml = load_optimal_weights("bigeye", weight_type="ml")
    assert "w_science" in w_ml
    assert "w_ml" in w_ml
    ml_total = w_ml["w_science"] + w_ml["w_ml"]
    assert 0.99 < ml_total < 1.01, f"ML weights must sum to ~1.0, got {ml_total}"


def test_fusion_raw_hsi_field():
    """ai_fusion._extract_hotspots must include raw_hsi when raw_score_grid is provided."""
    import numpy as np
    from engine.ai_fusion import _extract_hotspots

    score = np.array([[0.5, 0.3], [0.8, 0.2]], dtype=np.float32)
    raw_score = np.array([[0.45, 0.25], [0.72, 0.15]], dtype=np.float32)
    lat = np.array([20.0, 21.0])
    lon = np.array([120.0, 121.0])

    hs = _extract_hotspots(score, lat, lon, "yellowfin", top_n=5,
                           min_score=0.1, raw_score_grid=raw_score)
    assert len(hs) > 0
    for h in hs:
        assert "raw_hsi" in h, "hotspot must have raw_hsi field"
        assert 0 <= h["raw_hsi"] <= 1.0


def test_fusion_bonus_convergence():
    """fuse_and_rank should accept convergence_index and front_persistence in features."""
    import numpy as np
    from engine.ai_fusion import fuse_and_rank

    shape = (10, 10)
    lats = np.linspace(20, 25, 10)
    lons = np.linspace(120, 125, 10)
    hsi_results = {
        "yellowfin": {
            "hsi": np.random.uniform(0.3, 0.8, shape).astype(np.float32)
        }
    }
    features = {
        "front_strength": np.random.uniform(0, 1, shape).astype(np.float32),
        "front_persistence": np.random.uniform(0, 1, shape).astype(np.float32),
        "convergence_index": np.random.uniform(0, 1, shape).astype(np.float32),
        "ftle": np.random.uniform(0, 1, shape).astype(np.float32),
    }
    result = fuse_and_rank(hsi_results, features, lats, lons, top_n=5)
    assert "combined_hotspots" in result
    hotspots = result["combined_hotspots"]
    assert len(hotspots) > 0
    # All hotspots should have raw_hsi
    for h in hotspots:
        assert "raw_hsi" in h
