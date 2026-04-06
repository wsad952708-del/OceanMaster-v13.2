"""
OceanMaster v13.2 — 八層深度測試
===================================
Layer 1: 全模組 import 驗證 (83 modules)
Layer 2: 核心類別實例化 (11 classes)
Layer 3: 函數呼叫 + 回傳值驗證 (10 functions)
Layer 4: 資料流 — 特徵工程 pipeline (5 tests)
Layer 5: ML 模型 — 訓練 + 預測 + 一致性 (5 tests)
Layer 6: API 端點 — 完整請求回應 (5 tests)
Layer 7: 跨模組一致性 — FEATURE_NAMES 對齊 (8 tests)
Layer 8: 整合測試 — HSI + AI Fusion + 衛星簡報 (5 tests)

Run: python -m pytest tests/test_deep_8layer.py -v --tb=short
"""

import sys
import importlib
import inspect
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ═════════════════════════════════════════════════════
# Layer 1: 全模組 import 驗證 (所有 engine/*.py)
# ═════════════════════════════════════════════════════

_ENGINE_MODULES = [
    "engine.algorithms",
    "engine.ai_fusion",
    "engine.accuracy_booster",
    "engine.acoustic_proxy",
    "engine.ais_shadow_fishing",
    "engine.blend_optimizer",
    "engine.calibration",
    "engine.captain_reports",
    "engine.carbon_calculator",
    "engine.catch_composition",
    "engine.cmems_ssh",
    "engine.commercial_core_v2",
    "engine.competition_penalty",
    "engine.convergence_detector",
    "engine.cpue_estimator",
    "engine.data_fetcher_v2",
    "engine.data_sources",
    "engine.departure_optimizer",
    "engine.dissolved_oxygen",
    "engine.dvm_model",
    "engine.eddy_detector",
    "engine.eddy_maturity",
    "engine.env_cpue_estimator",
    "engine.fish_behavior_model",
    "engine.fleet_commander",
    "engine.food_chain_predictor",
    "engine.forage_engine",
    "engine.fuel_predictor",
    "engine.gebco_features",
    "engine.geojson_output",
    "engine.greenfish_hsi",
    "engine.hab_detector",
    "engine.habitat_compression",
    "engine.hotspot_downscaler",
    "engine.hsi_dynamic_weights",
    "engine.hsi_models",
    "engine.html_map_generator",
    "engine.kuroshio_engine",
    "engine.lagrangian_advection",
    "engine.land_mask",
    "engine.longline_drift",
    "engine.lunar_model",
    "engine.marine_heatwave",
    "engine.market_optimizer",
    "engine.migration_corridor",
    "engine.mld_forecast",
    "engine.mur_sst_loader",
    "engine.npz_model",
    "engine.ocean_color_fronts",
    "engine.ocean_physics",
    "engine.okubo_weiss",
    "engine.omz_model",
    "engine.pressure_features",
    "engine.rainfall_fetcher",
    "engine.roi_calculator",
    "engine.safety_checker",
    "engine.salinity_fetcher",
    "engine.shap_explainer",
    "engine.species_params",
    "engine.species_probability",
    "engine.sst_change_detector",
    "engine.tactical_features",
    "engine.temporal_forecast",
    "engine.text_briefing",
    "engine.thermocline_fetcher",
    "engine.typhoon_golden_zone",
    "engine.uncertainty_estimator",
    "engine.validation_tracker",
    "engine.vessel_lights",
    "engine.water_mass_classifier",
    "engine.wave_fetcher",
    "engine.weather_fetcher",
    "engine.zooplankton_proxy",
    # ML sub-modules
    "engine.ml.stacking_ensemble",
    "engine.ml.synthetic_training_data",
    "engine.ml.fao_data_loader",
    "engine.ml.validation",
    "engine.ml.wcpfc_data_loader",
    "engine.ml.transfish",
    "engine.ml.unet_fishing",
    # EEZ
    "engine.eez.eez_checker",
]


@pytest.mark.parametrize("module_name", _ENGINE_MODULES)
def test_layer1_import(module_name):
    """Layer 1: 每個 engine 子模組都能成功 import"""
    mod = importlib.import_module(module_name)
    assert mod is not None


# ═════════════════════════════════════════════════════
# Layer 2: 核心類別實例化
# ═════════════════════════════════════════════════════

class TestLayer2Instantiation:
    """Layer 2: 核心類別能成功建立實例"""

    def test_feature_engineer(self):
        from engine.ml.stacking_ensemble import FeatureEngineer
        fe = FeatureEngineer()
        assert hasattr(fe, 'FEATURE_NAMES')
        assert hasattr(fe, 'extract_features')

    def test_synthetic_generator(self):
        from engine.ml.synthetic_training_data import SyntheticCPUEGenerator
        gen = SyntheticCPUEGenerator()
        assert hasattr(gen, 'FEATURE_NAMES')
        assert hasattr(gen, 'generate_full_dataset')

    def test_hsi_models_has_functions(self):
        import engine.hsi_models as m
        assert hasattr(m, 'compute_all_hsi')
        assert hasattr(m, 'compute_hsi_yellowfin')
        assert hasattr(m, 'compute_hsi_bigeye')

    def test_thermocline_fetcher(self):
        from engine.thermocline_fetcher import ThermoclineFetcher
        tf = ThermoclineFetcher.__new__(ThermoclineFetcher)
        assert hasattr(tf, 'extract_temp_at_depth')
        assert hasattr(tf, 'compute_thermocline_gradient')

    def test_ai_fusion(self):
        from engine.ai_fusion import fuse_and_rank
        assert callable(fuse_and_rank)

    def test_text_briefing(self):
        from engine.text_briefing import generate_sat_briefing
        assert callable(generate_sat_briefing)

    def test_species_params(self):
        from engine.species_params import SPECIES
        assert isinstance(SPECIES, dict)
        assert "yellowfin" in SPECIES
        assert "bigeye" in SPECIES
        assert len(SPECIES) >= 6

    def test_npz_model(self):
        from engine.npz_model import NPZModel
        m = NPZModel()
        assert hasattr(m, 'run')

    def test_food_chain(self):
        from engine.food_chain_predictor import FoodChainPredictor
        fcp = FoodChainPredictor()
        # Check it has some callable method
        methods = [m for m in dir(fcp) if not m.startswith('_') and callable(getattr(fcp, m))]
        assert len(methods) > 0, "FoodChainPredictor has no methods"

    def test_safety_checker(self):
        from engine.safety_checker import SafetyLevel, SafetyResult
        assert SafetyLevel is not None
        assert SafetyResult is not None

    def test_calibration(self):
        from engine.calibration import WCPFCCalibrator
        cal = WCPFCCalibrator.__new__(WCPFCCalibrator)
        methods = [m for m in dir(cal) if not m.startswith('_')]
        assert len(methods) > 0, "WCPFCCalibrator has no methods"

    def test_enso_calibrator(self):
        from engine.enso_calibrator import ENSOCalibrator
        cal = ENSOCalibrator()
        assert hasattr(cal, 'get_species_modifier')
        assert hasattr(cal, 'apply_to_hsi')

    def test_lunar_model(self):
        from engine.lunar_model import LunarPhaseEngine
        lpe = LunarPhaseEngine()
        assert lpe is not None


# ═════════════════════════════════════════════════════
# Layer 3: 函數呼叫 + 回傳值驗證
# ═════════════════════════════════════════════════════

class TestLayer3FunctionCalls:
    """Layer 3: 核心函數能正確呼叫並回傳預期格式"""

    def test_hsi_yellowfin(self):
        from engine.hsi_models import compute_hsi_yellowfin
        result = compute_hsi_yellowfin(
            sst=np.array([26.0]), chl=np.array([0.2]), ssh=np.array([0.3]))
        # Returns dict with 'hsi' key
        assert isinstance(result, dict)
        assert 'hsi' in result
        val = float(np.squeeze(result['hsi']))
        assert 0 <= val <= 1

    def test_hsi_bigeye_with_t100(self):
        from engine.hsi_models import compute_hsi_bigeye
        r = compute_hsi_bigeye(
            sst=np.array([22.0]), chl=np.array([0.15]), ssh=np.array([0.2]),
            t100=np.array([10.0]))
        assert isinstance(r, dict)
        val = float(np.squeeze(r['hsi']))
        assert 0 <= val <= 1

    def test_hsi_yellowfin_with_gradient(self):
        from engine.hsi_models import compute_hsi_yellowfin
        r = compute_hsi_yellowfin(
            sst=np.array([27.0]), chl=np.array([0.3]), ssh=np.array([0.2]),
            t100=np.array([18.0]), gradient_strength=np.array([0.15]))
        assert isinstance(r, dict)
        val = float(np.squeeze(r['hsi']))
        assert 0 <= val <= 1

    def test_compute_all_hsi(self):
        from engine.hsi_models import compute_all_hsi
        ocean_features = {
            'sst': np.array([26.0]),
            'chl': np.array([0.2]),
            'ssh': np.array([0.3]),
        }
        results = compute_all_hsi(ocean_features)
        assert isinstance(results, dict)

    def test_thermocline_extract_temp(self):
        from engine.thermocline_fetcher import ThermoclineFetcher
        depths = np.array([0, 50, 100, 200, 500])
        temp_profile = np.array([28.0, 22.0, 15.0, 8.0, 4.0])
        tf = ThermoclineFetcher.__new__(ThermoclineFetcher)
        t100 = tf.extract_temp_at_depth(depths, temp_profile, target_depth=100)
        assert isinstance(t100, (int, float, np.floating))

    def test_thermocline_gradient(self):
        from engine.thermocline_fetcher import ThermoclineFetcher
        depths = np.array([0, 50, 100, 200, 500])
        temps = np.array([28.0, 22.0, 15.0, 8.0, 4.0])
        tf = ThermoclineFetcher.__new__(ThermoclineFetcher)
        result = tf.compute_thermocline_gradient(depths, temps)
        # Returns tuple (gradient_strength, gradient_depth)
        assert isinstance(result, (tuple, dict))

    def test_species_params_fields(self):
        from engine.species_params import SPECIES
        for sp_key, params in SPECIES.items():
            assert "Topt_C" in params, f"{sp_key} missing Topt_C"
            assert "name_zh" in params, f"{sp_key} missing name_zh"

    def test_enso_calibrator(self):
        from engine.enso_calibrator import ENSOCalibrator
        cal = ENSOCalibrator()
        result = cal.get_species_modifier("yellowfin")
        # Returns dict with hsi_modifier etc.
        assert isinstance(result, (int, float, dict))

    def test_lunar_engine(self):
        from engine.lunar_model import LunarPhaseEngine
        engine = LunarPhaseEngine()
        methods = [m for m in dir(engine) if not m.startswith('_')]
        assert len(methods) > 0, "LunarPhaseEngine has no public methods"

    def test_npz_model_run(self):
        from engine.npz_model import NPZModel
        m = NPZModel()
        # NPZ needs 2D arrays
        sst_2d = np.full((3, 3), 26.0)
        chl_2d = np.full((3, 3), 0.3)
        par_2d = np.full((3, 3), 40.0)
        result = m.run(sst=sst_2d, chl=chl_2d, par=par_2d)
        assert result is not None


# ═════════════════════════════════════════════════════
# Layer 4: 資料流 — 特徵工程 pipeline
# ═════════════════════════════════════════════════════

class TestLayer4DataFlow:
    """Layer 4: 特徵工程 pipeline 資料流驗證"""

    def test_synthetic_data_generation(self):
        from engine.ml.synthetic_training_data import SyntheticCPUEGenerator
        gen = SyntheticCPUEGenerator()
        X, y, names = gen.generate_full_dataset(n_samples=20, species="yellowfin")
        assert X.shape[0] == 20
        assert X.shape[1] == 59, f"Expected 59 features, got {X.shape[1]}"
        assert len(y) == 20
        assert len(names) == 59
        assert not np.any(np.isnan(X)), "X contains NaN"
        assert not np.any(np.isinf(X)), "X contains Inf"

    def test_synthetic_data_all_species(self):
        from engine.ml.synthetic_training_data import SyntheticCPUEGenerator
        gen = SyntheticCPUEGenerator()
        for sp in ["yellowfin", "bigeye", "skipjack", "albacore"]:
            X, y, names = gen.generate_full_dataset(n_samples=5, species=sp)
            assert X.shape[1] == 59, f"{sp}: expected 59 features, got {X.shape[1]}"

    def test_synthetic_seahawk_features_present(self):
        from engine.ml.synthetic_training_data import SyntheticCPUEGenerator
        gen = SyntheticCPUEGenerator()
        X, y, names = gen.generate_full_dataset(n_samples=50, species="bigeye")
        seahawk = ["t100", "gradient_strength", "delta_t_surface_100", "chl_lag15d"]
        for feat in seahawk:
            assert feat in names, f"Missing feature: {feat}"
            idx = names.index(feat)
            col = X[:, idx]
            assert not np.all(col == 0), f"{feat} is all zeros"
            assert np.std(col) > 0, f"{feat} has zero variance"

    def test_synthetic_t100_influence_on_cpue(self):
        from engine.ml.synthetic_training_data import SyntheticCPUEGenerator
        gen = SyntheticCPUEGenerator()
        X, y, names = gen.generate_full_dataset(n_samples=200, species="bigeye")
        t100_idx = names.index("t100")
        t100_vals = X[:, t100_idx]
        median_t100 = np.median(t100_vals)
        high_mask = t100_vals > median_t100
        low_mask = ~high_mask
        if np.sum(high_mask) > 5 and np.sum(low_mask) > 5:
            mean_high = np.mean(y[high_mask])
            mean_low = np.mean(y[low_mask])
            assert mean_high != mean_low, "T100 has no effect on CPUE"

    def test_feature_builder_dataclass(self):
        from pipeline.feature_builder import FeatureMatrix
        fm = FeatureMatrix()
        assert hasattr(fm, 't100')
        assert hasattr(fm, 'gradient_strength')
        assert hasattr(fm, 'delta_t_surface_100')
        assert hasattr(fm, 'chl_lag15d')


# ═════════════════════════════════════════════════════
# Layer 5: ML 模型 — 訓練 + 預測 + 一致性
# ═════════════════════════════════════════════════════

class TestLayer5MLPipeline:
    """Layer 5: ML 模型訓練、載入、預測流程"""

    def test_train_mini_model(self):
        from engine.ml.synthetic_training_data import SyntheticCPUEGenerator
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split

        gen = SyntheticCPUEGenerator()
        X, y, names = gen.generate_full_dataset(n_samples=300, species="yellowfin")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        # With 300 samples and 50 trees, R² should be reasonable on synthetic data
        assert score > -0.5, f"Model R² extremely bad: {score:.3f}"

    def test_model_files_exist(self):
        models_dir = PROJECT_ROOT / "models"
        for sp in ["yellowfin", "bigeye", "albacore", "skipjack"]:
            pkl_path = models_dir / f"stacking_{sp}.pkl"
            assert pkl_path.exists(), f"Missing model: {pkl_path}"

    def test_safe_model_load_all_species(self):
        from engine.ml.stacking_ensemble import FishingStackingModel
        for sp in ["yellowfin", "bigeye", "albacore", "skipjack"]:
            model = FishingStackingModel(species=sp)
            success = model.safe_model_load()
            assert success, f"safe_model_load failed for {sp}"

    def test_scaler_files_exist(self):
        models_dir = PROJECT_ROOT / "models"
        for sp in ["yellowfin", "bigeye", "albacore", "skipjack"]:
            scaler_path = models_dir / f"scaler_{sp}.pkl"
            assert scaler_path.exists(), f"Missing scaler: {scaler_path}"

    def test_stacking_ensemble_feature_count(self):
        from engine.ml.stacking_ensemble import FeatureEngineer
        assert len(FeatureEngineer.FEATURE_NAMES) == 59
        assert FeatureEngineer.EXPECTED_FEATURE_COUNT == 59


# ═════════════════════════════════════════════════════
# Layer 6: API 端點 — 完整請求回應
# ═════════════════════════════════════════════════════

class TestLayer6API:
    """Layer 6: FastAPI 端點完整請求/回應驗證"""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api.app import app
        return TestClient(app)

    def test_health_complete(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == "15.3"
        assert data["status"] == "healthy"
        assert "uptime_seconds" in data
        assert "models_loaded" in data

    def test_species_complete(self, client):
        resp = client.get("/species")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 6
        sp = data["species"][0]
        for field in ["key", "name_zh", "name_en", "scientific", "Topt_C"]:
            assert field in sp, f"Missing field: {field}"

    def test_hotspots_geojson(self, client):
        resp = client.get("/hotspots")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "FeatureCollection"

    def test_docs_accessible(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_openapi_schema(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "/health" in schema["paths"]
        assert "/predict" in schema["paths"]


# ═════════════════════════════════════════════════════
# Layer 7: 跨模組一致性 — FEATURE_NAMES 對齊
# ═════════════════════════════════════════════════════

class TestLayer7CrossModuleConsistency:
    """Layer 7: 確保所有模組的特徵名稱和數量一致"""

    def test_feature_names_match_stacking_vs_synthetic(self):
        from engine.ml.stacking_ensemble import FeatureEngineer
        from engine.ml.synthetic_training_data import SyntheticCPUEGenerator
        assert FeatureEngineer.FEATURE_NAMES == SyntheticCPUEGenerator.FEATURE_NAMES

    def test_feature_count_consistency(self):
        from engine.ml.stacking_ensemble import FeatureEngineer
        from engine.ml.synthetic_training_data import SyntheticCPUEGenerator
        assert len(FeatureEngineer.FEATURE_NAMES) == 59
        assert FeatureEngineer.EXPECTED_FEATURE_COUNT == 59
        assert len(SyntheticCPUEGenerator.FEATURE_NAMES) == 59

    def test_seahawk_features_in_all_modules(self):
        from engine.ml.stacking_ensemble import FeatureEngineer
        from engine.ml.synthetic_training_data import SyntheticCPUEGenerator
        seahawk = ["t100", "gradient_strength", "delta_t_surface_100", "chl_lag15d"]
        for feat in seahawk:
            assert feat in FeatureEngineer.FEATURE_NAMES, f"{feat} missing from FeatureEngineer"
            assert feat in SyntheticCPUEGenerator.FEATURE_NAMES, f"{feat} missing from SyntheticCPUEGenerator"

    def test_feature_builder_has_seahawk_fields(self):
        from pipeline.feature_builder import FeatureMatrix
        import dataclasses
        field_names = [f.name for f in dataclasses.fields(FeatureMatrix)]
        for field in ["t100", "gradient_strength", "delta_t_surface_100", "chl_lag15d"]:
            assert field in field_names, f"FeatureMatrix missing field: {field}"

    def test_hsi_models_seahawk_params(self):
        from engine.hsi_models import compute_hsi_yellowfin, compute_hsi_bigeye
        for fn in [compute_hsi_yellowfin, compute_hsi_bigeye]:
            sig = inspect.signature(fn)
            params = list(sig.parameters.keys())
            assert "t100" in params, f"{fn.__name__} missing t100 param"
            assert "gradient_strength" in params, f"{fn.__name__} missing gradient_strength param"

    def test_text_briefing_seahawk_section(self):
        from engine.text_briefing import generate_sat_briefing
        hotspot = {
            "rank": 1, "lat": 24.0, "lon": 131.0,
            "species": "yellowfin", "score": 0.85,
            "sst": 27.0, "chl": 0.3,
            "t100": 15.0, "gradient_strength": 0.12,
            "delta_t_surface_100": 12.0,
        }
        briefing = generate_sat_briefing([hotspot])
        assert isinstance(briefing, str)
        assert len(briefing) > 0

    def test_train_wcpfc_feature_count(self):
        from train_wcpfc import derive_features
        result = derive_features(25.0, 131.0, 6, 2020)
        # May return tuple (features, labels) or just array
        features = result[0] if isinstance(result, tuple) else result
        assert len(features) >= 40, f"Expected >=40 features, got {len(features)}"

    def test_version_consistency(self):
        from api.app import app
        assert app.version == "15.3"


# ═════════════════════════════════════════════════════
# Layer 8: 整合測試 — HSI + AI Fusion + 報告
# ═════════════════════════════════════════════════════

class TestLayer8Integration:
    """Layer 8: 多模組整合測試"""

    def test_hsi_grid_calculation(self):
        from engine.hsi_models import compute_hsi_yellowfin
        sst_grid = np.random.uniform(24, 30, 25)
        chl_grid = np.random.uniform(0.05, 0.5, 25)
        ssh_grid = np.full(25, 0.2)
        result = compute_hsi_yellowfin(sst=sst_grid, chl=chl_grid, ssh=ssh_grid)
        assert isinstance(result, dict)
        hsi = result['hsi']
        assert len(hsi) == 25
        assert np.all(hsi >= 0) and np.all(hsi <= 1)
        assert np.any(hsi > 0)

    def test_full_synthetic_train_predict(self):
        from engine.ml.synthetic_training_data import SyntheticCPUEGenerator
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler

        gen = SyntheticCPUEGenerator()
        X, y, names = gen.generate_full_dataset(n_samples=100, species="yellowfin")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = GradientBoostingRegressor(n_estimators=20, random_state=42)
        model.fit(X_scaled, y)

        X_new, _, _ = gen.generate_full_dataset(n_samples=5, species="yellowfin")
        predictions = model.predict(scaler.transform(X_new))
        assert len(predictions) == 5
        importances = model.feature_importances_
        assert len(importances) == 59

    def test_end_to_end_data_consistency(self):
        from engine.ml.synthetic_training_data import SyntheticCPUEGenerator
        from engine.ml.stacking_ensemble import FeatureEngineer

        gen = SyntheticCPUEGenerator()
        fe = FeatureEngineer()
        assert gen.FEATURE_NAMES == fe.FEATURE_NAMES

        X, y, names = gen.generate_full_dataset(n_samples=10, species="bigeye")
        assert X.shape[1] == len(fe.FEATURE_NAMES) == 59
        assert np.all(np.isfinite(X))
        assert np.all(np.isfinite(y))

    def test_multi_species_hsi_consistency(self):
        from engine.hsi_models import compute_all_hsi
        ocean_features = {
            'sst': np.array([25.0]),
            'chl': np.array([0.2]),
            'ssh': np.array([0.2]),
            't100': np.array([12.0]),
            'gradient_strength': np.array([0.1]),
        }
        results = compute_all_hsi(ocean_features)
        assert isinstance(results, dict)
        assert len(results) > 0

    def test_full_species_params_coverage(self):
        """所有 SPECIES 都有完整參數"""
        from engine.species_params import SPECIES
        required_fields = ["Topt_C", "T_min", "T_max", "name_zh", "name_en"]
        for sp_key, params in SPECIES.items():
            for field in required_fields:
                assert field in params, f"Species '{sp_key}' missing '{field}'"
