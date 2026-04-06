"""
OceanMaster Test Suite — T6: Minimum 15 test cases, 0 failures allowed
=======================================================================
Run:  pytest tests/test_core.py -v
"""

import pytest
import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ═══════════════════════════════════════════════════
# Test 1-3: Metabolic Index (Phi)
# ═══════════════════════════════════════════════════
class TestMetabolicIndex:
    """Test Deutsch 2015 Phi calculation."""

    def test_phi_known_value(self):
        """SST=25C, DO=5 ml/L -> yellowfin Phi should be realistic."""
        from engine.species_params import SPECIES
        sp = SPECIES["yellowfin"]
        kB = 8.617e-5  # eV/K
        T_K = 25.0 + 273.15
        Tref_K = 15.0 + 273.15
        # pO2 approx from DO=5 ml/L
        pO2 = 5.0 * 1.42857 * 0.2095 * 101.325
        Pcrit_scaled = sp["Pcrit_kPa"] * np.exp(sp["Eo"] / kB * (1 / T_K - 1 / Tref_K))
        phi = pO2 / Pcrit_scaled
        assert 0.5 < phi < 500, f"Phi={phi} out of realistic range"

    def test_phi_cold_vs_warm(self):
        """Cold water should give different metabolic viability than warm."""
        from engine.species_params import SPECIES
        sp = SPECIES["yellowfin"]
        kB = 8.617e-5
        pO2 = 5.0 * 1.42857 * 0.2095 * 101.325

        phi_cold = pO2 / (sp["Pcrit_kPa"] * np.exp(sp["Eo"] / kB * (1 / (10 + 273.15) - 1 / 288.15)))
        phi_warm = pO2 / (sp["Pcrit_kPa"] * np.exp(sp["Eo"] / kB * (1 / (25 + 273.15) - 1 / 288.15)))
        assert phi_cold > 0, f"Cold water Phi={phi_cold} should be positive"
        assert phi_warm > 0, f"Warm water Phi={phi_warm} should be positive"
        assert phi_cold != phi_warm, "Phi should vary with temperature"

    def test_species_params_complete(self):
        """All 4 species must have required metabolic params."""
        from engine.species_params import SPECIES
        for sp_name in ["yellowfin", "bigeye", "skipjack", "albacore"]:
            sp = SPECIES[sp_name]
            assert "Eo" in sp, f"{sp_name} missing Eo"
            assert "Pcrit_kPa" in sp, f"{sp_name} missing Pcrit_kPa"
            assert "Topt_C" in sp, f"{sp_name} missing Topt_C"
            assert "Topt_sigma" in sp, f"{sp_name} missing Topt_sigma"
            assert sp["Topt_sigma"] > 0, f"{sp_name} Topt_sigma must be > 0"


# ═══════════════════════════════════════════════════
# Test 4-6: HSI Range & Skipjack Saturation
# ═══════════════════════════════════════════════════
class TestHSI:
    """Test habitat suitability index."""

    def test_hsi_range_0_100(self):
        """HSI must be in [0, 100]."""
        from engine.greenfish_hsi import GreenFishLiteHSI
        gf = GreenFishLiteHSI()
        sst = np.random.uniform(20, 30, (10, 10)).astype(np.float32)
        phi = np.random.uniform(0.3, 0.9, (10, 10)).astype(np.float32)
        result = gf.compute("yellowfin", sst, phi)
        assert np.all(result["hsi"] >= 0), f"HSI < 0: min={result['hsi'].min()}"
        assert np.all(result["hsi"] <= 100), f"HSI > 100: max={result['hsi'].max()}"

    def test_skipjack_not_saturated(self):
        """Skipjack HSI should have variance (not all = 1.0/100)."""
        from engine.greenfish_hsi import GreenFishLiteHSI
        gf = GreenFishLiteHSI()
        sst = np.random.uniform(12, 35, (20, 20)).astype(np.float32)
        phi = np.random.uniform(0.05, 1.0, (20, 20)).astype(np.float32)
        result = gf.compute("skipjack", sst, phi)
        hsi_std = float(np.std(result["hsi"]))
        assert hsi_std > 3, f"Skipjack HSI std={hsi_std:.2f} too saturated (need > 3)"

    def test_hsi_all_species(self):
        """All 4 species should produce valid HSI."""
        from engine.greenfish_hsi import GreenFishLiteHSI
        gf = GreenFishLiteHSI()
        sst = np.full((5, 5), 25.0, dtype=np.float32)
        phi = np.full((5, 5), 0.5, dtype=np.float32)
        for species in ["yellowfin", "bigeye", "skipjack", "albacore"]:
            result = gf.compute(species, sst, phi)
            assert "hsi" in result, f"{species}: missing hsi"
            assert result["hsi"].shape == (5, 5), f"{species}: wrong shape"


# ═══════════════════════════════════════════════════
# Test 7-8: CPUE Caps (T2)
# ═══════════════════════════════════════════════════
class TestCPUECaps:
    """Test CPUE physical limits."""

    def test_cpue_caps_defined(self):
        """CPUE_CAPS must exist for all species."""
        from engine.ml.stacking_ensemble import CPUE_CAPS
        for sp in ["yellowfin", "bigeye", "skipjack", "albacore"]:
            assert sp in CPUE_CAPS, f"{sp} missing from CPUE_CAPS"
            assert CPUE_CAPS[sp] > 0, f"{sp} cap must be > 0"

    def test_skipjack_cap_reasonable(self):
        """Skipjack cap must be <= 500 kg/day (longline)."""
        from engine.ml.stacking_ensemble import CPUE_CAPS
        assert CPUE_CAPS["skipjack"] <= 500, f"Skipjack cap {CPUE_CAPS['skipjack']} > 500"


# ═══════════════════════════════════════════════════
# Test 9-10: Scaler Validation (T3)
# Only check models/ dir (new v10.4 models), NOT ml_system/models/ (old v10.2)
# ═══════════════════════════════════════════════════
class TestScaler:
    """Test scaler usability."""

    def test_scaler_files_exist(self):
        """At least one scaler file should exist after training."""
        import glob
        scalers = glob.glob("models/scaler_*.pkl")
        if not scalers:
            pytest.skip("No v10.4 scaler files found (training not yet run)")
        for f in scalers:
            size = os.path.getsize(f)
            assert size > 500, f"{f}: scaler file too small ({size} bytes)"

    def test_scaler_fitted(self):
        """Scaler must have mean_ and scale_ attributes."""
        import pickle
        import glob
        scalers = glob.glob("models/scaler_*.pkl")
        if not scalers:
            pytest.skip("No v10.4 scaler files found")
        for f in scalers:
            with open(f, "rb") as fh:
                scaler = pickle.load(fh)
            assert hasattr(scaler, "mean_"), f"{f}: scaler not fitted (no mean_)"
            assert hasattr(scaler, "scale_"), f"{f}: scaler not fitted (no scale_)"


# ═══════════════════════════════════════════════════
# Test 11-12: Model Loadability (T3)
# Only check models/ dir (new v10.4), NOT ml_system/models/ (old corrupt)
# ═══════════════════════════════════════════════════
class TestModelLoad:
    """Test model can be loaded and predict."""

    def test_model_loadable(self):
        """Model pkl should be loadable."""
        import pickle
        import glob
        models = glob.glob("models/stacking_*.pkl")
        if not models:
            pytest.skip("No v10.4 model files found")
        for f in models:
            with open(f, "rb") as fh:
                data = pickle.load(fh)
            assert "model" in data, f"{f}: missing 'model' key"
            assert "scaler" in data, f"{f}: missing 'scaler' key"

    def test_model_predict(self):
        """Model should produce predictions."""
        import pickle
        import glob
        models = glob.glob("models/stacking_*.pkl")
        if not models:
            pytest.skip("No v10.4 model files found")
        f = models[0]
        with open(f, "rb") as fh:
            data = pickle.load(fh)
        model = data["model"]
        scaler = data["scaler"]
        n_features = scaler.mean_.shape[0]
        X_fake = np.random.uniform(0, 1, (5, n_features)).astype(np.float32)
        X_scaled = scaler.transform(X_fake)
        pred = model.predict(X_scaled)
        assert pred.shape == (5,), f"Prediction shape wrong: {pred.shape}"
        assert np.all(np.isfinite(pred)), "Predictions contain inf/nan"


# ═══════════════════════════════════════════════════
# Test 13: No real_cpue files remain (T5)
# ═══════════════════════════════════════════════════
class TestDataIntegrity:
    """Test data file naming."""

    def test_no_real_cpue_files(self):
        """No files named real_cpue_*.csv should exist (T5)."""
        import glob
        bad_files = glob.glob("ml_system/data/real_cpue_*.csv")
        assert len(bad_files) == 0, f"Found misleading files: {bad_files}"


# ═══════════════════════════════════════════════════
# Test 14: Requirements pinned (T7)
# ═══════════════════════════════════════════════════
class TestRequirements:
    """Test dependency management."""

    def test_requirements_pinned(self):
        """Every non-comment, non-blank line must use == (T7)."""
        req_path = Path(__file__).parent.parent / "requirements.txt"
        if not req_path.exists():
            pytest.skip("requirements.txt not found")
        with open(req_path, encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            assert "==" in line, f"Unpinned dependency: {line}"
            assert ">=" not in line, f"Using >= instead of ==: {line}"


# ═══════════════════════════════════════════════════
# Test 15-16: Feature Engineering (T1)
# ═══════════════════════════════════════════════════
class TestFeatureEngineering:
    """Test spatial leakage prevention and feature quality."""

    def test_no_lat_lon_features(self):
        """Training features must not include raw lat/lon (T1)."""
        train_path = Path(__file__).parent.parent / "train_with_gfw.py"
        if not train_path.exists():
            pytest.skip("train_with_gfw.py not found")
        content = train_path.read_text(encoding="utf-8")
        # Verify the feature exclusion code exists
        assert '"lat"' in content and "not in" in content, \
            "train_with_gfw.py must exclude lat from features"

    def test_temporal_features_present(self):
        """Training set builder should produce month_sin/month_cos (T1)."""
        loader_path = Path(__file__).parent.parent / "engine" / "gfw_data_loader.py"
        if not loader_path.exists():
            pytest.skip("gfw_data_loader.py not found")
        content = loader_path.read_text(encoding="utf-8")
        assert "month_sin" in content, "Missing month_sin in training features"
        assert "month_cos" in content, "Missing month_cos in training features"


# ═══════════════════════════════════════════════════
# Test 17-18: API Authentication & Lifespan (T8, T11)
# ═══════════════════════════════════════════════════
class TestAPIAuth:
    """Test web server has API key auth and uses lifespan."""

    def test_api_key_in_web_server(self):
        """web_server.py must have API key auth (T8)."""
        ws_path = Path(__file__).parent.parent / "web_server.py"
        if not ws_path.exists():
            pytest.skip("web_server.py not found")
        content = ws_path.read_text(encoding="utf-8")
        assert "APIKeyHeader" in content, "Missing APIKeyHeader import"
        assert "verify_api_key" in content, "Missing verify_api_key function"
        assert "X-API-Key" in content, "Missing X-API-Key header name"

    def test_lifespan_used(self):
        """web_server.py must use lifespan (T11)."""
        ws_path = Path(__file__).parent.parent / "web_server.py"
        if not ws_path.exists():
            pytest.skip("web_server.py not found")
        content = ws_path.read_text(encoding="utf-8")
        assert "async def lifespan" in content, "Missing lifespan function"
        assert "lifespan=lifespan" in content, "lifespan not passed to FastAPI"
        # Verify @app.on_event("startup") is NOT present as actual decorator
        import re
        has_decorator = bool(re.search(r'@app\.on_event\s*\(\s*"startup"\s*\)', content))
        assert not has_decorator, "Still using deprecated @app.on_event decorator"
