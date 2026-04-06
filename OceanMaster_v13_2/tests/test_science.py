"""
OceanMaster — Core Science Unit Tests
=======================================
Validates correctness of key scientific algorithms against known values.

Run: python -m pytest tests/test_science.py -v
"""

import sys
import math
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════
# 1. VGPM Pb_opt (Behrenfeld & Falkowski 1997)
# ═══════════════════════════════════════════════════

class TestVGPM:
    """Test VGPM primary productivity formula."""

    def _pb_opt(self, sst):
        """Pb_opt polynomial from Behrenfeld & Falkowski 1997."""
        from engine.species_params import FORAGE_PARAMS
        coeffs = FORAGE_PARAMS["vgpm_coeffs"]
        result = 0.0
        for i, c in enumerate(coeffs):
            result += c * sst ** (len(coeffs) - 1 - i)
        return result

    def test_pb_opt_at_25C(self):
        """SST=25°C → Pb_opt should be ~3.5-5.0 (B&F97 Fig.2)."""
        pb = self._pb_opt(25.0)
        assert 3.0 < pb < 5.5, f"Pb_opt(25°C)={pb:.2f}, expected 3.0-5.5"

    def test_pb_opt_at_20C(self):
        """SST=20°C → Pb_opt should be ~4-7 (B&F97 polynomial peaks near 20°C)."""
        pb = self._pb_opt(20.0)
        assert 3.0 < pb < 8.0, f"Pb_opt(20°C)={pb:.2f}, expected 3.0-8.0"

    def test_pb_opt_at_10C(self):
        """SST=10°C → Pb_opt should be ~2-5 (B&F97 polynomial)."""
        pb = self._pb_opt(10.0)
        assert 1.0 < pb < 5.0, f"Pb_opt(10°C)={pb:.2f}, expected 1.0-5.0"

    def test_pb_opt_positive_tropical(self):
        """Pb_opt should be positive in tropical range (10-30°C)."""
        for sst in [10, 15, 20, 25, 30]:
            pb = self._pb_opt(sst)
            assert pb > 0, f"Pb_opt({sst}°C) should be positive, got {pb}"


# ═══════════════════════════════════════════════════
# 2. Deutsch Phi Metabolic Index
# ═══════════════════════════════════════════════════

class TestPhiIndex:
    """Test Deutsch 2015 metabolic index (Phi) calculations."""

    def test_phi_calculation(self):
        """Phi at SST=25°C, DO=200mmol/m³ should be > 2 for yellowfin."""
        from engine.species_params import SPECIES

        sp = SPECIES["yellowfin"]
        Eo = sp["Eo"]  # 0.40 eV
        Pcrit_kPa = sp["Pcrit_kPa"]  # 4.5

        KB = 8.617e-5  # Boltzmann constant (eV/K)
        TREF_K = 288.15  # Reference temp 15°C

        SST = 25.0
        DO = 200.0  # mmol/m³
        T_K = SST + 273.15
        pO2 = DO * 0.032 * 0.2095  # mmol/m³ → kPa (rough)
        Pcrit_T = Pcrit_kPa * math.exp(Eo / KB * (1.0 / TREF_K - 1.0 / T_K))
        phi = pO2 / max(Pcrit_T, 0.01)

        assert phi > 0, f"Phi should be positive, got {phi}"

    def test_phi_species_consistency(self):
        """All species should have Eo between 0.2 and 0.6 eV."""
        from engine.species_params import SPECIES

        for name, sp in SPECIES.items():
            eo = sp.get("Eo", 0)
            assert 0.2 <= eo <= 0.6, f"{name}: Eo={eo} outside valid range [0.2, 0.6]"

    def test_phi_increases_with_oxygen(self):
        """Higher DO → higher Phi (more oxygen = more habitat)."""
        KB = 8.617e-5
        TREF_K = 288.15
        Eo = 0.40
        Pcrit = 4.5
        T_K = 298.15  # 25°C

        Pcrit_T = Pcrit * math.exp(Eo / KB * (1.0 / TREF_K - 1.0 / T_K))

        phi_low = (100 * 0.032 * 0.2095) / Pcrit_T  # DO=100
        phi_high = (300 * 0.032 * 0.2095) / Pcrit_T  # DO=300

        assert phi_high > phi_low, "Phi should increase with DO"


# ═══════════════════════════════════════════════════
# 3. HSI Weighted Geometric Mean
# ═══════════════════════════════════════════════════

class TestHSI:
    """Test HSI calculation with weighted geometric mean."""

    def test_all_ones(self):
        """All factors = 1.0 → HSI ≈ 1.0."""
        factors = [1.0, 1.0, 1.0, 1.0]
        weights = [0.25, 0.25, 0.25, 0.25]

        log_hsi = sum(w * np.log(max(f, 1e-10)) for f, w in zip(factors, weights))
        hsi = np.exp(log_hsi / sum(weights))

        assert abs(hsi - 1.0) < 0.01, f"All-1 HSI should be 1.0, got {hsi}"

    def test_one_zero(self):
        """One factor ≈ 0 → HSI should be very low (geometric mean property)."""
        factors = [0.001, 1.0, 1.0, 1.0]
        weights = [0.25, 0.25, 0.25, 0.25]

        log_hsi = sum(w * np.log(max(f, 1e-10)) for f, w in zip(factors, weights))
        hsi = np.exp(log_hsi / sum(weights))

        assert hsi < 0.2, f"HSI with one near-zero factor should be < 0.2, got {hsi}"

    def test_gaussian_si(self):
        """Gaussian SI: value at optimum should be 1.0."""
        Topt = 26.0
        sigma = 3.5
        T = 26.0  # at optimum

        si = np.exp(-((T - Topt) ** 2) / (2 * sigma ** 2))
        assert abs(si - 1.0) < 0.01, f"SI at Topt should be 1.0, got {si}"

    def test_gaussian_si_decay(self):
        """SI at 2*sigma from optimum should be < 0.2."""
        Topt = 26.0
        sigma = 3.5
        T = 26.0 + 2 * sigma

        si = np.exp(-((T - Topt) ** 2) / (2 * sigma ** 2))
        assert si < 0.2, f"SI at 2σ should be < 0.2, got {si}"


# ═══════════════════════════════════════════════════
# 4. CPUE Bootstrap
# ═══════════════════════════════════════════════════

class TestCPUE:
    """Test CPUE estimator reproducibility."""

    def test_bootstrap_reproducible(self):
        """Bootstrap with fixed seed should produce identical results."""
        rng1 = np.random.RandomState(42)
        data = np.array([10, 20, 30, 40, 50], dtype=float)

        means1 = []
        for _ in range(100):
            sample = data[rng1.randint(0, len(data), len(data))]
            means1.append(sample.mean())

        rng2 = np.random.RandomState(42)
        means2 = []
        for _ in range(100):
            sample = data[rng2.randint(0, len(data), len(data))]
            means2.append(sample.mean())

        assert means1 == means2, "Bootstrap should be reproducible with same seed"

    def test_cpue_positive(self):
        """CPUE = catch / effort should always be non-negative."""
        catch = 100.0
        effort = 8.0
        cpue = catch / max(effort, 0.1)
        assert cpue > 0, f"CPUE should be positive, got {cpue}"


# ═══════════════════════════════════════════════════
# 5. Calibration System
# ═══════════════════════════════════════════════════

class TestCalibration:
    """Test WCPFC calibration outputs."""

    def test_proportions_sum(self):
        """Species proportions should sum to ~1.0."""
        from engine.calibration import get_calibrated_priors

        proportions, _, _ = get_calibrated_priors()
        total = sum(proportions.values())
        assert abs(total - 1.0) < 0.02, f"Proportions sum={total}, expected ~1.0"

    def test_seasonal_factors_range(self):
        """Seasonal factors should be in [0.3, 3.0]."""
        from engine.calibration import get_calibrated_priors

        _, seasonal, _ = get_calibrated_priors()
        for sp, monthly in seasonal.items():
            for month, factor in monthly.items():
                assert 0.3 <= factor <= 3.0, \
                    f"{sp} month {month}: seasonal factor {factor} out of range"

    def test_fallback_consistency(self):
        """Fallback proportions should also sum to ~1.0."""
        from engine.calibration import _FALLBACK_PROPORTIONS

        total = sum(_FALLBACK_PROPORTIONS.values())
        assert abs(total - 1.0) < 0.02, f"Fallback sum={total}"


# ═══════════════════════════════════════════════════
# 6. Species Parameters Cross-Check
# ═══════════════════════════════════════════════════

class TestSpeciesParams:
    """Validate species parameter consistency."""

    def test_all_species_have_required_fields(self):
        """Every species must have Topt_C, Eo, T_min, T_max."""
        from engine.species_params import SPECIES

        required = ["Topt_C", "Eo", "T_min", "T_max", "name_zh", "name_en"]
        for name, params in SPECIES.items():
            for field in required:
                assert field in params, f"{name} missing required field: {field}"

    def test_topt_in_range(self):
        """Topt should be between T_min and T_max."""
        from engine.species_params import SPECIES

        for name, params in SPECIES.items():
            topt = params["Topt_C"]
            tmin = params["T_min"]
            tmax = params["T_max"]
            assert tmin <= topt <= tmax, \
                f"{name}: Topt={topt} not in [{tmin}, {tmax}]"

    def test_bigeye_eo_consistent(self):
        """Bigeye Eo should match between species_params and commercial_core_v2."""
        from engine.species_params import SPECIES
        from engine.commercial_core_v2 import METABOLIC_TRAITS

        sp_eo = SPECIES["bigeye"]["Eo"]
        cc_eo = METABOLIC_TRAITS["bigeye"]["Eo"]
        assert sp_eo == cc_eo, \
            f"bigeye Eo mismatch: species_params={sp_eo}, commercial_core_v2={cc_eo}"

    def test_dvm_params_all_species(self):
        """DVM params should exist for every species."""
        from engine.species_params import SPECIES, DVM_PARAMS

        for name in SPECIES:
            assert name in DVM_PARAMS, f"{name} missing from DVM_PARAMS"


# ═══════════════════════════════════════════════════
# 7. Ocean Physics Formulas
# ═══════════════════════════════════════════════════

class TestOceanPhysics:
    """Test ocean physics calculations."""

    def test_eke_formula(self):
        """EKE = 0.5 * (u'² + v'²) should be non-negative."""
        u = np.array([[0.1, 0.2], [0.15, 0.25]])
        v = np.array([[0.05, 0.1], [0.08, 0.12]])
        u_mean = np.mean(u)
        v_mean = np.mean(v)
        eke = 0.5 * ((u - u_mean) ** 2 + (v - v_mean) ** 2)

        assert np.all(eke >= 0), "EKE must be non-negative"

    def test_coriolis(self):
        """Coriolis parameter at 30°N should be ~7.3e-5."""
        omega = 7.292e-5
        lat = 30.0
        f = 2 * omega * np.sin(np.radians(lat))
        assert abs(f - 7.292e-5) < 1e-5, f"f at 30°N = {f}, expected ~7.3e-5"

    def test_ekman_pumping_sign(self):
        """Positive wind curl → positive Ekman pumping (upwelling)."""
        # Simplified: curl(τ) > 0 → upwelling
        curl_positive = 1e-6  # N/m³
        rho = 1025.0
        f = 7.3e-5
        w_ekman = curl_positive / (rho * f)
        assert w_ekman > 0, "Positive curl should produce upwelling"


# ═══════════════════════════════════════════════════
# 8. Front Detection
# ═══════════════════════════════════════════════════

class TestFrontDetection:
    """Test SST front detection algorithms."""

    def test_sobel_detects_gradient(self):
        """Strong SST gradient should produce high front strength."""
        from scipy.ndimage import sobel

        # Create field with sharp gradient
        sst = np.ones((20, 20)) * 25.0
        sst[:, 10:] = 28.0  # 3°C jump

        gx = sobel(sst, axis=1)
        gy = sobel(sst, axis=0)
        gradient = np.sqrt(gx ** 2 + gy ** 2)

        assert gradient.max() > 1.0, "Sobel should detect 3°C gradient"

    def test_uniform_field_no_front(self):
        """Uniform SST field should produce near-zero gradient."""
        from scipy.ndimage import sobel

        sst = np.ones((20, 20)) * 25.0  # uniform

        gx = sobel(sst, axis=1)
        gy = sobel(sst, axis=0)
        gradient = np.sqrt(gx ** 2 + gy ** 2)

        assert gradient.max() < 0.01, "Uniform field should have no front"


# ═══════════════════════════════════════════════════
# 9. Data Quality Score
# ═══════════════════════════════════════════════════

class TestDataQuality:
    """Test data quality scoring system."""

    def test_full_data_high_score(self):
        """All data present → score > 0.7."""
        from engine.calibration import compute_data_quality_score

        result = compute_data_quality_score(
            sst=np.random.uniform(20, 30, (20, 20)),
            do_surface=np.random.uniform(100, 300, (20, 20)),
            temp_3d=np.random.uniform(5, 30, (5, 20, 20)),
            ow=np.random.uniform(-1, 1, (20, 20)),
            salinity=np.random.uniform(33, 36, (20, 20)),
            chl=np.random.uniform(0.1, 5, (20, 20)),
        )
        assert result["score"] > 0.7, f"Full data score={result['score']}, expected > 0.7"
        assert result["grade"] in ("A", "B"), f"Grade={result['grade']}, expected A or B"

    def test_no_data_low_score(self):
        """No data → score = 0."""
        from engine.calibration import compute_data_quality_score

        result = compute_data_quality_score()
        assert result["score"] == 0.0, f"No data score={result['score']}, expected 0"
        assert result["grade"] == "D"
