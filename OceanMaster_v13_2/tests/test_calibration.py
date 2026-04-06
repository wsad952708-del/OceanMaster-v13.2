"""
OceanMaster — Calibration System Tests
========================================
驗證 WCPFC 校準模組正確性:
1. WCPFCCalibrator 讀取 LONGLINE.CSV
2. 產出的 proportions 總和 ≈ 1.0
3. data_quality_score 回傳 0-1 範圍
4. species_probability 用校準資料後仍穩定

Run: pytest tests/test_calibration.py -v
"""

import os
import sys
import pytest
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


# ═══════════════════════════════════════════════════
# 1. Calibration Module Import
# ═══════════════════════════════════════════════════

def test_calibration_import():
    """calibration module must import without error."""
    from engine.calibration import (
        WCPFCCalibrator,
        compute_data_quality_score,
        get_calibrated_priors,
    )
    assert WCPFCCalibrator is not None
    assert compute_data_quality_score is not None
    assert get_calibrated_priors is not None


# ═══════════════════════════════════════════════════
# 2. WCPFCCalibrator
# ═══════════════════════════════════════════════════

def test_wcpfc_calibrator_load():
    """WCPFCCalibrator must load from LONGLINE.CSV if exists."""
    from engine.calibration import WCPFCCalibrator, WCPFC_CSV

    if not WCPFC_CSV.exists():
        pytest.skip("WCPFC LONGLINE.CSV not available")

    cal = WCPFCCalibrator()
    success = cal.load()
    assert success, "WCPFCCalibrator.load() failed"


def test_wcpfc_proportions_sum_to_one():
    """Calibrated species proportions must sum to ~1.0."""
    from engine.calibration import WCPFCCalibrator, WCPFC_CSV

    if not WCPFC_CSV.exists():
        pytest.skip("WCPFC LONGLINE.CSV not available")

    cal = WCPFCCalibrator()
    cal.load()

    props = cal.species_proportions
    assert len(props) > 0, "No proportions calculated"
    total = sum(props.values())
    assert abs(total - 1.0) < 0.01, f"Proportions sum to {total}, expected ~1.0"


def test_wcpfc_seasonal_factors_range():
    """Seasonal factors must be in [0.3, 3.0] range."""
    from engine.calibration import WCPFCCalibrator, WCPFC_CSV

    if not WCPFC_CSV.exists():
        pytest.skip("WCPFC LONGLINE.CSV not available")

    cal = WCPFCCalibrator()
    cal.load()

    for sp, monthly in cal.seasonal_factors.items():
        for month_key, factor in monthly.items():
            assert 0.3 <= factor <= 3.0, (
                f"{sp} month {month_key}: seasonal factor {factor} out of range"
            )


def test_wcpfc_residuals_range():
    """Credible residuals must be in (0, 1] range."""
    from engine.calibration import WCPFCCalibrator, WCPFC_CSV

    if not WCPFC_CSV.exists():
        pytest.skip("WCPFC LONGLINE.CSV not available")

    cal = WCPFCCalibrator()
    cal.load()

    for sp, r in cal.credible_residuals.items():
        assert 0.0 < r <= 1.0, f"{sp}: residual {r} out of range (0, 1]"


# ═══════════════════════════════════════════════════
# 3. Data Quality Score
# ═══════════════════════════════════════════════════

def test_data_quality_score_full_data():
    """Full data should produce high quality score."""
    from engine.calibration import compute_data_quality_score

    sst = np.random.uniform(20, 30, (20, 20)).astype(np.float32)
    do_surface = np.random.uniform(3, 7, (20, 20)).astype(np.float32)
    temp_3d = np.random.uniform(5, 30, (5, 20, 20)).astype(np.float32)
    ow = np.random.uniform(-1e-8, 1e-8, (20, 20)).astype(np.float32)
    salinity = np.full((20, 20), 35.0, dtype=np.float32)
    chl = np.random.uniform(0.1, 1.0, (20, 20)).astype(np.float32)

    result = compute_data_quality_score(
        sst=sst, do_surface=do_surface, temp_3d=temp_3d,
        ow=ow, salinity=salinity, chl=chl,
    )

    assert 0.0 <= result["score"] <= 1.0
    assert result["grade"] in ("A", "B", "C", "D")
    assert "breakdown" in result
    assert "note" in result
    # With all data present and clean, should be A or B
    assert result["score"] > 0.6, f"Full data should score > 0.6, got {result['score']}"


def test_data_quality_score_no_data():
    """No data should produce low quality score."""
    from engine.calibration import compute_data_quality_score

    result = compute_data_quality_score()
    assert result["score"] == 0.0
    assert result["grade"] == "D"


def test_data_quality_score_partial_data():
    """SST-only should produce intermediate score."""
    from engine.calibration import compute_data_quality_score

    sst = np.random.uniform(20, 30, (20, 20)).astype(np.float32)
    result = compute_data_quality_score(sst=sst)
    assert 0.0 < result["score"] < 0.8
    assert result["breakdown"]["sst"] > 0.5


def test_data_quality_score_nan_heavy():
    """SST with many NaNs should reduce score."""
    from engine.calibration import compute_data_quality_score

    sst = np.full((20, 20), np.nan, dtype=np.float32)
    sst[:5, :5] = 25.0  # Only 25/400 = 6.25% valid
    result = compute_data_quality_score(sst=sst)
    assert result["breakdown"]["sst"] < 0.5  # 6.25% valid → low coverage


# ═══════════════════════════════════════════════════
# 4. get_calibrated_priors
# ═══════════════════════════════════════════════════

def test_get_calibrated_priors_returns_triple():
    """get_calibrated_priors must return (proportions, seasonal, residuals)."""
    from engine.calibration import get_calibrated_priors

    props, seasonal, residuals = get_calibrated_priors()

    assert isinstance(props, dict)
    assert isinstance(seasonal, dict)
    assert isinstance(residuals, dict)

    # Must contain tuna species
    for sp in ["yellowfin", "bigeye", "albacore"]:
        assert sp in props, f"{sp} missing from proportions"
        assert sp in residuals, f"{sp} missing from residuals"


# ═══════════════════════════════════════════════════
# 5. Species Probability Integration
# ═══════════════════════════════════════════════════

def test_species_probability_with_calibration():
    """species_probability must work with calibrated priors."""
    from engine.species_probability import compute_species_probability

    hsi = {"yellowfin": 0.7, "bigeye": 0.5, "albacore": 0.3}
    result = compute_species_probability(hsi, month=6)

    assert "probabilities" in result
    assert "primary_species" in result
    assert "calibration_source" in result
    assert "note" in result

    # Probabilities must sum to ~1
    prob_sum = sum(result["probabilities"].values())
    assert abs(prob_sum - 1.0) < 0.01, f"Probs sum to {prob_sum}"

    # Primary species should be yellowfin (highest HSI)
    assert result["primary_species"] == "yellowfin"


def test_species_probability_all_zeros():
    """All-zero HSI should not crash."""
    from engine.species_probability import compute_species_probability

    result = compute_species_probability({})
    assert result["primary_species"] == "unknown"


# ═══════════════════════════════════════════════════
# 6. Commercial Core v2 Integration
# ═══════════════════════════════════════════════════

def test_commercial_hsi_returns_data_quality():
    """CommercialGradeHSI should return data_quality dict."""
    from engine.commercial_core_v2 import CommercialGradeHSI

    chsi = CommercialGradeHSI()
    sst = np.full((10, 10), 25.0, dtype=np.float32)
    chl = np.full((10, 10), 0.3, dtype=np.float32)
    do_surface = np.full((10, 10), 5.0, dtype=np.float32)
    forage = np.full((10, 10), 0.5, dtype=np.float32)
    front = np.full((10, 10), 0.02, dtype=np.float32)

    result = chsi.compute_ultimate_hsi(
        sst=sst, chl=chl, do_surface=do_surface,
        forage_index=forage, front_strength=front,
        species="yellowfin",
    )

    assert "confidence" in result, "Missing backward-compat confidence key"
    assert "data_quality" in result, "Missing data_quality dict"
    assert 0.0 <= result["confidence"] <= 1.0
    dq = result["data_quality"]
    assert "score" in dq
    assert "grade" in dq
    assert "note" in dq
