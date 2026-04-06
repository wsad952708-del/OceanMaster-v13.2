"""
OceanMaster v13.2 — Species Probability + Uncertainty Quantification
==================================================================
模仿 AZTI 的魚種機率輸出 + 不確定性量化。

使用 existing species HSI scores 做 Softmax 正規化 → 魚種機率分佈。
Credible Prediction Set 計算 prediction set（95% 信賴水準）。
"""

import logging
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger("OceanMaster.SpeciesProb")

# [v16-cal] 動態載入 WCPFC 校準資料 (替代硬編碼 prior)
# 如果 WCPFC 數據可用，使用真實漁獲統計；否則回退到 fallback
try:
    from engine.calibration import get_calibrated_priors
    _HISTORICAL_PROPORTIONS, _SEASONAL_BIAS, _CREDIBLE_RESIDUALS = get_calibrated_priors()
    _CALIBRATION_SOURCE = "WCPFC_LONGLINE_real_data"
except Exception:
    # Fallback: 原始硬編碼值 (帶警告)
    _CALIBRATION_SOURCE = "hardcoded_fallback"
    log.warning("  ⚠️ WCPFC 校準不可用，使用硬編碼 fallback prior")

    _HISTORICAL_PROPORTIONS = {
        "yellowfin": 0.28, "bigeye": 0.25, "skipjack": 0.20,
        "albacore": 0.15, "japanese_flying_squid": 0.06,
        "pacific_saury": 0.04, "other": 0.02,
    }

    _SEASONAL_BIAS = {
        "yellowfin": {1: 0.9, 2: 0.9, 3: 1.0, 4: 1.1, 5: 1.2, 6: 1.2,
                      7: 1.1, 8: 1.0, 9: 0.9, 10: 0.9, 11: 0.9, 12: 0.9},
        "bigeye":    {1: 1.1, 2: 1.1, 3: 1.0, 4: 0.9, 5: 0.9, 6: 0.9,
                      7: 1.0, 8: 1.1, 9: 1.2, 10: 1.2, 11: 1.1, 12: 1.1},
        "skipjack":  {1: 0.8, 2: 0.9, 3: 1.0, 4: 1.1, 5: 1.2, 6: 1.3,
                      7: 1.2, 8: 1.1, 9: 1.0, 10: 0.9, 11: 0.8, 12: 0.8},
        "albacore":  {1: 1.2, 2: 1.1, 3: 1.0, 4: 0.9, 5: 0.8, 6: 0.8,
                      7: 0.9, 8: 1.0, 9: 1.1, 10: 1.2, 11: 1.3, 12: 1.2},
    }

    _CREDIBLE_RESIDUALS = {
        "yellowfin": 0.18, "bigeye": 0.22, "skipjack": 0.15,
        "albacore": 0.25, "japanese_flying_squid": 0.30,
        "pacific_saury": 0.28,
    }


def compute_species_probability(
    species_hsi: Dict[str, float],
    month: int = 2,
    temperature: float = 25.0,
    lat: float = 20.0,
    confidence_level: float = 0.95,
) -> Dict:
    """
    計算各魚種的出現機率 + 95% Credible Prediction Set。

    Args:
        species_hsi: {species: hsi_score} dict
        month: 月份
        temperature: SST (°C)
        lat: 緯度
        confidence_level: 信賴水準

    Returns:
        {
            "probabilities": {"bigeye": 0.42, "yellowfin": 0.35, ...},
            "primary_species": "bigeye",
            "primary_prob": 0.42,
            "prediction_set_95": ["bigeye", "yellowfin"],
            "entropy": 1.23,  # 低 = 確定, 高 = 不確定
        }
    """
    if not species_hsi:
        return {
            "probabilities": {},
            "primary_species": "unknown",
            "primary_prob": 0.0,
            "prediction_set_95": [],
            "entropy": 0.0,
        }

    # 1. 對 HSI 做季節性修正
    adjusted_hsi = {}
    for sp, hsi in species_hsi.items():
        seasonal = _SEASONAL_BIAS.get(sp, {}).get(month, 1.0)
        prior = _HISTORICAL_PROPORTIONS.get(sp, 0.05)
        # Bayes: posterior ∝ likelihood(hsi) × prior(historical)
        adjusted_hsi[sp] = max(0.001, hsi * seasonal * (0.5 + prior))

    # 2. Softmax 正規化 → 機率分佈
    #    用 temperature scaling (τ=0.5) 讓分佈更尖銳
    tau = 0.5
    species_list = list(adjusted_hsi.keys())
    logits = np.array([adjusted_hsi[sp] for sp in species_list])
    logits_scaled = logits / tau
    exp_logits = np.exp(logits_scaled - np.max(logits_scaled))  # numerical stability
    probabilities = exp_logits / exp_logits.sum()

    prob_dict = {sp: round(float(p), 3) for sp, p in zip(species_list, probabilities)}

    # 3. Primary species
    primary_idx = np.argmax(probabilities)
    primary_species = species_list[primary_idx]
    primary_prob = float(probabilities[primary_idx])

    # 4. Credible Prediction Set (cumulative probability threshold)
    #    Include species until cumulative probability >= confidence_level
    #    Also add any species whose residual could push it into the set
    sorted_species = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    prediction_set = []
    cumsum = 0.0
    for sp, prob in sorted_species:
        residual = _CREDIBLE_RESIDUALS.get(sp, 0.25)
        # Include if: cumulative < threshold OR residual is large enough
        if cumsum < confidence_level or prob + residual > 0.15:
            prediction_set.append(sp)
            cumsum += prob
        if cumsum >= confidence_level and prob < 0.05:
            break

    # 5. Shannon entropy (uncertainty measure)
    probs_arr = probabilities[probabilities > 0]
    entropy = float(-np.sum(probs_arr * np.log2(probs_arr)))
    max_entropy = np.log2(len(species_list)) if len(species_list) > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    return {
        "probabilities": prob_dict,
        "primary_species": primary_species,
        "primary_prob": round(primary_prob, 3),
        "prediction_set_95": prediction_set,
        "entropy": round(entropy, 3),
        "normalized_entropy": round(normalized_entropy, 3),
        "uncertainty": "Low" if normalized_entropy < 0.4 else
                       "Medium" if normalized_entropy < 0.7 else "High",
        "calibration_source": _CALIBRATION_SOURCE,
        "note": "相對棲息地適合度，非校準後的出現機率",
    }


def enrich_hotspots_with_species_prob(
    hotspots: List[Dict],
    hsi_results: Dict[str, Dict],
    lats: np.ndarray,
    lons: np.ndarray,
    month: int = 2,
) -> List[Dict]:
    """
    批量為 hotspot 加入魚種機率分佈。

    Args:
        hotspots: hotspot list
        hsi_results: {species: {"hsi": ndarray, ...}}
        lats, lons: 1D coordinate arrays
        month: 月份
    """
    for h in hotspots:
        lat, lon = h.get("lat", 0), h.get("lon", 0)

        # 找最近的 grid index
        li = int(np.argmin(np.abs(lats - lat)))
        lj = int(np.argmin(np.abs(lons - lon)))

        # 收集各物種在此點的 HSI
        species_hsi = {}
        for sp, data in hsi_results.items():
            hsi_grid = data.get("hsi")
            if hsi_grid is not None and li < hsi_grid.shape[0] and lj < hsi_grid.shape[1]:
                val = float(hsi_grid[li, lj])
                if np.isfinite(val):
                    species_hsi[sp] = val

        sst = h.get("sst", 25.0)
        result = compute_species_probability(species_hsi, month=month, temperature=sst, lat=lat)

        h["species_probs"] = result["probabilities"]
        h["primary_species_prob"] = f"{result['primary_species']} ({result['primary_prob']*100:.0f}%)"
        h["prediction_set_95"] = result["prediction_set_95"]
        h["species_uncertainty"] = result["uncertainty"]

    n_low = sum(1 for h in hotspots if h.get("species_uncertainty") == "Low")
    log.info(
        f"  🐟 Species Probability: {len(hotspots)} hotspots, "
        f"{n_low}/{len(hotspots)} low uncertainty"
    )
    return hotspots
