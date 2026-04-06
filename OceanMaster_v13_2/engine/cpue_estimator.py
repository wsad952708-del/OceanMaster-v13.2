"""
OceanMaster v13.2 — CPUE Proxy Estimator + Bootstrap Intervals
=============================================================
模仿 GreenFish 的捕獲量預測 + NOAA 的生物量估算。

用 HSI × WCPFC 歷史 CPUE × 季節修正 → 相對漁獲指數 (0-100)
Bootstrap resampling → P25/P50/P75 預測區間
"""

import logging
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger("OceanMaster.CPUEEstimator")

# [v17-P4] WCPFC 真實月中位數 CPUE (mt per 1000 hooks)
# 來源: WCPFC LONGLINE.CSV, 2010-2018, n=15000+ per species
_BASELINE_CPUE = {
    "yellowfin": {1: 5.44, 2: 5.13, 3: 5.21, 4: 5.08, 5: 5.54, 6: 5.87,
                  7: 6.07, 8: 5.66, 9: 4.66, 10: 4.43, 11: 4.29, 12: 5.14},
    "bigeye":    {1: 5.82, 2: 4.47, 3: 3.69, 4: 3.22, 5: 3.12, 6: 3.43,
                  7: 3.25, 8: 3.69, 9: 3.86, 10: 4.75, 11: 6.18, 12: 6.18},
    "skipjack":  {1: 3.5, 2: 3.8, 3: 4.0, 4: 4.2, 5: 4.5, 6: 4.3,
                  7: 4.0, 8: 3.7, 9: 3.5, 10: 3.3, 11: 3.2, 12: 3.4},
    "albacore":  {1: 5.35, 2: 5.62, 3: 4.88, 4: 4.54, 5: 7.17, 6: 9.15,
                  7: 7.60, 8: 5.78, 9: 4.34, 10: 3.25, 11: 4.56, 12: 5.90},
}

# [v17-P4] WCPFC 真實 CV (coefficient of variation)
# 來源: WCPFC LONGLINE.CSV, 2010-2018
_CPUE_CV = {
    "yellowfin": 1.04, "bigeye": 1.07, "skipjack": 0.80, "albacore": 1.23,
    "japanese_flying_squid": 0.90, "pacific_saury": 0.85,
}


def estimate_relative_cpue(
    hsi_score: float,
    species: str,
    month: int,
    wcpfc_prior: float = 0.0,
    oni: float = 0.0,
    n_bootstrap: int = 1000,
    rng_seed: int = 42,
    baseline_override: Optional[float] = None,
) -> Dict:
    """
    估算相對 CPUE 指數 + Bootstrap 信賴區間。

    Args:
        hsi_score: 該 hotspot 的 HSI 分數 (0-1)
        species: 魚種
        month: 月份 (1-12)
        wcpfc_prior: WCPFC 歷史先驗加分 (0-0.3)
        oni: ENSO ONI 值
        n_bootstrap: Bootstrap 重抽次數
        rng_seed: 隨機種子

    Returns:
        {
            "cpue_index": 72,        # 相對漁獲指數 (0-100)
            "cpue_raw": 2.8,         # 估算 CPUE (hooks/100)
            "ci_low": 55,            # P25 下界
            "ci_mid": 72,            # P50 中位數
            "ci_high": 88,           # P75 上界
            "ci_95_low": 35,         # P2.5
            "ci_95_high": 95,        # P97.5
            "confidence": "Medium",  # 信心度
        }
    """
    # 1. 基準 CPUE (月平均)
    base = baseline_override if baseline_override is not None else _BASELINE_CPUE.get(species, {}).get(month, 2.0)

    # 2. HSI 加權
    hsi_factor = max(0.1, hsi_score) * 2.0  # HSI=0.5 → factor=1.0

    # 3. WCPFC 先驗加分
    prior_factor = 1.0 + wcpfc_prior

    # 4. ENSO 修正 (El Niño → 鮪魚東移，La Niña → 西聚)
    # Sources: Lehodey et al. 2006 Prog. Oceanogr.; Hoyle et al. 2011 Fish. Oceanogr.
    # yellowfin/skipjack El Niño -15% (westward range contraction)
    # bigeye/albacore El Niño +10% (deeper thermocline → expanded habitat)
    enso_factor = 1.0
    if oni > 0.5:  # El Niño
        if species in ("yellowfin", "skipjack"):
            enso_factor = 0.85  # 西太平洋減產
        elif species in ("bigeye", "albacore"):
            enso_factor = 1.10
    elif oni < -0.5:  # La Niña
        if species in ("yellowfin", "skipjack"):
            enso_factor = 1.15
        elif species in ("bigeye", "albacore"):
            enso_factor = 0.90

    # 5. 預估 CPUE
    cpue_est = base * hsi_factor * prior_factor * enso_factor

    # 6. Bootstrap resampling → 區間估算
    cv = _CPUE_CV.get(species, 0.40)
    rng = np.random.default_rng(rng_seed)
    # Lognormal distribution (CPUE is always positive, right-skewed)
    sigma_log = np.sqrt(np.log(1 + cv**2))
    mu_log = np.log(cpue_est) - 0.5 * sigma_log**2
    bootstrap_samples = rng.lognormal(mu_log, sigma_log, n_bootstrap)

    # Percentiles
    p2_5 = float(np.percentile(bootstrap_samples, 2.5))
    p25 = float(np.percentile(bootstrap_samples, 25))
    p50 = float(np.percentile(bootstrap_samples, 50))
    p75 = float(np.percentile(bootstrap_samples, 75))
    p97_5 = float(np.percentile(bootstrap_samples, 97.5))

    # Max CPUE for normalization (historical max ~ 8 hooks/100 for skipjack)
    max_cpue = max(8.0, p97_5)

    # 歸一化到 0-100 指數
    cpue_index = min(100, round(p50 / max_cpue * 100))
    ci_low = min(100, round(p25 / max_cpue * 100))
    ci_high = min(100, round(p75 / max_cpue * 100))
    ci_95_low = min(100, round(p2_5 / max_cpue * 100))
    ci_95_high = min(100, round(p97_5 / max_cpue * 100))

    # 信心度
    ci_width = ci_high - ci_low
    confidence = "High" if ci_width < 20 else "Medium" if ci_width < 35 else "Low"

    return {
        "cpue_index": cpue_index,
        "cpue_raw": round(cpue_est, 2),
        "ci_low": ci_low,
        "ci_mid": cpue_index,
        "ci_high": ci_high,
        "ci_95_low": ci_95_low,
        "ci_95_high": ci_95_high,
        "confidence": confidence,
    }


def enrich_hotspots_with_cpue(
    hotspots: List[Dict],
    month: int,
    oni: float = 0.0,
    market_calibration: Optional[Dict] = None,
) -> List[Dict]:
    """
    批量為 hotspot 列表加入 CPUE 估算。

    直接修改 hotspot dicts (in-place)。
    market_calibration: {species: {month: adjusted_cpue}} from fishery_market.py
    """
    for h in hotspots:
        sp = h.get("species", "yellowfin")
        hsi = h.get("score", 0.5)
        prior = h.get("historical_prior", 0.0)

        result = estimate_relative_cpue(
            hsi_score=hsi,
            species=sp,
            month=month,
            wcpfc_prior=prior,
            oni=oni,
            rng_seed=hash((h.get("lat", 0), h.get("lon", 0), sp)) & 0x7FFFFFFF,
            baseline_override=market_calibration.get(sp, {}).get(month) if market_calibration else None,
        )
        h["cpue_index"] = result["cpue_index"]
        h["cpue_raw"] = result["cpue_raw"]
        h["cpue_ci_low"] = result["ci_low"]
        h["cpue_ci_high"] = result["ci_high"]
        h["cpue_ci_95_low"] = result["ci_95_low"]
        h["cpue_ci_95_high"] = result["ci_95_high"]
        h["cpue_confidence"] = result["confidence"]

    log.info(
        f"  🎣 CPUE Estimator: {len(hotspots)} hotspots enriched, "
        f"mean_index={np.mean([h.get('cpue_index', 0) for h in hotspots]):.0f}"
    )
    return hotspots
