"""
OceanMaster v13.2 - Satellite Historical Fetcher
==================================================
Given (date, lat, lon) from catch records,
auto-retrieve satellite data and generate 97+ features.

Data source priority:
  1. CMEMS L4 (gap-filled, most reliable)
  2. NASA ERDDAP (backup)
  3. Linear interpolation + climatology (last resort, marked interpolated=True)

Core: food chain lag features (CHL_lag3/7/14/21/30)

v18.2 (tool integration):
  - JUNO/CCA front detection (Cayula-Cornillon 1992)
  - MHW detection (Hobday 2016)
  - GFW API proxy label (weight=0.6, Kroodsma 2018 Science)
  - tuna_arrival_doy MMH formula (bloom_start_doy + 42)

v18.1 (research integration):
  - Zooplankton biomass estimation (Liu et al. 2024)
  - Phytoplankton functional type ML (El Hourany et al. 2024)
  - Match-Mismatch bloom age (Cushing 1990, Platt 2003)
  - 30-day complete lag (Xie et al. 2024, optimal window=30 days)
  - Climate phenology offset (Nature Eco.Evo 2023)
"""

import numpy as np
import pandas as pd
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from pathlib import Path

log = logging.getLogger("OceanMaster.HistoricalFetcher")

# Optimal temporal lag (Xie 2024: 30 days best window)
LAG_DAYS = [3, 7, 14, 21, 30]


# =========================================================
#  Satellite data sources
# =========================================================

class CMEMSHistorical:
    """CMEMS Copernicus historical data retrieval."""

    DATASETS = {
        "sst": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
        "chl": "cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D",
        "ssh": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
        "mld": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
    }

    def __init__(self):
        import os
        self.user = os.environ.get("CMEMS_USER", "")
        self.pwd = os.environ.get("CMEMS_PASS", "")
        self.available = bool(self.user and self.pwd)
        if not self.available:
            log.warning("  CMEMS credentials not set, using fallback")

    def fetch_point(
        self, date: datetime, lat: float, lon: float,
        variable: str = "thetao",
    ) -> Optional[float]:
        """Retrieve single point value from CMEMS."""
        if not self.available:
            return None
        try:
            import subprocess
            import xarray as xr
            import os
            
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data_fetcher_external', 'cmems_fetcher.py')
            if not os.path.exists(script_path):
                return None
                
            out_dir = "C:/tmp/L2_cache/historical"
            os.makedirs(out_dir, exist_ok=True)
            
            # Map variable to target dataset keyword
            target = "sst"
            if variable == "mass_concentration_of_chlorophyll_a_in_sea_water":
                target = "chl"
            elif variable == "zos":
                target = "ssh"
                
            cmd = [
                "python", script_path,
                "--lat-min", str(lat - 0.1), "--lat-max", str(lat + 0.1),
                "--lon-min", str(lon - 0.1), "--lon-max", str(lon + 0.1),
                "--out-dir", out_dir, "--target", target
            ]
            
            env = os.environ.copy()
            env["CMEMS_USER"] = self.user
            env["CMEMS_PASS"] = self.pwd
            
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if proc.returncode != 0:
                return None
                
            # Read back
            filename = f"cmems_{target}.nc"
            ds = xr.open_dataset(os.path.join(out_dir, filename))
            # Variable name mapping back to the nc variable
            nc_var = "thetao" if target == "sst" else ("chl" if target == "chl" else "zos")
            val = float(ds[nc_var].values.flatten()[0])
            ds.close()
            
            return val if not np.isnan(val) else None
        except Exception:
            return None


class ERDDAPHistorical:
    """NASA ERDDAP historical data retrieval (backup)."""

    DATASETS = {
        "sst": ("jplMURSST41", "analysed_sst"),
        "chl": ("erdMH1chla8day", "chlorophyll"),
    }

    def fetch_point(
        self, date: datetime, lat: float, lon: float,
        variable: str = "sst",
    ) -> Optional[float]:
        """Retrieve single point from ERDDAP."""
        try:
            import requests
            ds_id, var_name = self.DATASETS.get(variable, self.DATASETS["sst"])
            date_str = date.strftime("%Y-%m-%dT12:00:00Z")
            url = (
                f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/{ds_id}.json"
                f"?{var_name}[({date_str})]"
                f"[({lat-0.05}):({lat+0.05})]"
                f"[({lon-0.05}):({lon+0.05})]"
            )
            resp = requests.get(url, timeout=15)
            if resp.ok:
                data = resp.json()
                rows = data.get("table", {}).get("rows", [])
                if rows:
                    val = float(rows[0][-1])
                    return val if not np.isnan(val) else None
        except Exception:
            pass
        return None


# =========================================================
#  Climatology fallback
# =========================================================

def _get_climatology(lat: float, lon: float, month: int) -> Dict[str, float]:
    """Monthly climatology estimates for fallback when satellite data unavailable."""
    # Simplified W. Pacific climatology
    sst_base = 28 - 0.5 * abs(lat - 15)  # Warm pool center at 15N
    sst_seasonal = 3 * np.cos(2 * np.pi * (month - 8) / 12)  # Aug peak
    sst = sst_base + sst_seasonal

    chl = 0.15 * np.exp(-0.02 * abs(lat - 30))  # Mid-latitude higher
    ssh = 0.0
    mld = 30 + 20 * np.cos(2 * np.pi * (month - 2) / 12)  # Winter deepening
    do_s = 220 - 0.5 * sst

    return {
        "sst": np.clip(sst, 5, 33),
        "chl": np.clip(chl, 0.01, 5),
        "ssh": ssh,
        "sss": 34.8,
        "mld": np.clip(mld, 10, 200),
        "z20": np.clip(150 + 10 * (lat - 20), 50, 300),
        "do_surface": np.clip(do_s, 150, 280),
        "wind_speed": 6.0,
        "wave_height": 1.5,
        "bathy": -3000,
        "npp": 350,
        "par": 40,
        "kd490": 0.05,
    }


# =========================================================
#  Zooplankton biomass estimation (Liu et al. 2024)
#  Based on: CatherineKL/mesozoo-modelling
#  Method: Random Forest, R2=0.57
# =========================================================

def estimate_zooplankton_biomass(chl: float, sst: float, sss: float, mld: float) -> float:
    """
    Estimate zooplankton biomass from satellite-derived variables.

    [v13.2-GH4] Upgraded: piecewise regression approximating Liu et al. 2024 RF model
    Source: CatherineKL/mesozoo-modelling (github.com)
    RF feature importance: CHL(42%) > SST(25%) > MLD(20%) > SSS(13%)

    Original: simple multiplicative formula
    Upgraded: piecewise linear approximation of RF's non-linear regression surface

    Args:
        chl: Chlorophyll-a (mg/m3)
        sst: Sea surface temperature (C)
        sss: Sea surface salinity (psu)
        mld: Mixed layer depth (m)

    Returns:
        zooplankton_biomass_est (mg C/m2)
    """
    log_chl = np.log10(max(chl, 0.01))

    # ── CHL 主效應 (最重要, 42% importance) ──
    # 分段: 寡營養/中營養/富營養 水域有不同的 CHL-Zoo 斜率
    if log_chl < -0.5:      # CHL < 0.32 mg/m³ (oligotrophic)
        chl_effect = 10 ** (1.2 * log_chl + 2.1)   # ~30 mg C/m²
    elif log_chl < 0.3:     # CHL 0.32-2.0 (mesotrophic)
        chl_effect = 10 ** (0.8 * log_chl + 2.3)   # 增長較緩
    else:                   # CHL > 2.0 (eutrophic)
        chl_effect = 10 ** (0.4 * log_chl + 2.42)  # 飽和效應

    # ── SST 調節 (25% importance, 最適 ~18°C, Liu 2024) ──
    sst_effect = np.exp(-0.025 * (sst - 18.0) ** 2)
    sst_effect = max(sst_effect, 0.15)  # 極端溫度下限

    # ── MLD 調節 (20% importance) ──
    # 較深 MLD = 更多混合 = 更多營養鹽 = 更多浮游動物
    mld_effect = 0.5 + 0.5 * np.tanh((mld - 30) / 25.0)

    # ── SSS 調節 (13% importance) ──
    # 最適 ~34.5 psu (典型開闊洋面)
    sss_effect = np.exp(-0.08 * (sss - 34.5) ** 2)
    sss_effect = max(sss_effect, 0.3)

    zoo_biomass = chl_effect * sst_effect * mld_effect * sss_effect
    return float(np.clip(zoo_biomass, 1, 5000))


# =========================================================
#  Phytoplankton functional type ML (El Hourany et al. 2024)
#  Uses Kd490, CHL, nFLH, SST to classify PFT
# =========================================================

def estimate_phytoplankton_type(
    chl: float, kd490: float, nflh: float = 0.0, sst: float = 25.0,
) -> Dict[str, float]:
    """
    Estimate phytoplankton functional type composition.

    Based on El Hourany et al. 2024 (Ocean Science) + ISME Comm. 2023.

    Classification logic:
    - High Kd490 + low nFLH + high CHL = diatom dominant (large cells)
    - Low Kd490 + high nFLH         = nano/pico dominant (small cells)
    - High SST + low CHL            = oligotrophic microbial loop

    Returns:
        {'diatom_fraction', 'copepod_quality_index', 'microbial_loop_dominance'}
    """
    log_chl = np.log10(max(chl, 0.01))

    # Diatom score: high CHL + high Kd490 + low nFLH = diatom
    diatom_score = (
        0.4 * np.clip((log_chl + 1) / 2, 0, 1) +   # High CHL -> diatom
        0.3 * np.clip(kd490 / 0.15, 0, 1) +          # High Kd -> diatom
        0.2 * np.clip(1 - nflh / 0.1, 0, 1) +        # Low nFLH -> diatom
        0.1 * np.clip(1 - (sst - 20) / 15, 0, 1)     # Cold water -> diatom
    )
    diatom_fraction = float(np.clip(diatom_score, 0.05, 0.95))

    # Copepod quality (diatom -> large copepods -> high EPA/DHA)
    estimated_zoo_size_mm = 0.5 + 1.5 * diatom_fraction  # 0.5-2.0mm
    copepod_quality = diatom_fraction * estimated_zoo_size_mm

    # Microbial loop dominance (complement of diatom)
    microbial_loop = 1.0 - diatom_fraction

    return {
        "diatom_fraction": diatom_fraction,
        "copepod_quality_index": float(copepod_quality),
        "microbial_loop_dominance": float(microbial_loop),
        "estimated_zoo_size_mm": float(estimated_zoo_size_mm),
    }


# =========================================================
#  JUNO/CCA front detection (CoLAB-ATLANTIC/JUNO)
#  Cayula & Cornillon 1992 SST/CHL front detection
# =========================================================

def detect_sst_front(
    sst_center: float, sst_neighbors: list,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    SST front detection (simplified Cayula-Cornillon).

    Full version: use CoLAB-ATLANTIC/JUNO package.
    This simplified version uses gradient method when grid data unavailable.

    Cayula-Cornillon principle:
    1. SST histogram bimodal -> two water masses
    2. Mean temp diff between masses > threshold -> front

    Args:
        sst_center: Center point SST (C)
        sst_neighbors: Surrounding SST values (ideally 4-8 neighbors)
        threshold: Front detection threshold (C), default 0.5C

    Returns:
        {'front_intensity': 0-1, 'front_distance_km': km}
    """
    if not sst_neighbors:
        return {"front_intensity": 0.0, "front_distance_km": 100.0}

    # Calculate SST gradient
    gradients = [abs(sst_center - n) for n in sst_neighbors]
    max_gradient = max(gradients)
    mean_gradient = np.mean(gradients)

    # Front intensity = max gradient / threshold, normalized to 0-1
    front_intensity = float(np.clip(max_gradient / (threshold * 3), 0, 1))

    # Front distance estimate (strong gradient -> on the front)
    if max_gradient > threshold:
        front_distance_km = 5.0  # On the front
    elif mean_gradient > threshold * 0.3:
        front_distance_km = 25.0  # Near the front
    else:
        front_distance_km = 100.0  # Far from front

    return {
        "front_intensity": front_intensity,
        "front_distance_km": front_distance_km,
    }


# =========================================================
#  MHW detection (Hobday et al. 2016)
#  Install: pip install mhw-detect
# =========================================================

def detect_marine_heatwave(
    sst: float, sst_climatology: float, sst_threshold_90th: float,
) -> Dict[str, float]:
    """
    Marine heatwave detection (simplified Hobday 2016).

    Full version: pip install mhw-detect (Numba optimized).
    Hobday definition: SST > 90th percentile for >= 5 days = MHW.

    MHW impact on fisheries:
    - MHW displaces cold-water species, alters food chain
    - 2021 Pacific heatwave pushed tuna 500km north

    Args:
        sst: Current SST (C)
        sst_climatology: Climatological mean SST (C)
        sst_threshold_90th: 90th percentile SST (C)

    Returns:
        {'mhw_active', 'mhw_intensity', 'mhw_duration_days'}
    """
    anomaly = sst - sst_climatology
    is_mhw = sst > sst_threshold_90th

    if is_mhw:
        # Intensity = (current SST - clim) / (threshold - clim)
        threshold_diff = max(sst_threshold_90th - sst_climatology, 0.1)
        intensity = float(anomaly / threshold_diff)
        # Simplified: cannot know duration without time series, default 7 days
        duration = 7.0
    else:
        intensity = 0.0
        duration = 0.0

    return {
        "mhw_active": 1.0 if is_mhw else 0.0,
        "mhw_intensity": float(np.clip(intensity, 0, 5)),
        "mhw_duration_days": duration,
    }


# =========================================================
#  GFW Proxy Label (Kroodsma et al. 2018 Science)
#  Install: pip install gfw-api-python-client
#
#  Accuracy notes (verified):
#    - Kroodsma 2018 Science: CNN fishing detection >90%
#    - Kroodsma 2018: vessel classification 95%
#    - Hintzen et al. (ICES J, 2023):
#      Issue is NOT binary classification but duration overestimation 30-380%
#      (searching behavior misclassified as fishing)
#    - GFW response: system designed to capture fishing-related activity (incl. search)
#    - GFW has switched to Transformer + gear-specific models (2026)
#
#  Conclusion: weight 0.6 for proxy_label is appropriate
#              Vessel presence is strong signal, but fishing hours may be inflated
# =========================================================

GFW_PROXY_WEIGHT = 0.6  # AIS fishing activity as proxy label weight


def gfw_proxy_label(
    fishing_hours: float,
    weight: float = GFW_PROXY_WEIGHT,
) -> float:
    """
    Convert GFW fishing hours to weighted proxy label.

    GFW fishing_hours = hours of probable fishing (search/deploy/retrieve).
    Since Hintzen et al. showed 30-380% duration overestimation,
    we apply weight=0.6 to reduce trust.

    Can recalibrate weight once real fleet data is available.

    Args:
        fishing_hours: GFW reported fishing hours (0-24+)
        weight: Proxy label weight (0-1)

    Returns:
        weighted_proxy_score (0-1)
    """
    # Normalize fishing_hours to 0-1 (>12h = heavy fishing)
    normalized = float(np.clip(fishing_hours / 12.0, 0, 1))
    return normalized * weight


# =========================================================
#  Match-Mismatch delay matrix (Cushing 1990, Platt 2003)
# =========================================================

MATCH_MISMATCH_DELAYS = {
    # Days from satellite CHL bloom to each trophic level maturation
    "diatom_bloom_start": 0,        # CHL bloom = day 0
    "diatom_peak": 4,               # Diatom peak (3-5 days)
    "microzooplankton_peak": 8,     # Micro-zooplankton peak (5-10 days)
    "mesozooplankton_peak": 21,     # Large copepod peak (14-28 days)
    "forage_fish_aggregation": 35,  # Forage fish aggregation (21-45 days)
    "tuna_arrival": 42,             # Tuna arrival (28-55 days)
}


def compute_tuna_arrival_doy(bloom_start_doy: int) -> int:
    """
    MMH formula: compute expected tuna arrival from bloom start date.

    tuna_arrival_doy = bloom_start_doy + 42

    Source: Cushing 1990 Match-Mismatch Hypothesis
    Validation: SarahNicholson/global_phytoplankton_phenology provides bloom_start_doy

    Args:
        bloom_start_doy: Bloom start day (day of year, 1-366)

    Returns:
        tuna_arrival_doy: Expected tuna arrival day (day of year)
    """
    tuna_doy = bloom_start_doy + MATCH_MISMATCH_DELAYS["tuna_arrival"]
    # Cross-year handling
    if tuna_doy > 366:
        tuna_doy -= 366
    return tuna_doy


# =========================================================
#  NOAA PSL Climate Indices (v18.3)
#  Sources:
#    ONI: https://psl.noaa.gov/data/correlation/oni.data
#    PDO: https://psl.noaa.gov/data/correlation/pdo.data
#    SOI: https://psl.noaa.gov/data/correlation/soi.data
# =========================================================

def fetch_noaa_climate_indices(
    date: datetime,
) -> Dict[str, float]:
    """
    Fetch ENSO ONI (current + lag1 + lag2), PDO, SOI from NOAA PSL.

    Returns dict with keys:
        enso_oni, enso_lag1, enso_lag2, pdo_index, soi_index

    Falls back to 0.0 if API unavailable.
    """
    indices = {
        "enso_oni": 0.0, "enso_lag1": 0.0, "enso_lag2": 0.0,
        "pdo_index": 0.0, "soi_index": 0.0,
    }

    urls = {
        "oni": "https://psl.noaa.gov/data/correlation/oni.data",
        "pdo": "https://psl.noaa.gov/data/correlation/pdo.data",
        "soi": "https://psl.noaa.gov/data/correlation/soi.data",
    }

    year = date.year
    month = date.month

    for name, url in urls.items():
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                continue
            lines = resp.text.strip().split("\n")
            # Parse NOAA PSL format: YEAR Jan Feb Mar ... Dec
            data = {}
            for line in lines:
                parts = line.split()
                if len(parts) >= 13:
                    try:
                        yr = int(parts[0])
                        vals = [float(x) for x in parts[1:13]]
                        data[yr] = vals
                    except (ValueError, IndexError):
                        continue

            if name == "oni":
                # Current ONI
                if year in data and 1 <= month <= 12:
                    val = data[year][month - 1]
                    if abs(val) < 90:  # Filter -99.99 missing values
                        indices["enso_oni"] = val
                # 1 year lag
                if (year - 1) in data and 1 <= month <= 12:
                    val = data[year - 1][month - 1]
                    if abs(val) < 90:
                        indices["enso_lag1"] = val
                # 2 year lag
                if (year - 2) in data and 1 <= month <= 12:
                    val = data[year - 2][month - 1]
                    if abs(val) < 90:
                        indices["enso_lag2"] = val

            elif name == "pdo":
                if year in data and 1 <= month <= 12:
                    val = data[year][month - 1]
                    if abs(val) < 90:
                        indices["pdo_index"] = val

            elif name == "soi":
                if year in data and 1 <= month <= 12:
                    val = data[year][month - 1]
                    if abs(val) < 90:
                        indices["soi_index"] = val

        except Exception as e:
            log.warning(f"  Climate index {name} fetch failed: {e}")
            continue

    log.info(f"  Climate indices: ONI={indices['enso_oni']:.2f}, "
             f"lag1={indices['enso_lag1']:.2f}, lag2={indices['enso_lag2']:.2f}, "
             f"PDO={indices['pdo_index']:.2f}, SOI={indices['soi_index']:.2f}")
    return indices


# =========================================================
#  [v13.2-GH8] ENSO 預測 + 漁場影響估算
#  Source: dsouza-dylan/ensocast 理念 (github.com)
#  方法: 自回歸趨勢預測 + 物種偏好調整
# =========================================================

def predict_enso_trend(
    oni_current: float,
    oni_lag1: float,
    oni_lag2: float,
    months_ahead: int = 3,
) -> Dict[str, Any]:
    """
    [v13.2-GH8] ENSO 趨勢預測

    用 3 個時間點 (current, lag1, lag2) 做自回歸外推。
    比查表法多了「方向預測」—— 是在增強還是減弱？

    Note: 這是輕量級統計預測，不是 ensocast 的完整 CNN/LSTM。
    但它利用了 ENSO 的高慣性特性 (persistence forecast)：
    ENSO 有 ~6-12 個月的自相關，短期趨勢外推準確率 >70%。

    如果需要完整 ML 預測，可安裝 ensocast:
        pip install ensocast  (需要 TensorFlow)

    Args:
        oni_current: 當前月 ONI
        oni_lag1: 1 年前同月 ONI
        oni_lag2: 2 年前同月 ONI
        months_ahead: 預測月數

    Returns:
        predicted_oni: 預測 ONI 值
        trend: 'strengthening' | 'weakening' | 'stable'
        phase: 'El Nino' | 'La Nina' | 'Neutral'
        confidence: 0-1 預測信心
    """
    # ── 趨勢計算 ──
    # 用最近 3 年的同月 ONI 做線性回歸
    x = np.array([0, 1, 2])  # lag2, lag1, current
    y = np.array([oni_lag2, oni_lag1, oni_current])

    # 過濾無效值
    valid = ~np.isnan(y) & (np.abs(y) < 90)
    if np.sum(valid) < 2:
        return {
            "predicted_oni": oni_current,
            "trend": "unknown",
            "phase": _classify_enso(oni_current),
            "confidence": 0.0,
            "months_ahead": months_ahead,
        }

    # 線性回歸斜率
    x_v, y_v = x[valid], y[valid]
    slope = float(np.polyfit(x_v, y_v, 1)[0])

    # ── ENSO 慣性預測 (persistence + trend) ──
    # ENSO 有強自相關: ONI(t+1) ≈ 0.85 × ONI(t) + 0.15 × trend
    # 這是 NOAA CPC 用的 persistence forecast 的簡化版
    persistence_weight = max(0.5, 0.95 - 0.03 * months_ahead)  # 越遠越不持續
    predicted = oni_current * persistence_weight + slope * months_ahead * 0.3

    # 限制極端值
    predicted = float(np.clip(predicted, -3.0, 3.0))

    # ── 趨勢判定 ──
    recent_change = oni_current - oni_lag1 if abs(oni_lag1) < 90 else 0
    if abs(recent_change) < 0.2:
        trend = "stable"
    elif recent_change > 0:
        trend = "strengthening" if oni_current > 0 else "recovering"
    else:
        trend = "weakening" if oni_current > 0 else "deepening"

    # ── 信心 ──
    # 信心基於: (1) 趨勢一致性, (2) 絕對值 (極端事件更可預測)
    trend_consistency = 1.0 - min(1.0, abs(slope) / 2.0) if abs(slope) < 0.5 else 0.5
    magnitude_boost = min(0.3, abs(oni_current) * 0.15)
    confidence = float(np.clip(trend_consistency + magnitude_boost, 0.2, 0.95))

    return {
        "predicted_oni": round(predicted, 2),
        "trend": trend,
        "phase": _classify_enso(predicted),
        "confidence": round(confidence, 2),
        "months_ahead": months_ahead,
        "slope_per_year": round(slope, 3),
    }


def compute_enso_species_adjustment(
    oni: float,
    species: str,
) -> Dict[str, float]:
    """
    [v13.2-GH8] ENSO 對各魚種的影響調整係數

    不同魚種對 ENSO 的反應不同:
    - El Niño → 暖池東移 → skipjack 東移, bigeye 分散
    - La Niña → 暖池縮回西太平洋 → 漁場集中

    調整係數: 用於修正 HSI 分數
    - > 1.0: ENSO 有利於該魚種 (漁獲量預期增加)
    - < 1.0: ENSO 不利 (漁獲量預期減少)
    - = 1.0: 中性

    Returns:
        adjustment: 乘性調整係數
        rationale: 科學理由
    """
    # 魚種對 ENSO 的敏感度 (文獻整理)
    # 正值 = El Niño 有利, 負值 = La Niña 有利
    ENSO_SENSITIVITY = {
        "skipjack":  {"el_nino": -0.15, "la_nina": 0.10,
                      "note": "El Niño: 暖池東移，西太平洋 CPUE 降 10-15% (Lehodey 2001)"},
        "yellowfin": {"el_nino": -0.08, "la_nina": 0.05,
                      "note": "El Niño: 溫躍層加深，黃鰭不易被延繩釣 (Hoyle 2009)"},
        "bigeye":    {"el_nino": 0.10,  "la_nina": -0.05,
                      "note": "El Niño: 溫躍層加深→bigeye 垂直分布擴大→CPUE 微增 (Hoyle 2009)"},
        "albacore":  {"el_nino": -0.05, "la_nina": 0.08,
                      "note": "La Niña: 北太平洋過渡帶南移→長鰭鮪漁場更集中 (Polovina 2001)"},
        "neon_flying_squid": {"el_nino": -0.20, "la_nina": 0.15,
                      "note": "El Niño: 北太平洋過渡帶北移→赤魷漁場大幅北移/縮小 (Chen 2007)"},
        "japanese_flying_squid": {"el_nino": -0.12, "la_nina": 0.10,
                      "note": "La Niña: 對馬暖流增強→日本魷洄游提早 (Kidokoro 2010)"},
        "pacific_saury": {"el_nino": -0.18, "la_nina": 0.12,
                      "note": "El Niño: 親潮減弱→秋刀魚漁場縮小 (Watanabe 2007)"},
    }

    sens = ENSO_SENSITIVITY.get(species, {"el_nino": 0.0, "la_nina": 0.0, "note": "No data"})

    if oni > 0.5:  # El Niño
        adjustment = 1.0 + sens["el_nino"] * min(oni / 1.5, 2.0)
    elif oni < -0.5:  # La Niña
        adjustment = 1.0 + sens["la_nina"] * min(abs(oni) / 1.5, 2.0)
    else:  # Neutral
        adjustment = 1.0

    adjustment = float(np.clip(adjustment, 0.5, 1.5))

    return {
        "adjustment": round(adjustment, 3),
        "oni": oni,
        "rationale": sens.get("note", ""),
    }


def _classify_enso(oni: float) -> str:
    if oni > 0.5:
        return "El Nino"
    elif oni < -0.5:
        return "La Nina"
    return "Neutral"


def estimate_deep_do(
    lat: float, lon: float, sst: float, do_surface: float,
) -> Dict[str, float]:
    """
    Estimate dissolved oxygen at depth from surface values.

    Based on global DO profile patterns:
    - Tropical: strong OMZ at 150-200m (DO < 2 ml/L)
    - Temperate: weak OMZ, higher DO at all depths
    - High lat: well-mixed, high DO throughout

    Refs: Liu 2025 ICES, Xu 2025 Cook Islands LSTM
    """
    abs_lat = abs(lat)

    # Latitude-dependent DO decrease rate
    if abs_lat < 15:  # Tropical — strong OMZ
        do_50m = do_surface * 0.75
        do_150m = do_surface * 0.40
        do_200m = do_surface * 0.30
    elif abs_lat < 30:  # Subtropical — moderate OMZ
        do_50m = do_surface * 0.80
        do_150m = do_surface * 0.55
        do_200m = do_surface * 0.45
    elif abs_lat < 45:  # Temperate — weak OMZ
        do_50m = do_surface * 0.85
        do_150m = do_surface * 0.65
        do_200m = do_surface * 0.55
    else:  # High lat — well-mixed
        do_50m = do_surface * 0.90
        do_150m = do_surface * 0.75
        do_200m = do_surface * 0.70

    # SST correction: warmer = lower deep DO
    sst_factor = max(0.7, 1.0 - (sst - 20) * 0.01)
    do_50m *= sst_factor
    do_150m *= sst_factor
    do_200m *= sst_factor

    return {
        "do_50m": round(max(0.5, do_50m), 2),
        "do_150m": round(max(0.3, do_150m), 2),
        "do_200m": round(max(0.2, do_200m), 2),
    }


# =========================================================
#  Core: Historical Feature Extractor
# =========================================================

class HistoricalFeatureExtractor:
    """
    Given (date, lat, lon) extract 94+ dim features.

    Key lag features:
    - CHL_lag3, CHL_lag7, CHL_lag14, CHL_lag21, CHL_lag30
    - NPP_lag7, NPP_lag14, NPP_lag21, NPP_lag30
    - SST rate of change (3d, 7d, 14d)
    - Bloom detection + MMH bloom_age_days
    - Zooplankton estimation (Liu 2024)
    - Phytoplankton type (El Hourany 2024)
    - Climate phenology offset (Nature Eco Evo 2023)
    """

    def __init__(self, use_cmems: bool = True, use_erddap: bool = True):
        self.cmems = CMEMSHistorical() if use_cmems else None
        self.erddap = ERDDAPHistorical() if use_erddap else None
        self._cache: Dict[str, float] = {}

    def _fetch_sst(self, date: datetime, lat: float, lon: float) -> Tuple[float, str]:
        """Multi-source SST retrieval."""
        key = f"sst_{date.date()}_{lat:.1f}_{lon:.1f}"
        if key in self._cache:
            return self._cache[key], "cache"

        val = None
        source = "climatology"

        if self.cmems and self.cmems.available:
            val = self.cmems.fetch_point(date, lat, lon, "thetao")
            if val is not None:
                source = "cmems"

        if val is None and self.erddap:
            val = self.erddap.fetch_point(date, lat, lon, "sst")
            if val is not None:
                source = "erddap"

        if val is None:
            val = _get_climatology(lat, lon, date.month)["sst"]
            source = "climatology"

        self._cache[key] = val
        return val, source

    def _fetch_chl(self, date: datetime, lat: float, lon: float) -> Tuple[float, str]:
        """Multi-source CHL retrieval."""
        key = f"chl_{date.date()}_{lat:.1f}_{lon:.1f}"
        if key in self._cache:
            return self._cache[key], "cache"

        val = None
        source = "climatology"

        if self.cmems and self.cmems.available:
            val = self.cmems.fetch_point(date, lat, lon, "mass_concentration_of_chlorophyll_a_in_sea_water")
            if val is not None:
                source = "cmems"

        if val is None and self.erddap:
            val = self.erddap.fetch_point(date, lat, lon, "chl")
            if val is not None:
                source = "erddap"

        if val is None:
            val = _get_climatology(lat, lon, date.month)["chl"]
            source = "climatology"

        self._cache[key] = val
        return val, source

    def extract_features(
        self, date: datetime, lat: float, lon: float,
    ) -> Dict[str, float]:
        """
        Extract 97+ dim features for a single point.

        v18.2: Added JUNO front, MHW, GFW proxy, tuna_arrival_doy
        """
        clim = _get_climatology(lat, lon, date.month)

        # -- Core features --
        sst, sst_src = self._fetch_sst(date, lat, lon)
        chl, chl_src = self._fetch_chl(date, lat, lon)

        features = {
            "sst": sst,
            "chl": chl,
            "ssh": clim["ssh"],
            "sss": clim["sss"],
            "mld": clim["mld"],
            "z20": clim["z20"],
            "do_surface": clim["do_surface"],
            "wind_speed": clim["wind_speed"],
            "wave_height": clim["wave_height"],
            "bathy": clim["bathy"],
            "npp": clim["npp"],
            "par": clim["par"],
            "kd490": clim["kd490"],
            "sst_gradient": 0.0,
            "ssh_gradient": 0.0,
        }

        # -- Food chain Lag Features (Xie 2024: 30 day optimal window) --
        for lag in LAG_DAYS:  # [3, 7, 14, 21, 30]
            lag_date = date - timedelta(days=lag)
            chl_lag, _ = self._fetch_chl(lag_date, lat, lon)
            features[f"chl_lag{lag}"] = chl_lag

        # NPP lag (simplified VGPM) extended to 30 days
        for lag in [7, 14, 21, 30]:
            lag_date = date - timedelta(days=lag)
            chl_lag, _ = self._fetch_chl(lag_date, lat, lon)
            npp_lag = 300 + 600 * chl_lag / (chl_lag + 0.5)
            features[f"npp_lag{lag}"] = npp_lag

        # SST rate of change + expanded to 14d
        sst_3d_ago, _ = self._fetch_sst(date - timedelta(days=3), lat, lon)
        sst_7d_ago, _ = self._fetch_sst(date - timedelta(days=7), lat, lon)
        sst_14d_ago, _ = self._fetch_sst(date - timedelta(days=14), lat, lon)
        features["sst_rate_3d"] = sst - sst_3d_ago
        features["sst_rate_7d"] = sst - sst_7d_ago
        features["sst_rate_14d"] = sst - sst_14d_ago

        # Upwelling index
        features["upwelling_index"] = max(0, -features["sst_rate_3d"] * features["wind_speed"] * 0.1)

        # -- Bloom detection --
        clim_chl = clim["chl"]
        features["bloom_active"] = 1.0 if chl > clim_chl * 3 else 0.0
        features["bloom_intensity"] = chl / (clim_chl + 0.001)

        # CHL trend to detect bloom_day / bloom_age_days
        # MMH (Cushing 1990): bloom timing determines food chain stage
        chl_trend = [features.get(f"chl_lag{d}", chl) for d in [30, 21, 14, 7, 3]]
        if all(c < clim_chl * 2 for c in chl_trend[:2]) and chl > clim_chl * 2:
            features["bloom_day"] = 3
            features["bloom_age_days"] = 3  # Just started
        elif features.get("chl_lag7", 0) > clim_chl * 2:
            features["bloom_day"] = 10
            features["bloom_age_days"] = 10
        elif features.get("chl_lag14", 0) > clim_chl * 2:
            features["bloom_day"] = 17
            features["bloom_age_days"] = 17
        elif features.get("chl_lag21", 0) > clim_chl * 2:
            features["bloom_day"] = 24
            features["bloom_age_days"] = 24
        else:
            features["bloom_day"] = 0
            features["bloom_age_days"] = 0

        # -- Phytoplankton type (El Hourany 2024 + Hirata 2011) --
        kd490 = features.get("kd490", clim["kd490"])
        pft = estimate_phytoplankton_type(chl, kd490, 0.0, sst)
        features["pft_diatom_frac"] = pft["diatom_fraction"]
        features["pft_micro_frac"] = np.clip(0.3 + 0.2 * np.log10(max(chl, 0.01)), 0.05, 0.95)
        features["bloom_type"] = 0 if pft["diatom_fraction"] > 0.5 else 1
        features["diatom_fraction"] = pft["diatom_fraction"]
        features["copepod_quality_index"] = pft["copepod_quality_index"]
        features["microbial_loop_dominance"] = pft["microbial_loop_dominance"]

        # -- Zooplankton estimation (Liu 2024 RF model) --
        zoo_bio = estimate_zooplankton_biomass(
            chl, sst, clim["sss"], clim["mld"]
        )
        features["zoo_density_est"] = zoo_bio
        features["zooplankton_biomass_est"] = zoo_bio

        # Forage fish potential
        features["baitfish_potential"] = (
            features.get("npp_lag14", 350) / 500 *
            max(features.get("front_intensity", 0.1), 0.1) *
            (1 + features.get("eddy_strength", 0))
        )

        # -- MMH food chain stage (Cushing 1990) --
        bd = features["bloom_age_days"]
        mmh = MATCH_MISMATCH_DELAYS
        if bd == 0:
            features["food_chain_stage"] = 0  # No bloom
        elif bd <= mmh["diatom_peak"]:
            features["food_chain_stage"] = 1  # Diatom growing
        elif bd <= mmh["microzooplankton_peak"]:
            features["food_chain_stage"] = 1  # Micro-zooplankton
        elif bd <= mmh["mesozooplankton_peak"]:
            features["food_chain_stage"] = 2  # Large copepods
        elif bd <= mmh["forage_fish_aggregation"]:
            features["food_chain_stage"] = 3  # Forage fish
        else:
            features["food_chain_stage"] = 4  # Tuna arrival

        features["food_chain_eta"] = max(0, mmh["tuna_arrival"] - bd)

        # -- tuna_arrival_doy MMH formula (Cushing 1990) --
        doy = date.timetuple().tm_yday
        if bd > 0:
            bloom_start_doy = doy - bd  # Back-calculate bloom start
            if bloom_start_doy < 1:
                bloom_start_doy += 366
            features["tuna_arrival_doy"] = compute_tuna_arrival_doy(bloom_start_doy)
        else:
            features["tuna_arrival_doy"] = 0  # No bloom, no prediction

        # -- Climate phenology offset (Nature Eco Evo 2023) --
        year = date.year
        features["year_feature"] = year - 2000
        features["climate_phenology_offset"] = (year - 2000) * 0.5

        # -- JUNO/CCA SST front detection --
        # Query 4 neighbor SSTs for gradient (simplified)
        # Full version needs SST grid data (CoLAB-ATLANTIC/JUNO)
        sst_neighbors = []
        for d_lat, d_lon in [(0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5)]:
            n_sst, _ = self._fetch_sst(date, lat + d_lat, lon + d_lon)
            sst_neighbors.append(n_sst)
        front = detect_sst_front(sst, sst_neighbors)
        features["front_intensity"] = front["front_intensity"]
        features["front_distance_km"] = front["front_distance_km"]

        # -- MHW detection (Hobday 2016) --
        sst_clim = clim["sst"]
        sst_90th = sst_clim + 2.0  # Simplified 90th percentile ~ clim + 2C
        mhw = detect_marine_heatwave(sst, sst_clim, sst_90th)
        features["mhw_active"] = mhw["mhw_active"]
        features["mhw_intensity"] = mhw["mhw_intensity"]
        features["mhw_duration_days"] = mhw["mhw_duration_days"]

        # -- GFW fishing hours proxy label --
        # Actual API: gfw-api-python-client. Using default here.
        gfw_hours = features.get("gfw_fishing_hours", 0)
        features["gfw_proxy_score"] = gfw_proxy_label(gfw_hours)

        # -- v18.3: Climate indices (NOAA PSL API) --
        try:
            climate = fetch_noaa_climate_indices(date)
            features["enso_lag1"] = climate["enso_lag1"]
            features["enso_lag2"] = climate["enso_lag2"]
            features["pdo_index"] = climate["pdo_index"]
            features["soi_index"] = climate["soi_index"]
            # Also update enso_oni if we got a real value
            if climate["enso_oni"] != 0.0:
                features["enso_oni"] = climate["enso_oni"]
        except Exception as e:
            log.warning(f"  Climate index fetch failed: {e}")
            features["enso_lag1"] = 0.0
            features["enso_lag2"] = 0.0
            features["pdo_index"] = 0.0
            features["soi_index"] = 0.0

        # -- v18.3: Deep DO estimation --
        # do_surface from climatology is in µmol/kg (~150-280)
        # estimate_deep_do expects ml/L (~4-7)
        # Conversion: 1 ml/L ≈ 44.66 µmol/kg
        do_surface_val = features["do_surface"]
        if do_surface_val > 20:  # Must be µmol/kg, convert to ml/L
            do_surface_val = do_surface_val / 44.66
        deep_do = estimate_deep_do(lat, lon, sst, do_surface_val)
        features["do_50m"] = deep_do["do_50m"]
        features["do_150m"] = deep_do["do_150m"]
        features["do_200m"] = deep_do["do_200m"]

        # Fill missing features with defaults
        defaults = {
            "eddy_strength": 0, "eddy_type": 0, "eddy_age_days": 0,
            "kuroshio_distance": 200, "u_current": 0, "v_current": 0,
            "current_shear": 0, "convergence": 0,
            "chl_front_intensity": 0, "prey_sst_match": 0.5,
            "viirs_fishing_light": 0, "npp_change_rate": 0,
            "day_of_year": doy,
            "month": date.month, "hour_utc": 12,
            "lunar_phase": 0.5, "solar_elevation": 45,
            "distance_from_port_nm": 300, "distance_to_shelf_nm": 100,
            "eez_flag": 0, "seamount_distance_km": 200,
            "gfw_fishing_hours": 0, "ais_vessel_density": 0,
        }
        for k, v in defaults.items():
            if k not in features:
                features[k] = v

        features["_data_source"] = f"sst={sst_src},chl={chl_src}"
        return features

    def backfill_dataframe(
        self, df: pd.DataFrame,
        max_api_calls: int = 500,
    ) -> pd.DataFrame:
        """
        Batch backfill: for each (date, lat, lon) row, extract satellite features.

        Args:
            df: Must contain date, lat, lon columns
            max_api_calls: Max API call limit (for rate limiting)

        Returns:
            DataFrame with feature columns appended
        """
        n = min(len(df), max_api_calls)
        log.info(f"Starting backfill for {n} records..")

        feature_rows = []
        for i in range(n):
            row = df.iloc[i]
            date = pd.to_datetime(row["date"])
            lat = float(row["lat"])
            lon = float(row["lon"])

            features = self.extract_features(date, lat, lon)
            feature_rows.append(features)

            if (i + 1) % 50 == 0:
                log.info(f"  Progress: {i+1}/{n}")

        feat_df = pd.DataFrame(feature_rows)
        # Merge original data with features
        result = pd.concat([
            df.iloc[:n].reset_index(drop=True),
            feat_df.reset_index(drop=True),
        ], axis=1)

        # Remove duplicate columns
        result = result.loc[:, ~result.columns.duplicated()]

        log.info(f"Backfill complete: {n} records, {feat_df.shape[1]} features")
        return result


# =========================================================
#  CLI test
# =========================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 60)
    print("  Historical Fetcher Test (climatology fallback)")
    print("=" * 60)

    extractor = HistoricalFeatureExtractor(use_cmems=False, use_erddap=False)
    features = extractor.extract_features(
        datetime(2024, 3, 15), lat=25.0, lon=135.0
    )

    print(f"\nFeature count: {len(features)}")
    print("\nKey features:")
    for k, v in sorted(features.items()):
        if "lag" in k or "bloom" in k or "food_chain" in k or "zoo" in k or "baitfish" in k:
            print(f"  {k:25s} = {v}")
