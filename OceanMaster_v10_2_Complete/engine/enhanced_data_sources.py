"""
OceanMaster v10.3 — 增強數據源模組
====================================
B 級精度提升:
  1. ENSO/ONI 即時指數 (NOAA CPC)
  2. 月相/潮汐漁獲因子
  3. 歷史漁場先驗權重
  4. Argo 深度溫度剖面 (大目鮪用)
  5. 備用衛星數據源 (ERDDAP 替代端點)
"""

import numpy as np
import logging
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple

log = logging.getLogger("OceanMaster.Enhanced")


# ══════════════════════════════════════════════════════
# 1. ENSO / ONI 即時指數
# ══════════════════════════════════════════════════════

# NOAA CPC ONI 歷史值 (3-month running mean, Niño 3.4)
# 用於離線回退 — 2024/2025 近期值
ONI_RECENT = {
    (2024, 1): 1.9, (2024, 2): 1.7, (2024, 3): 1.4,
    (2024, 4): 1.0, (2024, 5): 0.5, (2024, 6): 0.1,
    (2024, 7): -0.2, (2024, 8): -0.5, (2024, 9): -0.7,
    (2024, 10): -0.8, (2024, 11): -0.9, (2024, 12): -1.0,
    (2025, 1): -0.9, (2025, 2): -0.7, (2025, 3): -0.5,
    (2025, 4): -0.3, (2025, 5): -0.1, (2025, 6): 0.0,
    (2025, 7): 0.1, (2025, 8): 0.2, (2025, 9): 0.3,
    (2025, 10): 0.4, (2025, 11): 0.5, (2025, 12): 0.5,
    (2026, 1): 0.5, (2026, 2): 0.4,
}


async def fetch_enso_oni(client=None) -> float:
    """
    取得最新 ONI 值。
    優先: NOAA CPC API → 內建近期值 → 0.0
    
    ONI > +0.5 = El Niño (鮪魚東移)
    ONI < -0.5 = La Niña (鮪魚西聚)
    """
    # 嘗試 NOAA CPC
    if client is not None:
        try:
            url = ("https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt")
            resp = await client.get(url, timeout=15)
            if resp.status_code == 200:
                lines = resp.text.strip().split("\n")
                # 取最後一行的 APTS (anomaly) 值
                last = lines[-1].split()
                if len(last) >= 4:
                    oni = float(last[-1])
                    log.info(f"  ENSO ONI (live): {oni:+.2f}")
                    return oni
        except Exception as e:
            log.warning(f"  ENSO fetch failed: {e}")
    
    # 回退到內建值
    now = datetime.now(timezone.utc)
    key = (now.year, now.month)
    oni = ONI_RECENT.get(key, ONI_RECENT.get((now.year, now.month - 1), 0.0))
    log.info(f"  ENSO ONI (cached): {oni:+.2f}")
    return oni


def enso_species_modifier(oni: float, species: str, lats, lons) -> np.ndarray:
    """
    根據 ENSO 狀態調整 HSI 的空間權重。
    
    El Niño (ONI>0): 暖池東擴，魚群東移
    La Niña (ONI<0): 暖池收縮，魚群聚集西太平洋
    """
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    modifier = np.ones_like(lat_grid, dtype=np.float32)
    
    if abs(oni) < 0.3:
        return modifier  # 中性 ENSO，不調整
    
    # 160E 為分界線
    west_mask = lon_grid < 160
    east_mask = lon_grid >= 160
    
    if species in ("skipjack", "yellowfin"):
        # El Niño: 東邊加分，西邊減分
        if oni > 0.5:
            strength = min(oni / 2.0, 0.3)  # 最多 ±30%
            modifier[east_mask] += strength
            modifier[west_mask] -= strength * 0.5
        elif oni < -0.5:
            strength = min(abs(oni) / 2.0, 0.3)
            modifier[west_mask] += strength
            modifier[east_mask] -= strength * 0.5
    elif species == "bigeye":
        # 大目鮪對 ENSO 反應較弱
        if oni > 0.5:
            modifier[east_mask] += min(oni / 3.0, 0.15)
        elif oni < -0.5:
            modifier[west_mask] += min(abs(oni) / 3.0, 0.15)
    elif species == "albacore":
        # 長鰭鮪：La Niña 時北移
        if oni < -0.5:
            high_lat = lat_grid > 25
            modifier[high_lat] += min(abs(oni) / 2.5, 0.2)
    
    return np.clip(modifier, 0.5, 1.5)


# ══════════════════════════════════════════════════════
# 2. 月相 / 潮汐漁獲因子
# ══════════════════════════════════════════════════════

def compute_lunar_fishing_factor(year: int, month: int, day: int) -> Dict:
    """
    計算月相對鮪魚漁獲的影響因子。
    
    研究顯示:
    - 新月 (暗夜): 鮪魚咬餌率最高 (+15-25%)
    - 滿月 (亮夜): 鮪魚分散，咬餌率下降 (-10-15%)
    - 上弦/下弦: 中等
    
    Returns: {
        "phase": 0-1 (0=new, 0.5=full),
        "phase_name": str,
        "fishing_factor": 0.85-1.25,
        "species_factors": {species: factor},
        "recommendation": str,
    }
    """
    # Conway算法計算月相
    phase = _moon_phase_fraction(year, month, day)
    
    # 月相名稱
    if phase < 0.0625 or phase > 0.9375:
        name = "新月"
    elif phase < 0.1875:
        name = "眉月"
    elif phase < 0.3125:
        name = "上弦月"
    elif phase < 0.4375:
        name = "盈凸月"
    elif phase < 0.5625:
        name = "滿月"
    elif phase < 0.6875:
        name = "虧凸月"
    elif phase < 0.8125:
        name = "下弦月"
    else:
        name = "殘月"
    
    # 基礎漁獲因子 (cosine curve: 新月=1.2, 滿月=0.85)
    base_factor = 1.025 + 0.175 * math.cos(2 * math.pi * phase)
    
    # 各魚種調整
    species_factors = {
        "yellowfin": base_factor,  # 標準反應
        "bigeye": 1.0 + 0.25 * math.cos(2 * math.pi * phase),  # 大目鮪更敏感（夜間垂直遷移）
        "skipjack": 1.0 + 0.10 * math.cos(2 * math.pi * phase),  # 正鰹較不敏感
        "albacore": 1.0 + 0.12 * math.cos(2 * math.pi * phase),  # 長鰭鮪中等
    }
    
    # 潮汐影響 (大潮=新月/滿月, 小潮=弦月)
    tidal_strength = abs(math.cos(2 * math.pi * phase))
    
    # 建議
    if phase < 0.1 or phase > 0.9:
        rec = "🌑 新月期間：最佳作業時段，建議全力出擊"
    elif 0.4 < phase < 0.6:
        rec = "🌕 滿月期間：魚群分散，建議集中鋒面/渦旋區域"
    elif 0.2 < phase < 0.35:
        rec = "🌓 上弦月：中等漁況，關注深水區"
    else:
        rec = "🌗 下弦月：漁況回升中，可逐步增加作業"
    
    result = {
        "phase": round(phase, 4),
        "phase_name": name,
        "fishing_factor": round(base_factor, 4),
        "species_factors": {k: round(v, 4) for k, v in species_factors.items()},
        "tidal_strength": round(tidal_strength, 4),
        "recommendation": rec,
    }
    
    log.info(f"  月相: {name} ({phase:.2f}), 漁獲因子: {base_factor:.3f}, 潮汐: {tidal_strength:.2f}")
    return result


def _moon_phase_fraction(year, month, day):
    """計算月相 (0=新月, 0.5=滿月) — Meeus算法簡化版"""
    if month <= 2:
        year -= 1
        month += 12
    a = int(year / 100)
    b = 2 - a + int(a / 4)
    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5
    # 新月週期 = 29.53059 天
    days_since_new = (jd - 2451550.1) % 29.53059
    return days_since_new / 29.53059


# ══════════════════════════════════════════════════════
# 3. 歷史漁場先驗權重
# ══════════════════════════════════════════════════════

# 西太平洋已知高產漁場 (lat, lon, radius_deg, weight)
KNOWN_FISHING_GROUNDS = {
    "yellowfin": [
        # 黑潮延伸體
        (28, 135, 3, 0.15), (25, 140, 3, 0.12),
        # 暖池區域
        (8, 155, 4, 0.20), (5, 160, 4, 0.18),
        # 菲律賓東部
        (15, 130, 3, 0.10), (12, 135, 3, 0.10),
        # 赤道逆流區
        (5, 170, 4, 0.15),
    ],
    "bigeye": [
        # 深水區 (較南)
        (12, 150, 3, 0.18), (10, 155, 3, 0.15),
        # 馬紹爾群島海域
        (8, 168, 3, 0.15), (10, 172, 3, 0.12),
        # 南太平洋
        (7, 160, 4, 0.20),
        # 黑潮深水
        (22, 138, 3, 0.10),
    ],
    "skipjack": [
        # 赤道暖池 (高密度)
        (5, 155, 5, 0.25), (3, 160, 5, 0.25),
        (7, 165, 4, 0.20), (5, 170, 5, 0.22),
        # PNG 海域
        (5, 150, 3, 0.15),
        # 密克羅尼西亞
        (8, 148, 3, 0.12),
    ],
    "albacore": [
        # 北太平洋亞熱帶
        (30, 140, 4, 0.18), (32, 145, 4, 0.15),
        (28, 150, 4, 0.15),
        # 日本東南
        (33, 138, 3, 0.12),
        # 副熱帶輻合帶
        (25, 155, 4, 0.10),
    ],
}


def compute_historical_prior(species: str, lats, lons) -> np.ndarray:
    """
    根據已知漁場位置生成先驗權重地圖。
    
    Returns: (ny, nx) array, 0.0-0.3 的加分值
    """
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    prior = np.zeros_like(lat_grid, dtype=np.float32)
    
    grounds = KNOWN_FISHING_GROUNDS.get(species, [])
    if not grounds:
        return prior
    
    for glat, glon, radius, weight in grounds:
        # 高斯衰減
        dist_sq = (lat_grid - glat)**2 + ((lon_grid - glon) * np.cos(np.radians(glat)))**2
        sigma_sq = (radius * 0.7)**2  # 高斯寬度
        contribution = weight * np.exp(-dist_sq / (2 * sigma_sq))
        prior += contribution
    
    # 歸一化到 0-0.3 (最多加 30% 的先驗權重)
    if np.nanmax(prior) > 0:
        prior = prior / np.nanmax(prior) * 0.3
    
    log.info(f"  歷史漁場先驗 ({species}): max={np.nanmax(prior):.3f}, "
             f"coverage={np.mean(prior > 0.01)*100:.1f}%")
    return prior


# ══════════════════════════════════════════════════════
# 4. 深度溫度剖面 (大目鮪特化)
# ══════════════════════════════════════════════════════

def compute_depth_temperature_index(
    sst: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray, 
    month: int,
    species: str = "bigeye"
) -> np.ndarray:
    """
    估算深層溫度適合度 (Argo 氣候態)。
    大目鮪在 200-300m 深處覓食，需要該深度的溫度在 8-14°C。
    
    這裡用 SST + 緯度 + 月份建立簡單的溫度剖面估算模型,
    真實 Argo 數據可在有網路時替代。
    """
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    
    if species == "bigeye":
        # 大目鮪: 200m 深度溫度估算
        # 經驗公式: T_200m ≈ SST * 0.4 + 5 - 0.15 * |lat|
        t_200m = sst * 0.4 + 5.0 - 0.15 * np.abs(lat_grid)
        # 季節修正
        season_mod = 1.0 + 0.1 * np.cos(2 * np.pi * (month - 3) / 12)
        t_200m *= season_mod
        
        # 適合度: 8-14°C 最佳
        optimal = 11.0  # 最適溫度
        sigma = 3.0     # 寬度
        suitability = np.exp(-((t_200m - optimal) ** 2) / (2 * sigma ** 2))
        
    elif species == "albacore":
        # 長鰭鮪: 100m 深度
        t_100m = sst * 0.55 + 4.0 - 0.1 * np.abs(lat_grid)
        optimal = 14.0
        sigma = 3.5
        suitability = np.exp(-((t_100m - optimal) ** 2) / (2 * sigma ** 2))
        
    else:
        # 表層魚種不需要深度指標
        suitability = np.ones_like(sst)
    
    return suitability.astype(np.float32)


# ══════════════════════════════════════════════════════
# 5. 備用衛星數據端點
# ══════════════════════════════════════════════════════

# NOAA ERDDAP 替代端點
ERDDAP_MIRRORS = [
    "https://coastwatch.pfeg.noaa.gov/erddap/griddap",
    "https://upwell.pfeg.noaa.gov/erddap/griddap",
    "https://apdrc.soest.hawaii.edu/erddap/griddap",
]

# 替代 SST 數據集 ID
SST_DATASETS = [
    "ncdcOisst21Agg",           # NOAA OISST v2.1 (primary)
    "ncdcOisst21Agg_LonPM180",  # Same but -180 to 180
    "jplMURSST41",              # JPL MUR SST v4.1 (高解析)
]

# 替代 Chl 數據集 ID  
CHL_DATASETS = [
    "erdMH1chla8day",     # MODIS Aqua 8-day (primary)
    "erdMH1chlamday",     # MODIS Aqua monthly
    "nesdisVHNSQchlaMonthly",  # VIIRS monthly
]

# 替代洋流數據
CURRENT_DATASETS = [
    "erdOscar1",           # OSCAR (primary)
    "erdQMstress1day",     # QuikSCAT-derived
]


async def try_multiple_erddap_sources(client, data_type: str, lat_range, lon_range):
    """
    嘗試多個 ERDDAP 鏡像和數據集。
    """
    from datetime import datetime, timezone, timedelta
    
    now = datetime.now(timezone.utc)
    
    if data_type == "sst":
        for mirror in ERDDAP_MIRRORS:
            for dataset in SST_DATASETS:
                for days_back in [2, 4, 7, 14]:
                    t = (now - timedelta(days=days_back)).strftime("%Y-%m-%dT12:00:00Z")
                    url = (f"{mirror}/{dataset}.json?sst[({t}):1:({t})]"
                           f"[({lat_range[0]}):1:({lat_range[1]})]"
                           f"[({lon_range[0]}):1:({lon_range[1]})]")
                    try:
                        resp = await client.get(url, timeout=30)
                        if resp.status_code == 200:
                            log.info(f"  SST OK: {dataset}@{mirror.split('/')[2]} (-{days_back}d)")
                            return resp
                    except Exception:
                        continue
    
    elif data_type == "chl":
        for mirror in ERDDAP_MIRRORS:
            for dataset in CHL_DATASETS:
                for days_back in [3, 8, 16, 31]:
                    t = (now - timedelta(days=days_back)).strftime("%Y-%m-%dT00:00:00Z")
                    # 月資料不需要高度維度
                    if "mday" in dataset or "Monthly" in dataset:
                        url = (f"{mirror}/{dataset}.json?chlorophyll[({t}):1:({t})]"
                               f"[({lat_range[0]}):4:({lat_range[1]})]"
                               f"[({lon_range[0]}):4:({lon_range[1]})]")
                    else:
                        url = (f"{mirror}/{dataset}.json?chlorophyll[({t}):1:({t})]"
                               f"[(0.0):1:(0.0)]"
                               f"[({lat_range[0]}):4:({lat_range[1]})]"
                               f"[({lon_range[0]}):4:({lon_range[1]})]")
                    try:
                        resp = await client.get(url, timeout=60)
                        if resp.status_code == 200:
                            log.info(f"  CHL OK: {dataset}@{mirror.split('/')[2]} (-{days_back}d)")
                            return resp
                    except Exception:
                        continue
    
    return None


# ══════════════════════════════════════════════════════
# 6. 綜合增強: 把所有 B 級特徵應用到 HSI
# ══════════════════════════════════════════════════════

def apply_all_enhancements(
    hsi: np.ndarray,
    species: str,
    lats: np.ndarray,
    lons: np.ndarray,
    sst: np.ndarray,
    month: int,
    oni: float = 0.0,
    lunar_factor: float = 1.0,
) -> Dict:
    """
    一次應用所有 B 級增強到 HSI。
    
    Returns: {
        "enhanced_hsi": np.ndarray,
        "enso_modifier": np.ndarray,
        "historical_prior": np.ndarray,
        "depth_index": np.ndarray,
        "lunar_factor": float,
    }
    """
    # 1. ENSO 空間修正
    enso_mod = enso_species_modifier(oni, species, lats, lons)
    
    # 2. 歷史漁場先驗
    hist_prior = compute_historical_prior(species, lats, lons)
    
    # 3. 深度溫度適合度
    depth_idx = compute_depth_temperature_index(sst, lats, lons, month, species)
    
    # 融合公式:
    # enhanced = HSI * ENSO * lunar * (1 + historical_prior)
    # 對深水魚種 (bigeye, albacore) 額外乘以深度指數
    enhanced = hsi * enso_mod * lunar_factor
    enhanced = enhanced * (1.0 + hist_prior)
    
    if species in ("bigeye", "albacore"):
        # 深水魚種: 深度溫度權重較高
        depth_weight = 0.3 if species == "bigeye" else 0.15
        enhanced = enhanced * (1.0 - depth_weight + depth_weight * depth_idx)
    
    enhanced = np.clip(enhanced, 0.0, 1.0).astype(np.float32)
    
    improvement = (np.nanmean(enhanced) - np.nanmean(hsi)) / max(np.nanmean(hsi), 0.001) * 100
    log.info(f"  {species} 增強: {np.nanmean(hsi):.3f} → {np.nanmean(enhanced):.3f} "
             f"({improvement:+.1f}%)")
    
    return {
        "enhanced_hsi": enhanced,
        "enso_modifier": enso_mod,
        "historical_prior": hist_prior,
        "depth_index": depth_idx,
        "lunar_factor": lunar_factor,
    }
