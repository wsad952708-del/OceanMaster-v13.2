"""
OceanMaster v8.0 — 物種棲地適合度模型 (HSI)
=============================================
第二層：用逆向工程取得的物種偏好參數
將物理海洋特徵轉換成「這裡有沒有魚」的評分

每種魚的 SI (Suitability Index) 都是高斯分布：
  SI = exp(-(value - optimal)² / (2σ²))

合併方法：加權幾何平均（比 CATSAT 的固定 AMM 更靈活）
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

log = logging.getLogger("OceanMaster.HSI")


def gaussian_si(value: np.ndarray, optimal: float, sigma: float) -> np.ndarray:
    """高斯分布適合度指數：越靠近最適值越高"""
    return np.exp(-((value - optimal) ** 2) / (2 * sigma ** 2))


def range_si(value: np.ndarray, low: float, high: float) -> np.ndarray:
    """範圍適合度：在範圍內 = 1，範圍外線性衰減"""
    si = np.ones_like(value, dtype=float)
    below = value < low
    above = value > high
    width = max(high - low, 0.01)
    si[below] = np.exp(-((value[below] - low) ** 2) / (0.5 * width ** 2))
    si[above] = np.exp(-((value[above] - high) ** 2) / (0.5 * width ** 2))
    return np.clip(si, 0, 1)


def compute_hsi_skipjack(
    sst: np.ndarray,
    chl: Optional[np.ndarray] = None,
    ssh: Optional[np.ndarray] = None,
    front_strength: Optional[np.ndarray] = None,
    ftle: Optional[np.ndarray] = None,
    do_level: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    鰹魚 (Skipjack) HSI
    
    論文依據：
    - 最佳 SST: 29-30°C (赤道), 27-29°C (溫帶)
    - 最佳 Chl-a: 0.1-0.5 mg/m³
    - SSH: 0.48-0.58 m
    - 強烈趨向 SST 鋒面
    - 溶氧需 > 3.5 ml/L
    """
    # 各因子 SI
    si_sst = gaussian_si(sst, optimal=29.0, sigma=2.5)

    weights = [1.0]
    factors = [si_sst]

    if chl is not None:
        chl_safe = np.maximum(chl, 0.001)
        si_chl = gaussian_si(np.log10(chl_safe), optimal=np.log10(0.25), sigma=0.4)
        factors.append(si_chl)
        weights.append(0.8)

    if ssh is not None:
        si_ssh = range_si(ssh, 0.48, 0.58)
        factors.append(si_ssh)
        weights.append(0.6)

    if front_strength is not None:
        si_front = np.clip(front_strength * 1.5, 0, 1)
        factors.append(si_front)
        weights.append(0.8)

    if ftle is not None:
        ftle_norm = _normalize_0_1(ftle)
        factors.append(ftle_norm)
        weights.append(0.6)

    if do_level is not None:
        si_do = np.where(do_level > 3.5, 1.0, do_level / 3.5)
        factors.append(si_do)
        weights.append(0.5)

    hsi = _weighted_geometric_mean(factors, weights)
    return {"hsi": hsi, "si_sst": si_sst, "species": "skipjack"}


def compute_hsi_yellowfin(
    sst: np.ndarray,
    chl: Optional[np.ndarray] = None,
    ssh: Optional[np.ndarray] = None,
    front_strength: Optional[np.ndarray] = None,
    ftle: Optional[np.ndarray] = None,
    thermocline_depth: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    黃鰭鮪 (Yellowfin) HSI
    
    - 最佳 SST: 26-30°C
    - 溫躍層深度重要（但不如大目鮪關鍵）
    - SSH: -0.05 ~ 0.15 m
    """
    si_sst = gaussian_si(sst, optimal=28.0, sigma=2.0)

    weights = [1.0]
    factors = [si_sst]

    if chl is not None:
        chl_safe = np.maximum(chl, 0.001)
        si_chl = gaussian_si(np.log10(chl_safe), optimal=np.log10(0.2), sigma=0.3)
        factors.append(si_chl)
        weights.append(0.7)

    if ssh is not None:
        si_ssh = range_si(ssh, -0.05, 0.15)
        factors.append(si_ssh)
        weights.append(0.6)

    if front_strength is not None:
        si_front = np.clip(front_strength * 1.3, 0, 1)
        factors.append(si_front)
        weights.append(0.7)

    if ftle is not None:
        factors.append(_normalize_0_1(ftle))
        weights.append(0.5)

    if thermocline_depth is not None:
        si_therm = gaussian_si(thermocline_depth, optimal=120.0, sigma=50.0)
        factors.append(si_therm)
        weights.append(0.8)

    hsi = _weighted_geometric_mean(factors, weights)
    return {"hsi": hsi, "si_sst": si_sst, "species": "yellowfin"}


def compute_hsi_bigeye(
    sst: np.ndarray,
    chl: Optional[np.ndarray] = None,
    ssh: Optional[np.ndarray] = None,
    thermocline_depth: Optional[np.ndarray] = None,
    d20_depth: Optional[np.ndarray] = None,
    ftle: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    大目鮪 (Bigeye) HSI — 溫躍層是生死線
    
    - 表層 SST: 26-30°C（但不是關鍵）
    - 溫躍層深度 = 最關鍵因子（鉤子下在這裡）
    - D20 深度（20°C等溫線）= 延繩釣黃金參考線
    - SSH: -0.10 ~ 0.03 m（冷渦旋邊緣）
    """
    si_sst = range_si(sst, 26.0, 30.0)

    weights = [0.5]  # SST 對大目鮪不是最重要的
    factors = [si_sst]

    if thermocline_depth is not None:
        # 大目鮪偏好溫躍層在 150-250m 的區域
        si_therm = gaussian_si(thermocline_depth, optimal=200.0, sigma=60.0)
        factors.append(si_therm)
        weights.append(1.0)  # 最高權重！

    if d20_depth is not None:
        # D20 在 200-350m = 黃金延繩釣區
        si_d20 = gaussian_si(d20_depth, optimal=250.0, sigma=80.0)
        factors.append(si_d20)
        weights.append(0.9)

    if chl is not None:
        chl_safe = np.maximum(chl, 0.001)
        si_chl = gaussian_si(np.log10(chl_safe), optimal=np.log10(0.15), sigma=0.3)
        factors.append(si_chl)
        weights.append(0.5)

    if ssh is not None:
        si_ssh = range_si(ssh, -0.10, 0.03)
        factors.append(si_ssh)
        weights.append(0.7)

    if ftle is not None:
        factors.append(_normalize_0_1(ftle))
        weights.append(0.4)

    hsi = _weighted_geometric_mean(factors, weights)
    return {
        "hsi": hsi,
        "si_sst": si_sst,
        "species": "bigeye",
        "hook_depth_tip": "鉤子建議下在溫躍層下緣",
    }


def compute_hsi_squid(
    sst: np.ndarray,
    chl: Optional[np.ndarray] = None,
    front_strength: Optional[np.ndarray] = None,
    ftle: Optional[np.ndarray] = None,
    moon_phase: float = 0.5,
    viirs_lights: Optional[np.ndarray] = None,
    lat: Optional[np.ndarray] = None,
    lon: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    魷魚 (Squid) HSI — 月相 + 夜光是關鍵
    
    - SST: 14-22°C (日本魷魚) / 14-24°C (赤魷)
    - Chl-a: 0.2-3.0 mg/m³（比鮪魚高很多）
    - 月相：新月最佳（魷魚趨光性 → 集魚燈效果最好）
    - VIIRS 夜光：直接看到別的魷魚船 = 最強信號
    """
    si_sst = gaussian_si(sst, optimal=17.0, sigma=3.0)

    weights = [0.8]
    factors = [si_sst]

    if chl is not None:
        chl_safe = np.maximum(chl, 0.001)
        si_chl = gaussian_si(np.log10(chl_safe), optimal=np.log10(0.8), sigma=0.5)
        factors.append(si_chl)
        weights.append(0.7)

    if front_strength is not None:
        factors.append(np.clip(front_strength, 0, 1))
        weights.append(0.6)

    if ftle is not None:
        factors.append(_normalize_0_1(ftle))
        weights.append(0.5)

    # 月相因子：新月=1.0, 滿月=0.3
    si_moon = np.full_like(sst, 1.0 - 0.7 * moon_phase)
    factors.append(si_moon)
    weights.append(0.9)

    # VIIRS 漁船燈光：有燈光的地方加分
    if viirs_lights is not None and len(viirs_lights) > 0 and lat is not None and lon is not None:
        si_viirs = _viirs_heatmap(viirs_lights, lat, lon, radius_deg=0.5)
        factors.append(si_viirs)
        weights.append(1.0)  # 最強信號

    hsi = _weighted_geometric_mean(factors, weights)
    return {"hsi": hsi, "si_sst": si_sst, "species": "squid", "moon_phase": moon_phase}


# ═══════════════════════════════════════════════════
#  統一入口：計算所有物種的 HSI
# ═══════════════════════════════════════════════════

def compute_all_hsi(
    ocean_features: Dict[str, Any],
    species_list: list = None,
    moon_phase: float = 0.5,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    一次計算所有目標物種的 HSI
    
    ocean_features 應包含:
        sst, chl, ssh, front_strength, ftle, 
        thermocline_depth, d20_depth, viirs_lights, lat, lon
    """
    if species_list is None:
        species_list = ["skipjack", "yellowfin", "bigeye", "squid"]

    sst = ocean_features.get("sst")
    if sst is None:
        log.error("SST 數據缺失，無法計算 HSI")
        return {}

    chl = ocean_features.get("chl")
    ssh = ocean_features.get("ssh")
    fs = ocean_features.get("front_strength")
    ftle = ocean_features.get("ftle")
    td = ocean_features.get("thermocline_depth")
    d20 = ocean_features.get("d20_depth")
    viirs = ocean_features.get("viirs_lights")
    lat = ocean_features.get("lat")
    lon = ocean_features.get("lon")

    # 網格對齊（如果尺寸不同，做最簡單的裁切對齊）
    base_shape = sst.shape
    chl = _align(chl, base_shape)
    ssh = _align(ssh, base_shape)
    fs = _align(fs, base_shape)
    ftle = _align(ftle, base_shape)
    td = _align(td, base_shape)
    d20 = _align(d20, base_shape)

    results = {}

    for sp in species_list:
        log.info(f"計算 {sp} HSI...")
        try:
            if sp == "skipjack":
                results[sp] = compute_hsi_skipjack(sst, chl, ssh, fs, ftle)
            elif sp == "yellowfin":
                results[sp] = compute_hsi_yellowfin(sst, chl, ssh, fs, ftle, td)
            elif sp == "bigeye":
                results[sp] = compute_hsi_bigeye(sst, chl, ssh, td, d20, ftle)
            elif sp == "squid":
                results[sp] = compute_hsi_squid(
                    sst, chl, fs, ftle, moon_phase, viirs, lat, lon
                )
            else:
                log.warning(f"未知物種: {sp}")

            if sp in results:
                hsi = results[sp]["hsi"]
                log.info(
                    f"  {sp} HSI: 平均={np.nanmean(hsi):.3f}, "
                    f"最高={np.nanmax(hsi):.3f}, "
                    f"適合區佔比={np.nanmean(hsi > 0.6)*100:.1f}%"
                )
        except Exception as e:
            log.error(f"  {sp} HSI 計算失敗: {e}")

    return results


# ═══════════════════════════════════════════════════
#  工具函數
# ═══════════════════════════════════════════════════

def _weighted_geometric_mean(factors: list, weights: list) -> np.ndarray:
    """加權幾何平均（比算術平均更嚴格，任一因子太差會大幅降低結果）"""
    w = np.array(weights)
    w = w / w.sum()
    result = np.ones_like(factors[0], dtype=float)
    for f, wi in zip(factors, w):
        f_safe = np.clip(f, 1e-10, 1.0)
        result *= np.power(f_safe, wi)
    return np.clip(result, 0, 1)


def _normalize_0_1(arr: np.ndarray) -> np.ndarray:
    """正規化到 [0, 1]"""
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return np.zeros_like(arr)
    lo, hi = np.nanpercentile(finite, [2, 98])
    if hi - lo < 1e-10:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def _viirs_heatmap(
    lights: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    radius_deg: float = 0.5,
) -> np.ndarray:
    """將 VIIRS 光點轉換為連續熱力圖"""
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    heatmap = np.zeros_like(lat_grid, dtype=float)

    for pt in lights:
        if len(pt) < 2:
            continue
        d2 = (lat_grid - pt[0]) ** 2 + (lon_grid - pt[1]) ** 2
        heatmap += np.exp(-d2 / (2 * (radius_deg / 2) ** 2))

    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap


def _align(arr: Optional[np.ndarray], target_shape: tuple) -> Optional[np.ndarray]:
    """簡單網格對齊：裁切或填充到目標尺寸"""
    if arr is None:
        return None
    if arr.shape == target_shape:
        return arr
    # 裁切到最小公共尺寸
    ny = min(arr.shape[0], target_shape[0])
    nx = min(arr.shape[1], target_shape[1])
    result = np.full(target_shape, np.nan)
    result[:ny, :nx] = arr[:ny, :nx]
    return result


def get_moon_phase(year: int, month: int, day: int) -> float:
    """
    簡易月相計算 (0=新月, 0.5=滿月, 1=下個新月)
    精度約 ±1天，對漁場預測足夠
    """
    # 已知新月: 2000-01-06
    from datetime import date
    ref = date(2000, 1, 6)
    target = date(year, month, day)
    days = (target - ref).days
    cycle = 29.53058867  # 朔望月
    phase = (days % cycle) / cycle
    return phase
