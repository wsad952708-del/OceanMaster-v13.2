"""
OceanMaster v13.2 — 物種棲地適合度模型 (HSI)
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

    參數來源分類：
    - SST optimal=29°C, σ=2.5  [🧠 專家知識] 熱帶鮪漁業廣泛共識，無單一引用源
    - CHL optimal=0.25, σ=0.4  [🧠 專家知識] 典型赤道太平洋葉綠素範圍
    - SSH range 0.48-0.58m     [🔬 內部設定] 無文獻直接支持此精確範圍
    - 鋒面加權 ×1.5            [🔬 內部設定] 經驗性倍率
    - 溶氧 > 3.5 ml/L         [🧠 專家知識] 鰹魚高活動量，需高溶氧
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
    t100: Optional[np.ndarray] = None,
    gradient_strength: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    黃鰭鮪 (Yellowfin) HSI

    參數來源分類：
    - SST optimal=28°C, σ=2.0     [🧠 專家知識] 黃鰭鮪偏好 26-30°C 為漁業共識
    - CHL optimal=0.2, σ=0.3      [🧠 專家知識] 熱帶寡營養海域典型值
    - SSH range -0.05~0.15m       [🔬 內部設定] 無文獻直接支持此精確範圍
    - 溫躍層 optimal=120m, σ=50   [🧠 專家知識] 中層溫躍層結構偏好
    - T100 optimal=18°C, σ=4.0    [🔬 內部觀察] 開發期間 WCPFC 資料分析所得
    - 梯度強度 /0.15 clip         [🔬 內部設定] 經驗性正規化參數
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

    # [海鷹] T100: 100m 水溫是海鷹核心發現，解釋力 17.42%
    if t100 is not None:
        si_t100 = gaussian_si(t100, optimal=18.0, sigma=4.0)
        factors.append(si_t100)
        weights.append(0.8)

    # [海鷹] 溫度梯度強度: 梯度越強 = 溫躍層越明顯 = 越好
    if gradient_strength is not None:
        # 梯度 0.05-0.3 °C/m 是典型範圍，用 sigmoid 映射
        si_grad = np.clip(gradient_strength / 0.15, 0, 1)
        factors.append(si_grad)
        weights.append(0.5)

    hsi = _weighted_geometric_mean(factors, weights)
    return {"hsi": hsi, "si_sst": si_sst, "species": "yellowfin"}


def compute_hsi_bigeye(
    sst: np.ndarray,
    chl: Optional[np.ndarray] = None,
    ssh: Optional[np.ndarray] = None,
    thermocline_depth: Optional[np.ndarray] = None,
    d20_depth: Optional[np.ndarray] = None,
    ftle: Optional[np.ndarray] = None,
    t100: Optional[np.ndarray] = None,
    gradient_strength: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    大目鮪 (Bigeye) HSI — 溫躍層是生死線

    參數來源分類：
    - SST range 26-30°C, 權重 0.5   [🧠 專家知識] 表層水溫對大目鮪非決定性
    - 溫躍層 optimal=200m, σ=60     [🧠 專家知識] 大目鮪日間棲息深度共識
    - D20 optimal=250m, σ=80        [🧠 專家知識] 延繩釣黃金參考線
    - CHL optimal=0.15, σ=0.3       [🔬 內部設定] 深海物種低葉綠素偏好
    - SSH range -0.10~0.03m         [🔬 內部設定] 冷渦旋邊緣假設
    - T100 optimal=12°C, σ=3.0      [🔬 內部觀察] 開發期間 WCPFC 資料分析所得，
                                      非來自已發表文獻，尚需外部驗證
    - 梯度強度 /0.15 clip           [🔬 內部設定] 經驗性正規化參數
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

    # [海鷹核心] T100: 大目鮪最重要的環境因子
    # 12°C 是大目鮪貪食深度的典型溫度
    if t100 is not None:
        si_t100 = gaussian_si(t100, optimal=12.0, sigma=3.0)
        factors.append(si_t100)
        weights.append(1.0)  # 與溫躍層並列最高權重

    # [海鷹] 溫度梯度: 梯度越強 = 溫躍層越明顯
    if gradient_strength is not None:
        si_grad = np.clip(gradient_strength / 0.15, 0, 1)
        factors.append(si_grad)
        weights.append(0.6)

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
    species: str = "squid_todarodes",
) -> Dict[str, np.ndarray]:
    """
    魷魚 (Squid) HSI — 月相 + 夜光是關鍵

    參數來源分類：
    - SST/CHL 參數從 config.py 讀取  [🔬 內部設定] 各魷魚種獨立配置
    - 月相 cos 映射, 新月=1.0        [🧠 專家知識] 魷釣漁業普遍共識
    - VIIRS 燈光 radius=0.5°         [🔬 內部設定] 經驗性搜索半徑
    - 月相權重=0.30, VIIRS=0.40      [🔬 內部設定] 無系統性敏感度分析
    """
    import config
    params = config.SPECIES_PARAMS.get(species, config.SPECIES_PARAMS.get("squid_todarodes", {}))
    
    opt_sst = params.get("sst_optimal", 17.0)
    sig_sst = params.get("sst_sigma", 3.0)
    opt_chl = params.get("chl_optimal", 0.5)
    sig_chl = params.get("chl_sigma", 0.5)

    si_sst = gaussian_si(sst, optimal=opt_sst, sigma=sig_sst)

    weights = [0.25]
    factors = [si_sst]

    if chl is not None:
        chl_safe = np.maximum(chl, 0.001)
        si_chl = gaussian_si(np.log10(chl_safe), optimal=np.log10(opt_chl), sigma=sig_chl)
        factors.append(si_chl)
        weights.append(0.15)

    if front_strength is not None:
        factors.append(np.clip(front_strength, 0, 1))
        weights.append(0.20)

    if ftle is not None:
        factors.append(_normalize_0_1(ftle))
        weights.append(0.10)

    # 月相因子：新月=1.0, 滿月=0.15 (floor 避免 geometric mean 歸零)
    si_moon = np.full_like(sst, 0.15 + 0.85 * (1.0 + np.cos(2 * np.pi * moon_phase)) / 2.0)
    factors.append(si_moon)
    weights.append(0.30)

    # VIIRS 漁船燈光：有燈光的地方加分
    if viirs_lights is not None and len(viirs_lights) > 0 and lat is not None and lon is not None:
        si_viirs = _viirs_heatmap(viirs_lights, lat, lon, radius_deg=0.5)
        factors.append(si_viirs)
        weights.append(0.40)  # 最強信號

    hsi = _weighted_geometric_mean(factors, weights)
    return {"hsi": hsi, "si_sst": si_sst, "species": species, "moon_phase": moon_phase}


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
    # [海鷹] 新特徵
    t100 = ocean_features.get("t100")
    gradient_str = ocean_features.get("gradient_strength")
    chl_lag15d = ocean_features.get("chl_lag15d")

    # 網格對齊（如果尺寸不同，做最簡單的裁切對齊）
    base_shape = sst.shape
    chl = _align(chl, base_shape)
    ssh = _align(ssh, base_shape)
    fs = _align(fs, base_shape)
    ftle = _align(ftle, base_shape)
    td = _align(td, base_shape)
    d20 = _align(d20, base_shape)
    t100 = _align(t100, base_shape)
    gradient_str = _align(gradient_str, base_shape)
    chl_lag15d = _align(chl_lag15d, base_shape)

    # [蒼鷺] CHL 時滯融合: 若有 15 天前 CHL，以 0.6:0.4 融合剛多於即時 CHL
    chl_effective = chl
    if chl_lag15d is not None and chl is not None:
        chl_effective = np.where(
            np.isfinite(chl_lag15d),
            0.4 * np.nan_to_num(chl, nan=0) + 0.6 * np.nan_to_num(chl_lag15d, nan=0),
            chl
        ).astype(np.float32)
        log.info(f"  [蒼鷺] CHL lag fusion: 0.4*now + 0.6*lag15d, "
                 f"effective mean={np.nanmean(chl_effective):.4f}")

    results = {}

    for sp in species_list:
        log.info(f"計算 {sp} HSI...")
        try:
            if sp == "skipjack":
                results[sp] = compute_hsi_skipjack(sst, chl_effective, ssh, fs, ftle)
            elif sp == "yellowfin":
                results[sp] = compute_hsi_yellowfin(sst, chl_effective, ssh, fs, ftle, td,
                                                     t100=t100, gradient_strength=gradient_str)
            elif sp == "bigeye":
                results[sp] = compute_hsi_bigeye(sst, chl_effective, ssh, td, d20, ftle,
                                                  t100=t100, gradient_strength=gradient_str)
            elif sp in ("squid", "squid_todarodes", "squid_ommastrephes",
                         "neon_flying_squid", "japanese_flying_squid"):
                results[sp] = compute_hsi_squid(
                    sst, chl_effective, fs, ftle, moon_phase, viirs, lat, lon,
                    species=sp
                )
            elif sp == "albacore":
                # 長鰭鮪: 偏好冷水 (Topt=18.5°C), 類似 yellowfin 但溫度參數不同
                si_sst = gaussian_si(sst, 18.5, 4.0)
                si_chl = gaussian_si(np.log10(np.maximum(chl, 0.001)),
                                     optimal=np.log10(0.25), sigma=0.4) if chl is not None else np.ones_like(sst) * 0.5
                si_front = np.clip(fs * 1.5, 0, 1) if fs is not None else np.ones_like(sst) * 0.3
                hsi = _weighted_geometric_mean(
                    [si_sst, si_chl, si_front],
                    [0.5, 0.25, 0.25]
                )
                results[sp] = {"hsi": hsi, "si_sst": si_sst, "si_chl": si_chl}
            elif sp == "mahi_mahi":
                # 鬼頭刀: 表層暖水 (Topt=27°C), 強 FAD 關聯
                si_sst = gaussian_si(sst, 27.0, 3.0)
                si_chl = gaussian_si(chl, 0.35, 0.2) if chl is not None else np.ones_like(sst) * 0.5
                si_front = np.clip(fs * 2.0, 0, 1) if fs is not None else np.ones_like(sst) * 0.3
                hsi = _weighted_geometric_mean(
                    [si_sst, si_chl, si_front],
                    [0.5, 0.2, 0.3]
                )
                results[sp] = {"hsi": hsi, "si_sst": si_sst, "si_chl": si_chl}
            elif sp == "blue_marlin":
                # 旗魚: 暖水 (Topt=26°C), 溫躍層覓食
                si_sst = gaussian_si(sst, 26.0, 3.5)
                si_chl = gaussian_si(chl, 0.25, 0.15) if chl is not None else np.ones_like(sst) * 0.5
                si_td = gaussian_si(td, 200, 80) if td is not None else np.ones_like(sst) * 0.5
                hsi = _weighted_geometric_mean(
                    [si_sst, si_chl, si_td],
                    [0.45, 0.2, 0.35]
                )
                results[sp] = {"hsi": hsi, "si_sst": si_sst, "si_chl": si_chl}
            elif sp == "mackerel_scad":
                # 竹筴魚: 溫帶偏好 (Topt=24°C), 高 Chl-a
                si_sst = gaussian_si(sst, 24.0, 4.0)
                si_chl = gaussian_si(chl, 1.0, 0.8) if chl is not None else np.ones_like(sst) * 0.5
                si_front = np.clip(fs * 1.5, 0, 1) if fs is not None else np.ones_like(sst) * 0.3
                hsi = _weighted_geometric_mean(
                    [si_sst, si_chl, si_front],
                    [0.4, 0.35, 0.25]
                )
                results[sp] = {"hsi": hsi, "si_sst": si_sst, "si_chl": si_chl}
            else:
                # [v13.2] 通用 fallback: 用 species_params 的 Topt_C 如果存在
                try:
                    from engine.species_params import SPECIES as SP_PARAMS
                    sp_data = SP_PARAMS.get(sp, {})
                    t_opt = sp_data.get("Topt_C", 25.0)
                    t_sigma = sp_data.get("Topt_sigma", 4.0)
                    si_sst = gaussian_si(sst, t_opt, t_sigma)
                    si_chl = gaussian_si(chl, 0.3, 0.2) if chl is not None else np.ones_like(sst) * 0.5
                    hsi = _weighted_geometric_mean([si_sst, si_chl], [0.6, 0.4])
                    results[sp] = {"hsi": hsi, "si_sst": si_sst}
                    log.info(f"  {sp}: 使用通用 HSI (Topt={t_opt}°C)")
                except Exception:
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


# ═══════════════════════════════════════════════════════════
# [v13.2-P1] 產卵棲地 HSI
# ═══════════════════════════════════════════════════════════

def compute_spawning_habitat(
    sst: np.ndarray,
    month: int,
    species_key: str,
    chl: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """計算物種的產卵棲地適合度指數 (Spawning Habitat HSI)

    基於 species_params.py 的 T_spawning 和 T_spawning_months，
    結合 SST 熱適合度 × 季節適合度 × CHL 營養適合度。

    Args:
        sst: SST 網格 (°C)
        month: 當前月份 (1-12)
        species_key: 物種鍵 (e.g. "yellowfin", "bigeye")
        chl: (可選) Chl-a 網格 (mg/m³)

    Returns:
        {"spawning_hsi": ndarray, "si_sst": ndarray,
         "si_season": float, "species": str, "in_season": bool}
    """
    from engine.species_params import SPECIES

    sp = SPECIES.get(species_key, {})
    t_spawn = sp.get("T_spawning", (26, 30))
    spawn_months = sp.get("T_spawning_months", [])
    chl_optimal = sp.get("chl_optimal", (0.1, 0.5))

    # ── 1. SST 熱適合度 ──
    # 使用 range_si: T_spawning 範圍內 = 1.0, 範圍外 Gaussian 衰減
    si_sst = range_si(sst, t_spawn[0], t_spawn[1])

    # ── 2. 季節適合度 ──
    in_season = month in spawn_months
    if in_season:
        si_season = 1.0
    elif spawn_months:
        # 計算距離最近產卵月的環形距離, 給予衰減
        min_dist = min(
            min(abs(month - m), 12 - abs(month - m))
            for m in spawn_months
        )
        si_season = max(0.0, np.exp(-0.5 * (min_dist / 1.5) ** 2))
    else:
        si_season = 0.5  # 無產卵月資料 → 中性

    # ── 3. CHL 營養適合度 (可選) ──
    if chl is not None:
        si_chl = range_si(chl, chl_optimal[0], chl_optimal[1])
    else:
        si_chl = np.ones_like(sst)

    # ── 組合: 加權幾何平均 ──
    # 權重: SST (0.5) + season (0.3) + CHL (0.2)
    hsi = np.power(
        np.clip(si_sst, 1e-6, 1.0) ** 0.5
        * np.clip(si_season, 1e-6, 1.0) ** 0.3
        * np.clip(si_chl, 1e-6, 1.0) ** 0.2,
        1.0  # 已在指數中加權，不再做根號
    )

    name_zh = sp.get("name_zh", species_key)
    log.info(f"  🥚 Spawning HSI ({name_zh}): "
             f"mean={np.nanmean(hsi):.3f}, in_season={in_season}, "
             f"si_season={si_season:.2f}")

    return {
        "spawning_hsi": hsi.astype(np.float32),
        "si_sst": si_sst.astype(np.float32),
        "si_season": float(si_season),
        "species": species_key,
        "in_season": in_season,
    }


def compute_all_spawning_hsi(
    sst: np.ndarray,
    month: int,
    species_list: Optional[list] = None,
    chl: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, Any]]:
    """批次計算所有物種的產卵棲地 HSI

    Args:
        sst: SST 網格
        month: 當前月份
        species_list: 物種列表 (None = 全部有 T_spawning 的物種)
        chl: (可選) Chl-a 網格

    Returns:
        {species_key: spawning_result_dict, ...}
    """
    from engine.species_params import SPECIES

    if species_list is None:
        species_list = [
            k for k, v in SPECIES.items()
            if "T_spawning" in v and "T_spawning_months" in v
        ]

    results = {}
    for sp in species_list:
        try:
            results[sp] = compute_spawning_habitat(sst, month, sp, chl)
        except Exception as e:
            log.warning(f"  Spawning HSI ({sp}) failed: {e}")
    return results


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

