"""
OceanMaster v13.2 — AI 融合引擎 + 熱點提取
============================================
第三層：將所有物理特徵 + HSI 分數融合成最終漁場評分

融合邏輯：
  最終評分 = HSI × (1 + bonus) → percentile rescaling → 0.15–0.90
  + 物種-水深驗證
  + 跨物種座標去重
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timezone
import logging

log = logging.getLogger("OceanMaster.Fusion")

# [v15.1] V3.0 #2: EEZ / MPA 地理圍欄
try:
    from engine.eez_filter import get_eez_filter
    HAS_EEZ = True
except ImportError:
    HAS_EEZ = False

# [v15.3] V3.0 #4: AIS 競爭密度懲罰
try:
    from engine.competition_penalty import get_competition_penalty
    HAS_COMPETITION = True
except ImportError:
    HAS_COMPETITION = False

# ── [v13.2] 商業級排除區 — 封閉/邊緣海域、商業延繩釣不作業的海域 ──
# (lat_min, lat_max, lon_min, lon_max)
_EXCLUDED_ZONES = [
    # ===== 封閉/邊緣海域 =====
    (4.0, 10.0, 117.0, 124.0),     # 蘇祿海 (Sulu Sea) — 島嶼密佈、淺礁多
    (-2.0, 6.0, 118.0, 128.0),     # 西里伯斯海 (Celebes Sea) — 印尼/菲律賓封閉
    (-8.0, -3.0, 105.0, 120.0),    # 爪哇海 (Java Sea) — 平均深度 46m
    (30.0, 41.0, 117.0, 127.0),    # 黃海+渤海 — 平均深度 44m
    (-1.0, 8.0, 106.0, 117.0),     # 南海南部淺水區 + 納土納
    (0.0, 8.0, 95.0, 106.0),       # 麻六甲海峽+安達曼海東
    (8.0, 22.0, 105.0, 112.0),     # 東京灣+海南東+南海西岸近海
    # ===== 中國沿岸 (新增) =====
    (23.0, 28.0, 116.5, 120.5),    # 福建沿岸 (廈門→福州→浙南)
    (20.0, 23.5, 109.0, 117.5),    # 廣東沿岸 (珠江口→汕頭→香港→澳門)
    (27.0, 32.0, 118.0, 123.0),    # 浙江沿岸 (溫州→寧波→舟山→上海)
    (32.0, 36.0, 118.0, 123.0),    # 江蘇沿岸 (南通→連雲港)
    (24.0, 30.0, 120.0, 124.0),    # 東海大陸棚 (平均深度 <200m)
    # ===== 近岸淺水區 =====
    (21.5, 26.0, 119.0, 122.5),    # 台灣近海 (含澎湖)
    (33.0, 38.0, 125.0, 130.0),    # 韓國近海
    (30.0, 36.0, 129.0, 133.0),    # 日本瀨戶內海+近畿沿岸
]

# [v13.2-P2] 陸地遮罩 — 使用共享模組 (與 main_v10_3.py 同一份數據)
from engine.land_mask import KNOWN_LAND_BBOXES, near_known_land, FUSION_BUFFER_DEG

_KNOWN_LAND_BBOXES_FUSION = KNOWN_LAND_BBOXES  # backward compat
_LAND_BUFFER = FUSION_BUFFER_DEG


def _near_land_bbox(lat: float, lon: float) -> bool:
    """檢查是否在已知陸地 bbox + buffer 範圍內 (delegates to shared module)."""
    return near_known_land(lat, lon, buffer_deg=FUSION_BUFFER_DEG)


def _in_excluded_zone(lat: float, lon: float) -> bool:
    """檢查是否在商業排除區域內。"""
    for lat1, lat2, lon1, lon2 in _EXCLUDED_ZONES:
        if lat1 <= lat <= lat2 and lon1 <= lon <= lon2:
            return True
    return False


def _is_valid_fishing_point(lat: float, lon: float, depth_m: float,
                            species: str) -> bool:
    """[v13.2] 商業級漁場驗證 — 綜合判斷此座標是否為合法漁場。

    Required:
      1. 水深 >= 200m (遠洋延繩釣最低作業深度)
      2. 不在封閉/邊緣海排除區
      3. 不在已知陸地 bbox 內 (除非水深 > 800m — 真正深海)
      4. 物種水深範圍匹配
    """
    # R1: 最低水深
    if depth_m < 200:
        return False

    # R2: 封閉海域排除
    if _in_excluded_zone(lat, lon):
        return False

    # R3: 近陸地 — 除非真正深海 (>800m)
    if _near_land_bbox(lat, lon) and depth_m < 800:
        return False

    # R4: 物種水深範圍
    d_min, d_max = SPECIES_DEPTH_RANGE.get(species, (200, 8000))
    if depth_m < d_min or depth_m > d_max:
        return False

    return True

# 物種合理水深範圍 (min_depth_m, max_depth_m)  — 商業遠洋延繩釣作業範圍
SPECIES_DEPTH_RANGE = {
    "skipjack":  (200,  6000),
    "yellowfin": (200,  6000),
    "bigeye":    (200,  6000),
    "albacore":  (200,  6000),
    "swordfish": (200,  6000),
    "squid":     (200,  5000),
    "neon_flying_squid": (200, 6000),
    "japanese_flying_squid": (200, 5000),
}


def fuse_and_rank(
    hsi_results: Dict[str, Dict[str, np.ndarray]],
    ocean_features: Dict[str, Any],
    lat: np.ndarray,
    lon: np.ndarray,
    top_n: int = 20,
) -> Dict[str, Any]:
    """
    融合所有圖層並提取最佳漁場熱點

    Steps:
    1. 將每個物種的 HSI 分數疊加
    2. 額外加分：SST 鋒面、FTLE 脊線、VIIRS 燈光
    3. 安全扣分：高浪區、低溶氧區
    4. 排名提取 Top-N 熱點
    """
    log.info("=" * 50)
    log.info("開始 AI 融合...")

    base_shape = lat.shape if lat.ndim == 2 else (len(lat), len(lon))
    lat_1d = lat if lat.ndim == 1 else lat[:, 0]
    lon_1d = lon if lon.ndim == 1 else lon[0, :]

    # ── 1. 合併所有物種 HSI ──
    all_species_scores = {}
    for species, data in hsi_results.items():
        hsi = data.get("hsi")
        if hsi is not None:
            # 確保尺寸匹配
            if hsi.shape != base_shape:
                ny = min(hsi.shape[0], base_shape[0])
                nx = min(hsi.shape[1], base_shape[1])
                aligned = np.full(base_shape, np.nan)
                aligned[:ny, :nx] = hsi[:ny, :nx]
                hsi = aligned
            all_species_scores[species] = hsi

    if not all_species_scores:
        log.error("沒有任何 HSI 結果可融合")
        return {"hotspots": [], "error": "No HSI data"}

    # ── 2. 額外加分層 ──
    # [v17] 競品整合: 增加鋒面持久性/匯聚帶加分 (INCOIS PFZ + Aker BioMarine)
    bonus = np.zeros(base_shape)

    # SST 鋒面加分 (+20%)
    front_strength = ocean_features.get("front_strength")
    if front_strength is not None:
        fs = _safe_resize(front_strength, base_shape)
        bonus += 0.2 * np.nan_to_num(fs, nan=0)

    # [v17] 鋒面持久性加分 (+10%) — INCOIS 研究: 持續鋒面 >> 瞬時鋒面
    front_persistence = ocean_features.get("front_persistence")
    if front_persistence is not None:
        fp = _safe_resize(front_persistence, base_shape)
        bonus += 0.10 * np.nan_to_num(fp, nan=0)

    # FTLE 脊線加分 (+15%)
    ftle = ocean_features.get("ftle")
    if ftle is not None:
        ft = _safe_resize(ftle, base_shape)
        ft_norm = _normalize(ft)
        bonus += 0.15 * ft_norm

    # [v17] 海流匯聚帶加分 (+10%) — Aker BioMarine: 漂流物匯聚預測魚群
    convergence = ocean_features.get("convergence_index")
    if convergence is not None:
        cv = _safe_resize(convergence, base_shape)
        bonus += 0.10 * np.clip(np.nan_to_num(cv, nan=0), 0, 1)

    # VIIRS 漁船燈光加分 (+25%)
    viirs = ocean_features.get("viirs_lights")
    if viirs is not None and len(viirs) > 0:
        viirs_map = _viirs_to_grid(viirs, lat_1d, lon_1d)
        if viirs_map.shape == base_shape:
            bonus += 0.25 * viirs_map

    # [v17-P5] 風速 × 鋒面位移 加分/懲罰
    # 科學依據: 中等風速 (5-15 m/s) 增強鋒面位移 → Ekman transport → 營養鹽上湧
    # 高風速 (>15 m/s) → 海況惡劣，作業困難
    ws = ocean_features.get("wind_speed")
    if ws is not None:
        ws_grid = _safe_resize(ws, base_shape)
        ws_safe = np.nan_to_num(ws_grid, nan=0)

        # 中等風速 + 鋒面存在 → 加分 (+8%)
        # 風推動鋒面位移，增強上升流和營養鹽混合
        if front_strength is not None:
            moderate_wind = (ws_safe >= 5.0) & (ws_safe <= 15.0)
            fs_safe = np.nan_to_num(fs, nan=0)
            wind_front_bonus = 0.08 * fs_safe * moderate_wind.astype(float)
            bonus += wind_front_bonus
            n_bonus = int(np.sum(wind_front_bonus > 0.005))
            if n_bonus > 0:
                log.info(f"  💨 Wind-front displacement bonus: {n_bonus} cells, "
                         f"mean={np.nanmean(wind_front_bonus[wind_front_bonus > 0]):.4f}")

        # 高風速懲罰 (-15%)
        high_wind = ws_safe > 15.0
        if np.any(high_wind):
            penalty = -0.15 * np.clip((ws_safe - 15.0) / 10.0, 0, 1) * high_wind.astype(float)
            bonus += penalty
            n_penalty = int(np.sum(high_wind))
            log.info(f"  🌬️ High wind penalty: {n_penalty} cells >15 m/s")

    # [v18] GFW Fishing Effort Prior — 群體驗證加分
    # Ref: Kroodsma et al. (2018) Science 359:904-908
    # 當 GFW AIS 數據顯示近期有漁船在此作業 → 加分 (最多+15%)
    ais_density = ocean_features.get("ais_fishing_density")
    if ais_density is not None:
        ad = _safe_resize(ais_density, base_shape)
        ad_safe = np.nan_to_num(ad, nan=0)
        # ais_fishing_density 0-5 scale → normalize to 0-1
        gfw_prior = np.clip(ad_safe / 5.0, 0, 1)
        bonus += 0.15 * gfw_prior
        n_gfw = int(np.sum(gfw_prior > 0.1))
        if n_gfw > 0:
            log.info(f"  🚢 GFW fishing effort prior: {n_gfw} cells boosted, "
                     f"mean={np.nanmean(gfw_prior[gfw_prior > 0.1]):.3f}")

    # [v17-P5-fix] 限制 bonus 合理範圍，防止極端疊加
    bonus = np.clip(bonus, -0.30, 0.75)

    # ── 3. 為每個物種計算最終分數 ──
    species_hotspots = {}

    # 獲取深度數據 (若有)
    bathy_grid = ocean_features.get("bathy")

    for species, hsi in all_species_scores.items():
        # 最終分數 = HSI × (1 + bonus)
        raw_score = hsi * (1 + bonus)
        # [v17-fix] max-normalize 而非 clip(1.0), 保留相對差異
        rs_max = np.nanmax(raw_score)
        if rs_max > 0:
            raw_score_clipped = (raw_score / rs_max).astype(np.float32)
        else:
            raw_score_clipped = np.zeros_like(raw_score, dtype=np.float32)
        raw_score_clipped = np.clip(raw_score_clipped, 0.0, 1.0)

        # [v16.0-fix] Percentile-based rescaling — 消除 1.0 飽和
        # 將原始分數映射到 0.15–0.95 範圍，保留相對排名
        # 上限 0.95 (非 1.0) — 沒有任何漁場條件是 100% 完美的
        final_score = _percentile_rescale(raw_score, out_min=0.15, out_max=0.95)

        # [海鷹/蒼鷺] 2D Gaussian 空間平滑 — 降低雜訊，穩定 hotspot 位置
        # 魷魚用精細級 sigma=1.0 (0.25°), 鮪魚用粗略級 sigma=2.0 (0.5°)
        try:
            from scipy.ndimage import gaussian_filter as _gf
            _sigma = 1.0 if 'squid' in species else 2.0
            final_score = _gf(
                np.nan_to_num(final_score, nan=0), sigma=_sigma
            ).astype(np.float32)
            # 平滑後重新 clip 到合理範圍
            final_score = np.clip(final_score, 0.0, 0.95)
        except ImportError:
            pass  # scipy 不可用，跳過平滑

        # [v16.0-fix] 用 final_score 排名（已校準到 0.15-0.95）
        # 避免 raw max-normalized 導致 100% 不切實際的顯示
        hotspots = _extract_hotspots(
            final_score, lat_1d, lon_1d, species, top_n=top_n,
            bathy_grid=bathy_grid,
            raw_score_grid=raw_score_clipped,
        )

        # [v15.3-audit] 將 ml_type 注入每個 hotspot，讓下游可知模型來源
        _ml_type = hsi_results.get(species, {}).get("ml_type", "science-only")
        for h in hotspots:
            h["ml_type"] = _ml_type

        species_hotspots[species] = {
            "score_grid": final_score,
            "hotspots": hotspots,
            "stats": {
                "mean_score": float(np.nanmean(final_score)),
                "max_score": float(np.nanmax(final_score)),
                "suitable_pct": float(np.nanmean(final_score > 0.3) * 100),
                "hotspot_count": len(hotspots),
            },
        }

        log.info(
            f"  {species}: 最高={np.nanmax(final_score):.3f}, "
            f"適合區={np.nanmean(final_score > 0.3)*100:.1f}%, "
            f"熱點={len(hotspots)}個"
        )

    # ── 4. 全物種綜合熱點（保證每物種最少配額 + 座標去重） ──
    n_species = len(species_hotspots)
    per_species_min = max(3, top_n // n_species) if n_species > 0 else top_n

    combined_hotspots = []
    remaining_candidates = []
    used_coords = set()  # (lat, lon) 去重

    for species, data in species_hotspots.items():
        sorted_hs = sorted(data["hotspots"], key=lambda x: x["score"], reverse=True)
        added = 0
        for h in sorted_hs:
            coord_key = (round(h["lat"], 2), round(h["lon"], 2))
            if coord_key in used_coords:
                continue  # 跨物種座標去重
            if added < per_species_min:
                combined_hotspots.append(h)
                used_coords.add(coord_key)
                added += 1
            else:
                remaining_candidates.append(h)

    # Fill remaining slots with highest-scoring candidates across species
    remaining_candidates.sort(key=lambda x: x["score"], reverse=True)
    slots_left = top_n - len(combined_hotspots)
    for h in remaining_candidates:
        if slots_left <= 0:
            break
        coord_key = (round(h["lat"], 2), round(h["lon"], 2))
        if coord_key not in used_coords:
            combined_hotspots.append(h)
            used_coords.add(coord_key)
            slots_left -= 1

    # Final sort by score
    combined_hotspots.sort(key=lambda x: x["score"], reverse=True)
    combined_hotspots = combined_hotspots[:top_n]

    # Re-rank
    for i, h in enumerate(combined_hotspots):
        h["rank"] = i + 1

    log.info(f"融合完成: 共 {len(combined_hotspots)} 個熱點 (去重後)")

    # ── 5. [v15.1] EEZ 地理圍欄 ──
    if HAS_EEZ:
        eez = get_eez_filter()
        combined_hotspots = eez.filter_hotspots(combined_hotspots)
        for sp_data in species_hotspots.values():
            eez.filter_hotspots(sp_data["hotspots"])

    # ── 6. [v15.3] AIS 競爭密度懲罰 ──
    if HAS_COMPETITION:
        cp = get_competition_penalty()
        cp_ocean = {
            "ais_fishing_density": ocean_features.get("ais_fishing_density"),
            "lat": lat_1d,
            "lon": lon_1d,
        }
        combined_hotspots = cp.apply_penalty(combined_hotspots, cp_ocean)
        for sp_data in species_hotspots.values():
            cp.apply_penalty(sp_data["hotspots"], cp_ocean)

    return {
        "species": species_hotspots,
        "combined_hotspots": combined_hotspots,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "area": {
            "lat_range": [float(lat_1d.min()), float(lat_1d.max())],
            "lon_range": [float(lon_1d.min()), float(lon_1d.max())],
        },
    }


def _extract_hotspots(
    score: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    species: str,
    top_n: int = 20,
    min_score: float = 0.15,
    min_distance_deg: float = 0.5,
    bathy_grid: "np.ndarray | None" = None,
    raw_score_grid: "np.ndarray | None" = None,  # [v17] 原始未 rescale 分數
) -> List[Dict[str, Any]]:
    """
    從分數網格中提取最佳漁場熱點

    [v13.2] 商業級驗證:
      1. 非極大值抑制 (NMS)
      2. 物種水深驗證
      3. 封閉海域排除
      4. 陸地幾何排除
      5. 最低水深 200m
    """
    score_filled = np.nan_to_num(score, nan=0)

    # 找所有超過閾值的點
    candidates = []
    ny, nx = score_filled.shape
    n_excluded = 0
    _bathy_warned = False
    if bathy_grid is None or bathy_grid.size == 0:
        log.warning("No bathymetry data — assuming 2000m depth, shallow/land hotspots may not be filtered")
        _bathy_warned = True
    for iy in range(ny):
        for ix in range(nx):
            if score_filled[iy, ix] >= min_score:
                pt_lat = float(lat[iy])
                pt_lon = float(lon[ix])

                # [v13.2] 計算水深
                if bathy_grid is not None and iy < bathy_grid.shape[0] and ix < bathy_grid.shape[1]:
                    depth_m = float(-bathy_grid[iy, ix])  # bathy is negative
                else:
                    depth_m = 2000.0  # 假設深海 (無 bathy 資料)

                # [v13.2] 商業級漁場驗證
                if not _is_valid_fishing_point(pt_lat, pt_lon, depth_m, species):
                    n_excluded += 1
                    continue

                candidates.append({
                    "lat": pt_lat,
                    "lon": pt_lon,
                    "score": float(score_filled[iy, ix]),
                    "iy": iy,
                    "ix": ix,
                })

    if n_excluded > 0:
        log.info(f"  {species}: excluded {n_excluded} points (land/shallow/marginal sea)")

    # 按分數降序
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # NMS: 去除太近的點
    selected = []
    for c in candidates:
        too_close = False
        for s in selected:
            dist = np.sqrt((c["lat"] - s["lat"]) ** 2 + (c["lon"] - s["lon"]) ** 2)
            if dist < min_distance_deg:
                too_close = True
                break
        if not too_close:
            hs_dict = {
                "lat": c["lat"],
                "lon": c["lon"],
                "score": c["score"],
                "species": species,
                "rank": len(selected) + 1,
            }
            # [v17] 雙軌: 存放 raw_hsi (未 percentile rescale 的真實 HSI)
            if raw_score_grid is not None:
                iy, ix = c.get("iy", 0), c.get("ix", 0)
                hs_dict["raw_hsi"] = float(raw_score_grid[iy, ix])
            selected.append(hs_dict)
            if len(selected) >= top_n:
                break

    return selected


def _percentile_rescale(
    arr: np.ndarray,
    out_min: float = 0.15,
    out_max: float = 0.95,
) -> np.ndarray:
    """
    [v16.0] Percentile-based rescaling — 將飽和的 HSI 映射到合理範圍。
    使用 5th-95th percentile 的點作為錨定，避免極端值影響。
    out_max=0.95: 沒有完美漁場，最高 95%。
    """
    finite = arr[np.isfinite(arr) & (arr > 0)]
    if len(finite) == 0:
        return np.zeros_like(arr)
    p5, p95 = np.nanpercentile(finite, [5, 95])
    if p95 - p5 < 1e-10:
        # 所有值幾乎相等 → 回傳中等值
        return np.where(arr > 0, (out_min + out_max) / 2, 0.0).astype(np.float32)
    normed = (arr - p5) / (p95 - p5)  # 0-1 range based on percentiles
    scaled = out_min + normed * (out_max - out_min)
    return np.clip(scaled, 0.0, out_max).astype(np.float32)


# ═══════════════════════════════════════════════════
#  安全評估 + 航線建議
# ═══════════════════════════════════════════════════

# DEPRECATED: 安全評估僅用距離，不使用氣象數據，主程式有獨立安全模組覆蓋此值
def assess_safety(
    hotspots: List[Dict],
    weather: Optional[Dict] = None,
    current_lat: float = 25.0,
    current_lon: float = 140.0,
) -> List[Dict]:
    """[DEPRECATED] 為每個熱點加入安全評估和距離資訊。主程式有獨立安全模組覆蓋此值。"""
    for h in hotspots:
        # 距離（大圓距離簡化版）
        dlat = h["lat"] - current_lat
        dlon = h["lon"] - current_lon
        dist_nm = np.sqrt(dlat**2 + (dlon * np.cos(np.radians(current_lat)))**2) * 60
        h["distance_nm"] = round(float(dist_nm), 1)

        # 預估航行時間（假設 10 節）
        h["travel_hours"] = round(dist_nm / 10, 1)

        # 安全等級（簡化版，正式版需要氣象數據）
        h["safety"] = "🟢 安全" if dist_nm < 500 else "🟡 注意"

    return hotspots


# ═══════════════════════════════════════════════════
#  工具函數
# ═══════════════════════════════════════════════════

def _safe_resize(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    if arr.shape == target_shape:
        return arr
    result = np.full(target_shape, np.nan)
    ny = min(arr.shape[0], target_shape[0])
    nx = min(arr.shape[1], target_shape[1])
    result[:ny, :nx] = arr[:ny, :nx]
    return result


def _normalize(arr: np.ndarray) -> np.ndarray:
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return np.zeros_like(arr)
    lo, hi = np.nanpercentile(finite, [2, 98])
    if hi - lo < 1e-10:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def _viirs_to_grid(
    lights: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    sigma: float = 0.3,
) -> np.ndarray:
    """VIIRS 光點 → 連續熱力圖"""
    lat_g, lon_g = np.meshgrid(lat, lon, indexing="ij")
    heat = np.zeros_like(lat_g, dtype=float)
    for pt in lights:
        if len(pt) < 2:
            continue
        try:
            pt_lat, pt_lon = float(pt[0]), float(pt[1])
        except (TypeError, ValueError):
            continue
        d2 = (lat_g - pt_lat)**2 + (lon_g - pt_lon)**2
        heat += np.exp(-d2 / (2 * sigma**2))
    mx = heat.max()
    if mx > 0:
        heat /= mx
    return heat
