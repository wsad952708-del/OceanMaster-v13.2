"""
OceanMaster v10.5 — AI 融合引擎 + 熱點提取
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

# 物種合理水深範圍 (min_depth_m, max_depth_m)  — 淺於 min 或深於 max 的直接排除
SPECIES_DEPTH_RANGE = {
    "skipjack":  (50,   6000),
    "yellowfin": (50,   6000),
    "bigeye":    (100,  6000),
    "albacore":  (100,  6000),
    "swordfish": (100,  6000),
    "squid":     (30,   5000),
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
    bonus = np.zeros(base_shape)

    # SST 鋒面加分 (+20%)
    front_strength = ocean_features.get("front_strength")
    if front_strength is not None:
        fs = _safe_resize(front_strength, base_shape)
        bonus += 0.2 * np.nan_to_num(fs, nan=0)

    # FTLE 脊線加分 (+15%)
    ftle = ocean_features.get("ftle")
    if ftle is not None:
        ft = _safe_resize(ftle, base_shape)
        ft_norm = _normalize(ft)
        bonus += 0.15 * ft_norm

    # VIIRS 漁船燈光加分 (+25%)
    viirs = ocean_features.get("viirs_lights")
    if viirs is not None and len(viirs) > 0:
        viirs_map = _viirs_to_grid(viirs, lat_1d, lon_1d)
        if viirs_map.shape == base_shape:
            bonus += 0.25 * viirs_map

    # ── 3. 為每個物種計算最終分數 ──
    species_hotspots = {}

    # 獲取深度數據 (若有)
    bathy_grid = ocean_features.get("bathy")

    for species, hsi in all_species_scores.items():
        # 最終分數 = HSI × (1 + bonus)
        raw_score = hsi * (1 + bonus)

        # [v10.5] Percentile-based rescaling — 消除 1.0 飽和
        # 將原始分數映射到 0.15–0.90 範圍，保留相對排名
        final_score = _percentile_rescale(raw_score, out_min=0.15, out_max=0.90)

        # 提取 Top-N 熱點 (含物種水深驗證)
        hotspots = _extract_hotspots(
            final_score, lat_1d, lon_1d, species, top_n=top_n,
            bathy_grid=bathy_grid,
        )
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
) -> List[Dict[str, Any]]:
    """
    從分數網格中提取最佳漁場熱點

    使用非極大值抑制（NMS）+ 物種水深驗證
    """
    score_filled = np.nan_to_num(score, nan=0)
    depth_range = SPECIES_DEPTH_RANGE.get(species, (10, 8000))

    # 找所有超過閾值的點
    candidates = []
    ny, nx = score_filled.shape
    for iy in range(ny):
        for ix in range(nx):
            if score_filled[iy, ix] >= min_score:
                # [v10.5] 物種水深驗證
                if bathy_grid is not None and iy < bathy_grid.shape[0] and ix < bathy_grid.shape[1]:
                    depth_m = float(-bathy_grid[iy, ix])  # bathy is negative
                    if depth_m < depth_range[0] or depth_m > depth_range[1]:
                        continue  # 水深不適合此物種
                candidates.append({
                    "lat": float(lat[iy]),
                    "lon": float(lon[ix]),
                    "score": float(score_filled[iy, ix]),
                    "iy": iy,
                    "ix": ix,
                })

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
            selected.append({
                "lat": c["lat"],
                "lon": c["lon"],
                "score": c["score"],
                "species": species,
                "rank": len(selected) + 1,
            })
            if len(selected) >= top_n:
                break

    return selected


def _percentile_rescale(
    arr: np.ndarray,
    out_min: float = 0.15,
    out_max: float = 0.90,
) -> np.ndarray:
    """
    Percentile-based rescaling — 將飽和的 HSI 映射到合理範圍。
    使用 5th-95th percentile 的點作為錨定，避免極端值影響。
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

def assess_safety(
    hotspots: List[Dict],
    weather: Optional[Dict] = None,
    current_lat: float = 25.0,
    current_lon: float = 140.0,
) -> List[Dict]:
    """為每個熱點加入安全評估和距離資訊"""
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
        d2 = (lat_g - pt[0])**2 + (lon_g - pt[1])**2
        heat += np.exp(-d2 / (2 * sigma**2))
    mx = heat.max()
    if mx > 0:
        heat /= mx
    return heat
