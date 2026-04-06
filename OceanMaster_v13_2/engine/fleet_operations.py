"""
OceanMaster v13.2 — 船隊作業模組 (Fleet Operations)
====================================================
[v14.0] 兩大功能:

1. 海流剪切預警 (Current Shear Warning)
   - 計算表層 vs 300m 深層海流向量差
   - 剪切強度 > 閾值 → 幹繩纏繞警告
   - 科學依據: Ward & Hindmarsh (2007) — longline tangling in shear zones

2. TSP 最佳巡航路線 (Optimal Patrol Route)
   - 取 Top-N 高分熱點 (預設 7)
   - Nearest Neighbor 啟發式
   - 輸出 Day 1~N 順序 + 總距離
   - GeoJSON 航線用於地圖顯示
"""

import numpy as np
import math
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone

log = logging.getLogger("OceanMaster.Fleet")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. 海流剪切預警 (Current Shear)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 剪切閾值 (m/s) — 超過此值警告幹繩纏繞風險
SHEAR_WARN_THRESHOLD = 0.15   # 中等風險
SHEAR_DANGER_THRESHOLD = 0.30  # 高風險 (強烈不建議作業)


def compute_current_shear(
    u_surface: Optional[np.ndarray],
    v_surface: Optional[np.ndarray],
    u_deep: Optional[np.ndarray],
    v_deep: Optional[np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
) -> Optional[Dict[str, np.ndarray]]:
    """
    計算表層 vs 深層海流剪切強度

    剪切 = |V_surface - V_deep| (速度向量差的模)
    物理意義: 高剪切 = 不同深度水流方向/速度差異大
               → 延繩下水後容易被扭曲/纏繞

    Args:
        u_surface, v_surface: 表層海流 (m/s), 2D grids
        u_deep, v_deep: 深層海流 (m/s, 通常 200-300m), 2D grids
        lats, lons: 座標

    Returns:
        dict with:
            shear_magnitude: 2D, 剪切強度 (m/s)
            shear_direction: 2D, 剪切方向 (degrees)
            risk_level: 2D, 0=安全, 1=警告, 2=危險
    """
    if u_surface is None or v_surface is None:
        log.info("  海流剪切: 無表層海流數據")
        return None

    # 如果沒有深層數據, 用表層數據的 30% 作為粗估深層
    if u_deep is None or v_deep is None:
        log.info("  海流剪切: 無深層數據, 使用表層 30% 估算")
        u_deep = u_surface * 0.3
        v_deep = v_surface * 0.3

    # 確保形狀一致
    shape = u_surface.shape
    if u_deep.shape != shape:
        u_deep = _resize_to(u_deep, shape)
        v_deep = _resize_to(v_deep, shape)

    # 剪切向量 = surface - deep
    du = np.nan_to_num(u_surface, nan=0) - np.nan_to_num(u_deep, nan=0)
    dv = np.nan_to_num(v_surface, nan=0) - np.nan_to_num(v_deep, nan=0)

    # 剪切強度 (m/s)
    shear_mag = np.sqrt(du ** 2 + dv ** 2).astype(np.float32)

    # 剪切方向 (degrees, 0=N, 90=E)
    shear_dir = np.degrees(np.arctan2(du, dv)).astype(np.float32)
    shear_dir[shear_dir < 0] += 360

    # 風險等級
    risk = np.zeros_like(shear_mag, dtype=np.int8)
    risk[shear_mag >= SHEAR_WARN_THRESHOLD] = 1
    risk[shear_mag >= SHEAR_DANGER_THRESHOLD] = 2

    n_warn = int(np.sum(risk == 1))
    n_danger = int(np.sum(risk == 2))
    log.info(f"  海流剪切: max={shear_mag.max():.3f} m/s, "
             f"警告={n_warn} cells, 危險={n_danger} cells")

    return {
        "shear_magnitude": shear_mag,
        "shear_direction": shear_dir,
        "risk_level": risk,
    }


def enrich_hotspots_with_shear(
    hotspots: List[Dict],
    shear_data: Optional[Dict[str, np.ndarray]],
    lats: np.ndarray,
    lons: np.ndarray,
) -> List[Dict]:
    """為每個熱點加入海流剪切資訊"""
    if shear_data is None:
        return hotspots

    shear_mag = shear_data["shear_magnitude"]
    risk_grid = shear_data["risk_level"]

    for h in hotspots:
        li = int(np.argmin(np.abs(lats - h["lat"])))
        lj = int(np.argmin(np.abs(lons - h["lon"])))

        if li < shear_mag.shape[0] and lj < shear_mag.shape[1]:
            mag = float(shear_mag[li, lj])
            risk = int(risk_grid[li, lj])
            h["current_shear_ms"] = round(mag, 3)
            h["shear_risk_level"] = risk

            if risk == 2:
                h["shear_warning"] = (
                    f"🔴 海流剪切 {mag:.2f} m/s — 強烈扭力，"
                    f"幹繩纏繞風險極高，建議迴避此區"
                )
            elif risk == 1:
                h["shear_warning"] = (
                    f"🟡 海流剪切 {mag:.2f} m/s — 中等扭力，"
                    f"延繩投放需注意繩張方向"
                )

    return hotspots


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2. TSP 最佳巡航路線 (Nearest Neighbor)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """兩點間大圓距離 (海浬)"""
    R = 3440.065  # 地球半徑 (海浬)
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def compute_optimal_route(
    hotspots: List[Dict],
    vessel_lat: float = 22.61,
    vessel_lon: float = 120.28,
    max_stops: int = 7,
    cruise_speed_kt: float = 10.0,
) -> Dict[str, Any]:
    """
    Nearest Neighbor TSP — 最短串聯巡航路線

    從船舶當前位置出發，依序前往最近的未訪熱點，
    直到訪完 max_stops 個熱點。

    Args:
        hotspots: 按 HSI 排序的熱點列表
        vessel_lat, vessel_lon: 船舶位置
        max_stops: 最多停靠點 (預設 7)
        cruise_speed_kt: 巡航速度 (節)

    Returns:
        dict:
            route: [{day, lat, lon, species, score, distance_nm, eta_hours}, ...]
            total_distance_nm: 總距離
            total_hours: 總航時
            geojson: GeoJSON LineString
    """
    if not hotspots:
        return {"route": [], "total_distance_nm": 0, "total_hours": 0, "geojson": None}

    # 取前 max_stops 個高分熱點作為候選
    candidates = hotspots[:max_stops]

    # Nearest Neighbor 演算法
    visited = []
    remaining = list(range(len(candidates)))
    current_lat, current_lon = vessel_lat, vessel_lon
    total_dist = 0

    while remaining:
        # 找最近的未訪候選點
        best_idx = None
        best_dist = float('inf')
        for idx in remaining:
            h = candidates[idx]
            d = _haversine_nm(current_lat, current_lon, h["lat"], h["lon"])
            if d < best_dist:
                best_dist = d
                best_idx = idx

        if best_idx is None:
            break

        remaining.remove(best_idx)
        total_dist += best_dist
        h = candidates[best_idx]
        current_lat, current_lon = h["lat"], h["lon"]

        visited.append({
            "day": len(visited) + 1,
            "lat": h["lat"],
            "lon": h["lon"],
            "species": h.get("species", "unknown"),
            "score": round(h.get("score", 0), 3),
            "species_name": h.get("species", "unknown"),
            "leg_distance_nm": round(best_dist, 1),
            "cumulative_distance_nm": round(total_dist, 1),
            "eta_hours": round(total_dist / cruise_speed_kt, 1),
        })

    total_hours = total_dist / cruise_speed_kt

    # GeoJSON LineString
    coordinates = [[vessel_lon, vessel_lat]]  # 起點
    for stop in visited:
        coordinates.append([stop["lon"], stop["lat"]])

    geojson = {
        "type": "Feature",
        "properties": {
            "type": "optimized_route",
            "total_distance_nm": round(total_dist, 1),
            "total_hours": round(total_hours, 1),
            "stops": len(visited),
        },
        "geometry": {
            "type": "LineString",
            "coordinates": coordinates,
        },
    }

    log.info(f"  TSP 巡航路線: {len(visited)} 站, "
             f"總距離 {total_dist:.0f} 海浬, "
             f"約 {total_hours:.0f} 小時")

    return {
        "route": visited,
        "total_distance_nm": round(total_dist, 1),
        "total_hours": round(total_hours, 1),
        "geojson": geojson,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  工具函數
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _resize_to(arr: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """安全 resize — 取共同尺寸"""
    result = np.zeros(target_shape)
    ny = min(arr.shape[0], target_shape[0])
    nx = min(arr.shape[1], target_shape[1])
    result[:ny, :nx] = arr[:ny, :nx]
    return result
