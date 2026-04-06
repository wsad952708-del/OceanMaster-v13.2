"""
OceanMaster v10.3 — 氣壓場特徵提取
=====================================
Task #14: 基於 pressure_msl 計算氣壓梯度 + 低壓槽距離

業務邏輯:
  距低壓槽 100-200km = 魚群聚集區 (上升流增強 → 營養鹽湧升)
  氣壓梯度 > 2 hPa/度 = 強風帶 (風生混合 → CHL 增加)

輸出特徵 (可直接加入 ML Pipeline):
  - pressure_gradient: 2D (hPa/deg)
  - distance_to_low_pressure: 2D (km)
  - low_pressure_bonus: 2D (0-1, 100-200km 內加分)
"""

import logging
import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("OceanMaster.Pressure")


def compute_pressure_features(
    pressure_msl: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    low_pressure_threshold: float = 1005.0,
) -> Dict[str, Any]:
    """
    從海平面氣壓場提取漁業相關特徵。

    Args:
        pressure_msl: 2D (ny, nx) 海平面氣壓 (hPa)
        lats: 1D (ny,) 緯度
        lons: 1D (nx,) 經度
        low_pressure_threshold: 低壓中心判定閾值 (hPa)

    Returns:
        {
            'pressure_gradient': 2D (hPa/deg) — 氣壓梯度大小
            'distance_to_low_pressure': 2D (km) — 距最近低壓中心
            'low_pressure_bonus': 2D (0-1) — 100-200km 加分
            'low_pressure_centers': List[(lat, lon, pressure)] — 低壓中心們
            'gradient_direction_u': 2D — 梯度 x 分量
            'gradient_direction_v': 2D — 梯度 y 分量
        }
    """
    ny, nx = pressure_msl.shape

    # ── 1. 氣壓梯度 (Sobel-like finite difference) ──
    grad_y, grad_x = _pressure_gradient(pressure_msl, lats, lons)
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

    # ── 2. 低壓中心偵測 ──
    low_centers = _detect_low_pressure_centers(
        pressure_msl, lats, lons, low_pressure_threshold
    )

    # ── 3. 距低壓中心距離 ──
    dist_grid = _distance_to_centers(lats, lons, low_centers)

    # ── 4. 低壓漁場加分 (100-200km 為最適區) ──
    bonus = _low_pressure_bonus(dist_grid)

    log.info(
        f"  Pressure: gradient=[{np.nanmin(gradient_mag):.2f},"
        f"{np.nanmax(gradient_mag):.2f}] hPa/deg, "
        f"{len(low_centers)} low-P centers, "
        f"bonus>0.5: {int(np.sum(bonus > 0.5))} cells"
    )

    return {
        "pressure_gradient": gradient_mag.astype(np.float32),
        "distance_to_low_pressure": dist_grid.astype(np.float32),
        "low_pressure_bonus": bonus.astype(np.float32),
        "low_pressure_centers": low_centers,
        "gradient_direction_u": grad_x.astype(np.float32),
        "gradient_direction_v": grad_y.astype(np.float32),
    }


def _pressure_gradient(
    p: np.ndarray, lats: np.ndarray, lons: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    氣壓梯度 (hPa/deg)。
    使用 central difference，邊界使用 forward/backward。
    """
    ny, nx = p.shape
    dlat = np.diff(lats).mean() if len(lats) > 1 else 1.0
    dlon = np.diff(lons).mean() if len(lons) > 1 else 1.0

    # Y (lat) 方向
    grad_y = np.zeros_like(p)
    if ny > 2:
        grad_y[1:-1, :] = (p[2:, :] - p[:-2, :]) / (2 * dlat)
        grad_y[0, :] = (p[1, :] - p[0, :]) / dlat
        grad_y[-1, :] = (p[-1, :] - p[-2, :]) / dlat
    elif ny == 2:
        grad_y[:, :] = (p[1, :] - p[0, :]) / dlat

    # X (lon) 方向
    grad_x = np.zeros_like(p)
    if nx > 2:
        grad_x[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2 * dlon)
        grad_x[:, 0] = (p[:, 1] - p[:, 0]) / dlon
        grad_x[:, -1] = (p[:, -1] - p[:, -2]) / dlon
    elif nx == 2:
        grad_x[:, :] = (p[:, 1] - p[:, 0]) / dlon

    return grad_y, grad_x


def _detect_low_pressure_centers(
    p: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    threshold: float,
    min_distance_deg: float = 2.0,
) -> List[Tuple[float, float, float]]:
    """
    偵測低壓中心 (local minimum < threshold)。
    使用 NMS 避免過於密集。
    """
    ny, nx = p.shape
    candidates = []

    for iy in range(1, ny - 1):
        for ix in range(1, nx - 1):
            val = p[iy, ix]
            if val > threshold:
                continue
            # 檢查是否為 3×3 局部最小值
            window = p[max(0, iy-1):iy+2, max(0, ix-1):ix+2]
            if val <= np.nanmin(window):
                candidates.append((float(lats[iy]), float(lons[ix]), float(val)))

    # Sort by pressure (lowest first)
    candidates.sort(key=lambda x: x[2])

    # NMS
    selected = []
    for c in candidates:
        too_close = False
        for s in selected:
            dist = math.sqrt((c[0] - s[0])**2 + (c[1] - s[1])**2)
            if dist < min_distance_deg:
                too_close = True
                break
        if not too_close:
            selected.append(c)

    return selected


def _distance_to_centers(
    lats: np.ndarray,
    lons: np.ndarray,
    centers: List[Tuple[float, float, float]],
) -> np.ndarray:
    """每個網格點到最近低壓中心的距離 (km)"""
    ny, nx = len(lats), len(lons)
    dist = np.full((ny, nx), 9999.0, dtype=np.float32)

    if not centers:
        return dist

    for c_lat, c_lon, _ in centers:
        for iy in range(ny):
            for ix in range(nx):
                d = _haversine_km(lats[iy], lons[ix], c_lat, c_lon)
                dist[iy, ix] = min(dist[iy, ix], d)

    return dist


def _low_pressure_bonus(
    dist: np.ndarray,
    optimal_min_km: float = 100.0,
    optimal_max_km: float = 200.0,
    decay_km: float = 100.0,
) -> np.ndarray:
    """
    低壓漁場加分:
      100-200km = 1.0 (最適區)
      <100km = 線性衰減 (太近，風太強)
      200-300km = 線性衰減
      >300km = 0
    """
    bonus = np.zeros_like(dist)

    # 最適區
    optimal_mask = (dist >= optimal_min_km) & (dist <= optimal_max_km)
    bonus[optimal_mask] = 1.0

    # 太近 (< 100km)
    near_mask = dist < optimal_min_km
    bonus[near_mask] = np.clip(dist[near_mask] / optimal_min_km, 0, 1)

    # 遠處衰減 (200-300km)
    far_mask = (dist > optimal_max_km) & (dist <= optimal_max_km + decay_km)
    bonus[far_mask] = np.clip(
        1.0 - (dist[far_mask] - optimal_max_km) / decay_km,
        0, 1,
    )

    return bonus.astype(np.float32)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """大圓距離"""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
