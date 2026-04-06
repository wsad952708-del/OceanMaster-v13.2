"""
OceanMaster v13.2 — 熱點降尺度引擎 (Hotspot Downscaler)
========================================================
[v13.5] 利用高解析度衛星數據對粗網格熱點進行次網格定位
[v13.9] 新增陸地驗證 — 移動後的座標必須在海洋上

問題: WCPFC 5°×5° 訓練數據 + 0.25° 操作網格 → 熱點定位精度不足
方案: 在每個熱點的 ±0.5° 鄰域內, 使用以下衛星特徵微調座標:

1. SST 梯度鋒面 — 溫度鋒面 = 魚群聚集帶
2. 葉綠素峰值 — 高 Chl = 餌料豐富
3. SSH 渦旋邊緣 — 冷渦邊界 = 上升流
4. 加權疊加 → 在鄰域內找最佳次網格「海洋」點

精確度等級:
  A: 3層衛星數據校正 (SST+Chl+SSH) — 精度 ~0.1°
  B: 2層衛星數據校正 (SST+Chl 或 SST+SSH) — 精度 ~0.25°
  C: 1層或無衛星校正 — 原始網格精度 ~0.5°
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional

log = logging.getLogger("OceanMaster.Downscaler")

# [v13.9] 陸地驗證
try:
    from engine.land_mask import near_known_land, FUSION_BUFFER_DEG
    LAND_MASK_OK = True
except ImportError:
    LAND_MASK_OK = False
    log.warning("land_mask not available — downscaler cannot validate land/ocean")


def _is_ocean(lat: float, lon: float) -> bool:
    """檢查座標是否為海洋 (非陸地)"""
    if not LAND_MASK_OK:
        return True  # 無法驗證時假設為海洋
    return not near_known_land(lat, lon, buffer_deg=FUSION_BUFFER_DEG)


def downscale_hotspots(
    hotspots: List[Dict],
    sst: Optional[np.ndarray],
    chl: Optional[np.ndarray],
    ssh: Optional[np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
    search_radius_deg: float = 0.5,
) -> List[Dict]:
    """
    對每個熱點做次網格降尺度精修

    [v13.9] 所有精修後的座標必須通過陸地驗證:
    1. 嘗試移動到衛星最佳點
    2. 如果最佳點在陸地上 → 在候選點中找最近的海洋點
    3. 如果都在陸地 → 退回原始座標
    4. 絕對不能輸出陸地座標
    """
    if not hotspots:
        return hotspots

    # 計算可用的衛星圖層
    layers_available = 0
    sst_gradient = None
    chl_norm = None
    ssh_edge = None

    if sst is not None and sst.size > 0:
        sst_gradient = _compute_sst_gradient(sst)
        layers_available += 1

    if chl is not None and chl.size > 0:
        chl_norm = _normalize_chl(chl)
        layers_available += 1

    if ssh is not None and ssh.size > 0:
        ssh_edge = _compute_ssh_edge(ssh)
        layers_available += 1

    if layers_available == 0:
        log.info("  降尺度: 無衛星數據可用, 保持原始座標")
        for h in hotspots:
            h["precision_grade"] = "C"
            h["precision_label"] = "原始網格 (~0.5°)"
        return hotspots

    # 確定精確度等級
    grade = "A" if layers_available >= 3 else ("B" if layers_available >= 2 else "C")
    precision_label = {
        "A": "高解析度衛星校正 (~0.1°)",
        "B": "衛星部分校正 (~0.25°)",
        "C": "原始網格 (~0.5°)",
    }

    n_refined = 0
    n_land_rejected = 0
    for h in hotspots:
        orig_lat, orig_lon = h["lat"], h["lon"]

        # [v13.9] 原始座標也必須在海洋上 (防禦性檢查)
        if not _is_ocean(orig_lat, orig_lon):
            log.warning(f"  ❌ 原始座標 ({orig_lat:.2f},{orig_lon:.2f}) 在陸地! 跳過降尺度")
            h["precision_grade"] = "C"
            h["precision_label"] = precision_label["C"]
            continue

        # 找搜尋窗口在網格中的索引範圍
        lat_mask = (lats >= orig_lat - search_radius_deg) & (lats <= orig_lat + search_radius_deg)
        lon_mask = (lons >= orig_lon - search_radius_deg) & (lons <= orig_lon + search_radius_deg)

        i_indices = np.where(lat_mask)[0]
        j_indices = np.where(lon_mask)[0]

        if len(i_indices) == 0 or len(j_indices) == 0:
            h["precision_grade"] = grade
            h["precision_label"] = precision_label[grade]
            continue

        i_min, i_max = i_indices[0], i_indices[-1] + 1
        j_min, j_max = j_indices[0], j_indices[-1] + 1

        # 加權評分
        combined_score = np.zeros((i_max - i_min, j_max - j_min))

        if sst_gradient is not None:
            sub = _safe_sub(sst_gradient, i_min, i_max, j_min, j_max)
            if sub is not None:
                combined_score += 0.4 * sub

        if chl_norm is not None:
            sub = _safe_sub(chl_norm, i_min, i_max, j_min, j_max)
            if sub is not None:
                combined_score += 0.3 * sub

        if ssh_edge is not None:
            sub = _safe_sub(ssh_edge, i_min, i_max, j_min, j_max)
            if sub is not None:
                combined_score += 0.3 * sub

        # 距離衰減 (離原始點越遠, 得分越低)
        local_lats = lats[i_min:i_max]
        local_lons = lons[j_min:j_max]
        lat_grid, lon_grid = np.meshgrid(local_lats, local_lons, indexing='ij')
        dist = np.sqrt((lat_grid - orig_lat) ** 2 + (lon_grid - orig_lon) ** 2)
        dist_weight = np.exp(-dist ** 2 / (2 * (search_radius_deg * 0.6) ** 2))
        combined_score *= dist_weight

        # [v13.9] 陸地遮罩 — 把陸地點的分數設為 -inf
        if LAND_MASK_OK:
            for li in range(len(local_lats)):
                for lj in range(len(local_lons)):
                    if not _is_ocean(float(local_lats[li]), float(local_lons[lj])):
                        combined_score[li, lj] = -999

        # 找最佳海洋點
        if combined_score.size > 0 and np.max(combined_score) > 0:
            best_idx = np.unravel_index(np.argmax(combined_score), combined_score.shape)
            new_lat = float(local_lats[best_idx[0]])
            new_lon = float(local_lons[best_idx[1]])

            # [v13.9] 最終驗證: 確認最佳點確實在海洋
            if not _is_ocean(new_lat, new_lon):
                n_land_rejected += 1
                log.debug(f"  🏔 降尺度最佳點 ({new_lat:.2f},{new_lon:.2f}) 在陸地, 保留原始座標")
                h["precision_grade"] = "C"
                h["precision_label"] = "陸地迴避 (原始網格)"
                continue

            # 計算偏移量
            offset_deg = np.sqrt((new_lat - orig_lat) ** 2 + (new_lon - orig_lon) ** 2)

            if offset_deg > 0.01:  # 至少偏移 0.01° 才算有意義
                h["lat"] = round(new_lat, 3)
                h["lon"] = round(new_lon, 3)
                h["downscale_offset_deg"] = round(offset_deg, 3)
                h["original_lat"] = orig_lat
                h["original_lon"] = orig_lon
                n_refined += 1

        h["precision_grade"] = grade
        h["precision_label"] = precision_label[grade]

    if n_land_rejected > 0:
        log.warning(f"  🏔 降尺度: {n_land_rejected} 個候選點因陸地而被拒")
    log.info(f"  降尺度: {n_refined}/{len(hotspots)} 個熱點精修 "
             f"(精度: {grade}, {layers_available} 衛星層)")

    return hotspots


def _compute_sst_gradient(sst: np.ndarray) -> np.ndarray:
    """SST gradient magnitude → 鋒面強度 (標準化 0~1)"""
    mean_val = np.nanmean(sst) if np.any(np.isfinite(sst)) else 25.0
    sst_clean = np.nan_to_num(sst, nan=mean_val)
    dy = np.gradient(sst_clean, axis=0)
    dx = np.gradient(sst_clean, axis=1)
    grad = np.sqrt(dy ** 2 + dx ** 2)
    p95 = np.percentile(grad[grad > 0], 95) if np.any(grad > 0) else 0.01
    return np.clip(grad / max(p95, 0.001), 0, 1).astype(np.float32)


def _normalize_chl(chl: np.ndarray) -> np.ndarray:
    """葉綠素濃度標準化到 0~1 (log-scale)"""
    chl_clean = np.nan_to_num(chl, nan=0)
    chl_pos = np.maximum(chl_clean, 0.01)
    chl_log = np.log10(chl_pos)
    chl_min, chl_max = np.percentile(chl_log, [5, 95])
    if chl_max - chl_min < 0.01:
        return np.zeros_like(chl, dtype=np.float32)
    return np.clip((chl_log - chl_min) / (chl_max - chl_min), 0, 1).astype(np.float32)


def _compute_ssh_edge(ssh: np.ndarray) -> np.ndarray:
    """SSH gradient → 渦旋邊緣 (標準化 0~1)"""
    ssh_clean = np.nan_to_num(ssh, nan=0)
    dy = np.gradient(ssh_clean, axis=0)
    dx = np.gradient(ssh_clean, axis=1)
    grad = np.sqrt(dy ** 2 + dx ** 2)
    p95 = np.percentile(grad[grad > 0], 95) if np.any(grad > 0) else 0.01
    return np.clip(grad / max(p95, 0.001), 0, 1).astype(np.float32)


def _safe_sub(arr: np.ndarray, i0: int, i1: int, j0: int, j1: int) -> Optional[np.ndarray]:
    """安全取子陣列"""
    if arr is None:
        return None
    ny, nx = arr.shape
    i1 = min(i1, ny)
    j1 = min(j1, nx)
    if i0 >= i1 or j0 >= j1:
        return None
    sub = arr[i0:i1, j0:j1].copy()
    return np.nan_to_num(sub, nan=0)
