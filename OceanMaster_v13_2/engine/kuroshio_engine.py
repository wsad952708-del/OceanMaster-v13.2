"""
OceanMaster v13.2 — 黑潮特徵引擎 (Kuroshio Current Engine)
=========================================================
台灣漁業最關鍵的物理海洋特徵引擎。

科學依據:
  - 黑潮主軸位於 SST 梯度最大處 (Jan et al. 2015)
  - 黑潮入侵台灣海峽時帶來暖水+魚群 (Hsin et al. 2013)
  - 黑潮邊緣渦旋是鬼頭刀/旗魚聚集的熱點 (Hsiao et al. 2011)
  - 黑潮流速 0.5-1.5 m/s, 寬度 ~100-150 km (Lien et al. 2014)

提供:
  1. 黑潮主軸位置估計 (SST 梯度 + 海流極大值)
  2. 黑潮入侵強度指標 (台灣海峽南口溫差)
  3. 邊緣暖渦偵測 (黑潮蛇行 → warm filament)
  4. 物種-黑潮交互 HSI 修正
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple

log = logging.getLogger("OceanMaster.Kuroshio")

# ─── 黑潮氣候態軸線 (每月, 緯度→經度) ───────────────────
# 來源: Jan et al. 2015, Lien et al. 2014
# 格式: {lat: lon_of_axis} — 黑潮主軸在該緯度的典型經度
KUROSHIO_CLIMATOLOGY_AXIS = {
    # 呂宋海峽 → 台灣東部 → 東海
    20.0: 121.5,  # 巴士海峽
    21.0: 121.3,
    22.0: 121.0,  # 台灣南端 (鵝鑾鼻)
    23.0: 121.5,  # 台東外海
    24.0: 122.0,  # 花蓮外海
    25.0: 122.5,  # 宜蘭外海
    26.0: 123.0,  # 台灣東北角
    27.0: 124.0,  # 進入東海
    28.0: 125.5,
    29.0: 127.0,
    30.0: 128.5,  # 九州以南
    31.0: 130.0,
    32.0: 131.5,
    33.0: 133.0,
    34.0: 134.5,
    35.0: 136.0,  # 紀伊半島
}

# 黑潮典型半寬 (km) — 兩側距離
KUROSHIO_HALF_WIDTH_KM = 75.0

# 黑潮 SST 跨軸溫差 (°C) — 冷側到暖側
KUROSHIO_SST_CONTRAST = {
    "winter": 4.0,   # 冬季溫差顯著 (Dec-Feb)
    "spring": 3.0,   # Mar-May
    "summer": 1.5,   # 夏季溫差縮小 (Jun-Aug)
    "autumn": 2.5,   # Sep-Nov
}

# 台灣海峽南口座標 (用於入侵指數)
STRAIT_SOUTH_GATE = {
    "warm_point": (22.5, 120.5),  # 黑潮暖水入侵點
    "cold_point": (23.5, 119.5),  # 海峽內冷水點
}


def _haversine_km(lat1, lon1, lat2, lon2):
    """兩點間距離 (km)"""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


class KuroshioEngine:
    """
    黑潮 (Kuroshio Current) 特徵提取器

    Pipeline:
      1. estimate_axis()         — 用 SST 梯度估計黑潮主軸
      2. compute_distance()      — 計算每個網格點到黑潮軸的距離
      3. intrusion_index()       — 台灣海峽黑潮入侵強度
      4. edge_eddies()           — 黑潮邊緣渦旋偵測
      5. species_kuroshio_boost()— 物種特異性加成
    """

    def __init__(self):
        self._axis_lats = np.array(sorted(KUROSHIO_CLIMATOLOGY_AXIS.keys()))
        self._axis_lons = np.array([KUROSHIO_CLIMATOLOGY_AXIS[lat] for lat in self._axis_lats])

    def estimate_axis(
        self,
        sst: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        u_current: Optional[np.ndarray] = None,
        v_current: Optional[np.ndarray] = None,
        adt: Optional[np.ndarray] = None,
        month: int = 1,
        sst_previous: Optional[np.ndarray] = None,  # [v13.2] 48hr 前 SST
    ) -> Dict:
        """
        估計黑潮主軸位置

        方法:
          1. 在每個緯度帶 (±0.5°), 找 SST 東西方向梯度最大的經度
          2. 若有海流資料, 用流速極大值修正
          3. 與氣候態加權平均 (避免單日噪音)

        Returns:
            dict with axis_lats, axis_lons, axis_speed, method
        """
        axis_lons_estimated = []
        axis_speeds = []

        for lat_target in self._axis_lats:
            # 找最近的緯度 index
            lat_idx = np.argmin(np.abs(lats - lat_target))
            if lat_idx < 0 or lat_idx >= sst.shape[0]:
                axis_lons_estimated.append(KUROSHIO_CLIMATOLOGY_AXIS[lat_target])
                axis_speeds.append(1.0)
                continue

            sst_row = sst[lat_idx, :]

            # SST 東西梯度
            if np.sum(np.isfinite(sst_row)) < 5:
                axis_lons_estimated.append(KUROSHIO_CLIMATOLOGY_AXIS[lat_target])
                axis_speeds.append(1.0)
                continue

            sst_filled = np.where(np.isfinite(sst_row), sst_row, np.nanmean(sst_row))
            grad = np.abs(np.gradient(sst_filled))

            # 限制搜索範圍在氣候態 ±3° 內
            clim_lon = KUROSHIO_CLIMATOLOGY_AXIS[lat_target]
            lon_mask = (lons >= clim_lon - 3.0) & (lons <= clim_lon + 3.0)

            if not np.any(lon_mask):
                axis_lons_estimated.append(clim_lon)
                axis_speeds.append(1.0)
                continue

            # 在搜索範圍內找梯度最大
            grad_masked = np.where(lon_mask, grad, 0.0)
            max_grad = np.nanmax(grad_masked) if np.any(grad_masked) else 0.0
            
            method = "sst_gradient"
            estimated_lon = None
            
            # [v13.2] 颱風擾動檢查 — 48hr SST 梯度突變則 fallback 氣候態
            typhoon_unstable = False
            if sst_previous is not None:
                prev_row = sst_previous[lat_idx, :]
                prev_filled = np.where(np.isfinite(prev_row), prev_row, np.nanmean(prev_row))
                prev_grad = np.abs(np.gradient(prev_filled))
                prev_max_grad = np.nanmax(np.where(lon_mask, prev_grad, 0.0)) if np.any(lon_mask) else 0.0
                # 如果梯度從 < 0.05 突變到 > 0.3 → 疑似颱風 cold wake
                if prev_max_grad < 0.05 and max_grad > 0.3:
                    typhoon_unstable = True
                    method = "climatology_typhoon_fallback"
                    estimated_lon = clim_lon
                    log.warning(f"  Kuroshio lat={lat_target}: SST gradient spike "
                                f"({prev_max_grad:.3f}→{max_grad:.3f}), typhoon fallback")

            # [v13.1] 夏季 SST 梯度微弱，切換為 ADT
            if not typhoon_unstable and month in [5, 6, 7, 8, 9] and max_grad < 0.15 and adt is not None:
                adt_row = adt[lat_idx, :]
                adt_masked = np.where(lon_mask, adt_row, 0.0)
                valid_adt = np.where((adt_masked >= 1.0) & (adt_masked <= 1.25))[0]
                if len(valid_adt) > 0:
                    best_adt_idx = valid_adt[np.argmin(np.abs(adt_masked[valid_adt] - 1.15))]
                    sst_lon = lons[best_adt_idx]
                    estimated_lon = 0.8 * sst_lon + 0.2 * clim_lon
                    method = "adt_contour"
            
            if estimated_lon is None:
                best_idx = np.argmax(grad_masked) if max_grad > 0 else np.argmin(np.abs(lons - clim_lon))
                sst_lon = lons[best_idx]
                # [v13.2-audit] SST 梯度 70% + 氣候態 30% 混合 (僅當無 typhoon/ADT fallback 時)
                estimated_lon = 0.7 * sst_lon + 0.3 * clim_lon

            # 若有海流, 用流速修正
            speed = 1.0
            if u_current is not None and v_current is not None:
                u_row = u_current[lat_idx, :]
                v_row = v_current[lat_idx, :]
                speed_row = np.sqrt(u_row ** 2 + v_row ** 2)
                speed_masked = np.where(lon_mask, speed_row, 0.0)
                if np.nanmax(speed_masked) > 0.3:  # 黑潮最小速度 ~0.3 m/s
                    speed_idx = np.nanargmax(speed_masked)
                    current_lon = lons[speed_idx]
                    # 融合: 視主要方法決定權重
                    if method == "adt_contour":
                        estimated_lon = 0.50 * estimated_lon + 0.40 * current_lon + 0.10 * clim_lon
                    else:
                        estimated_lon = 0.50 * estimated_lon + 0.35 * current_lon + 0.15 * clim_lon
                    method += "+current"
                    speed = float(np.nanmax(speed_masked))

            axis_lons_estimated.append(float(estimated_lon))
            axis_speeds.append(speed)

        # [v13.2] 空間一致性約束 — 相鄰緯度間經度跳變 < 1.5°
        for k in range(1, len(axis_lons_estimated)):
            delta = abs(axis_lons_estimated[k] - axis_lons_estimated[k - 1])
            if delta > 1.5:
                # 用線性內插修正
                axis_lons_estimated[k] = (
                    axis_lons_estimated[k - 1] + 
                    np.sign(axis_lons_estimated[k] - axis_lons_estimated[k - 1]) * 1.5
                )
                log.warning(f"  Kuroshio axis spatial jump at lat={self._axis_lats[k]}: "
                            f"delta={delta:.2f}°, clamped to 1.5°")

        return {
            "axis_lats": self._axis_lats.tolist(),
            "axis_lons": axis_lons_estimated,
            "axis_speed": axis_speeds,
            "method": method,
        }

    def compute_distance_grid(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        axis_result: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        計算每個網格點到黑潮主軸的距離 (km)

        Returns:
            2D ndarray (ny, nx) — 距離 (km), 正=東側(暖側), 負=西側(冷側)
        """
        if axis_result is None:
            ax_lats = self._axis_lats
            ax_lons = self._axis_lons
        else:
            ax_lats = np.array(axis_result["axis_lats"])
            ax_lons = np.array(axis_result["axis_lons"])

        ny, nx = len(lats), len(lons)
        distance = np.full((ny, nx), 999.0, dtype=np.float32)

        for i, lat in enumerate(lats):
            # 內插出此緯度的軸線經度
            if lat < ax_lats[0] or lat > ax_lats[-1]:
                continue
            axis_lon = float(np.interp(lat, ax_lats, ax_lons))

            for j, lon in enumerate(lons):
                # 計算到軸線的經度差 (近似距離)
                dlon = lon - axis_lon
                dist_km = dlon * 111.32 * np.cos(np.radians(lat))  # 正=東, 負=西
                distance[i, j] = dist_km

        return distance

    def compute_intrusion_index(
        self,
        sst: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        month: int = 1,
    ) -> Dict:
        """
        計算黑潮入侵台灣海峽的強度指標

        原理:
          黑潮暖水從台灣南端入侵海峽時, 海峽南口 SST 升高
          入侵指數 = (暖點SST - 冷點SST) / 季節典型溫差

        > 1.0 = 強入侵 (冬季常見)
        = 0.5-1.0 = 正常
        < 0.5 = 弱入侵

        Returns:
            dict with intrusion_index, warm_sst, cold_sst, season
        """
        wp = STRAIT_SOUTH_GATE["warm_point"]
        cp = STRAIT_SOUTH_GATE["cold_point"]

        # 找最近網格點
        wi = np.argmin(np.abs(lats - wp[0]))
        wj = np.argmin(np.abs(lons - wp[1]))
        ci = np.argmin(np.abs(lats - cp[0]))
        cj = np.argmin(np.abs(lons - cp[1]))

        warm_sst = float(sst[wi, wj]) if np.isfinite(sst[wi, wj]) else 26.0
        cold_sst = float(sst[ci, cj]) if np.isfinite(sst[ci, cj]) else 22.0

        # 季節
        season = "winter" if month in [12, 1, 2] else \
                 "spring" if month in [3, 4, 5] else \
                 "summer" if month in [6, 7, 8] else "autumn"

        typical_contrast = KUROSHIO_SST_CONTRAST[season]
        actual_contrast = warm_sst - cold_sst
        intrusion_index = actual_contrast / max(typical_contrast, 0.5)

        return {
            "intrusion_index": float(np.clip(intrusion_index, 0.0, 3.0)),
            "warm_sst": warm_sst,
            "cold_sst": cold_sst,
            "actual_contrast": actual_contrast,
            "season": season,
            "typical_contrast": typical_contrast,
        }

    def detect_edge_features(
        self,
        sst: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        distance_grid: np.ndarray,
    ) -> np.ndarray:
        """
        偵測黑潮邊緣特徵 (渦旋/暖絲)

        黑潮邊緣 (距軸 50-150 km) 的 SST 異常高 → 暖渦/暖絲
        這些區域是頂級捕食者的主要覓食區

        Returns:
            2D ndarray (ny, nx) — 邊緣特徵強度 (0-1)
        """
        ny, nx = sst.shape
        edge_score = np.zeros((ny, nx), dtype=np.float32)

        abs_dist = np.abs(distance_grid)

        # 邊緣帶: 距軸 30-150 km
        edge_mask = (abs_dist >= 30) & (abs_dist <= 150)

        if not np.any(edge_mask):
            return edge_score

        # 在邊緣帶中, SST 高於鄰近平均 → 可能是暖渦/暖絲
        try:
            from scipy.ndimage import uniform_filter
            sst_smooth = uniform_filter(
                np.nan_to_num(sst, nan=np.nanmean(sst)), size=5
            )
            sst_anomaly = sst - sst_smooth
            # 正異常 + 邊緣帶 → 暖渦特徵
            warm_anomaly = np.clip(sst_anomaly / 2.0, 0, 1)
            edge_score = np.where(edge_mask, warm_anomaly, 0.0).astype(np.float32)
        except ImportError:
            # 無 scipy — 用簡化邊緣偵測
            # 距軸距離的高斯形邊緣分數
            edge_gaussian = np.exp(-((abs_dist - 80) ** 2) / (2 * 40 ** 2))
            edge_score = np.where(edge_mask, edge_gaussian * 0.5, 0.0).astype(np.float32)

        return edge_score

    def compute_species_boost(
        self,
        distance_grid: np.ndarray,
        species: str,
        intrusion_result: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        計算物種-黑潮交互 HSI 修正 (乘法因子)

        不同物種對黑潮距離的響應:
          - 鬼頭刀: 高親和, 邊緣最佳 (50-100km)
          - 旗魚: 高親和, 軸線附近
          - 黃鰭鮪: 中等, 暖側 (0-100km)
          - 正鰹: 中等, 暖池交匯
          - 秋刀魚: 低親和, 偏好冷側

        Returns:
            2D ndarray — boost factor (0.8-1.5)
        """
        from engine.species_params import KUROSHIO_SPECIES_AFFINITY

        affinity = KUROSHIO_SPECIES_AFFINITY.get(species, 0.3)
        abs_dist = np.abs(distance_grid)

        # 基礎: 距離衰減 (越近黑潮, affinity 高的物種加成越大)
        # 使用高斯: optimal_distance 因物種而異
        if affinity >= 0.8:
            # 高親和 (鬼頭刀/旗魚): 邊緣最佳
            optimal_dist = 60.0
            sigma_dist = 50.0
        elif affinity >= 0.5:
            # 中親和 (黃鰭/正鰹): 軸線附近最佳
            optimal_dist = 30.0
            sigma_dist = 80.0
        else:
            # 低親和 (長鰭/秋刀魚): 距離影響小
            optimal_dist = 100.0
            sigma_dist = 150.0

        dist_score = np.exp(-((abs_dist - optimal_dist) ** 2) / (2 * sigma_dist ** 2))

        # boost = 1.0 + affinity * dist_score * scale
        scale = 0.5  # 最大 boost = 1.0 + 0.9 * 1.0 * 0.5 = 1.45
        boost = 1.0 + affinity * dist_score * scale

        # 入侵強度修正 (入侵越強, 近海物種加成越大)
        if intrusion_result is not None:
            intr = intrusion_result.get("intrusion_index", 1.0)
            if intr > 1.0 and affinity >= 0.5:
                boost *= 1.0 + 0.1 * (intr - 1.0)  # 強入侵額外加成

        return np.clip(boost, 0.8, 1.5).astype(np.float32)

    def analyze_full(
        self,
        sst: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        u_current: Optional[np.ndarray] = None,
        v_current: Optional[np.ndarray] = None,
        adt: Optional[np.ndarray] = None,
        month: int = 1,
    ) -> Dict:
        """
        完整黑潮分析

        Returns:
            dict with:
              - axis: 軸線座標 dict
              - distance_grid: 距離場 (km), 正=東(暖), 負=西(冷)
              - intrusion: 入侵指數 dict
              - edge_features: 邊緣特徵強度 (0-1)
        """
        log.info("  Kuroshio: estimating axis position...")
        axis = self.estimate_axis(sst, lats, lons, u_current, v_current, adt, month)

        log.info("  Kuroshio: computing distance grid...")
        distance_grid = self.compute_distance_grid(lats, lons, axis)

        log.info("  Kuroshio: computing intrusion index...")
        intrusion = self.compute_intrusion_index(sst, lats, lons, month)

        log.info("  Kuroshio: detecting edge features...")
        edge_features = self.detect_edge_features(sst, lats, lons, distance_grid)

        mean_dist = np.nanmean(np.abs(distance_grid[distance_grid < 900]))
        log.info(
            f"  Kuroshio: axis method={axis['method']}, "
            f"mean_dist={mean_dist:.0f}km, "
            f"intrusion={intrusion['intrusion_index']:.2f} ({intrusion['season']}), "
            f"edge_pts={np.sum(edge_features > 0.3)}"
        )

        return {
            "axis": axis,
            "distance_grid": distance_grid,
            "intrusion": intrusion,
            "edge_features": edge_features,
        }
