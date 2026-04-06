"""
OceanMaster — VIIRS 漁船燈光驗證模組
=====================================
用衛星夜間燈光來驗證漁場預測的準確性。

原理:
  集魚燈漁船（如魷魚船、秋刀魚船）夜間開燈吸引魚群，
  VIIRS Day-Night Band 可偵測到這些燈光。
  漁船聚集的地方 = 真實漁場 = 我們的「免費地面真相」。

數據源:
  1. Global Fishing Watch — VIIRS Boat Detection API
  2. EOG (Earth Observation Group) — VIIRS VBD 下載
  3. OceanDataFetcher 已有的 viirs_lights 數據

使用方式:
  只作為驗證層(不影響 HSI 預測)，計算:
  - 預測熱點半徑 50nm 內的漁船數量
  - 預測可信度評分調整
"""

import numpy as np
import logging
from typing import Dict, List, Optional

log = logging.getLogger("OceanMaster.VesselLights")


class VesselLightValidator:
    """
    漁船燈光驗證器

    使用 VIIRS 夜間燈光數據驗證漁場預測。

    使用:
        validator = VesselLightValidator()
        result = validator.validate(hotspots, viirs_lights)
        # result['validation_score'] → 預測可信度
    """

    def __init__(self, search_radius_nm: float = 50.0):
        """
        Parameters:
            search_radius_nm: 搜索半徑 (海里)，預設 50nm
        """
        self.search_radius_deg = search_radius_nm / 60.0  # 1° ≈ 60 nm

    def validate(self, hotspots: List[Dict],
                 viirs_lights: np.ndarray) -> Dict:
        """
        驗證熱點預測

        Parameters:
            hotspots: 預測熱點列表 [{lat, lon, hsi, species, ...}]
            viirs_lights: Nx3 array (lat, lon, brightness)

        Returns:
            dict with:
              - total_vessels: 區域內總漁船數
              - validated_spots: 有漁船附近的熱點數
              - validation_score: 0-1 驗證分數
              - spots_with_validation: 帶驗證資訊的熱點列表
        """
        if viirs_lights is None or len(viirs_lights) == 0:
            log.info("  無 VIIRS 燈光數據，跳過驗證")
            return {
                "total_vessels": 0,
                "validated_spots": 0,
                "validation_score": None,
                "spots_with_validation": hotspots,
            }

        # 形狀驗證: 確保是 Nx3 或 Nx2 array
        viirs_lights = np.atleast_2d(np.asarray(viirs_lights, dtype=np.float64))
        if viirs_lights.ndim != 2 or viirs_lights.shape[1] < 2:
            log.warning(f"  VIIRS 數據格式異常: shape={viirs_lights.shape}，跳過驗證")
            return {
                "total_vessels": 0,
                "validated_spots": 0,
                "validation_score": None,
                "spots_with_validation": hotspots,
            }

        total_vessels = len(viirs_lights)
        validated_count = 0
        enhanced_spots = []

        for spot in hotspots:
            lat = spot.get("lat", 0)
            lon = spot.get("lon", 0)

            # 搜索半徑內的漁船
            nearby = self._count_nearby(lat, lon, viirs_lights)

            spot_enhanced = dict(spot)
            spot_enhanced["vessel_count_50nm"] = nearby
            spot_enhanced["vessel_validated"] = nearby > 0

            # 調整信心度
            base_conf = spot.get("confidence", 0.5)
            if nearby >= 5:
                spot_enhanced["confidence_adjusted"] = min(base_conf * 1.3, 1.0)
            elif nearby >= 1:
                spot_enhanced["confidence_adjusted"] = min(base_conf * 1.1, 1.0)
            else:
                spot_enhanced["confidence_adjusted"] = base_conf * 0.9

            if nearby > 0:
                validated_count += 1

            enhanced_spots.append(spot_enhanced)

        if len(hotspots) > 0:
            validation_score = validated_count / len(hotspots)
        else:
            validation_score = 0

        log.info(f"  VIIRS 驗證: {total_vessels} 漁船, "
                 f"{validated_count}/{len(hotspots)} 熱點有漁船附近 "
                 f"(score={validation_score:.2f})")

        return {
            "total_vessels": total_vessels,
            "validated_spots": validated_count,
            "validation_score": validation_score,
            "spots_with_validation": enhanced_spots,
        }

    def _count_nearby(self, lat: float, lon: float,
                      viirs_lights: np.ndarray) -> int:
        """計算搜索半徑內的漁船數量"""
        if len(viirs_lights) == 0:
            return 0

        dlat = viirs_lights[:, 0] - lat
        dlon = (viirs_lights[:, 1] - lon) * np.cos(np.radians(lat))
        dist = np.sqrt(dlat**2 + dlon**2)

        return int(np.sum(dist <= self.search_radius_deg))

    def compute_spatial_correlation(self, hsi_grid: np.ndarray,
                                     viirs_lights: np.ndarray,
                                     lats: np.ndarray,
                                     lons: np.ndarray) -> float:
        """
        計算 HSI 場與漁船分布的空間相關性

        將漁船位置投影到 HSI 網格上，計算 Pearson 相關係數。
        相關性越高 → 預測越準確。

        Returns:
            correlation: -1 到 1 的相關係數
        """
        if viirs_lights is None or len(viirs_lights) == 0:
            return 0.0

        ny, nx = hsi_grid.shape

        # 將漁船投影到網格
        vessel_density = np.zeros((ny, nx), dtype=np.float32)

        for v in viirs_lights:
            vlat, vlon = v[0], v[1]
            j = np.argmin(np.abs(lats - vlat))
            i = np.argmin(np.abs(lons - vlon))
            if 0 <= j < ny and 0 <= i < nx:
                vessel_density[j, i] += 1

        # 高斯平滑 (船不會剛好在一個格點上)
        try:
            from scipy.ndimage import gaussian_filter
            vessel_density = gaussian_filter(vessel_density, sigma=2)
        except ImportError:
            pass

        # Pearson 相關
        hsi_flat = hsi_grid.flatten()
        vd_flat = vessel_density.flatten()

        mask = ~(np.isnan(hsi_flat) | np.isnan(vd_flat))
        if np.sum(mask) < 10:
            return 0.0

        hsi_f = hsi_flat[mask]
        vd_f = vd_flat[mask]

        if np.std(hsi_f) < 1e-10 or np.std(vd_f) < 1e-10:
            return 0.0

        correlation = np.corrcoef(hsi_f, vd_f)[0, 1]

        log.info(f"  HSI-漁船空間相關: r={correlation:.3f}")
        return float(correlation)
