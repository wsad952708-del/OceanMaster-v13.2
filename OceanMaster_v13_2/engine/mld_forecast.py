"""
OceanMaster v13.2 — 混合層深度預報
====================================
Task #17: CMEMS 5-day MLD Forecast

數據源:
  1. CMEMS PHY_001_024 (5-day forecast, via copernicusmarine)
  2. 現有 HYCOM temp_3d → 計算 MLD (fallback)
  3. 氣候態 MLD (seasonal average)

輸出:
  mld_forecast[day0..day4]: 未來 5 天 MLD (m)
"""

import asyncio
import json
import logging
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from engine.base_fetcher import BaseFetcher, DataSource

log = logging.getLogger("OceanMaster.MLD")

# 全球混合層深度氣候態 (m, 按月)
MLD_CLIMATOLOGY = {
    1:  80, 2:  90, 3:  75, 4:  50,
    5:  30, 6:  20, 7:  15, 8:  15,
    9:  20, 10: 30, 11: 50, 12: 70,
}


class MLDForecaster(BaseFetcher):
    """混合層深度預報引擎"""

    def __init__(self):
        super().__init__()

    async def fetch_mld_forecast(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        lats: np.ndarray,
        lons: np.ndarray,
        temp_3d: Optional[np.ndarray] = None,
        sst: Optional[np.ndarray] = None,
        month: int = 1,
    ) -> Dict[str, Any]:
        """
        取得 5 天 MLD 預報。

        Cascade:
          1. CMEMS PHY_001_024 forecast (if copernicusmarine available)
          2. HYCOM temp_3d → 計算密度基準 MLD
          3. 氣候態

        Returns:
            {
                'mld_current': 2D (m) — 目前 MLD
                'mld_forecast': Dict[int, 2D] — {day: MLD array}
                'source': str
            }
        """
        ny, nx = len(lats), len(lons)

        # ─── 1. CMEMS Forecast ───
        cmems_data = await self._try_cmems_forecast(lat_range, lon_range, lats, lons)
        if cmems_data is not None:
            return cmems_data

        # ─── 2. HYCOM temp_3d → MLD ───
        if temp_3d is not None:
            mld = self._compute_mld_from_temp3d(temp_3d)
            if mld is not None:
                # 簡單持續預報 (衰減向氣候態)
                clim = float(MLD_CLIMATOLOGY.get(month, 40))
                forecast = {}
                for d in range(5):
                    alpha = d / 7.0  # 衰減因子
                    forecast[d] = ((1 - alpha) * mld + alpha * clim).astype(np.float32)

                log.info(f"  MLD: from HYCOM, current={np.nanmean(mld):.0f}m")
                return {
                    "mld_current": mld.astype(np.float32),
                    "mld_forecast": forecast,
                    "source": "HYCOM-derived",
                }

        # ─── 3. 氣候態 ───
        clim_val = float(MLD_CLIMATOLOGY.get(month, 40))
        mld_clim = np.full((ny, nx), clim_val, dtype=np.float32)

        log.info(f"  MLD: climatology={clim_val:.0f}m (month={month})")
        return {
            "mld_current": mld_clim,
            "mld_forecast": {d: mld_clim.copy() for d in range(5)},
            "source": "CLIM",
        }

    async def _try_cmems_forecast(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> Optional[Dict]:
        """嘗試 CMEMS PHY_001_024 MLD forecast"""
        try:
            import os
            if not os.environ.get("CMEMS_PASS"):
                return None

            # 使用 copernicusmarine 庫取得 MLD
            # [v13.2] CMEMS MLD 預報尚未實作。
            # 回退行為: 返回 None → 呼叫端使用 _compute_mld_from_temp3d() 從 HYCOM 3D 溫度計算。
            # 未來: 當 copernicusmarine 庫穩定後，可直接取得 CMEMS 的 MLD 分析/預報產品。
            log.debug("  CMEMS MLD forecast: not yet implemented, using temp3d fallback")
            return None

        except ImportError:
            return None

    @staticmethod
    def _compute_mld_from_temp3d(
        temp_3d: np.ndarray,
        depth_levels: Optional[np.ndarray] = None,
        delta_t: float = 0.5,
    ) -> Optional[np.ndarray]:
        """
        從 3D 溫度剖面計算 MLD (溫度差基準法)。
        MLD = SST - delta_t 的深度

        Args:
            temp_3d: (n_depths, ny, nx)
            depth_levels: 深度值 (m)
            delta_t: 溫度差異閾值 (°C)
        """
        if temp_3d.ndim != 3:
            return None

        n_depths, ny, nx = temp_3d.shape

        if depth_levels is None:
            # 標準 HYCOM 深度表
            depth_levels = np.array([0, 10, 20, 30, 50, 75, 100,
                                     125, 150, 200, 250, 300, 400, 500])[:n_depths]

        mld = np.full((ny, nx), np.nan, dtype=np.float32)

        for iy in range(ny):
            for ix in range(nx):
                profile = temp_3d[:, iy, ix]
                sst = profile[0]
                if np.isnan(sst):
                    continue

                threshold = sst - delta_t
                for k in range(1, n_depths):
                    if np.isnan(profile[k]):
                        continue
                    if profile[k] < threshold:
                        # 線性插值
                        if k > 0 and not np.isnan(profile[k-1]):
                            frac = (threshold - profile[k-1]) / (profile[k] - profile[k-1])
                            mld[iy, ix] = depth_levels[k-1] + \
                                frac * (depth_levels[k] - depth_levels[k-1])
                        else:
                            mld[iy, ix] = depth_levels[k]
                        break
                else:
                    mld[iy, ix] = depth_levels[-1]

        return mld
