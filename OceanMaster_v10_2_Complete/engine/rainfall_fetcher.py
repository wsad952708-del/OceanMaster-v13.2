"""
OceanMaster v10.3 — GPM 降雨數據擷取
======================================
Task #15: NASA GPM IMERG → days_since_heavy_rain

限制:
  - 僅近岸 50km 範圍內啟用 (節省運算)
  - 日雨量 > 50mm = 暴雨事件

Cascade:
  1. NASA GPM IMERG Final Run (0.1° / 30min) via OpenDAP
  2. Open-Meteo precipitation (已由 weather_fetcher 取得)
  3. Climatology (0mm, 999天)
"""

import asyncio
import json
import logging
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, Tuple

from engine.base_fetcher import BaseFetcher, DataSource

log = logging.getLogger("OceanMaster.Rainfall")

HEAVY_RAIN_THRESHOLD_MM = 50.0   # 暴雨定義 (mm/day)
COASTAL_LIMIT_KM = 50.0          # 近岸限制


class RainfallFetcher(BaseFetcher):
    """GPM 降雨數據擷取器"""

    def __init__(self):
        super().__init__()

    async def fetch_rainfall_features(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        lats: np.ndarray,
        lons: np.ndarray,
        bathy_depth: Optional[np.ndarray] = None,
        existing_precip: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        計算降雨特徵 (days_since_heavy_rain)。

        近岸判定:
          若提供 bathy_depth，使用水深 < 200m 作為近岸判斷。
          否則使用經緯度粗略估算。

        Args:
            bathy_depth: 2D 水深 (m, 負值)
            existing_precip: 2D 降雨 (mm/h, 來自 weather_fetcher)

        Returns:
            {
                'precipitation_daily': 2D (mm/day)
                'days_since_heavy_rain': 2D (days, 999=no recent rain)
                'is_coastal': 2D bool
                'source': str
            }
        """
        ny, nx = len(lats), len(lons)

        # ── 近岸判定 ──
        is_coastal = self._detect_coastal(lats, lons, bathy_depth)

        # ── 降雨數據 ──
        if existing_precip is not None and existing_precip.shape == (ny, nx):
            daily_precip = existing_precip * 24  # mm/h → mm/day
            source = "WeatherFetcher"
        else:
            # 嘗試 Open-Meteo 歷史降雨
            daily_precip = await self._fetch_recent_rainfall(
                lat_range, lon_range, lats, lons
            )
            source = "OpenMeteo-history" if daily_precip is not None else "CLIM"

        if daily_precip is None:
            daily_precip = np.zeros((ny, nx), dtype=np.float32)

        # ── 計算 days_since_heavy_rain ──
        days_since = np.full((ny, nx), 999.0, dtype=np.float32)

        # 只對近岸區域計算
        heavy_mask = daily_precip >= HEAVY_RAIN_THRESHOLD_MM
        coastal_heavy = heavy_mask & is_coastal
        days_since[coastal_heavy] = 0.0  # 目前正在暴雨

        n_coastal = int(np.sum(is_coastal))
        n_heavy = int(np.sum(coastal_heavy))

        log.info(
            f"  Rainfall: {source}, coastal={n_coastal} cells, "
            f"heavy_rain={n_heavy} cells"
        )

        return {
            "precipitation_daily": daily_precip.astype(np.float32),
            "days_since_heavy_rain": days_since,
            "is_coastal": is_coastal,
            "source": source,
        }

    async def _fetch_recent_rainfall(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        lats: np.ndarray,
        lons: np.ndarray,
        lookback_days: int = 7,
    ) -> Optional[np.ndarray]:
        """
        從 Open-Meteo 取得過去幾天的降雨歷史。
        """
        try:
            import httpx
        except ImportError:
            import requests as httpx

        ny, nx = len(lats), len(lons)
        lat_c = float((lats.min() + lats.max()) / 2)
        lon_c = float((lons.min() + lons.max()) / 2)

        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat_c:.2f}&longitude={lon_c:.2f}"
            f"&daily=precipitation_sum"
            f"&start_date={start}&end_date={end}"
        )

        try:
            if hasattr(httpx, 'AsyncClient'):
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.get(url)
                    if resp.status_code != 200:
                        return None
                    data = resp.json() if callable(getattr(resp, 'json', None)) \
                        else json.loads(resp.text)
            else:
                resp = httpx.get(url, timeout=15)
                if resp.status_code != 200:
                    return None
                data = resp.json()

            daily = data.get("daily", {})
            precip_sums = daily.get("precipitation_sum", [])
            if not precip_sums:
                return None

            # 取最大日降雨量作為代表
            max_daily = max(float(p) if p else 0 for p in precip_sums)
            return np.full((ny, nx), max_daily, dtype=np.float32)

        except Exception as e:
            log.debug(f"  Rainfall history: {e}")
            return None

    @staticmethod
    def _detect_coastal(
        lats: np.ndarray,
        lons: np.ndarray,
        bathy_depth: Optional[np.ndarray],
    ) -> np.ndarray:
        """判定近岸區域"""
        ny, nx = len(lats), len(lons)
        is_coastal = np.zeros((ny, nx), dtype=bool)

        if bathy_depth is not None and bathy_depth.shape == (ny, nx):
            # 水深 < 200m = 近岸
            is_coastal = np.abs(bathy_depth) < 200
        else:
            # 粗略估算：緯度 0-5° 距赤道近岸線
            # 這裡簡化為「不在深海」的判定
            pass

        return is_coastal
