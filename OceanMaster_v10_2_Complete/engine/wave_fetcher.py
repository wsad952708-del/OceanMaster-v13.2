"""
OceanMaster v10.3 — WaveWatch III 有效波高整合
===============================================
Task #13: Significant Wave Height (SWH) 深度整合

Cascade 鏈:
  1. NOAA WW3 ERDDAP (NWW3_Global_Best, Thgt 0.5°)
  2. CMEMS Wave (GLOBAL_ANALYSISFORECAST_WAV, VHM0)
  3. 風速估算 (0.6 × wind_speed_10m)
  4. 氣候態 (1.5m)

精度影響:
  - SWH 直接用於 safety_checker (波高 > 2.5m 警告)
  - SWH 作為 ML 特徵 (魚群行為受波浪影響)
  - FTLE 計算的 Stokes drift 修正
"""

import asyncio
import json
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from engine.base_fetcher import BaseFetcher, DataSource, CascadeResult

log = logging.getLogger("OceanMaster.Wave")

# WW3 ERDDAP datasets
WW3_DATASETS = [
    ("NWW3_Global_Best", "Thgt", "WW3-Global"),        # Global 0.5°
]

# ERDDAP mirrors
ERDDAP_MIRRORS = [
    "https://coastwatch.pfeg.noaa.gov/erddap/griddap",
    "https://polarwatch.noaa.gov/erddap/griddap",
    "https://coastwatch.noaa.gov/erddap/griddap",
]

CLIM_WAVE_HEIGHT = 1.5  # 全球年平均有效波高 (m)


class WaveFetcher(BaseFetcher):
    """有效波高數據擷取器 (繼承 BaseFetcher cascade 機制)"""

    def __init__(self):
        super().__init__()

    async def fetch_wave_height(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        lats: np.ndarray,
        lons: np.ndarray,
        wind_speed: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        擷取有效波高 (Significant Wave Height, SWH)。

        Cascade:
          1. WW3 ERDDAP → 2. Open-Meteo Marine → 3. 風速估算 → 4. 氣候態

        Returns:
            {
                'wave_height': 2D np.ndarray (m),
                'wave_period': 2D np.ndarray (s) or None,
                'wave_direction': 2D np.ndarray (deg) or None,
                'source': str,
            }
        """
        lat_min, lat_max = lat_range
        lon_min, lon_max = lon_range
        ny, nx = len(lats), len(lons)

        # ─── 1. WW3 ERDDAP ───
        ww3_data = await self._fetch_ww3(lat_min, lat_max, lon_min, lon_max, lats, lons)
        if ww3_data is not None:
            return ww3_data

        # ─── 2. Open-Meteo Marine (scatter → grid) ───
        marine_data = await self._fetch_open_meteo_marine(
            lat_min, lat_max, lon_min, lon_max, lats, lons
        )
        if marine_data is not None:
            return marine_data

        # ─── 3. 風速估算: SWH ≈ 0.6 × U10 (Wilson 1965 經驗公式) ───
        if wind_speed is not None and wind_speed.shape == (ny, nx):
            estimated = 0.6 * np.clip(wind_speed, 0, 40)
            log.info(f"  Wave: estimated from wind, "
                     f"mean={np.nanmean(estimated):.1f}m")
            return {
                "wave_height": estimated.astype(np.float32),
                "wave_period": None,
                "wave_direction": None,
                "source": "WIND-EST",
            }

        # ─── 4. 氣候態 ───
        log.info(f"  Wave: climatology fallback ({CLIM_WAVE_HEIGHT}m)")
        return {
            "wave_height": np.full((ny, nx), CLIM_WAVE_HEIGHT, dtype=np.float32),
            "wave_period": None,
            "wave_direction": None,
            "source": "CLIM",
        }

    async def _fetch_ww3(
        self,
        lat_min: float, lat_max: float,
        lon_min: float, lon_max: float,
        lats: np.ndarray, lons: np.ndarray,
    ) -> Optional[Dict]:
        """NOAA WaveWatch III via ERDDAP"""
        try:
            import httpx
        except ImportError:
            import requests as httpx

        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)

        for ds_id, var, tag in WW3_DATASETS:
            for mirror in ERDDAP_MIRRORS:
                for day_offset in [0, 1, 2, 3]:
                    target = (now - timedelta(days=day_offset)).strftime("%Y-%m-%dT00:00:00Z")
                    url = (
                        f"{mirror}/{ds_id}.json?"
                        f"{var}[({target}):1:({target})]"
                        f"[({lat_min}):1:({lat_max})]"
                        f"[({lon_min}):1:({lon_max})]"
                    )
                    try:
                        if hasattr(httpx, 'AsyncClient'):
                            async with httpx.AsyncClient(timeout=25) as client:
                                resp = await client.get(url)
                                if resp.status_code != 200:
                                    continue
                                data = resp.json() if callable(getattr(resp, 'json', None)) \
                                    else json.loads(resp.text)
                        else:
                            resp = httpx.get(url, timeout=25)
                            if resp.status_code != 200:
                                continue
                            data = resp.json()

                        # Parse ERDDAP JSON Table
                        table = data.get("table", {})
                        cols = table.get("columnNames", [])
                        rows = table.get("rows", [])

                        if not rows or var not in cols:
                            continue

                        var_idx = cols.index(var)
                        lat_idx = cols.index("latitude") if "latitude" in cols else None
                        lon_idx = cols.index("longitude") if "longitude" in cols else None

                        if lat_idx is None or lon_idx is None:
                            continue

                        # 提取散點
                        pts_lat = np.array([r[lat_idx] for r in rows])
                        pts_lon = np.array([r[lon_idx] for r in rows])
                        pts_val = np.array([r[var_idx] if r[var_idx] is not None else np.nan
                                            for r in rows])

                        valid = np.isfinite(pts_val)
                        if np.sum(valid) < 4:
                            continue

                        # Regrid to target lats/lons
                        grid = self._regrid_scatter(
                            pts_lat[valid], pts_lon[valid], pts_val[valid],
                            lats, lons, CLIM_WAVE_HEIGHT,
                        )

                        log.info(f"  Wave: {tag} ({mirror.split('/')[2][:15]}, -{day_offset}d), "
                                 f"mean={np.nanmean(grid):.1f}m")
                        return {
                            "wave_height": grid.astype(np.float32),
                            "wave_period": None,
                            "wave_direction": None,
                            "source": f"{tag}(-{day_offset}d)",
                        }

                    except Exception as e:
                        log.debug(f"  WW3 {mirror[:30]}: {e}")
                        continue

        return None

    async def _fetch_open_meteo_marine(
        self,
        lat_min: float, lat_max: float,
        lon_min: float, lon_max: float,
        lats: np.ndarray, lons: np.ndarray,
    ) -> Optional[Dict]:
        """Open-Meteo Marine API (scatter 取樣 → griddata 插值)"""
        try:
            import httpx
        except ImportError:
            import requests as httpx

        # Sample 3×3 grid
        s_lats = np.linspace(lat_min, lat_max, 3)
        s_lons = np.linspace(lon_min, lon_max, 3)

        pts_lat, pts_lon, pts_wh, pts_wp, pts_wd = [], [], [], [], []

        for la in s_lats:
            for lo in s_lons:
                url = (
                    f"https://marine-api.open-meteo.com/v1/marine?"
                    f"latitude={la:.2f}&longitude={lo:.2f}"
                    f"&current=wave_height,wave_period,wave_direction"
                )
                try:
                    if hasattr(httpx, 'AsyncClient'):
                        async with httpx.AsyncClient(timeout=12) as client:
                            resp = await client.get(url)
                            if resp.status_code != 200:
                                continue
                            data = resp.json() if callable(getattr(resp, 'json', None)) \
                                else json.loads(resp.text)
                    else:
                        resp = httpx.get(url, timeout=12)
                        if resp.status_code != 200:
                            continue
                        data = resp.json()

                    current = data.get("current", {})
                    wh = current.get("wave_height")
                    if wh is not None:
                        pts_lat.append(la)
                        pts_lon.append(lo)
                        pts_wh.append(float(wh))
                        pts_wp.append(float(current.get("wave_period", 0)))
                        pts_wd.append(float(current.get("wave_direction", 0)))
                except Exception:
                    continue

        if len(pts_wh) < 2:
            return None

        wh_grid = self._regrid_scatter(
            np.array(pts_lat), np.array(pts_lon), np.array(pts_wh),
            lats, lons, CLIM_WAVE_HEIGHT,
        )

        log.info(f"  Wave: Open-Meteo Marine ({len(pts_wh)} pts), "
                 f"mean={np.nanmean(wh_grid):.1f}m")

        return {
            "wave_height": wh_grid.astype(np.float32),
            "wave_period": None,  # 未來可加入
            "wave_direction": None,
            "source": "OpenMeteo-Marine",
        }

    @staticmethod
    def _regrid_scatter(
        src_lat: np.ndarray, src_lon: np.ndarray, src_val: np.ndarray,
        dst_lat: np.ndarray, dst_lon: np.ndarray,
        fill_value: float,
    ) -> np.ndarray:
        """散點 → 規則網格 (linear + nearest 填充)"""
        ny, nx = len(dst_lat), len(dst_lon)
        try:
            from scipy.interpolate import griddata
            pts = np.column_stack([src_lat, src_lon])
            lat_g, lon_g = np.meshgrid(dst_lat, dst_lon, indexing="ij")
            target = np.column_stack([lat_g.ravel(), lon_g.ravel()])

            interp = griddata(pts, src_val, target, method="linear")
            nn = griddata(pts, src_val, target, method="nearest")
            result = np.where(np.isnan(interp), nn, interp)
            return result.reshape(ny, nx).astype(np.float32)
        except ImportError:
            return np.full((ny, nx), fill_value, dtype=np.float32)
