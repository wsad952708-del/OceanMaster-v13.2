"""
OceanMaster v10.3 — 海洋氣象擷取引擎
======================================
Task #10: Open-Meteo 基礎氣象擴充

新增欄位:
  - pressure_msl  海平面氣壓 (hPa)
  - wave_height    有效波高 (m)
  - precipitation  降雨量 (mm/h)

Cascade 鏈:
  1. Open-Meteo Marine API (wave_height)
  2. Open-Meteo Forecast API (pressure_msl, precipitation, wind)
  3. Climatology fallback (1013.25 hPa, 1.5m, 0mm)

多點網格策略: 取 3×3 → 5×5 格點再 scipy.griddata 插值
"""

import asyncio
import json
import logging
import numpy as np
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from engine.base_fetcher import BaseFetcher, DataSource, CascadeResult

log = logging.getLogger("OceanMaster.Weather")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  常數: 氣候態平均值
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CLIM_PRESSURE = 1013.25    # hPa (標準大氣壓)
CLIM_WAVE_HEIGHT = 1.5     # m   (全球海洋年平均)
CLIM_PRECIPITATION = 0.0   # mm/h


class WeatherFetcher(BaseFetcher):
    """海洋氣象數據擷取器 (繼承 BaseFetcher cascade 機制)"""

    def __init__(self):
        super().__init__()

    async def fetch_marine_weather(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        ny: int = 0,
        nx: int = 0,
    ) -> Dict[str, np.ndarray]:
        """
        擷取完整氣象數據: wind + pressure + wave + precipitation

        Args:
            lat_range: (lat_min, lat_max)
            lon_range: (lon_min, lon_max)
            ny, nx: 輸出網格大小 (0 = 自動)

        Returns:
            {
                'wind_u': 2D,  'wind_v': 2D,
                'wind_speed': 2D,
                'pressure_msl': 2D,
                'wave_height': 2D,
                'precipitation': 2D,
                'source': str,
            }
        """
        lat_min, lat_max = lat_range
        lon_min, lon_max = lon_range

        if ny == 0:
            ny = max(5, int((lat_max - lat_min) / 0.5))
        if nx == 0:
            nx = max(5, int((lon_max - lon_min) / 0.5))

        lats = np.linspace(lat_min, lat_max, ny)
        lons = np.linspace(lon_min, lon_max, nx)

        # ── 取樣點 (3×3 → 5×5) ──
        sample_lats, sample_lons = self._grid_sample_points(
            lat_min, lat_max, lon_min, lon_max, n=5
        )

        # ── 並行抓取: Marine API + Forecast API ──
        marine_task = self._fetch_marine_api(sample_lats, sample_lons)
        forecast_task = self._fetch_forecast_api(sample_lats, sample_lons)

        marine_data, forecast_data = await asyncio.gather(
            marine_task, forecast_task, return_exceptions=True
        )

        # ── 處理 Marine API 結果 (wave_height) ──
        wave_pts = []
        if isinstance(marine_data, dict) and "points" in marine_data:
            wave_pts = marine_data["points"]

        # ── 處理 Forecast API 結果 (pressure, precipitation, wind) ──
        pressure_pts = []
        precip_pts = []
        wind_pts = []
        if isinstance(forecast_data, dict) and "points" in forecast_data:
            pressure_pts = forecast_data["points"]
            precip_pts = forecast_data.get("precip_points", [])
            wind_pts = forecast_data.get("wind_points", [])

        # ── 插值到目標網格 ──
        wave_grid = self._interpolate_to_grid(
            wave_pts, "wave_height", lats, lons, CLIM_WAVE_HEIGHT
        )
        pressure_grid = self._interpolate_to_grid(
            pressure_pts, "pressure_msl", lats, lons, CLIM_PRESSURE
        )
        precip_grid = self._interpolate_to_grid(
            precip_pts, "precipitation", lats, lons, CLIM_PRECIPITATION
        )

        # ── 風場 ──
        wind_u_grid, wind_v_grid = self._interpolate_wind(
            wind_pts, lats, lons
        )
        wind_speed = np.sqrt(wind_u_grid**2 + wind_v_grid**2)

        source_parts = []
        if wave_pts:
            source_parts.append("Marine-API")
        if pressure_pts:
            source_parts.append("Forecast-API")
        if not source_parts:
            source_parts.append("CLIM")

        result = {
            "wind_u": wind_u_grid,
            "wind_v": wind_v_grid,
            "wind_speed": wind_speed,
            "pressure_msl": pressure_grid,
            "wave_height": wave_grid,
            "precipitation": precip_grid,
            "source": "+".join(source_parts),
        }

        log.info(
            f"  Weather: pressure=[{np.nanmin(pressure_grid):.0f},"
            f"{np.nanmax(pressure_grid):.0f}]hPa, "
            f"wave=[{np.nanmin(wave_grid):.1f},{np.nanmax(wave_grid):.1f}]m, "
            f"wind={np.nanmean(wind_speed):.1f}m/s ({result['source']})"
        )
        return result

    # ─── Open-Meteo Marine API (波高) ───

    async def _fetch_marine_api(
        self, sample_lats: List[float], sample_lons: List[float]
    ) -> Optional[Dict]:
        """呼叫 Open-Meteo Marine API 取得 wave_height"""
        try:
            import httpx
        except ImportError:
            import requests as httpx

        points = []
        for lat, lon in zip(sample_lats, sample_lons):
            url = (
                f"https://marine-api.open-meteo.com/v1/marine?"
                f"latitude={lat:.2f}&longitude={lon:.2f}"
                f"&current=wave_height,wave_period,wave_direction"
            )
            try:
                if hasattr(httpx, 'AsyncClient'):
                    async with httpx.AsyncClient(timeout=15) as client:
                        resp = await client.get(url)
                        if resp.status_code != 200:
                            continue
                        data = resp.json() if hasattr(resp, 'json') and callable(resp.json) \
                            else json.loads(resp.text)
                else:
                    resp = httpx.get(url, timeout=15)
                    if resp.status_code != 200:
                        continue
                    data = resp.json()

                current = data.get("current", {})
                wh = current.get("wave_height")
                if wh is not None:
                    points.append({
                        "lat": lat, "lon": lon,
                        "wave_height": float(wh),
                        "wave_period": current.get("wave_period", 0),
                        "wave_direction": current.get("wave_direction", 0),
                    })
            except Exception as e:
                log.debug(f"  Marine API ({lat:.1f},{lon:.1f}): {e}")
                continue

        if not points:
            return None

        log.info(f"  Marine API: {len(points)} wave points fetched")
        return {"points": points}

    # ─── Open-Meteo Forecast API (氣壓/降雨/風) ───

    async def _fetch_forecast_api(
        self, sample_lats: List[float], sample_lons: List[float]
    ) -> Optional[Dict]:
        """呼叫 Open-Meteo Forecast API 取得 pressure_msl + precipitation + wind"""
        try:
            import httpx
        except ImportError:
            import requests as httpx

        points = []
        precip_pts = []
        wind_pts = []

        for lat, lon in zip(sample_lats, sample_lons):
            url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={lat:.2f}&longitude={lon:.2f}"
                f"&current=pressure_msl,precipitation,wind_speed_10m,"
                f"wind_direction_10m"
            )
            try:
                if hasattr(httpx, 'AsyncClient'):
                    async with httpx.AsyncClient(timeout=15) as client:
                        resp = await client.get(url)
                        if resp.status_code != 200:
                            continue
                        data = resp.json() if hasattr(resp, 'json') and callable(resp.json) \
                            else json.loads(resp.text)
                else:
                    resp = httpx.get(url, timeout=15)
                    if resp.status_code != 200:
                        continue
                    data = resp.json()

                current = data.get("current", {})
                pressure = current.get("pressure_msl")
                precip = current.get("precipitation", 0)
                wspd = current.get("wind_speed_10m", 0)
                wdir = current.get("wind_direction_10m", 0)

                if pressure is not None:
                    points.append({
                        "lat": lat, "lon": lon,
                        "pressure_msl": float(pressure),
                    })
                precip_pts.append({
                    "lat": lat, "lon": lon,
                    "precipitation": float(precip) if precip else 0.0,
                })
                if wspd is not None:
                    u = -wspd * np.sin(np.radians(wdir))
                    v = -wspd * np.cos(np.radians(wdir))
                    wind_pts.append({
                        "lat": lat, "lon": lon,
                        "u": float(u), "v": float(v),
                        "speed": float(wspd),
                    })
            except Exception as e:
                log.debug(f"  Forecast API ({lat:.1f},{lon:.1f}): {e}")
                continue

        if not points:
            return None

        log.info(f"  Forecast API: {len(points)} pressure points")
        return {
            "points": points,
            "precip_points": precip_pts,
            "wind_points": wind_pts,
        }

    # ─── 內部工具 ───

    @staticmethod
    def _grid_sample_points(
        lat_min: float, lat_max: float,
        lon_min: float, lon_max: float,
        n: int = 5,
    ) -> Tuple[List[float], List[float]]:
        """產生 n×n 均勻取樣點"""
        lats_s = np.linspace(lat_min, lat_max, n)
        lons_s = np.linspace(lon_min, lon_max, n)
        lat_list, lon_list = [], []
        for la in lats_s:
            for lo in lons_s:
                lat_list.append(float(la))
                lon_list.append(float(lo))
        return lat_list, lon_list

    @staticmethod
    def _interpolate_to_grid(
        points: List[Dict],
        field: str,
        lats: np.ndarray,
        lons: np.ndarray,
        fill_value: float,
    ) -> np.ndarray:
        """將散點資料插值到 2D 網格"""
        ny, nx = len(lats), len(lons)
        grid = np.full((ny, nx), fill_value, dtype=np.float32)

        if not points:
            return grid

        try:
            from scipy.interpolate import griddata

            pts = np.array([[p["lat"], p["lon"]] for p in points])
            vals = np.array([p[field] for p in points])

            lat_g, lon_g = np.meshgrid(lats, lons, indexing="ij")
            target = np.column_stack([lat_g.ravel(), lon_g.ravel()])

            interp = griddata(pts, vals, target, method="linear")
            interp_nn = griddata(pts, vals, target, method="nearest")

            result = np.where(np.isnan(interp), interp_nn, interp)
            grid = result.reshape(ny, nx).astype(np.float32)

            # 填充殘餘 NaN
            grid = np.where(np.isfinite(grid), grid, fill_value)

        except ImportError:
            # scipy 不可用 → 簡單最近鄰
            lat_g, lon_g = np.meshgrid(lats, lons, indexing="ij")
            for p in points:
                iy = np.argmin(np.abs(lats - p["lat"]))
                ix = np.argmin(np.abs(lons - p["lon"]))
                grid[iy, ix] = p[field]

        return grid

    @staticmethod
    def _interpolate_wind(
        wind_pts: List[Dict],
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """風場 u/v 散點 → 2D 網格"""
        ny, nx = len(lats), len(lons)
        if not wind_pts:
            return (
                np.zeros((ny, nx), dtype=np.float32),
                np.zeros((ny, nx), dtype=np.float32),
            )

        pts_arr = np.array([[p["lat"], p["lon"]] for p in wind_pts])
        u_vals = np.array([p["u"] for p in wind_pts])
        v_vals = np.array([p["v"] for p in wind_pts])

        try:
            from scipy.interpolate import griddata

            lat_g, lon_g = np.meshgrid(lats, lons, indexing="ij")
            target = np.column_stack([lat_g.ravel(), lon_g.ravel()])

            u_grid = griddata(pts_arr, u_vals, target, method="nearest").reshape(ny, nx)
            v_grid = griddata(pts_arr, v_vals, target, method="nearest").reshape(ny, nx)

            return u_grid.astype(np.float32), v_grid.astype(np.float32)

        except ImportError:
            u_mean = float(np.mean(u_vals))
            v_mean = float(np.mean(v_vals))
            return (
                np.full((ny, nx), u_mean, dtype=np.float32),
                np.full((ny, nx), v_mean, dtype=np.float32),
            )
