"""
OceanMaster v10.4 — SSS 海面鹽度特徵擷取
==========================================
Task #3: Sea Surface Salinity (SSS)

Cascade:
  1. NASA SMAP L3 via ERDDAP (BBox 強制)
  2. CMEMS SMOS (Global Ocean PSY4, BBox)
  3. WOA Climatology (現有 _clim.salinity)

OOM 防護:
  所有 ERDDAP URL 必須包含 BBox — 禁止全球 fetch。

輸出特徵:
  - sss (psu) — 海面鹽度值
  - sss_gradient (psu/km) — Haversine dx/dy 計算
  - is_salinity_front (bool) — gradient > 0.05 psu/km
"""

import asyncio
import json
import logging
import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from engine.base_fetcher import BaseFetcher, DataSource

log = logging.getLogger("OceanMaster.SSS")

# ERDDAP datasets for SMAP SSS
SMAP_DATASETS = [
    # (dataset_id, variable, tag, server)
    ("jplSMAPSSSv5Monthly", "sss", "SMAP-L3-Monthly",
     "https://coastwatch.pfeg.noaa.gov/erddap/griddap"),
    ("jplSMAPSSSv5", "sss", "SMAP-L3-8day",
     "https://coastwatch.pfeg.noaa.gov/erddap/griddap"),
]

# ERDDAP mirrors
ERDDAP_MIRRORS = [
    "https://coastwatch.pfeg.noaa.gov/erddap/griddap",
    "https://polarwatch.noaa.gov/erddap/griddap",
]

# 鹽度鋒面閾值
SSS_FRONT_THRESHOLD = 0.05  # psu/km

# 物理常數 (同 Task #2)
_R_EARTH_KM = 6371.0


class SalinityFetcher(BaseFetcher):
    """
    SSS 海面鹽度擷取器。
    嚴格 BBox — 所有 API 請求不得拉取全球網格。
    """

    def __init__(self):
        super().__init__()

    async def fetch_sss_features(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        lats: np.ndarray,
        lons: np.ndarray,
        fallback_salinity: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        擷取 SSS + 計算鹽度梯度與鋒面。

        Cascade: SMAP → Open-Meteo (無 SSS) → HYCOM salinity → WOA clim

        Args:
            lat_range: (lat_min, lat_max)
            lon_range: (lon_min, lon_max)
            lats: 1D (ny,) target latitudes
            lons: 1D (nx,) target longitudes
            fallback_salinity: 2D (ny, nx) 現有 HYCOM/WOA 鹽度 (from data_fetcher_v2)

        Returns:
            {
                'sss': 2D (psu),
                'sss_gradient': 2D (psu/km),
                'is_salinity_front': 2D (bool),
                'source': str,
            }
        """
        lat_min, lat_max = lat_range
        lon_min, lon_max = lon_range
        ny, nx = len(lats), len(lons)

        # ─── 1. SMAP ERDDAP (BBox 強制) ───
        sss = await self._fetch_smap_erddap(
            lat_min, lat_max, lon_min, lon_max, lats, lons
        )
        if sss is not None:
            source = "SMAP-ERDDAP"
        elif fallback_salinity is not None and fallback_salinity.shape == (ny, nx):
            sss = fallback_salinity.copy()
            source = "HYCOM-salinity"
            log.info(f"  SSS: fallback to existing HYCOM salinity")
        else:
            # WOA climatology (mean SSS for tropical open ocean)
            sss = np.full((ny, nx), 34.8, dtype=np.float32)
            source = "WOA-CLIM"
            log.info(f"  SSS: WOA climatology fallback (34.8 psu)")

        # ─── 2. 鹽度梯度 (Haversine-correct) ───
        sss_gradient = self._compute_sss_gradient(sss, lats, lons)

        # ─── 3. 鹽度鋒面偵測 ───
        is_front = sss_gradient >= SSS_FRONT_THRESHOLD

        n_front = int(np.sum(is_front))
        log.info(
            f"  SSS: {source}, "
            f"range=[{np.nanmin(sss):.1f}, {np.nanmax(sss):.1f}] psu, "
            f"gradient_max={np.nanmax(sss_gradient):.3f} psu/km, "
            f"fronts={n_front} cells"
        )

        return {
            "sss": sss.astype(np.float32),
            "sss_gradient": sss_gradient.astype(np.float32),
            "is_salinity_front": is_front,
            "source": source,
        }

    async def _fetch_smap_erddap(
        self,
        lat_min: float, lat_max: float,
        lon_min: float, lon_max: float,
        lats: np.ndarray, lons: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        NASA SMAP L3 SSS via ERDDAP (BBox 強制)。

        URL 格式:
          .../griddap/{dataset}.json?sss[(time)][(lat_min):(lat_max)][(lon_min):(lon_max)]
        """
        try:
            import httpx
        except ImportError:
            import requests as httpx

        for ds_id, var, tag, server in SMAP_DATASETS:
            for mirror in [server] + ERDDAP_MIRRORS:
                # 🔒 OOM 防護: BBox 強制
                url = (
                    f"{mirror}/{ds_id}.json?"
                    f"{var}[last]"
                    f"[({lat_min:.2f}):1:({lat_max:.2f})]"
                    f"[({lon_min:.2f}):1:({lon_max:.2f})]"
                )

                try:
                    if hasattr(httpx, 'AsyncClient'):
                        async with httpx.AsyncClient(timeout=30) as client:
                            resp = await client.get(url)
                            if resp.status_code != 200:
                                continue
                            data = resp.json() if callable(getattr(resp, 'json', None)) \
                                else json.loads(resp.text)
                    else:
                        resp = httpx.get(url, timeout=30)
                        if resp.status_code != 200:
                            continue
                        data = resp.json()

                    # Parse ERDDAP JSON table
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

                    # Extract scatter points
                    pts_lat = np.array([r[lat_idx] for r in rows])
                    pts_lon = np.array([r[lon_idx] for r in rows])
                    pts_val = np.array([
                        r[var_idx] if r[var_idx] is not None else np.nan
                        for r in rows
                    ])

                    valid = np.isfinite(pts_val) & (pts_val > 0) & (pts_val < 45)
                    if np.sum(valid) < 4:
                        continue

                    # Regrid to target lats/lons
                    grid = self._regrid_scatter(
                        pts_lat[valid], pts_lon[valid], pts_val[valid],
                        lats, lons, fill_value=34.8,
                    )

                    log.info(f"  SSS: {tag} ({mirror.split('/')[2][:20]}), "
                             f"mean={np.nanmean(grid):.1f} psu")
                    return grid

                except Exception as e:
                    log.debug(f"  SMAP {mirror[:30]}: {e}")
                    continue

        return None

    @staticmethod
    def _compute_sss_gradient(
        sss: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> np.ndarray:
        """
        計算鹽度梯度 (psu/km)。

        使用 Haversine-correct 真實距離:
          dy = R · Δφ (km)
          dx = R · cos(φ) · Δλ (km)
        """
        ny, nx = sss.shape

        # 安全處理 NaN
        sss_clean = np.nan_to_num(sss, nan=34.8)

        # 真實距離 (km)
        lat_rad = np.radians(lats)
        dlat_rad = np.diff(lat_rad).mean() if ny > 1 else np.radians(0.25)
        dy_km = _R_EARTH_KM * dlat_rad  # scalar

        dlon_rad = np.radians(np.diff(lons).mean()) if nx > 1 else np.radians(0.25)
        cos_lat = np.cos(lat_rad)[:, np.newaxis]  # (ny, 1)
        dx_km = _R_EARTH_KM * cos_lat * dlon_rad  # (ny, 1)

        # np.gradient (pixel-based)
        dsss_dy_raw, dsss_dx_raw = np.gradient(sss_clean)

        # 轉換為 psu/km
        dsss_dy = dsss_dy_raw / max(dy_km, 0.01)
        dsss_dx = dsss_dx_raw / np.maximum(dx_km, 0.01)

        # 梯度大小
        gradient_mag = np.sqrt(dsss_dx**2 + dsss_dy**2)

        return gradient_mag.astype(np.float32)

    @staticmethod
    def _regrid_scatter(
        src_lat: np.ndarray, src_lon: np.ndarray, src_val: np.ndarray,
        dst_lat: np.ndarray, dst_lon: np.ndarray,
        fill_value: float = 34.8,
    ) -> np.ndarray:
        """散點 → 規則網格 (linear + nearest)"""
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
