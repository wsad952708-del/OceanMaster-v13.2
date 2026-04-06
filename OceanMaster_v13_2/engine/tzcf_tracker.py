"""
OceanMaster v13.2 — TZCF Tracker (Transition Zone Chlorophyll Front)
====================================================================
蒼鷺 AI 隱式依賴的關鍵特徵 → 顯式計算。

TZCF = Chl-a ≈ 0.2 mg/m³ 等值線，標示亞熱帶→溫帶的過渡帶。
北太平洋柔魚（Ommastrephes bartramii）大量聚集在 TZCF 附近。

Scientific basis:
  - Polovina et al. (2001): TZCF (18°C isotherm ∩ 0.2 mg/m³ Chl-a)
    is the primary foraging habitat for North Pacific loggerhead turtles
    and squid.
  - Bograd et al. (2004): TZCF位置與北太平洋漁場高度相關。
  - 蒼鷺 AI 用 SST+Chl-a 作為輸入，U-Net 隱式學習了 TZCF 位置。
    OceanMaster 顯式計算更高效。

Output:
  - TZCF latitude at each longitude (contour extraction)
  - Distance-to-TZCF 2D field → feature for ML/DL models
"""

import numpy as np
import logging
from typing import Dict, Any, Optional

log = logging.getLogger("OceanMaster.TZCF")


class TZCFTracker:
    """
    Transition Zone Chlorophyll Front tracker.

    Finds the Chl-a = threshold contour and computes
    distance-to-TZCF as a spatial feature.
    """

    def __init__(self, chl_threshold: float = 0.2):
        """
        Args:
            chl_threshold: Chl-a value defining TZCF (mg/m³).
                           Default 0.2 from Polovina et al. (2001).
        """
        self.chl_threshold = chl_threshold

    def compute(
        self,
        chl: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compute TZCF location and distance field.

        Args:
            chl: 2D Chl-a field (ny, nx), mg/m³
            lats: 1D latitude array (ny,), assumed sorted
            lons: 1D longitude array (nx,)

        Returns:
            {
                "tzcf_latitude": 1D array (nx,) — TZCF latitude per longitude
                                 NaN where no crossing found
                "distance_to_tzcf": 2D array (ny, nx) — signed distance in degrees
                                    positive = north of TZCF, negative = south
                "tzcf_mask": 2D bool (ny, nx) — within ±1° of TZCF
                "mean_tzcf_lat": float — mean TZCF latitude
                "valid_fraction": float — fraction of longitudes with detected TZCF
            }
        """
        ny, nx = chl.shape
        chl_clean = np.nan_to_num(chl, nan=0.0)

        tzcf_lat = np.full(nx, np.nan, dtype=np.float64)

        for ix in range(nx):
            col = chl_clean[:, ix]

            # Find where Chl-a crosses the threshold
            # TZCF: going from low (subtropical) to high (temperate) Chl-a
            # = northward in NH, southward in SH
            crossings = []
            for iy in range(ny - 1):
                v0, v1 = col[iy], col[iy + 1]
                if v0 == 0 and v1 == 0:
                    continue

                # Linear interpolation of crossing point
                if (v0 - self.chl_threshold) * (v1 - self.chl_threshold) < 0:
                    # Crossing between iy and iy+1
                    frac = (self.chl_threshold - v0) / (v1 - v0 + 1e-12)
                    lat_cross = lats[iy] + frac * (lats[iy + 1] - lats[iy])
                    crossings.append(lat_cross)

            if crossings:
                # Take the crossing closest to the expected TZCF latitude (~30-40°N)
                # In southern hemisphere, take the one closest to -30 to -40°S
                if np.mean(lats) > 0:
                    # Northern hemisphere — TZCF around 30-40°N
                    tzcf_lat[ix] = min(crossings, key=lambda x: abs(x - 35.0))
                else:
                    # Southern hemisphere
                    tzcf_lat[ix] = min(crossings, key=lambda x: abs(x + 35.0))

        # Distance-to-TZCF field (signed, in degrees latitude)
        distance_to_tzcf = np.full((ny, nx), np.nan, dtype=np.float32)
        for ix in range(nx):
            if not np.isnan(tzcf_lat[ix]):
                distance_to_tzcf[:, ix] = lats - tzcf_lat[ix]

        # Fill NaN columns with nearest neighbor
        valid_cols = ~np.isnan(tzcf_lat)
        if np.any(valid_cols) and not np.all(valid_cols):
            valid_idx = np.where(valid_cols)[0]
            for ix in range(nx):
                if np.isnan(tzcf_lat[ix]):
                    nearest = valid_idx[np.argmin(np.abs(valid_idx - ix))]
                    distance_to_tzcf[:, ix] = lats - tzcf_lat[nearest]

        # Replace remaining NaN with 0
        distance_to_tzcf = np.nan_to_num(distance_to_tzcf, nan=0.0)

        # TZCF proximity mask: within ±1 degree
        tzcf_mask = np.abs(distance_to_tzcf) < 1.0

        valid_frac = float(np.sum(valid_cols) / max(nx, 1))
        mean_lat = float(np.nanmean(tzcf_lat)) if np.any(valid_cols) else np.nan

        log.info(
            f"  TZCF: threshold={self.chl_threshold} mg/m³, "
            f"valid={valid_frac*100:.0f}% of longitudes, "
            f"mean latitude={mean_lat:.1f}°"
        )

        return {
            "tzcf_latitude": tzcf_lat.astype(np.float32),
            "distance_to_tzcf": distance_to_tzcf,
            "tzcf_mask": tzcf_mask,
            "mean_tzcf_lat": mean_lat,
            "valid_fraction": valid_frac,
        }
