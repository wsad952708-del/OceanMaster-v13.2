"""
OceanMaster v13.2 — Convergence Zone Detector
================================================
Detect ocean surface convergence zones from velocity field divergence.

Where currents converge → surface water sinks → floating debris, plankton,
and baitfish accumulate → predators aggregate.

This is THE most direct physical mechanism for fish aggregation.

Physics:
  div(V) = ∂u/∂x + ∂v/∂y
  div < 0 → convergence (water sinking, stuff accumulates)
  div > 0 → divergence (upwelling, water spreading)

  Convergence lines = where div transitions from + to -

Ref: Bakun (2006) Global Change Biology 12:1-15
     Olson et al. (1994) Bull. Mar. Sci. 55:538-558
"""

import numpy as np
import logging
from typing import Dict, Any

log = logging.getLogger("OceanMaster.Conv")


class ConvergenceDetector:
    """
    [v16.0] Detect convergence/divergence zones from u/v currents.

    Convergence zones are THE primary physical mechanism for prey
    accumulation in the open ocean. Fish follow prey → prey follows
    convergence.
    """

    def compute(
        self,
        u: np.ndarray,
        v: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compute velocity divergence and detect convergence zones.

        Args:
            u: 2D zonal velocity (m/s)
            v: 2D meridional velocity (m/s)
            lats, lons: 1D coordinate arrays

        Returns:
            {
                "divergence": 2D (s⁻¹, negative = convergence),
                "convergence_mask": 2D bool,
                "convergence_strength": 2D (0-1, normalized),
                "convergence_lines": list of (lat, lon) along zero-crossing,
                "convergence_pct": float,
                "upwelling_divergence_mask": 2D bool (div > 0),
                "advisory": str,
            }
        """
        ny, nx = u.shape

        # Grid spacing in meters
        lat_mid = np.mean(lats)
        dy = np.abs(np.diff(lats).mean()) * 111320.0
        dx = np.abs(np.diff(lons).mean()) * 111320.0 * np.cos(np.radians(lat_mid))
        if dx < 1 or dy < 1:
            dx = dy = 25000.0

        # Divergence: div(V) = ∂u/∂x + ∂v/∂y
        du_dx = np.gradient(u, dx, axis=1)
        dv_dy = np.gradient(v, dy, axis=0)
        div = (du_dx + dv_dy).astype(np.float32)

        # Convergence = negative divergence
        div_std = max(float(np.std(div[np.isfinite(div)])), 1e-12)
        convergence_mask = div < -0.5 * div_std
        upwelling_mask = div > 0.5 * div_std

        # Normalized convergence strength (0-1)
        conv_strength = np.clip(-div / (3 * div_std), 0, 1).astype(np.float32)

        # Convergence lines (zero-crossing of divergence) — vectorized
        # [v13.2-audit] Replace O(n²) nested loops with numpy vectorization
        conv_lines = []

        # Horizontal zero-crossings: sign change between div[i,j] and div[i,j+1]
        h_cross = div[:ny-1, :nx-1] * div[:ny-1, 1:nx] < 0
        if np.any(h_cross):
            h_i, h_j = np.where(h_cross)
            h_frac = div[h_i, h_j] / (div[h_i, h_j] - div[h_i, np.minimum(h_j+1, nx-1)])
            h_lat = lats[h_i]
            h_lon = lons[h_j] + h_frac * (lons[np.minimum(h_j+1, nx-1)] - lons[h_j])
            conv_lines.extend(zip(h_lat.astype(float).tolist(),
                                  h_lon.astype(float).tolist()))

        # Vertical zero-crossings: sign change between div[i,j] and div[i+1,j]
        v_cross = div[:ny-1, :nx-1] * div[1:ny, :nx-1] < 0
        if np.any(v_cross):
            v_i, v_j = np.where(v_cross)
            v_frac = div[v_i, v_j] / (div[v_i, v_j] - div[np.minimum(v_i+1, ny-1), v_j])
            v_lat = lats[v_i] + v_frac * (lats[np.minimum(v_i+1, ny-1)] - lats[v_i])
            v_lon = lons[v_j]
            conv_lines.extend(zip(v_lat.astype(float).tolist(),
                                  v_lon.astype(float).tolist()))

        conv_pct = float(np.mean(convergence_mask) * 100)
        upw_pct = float(np.mean(upwelling_mask) * 100)

        if conv_pct > 25:
            advisory = f"強收斂帶: {conv_pct:.0f}% — 餌料高度集中，黃金漁場"
        elif conv_pct > 10:
            advisory = f"中等收斂: {conv_pct:.0f}% — 部分區域有餌料聚集"
        else:
            advisory = "收斂弱 — 餌料分散"

        log.info(f"  Convergence: {conv_pct:.1f}%, divergence: {upw_pct:.1f}%, "
                 f"lines: {len(conv_lines)} pts")

        return {
            "divergence": div,
            "convergence_mask": convergence_mask,
            "convergence_strength": conv_strength,
            "convergence_lines": conv_lines[:500],  # cap for memory
            "convergence_pct": round(conv_pct, 1),
            "upwelling_divergence_mask": upwelling_mask,
            "upwelling_pct": round(upw_pct, 1),
            "advisory": advisory,
        }
