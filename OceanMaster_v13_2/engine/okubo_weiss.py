"""
OceanMaster v13.2 — Okubo-Weiss Parameter + Strain Rate [#1 + #6]
==================================================================
Classify ocean flow into:
  - Rotation-dominated (eddy cores): OW < -threshold
  - Strain-dominated (eddy edges/filaments): OW > +threshold
  - Background: |OW| < threshold

Tuna CPUE is 3-5x higher in strain-dominated regions (OW > 0).
Ref: Sabarros et al. (2009), Okubo (1970), Weiss (1991)

Also computes:
  - Normal strain (s_n): stretching/compression
  - Shear strain (s_s): shearing deformation
  - Relative vorticity (ω): rotation
"""

import numpy as np
import logging
from typing import Dict, Any

log = logging.getLogger("OceanMaster.OW")


class OkuboWeissAnalyzer:
    """
    [v16.0 #1+#6] Okubo-Weiss parameter and strain rate from u/v currents.

    OW = s_n² + s_s² - ω²
      s_n = ∂u/∂x - ∂v/∂y  (normal strain)
      s_s = ∂v/∂x + ∂u/∂y  (shear strain)
      ω   = ∂v/∂x - ∂u/∂y  (relative vorticity)

    OW > 0 → strain dominates → filaments → prey compressed
    OW < 0 → rotation dominates → eddy cores → trapped water
    """

    def __init__(self, ow_threshold: float = 0.2):
        """
        Args:
            ow_threshold: σ multiplier for OW classification.
                          |OW| > threshold * σ(OW) → significant.
        """
        self.ow_threshold_sigma = ow_threshold

    def compute(
        self,
        u: np.ndarray,
        v: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compute Okubo-Weiss, strain rate, and vorticity fields.

        Args:
            u: 2D zonal velocity (m/s)
            v: 2D meridional velocity (m/s)
            lats: 1D latitude array
            lons: 1D longitude array

        Returns:
            {
                "okubo_weiss": 2D OW field (s⁻²),
                "strain_rate": 2D total strain (s⁻¹),
                "normal_strain": 2D s_n (s⁻¹),
                "shear_strain": 2D s_s (s⁻¹),
                "vorticity": 2D ω (s⁻¹),
                "ow_classification": 2D int (-1=rotation, 0=background, 1=strain),
                "strain_pct": float (% of grid in strain-dominated),
                "rotation_pct": float (% in rotation-dominated),
                "advisory": str,
            }
        """
        ny, nx = u.shape

        # Grid spacing in meters
        lat_mid = np.mean(lats)
        dy = np.abs(np.diff(lats).mean()) * 111320.0  # deg → m
        dx = np.abs(np.diff(lons).mean()) * 111320.0 * np.cos(np.radians(lat_mid))

        if dx < 1 or dy < 1:
            dx = dy = 25000.0  # 0.25° default

        # Velocity gradients (central differences, edges = forward/backward)
        du_dx = np.gradient(u, dx, axis=1)
        du_dy = np.gradient(u, dy, axis=0)
        dv_dx = np.gradient(v, dx, axis=1)
        dv_dy = np.gradient(v, dy, axis=0)

        # Components
        s_n = du_dx - dv_dy          # Normal strain
        s_s = dv_dx + du_dy          # Shear strain
        omega = dv_dx - du_dy        # Relative vorticity

        # Okubo-Weiss parameter
        ow = (s_n**2 + s_s**2 - omega**2).astype(np.float32)

        # Total strain rate
        strain = np.sqrt(s_n**2 + s_s**2).astype(np.float32)

        # Classification based on OW relative to its std
        ow_finite = ow[np.isfinite(ow)]
        if len(ow_finite) > 0:
            ow_sigma = float(np.std(ow_finite))
        else:
            ow_sigma = 1e-10
        threshold = self.ow_threshold_sigma * max(ow_sigma, 1e-12)

        classification = np.zeros((ny, nx), dtype=np.int8)
        classification[ow > threshold] = 1     # Strain-dominated
        classification[ow < -threshold] = -1   # Rotation-dominated

        strain_pct = float(np.mean(classification == 1) * 100)
        rotation_pct = float(np.mean(classification == -1) * 100)

        if strain_pct > 30:
            advisory = "高應變區域多 — 渦旋邊緣 filament 發達，餌料壓縮區"
        elif strain_pct > 10:
            advisory = "中等應變 — 部分渦旋邊緣活躍"
        else:
            advisory = "應變弱 — 渦旋邊緣不明顯"

        log.info(f"  OW: strain={strain_pct:.1f}%, rotation={rotation_pct:.1f}%, "
                 f"σ(OW)={ow_sigma:.2e}")

        return {
            "okubo_weiss": ow,
            "strain_rate": strain,
            "normal_strain": s_n.astype(np.float32),
            "shear_strain": s_s.astype(np.float32),
            "vorticity": omega.astype(np.float32),
            "ow_classification": classification,
            "strain_pct": round(strain_pct, 1),
            "rotation_pct": round(rotation_pct, 1),
            "ow_sigma": ow_sigma,
            "advisory": advisory,
        }
