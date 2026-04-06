"""
OceanMaster v13.2 — Eddy Biological Maturity Index
====================================================
Estimate the biological maturity of mesoscale eddies.

Young eddy (< 2 weeks): just formed, no biological enhancement yet
Mature eddy (2-8 weeks): phytoplankton bloom developing, prey accumulating
Old eddy (> 8 weeks): fully developed ecosystem, highest CPUE
Decaying eddy: weakening, ecosystem dispersing

Ref: Gaube et al. (2014) GRL 41:3619-3625
     Chelton et al. (2011) Prog. Oceanogr. 91:167-216
"""

import numpy as np
import logging
from typing import Dict, Any, Optional

log = logging.getLogger("OceanMaster.EddyAge")


class EddyMaturityEstimator:
    """
    [v16.0] Estimate eddy biological maturity from SSH anomaly + CHL patterns.

    Direct eddy age tracking requires historical time series.
    We use a proxy: SSH anomaly strength × CHL anomaly in the eddy
    to estimate biological development stage.

    Mature eddies have:
    - Strong SSH anomaly (deep, well-formed)
    - Elevated CHL inside cyclonic eddies (nutrient pumping)
    - Suppressed CHL inside anti-cyclonic eddies (downwelling)
    """

    def compute(
        self,
        ssh: np.ndarray,
        chl: np.ndarray,
        eddy_type: Optional[np.ndarray] = None,
        eke: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Estimate eddy maturity.

        Args:
            ssh: SSH anomaly (m)
            chl: Chlorophyll-a (mg/m³)
            eddy_type: from eddy_detector (-1=cyclonic, 0=none, 1=anti-cyclonic)
            eke: Eddy kinetic energy (optional)

        Returns:
            {
                "maturity_index": 2D (0-1, higher = more mature),
                "maturity_stage": 2D int (0=none, 1=young, 2=developing, 3=mature, 4=old),
                "bio_enhancement": 2D (CHL anomaly in eddy),
                "fishing_score": 2D (0-1, predicted CPUE enhancement from eddy age),
                "advisory": str,
            }
        """
        ny, nx = ssh.shape

        # SSH anomaly strength (normalized)
        ssh_safe = np.nan_to_num(ssh, nan=0)
        ssh_abs = np.abs(ssh_safe)
        ssh_std = max(float(np.std(ssh_abs[ssh_abs > 0])), 0.01)
        ssh_strength = np.clip(ssh_abs / (3 * ssh_std), 0, 1).astype(np.float32)

        # CHL anomaly (local anomaly relative to regional mean)
        chl_safe = np.nan_to_num(chl, nan=0.3)
        chl_mean = max(float(np.nanmean(chl_safe)), 0.01)
        chl_anomaly = ((chl_safe - chl_mean) / chl_mean).astype(np.float32)

        # Biological enhancement in eddy
        # Cyclonic (cold core): positive CHL anomaly = nutrient pumping = GOOD
        # Anti-cyclonic (warm core): negative CHL anomaly = downwelling = varies
        if eddy_type is not None:
            # Cyclonic eddies with high CHL = mature, biologically productive
            cyclonic_bio = np.where(
                eddy_type == -1,
                np.clip(chl_anomaly, 0, 2),  # only positive anomaly matters
                0
            )
            # Anti-cyclonic eddies: fish aggregate at EDGES, not center
            # Edge proxy: high SSH gradient
            ssh_gy, ssh_gx = np.gradient(ssh_safe)
            ssh_grad = np.sqrt(ssh_gx**2 + ssh_gy**2)
            ssh_grad_norm = np.clip(ssh_grad / max(float(np.std(ssh_grad)), 1e-6), 0, 1)
            anticyc_edge = np.where(eddy_type == 1, ssh_grad_norm, 0)

            bio_enhancement = (cyclonic_bio + anticyc_edge * 0.5).astype(np.float32)
        else:
            bio_enhancement = np.clip(chl_anomaly, 0, 2).astype(np.float32)

        # EKE contribution (high EKE = active eddy)
        if eke is not None:
            eke_safe = np.nan_to_num(eke, nan=0)
            eke_norm = np.clip(eke_safe / max(float(np.std(eke_safe)), 1e-6), 0, 1)
        else:
            eke_norm = ssh_strength * 0.5  # proxy

        # Maturity index (composite)
        maturity = (0.35 * ssh_strength + 0.35 * bio_enhancement + 0.30 * eke_norm)
        maturity = np.clip(maturity, 0, 1).astype(np.float32)

        # Stage classification
        stage = np.zeros((ny, nx), dtype=np.int8)
        has_eddy = ssh_strength > 0.1
        stage[has_eddy & (maturity < 0.2)] = 1   # Young
        stage[has_eddy & (maturity >= 0.2) & (maturity < 0.5)] = 2  # Developing
        stage[has_eddy & (maturity >= 0.5) & (maturity < 0.8)] = 3  # Mature
        stage[has_eddy & (maturity >= 0.8)] = 4   # Old/peaked

        # Fishing score: mature and old eddies have highest CPUE
        fishing = np.zeros_like(maturity)
        fishing[stage == 1] = 0.2   # Young: low
        fishing[stage == 2] = 0.5   # Developing: moderate
        fishing[stage == 3] = 0.9   # Mature: excellent
        fishing[stage == 4] = 0.7   # Old: still good but declining

        mature_pct = float(np.mean(stage >= 3) * 100)
        advisory = (f"成熟渦旋: {mature_pct:.0f}% 區域有生物成熟渦旋 — CPUE +50-90%"
                    if mature_pct > 5
                    else "渦旋多為年輕或不活躍 — 生物增強效應有限")

        log.info(f"  Eddy maturity: young={float(np.mean(stage==1))*100:.0f}%, "
                 f"dev={float(np.mean(stage==2))*100:.0f}%, "
                 f"mature={float(np.mean(stage==3))*100:.0f}%, "
                 f"old={float(np.mean(stage==4))*100:.0f}%")

        return {
            "maturity_index": maturity,
            "maturity_stage": stage,
            "bio_enhancement": bio_enhancement,
            "fishing_score": fishing,
            "mature_pct": round(mature_pct, 1),
            "advisory": advisory,
        }
