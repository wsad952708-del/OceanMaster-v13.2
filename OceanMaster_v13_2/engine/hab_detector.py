"""
OceanMaster v13.2 — HAB Harmful Algal Bloom Detector [F2]
============================================================
Detect potential harmful algal blooms (red tide) from CHL anomalies.
Advisory: fish may flee or become toxic in HAB zones.

Uses existing CHL + SST data — zero new APIs.
Ref: Stumpf et al. (2003) Harmful Algae 2:311-317
"""

import numpy as np
import logging
from typing import Dict, Any

log = logging.getLogger("OceanMaster.HAB")


class HABDetector:
    """Detect potential Harmful Algal Blooms from CHL anomalies."""

    # CHL thresholds for HAB risk
    CHL_MODERATE = 5.0    # mg/m³ — moderate bloom
    CHL_HIGH = 15.0       # mg/m³ — high bloom (potential HAB)
    CHL_EXTREME = 30.0    # mg/m³ — extreme (very likely HAB)

    # SST range where HABs are most common (subtropical)
    SST_HAB_MIN = 20.0
    SST_HAB_MAX = 32.0

    def detect(
        self,
        chl: np.ndarray,
        sst: np.ndarray,
        chl_climatology: np.ndarray = None,
    ) -> Dict[str, Any]:
        """
        Detect HAB risk zones.

        Returns:
            {
                "hab_risk": 2D float (0-1),
                "hab_level": 2D int (0=none, 1=moderate, 2=high, 3=extreme),
                "hab_mask": 2D bool (True=potential HAB),
                "affected_pct": float,
                "advisory": str,
            }
        """
        chl_safe = np.nan_to_num(chl, nan=0.0)
        sst_safe = np.nan_to_num(sst, nan=25.0)

        # CHL anomaly
        if chl_climatology is not None:
            chl_anom = chl_safe - chl_climatology
        else:
            chl_anom = chl_safe - np.nanmedian(chl_safe)

        # Risk score: high CHL + favorable SST = higher risk
        sst_factor = np.where(
            (sst_safe >= self.SST_HAB_MIN) & (sst_safe <= self.SST_HAB_MAX),
            1.0, 0.3
        )

        risk = np.clip(chl_safe / self.CHL_EXTREME, 0, 1) * sst_factor
        risk = risk.astype(np.float32)

        # Level
        level = np.zeros_like(chl_safe, dtype=np.int32)
        level[chl_safe >= self.CHL_MODERATE] = 1
        level[chl_safe >= self.CHL_HIGH] = 2
        level[chl_safe >= self.CHL_EXTREME] = 3

        hab_mask = level >= 2
        affected = float(np.sum(hab_mask)) / max(chl_safe.size, 1) * 100
        max_lev = int(np.max(level))

        names = {0: "無", 1: "中度藻華", 2: "高度藻華(可能赤潮)", 3: "極端藻華(赤潮)"}
        if max_lev >= 2:
            advisory = f"⚠️ {names[max_lev]} — 魚群可能逃離或具毒性風險"
        elif max_lev == 1:
            advisory = "🌊 中度藻華，注意水質"
        else:
            advisory = "水質正常"

        log.info(f"  HAB: level={max_lev}, affected={affected:.1f}%")
        return {
            "hab_risk": risk,
            "hab_level": level,
            "hab_mask": hab_mask,
            "affected_pct": round(affected, 1),
            "advisory": advisory,
        }
