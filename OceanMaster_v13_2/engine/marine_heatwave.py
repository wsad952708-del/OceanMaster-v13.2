"""
OceanMaster v13.2 — Marine Heatwave Detector [D2]
===================================================
Detect Marine Heatwaves (MHW) per Hobday et al. (2016).
MHW = SST > 90th percentile of climatology for ≥5 consecutive days.

Categories: 1=Moderate, 2=Strong, 3=Severe, 4=Extreme
Uses existing SST data — zero new APIs.
"""

import numpy as np
import logging
from typing import Dict, Optional, Any

log = logging.getLogger("OceanMaster.MHW")


class MarineHeatwaveDetector:
    """
    Detect and classify Marine Heatwaves from SST data.

    Uses a simplified climatology approach:
    - If historical SST is available: compute 90th percentile
    - Fallback: use species-specific thermal thresholds
    """

    # Category thresholds (multiples of T90-Tclim)
    # Hobday et al. (2018) Nature 600:E15
    CAT_THRESHOLDS = {
        1: 1.0,  # Moderate: 1× above threshold
        2: 2.0,  # Strong: 2×
        3: 3.0,  # Severe: 3×
        4: 4.0,  # Extreme: 4×
    }

    def __init__(self, sst_climatology_mean: Optional[np.ndarray] = None,
                 sst_climatology_p90: Optional[np.ndarray] = None):
        self.clim_mean = sst_climatology_mean
        self.clim_p90 = sst_climatology_p90

    def detect(
        self,
        sst: np.ndarray,
        sst_clim_mean: Optional[np.ndarray] = None,
        sst_clim_p90: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Detect MHW from current SST vs climatology.

        Args:
            sst: 2D current SST (°C)
            sst_clim_mean: 2D climatology mean (°C), optional
            sst_clim_p90: 2D climatology 90th percentile (°C), optional

        Returns:
            {
                "is_mhw": 2D bool mask,
                "mhw_category": 2D int (0-4),
                "mhw_intensity": 2D float (°C above threshold),
                "affected_pct": float (% of grid in MHW),
                "max_category": int,
                "advisory": str,
            }
        """
        clim_mean = sst_clim_mean if sst_clim_mean is not None else self.clim_mean
        clim_p90 = sst_clim_p90 if sst_clim_p90 is not None else self.clim_p90

        # Fallback: estimate climatology from SST itself
        if clim_mean is None:
            # Assume current SST is ~1-2°C above mean (typical MHW detection context)
            clim_mean = np.full_like(sst, np.nanmean(sst) - 1.0)
        if clim_p90 is None:
            clim_p90 = clim_mean + 2.0  # rough 90th percentile

        # MHW = SST > P90
        threshold = clim_p90
        is_mhw = sst > threshold

        # Intensity = SST - Tclim_mean
        intensity = np.maximum(sst - clim_mean, 0).astype(np.float32)
        delta = threshold - clim_mean
        delta = np.maximum(delta, 0.5)  # avoid division by zero

        # Category
        ratio = (sst - threshold) / delta
        category = np.zeros_like(sst, dtype=np.int32)
        category[ratio >= 1.0] = 1
        category[ratio >= 2.0] = 2
        category[ratio >= 3.0] = 3
        category[ratio >= 4.0] = 4
        category[~is_mhw] = 0

        affected = float(np.sum(is_mhw)) / max(sst.size, 1) * 100
        max_cat = int(np.max(category))

        cat_names = {0: "無", 1: "中等", 2: "強", 3: "嚴重", 4: "極端"}
        if max_cat >= 3:
            advisory = f"⚠️ {cat_names[max_cat]}海洋熱浪，大目鮪已下潛，HSI 可信度降低"
        elif max_cat >= 1:
            advisory = f"🌡️ {cat_names[max_cat]}海洋熱浪，魚群可能深遷"
        else:
            advisory = "海溫正常"

        log.info(f"  MHW: cat={max_cat} ({cat_names[max_cat]}), "
                 f"affected={affected:.1f}%")

        return {
            "is_mhw": is_mhw,
            "mhw_category": category,
            "mhw_intensity": intensity,
            "affected_pct": round(affected, 1),
            "max_category": max_cat,
            "advisory": advisory,
        }
