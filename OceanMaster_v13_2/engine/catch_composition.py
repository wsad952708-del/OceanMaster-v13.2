"""
OceanMaster v13.2 — Catch Composition Forecast [H2]
=====================================================
Predict species composition based on environmental conditions.
Uses species-specific thermal/depth preferences.

Ref: Lehodey et al. (2006) Prog. Oceanogr. 71:168-201
"""

import numpy as np
import logging
from typing import Dict, List, Any

log = logging.getLogger("OceanMaster.CatchComp")

# Species thermal/depth preference profiles
SPECIES_PROFILES = {
    "yellowfin": {
        "name_zh": "黄鰭鮪",
        "sst_optimal": (26, 30),
        "depth_range": (0, 150),
        "do_min": 3.5,
        "chl_preference": "moderate",  # 0.1-0.5 mg/m³
    },
    "bigeye": {
        "name_zh": "大目鮪",
        "sst_optimal": (22, 28),
        "depth_range": (100, 400),
        "do_min": 2.0,
        "chl_preference": "moderate",
    },
    "skipjack": {
        "name_zh": "正鰹",
        "sst_optimal": (27, 31),
        "depth_range": (0, 100),
        "do_min": 4.0,
        "chl_preference": "high",  # > 0.3 mg/m³
    },
    "albacore": {
        "name_zh": "長鰭鮪",
        "sst_optimal": (18, 24),
        "depth_range": (50, 300),
        "do_min": 3.0,
        "chl_preference": "low",  # < 0.2 mg/m³
    },
    "swordfish": {
        "name_zh": "旗魚",
        "sst_optimal": (20, 28),
        "depth_range": (200, 800),
        "do_min": 2.0,
        "chl_preference": "moderate",
    },
}


class CatchCompositionForecaster:
    """[v16.0 H2] Predict species probability based on local conditions."""

    def __init__(self, species_list: List[str] = None):
        self.species = species_list or list(SPECIES_PROFILES.keys())

    def forecast(
        self,
        sst: float,
        depth: float = -3000.0,
        do_ml_per_l: float = 4.0,
        chl: float = 0.3,
        mld: float = 50.0,
    ) -> Dict[str, Any]:
        """
        Predict species probability breakdown.

        Args:
            sst: SST (°C)
            depth: bottom depth (m, negative or positive)
            do_ml_per_l: dissolved oxygen (ml/L)
            chl: chlorophyll (mg/m³)
            mld: mixed layer depth (m)

        Returns:
            {
                "probabilities": {species: float},
                "dominant_species": str,
                "composition_advisory": str,
            }
        """
        depth_abs = abs(depth)
        probs = {}

        for sp in self.species:
            prof = SPECIES_PROFILES.get(sp)
            if prof is None:
                continue

            # SST suitability (Gaussian decay from optimal range)
            sst_lo, sst_hi = prof["sst_optimal"]
            sst_mid = (sst_lo + sst_hi) / 2
            sst_sigma = (sst_hi - sst_lo) / 2
            sst_score = np.exp(-0.5 * ((sst - sst_mid) / max(sst_sigma, 1)) ** 2)

            # Depth suitability
            d_lo, d_hi = prof["depth_range"]
            if d_lo <= depth_abs <= d_hi:
                depth_score = 1.0
            elif depth_abs < d_lo:
                depth_score = max(0, 1.0 - (d_lo - depth_abs) / 200)
            else:
                depth_score = max(0, 1.0 - (depth_abs - d_hi) / 500)

            # DO suitability
            do_score = min(1.0, do_ml_per_l / max(prof["do_min"], 0.1))

            # CHL preference
            if prof["chl_preference"] == "high":
                chl_score = min(1.0, chl / 0.3)
            elif prof["chl_preference"] == "low":
                chl_score = min(1.0, 0.2 / max(chl, 0.01))
            else:
                chl_score = 1.0 - abs(chl - 0.3) / 0.5
                chl_score = max(0, min(1.0, chl_score))

            probs[sp] = float(sst_score * depth_score * do_score * chl_score)

        # Normalize to percentages
        total = sum(probs.values())
        if total > 0:
            probs = {sp: round(v / total * 100, 1) for sp, v in probs.items()}
        else:
            probs = {sp: round(100 / len(self.species), 1) for sp in self.species}

        dominant = max(probs, key=probs.get) if probs else "unknown"
        dom_pct = probs.get(dominant, 0)
        dom_zh = SPECIES_PROFILES.get(dominant, {}).get("name_zh", dominant)

        advisory = f"主要物種: {dom_zh} ({dom_pct:.0f}%)"
        if dom_pct < 30:
            advisory += " — 混合漁場，多物種可能"

        return {
            "probabilities": probs,
            "dominant_species": dominant,
            "dominant_pct": dom_pct,
            "composition_advisory": advisory,
        }
