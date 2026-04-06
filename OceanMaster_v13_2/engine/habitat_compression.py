"""
OceanMaster v13.2 — Habitat Compression Index [#3]
====================================================
When OMZ shoals (low DO rises) + MLD deepens → pelagic habitat compresses.
Fish are squeezed into a thin surface layer → CPUE increases dramatically.

Ref: Stramma et al. (2012) Nature Climate Change 2:33-37
     Prince & Goodyear (2006) Fish. Oceanogr. 15:451-460
"""

import numpy as np
import logging
from typing import Dict, Any, Optional

log = logging.getLogger("OceanMaster.HCI")

# Species depth preferences (m)
SPECIES_DEPTH = {
    "skipjack":  {"min": 0, "max": 150, "preferred": 80},
    "yellowfin": {"min": 0, "max": 250, "preferred": 120},
    "bigeye":    {"min": 50, "max": 500, "preferred": 300},
    "albacore":  {"min": 30, "max": 350, "preferred": 200},
    "swordfish": {"min": 100, "max": 800, "preferred": 400},
}


class HabitatCompressionAnalyzer:
    """
    [v16.0 #3] Compute habitat compression index.

    HCI = available_habitat / species_preferred_range
    When HCI < 0.5 → severe compression → CPUE boost 2-4x
    When HCI > 1.0 → no compression → normal fishing

    Available habitat = OMZ_depth - MLD (vertical space with enough O₂)
    """

    def __init__(self, do_threshold: float = 2.0):
        """
        Args:
            do_threshold: DO level (ml/L) below which habitat is uninhabitable.
        """
        self.do_threshold = do_threshold

    def compute(
        self,
        mld: np.ndarray,
        do_surface: np.ndarray,
        species: str = "yellowfin",
        z20: Optional[np.ndarray] = None,
        temp_3d: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Compute habitat compression index.

        Args:
            mld: Mixed layer depth (m)
            do_surface: Surface dissolved oxygen (ml/L)
            species: Target species
            z20: 20°C isotherm depth (m) — proxy for OMZ upper bound
            temp_3d: 3D temperature (optional, for better OMZ estimation)

        Returns:
            {
                "hci": 2D compression index (0-2, <0.5 = severe compression),
                "available_depth": 2D (m, habitable vertical range),
                "omz_depth_est": 2D (m, estimated OMZ upper boundary),
                "compression_mask": 2D bool (HCI < 0.5),
                "compression_pct": float,
                "cpue_boost": 2D (estimated CPUE multiplier),
                "advisory": str,
            }
        """
        ny, nx = mld.shape
        sp_info = SPECIES_DEPTH.get(species, SPECIES_DEPTH["yellowfin"])
        pref_range = sp_info["max"] - sp_info["min"]

        # Estimate OMZ upper boundary depth
        # Method 1: If Z20 available, OMZ ≈ Z20 + offset (deeper than thermocline)
        if z20 is not None:
            omz_depth = z20 + 100  # OMZ typically 100m below Z20
        else:
            # Method 2: Estimate from surface DO
            # Lower surface DO → shallower OMZ
            # DO=5+ → OMZ at ~500m; DO=2 → OMZ at ~150m
            do_safe = np.clip(do_surface, 0.5, 8.0)
            omz_depth = 100 + 80 * do_safe  # rough linear mapping
        omz_depth = np.clip(omz_depth, 50, 1000).astype(np.float32)

        # Available habitat = OMZ_depth - MLD
        mld_safe = np.clip(mld, 5, 500)
        available = np.clip(omz_depth - mld_safe, 0, 1000).astype(np.float32)

        # HCI = available / preferred_range
        hci = (available / max(pref_range, 10)).astype(np.float32)
        hci = np.clip(hci, 0, 3.0)

        # Compression detection
        compression = hci < 0.5
        compression_pct = float(np.mean(compression) * 100)

        # CPUE boost estimate (Stramma et al. 2012)
        # HCI < 0.3 → ~4x; HCI < 0.5 → ~2x; HCI > 1 → 1x
        cpue_boost = np.ones_like(hci)
        cpue_boost[hci < 1.0] = 1.0 + (1.0 - hci[hci < 1.0]) * 2
        cpue_boost[hci < 0.5] = 2.0 + (0.5 - hci[hci < 0.5]) * 4
        cpue_boost = np.clip(cpue_boost, 1.0, 5.0).astype(np.float32)

        if compression_pct > 20:
            advisory = f"嚴重棲息壓縮: {compression_pct:.0f}% — 魚群被擠到表層，CPUE 預計 2-4x"
        elif compression_pct > 5:
            advisory = f"部分棲息壓縮: {compression_pct:.0f}%"
        else:
            advisory = "棲息空間充足 — 無壓縮效應"

        log.info(f"  HCI ({species}): compression={compression_pct:.1f}%, "
                 f"mean HCI={np.nanmean(hci):.2f}, "
                 f"mean boost={np.nanmean(cpue_boost):.1f}x")

        return {
            "hci": hci,
            "available_depth": available,
            "omz_depth_est": omz_depth,
            "compression_mask": compression,
            "compression_pct": round(compression_pct, 1),
            "cpue_boost": cpue_boost,
            "advisory": advisory,
        }
