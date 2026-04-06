"""
OceanMaster v13.2 — Water Mass Classifier
===========================================
Classify ocean grid points into water mass types from T-S properties.

Water mass classification tells fishing companies:
  "You're fishing in Kuroshio water" or "North Pacific Intermediate Water"
  Each water mass has different productivity and species associations.

Key water masses in Western Pacific:
  - Kuroshio Current Water (KCW): warm, saline → yellowfin, skipjack
  - North Pacific Subtropical Mode Water (STMW): 16-19°C, 34.7-34.9 PSU → albacore
  - North Pacific Intermediate Water (NPIW): 5-10°C, 34.0-34.5 PSU → deep bigeye
  - South Pacific Tropical Water (SPTW): >25°C, >35.5 PSU → skipjack
  - Equatorial Under-Current (EUC): cool, low salinity → upwelling zones

Ref: Suga et al. (2000) J. Phys. Oceanogr. 30:784-798
     Hanawa & Talley (2001) Ocean Circulation & Climate
"""

import numpy as np
import logging
from typing import Dict, Any

log = logging.getLogger("OceanMaster.WaterMass")

# T-S envelopes for each water mass (simplified for surface classification)
WATER_MASSES = {
    "Kuroshio Current": {
        "t_min": 24, "t_max": 30, "s_min": 34.5, "s_max": 35.2,
        "species": ["yellowfin", "skipjack", "bigeye"],
        "productivity": "高",
    },
    "North Pacific Subtropical": {
        "t_min": 18, "t_max": 25, "s_min": 34.8, "s_max": 35.5,
        "species": ["yellowfin", "albacore"],
        "productivity": "中",
    },
    "STMW (副熱帶模態水)": {
        "t_min": 15, "t_max": 19, "s_min": 34.6, "s_max": 34.9,
        "species": ["albacore", "swordfish"],
        "productivity": "中低",
    },
    "Equatorial (赤道水)": {
        "t_min": 27, "t_max": 32, "s_min": 33.5, "s_max": 35.0,
        "species": ["skipjack", "yellowfin"],
        "productivity": "高",
    },
    "South China Sea": {
        "t_min": 25, "t_max": 31, "s_min": 33.0, "s_max": 34.5,
        "species": ["yellowfin", "skipjack"],
        "productivity": "中高",
    },
    "Cold Subpolar": {
        "t_min": 5, "t_max": 15, "s_min": 33.0, "s_max": 34.5,
        "species": ["albacore"],
        "productivity": "低",
    },
}


class WaterMassClassifier:
    """
    [v16.0] Classify each grid point into water mass type from T-S properties.
    """

    def classify(
        self,
        sst: np.ndarray,
        salinity: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Classify water masses.

        Args:
            sst: 2D SST grid (°C)
            salinity: 2D salinity grid (PSU)

        Returns:
            {
                "classification": 2D int (0-N, each class ID),
                "class_names": list of str,
                "class_map": 2D of class names (as object array),
                "dominant_mass": str (most common),
                "species_by_mass": dict,
                "advisory": str,
            }
        """
        ny, nx = sst.shape
        sst_safe = np.nan_to_num(sst, nan=25.0)
        sal_safe = np.nan_to_num(salinity, nan=35.0)

        class_names = ["Unknown"] + list(WATER_MASSES.keys())
        classification = np.zeros((ny, nx), dtype=np.int8)
        best_distance = np.full((ny, nx), 999.0, dtype=np.float32)

        for idx, (name, props) in enumerate(WATER_MASSES.items(), start=1):
            t_center = (props["t_min"] + props["t_max"]) / 2
            s_center = (props["s_min"] + props["s_max"]) / 2
            t_range = max(props["t_max"] - props["t_min"], 1)
            s_range = max(props["s_max"] - props["s_min"], 0.1)

            # Normalized distance in T-S space
            dt = (sst_safe - t_center) / t_range
            ds = (sal_safe - s_center) / s_range
            dist = np.sqrt(dt**2 + ds**2)

            # Check if within envelope
            in_range = (
                (sst_safe >= props["t_min"] - 2) & (sst_safe <= props["t_max"] + 2) &
                (sal_safe >= props["s_min"] - 0.5) & (sal_safe <= props["s_max"] + 0.5)
            )

            # Assign closest water mass
            update = in_range & (dist < best_distance)
            classification[update] = idx
            best_distance[update] = dist[update]

        # Dominant water mass
        counts = np.bincount(classification.ravel(), minlength=len(class_names))
        dominant_idx = int(np.argmax(counts[1:]) + 1) if counts[1:].max() > 0 else 0
        dominant = class_names[dominant_idx]

        # Species associations
        species_by_mass = {}
        for name, props in WATER_MASSES.items():
            species_by_mass[name] = props["species"]

        pct_classified = float(np.mean(classification > 0) * 100)
        advisory = f"主要水團: {dominant} ({pct_classified:.0f}% 已分類)"

        log.info(f"  Water mass: dominant={dominant}, classified={pct_classified:.0f}%")
        for idx, name in enumerate(class_names):
            pct = float(np.mean(classification == idx) * 100)
            if pct > 1:
                log.info(f"    {name}: {pct:.1f}%")

        return {
            "classification": classification,
            "class_names": class_names,
            "dominant_mass": dominant,
            "species_by_mass": species_by_mass,
            "classified_pct": round(pct_classified, 1),
            "advisory": advisory,
        }
