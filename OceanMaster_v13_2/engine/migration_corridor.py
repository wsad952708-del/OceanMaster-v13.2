"""
OceanMaster v13.2 — Tuna Migration Corridor Predictor
=======================================================
Predict tuna migration corridors from SST isotherms and current patterns.

Key insight: tunas follow preferred SST isotherms as they migrate seasonally.
  - Skipjack: follows 26-29°C isotherm
  - Yellowfin: follows 24-28°C isotherm
  - Bigeye: follows 20-26°C isotherm (wider range due to deep diving)
  - Albacore: follows 15-20°C isotherm (temperate species)

Corridors = where preferred isotherm aligns with favorable current direction.

Ref: Lehodey et al. (1997) Nature 389:715-718
     Fromentin & Fonteneau (2001) ICES J. Mar. Sci. 58:1231-1240
"""

import numpy as np
import logging
from typing import Dict, Any, Optional

log = logging.getLogger("OceanMaster.MigCor")

SPECIES_SST_CORRIDORS = {
    "skipjack":  {"min": 26.0, "max": 29.0, "optimal": 27.5},
    "yellowfin": {"min": 24.0, "max": 28.0, "optimal": 26.0},
    "bigeye":    {"min": 20.0, "max": 26.0, "optimal": 23.0},
    "albacore":  {"min": 15.0, "max": 20.0, "optimal": 17.5},
    "swordfish": {"min": 13.0, "max": 25.0, "optimal": 18.0},
}


class MigrationCorridorPredictor:
    """
    [v16.0] Predict tuna migration corridors from SST isotherms + currents.

    A corridor exists where:
    1. SST is within species' preferred range
    2. SST gradient is moderate (isotherm = path, not barrier)
    3. Current direction aligns with isotherm direction (fish riding the current)
    """

    def compute(
        self,
        sst: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        species: str = "yellowfin",
        u: Optional[np.ndarray] = None,
        v: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Predict migration corridor.

        Args:
            sst: 2D SST grid (°C)
            lats, lons: 1D coordinate arrays
            species: target species
            u, v: optional current velocity

        Returns:
            {
                "corridor_mask": 2D bool,
                "corridor_strength": 2D (0-1),
                "isotherm_optimal": 2D (distance to optimal SST),
                "current_alignment": 2D (0-1, how well current aligns with isotherm),
                "corridor_pct": float,
                "migration_direction": str,
                "advisory": str,
            }
        """
        ny, nx = sst.shape
        sp = SPECIES_SST_CORRIDORS.get(species, SPECIES_SST_CORRIDORS["yellowfin"])

        # Layer 1: SST within preferred range (Gaussian around optimal)
        sst_safe = np.nan_to_num(sst, nan=25.0)
        sigma = (sp["max"] - sp["min"]) / 4
        sst_score = np.exp(-0.5 * ((sst_safe - sp["optimal"]) / max(sigma, 0.5)) ** 2)
        sst_score = sst_score.astype(np.float32)

        # Layer 2: SST gradient direction (isotherms = perpendicular to gradient)
        gy, gx = np.gradient(sst_safe)
        grad_mag = np.sqrt(gx**2 + gy**2)
        grad_mag_safe = np.clip(grad_mag, 1e-6, None)

        # Moderate gradient is better (too strong = front barrier, too weak = no corridor)
        grad_mean = max(float(np.mean(grad_mag)), 1e-6)
        grad_score = np.exp(-0.5 * ((grad_mag - grad_mean) / max(grad_mean, 0.01)) ** 2)
        grad_score = grad_score.astype(np.float32)

        # Layer 3: Current alignment with isotherm direction
        if u is not None and v is not None:
            # Isotherm direction = perpendicular to SST gradient
            iso_dx = -gy / grad_mag_safe  # perpendicular
            iso_dy = gx / grad_mag_safe

            # Current direction
            cur_mag = np.sqrt(u**2 + v**2)
            cur_mag_safe = np.clip(cur_mag, 1e-6, None)
            cur_dx = u / cur_mag_safe
            cur_dy = v / cur_mag_safe

            # Alignment = |cos(angle between current and isotherm)|
            alignment = np.abs(cur_dx * iso_dx + cur_dy * iso_dy)
            alignment = alignment.astype(np.float32)
        else:
            alignment = np.ones((ny, nx), dtype=np.float32) * 0.5

        # Composite corridor strength
        corridor = (0.5 * sst_score + 0.25 * grad_score + 0.25 * alignment)
        corridor = corridor.astype(np.float32)

        # Corridor mask (strength > 0.5)
        corridor_mask = corridor > 0.5
        corridor_pct = float(np.mean(corridor_mask) * 100)

        # Migration direction (dominant current direction in corridor)
        direction = "不明"
        if u is not None and v is not None and np.any(corridor_mask):
            u_mean = float(np.nanmean(u[corridor_mask]))
            v_mean = float(np.nanmean(v[corridor_mask]))
            angle = np.degrees(np.arctan2(v_mean, u_mean))
            if -45 <= angle < 45:
                direction = "東向"
            elif 45 <= angle < 135:
                direction = "北向"
            elif angle >= 135 or angle < -135:
                direction = "西向"
            else:
                direction = "南向"

        if corridor_pct > 30:
            advisory = f"寬廣遷徙走廊: {corridor_pct:.0f}% ({sp['min']}-{sp['max']}°C), 方向: {direction}"
        elif corridor_pct > 10:
            advisory = f"中等遷徙走廊: {corridor_pct:.0f}%, 方向: {direction}"
        else:
            advisory = f"走廊狹窄或不在本區域 — {species} 可能在更遠處"

        log.info(f"  Corridor ({species}): {corridor_pct:.1f}%, "
                 f"dir={direction}, sst_range={sp['min']}-{sp['max']}°C")

        return {
            "corridor_mask": corridor_mask,
            "corridor_strength": corridor,
            "isotherm_score": sst_score,
            "current_alignment": alignment,
            "corridor_pct": round(corridor_pct, 1),
            "migration_direction": direction,
            "sst_range": (sp["min"], sp["max"]),
            "advisory": advisory,
        }
