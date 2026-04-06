"""
OceanMaster v13.2 — AIS Competition Density Penalty
=====================================================
V3.0 #4: Penalize hotspots with high nearby fishing vessel density.

Concept:
  ais_fishing_density (existing ML feature) is a POSITIVE signal:
  "more boats = more fish found here"

  Competition penalty is the INVERSE:
  "too many boats = overcrowded, go elsewhere for hidden grounds"

  Score adjustment:
    penalty = -competition_weight * saturate(vessel_count / threshold)

  Default: 50km radius, threshold=10 vessels, max penalty 20%

Data source:
  Uses ais_fishing_density from ocean_data (GFW API or heatmap).
  If unavailable, uses vessel_count from AIS shadow fishing events.

Usage:
  from engine.competition_penalty import CompetitionPenalty
  cp = CompetitionPenalty()
  hotspots = cp.apply_penalty(hotspots, ocean_data)
"""

import os
import logging
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger("OceanMaster.Competition")


class CompetitionPenalty:
    """
    AIS-based competition density penalty for hotspots.

    Hotspots near high vessel density get score reduction,
    pushing "hidden" low-competition spots higher in ranking.
    """

    def __init__(
        self,
        radius_km: float = 50.0,
        vessel_threshold: int = 10,
        max_penalty_pct: float = 0.20,
    ):
        """
        Args:
            radius_km: search radius for counting competitor vessels
            vessel_threshold: number of vessels at which max penalty applies
            max_penalty_pct: maximum score reduction (0.20 = 20%)
        """
        # Allow env var overrides for tuning
        self.radius_km = float(os.getenv(
            "COMPETITION_RADIUS_KM", str(radius_km)))
        self.vessel_threshold = int(os.getenv(
            "COMPETITION_VESSEL_THRESHOLD", str(vessel_threshold)))
        self.max_penalty_pct = float(os.getenv(
            "COMPETITION_MAX_PENALTY", str(max_penalty_pct)))

        log.info(
            f"Competition penalty: radius={self.radius_km}km, "
            f"threshold={self.vessel_threshold} vessels, "
            f"max_penalty={self.max_penalty_pct*100:.0f}%"
        )

    def estimate_vessel_count(
        self,
        lat: float,
        lon: float,
        ais_density_grid: Optional[np.ndarray] = None,
        lat_arr: Optional[np.ndarray] = None,
        lon_arr: Optional[np.ndarray] = None,
    ) -> float:
        """
        Estimate number of competing vessels near (lat, lon).

        Uses ais_fishing_density grid if available,
        otherwise returns 0 (no penalty applied).

        The ais_fishing_density values are typically 0-5 scale
        where 1.0 ~ 3-5 vessels, 3.0 ~ 15+ vessels.
        """
        if ais_density_grid is None or lat_arr is None or lon_arr is None:
            return 0.0

        # Find nearest grid cell
        try:
            iy = int(np.argmin(np.abs(lat_arr - lat)))
            ix = int(np.argmin(np.abs(lon_arr - lon)))

            if iy >= ais_density_grid.shape[0] or ix >= ais_density_grid.shape[1]:
                return 0.0

            # Sample density in surrounding cells (~50km radius)
            # At ~0.25 deg resolution, 50km ~ 2 cells
            r_cells = max(1, int(self.radius_km / 25))
            iy_lo = max(0, iy - r_cells)
            iy_hi = min(ais_density_grid.shape[0], iy + r_cells + 1)
            ix_lo = max(0, ix - r_cells)
            ix_hi = min(ais_density_grid.shape[1], ix + r_cells + 1)

            region = ais_density_grid[iy_lo:iy_hi, ix_lo:ix_hi]
            density = float(np.nanmean(region))

            # [v13.2] density→vessel 換算係數
            # 量綱: GFW 4Wings apparent_fishing_hours 歸一化後 density ∈ [0,5]
            # density=1.0 ≈ 3-5 vessels/cell (基於 WCPFC 2019-2023 年報交叉比對)
            # factor=4.0 取中間值。不同海域 (如西太平洋 vs 印度洋) 可能需要微調。
            # 可通過環境變數 COMPETITION_DENSITY_FACTOR 覆蓋。
            import os
            DENSITY_TO_VESSEL_FACTOR = float(os.environ.get("COMPETITION_DENSITY_FACTOR", "4.0"))
            estimated_vessels = density * DENSITY_TO_VESSEL_FACTOR

            return max(0.0, estimated_vessels)

        except (IndexError, ValueError):
            return 0.0

    def compute_penalty(self, vessel_count: float) -> float:
        """
        Compute score penalty factor (0 to max_penalty_pct).

        Returns:
            Penalty as fraction (e.g., 0.15 = 15% reduction)
        """
        if vessel_count <= 0:
            return 0.0

        # Saturating curve: penalty = max_penalty * tanh(count / threshold)
        saturation = min(vessel_count / self.vessel_threshold, 1.0)
        return self.max_penalty_pct * saturation

    def apply_penalty(
        self,
        hotspots: List[Dict[str, Any]],
        ocean_data: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Apply competition penalty to all hotspots.

        Adds to each hotspot:
          - competition_vessels: estimated nearby vessel count
          - competition_penalty: penalty fraction applied
          - score_before_competition: original score

        Ocean_data should contain:
          - ais_fishing_density: 2D grid
          - lat, lon: 1D coordinate arrays
        """
        ocean_data = ocean_data or {}
        ais_grid = ocean_data.get("ais_fishing_density")
        lat_arr = ocean_data.get("lat")
        lon_arr = ocean_data.get("lon")

        # Convert lat/lon to numpy arrays if needed
        if lat_arr is not None and not isinstance(lat_arr, np.ndarray):
            lat_arr = np.array(lat_arr)
        if lon_arr is not None and not isinstance(lon_arr, np.ndarray):
            lon_arr = np.array(lon_arr)

        n_penalized = 0

        for hs in hotspots:
            lat = hs.get("lat", 0)
            lon = hs.get("lon", 0)

            vessel_count = self.estimate_vessel_count(
                lat, lon, ais_grid, lat_arr, lon_arr
            )
            penalty = self.compute_penalty(vessel_count)

            hs["competition_vessels"] = round(vessel_count, 1)
            hs["competition_penalty"] = round(penalty, 4)

            if penalty > 0:
                original_score = hs.get("score", 0)
                hs["score_before_competition"] = original_score
                hs["score"] = round(original_score * (1 - penalty), 4)
                n_penalized += 1

        # Re-sort by adjusted score
        hotspots.sort(key=lambda x: x.get("score", 0), reverse=True)
        for i, hs in enumerate(hotspots):
            hs["rank"] = i + 1

        if n_penalized:
            log.info(
                f"  Competition penalty: {n_penalized}/{len(hotspots)} "
                f"hotspots penalized (radius={self.radius_km}km)"
            )

        return hotspots

    def compute_fatigue_penalty(
        self,
        vessel_count: float,
        consecutive_heavy_days: int = 0,
    ) -> float:
        """
        [v18] Fishing pressure fatigue penalty.

        If a hotspot has been heavily fished for multiple consecutive days,
        fish stocks may be locally depleted.

        Ref: Catchwise AIS temporal analysis concept.

        Args:
            vessel_count: current vessel count near hotspot
            consecutive_heavy_days: days of continuous heavy fishing (>=5 vessels/day)

        Returns:
            Additional penalty fraction (0 to 0.30)
        """
        if consecutive_heavy_days <= 1:
            return 0.0
        # 每天連續重度作業 → 額外扣分
        # 7 天 → 0.30 (最大); 3 天 → 0.09
        fatigue = min(0.30, 0.01 * consecutive_heavy_days**2)
        return fatigue


# Global singleton
_global_competition: Optional[CompetitionPenalty] = None


def get_competition_penalty() -> CompetitionPenalty:
    """Get global CompetitionPenalty instance."""
    global _global_competition
    if _global_competition is None:
        _global_competition = CompetitionPenalty()
    return _global_competition
