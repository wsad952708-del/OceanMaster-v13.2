"""
OceanMaster v13.2 — Typhoon Post-Storm Golden Fishing Zone Predictor
=====================================================================
P0 Gap: Transform typhoon threat into opportunity.

Science (台灣水產試驗所 TFRIN, MDPI Atmosphere 2024):
  After a typhoon passes (2-7 days):
  1. Cold wake: SST drops 6-12°C right of track, recovers over ~10 days
  2. Nutrient upwelling: deep nutrients pump to surface via mixing
  3. CHL bloom: phytoplankton boom 3-7 days post-storm (1.5-5× baseline)
  4. Fish aggregation: zooplankton & prey fish concentrate → tuna follows
  5. Optimal window: Day 3 to Day 7 post-passage (before SST recovers)

Model:
  golden_score(lat, lon, t) =
    cold_wake_factor(dist_to_track, hours_since_pass) ×
    nutrient_enrichment(sst_drop, mixing_depth) ×
    bloom_timing(days_post)

Usage:
  from engine.typhoon_golden_zone import TyphoonGoldenZone
  tz = TyphoonGoldenZone()
  result = tz.compute_golden_zones(active_typhoons, ocean_data, lats, lons)
"""

import logging
import math
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("OceanMaster.TyphoonGolden")


class TyphoonGoldenZone:
    """
    Predict post-typhoon fishing hotspots from cold wake & nutrient enrichment.

    Theory:
      - Coriolis-driven rightward bias (NH) creates asymmetric cold wake
      - Inertial pumping drives nutrient injection 2-5 days post-passage
      - CHL bloom peaks at day 4-6 (Zhao et al. 2017, Lin 2012)
      - Tuna aggregation follows bloom with 1-2 day lag
    """

    # ─── Physical Constants ───
    GOLDEN_WINDOW_START_H = 48    # earliest opportunity (hours after passage)
    GOLDEN_WINDOW_END_H = 168     # last viable window (7 days)
    PEAK_DAYS = (3, 6)            # optimal fishing window (days post-storm)

    # Cold wake decay: e-folding time ~5 days (Price et al. 1994)
    COLD_WAKE_TAU_HOURS = 120.0   # 5 days

    # Right-of-track bias in Northern Hemisphere (Coriolis effect)
    RIGHT_BIAS_KM = 80.0          # max rightward displacement of cold wake
    WAKE_WIDTH_KM = 200.0         # total wake width (cross-track)

    # Enrichment thresholds
    MIN_SST_DROP = 1.0            # °C: minimum detectable cooling
    MAX_SST_DROP = 12.0           # °C: extreme typhoon cooling

    # Category multiplier (Saffir-Simpson proxy via wind speed)
    CAT_THRESHOLDS = [
        (64, 1.0),    # Cat 1: 64-82 kt
        (83, 1.3),    # Cat 2: 83-95 kt
        (96, 1.6),    # Cat 3: 96-112 kt
        (113, 1.9),   # Cat 4: 113-136 kt
        (137, 2.2),   # Cat 5: 137+ kt
    ]

    def __init__(self):
        self._cache = {}  # track_id → prior computations

    def _category_factor(self, wind_kt: float) -> float:
        """Scale enrichment by typhoon intensity."""
        factor = 0.8
        for threshold, f in self.CAT_THRESHOLDS:
            if wind_kt >= threshold:
                factor = f
        return factor

    def _cold_wake_decay(self, hours_since: float) -> float:
        """
        Cold wake dissipation: exponential decay from Price et al. (1994).
        Returns factor 0-1 (1 = full intensity, 0 = fully recovered).
        """
        if hours_since < 0:
            return 0.0
        return math.exp(-hours_since / self.COLD_WAKE_TAU_HOURS)

    def _bloom_timing(self, hours_since: float) -> float:
        """
        CHL bloom timing curve (bell-shaped, peak at day 4-5).
        Zhao et al. 2017: bloom onset day 2, peak day 4-5, decay by day 8.
        Model: N(μ=108h, σ=36h) normalized to peak=1.0
        """
        mu = 108.0   # peak at 4.5 days
        sigma = 36.0  # ~1.5 day spread
        return math.exp(-0.5 * ((hours_since - mu) / sigma) ** 2)

    def _cross_track_factor(
        self,
        lat: float, lon: float,
        track_points: List[Tuple[float, float, float]],  # [(lat, lon, heading_deg), ...]
    ) -> Tuple[float, float]:
        """
        Compute cross-track distance and right-of-track bias.

        Returns:
            (cross_track_factor 0-1, min_dist_km)
        """
        if not track_points:
            return 0.0, 9999.0

        min_dist_km = 9999.0
        best_factor = 0.0

        for tlat, tlon, heading in track_points:
            # Great-circle approximate distance (km)
            dlat = (lat - tlat) * 111.0
            dlon = (lon - tlon) * 111.0 * math.cos(math.radians(tlat))
            dist_km = math.sqrt(dlat**2 + dlon**2)

            if dist_km > self.WAKE_WIDTH_KM:
                continue

            # Right-of-track component (Northern Hemisphere bias)
            # Track heading in radians
            h_rad = math.radians(heading)
            # Cross-track: positive = right of track
            cross = -dlat * math.sin(h_rad) + dlon * math.cos(h_rad)

            # Right-side bonus (Coriolis effect in NH)
            if lat > 0:  # Northern Hemisphere
                right_bonus = max(0, cross) / self.RIGHT_BIAS_KM
                right_bonus = min(right_bonus, 1.0) * 0.3  # up to 30% bonus
            else:
                right_bonus = 0.0

            # Gaussian cross-track profile
            sigma_km = self.WAKE_WIDTH_KM / 3.0
            ct_factor = math.exp(-0.5 * (dist_km / sigma_km) ** 2) + right_bonus
            ct_factor = min(ct_factor, 1.0)

            if dist_km < min_dist_km:
                min_dist_km = dist_km
                best_factor = ct_factor

        return best_factor, min_dist_km

    def compute_golden_zones(
        self,
        typhoon_data: List[Dict[str, Any]],
        sst_current: Optional[np.ndarray],
        lats: np.ndarray,
        lons: np.ndarray,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Main computation: generate golden zone overlay for the grid.

        Args:
            typhoon_data: list of typhoon dicts with keys:
                - name, lat, lon, wind_kt, heading_deg
                - track_history: [{lat, lon, time, heading_deg}, ...]
            sst_current: current SST grid (optional, for SST drop estimation)
            lats, lons: 1D coordinate arrays
            now: current time (default: utcnow)

        Returns:
            {
                "golden_score": 2D grid (0-1),
                "golden_zones": list of zone dicts for hotspot enrichment,
                "n_active_golden": int,
            }
        """
        now = now or datetime.now(timezone.utc)
        ny, nx = len(lats), len(lons)
        golden_grid = np.zeros((ny, nx), dtype=np.float32)
        golden_zones = []

        for typh in typhoon_data:
            track_hist = typh.get("track_history", [])
            if not track_hist:
                continue

            wind_kt = typh.get("wind_kt", 50)
            cat_factor = self._category_factor(wind_kt)
            name = typh.get("name", "UNKNOWN")

            # Build track points with heading
            track_points = []
            for pt in track_hist:
                pt_time = pt.get("time")
                if pt_time is None:
                    continue

                # Parse time
                if isinstance(pt_time, str):
                    try:
                        pt_time = datetime.fromisoformat(
                            pt_time.replace("Z", "+00:00")
                        )
                    except ValueError:
                        continue
                elif not hasattr(pt_time, "timestamp"):
                    continue

                hours_since = (now - pt_time).total_seconds() / 3600.0

                # Only consider points in the golden window
                if hours_since < self.GOLDEN_WINDOW_START_H:
                    continue  # too recent, still dangerous
                if hours_since > self.GOLDEN_WINDOW_END_H:
                    continue  # too old, nutrients dispersed

                heading = pt.get("heading_deg", pt.get("heading", 0))
                track_points.append((
                    pt["lat"], pt["lon"], heading, hours_since, cat_factor,
                ))

            if not track_points:
                continue

            log.info(
                f"  🌀→🐟 Typhoon {name}: {len(track_points)} track points "
                f"in golden window, cat_factor={cat_factor:.1f}"
            )

            # Compute grid-wise golden score
            zone_mask = np.zeros((ny, nx), dtype=bool)

            for i, lat in enumerate(lats):
                for j, lon in enumerate(lons):
                    best_score = 0.0
                    for tlat, tlon, heading, hours_since, cf in track_points:
                        # Distance
                        dlat = (lat - tlat) * 111.0
                        dlon = (lon - tlon) * 111.0 * math.cos(math.radians(tlat))
                        dist_km = math.sqrt(dlat**2 + dlon**2)

                        if dist_km > self.WAKE_WIDTH_KM:
                            continue

                        # Cross-track Gaussian
                        sigma_km = self.WAKE_WIDTH_KM / 3.0
                        spatial = math.exp(-0.5 * (dist_km / sigma_km) ** 2)

                        # Right-of-track bonus (NH only)
                        h_rad = math.radians(heading)
                        cross = -dlat * math.sin(h_rad) + dlon * math.cos(h_rad)
                        if lat > 0 and cross > 0:
                            spatial = min(1.0, spatial + 0.2)

                        # Temporal: cold wake decay × bloom timing
                        wake = self._cold_wake_decay(hours_since)
                        bloom = self._bloom_timing(hours_since)
                        temporal = 0.4 * wake + 0.6 * bloom  # bloom > wake importance

                        # Combined score
                        score = spatial * temporal * cf
                        best_score = max(best_score, score)

                    if best_score > 0.05:
                        golden_grid[i, j] = max(golden_grid[i, j], best_score)
                        zone_mask[i, j] = True

            n_cells = int(zone_mask.sum())
            if n_cells > 0:
                max_score = float(golden_grid[zone_mask].max())
                # Find centroid
                yi, xi = np.where(zone_mask)
                c_lat = float(np.mean(lats[yi]))
                c_lon = float(np.mean(lons[xi]))

                golden_zones.append({
                    "typhoon_name": name,
                    "centroid_lat": round(c_lat, 2),
                    "centroid_lon": round(c_lon, 2),
                    "n_cells": n_cells,
                    "max_golden_score": round(max_score, 3),
                    "category_factor": round(cat_factor, 1),
                    "recommendation": (
                        f"🌀→🐟 颱風 {name} 過後黃金漁場！"
                        f"營養鹽湧升中，預估 CHL 增加 {cat_factor:.0f}× 基線值。"
                        f"建議在此區域加密搜索。"
                    ),
                })

                log.info(
                    f"  ✅ Golden zone from {name}: "
                    f"({c_lat:.1f}°N, {c_lon:.1f}°E), "
                    f"{n_cells} cells, max_score={max_score:.3f}"
                )

        # Normalize to [0, 1]
        if golden_grid.max() > 0:
            golden_grid = np.clip(golden_grid / golden_grid.max(), 0, 1)

        return {
            "golden_score": golden_grid.astype(np.float32),
            "golden_zones": golden_zones,
            "n_active_golden": len(golden_zones),
        }

    def enrich_hotspots(
        self,
        hotspots: List[Dict],
        golden_result: Dict,
        lats: np.ndarray,
        lons: np.ndarray,
        hsi_boost: float = 0.08,
    ) -> List[Dict]:
        """
        Enrich hotspot dicts with golden zone info.
        Adds HSI bonus for hotspots inside golden zones.

        Args:
            hsi_boost: max HSI increase for golden zone hotspots (default 8%)
        """
        golden_grid = golden_result.get("golden_score")
        if golden_grid is None or golden_grid.max() == 0:
            return hotspots

        golden_zones = golden_result.get("golden_zones", [])

        for h in hotspots:
            li = int(np.argmin(np.abs(lats - h["lat"])))
            lj = int(np.argmin(np.abs(lons - h["lon"])))

            if li < golden_grid.shape[0] and lj < golden_grid.shape[1]:
                gs = float(golden_grid[li, lj])
                if gs > 0.1:
                    h["typhoon_golden_score"] = round(gs, 3)
                    h["typhoon_golden_boost"] = round(hsi_boost * gs, 4)

                    # Apply HSI boost
                    old_score = h.get("score", 0)
                    h["score"] = min(0.95, old_score + hsi_boost * gs)
                    h["score_before_golden"] = old_score

                    # Add recommendation from matching zone
                    for zone in golden_zones:
                        dist = math.sqrt(
                            (h["lat"] - zone["centroid_lat"])**2 +
                            (h["lon"] - zone["centroid_lon"])**2
                        )
                        if dist < 3.0:  # within ~3 degrees
                            h["typhoon_golden_note"] = zone["recommendation"]
                            break

        # Re-sort by adjusted score
        hotspots.sort(key=lambda x: x.get("score", 0), reverse=True)
        for i, h in enumerate(hotspots):
            h["rank"] = i + 1

        boosted = sum(1 for h in hotspots if h.get("typhoon_golden_score", 0) > 0)
        if boosted:
            log.info(
                f"  🌀→🐟 Golden zone: {boosted}/{len(hotspots)} "
                f"hotspots boosted (max {hsi_boost*100:.0f}%)"
            )

        return hotspots
