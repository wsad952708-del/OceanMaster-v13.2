"""
OceanMaster v13.2 — Ocean Diagnostics Suite [#4, #5, #7, #8, #9, #10]
=======================================================================
Derived oceanographic diagnostics from existing data:

  #4  Spiciness — water mass boundary detection from T-S
  #5  Buoyancy frequency N² — stratification strength
  #7  Mixed layer heat content OHC — thermal stability
  #8  Multi-species overlap — biodiversity hotspot zones
  #9  Optimal fishing time — best hours based on DVM + moon + tide
  #10 Cross-shelf transport — nutrient transport index

Ref: Flament (2002) Prog. Oceanogr. 54:493-501 [spiciness]
     Gill (1982) Atmosphere-Ocean Dynamics [N²]
     Stramma et al. (2012) [habitat compression/OHC]
"""

import numpy as np
import logging
import math
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

log = logging.getLogger("OceanMaster.Diag")


# ═══════════════════════════════════════════════════
# #4: Spiciness — T-S water mass boundary detection
# ═══════════════════════════════════════════════════

class SpicinessAnalyzer:
    """
    [v16.0 #4] Detect water mass boundaries via spiciness.

    Spiciness = temperature variation along isopycnals (constant density).
    High spiciness gradient = different water masses meeting = frontal zone.

    Simplified: π ≈ T - α·S  (linearized on potential density surface)
    where α adjusts for salinity contribution to density.
    """

    def compute(
        self,
        sst: np.ndarray,
        salinity: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compute spiciness field and gradient.

        Args:
            sst: SST (°C)
            salinity: Surface salinity (PSU)

        Returns:
            {
                "spiciness": 2D field,
                "spiciness_gradient": 2D magnitude,
                "water_mass_front": 2D bool,
                "front_pct": float,
                "advisory": str,
            }
        """
        # Linearized spiciness (Flament 2002, simplified)
        # π ≈ T + β·S where β ≈ 5.6°C/PSU for surface ocean
        # This picks up T-S co-variation = water mass identity
        sal_safe = np.nan_to_num(salinity, nan=35.0)
        sst_safe = np.nan_to_num(sst, nan=25.0)

        spice = sst_safe + 5.6 * (sal_safe - 35.0)
        spice = spice.astype(np.float32)

        # Gradient
        gy, gx = np.gradient(spice)
        grad = np.sqrt(gx**2 + gy**2).astype(np.float32)

        # Front detection: gradient > 2σ
        g_std = float(np.std(grad[np.isfinite(grad)])) if np.any(np.isfinite(grad)) else 1.0
        front = grad > 2.0 * g_std
        front_pct = float(np.mean(front) * 100)

        advisory = (f"水團邊界: {front_pct:.0f}% 的網格偵測到 spiciness 鋒面"
                    if front_pct > 5 else "水團均質 — 無明顯邊界")

        log.info(f"  Spiciness: range=[{spice.min():.1f}, {spice.max():.1f}], "
                 f"fronts={front_pct:.1f}%")

        return {
            "spiciness": spice,
            "spiciness_gradient": grad,
            "water_mass_front": front,
            "front_pct": round(front_pct, 1),
            "advisory": advisory,
        }


# ═══════════════════════════════════════════════════
# #5: Buoyancy Frequency N² — Stratification
# ═══════════════════════════════════════════════════

class StratificationAnalyzer:
    """
    [v16.0 #5] Buoyancy frequency N² from temperature/salinity profiles.

    N² = -(g/ρ₀) × dρ/dz

    High N² = strong stratification → nutrients locked below
    Low N²  = weak stratification → easy mixing → nutrients rise
    """

    G = 9.81       # m/s²
    RHO_0 = 1025.0  # reference density kg/m³

    def compute(
        self,
        temp_3d: np.ndarray,
        depths: np.ndarray,
        salinity: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Compute N² from vertical temperature profile.

        Args:
            temp_3d: 3D temperature (n_depths, ny, nx)
            depths: 1D depth levels (m), e.g. [0, 50, 100, 200, 300, 500]
            salinity: 2D surface salinity (optional, improves density estimate)

        Returns:
            {
                "n2_profile": 3D (n_depths-1, ny, nx) — N² at each layer interface,
                "n2_mean": 2D — depth-averaged N²,
                "pycnocline_depth": 2D — depth of maximum N² (m),
                "weak_strat_mask": 2D bool — areas with weak stratification,
                "advisory": str,
            }
        """
        nd, ny, nx = temp_3d.shape
        if nd < 2:
            return self._empty(ny, nx)

        # Density profile estimate (simplified linear EOS)
        # ρ = ρ₀ × (1 - α_T × (T - T_ref) + β_S × (S - S_ref))
        alpha_t = 2e-4  # thermal expansion (1/°C)
        beta_s = 7.5e-4  # haline contraction (1/PSU)
        t_ref = 10.0
        s_ref = 35.0

        # Build density profile
        rho_3d = np.full_like(temp_3d, self.RHO_0)
        for k in range(nd):
            dt = np.nan_to_num(temp_3d[k], nan=t_ref) - t_ref
            if salinity is not None:
                ds = np.nan_to_num(salinity, nan=s_ref) - s_ref
            else:
                ds = 0.0
            rho_3d[k] = self.RHO_0 * (1 - alpha_t * dt + beta_s * ds)

        # N² at each layer interface
        n2 = np.zeros((nd - 1, ny, nx), dtype=np.float32)
        for k in range(nd - 1):
            dz = max(depths[k + 1] - depths[k], 1.0)
            drho = rho_3d[k + 1] - rho_3d[k]
            n2[k] = (self.G / self.RHO_0) * drho / dz  # positive = stable

        # Depth-averaged N²
        n2_mean = np.mean(n2, axis=0).astype(np.float32)

        # Pycnocline = depth of max N²
        k_max = np.argmax(n2, axis=0)
        pyc_depth = np.zeros((ny, nx), dtype=np.float32)
        for j in range(ny):
            for i in range(nx):
                km = k_max[j, i]
                pyc_depth[j, i] = (depths[km] + depths[min(km + 1, nd - 1)]) / 2

        # Weak stratification (N² < 1e-5 s⁻²)
        weak = n2_mean < 1e-5

        weak_pct = float(np.mean(weak) * 100)
        advisory = (f"弱層化: {weak_pct:.0f}% — 營養鹽容易混合上升"
                    if weak_pct > 20 else "層化正常 — 營養鹽需要動力機制上升")

        log.info(f"  N²: mean={np.nanmean(n2_mean):.2e}, "
                 f"pycnocline={np.nanmean(pyc_depth):.0f}m, "
                 f"weak={weak_pct:.1f}%")

        return {
            "n2_profile": n2,
            "n2_mean": n2_mean,
            "pycnocline_depth": pyc_depth,
            "weak_strat_mask": weak,
            "weak_strat_pct": round(weak_pct, 1),
            "advisory": advisory,
        }

    @staticmethod
    def _empty(ny, nx):
        return {
            "n2_profile": np.zeros((1, ny, nx), dtype=np.float32),
            "n2_mean": np.zeros((ny, nx), dtype=np.float32),
            "pycnocline_depth": np.full((ny, nx), 100.0, dtype=np.float32),
            "weak_strat_mask": np.zeros((ny, nx), dtype=bool),
            "weak_strat_pct": 0.0,
            "advisory": "N² calculation skipped (insufficient depth layers)",
        }


# ═══════════════════════════════════════════════════
# #7: Mixed Layer Heat Content (OHC)
# ═══════════════════════════════════════════════════

def compute_ohc(sst: np.ndarray, mld: np.ndarray) -> Dict[str, Any]:
    """
    [v16.0 #7] Mixed layer ocean heat content.

    OHC = ρ × Cp × SST × MLD  (J/m²)

    High OHC → thermally stable, warm-core eddies favored by yellowfin.
    Low OHC → cold, shallow mixed layer → upwelling signatures.

    Args:
        sst: SST (°C)
        mld: Mixed layer depth (m)

    Returns:
        {"ohc": 2D (GJ/m²), "ohc_anomaly": 2D, "advisory": str}
    """
    rho = 1025.0  # kg/m³
    cp = 3985.0   # J/(kg·°C)

    sst_safe = np.nan_to_num(sst, nan=25.0)
    mld_safe = np.clip(np.nan_to_num(mld, nan=50.0), 5, 500)

    ohc = (rho * cp * sst_safe * mld_safe / 1e9).astype(np.float32)  # GJ/m²

    ohc_mean = float(np.nanmean(ohc))
    ohc_anomaly = (ohc - ohc_mean).astype(np.float32)

    advisory = f"OHC: mean={ohc_mean:.1f} GJ/m² | 高值=暖核穩定，低值=湧升活躍"

    log.info(f"  OHC: mean={ohc_mean:.1f} GJ/m², "
             f"range=[{ohc.min():.1f}, {ohc.max():.1f}]")

    return {
        "ohc": ohc,
        "ohc_anomaly": ohc_anomaly,
        "ohc_mean": ohc_mean,
        "advisory": advisory,
    }


# ═══════════════════════════════════════════════════
# #8: Multi-species Overlap Zones
# ═══════════════════════════════════════════════════

def compute_species_overlap(
    species_probabilities: Dict[str, float],
    threshold: float = 15.0,
) -> Dict[str, Any]:
    """
    [v16.0 #8] Identify multi-species overlap zones.

    Locations where multiple species have high probability indicate
    rich ecosystems with abundant prey base.

    Args:
        species_probabilities: {species: probability%} from CatchCompositionForecaster
        threshold: minimum % to count as "present"

    Returns:
        {"n_species": int, "overlap_index": float, "dominant_pair": tuple, ...}
    """
    present = {sp: prob for sp, prob in species_probabilities.items()
               if prob >= threshold}
    n_species = len(present)

    # Overlap index: Shannon diversity-like
    total = sum(present.values())
    if total > 0:
        probs_norm = [p / total for p in present.values()]
        diversity = -sum(p * np.log(p + 1e-10) for p in probs_norm)
        max_diversity = np.log(max(len(probs_norm), 2))
        overlap_index = diversity / max(max_diversity, 0.01)
    else:
        overlap_index = 0.0

    # Dominant pair
    sorted_sp = sorted(present.items(), key=lambda x: x[1], reverse=True)
    dominant_pair = (sorted_sp[0][0], sorted_sp[1][0]) if len(sorted_sp) >= 2 else None

    if n_species >= 4:
        advisory = f"生態豐富: {n_species} 物種共存 — 高生物多樣性漁場"
    elif n_species >= 2:
        advisory = f"中等多樣性: {n_species} 物種"
    else:
        advisory = "單一物種主導"

    return {
        "n_species_present": n_species,
        "overlap_index": round(float(overlap_index), 3),
        "present_species": list(present.keys()),
        "dominant_pair": dominant_pair,
        "advisory": advisory,
    }


# ═══════════════════════════════════════════════════
# #9: Optimal Fishing Time Window
# ═══════════════════════════════════════════════════

class FishingTimeOptimizer:
    """
    [v16.0 #9] Predict optimal fishing hours within a day.

    Combines:
    - DVM (diel vertical migration): dawn/dusk = best
    - Moon phase: new moon > full moon for surface species
    - Species behavior: bigeye = nighttime deep, skipjack = daytime surface
    """

    SPECIES_BEST_HOURS = {
        "skipjack":  {"dawn": (5, 8), "dusk": (16, 19), "weight": 0.7},
        "yellowfin": {"dawn": (5, 7), "dusk": (17, 19), "weight": 0.8},
        "bigeye":    {"dawn": (4, 6), "dusk": (18, 21), "weight": 0.9},
        "albacore":  {"dawn": (5, 8), "dusk": (16, 19), "weight": 0.6},
        "swordfish": {"dawn": (3, 5), "dusk": (19, 22), "weight": 0.95},
    }

    def compute(
        self,
        species: str,
        lunar_illumination: float = 0.5,
        lat: float = 10.0,
        utc_offset: int = 8,
    ) -> Dict[str, Any]:
        """
        Predict optimal fishing time windows.

        Returns:
            {
                "best_hours_local": list of (start, end) tuples,
                "hourly_score": list of 24 floats (0-1),
                "peak_hour_local": int,
                "advisory": str,
            }
        """
        sp_info = self.SPECIES_BEST_HOURS.get(species, self.SPECIES_BEST_HOURS["yellowfin"])
        crepuscular_weight = sp_info["weight"]

        # Hourly fishing score (0-1) — UTC
        scores = np.zeros(24, dtype=np.float32)

        # Dawn peak
        dawn_start, dawn_end = sp_info["dawn"]
        for h in range(dawn_start, dawn_end + 1):
            h_mod = h % 24
            center = (dawn_start + dawn_end) / 2
            scores[h_mod] = crepuscular_weight * np.exp(-0.5 * ((h - center) / 1.5) ** 2)

        # Dusk peak
        dusk_start, dusk_end = sp_info["dusk"]
        for h in range(dusk_start, min(dusk_end + 1, 24)):
            center = (dusk_start + dusk_end) / 2
            scores[h % 24] = crepuscular_weight * np.exp(-0.5 * ((h - center) / 1.5) ** 2)

        # Moon correction: new moon boosts night scores, full moon suppresses
        moon_factor = 1.0 - 0.4 * lunar_illumination  # 0.6-1.0
        for h in range(20, 24):
            scores[h] *= moon_factor
        for h in range(0, 5):
            scores[h] *= moon_factor

        # Baseline minimum (always some chance)
        scores = np.clip(scores + 0.1, 0, 1)

        # Convert to local time
        local_scores = np.roll(scores, utc_offset)
        peak_hour = int(np.argmax(local_scores))

        # Best windows (score > 0.5)
        best_hours = []
        in_window = False
        win_start = 0
        for h in range(24):
            if local_scores[h] > 0.4 and not in_window:
                win_start = h
                in_window = True
            elif local_scores[h] <= 0.4 and in_window:
                best_hours.append((win_start, h - 1))
                in_window = False
        if in_window:
            best_hours.append((win_start, 23))

        windows_str = ", ".join(f"{s:02d}:00-{e:02d}:59" for s, e in best_hours)
        advisory = f"最佳時段 (UTC+{utc_offset}): {windows_str} | 高峰: {peak_hour:02d}:00"

        return {
            "best_hours_local": best_hours,
            "hourly_score": local_scores.tolist(),
            "peak_hour_local": peak_hour,
            "species": species,
            "advisory": advisory,
        }


# ═══════════════════════════════════════════════════
# #10: Cross-shelf Transport Index
# ═══════════════════════════════════════════════════

def compute_cross_shelf_transport(
    u: np.ndarray,
    v: np.ndarray,
    bathy: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> Dict[str, Any]:
    """
    [v16.0 #10] Cross-shelf transport — nutrient flux from deep to shelf.

    Computes current component perpendicular to bathymetric contours.
    Positive = onshore (deep → shallow) = nutrient import.
    Negative = offshore = nutrient export.

    Args:
        u, v: 2D current velocity (m/s)
        bathy: 2D bathymetry (m, negative = depth below sea level)
        lats, lons: 1D coordinate arrays

    Returns:
        {"cross_shelf": 2D (m/s, positive=onshore), ...}
    """
    ny, nx = bathy.shape
    depth = np.abs(bathy)

    # Bathymetric gradient (points toward shallow)
    gy = np.gradient(depth, axis=0)
    gx = np.gradient(depth, axis=1)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_mag = np.clip(grad_mag, 1e-6, None)

    # Unit normal to depth contours (pointing toward shallow)
    nx_hat = gx / grad_mag
    ny_hat = gy / grad_mag

    # Cross-shelf component = current · depth_gradient_direction
    # Depth gradient points from deep -> shallow (toward shallower isobath)
    # Positive cross = flowing toward shallow = onshore = nutrient import
    cross = -(u * nx_hat + v * ny_hat).astype(np.float32)

    onshore_pct = float(np.mean(cross > 0.02) * 100)

    if onshore_pct > 30:
        advisory = f"向岸輸送活躍: {onshore_pct:.0f}% — 深層營養鹽向淺水輸送"
    elif onshore_pct > 10:
        advisory = f"部分向岸輸送: {onshore_pct:.0f}%"
    else:
        advisory = "向岸輸送弱 — 營養鹽來自其他機制"

    log.info(f"  Cross-shelf: onshore={onshore_pct:.1f}%, "
             f"mean={np.nanmean(cross):.4f} m/s")

    return {
        "cross_shelf": cross,
        "onshore_pct": round(onshore_pct, 1),
        "bathy_gradient_mag": grad_mag.astype(np.float32),
        "advisory": advisory,
    }
