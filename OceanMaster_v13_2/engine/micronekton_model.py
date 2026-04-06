"""
OceanMaster v13.2 — 3D Micronekton Habitat Model
=================================================
P1 Gap: Simulate 6 functional groups of micronekton (2-20cm organisms).
This is the core technology behind CLS CATSAT (€3,000/year subscription).

Science (Lehodey et al. 2010, 2015 — SEAPODYM):
  Micronekton are the direct prey of tuna and billfish.
  6 functional groups based on DVM (diel vertical migration) behavior:
    1. Epipelagic (EP): non-migrant, stays 0-200m (photic zone)
    2. Upper meso migrant (UMM): day 200-500m, night 0-200m
    3. Lower meso migrant (LMM): day 500-1000m, night 0-200m
    4. Highly migratory (HM): day 500-1000m, night 0-50m
    5. Upper meso non-migrant (UMNM): stays 200-500m
    6. Bathypelagic non-migrant (BNM): stays 500-1000m

  Biomass productivity equation (simplified):
    dB/dt = f(NPP, T, Light) × recruitment − mortality × B

  We compute relative biomass density per functional group per grid cell.

Usage:
    from engine.micronekton_model import MicronektonModel
    mn = MicronektonModel()
    result = mn.compute(sst, npp, mld, z20, chl, depth, moon_illum, lats, lons)
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("OceanMaster.Micronekton")


# ─── 6 Functional Groups ───
FUNCTIONAL_GROUPS = {
    "epipelagic": {
        "name": "表層浮游性", "abbrev": "EP",
        "day_depth": (0, 200), "night_depth": (0, 200),
        "migrant": False,
        "temp_opt": (18, 28), "temp_tol": 6,
        "npp_weight": 0.35,  # directly dependent on surface NPP
        "light_pref": 0.8,   # prefers lit zones
    },
    "upper_meso_migrant": {
        "name": "中層遷移型", "abbrev": "UMM",
        "day_depth": (200, 500), "night_depth": (0, 200),
        "migrant": True,
        "temp_opt": (12, 22), "temp_tol": 8,
        "npp_weight": 0.25,
        "light_pref": -0.3,  # avoids strong light (day deep)
    },
    "lower_meso_migrant": {
        "name": "深層遷移型", "abbrev": "LMM",
        "day_depth": (500, 1000), "night_depth": (0, 200),
        "migrant": True,
        "temp_opt": (6, 16), "temp_tol": 10,
        "npp_weight": 0.15,
        "light_pref": -0.6,
    },
    "highly_migratory": {
        "name": "高度遷移型", "abbrev": "HM",
        "day_depth": (500, 1000), "night_depth": (0, 50),
        "migrant": True,
        "temp_opt": (8, 20), "temp_tol": 10,
        "npp_weight": 0.20,
        "light_pref": -0.8,
    },
    "upper_meso_nonmigrant": {
        "name": "中層非遷移型", "abbrev": "UMNM",
        "day_depth": (200, 500), "night_depth": (200, 500),
        "migrant": False,
        "temp_opt": (8, 18), "temp_tol": 8,
        "npp_weight": 0.10,
        "light_pref": -0.2,
    },
    "bathypelagic_nonmigrant": {
        "name": "深海非遷移型", "abbrev": "BNM",
        "day_depth": (500, 1000), "night_depth": (500, 1000),
        "migrant": False,
        "temp_opt": (4, 12), "temp_tol": 6,
        "npp_weight": 0.05,
        "light_pref": -1.0,
    },
}


class MicronektonModel:
    """
    3D micronekton habitat model inspired by SEAPODYM (Lehodey et al. 2010).

    Computes relative biomass density for 6 functional groups based on:
    - NPP (net primary production) → drives food web base
    - SST → thermal suitability per group
    - MLD → mixed layer depth affects nutrient availability
    - Z20 → thermocline depth affects vertical structure
    - Bathymetry → constrains depth groups
    - Lunar illumination → affects DVM amplitude

    Output: total prey density index (0-1) suitable for tuna foraging.
    """

    def __init__(self):
        self.groups = FUNCTIONAL_GROUPS

    def _thermal_preference(
        self, sst: np.ndarray, opt_range: Tuple[float, float], tolerance: float
    ) -> np.ndarray:
        """
        Gaussian thermal preference function.
        Returns 0-1 suitability based on temperature.
        """
        t_opt = (opt_range[0] + opt_range[1]) / 2.0
        sigma = tolerance / 2.0
        return np.exp(-0.5 * ((sst - t_opt) / max(sigma, 1.0)) ** 2)

    def _npp_productivity(self, npp: np.ndarray, weight: float) -> np.ndarray:
        """
        Convert NPP to productivity index.
        Saturating function: high NPP → diminishing returns.
        Ref: Eppley-VGPM model relationship.
        """
        # Half-saturation at ~2000 mgC/m²/day
        k_half = 2000.0
        return weight * npp / (npp + k_half)

    def _depth_availability(
        self,
        depth: np.ndarray,
        group_depth: Tuple[int, int],
    ) -> np.ndarray:
        """
        Check if seafloor depth allows this functional group to exist.
        Group can't exist deeper than the seafloor.
        """
        min_depth_needed = group_depth[0]
        return np.where(depth >= min_depth_needed, 1.0, 0.0)

    def _dvm_lunar_factor(
        self, moon_illum: float, is_migrant: bool, light_pref: float
    ) -> float:
        """
        Lunar illumination effect on DVM behavior.
        - New moon (illum ≈ 0): DVM very active → migrant groups reach surface
        - Full moon (illum ≈ 1): DVM suppressed → migrants stay deeper
        Ref: Benoit-Bird et al. 2009; Drazen et al. 2011
        """
        if not is_migrant:
            return 1.0  # non-migrants unaffected

        # New moon bonus for migrants: they reach surface better
        # Full moon penalty: light suppresses surface migration
        moon_factor = 1.0 - 0.4 * moon_illum  # 0.6-1.0 range
        return max(0.3, moon_factor)

    def _mld_factor(self, mld: np.ndarray, group_depth: Tuple[int, int]) -> np.ndarray:
        """
        Mixed layer depth effect.
        Deeper MLD → more nutrients mixed → benefits deeper groups.
        Shallow MLD → benefits surface groups.
        """
        group_mid = (group_depth[0] + group_depth[1]) / 2.0
        if group_mid < 200:
            # Surface group: benefits from shallow MLD (nutrients concentrated)
            return np.clip(1.0 - (mld - 50) / 200, 0.3, 1.0)
        else:
            # Deep group: benefits from deep MLD
            return np.clip(mld / 200, 0.3, 1.0)

    def compute(
        self,
        sst: np.ndarray,
        npp: np.ndarray,
        mld: np.ndarray,
        z20: np.ndarray,
        chl: np.ndarray,
        depth: np.ndarray,
        moon_illum: float = 0.5,
        lats: Optional[np.ndarray] = None,
        lons: Optional[np.ndarray] = None,
        is_night: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute micronekton biomass density for all 6 functional groups.

        Args:
            sst: 2D SST grid (°C)
            npp: 2D NPP grid (mgC/m²/day)
            mld: 2D MLD grid (m)
            z20: 2D Z20 grid (m, thermocline depth)
            chl: 2D CHL grid (mg/m³)
            depth: 2D bathymetry grid (m, positive down)
            moon_illum: 0-1 lunar illumination
            is_night: whether current time is nighttime

        Returns:
            {
                "total_prey_density": 2D grid (0-1),
                "surface_prey": 2D grid — prey available at 0-200m,
                "deep_prey": 2D grid — prey at 200-1000m,
                "groups": {name: 2D density, ...},
                "tuna_foraging_index": 2D grid (0-1) — combined prey availability for tuna,
                "summary": dict
            }
        """
        shape = sst.shape
        group_densities = {}
        total_density = np.zeros(shape, dtype=np.float32)
        surface_prey = np.zeros(shape, dtype=np.float32)
        deep_prey = np.zeros(shape, dtype=np.float32)

        for gname, gparams in self.groups.items():
            # 1. Thermal suitability
            thermal = self._thermal_preference(
                sst, gparams["temp_opt"], gparams["temp_tol"]
            )

            # 2. NPP-driven productivity
            prod = self._npp_productivity(npp, gparams["npp_weight"])

            # 3. Depth availability
            depth_avail = self._depth_availability(depth, gparams["day_depth"])

            # 4. Lunar DVM factor
            lunar = self._dvm_lunar_factor(
                moon_illum, gparams["migrant"], gparams["light_pref"]
            )

            # 5. MLD factor
            mld_f = self._mld_factor(mld, gparams["day_depth"])

            # 6. CHL boost (phytoplankton → zooplankton → micronekton cascade)
            chl_boost = np.clip(chl / 0.5, 0.5, 2.0)  # normalize around 0.5 mg/m³

            # Combined biomass density
            density = thermal * prod * depth_avail * lunar * mld_f * chl_boost

            # Clip to [0, 1]
            density = np.clip(density, 0, 1).astype(np.float32)
            group_densities[gname] = density
            total_density += density

            # Classify into surface vs deep prey
            if is_night:
                target_depth = gparams["night_depth"]
            else:
                target_depth = gparams["day_depth"]

            if target_depth[1] <= 200:
                surface_prey += density
            elif target_depth[0] >= 200:
                deep_prey += density
            else:
                # Split
                surface_prey += density * 0.5
                deep_prey += density * 0.5

        # Normalize total to 0-1
        max_total = total_density.max()
        if max_total > 0:
            total_density = total_density / max_total

        max_surface = surface_prey.max()
        if max_surface > 0:
            surface_prey = surface_prey / max_surface

        max_deep = deep_prey.max()
        if max_deep > 0:
            deep_prey = deep_prey / max_deep

        # Tuna foraging index: weighted combination
        # Tuna prefer areas where prey migrates to surface (migrants at night)
        if is_night:
            # Night: migrant groups at surface → high foraging opportunity
            tuna_fi = 0.7 * surface_prey + 0.3 * total_density
        else:
            # Day: tuna must dive for prey → total density matters more
            tuna_fi = 0.4 * surface_prey + 0.6 * total_density

        tuna_fi = np.clip(tuna_fi, 0, 1).astype(np.float32)

        # Summary stats
        group_means = {
            gparams["abbrev"]: float(np.mean(group_densities[gname]))
            for gname, gparams in self.groups.items()
        }
        dominant_group = max(group_means, key=group_means.get)

        summary = {
            "total_mean_density": round(float(np.mean(total_density)), 4),
            "surface_prey_mean": round(float(np.mean(surface_prey)), 4),
            "deep_prey_mean": round(float(np.mean(deep_prey)), 4),
            "tuna_foraging_mean": round(float(np.mean(tuna_fi)), 4),
            "dominant_group": dominant_group,
            "group_means": {k: round(v, 4) for k, v in group_means.items()},
            "moon_illumination": round(moon_illum, 2),
            "is_night": is_night,
        }

        log.info(
            f"  🦐 Micronekton: total={summary['total_mean_density']:.3f}, "
            f"surface={summary['surface_prey_mean']:.3f}, "
            f"deep={summary['deep_prey_mean']:.3f}, "
            f"dominant={dominant_group}, "
            f"tuna_FI={summary['tuna_foraging_mean']:.3f}"
        )

        return {
            "total_prey_density": total_density,
            "surface_prey": surface_prey,
            "deep_prey": deep_prey,
            "tuna_foraging_index": tuna_fi,
            "groups": group_densities,
            "summary": summary,
        }

    def enrich_hotspot(self, hotspot: Dict, result: Dict, lats, lons) -> Dict:
        """Add micronekton data to a single hotspot dict."""
        lat, lon = hotspot.get("lat", 0), hotspot.get("lon", 0)
        li = int(np.argmin(np.abs(lats - lat)))
        lj = int(np.argmin(np.abs(lons - lon)))

        tfi = result["tuna_foraging_index"]
        if li < tfi.shape[0] and lj < tfi.shape[1]:
            hotspot["micronekton_density"] = round(
                float(result["total_prey_density"][li, lj]), 3
            )
            hotspot["surface_prey_density"] = round(
                float(result["surface_prey"][li, lj]), 3
            )
            hotspot["deep_prey_density"] = round(
                float(result["deep_prey"][li, lj]), 3
            )
            hotspot["tuna_foraging_index"] = round(
                float(tfi[li, lj]), 3
            )

            # Per-group breakdown
            groups_at_point = {}
            for gname, gparams in self.groups.items():
                gd = result["groups"].get(gname)
                if gd is not None and li < gd.shape[0] and lj < gd.shape[1]:
                    groups_at_point[gparams["abbrev"]] = round(float(gd[li, lj]), 3)
            hotspot["micronekton_groups"] = groups_at_point

        return hotspot
