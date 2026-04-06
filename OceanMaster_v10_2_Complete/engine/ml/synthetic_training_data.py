"""
OceanMaster v12 — Phase 9: Synthetic CPUE Training Data Generator  # [v12-phase9]
==================================================================================
Generate scientifically realistic synthetic CPUE training data aligned with
FeatureEngineer.FEATURE_NAMES (44 features).

Based on known relationships from fisheries science:
  - CPUE correlates with SST (species-specific optimal range)
  - CPUE correlates with Chl-a (proxy for food availability)
  - CPUE correlates with Φ metabolic index (Deutsch 2015)
  - CPUE has seasonal patterns (migration + spawning)
  - CPUE has spatial patterns (latitude + proximity to fronts)
  - CPUE has noise (weather, luck, reporting bias)

Usage:
    gen = SyntheticCPUEGenerator(seed=42)
    X, y, feature_names = gen.generate_full_dataset(n_samples=2000, species="yellowfin")
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, List, Optional

log = logging.getLogger("OceanMaster.SyntheticData")

# ── Species-specific parameters (from fisheries literature) ──
SPECIES_PARAMS = {
    "yellowfin": {
        "sst_opt": 28.0, "sst_sigma": 2.0,       # Tropical pelagic
        "chl_opt_log": -0.5, "chl_sigma": 0.4,     # Moderate productivity
        "depth_opt": -150, "depth_sigma": 100,      # Shallow thermocline
        "lat_center": 15, "lat_sigma": 10,
        "spawning_months": [4, 5, 6, 7, 8, 9],
        "migration_phase": 0.0,
        "cpue_scale": 120,                          # kg/day baseline
    },
    "bigeye": {
        "sst_opt": 23.0, "sst_sigma": 3.0,         # Deeper, cooler
        "chl_opt_log": -0.6, "chl_sigma": 0.5,
        "depth_opt": -300, "depth_sigma": 150,      # Deep thermocline specialist
        "lat_center": 20, "lat_sigma": 12,
        "spawning_months": [3, 4, 5, 6, 7],
        "migration_phase": 0.5,
        "cpue_scale": 80,
    },
    "albacore": {
        "sst_opt": 18.5, "sst_sigma": 3.5,         # Temperate
        "chl_opt_log": -0.3, "chl_sigma": 0.5,
        "depth_opt": -200, "depth_sigma": 120,
        "lat_center": 30, "lat_sigma": 8,
        "spawning_months": [3, 4, 5, 6],
        "migration_phase": 1.0,
        "cpue_scale": 60,
    },
    "skipjack": {
        "sst_opt": 29.0, "sst_sigma": 1.5,         # Surface tropical
        "chl_opt_log": -0.4, "chl_sigma": 0.4,
        "depth_opt": -80, "depth_sigma": 50,
        "lat_center": 10, "lat_sigma": 8,
        "spawning_months": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # Year-round
        "migration_phase": 0.3,
        "cpue_scale": 150,
    },
}


class SyntheticCPUEGenerator:
    """
    Generate realistic synthetic CPUE training data matching
    FeatureEngineer.FEATURE_NAMES (44 features) exactly.
    """

    # Must match FeatureEngineer.FEATURE_NAMES in stacking_ensemble.py
    FEATURE_NAMES = [
        # 基礎環境
        "sst", "chl_log", "ssh", "current_speed", "current_dir",
        # 衍生物理
        "sst_gradient", "chl_gradient", "front_strength",
        "ftle", "ftle_ridge",
        "thermocline_depth", "d20_depth", "mld",
        # 距離特徵
        "dist_to_front", "dist_to_eddy",
        "dist_to_seamount", "dist_to_shelf_break",
        # 地形特徵
        "bathy_depth", "bathy_slope",
        # 時間特徵
        "moon_phase",
        "season_sin", "season_cos", "day_of_year_sin", "day_of_year_cos",
        # 時序特徵
        "sst_7d_trend", "chl_30d_anomaly",
        # 交互
        "sst_x_chl", "front_x_ftle", "ssh_x_thermo",
        # 窗口統計
        "sst_local_std", "chl_local_mean", "current_local_mean",
        # AIS / VIIRS
        "ais_fishing_density", "viirs_light_density",
        # v11 新增 10 個科學特徵
        "lunar_cpue_modifier",
        "zooplankton_index",
        "spawning_season",
        "enso_oni",
        "omz_compression",
        "eddy_enrichment",
        "dvm_accessible_depth",
        "salinity_front_strength",
        "productivity_front",
        "habitat_compression_ratio",
    ]

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_ocean_conditions(
        self,
        n_samples: int = 2000,
        lat_range: Tuple[float, float] = (5, 35),
        lon_range: Tuple[float, float] = (120, 175),
        date_range: Optional[Tuple[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Generate realistic synthetic ocean conditions for n_samples points.
        Each row = one observation (lat, lon, time, + 44 features).
        """
        rng = self.rng

        # ── Spatial coordinates ──
        lat = rng.uniform(lat_range[0], lat_range[1], n_samples)
        lon = rng.uniform(lon_range[0], lon_range[1], n_samples)
        doy = rng.integers(1, 366, n_samples)  # day of year
        month = np.clip((doy / 30.5).astype(int) + 1, 1, 12)

        data = {}
        data["_lat"] = lat
        data["_lon"] = lon
        data["_doy"] = doy
        data["_month"] = month

        # ═════════════════════════════════════════
        # 1. 基礎環境特徵
        # ═════════════════════════════════════════

        # SST: latitude gradient + seasonal cycle + noise
        sst_base = 32.0 - 0.5 * (lat - 5)  # ~32°C at 5°N, ~17°C at 35°N
        sst_seasonal = 2.0 * np.sin(2 * np.pi * (doy - 80) / 365)  # peak ~Mar 21
        sst_noise = rng.normal(0, 0.8, n_samples)
        data["sst"] = np.clip(sst_base + sst_seasonal + sst_noise, 10, 34)

        # Chl-a (log10): log-normal, higher near coasts and at higher latitudes
        chl_base = -0.8 + 0.02 * (lat - 5)  # more productive at higher lat
        coast_boost = np.where(
            (lon < 130) | (lon > 170), 0.3, 0.0  # near landmasses
        )
        chl_noise = rng.normal(0, 0.3, n_samples)
        data["chl_log"] = np.clip(chl_base + coast_boost + chl_noise, -2.0, 1.5)

        # SSH: mesoscale variability
        data["ssh"] = rng.normal(0, 0.1, n_samples)

        # Current speed (kn): 0-2
        data["current_speed"] = rng.exponential(0.3, n_samples)
        data["current_speed"] = np.clip(data["current_speed"], 0, 2.0)

        # Current direction (radians)
        data["current_dir"] = rng.uniform(-np.pi, np.pi, n_samples)

        # ═════════════════════════════════════════
        # 2. 衍生物理特徵
        # ═════════════════════════════════════════

        # SST gradient: higher near fronts
        data["sst_gradient"] = rng.exponential(0.02, n_samples)
        data["sst_gradient"] = np.clip(data["sst_gradient"], 0, 0.5)

        # Chl gradient
        data["chl_gradient"] = rng.exponential(0.01, n_samples)
        data["chl_gradient"] = np.clip(data["chl_gradient"], 0, 0.3)

        # Front strength: correlated with SST gradient
        data["front_strength"] = np.clip(
            data["sst_gradient"] * 10 + rng.normal(0, 0.1, n_samples), 0, 1
        )

        # FTLE: Lagrangian coherent structures
        data["ftle"] = rng.exponential(0.05, n_samples)
        data["ftle"] = np.clip(data["ftle"], 0, 0.5)

        # FTLE ridge (binary-ish)
        data["ftle_ridge"] = (data["ftle"] > 0.1).astype(float)

        # Thermocline depth: deeper in tropics
        thermo_base = 100 + 3 * (lat - 5) + rng.normal(0, 20, n_samples)
        data["thermocline_depth"] = np.clip(thermo_base, 30, 400)

        # D20 depth: closely related to thermocline
        data["d20_depth"] = data["thermocline_depth"] + rng.normal(50, 15, n_samples)
        data["d20_depth"] = np.clip(data["d20_depth"], 50, 500)

        # Mixed layer depth
        data["mld"] = 30 + rng.exponential(20, n_samples)
        data["mld"] = np.clip(data["mld"], 10, 200)

        # ═════════════════════════════════════════
        # 3. 距離特徵
        # ═════════════════════════════════════════

        # Distance to nearest front (km) — closer = better fishing
        data["dist_to_front"] = rng.exponential(50, n_samples)
        data["dist_to_front"] = np.clip(data["dist_to_front"], 0, 500)

        # Distance to nearest eddy (km)
        data["dist_to_eddy"] = rng.exponential(80, n_samples)
        data["dist_to_eddy"] = np.clip(data["dist_to_eddy"], 0, 500)

        # Distance to seamount (km)
        data["dist_to_seamount"] = rng.exponential(150, n_samples)
        data["dist_to_seamount"] = np.clip(data["dist_to_seamount"], 0, 500)

        # Distance to shelf break (km)
        data["dist_to_shelf_break"] = rng.exponential(100, n_samples)
        data["dist_to_shelf_break"] = np.clip(data["dist_to_shelf_break"], 0, 500)

        # ═════════════════════════════════════════
        # 4. 地形特徵
        # ═════════════════════════════════════════

        # Bathymetry depth (m, negative = deeper)
        data["bathy_depth"] = -(1000 + rng.exponential(2500, n_samples))
        data["bathy_depth"] = np.clip(data["bathy_depth"], -8000, -50)

        # Bathymetry slope (degrees)
        data["bathy_slope"] = rng.exponential(1.5, n_samples)
        data["bathy_slope"] = np.clip(data["bathy_slope"], 0, 30)

        # ═════════════════════════════════════════
        # 5. 時間特徵
        # ═════════════════════════════════════════

        # Moon phase: 0-1 (0=new, 0.5=full)
        data["moon_phase"] = rng.uniform(0, 1, n_samples)

        # Seasonal encoding
        data["season_sin"] = np.sin(2 * np.pi * doy / 365.25)
        data["season_cos"] = np.cos(2 * np.pi * doy / 365.25)
        data["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
        data["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.25)

        # ═════════════════════════════════════════
        # 6. 時序特徵
        # ═════════════════════════════════════════

        # SST 7-day trend (°C/day)
        data["sst_7d_trend"] = rng.normal(0, 0.05, n_samples)

        # Chl 30-day anomaly
        data["chl_30d_anomaly"] = rng.normal(0, 0.2, n_samples)

        # ═════════════════════════════════════════
        # 7. 交互特徵
        # ═════════════════════════════════════════

        data["sst_x_chl"] = data["sst"] * data["chl_log"]
        data["front_x_ftle"] = data["front_strength"] * data["ftle"]
        data["ssh_x_thermo"] = data["ssh"] * data["thermocline_depth"]

        # ═════════════════════════════════════════
        # 8. 窗口統計
        # ═════════════════════════════════════════

        data["sst_local_std"] = rng.exponential(0.5, n_samples)
        data["sst_local_std"] = np.clip(data["sst_local_std"], 0, 5)

        data["chl_local_mean"] = 10 ** data["chl_log"] + rng.normal(0, 0.05, n_samples)
        data["chl_local_mean"] = np.clip(data["chl_local_mean"], 0.01, 20)

        data["current_local_mean"] = data["current_speed"] + rng.normal(0, 0.05, n_samples)
        data["current_local_mean"] = np.clip(data["current_local_mean"], 0, 3)

        # ═════════════════════════════════════════
        # 9. AIS / VIIRS
        # ═════════════════════════════════════════

        data["ais_fishing_density"] = rng.exponential(0.3, n_samples)
        data["ais_fishing_density"] = np.clip(data["ais_fishing_density"], 0, 5)

        data["viirs_light_density"] = rng.exponential(0.1, n_samples)
        data["viirs_light_density"] = np.clip(data["viirs_light_density"], 0, 3)

        # ═════════════════════════════════════════
        # 10. v11 新增 10 個科學特徵
        # ═════════════════════════════════════════

        # Lunar CPUE modifier (0.7 - 1.3)
        # Dark (new moon) → better fishing → modifier ~1.25
        # Full moon → worse fishing → modifier ~0.85
        moon = data["moon_phase"]
        data["lunar_cpue_modifier"] = 1.0 - 0.3 * np.cos(2 * np.pi * (moon - 0.5))
        data["lunar_cpue_modifier"] += rng.normal(0, 0.03, n_samples)
        data["lunar_cpue_modifier"] = np.clip(data["lunar_cpue_modifier"], 0.5, 1.5)

        # Zooplankton index: correlated with Chl-a
        zoo_base = 0.3 + 0.3 * (data["chl_log"] + 1) / 2
        data["zooplankton_index"] = np.clip(
            zoo_base + rng.normal(0, 0.1, n_samples), 0, 1
        )

        # Spawning season (binary)
        data["spawning_season"] = np.zeros(n_samples)
        # Will be set per-species in generate_cpue_labels

        # ENSO ONI index (-3 to +3)
        # Simulate random ENSO state for the dataset
        oni_state = rng.choice([-1.5, -0.5, 0.0, 0.5, 1.5], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1])
        data["enso_oni"] = oni_state + rng.normal(0, 0.2, n_samples)

        # OMZ compression (0-1, 1=no compression)
        # Higher SST → lower DO → more compression
        omz_base = 1.0 - 0.02 * np.maximum(data["sst"] - 25, 0)
        data["omz_compression"] = np.clip(
            omz_base + rng.normal(0, 0.05, n_samples), 0.3, 1.0
        )

        # Eddy enrichment (0-1)
        # Closer to eddy → higher enrichment
        data["eddy_enrichment"] = np.clip(
            np.exp(-data["dist_to_eddy"] / 100) + rng.normal(0, 0.05, n_samples),
            0, 1
        )

        # DVM accessible depth (m)
        # Related to thermocline and moon phase
        dvm_base = data["thermocline_depth"] * (1 + 0.3 * moon)
        data["dvm_accessible_depth"] = np.clip(
            dvm_base + rng.normal(0, 20, n_samples), 50, 600
        )

        # Salinity front strength (0-1)
        data["salinity_front_strength"] = rng.exponential(0.15, n_samples)
        data["salinity_front_strength"] = np.clip(data["salinity_front_strength"], 0, 1)

        # Productivity front (composite of SST + Chl fronts)
        data["productivity_front"] = np.clip(
            0.5 * data["front_strength"] + 0.5 * data["chl_gradient"] * 10
            + rng.normal(0, 0.05, n_samples),
            0, 1,
        )

        # Habitat compression ratio
        # OMZ / thermocline compression
        data["habitat_compression_ratio"] = np.clip(
            data["omz_compression"] * (data["thermocline_depth"] / 200),
            0.1, 3.0,
        )

        return pd.DataFrame(data)

    def generate_cpue_labels(
        self,
        conditions: pd.DataFrame,
        species: str = "yellowfin",
    ) -> np.ndarray:
        """
        Generate scientifically grounded CPUE labels from ocean conditions.

        CPUE = base_cpue × food_factor × metabolic_factor × spatial_factor
               × lunar_factor × seasonal_factor + noise

        Returns CPUE normalized to [0, 1] range.
        """
        rng = self.rng
        params = SPECIES_PARAMS.get(species, SPECIES_PARAMS["yellowfin"])
        n = len(conditions)

        sst = conditions["sst"].values
        chl_log = conditions["chl_log"].values
        moon = conditions["moon_phase"].values
        doy = conditions["_doy"].values
        month = conditions["_month"].values
        lat = conditions["_lat"].values
        front = conditions["front_strength"].values
        eddy_enrich = conditions["eddy_enrichment"].values
        thermo = conditions["thermocline_depth"].values
        bathy = conditions["bathy_depth"].values
        omz = conditions["omz_compression"].values
        zoo = conditions["zooplankton_index"].values

        # ── 1. SST suitability: Gaussian around species optimum ──
        sst_si = np.exp(
            -((sst - params["sst_opt"]) ** 2) / (2 * params["sst_sigma"] ** 2)
        )

        # ── 2. Food factor: log(1 + Chl) × zooplankton ──
        chl_linear = 10 ** chl_log
        chl_si = np.exp(
            -((chl_log - params["chl_opt_log"]) ** 2) / (2 * params["chl_sigma"] ** 2)
        )
        food_factor = chl_si * (0.5 + 0.5 * zoo)

        # ── 3. Metabolic viability (Deutsch 2015 Φ proxy) ──
        # Simplified: SST → metabolic rate, OMZ → oxygen limitation
        phi_proxy = np.clip(omz * sst_si, 0, 1)

        # ── 4. Spatial factor: front + eddy + bathymetry ──
        front_bonus = 0.2 * front
        eddy_bonus = 0.15 * eddy_enrich
        # Bathymetry suitability
        bathy_si = np.exp(
            -((bathy - params["depth_opt"]) ** 2) / (2 * params["depth_sigma"] ** 2)
        )
        spatial_factor = 1.0 + front_bonus + eddy_bonus + 0.1 * bathy_si

        # ── 5. Lunar factor: dark = better ──
        lunar_factor = 1.0 - 0.3 * np.abs(moon - 0.0)  # 0 = new moon = best
        lunar_factor = np.clip(lunar_factor, 0.6, 1.2)

        # ── 6. Seasonal factor: migration + spawning ──
        phase = params["migration_phase"]
        seasonal = 0.7 + 0.3 * np.sin(2 * np.pi * doy / 365 + phase)

        # Spawning boost
        is_spawning = np.isin(month, params["spawning_months"]).astype(float)
        seasonal *= (1.0 + 0.15 * is_spawning)

        # ── 7. Latitude suitability ──
        lat_si = np.exp(
            -((lat - params["lat_center"]) ** 2) / (2 * params["lat_sigma"] ** 2)
        )

        # ── Combine ──
        cpue_raw = (
            params["cpue_scale"]
            * sst_si
            * food_factor
            * phi_proxy
            * spatial_factor
            * lunar_factor
            * seasonal
            * lat_si
        )

        # ── Add realistic noise: 20-30% CV ──
        cv = rng.uniform(0.20, 0.30)
        noise = rng.lognormal(0, cv, n)
        cpue_raw *= noise

        # ── Normalize to [0, 1] ──
        cpue_max = np.percentile(cpue_raw, 99)  # avoid extreme outlier stretching
        cpue_norm = np.clip(cpue_raw / max(cpue_max, 1e-6), 0, 1)

        # Update spawning_season column in conditions
        conditions["spawning_season"] = is_spawning

        return cpue_norm

    def generate_full_dataset(
        self,
        n_samples: int = 2000,
        species: str = "yellowfin",
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate complete training dataset: (X, y, feature_names).
        X has exactly 44 features matching FeatureEngineer.FEATURE_NAMES.
        """
        log.info(f"Generating {n_samples} synthetic samples for {species}...")

        # Generate conditions
        conditions = self.generate_ocean_conditions(n_samples)

        # Generate CPUE labels
        y = self.generate_cpue_labels(conditions, species)

        # Extract only the 44 feature columns (drop _lat, _lon, _doy, _month)
        feature_names = list(self.FEATURE_NAMES)
        X = conditions[feature_names].values.astype(np.float64)

        # Sanity check
        assert X.shape[1] == 44, f"Expected 44 features, got {X.shape[1]}"
        assert len(y) == n_samples

        log.info(f"  Dataset: X={X.shape}, y mean={np.mean(y):.3f}, std={np.std(y):.3f}")
        return X, y, feature_names

    def save_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        output_dir: str = "data/training",
        species: str = "yellowfin",
    ) -> dict:
        """Save dataset to CSV + NPY files."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # CSV (human-readable)
        df = pd.DataFrame(X, columns=feature_names)
        df["cpue_target"] = y
        csv_path = out / f"synthetic_{species}.csv"
        df.to_csv(csv_path, index=False)

        # NPY (fast loading)
        np.save(out / f"X_{species}.npy", X)
        np.save(out / f"y_{species}.npy", y)

        log.info(f"  Saved: {csv_path} ({csv_path.stat().st_size // 1024} KB)")
        return {
            "csv_path": str(csv_path),
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
        }
