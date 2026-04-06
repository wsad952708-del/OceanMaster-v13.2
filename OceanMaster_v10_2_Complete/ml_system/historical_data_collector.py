"""
OceanMaster — Historical Data Collector & Synthetic CPUE Generator
===================================================================
Generates scientifically-grounded synthetic training data when real FAO
CPUE data is unavailable.

The synthetic data models the relationship:
    CPUE = f(SST, Chl-a, SSH, DO, Phi, season, lat, lon) + noise

Based on:
  - Deutsch 2015 (Science): Metabolic Index Φ → CPUE correlation
  - WCPFC public CPUE ranges: yellowfin 20-150 kg/day, bigeye 10-80 kg/day
  - Seasonal patterns from peer-reviewed tuna ecology literature
  - Spatial distributions from WCPFC Statistical Areas

Author: OceanMaster ML System
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging

log = logging.getLogger("OceanMaster.DataCollector")

# ═══════════════════════════════════════════════════
#  Physical Constants (from commercial_core_v2.py)
# ═══════════════════════════════════════════════════
KB = 8.617e-5       # Boltzmann constant (eV/K)
TREF_K = 288.15     # Reference temperature 15°C (K)

METABOLIC_TRAITS = {
    "yellowfin": {
        "Eo": 0.40, "Pcrit_kPa": 4.8, "Topt_C": 28.0,
        "Topt_sigma": 2.0, "phi_crit": 2.5,
    },
    "bigeye": {
        "Eo": 0.35, "Pcrit_kPa": 3.5, "Topt_C": 18.0,
        "Topt_sigma": 3.0, "phi_crit": 2.0,
    },
    "skipjack": {
        "Eo": 0.45, "Pcrit_kPa": 5.5, "Topt_C": 26.0,
        "Topt_sigma": 3.5, "phi_crit": 3.0,
    },
    "albacore": {
        "Eo": 0.38, "Pcrit_kPa": 4.0, "Topt_C": 20.0,
        "Topt_sigma": 3.0, "phi_crit": 2.0,
    },
}

# CPUE ranges from WCPFC public statistics (kg/day)
# Enhanced: include lognormal parameters for realistic distribution
CPUE_RANGES = {
    "yellowfin": {"min": 2, "max": 250, "median": 42, "log_mu": 3.5, "log_sigma": 0.8},
    "bigeye":    {"min": 1, "max": 150, "median": 22, "log_mu": 2.9, "log_sigma": 0.9},
    "skipjack":  {"min": 5, "max": 400, "median": 55, "log_mu": 3.8, "log_sigma": 0.7},
    "albacore":  {"min": 2, "max": 180, "median": 28, "log_mu": 3.1, "log_sigma": 0.85},
}

# Known fishing ground clusters (lat, lon, radius_deg, intensity)
# Based on WCPFC statistical areas and historical catch patterns
FISHING_GROUNDS = {
    "yellowfin": [
        (13, 145, 8, 1.5),   # Western Central Pacific hotspot
        (25, 140, 5, 1.3),   # Near Kuroshio
        (5, 155, 6, 1.2),    # Equatorial convergence
        (20, 170, 4, 1.1),   # Eastern Pacific boundary
    ],
    "bigeye": [
        (10, 150, 7, 1.5),   # Deep thermocline zone
        (25, 135, 5, 1.4),   # Kuroshio Extension
        (0, 160, 6, 1.3),    # Equatorial
    ],
    "skipjack": [
        (5, 155, 10, 1.5),   # Western equatorial
        (0, 165, 8, 1.4),    # Central equatorial
        (15, 140, 5, 1.2),   # Western Pacific
    ],
    "albacore": [
        (30, 160, 6, 1.5),   # North Pacific migration
        (25, 140, 5, 1.3),   # Kuroshio region
        (32, 170, 4, 1.2),   # Transition zone
    ],
}


class OceanEnvironmentSimulator:
    """
    Simulates realistic ocean environment variables for the Western
    Central Pacific (Area 71) based on climatological patterns.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def simulate_sst(self, lat: float, lon: float, month: int) -> float:
        """SST model: latitude + season + El Niño-like variability"""
        # Base: tropical SST decreases with latitude
        base = 30.0 - 0.35 * abs(lat - 5.0)
        # Seasonal cycle (stronger at higher latitudes)
        seasonal_amp = 0.5 + 0.15 * abs(lat - 15)
        seasonal = seasonal_amp * np.cos(2 * np.pi * (month - 2) / 12)
        # Longitude effect (warmer in western warm pool)
        lon_effect = 0.5 * np.exp(-((lon - 145) / 30) ** 2)
        # Random variability (~1°C)
        noise = self.rng.normal(0, 0.8)
        return np.clip(base + seasonal + lon_effect + noise, 15.0, 33.0)

    def simulate_chlorophyll(self, lat: float, lon: float, month: int) -> float:
        """Chl-a model: higher near coasts/upwelling, seasonal bloom"""
        # Base: oligotrophic open ocean
        base = 0.08
        # Latitude effect (more productive at higher lats)
        lat_effect = 0.02 * abs(lat - 10)
        # Coastal proximity (crude)
        if lon < 130 or lon > 170:
            coast = 0.15
        else:
            coast = 0.0
        # Spring bloom
        bloom = 0.05 * np.exp(-((month - 4) / 2.5) ** 2)
        noise = self.rng.lognormal(-3.0, 0.5)
        return np.clip(base + lat_effect + coast + bloom + noise, 0.01, 5.0)

    def simulate_ssh(self, lat: float, lon: float, month: int) -> float:
        """SSH anomaly (meters): mesoscale eddies + large-scale gradient"""
        base = 0.05 * np.sin(2 * np.pi * lat / 40)
        eddy = self.rng.normal(0, 0.08)
        seasonal = 0.03 * np.cos(2 * np.pi * (month - 1) / 12)
        return base + eddy + seasonal

    def simulate_do(self, lat: float, lon: float, month: int, sst: float) -> float:
        """Dissolved Oxygen (mL/L): decreases with SST, varies with depth/region"""
        # Warmer water holds less oxygen
        do_base = 8.5 - 0.15 * sst
        # OMZ effect at certain latitudes
        if 5 < lat < 15 and 130 < lon < 160:
            omz_effect = -0.5
        else:
            omz_effect = 0.0
        noise = self.rng.normal(0, 0.3)
        return np.clip(do_base + omz_effect + noise, 1.0, 8.0)

    def simulate_current_speed(self, lat: float, lon: float) -> float:
        """Surface current speed (m/s)"""
        base = 0.2 + 0.1 * np.sin(2 * np.pi * lat / 30)
        noise = self.rng.exponential(0.1)
        return np.clip(base + noise, 0.01, 1.5)

    def simulate_front_strength(self, lat: float, lon: float) -> float:
        """SST front strength (°C/km): higher near convergence zones"""
        # Fronts more common near subtropical convergence
        if 20 < lat < 30:
            base = 0.03
        elif 10 < lat < 20:
            base = 0.02
        else:
            base = 0.01
        noise = self.rng.exponential(0.01)
        return np.clip(base + noise, 0.0, 0.2)

    def simulate_eddy_strength(self, lat: float, lon: float) -> float:
        """Eddy kinetic energy proxy (cm²/s²)"""
        base = 100 + 50 * np.sin(2 * np.pi * (lat - 10) / 25)
        noise = self.rng.exponential(30)
        return np.clip(base + noise, 10, 500)


def compute_metabolic_index(
    sst_c: float, do_ml_l: float, species: str
) -> float:
    """
    Compute Deutsch 2015 Metabolic Index Φ.

    Φ = pO₂ / [Pcrit × exp(Eo/kB × (1/T - 1/Tref))]
    """
    traits = METABOLIC_TRAITS.get(species, METABOLIC_TRAITS["yellowfin"])
    T_K = sst_c + 273.15

    # Convert DO (mL/L) to pO2 (kPa) using Weiss 1970 / Garcia-Gordon
    # Simplified: at surface, DO_sat ≈ 8.0 mL/L → ~21 kPa
    po2 = do_ml_l * (21.0 / 8.0)

    metabolic_demand = traits["Pcrit_kPa"] * np.exp(
        traits["Eo"] / KB * (1.0 / T_K - 1.0 / TREF_K)
    )
    phi = po2 / max(metabolic_demand, 0.01)
    return np.clip(phi, 0, 20)


def compute_seapodym_hsi(
    sst_c: float, chl: float, species: str
) -> float:
    """
    Simplified SEAPODYM Habitat Suitability Index.

    thermal_score = exp(-0.5 * ((T - T_opt) / σ)²)
    feeding_score = sigmoid(log10(chl))
    HSI = geometric_mean(thermal, feeding)
    """
    traits = METABOLIC_TRAITS.get(species, METABOLIC_TRAITS["yellowfin"])
    thermal = np.exp(-0.5 * ((sst_c - traits["Topt_C"]) / traits["Topt_sigma"]) ** 2)
    chl_log = np.log10(max(chl, 0.001))
    feeding = 1.0 / (1.0 + np.exp(-3.0 * (chl_log + 0.5)))
    return np.sqrt(thermal * feeding)


class SyntheticCPUEGenerator:
    """
    Generates scientifically-grounded synthetic CPUE training data.

    The CPUE model:
      CPUE_base = f(Φ, HSI, front_strength, season)
      CPUE = CPUE_base × effort_correction × noise

    This produces data with known ground-truth relationships that
    the ML model should be able to recover.
    """

    def __init__(
        self,
        species: str = "yellowfin",
        seed: int = 42,
        years: Tuple[int, int] = (2015, 2024),
        area: Dict = None,
    ):
        self.species = species
        self.rng = np.random.default_rng(seed)
        self.env = OceanEnvironmentSimulator(seed)
        self.years = years
        self.area = area or {
            "lat_min": 0, "lat_max": 35,
            "lon_min": 120, "lon_max": 175,
        }
        self.cpue_range = CPUE_RANGES.get(species, CPUE_RANGES["yellowfin"])
        self.traits = METABOLIC_TRAITS.get(species, METABOLIC_TRAITS["yellowfin"])

    def generate(self, n_samples: int = 5000) -> pd.DataFrame:
        """
        Generate n_samples of synthetic CPUE records.

        Enhanced v10.3:
          - Lognormal CPUE distribution (matches real FAO/WCPFC data)
          - Spatial autocorrelation via fishing ground clusters
          - Weighted sampling: 60% near known grounds, 40% random
          - Stronger SST-CPUE Gaussian correlation
          - CHL-CPUE positive correlation
        """
        log.info(f"Generating {n_samples} synthetic CPUE records for {self.species}...")

        fishing_grounds = FISHING_GROUNDS.get(self.species, FISHING_GROUNDS["yellowfin"])

        records = []
        n_clustered = int(n_samples * 0.6)  # 60% near fishing grounds
        n_random = n_samples - n_clustered

        for i in range(n_samples):
            year = self.rng.integers(self.years[0], self.years[1] + 1)
            month = self.rng.integers(1, 13)
            day = self.rng.integers(1, 29)
            date = datetime(year, month, day)

            # Spatial sampling: cluster around known grounds or random
            if i < n_clustered:
                # Pick a random fishing ground and sample near it
                gnd = fishing_grounds[self.rng.integers(0, len(fishing_grounds))]
                lat = gnd[0] + self.rng.normal(0, gnd[2] * 0.5)
                lon = gnd[1] + self.rng.normal(0, gnd[2] * 0.5)
            else:
                lat = self.rng.uniform(self.area["lat_min"], self.area["lat_max"])
                lon = self.rng.uniform(self.area["lon_min"], self.area["lon_max"])

            # Clamp to area
            lat = np.clip(lat, self.area["lat_min"], self.area["lat_max"])
            lon = np.clip(lon, self.area["lon_min"], self.area["lon_max"])

            # Simulate environment
            sst = self.env.simulate_sst(lat, lon, month)
            chl = self.env.simulate_chlorophyll(lat, lon, month)
            ssh = self.env.simulate_ssh(lat, lon, month)
            do = self.env.simulate_do(lat, lon, month, sst)
            current_speed = self.env.simulate_current_speed(lat, lon)
            front_strength = self.env.simulate_front_strength(lat, lon)
            eddy_strength = self.env.simulate_eddy_strength(lat, lon)

            phi = compute_metabolic_index(sst, do, self.species)
            hsi = compute_seapodym_hsi(sst, chl, self.species)

            cpue = self._compute_cpue(
                phi, hsi, sst, chl, front_strength, eddy_strength,
                month, lat, lon
            )

            records.append({
                "date": date,
                "year": year,
                "month": month,
                "day_of_year": date.timetuple().tm_yday,
                "lat": round(lat, 2),
                "lon": round(lon, 2),
                "sst": round(sst, 2),
                "chl": round(chl, 4),
                "ssh": round(ssh, 4),
                "do": round(do, 2),
                "current_speed": round(current_speed, 3),
                "front_strength": round(front_strength, 4),
                "eddy_strength": round(eddy_strength, 1),
                "phi": round(phi, 3),
                "hsi": round(hsi, 4),
                "cpue_kg_per_day": round(cpue, 2),
                "species": self.species,
            })

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Data quality stats
        log.info(f"  Generated {len(df)} records")
        log.info(f"  CPUE range: {df['cpue_kg_per_day'].min():.1f} - "
                 f"{df['cpue_kg_per_day'].max():.1f} kg/day")
        log.info(f"  CPUE mean±std: {df['cpue_kg_per_day'].mean():.1f} "
                 f"± {df['cpue_kg_per_day'].std():.1f}")
        log.info(f"  CPUE median: {df['cpue_kg_per_day'].median():.1f} kg/day")
        log.info(f"  SST-CPUE corr: {df['sst'].corr(df['cpue_kg_per_day']):.3f}")
        log.info(f"  CHL-CPUE corr: {df['chl'].corr(df['cpue_kg_per_day']):.3f}")
        log.info(f"  Phi-CPUE corr: {df['phi'].corr(df['cpue_kg_per_day']):.3f}")
        return df

    def _compute_cpue(
        self, phi, hsi, sst, chl, front_strength, eddy_strength,
        month, lat, lon
    ) -> float:
        """
        Enhanced CPUE model with FAO-realistic distributions.

        Key improvements:
          1. Lognormal base distribution (matching real CPUE statistics)
          2. Gaussian SST-CPUE relationship (Bell curve around T_opt)
          3. CHL positive correlation (log-linear)
          4. Spatial autocorrelation via fishing ground proximity
          5. Stronger seasonal modulation with sinusoidal fit
          6. Phi-CPUE correlation as primary driver
        """
        # ── 1. Base lognormal CPUE (FAO-realistic right-skewed distribution)
        log_mu = self.cpue_range["log_mu"]
        log_sigma = self.cpue_range["log_sigma"]
        base_cpue = self.rng.lognormal(log_mu, log_sigma * 0.3)

        # ── 2. Gaussian SST response (strongest environmental signal)
        t_opt = self.traits["Topt_C"]
        t_sigma = self.traits["Topt_sigma"]
        thermal_score = np.exp(-0.5 * ((sst - t_opt) / t_sigma) ** 2)

        # ── 3. CHL positive correlation (log-linear)
        chl_score = np.clip(0.3 + 0.4 * np.log10(max(chl, 0.01) / 0.1), 0.1, 1.0)

        # ── 4. Phi (metabolic index) as primary driver
        phi_crit = self.traits["phi_crit"]
        phi_score = np.clip((phi - 1.0) / (phi_crit * 2.0), 0.05, 1.0)

        # ── 5. Spatial autocorrelation (fishing ground proximity)
        fishing_grounds = FISHING_GROUNDS.get(self.species, [])
        spatial_score = 0.5  # baseline
        for g_lat, g_lon, g_rad, g_int in fishing_grounds:
            dist = np.sqrt((lat - g_lat)**2 + ((lon - g_lon) * np.cos(np.radians(lat)))**2)
            proximity = np.exp(-0.5 * (dist / g_rad) ** 2)
            spatial_score = max(spatial_score, g_int * proximity)

        # ── 6. Seasonal modulation (sinusoidal, species-specific)
        PEAK_MONTHS = {
            "yellowfin": 7.0,  # July peak
            "bigeye": 4.5,     # April-May peak
            "skipjack": 7.5,   # July-Aug peak
            "albacore": 6.0,   # June peak
        }
        peak = PEAK_MONTHS.get(self.species, 7.0)
        season_mod = 0.6 + 0.4 * np.cos(2 * np.pi * (month - peak) / 12)

        # ── 7. Front and eddy bonus
        front_bonus = 1.0 + 0.6 * np.clip(front_strength / 0.05, 0, 1)
        eddy_bonus = 1.0 + 0.2 * np.clip(eddy_strength / 300, 0, 1)

        # ── 8. Combine with weighted contributions
        quality = (
            0.35 * thermal_score +
            0.25 * phi_score +
            0.20 * chl_score +
            0.10 * hsi +
            0.10 * np.clip(eddy_strength / 500, 0, 1)
        )

        cpue = base_cpue * quality * spatial_score * season_mod
        cpue *= front_bonus * eddy_bonus

        # ── 9. Multiplicative noise (25% CV — realistic for longline)
        noise = self.rng.lognormal(0, 0.25)
        cpue *= noise

        return np.clip(cpue, self.cpue_range["min"], self.cpue_range["max"])


class FAODataSimulator:
    """
    Generates data that mimics the format of real FAO CPUE downloads,
    suitable for direct use in the training pipeline.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def generate_fao_format(
        self,
        species_list: List[str] = None,
        n_per_species: int = 2000,
        output_dir: str = "data",
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate FAO-format CSV files for each species.

        Returns dict of {species: DataFrame}
        """
        species_list = species_list or ["yellowfin", "bigeye", "albacore"]
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        all_data = {}
        for sp in species_list:
            gen = SyntheticCPUEGenerator(species=sp, seed=self.seed)
            df = gen.generate(n_per_species)

            # Save as FAO-format CSV
            fao_df = df[[
                "year", "month", "lat", "lon", "sst", "chl", "ssh", "do",
                "current_speed", "front_strength", "eddy_strength",
                "phi", "hsi", "cpue_kg_per_day", "species", "day_of_year"
            ]].copy()

            csv_path = output_path / f"synthetic_cpue_{sp}_{df['year'].min()}_{df['year'].max()}.csv"
            fao_df.to_csv(csv_path, index=False)
            log.info(f"  Saved: {csv_path} ({len(fao_df)} records)")
            all_data[sp] = fao_df

        # Also save combined dataset
        combined = pd.concat(all_data.values(), ignore_index=True)
        combined_path = output_path / "synthetic_cpue_all_species.csv"
        combined.to_csv(combined_path, index=False)
        log.info(f"  Combined: {combined_path} ({len(combined)} records)")

        return all_data


# ═══════════════════════════════════════════════════
#  Standalone execution
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    print("=" * 70)
    print("OceanMaster — Synthetic CPUE Data Generator")
    print("=" * 70)

    sim = FAODataSimulator(seed=42)
    data = sim.generate_fao_format(
        species_list=["yellowfin", "bigeye", "albacore"],
        n_per_species=2000,
        output_dir="ml_system/data",
    )

    for sp, df in data.items():
        print(f"\n{sp.upper()}:")
        print(f"  Records: {len(df)}")
        print(f"  CPUE range: {df['cpue_kg_per_day'].min():.1f} - "
              f"{df['cpue_kg_per_day'].max():.1f} kg/day")
        print(f"  CPUE mean±std: {df['cpue_kg_per_day'].mean():.1f} "
              f"± {df['cpue_kg_per_day'].std():.1f}")
        print(f"  Years: {df['year'].min()}-{df['year'].max()}")
        print(f"  SST range: {df['sst'].min():.1f} - {df['sst'].max():.1f} °C")
        print(f"  Phi range: {df['phi'].min():.2f} - {df['phi'].max():.2f}")
