"""
OceanMaster — Feature Engineering Pipeline
============================================
Transforms raw ocean/CPUE data into ML-ready feature matrices.

Integrates:
  - OceanMaster scientific features (Φ, SEAPODYM HSI)
  - Derived physical features (gradients, fronts, eddies)
  - Temporal features (season, moon phase, day-of-year encoding)
  - Spatial features (lat/lon, region encoding)
  - Interaction features (SST×Chl, front×FTLE)
  - Lag features (rolling averages)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from typing import Dict, List, Optional, Tuple
import logging
import joblib
from pathlib import Path

log = logging.getLogger("OceanMaster.FeatureEng")


class FeatureEngineeringPipeline:
    """
    Complete feature engineering pipeline for OceanMaster ML training.

    Takes a DataFrame of CPUE records with ocean variables and produces
    a scaled feature matrix ready for model training.
    """

    # Core feature columns from the synthetic/FAO data
    BASE_FEATURES = [
        "sst", "chl", "ssh", "do", "current_speed",
        "front_strength", "eddy_strength", "phi", "hsi",
    ]

    TEMPORAL_FEATURES = [
        "month_sin", "month_cos", "day_of_year_sin", "day_of_year_cos",
    ]

    SPATIAL_FEATURES = [
        "lat_norm", "lon_norm", "lat_lon_interaction",
    ]

    DERIVED_FEATURES = [
        "chl_log", "sst_squared", "phi_hsi_product",
        "sst_chl_interaction", "front_eddy_interaction",
        "phi_over_crit", "thermal_stress",
    ]

    REGION_FEATURES = [
        "region_kuroshio", "region_warm_pool", "region_subtropical",
    ]

    def __init__(self, species: str = "yellowfin"):
        self.species = species
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_fitted = False

        # Species-specific parameters
        self._phi_crit = {
            "yellowfin": 2.5, "bigeye": 2.0,
            "skipjack": 3.0, "albacore": 2.0,
        }.get(species, 2.5)

        self._t_opt = {
            "yellowfin": 28.0, "bigeye": 18.0,
            "skipjack": 26.0, "albacore": 20.0,
        }.get(species, 28.0)

    def transform(
        self,
        df: pd.DataFrame,
        fit: bool = True,
        include_target: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Full feature engineering pipeline.

        Args:
            df: Raw CPUE DataFrame
            fit: Whether to fit the scaler (True for training, False for inference)
            include_target: If True, also return the CPUE target vector

        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (if include_target) or empty array
            feature_names: List of feature column names
        """
        log.info(f"Feature engineering: {len(df)} samples, species={self.species}")

        features = pd.DataFrame(index=df.index)

        # 1. Base environmental features
        for col in self.BASE_FEATURES:
            if col in df.columns:
                features[col] = df[col].fillna(df[col].median())
            else:
                features[col] = 0.0

        # 2. Temporal features (cyclical encoding)
        if "month" in df.columns:
            features["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            features["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        else:
            features["month_sin"] = 0.0
            features["month_cos"] = 0.0

        if "day_of_year" in df.columns:
            features["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
            features["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
        else:
            features["day_of_year_sin"] = 0.0
            features["day_of_year_cos"] = 0.0

        # 3. Spatial features (normalized)
        if "lat" in df.columns:
            features["lat_norm"] = (df["lat"] - 17.5) / 17.5  # Center of study area
            features["lon_norm"] = (df["lon"] - 147.5) / 27.5
            features["lat_lon_interaction"] = features["lat_norm"] * features["lon_norm"]
        else:
            features["lat_norm"] = 0.0
            features["lon_norm"] = 0.0
            features["lat_lon_interaction"] = 0.0

        # 4. Derived features
        features["chl_log"] = np.log10(np.maximum(features["chl"], 0.001))
        features["sst_squared"] = features["sst"] ** 2
        features["phi_hsi_product"] = features["phi"] * features["hsi"]
        features["sst_chl_interaction"] = features["sst"] * features["chl_log"]
        features["front_eddy_interaction"] = (
            features["front_strength"] * features["eddy_strength"] / 100
        )
        features["phi_over_crit"] = features["phi"] / self._phi_crit
        features["thermal_stress"] = np.abs(features["sst"] - self._t_opt)

        # 5. Region encoding
        if "lat" in df.columns and "lon" in df.columns:
            features["region_kuroshio"] = (
                (df["lat"] > 20) & (df["lat"] < 30) &
                (df["lon"] > 125) & (df["lon"] < 145)
            ).astype(float)
            features["region_warm_pool"] = (
                (df["lat"] > 0) & (df["lat"] < 15) &
                (df["lon"] > 140) & (df["lon"] < 175)
            ).astype(float)
            features["region_subtropical"] = (
                (df["lat"] > 25) & (df["lat"] < 35)
            ).astype(float)
        else:
            features["region_kuroshio"] = 0.0
            features["region_warm_pool"] = 0.0
            features["region_subtropical"] = 0.0

        # Finalize feature names
        self.feature_names = list(features.columns)
        X = features.values.astype(np.float64)

        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise RuntimeError("Pipeline not fitted. Call with fit=True first.")
            X_scaled = self.scaler.transform(X)

        # Target
        y = np.array([])
        if include_target and "cpue_kg_per_day" in df.columns:
            y = df["cpue_kg_per_day"].values.astype(np.float64)

        log.info(f"  Output: X={X_scaled.shape}, features={len(self.feature_names)}")
        return X_scaled, y, self.feature_names

    def save(self, path: str = "models/feature_pipeline.pkl"):
        """Save fitted pipeline to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "species": self.species,
            "is_fitted": self.is_fitted,
            "phi_crit": self._phi_crit,
            "t_opt": self._t_opt,
        }, path)
        log.info(f"Pipeline saved: {path}")

    def load(self, path: str = "models/feature_pipeline.pkl"):
        """Load fitted pipeline from disk."""
        data = joblib.load(path)
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self.species = data["species"]
        self.is_fitted = data["is_fitted"]
        self._phi_crit = data["phi_crit"]
        self._t_opt = data["t_opt"]
        log.info(f"Pipeline loaded: {path} ({len(self.feature_names)} features)")

    def get_feature_summary(self) -> pd.DataFrame:
        """Return summary of feature names and categories."""
        categories = {}
        for name in self.feature_names:
            if name in self.BASE_FEATURES:
                cat = "Base Environmental"
            elif name in self.TEMPORAL_FEATURES:
                cat = "Temporal"
            elif name in self.SPATIAL_FEATURES:
                cat = "Spatial"
            elif name in [f.replace("region_", "") for f in self.REGION_FEATURES]:
                cat = "Region"
            elif "region_" in name:
                cat = "Region"
            else:
                cat = "Derived"
            categories[name] = cat

        return pd.DataFrame([
            {"feature": k, "category": v}
            for k, v in categories.items()
        ])


# ═══════════════════════════════════════════════════
#  Standalone test
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    from historical_data_collector import SyntheticCPUEGenerator

    print("=" * 70)
    print("Feature Engineering Pipeline — Test")
    print("=" * 70)

    # Generate test data
    gen = SyntheticCPUEGenerator(species="yellowfin", seed=42)
    df = gen.generate(500)

    # Run pipeline
    pipeline = FeatureEngineeringPipeline(species="yellowfin")
    X, y, names = pipeline.transform(df, fit=True, include_target=True)

    print(f"\nFeature matrix: {X.shape}")
    print(f"Target vector: {y.shape}")
    print(f"Feature names ({len(names)}):")
    for i, name in enumerate(names):
        print(f"  {i+1:2d}. {name}")

    # Summary stats
    print(f"\nX stats: mean={X.mean():.4f}, std={X.std():.4f}")
    print(f"y stats: mean={y.mean():.1f}, std={y.std():.1f}, "
          f"range=[{y.min():.1f}, {y.max():.1f}]")

    # Save
    pipeline.save("ml_system/models/feature_pipeline.pkl")
    print("\nPipeline saved successfully.")
