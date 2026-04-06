"""
OceanMaster v12 — Phase 9: FAO Data Loader  # [v12-phase9]
============================================================
Load and process FAO/RFMO CPUE data for ML training.
Ready for real data when available — currently provides the interface.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, List, Optional

log = logging.getLogger("OceanMaster.FAO")


class FAODataLoader:
    """
    Load and process FAO CPUE data for ML training.

    Expected CSV format (from FAO FishStatJ or WCPFC):
      year, month, lat_5x5, lon_5x5, species_code, catch_mt, effort_days, cpue

    Also supports:
      - WCPFC public domain data
      - IOTC nominal catch data
      - IATTC public data
    """

    SPECIES_MAP = {
        "YFT": "yellowfin",
        "BET": "bigeye",
        "ALB": "albacore",
        "SKJ": "skipjack",
    }

    REQUIRED_COLUMNS = ["year", "month", "lat_5x5", "lon_5x5", "species_code"]
    CPUE_COLUMNS = ["cpue", "catch_mt", "catch_kg"]

    def __init__(self):
        self.raw_df: Optional[pd.DataFrame] = None

    def load(self, csv_path: str) -> pd.DataFrame:
        """Load and clean FAO/RFMO CPUE data."""
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"FAO data not found: {csv_path}")

        df = pd.read_csv(csv_path)
        log.info(f"Loaded FAO data: {len(df)} records from {csv_path}")

        # Normalize column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Map species codes
        if "species_code" in df.columns:
            df["species"] = df["species_code"].map(self.SPECIES_MAP)
            df = df.dropna(subset=["species"])

        # Compute CPUE if not present
        if "cpue" not in df.columns:
            if "catch_mt" in df.columns and "effort_days" in df.columns:
                df["cpue"] = (df["catch_mt"] * 1000) / df["effort_days"].clip(lower=0.1)
            elif "catch_kg" in df.columns and "effort_days" in df.columns:
                df["cpue"] = df["catch_kg"] / df["effort_days"].clip(lower=0.1)
            else:
                raise ValueError("Cannot compute CPUE: need catch + effort columns")

        # Clean
        df = df[df["cpue"] > 0].copy()
        df = df.dropna(subset=["cpue"])

        log.info(f"  Cleaned: {len(df)} valid CPUE records")
        self.raw_df = df
        return df

    def align_with_oceanmaster(
        self,
        cpue_df: pd.DataFrame,
        feature_names: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match CPUE records with OceanMaster feature vector format.

        Since we can't fetch real ocean data for each historical point,
        this creates a feature matrix from the available columns and
        fills missing features with climatological defaults.
        """
        from engine.ml.stacking_ensemble import FeatureEngineer

        n = len(cpue_df)
        n_features = len(feature_names)
        X = np.zeros((n, n_features))

        # Map available columns
        col_map = {
            "sst": "sst",
            "chl_log": None,  # Will compute from chl if available
            "ssh": "ssh",
            "current_speed": "current_speed",
        }

        for i, fname in enumerate(feature_names):
            src_col = col_map.get(fname)
            if src_col and src_col in cpue_df.columns:
                X[:, i] = cpue_df[src_col].values
            elif fname == "chl_log" and "chl" in cpue_df.columns:
                X[:, i] = np.log10(cpue_df["chl"].clip(lower=0.001).values)
            elif fname == "moon_phase":
                # Approximate from date
                X[:, i] = 0.5  # default
            elif fname == "season_sin" and "month" in cpue_df.columns:
                doy = cpue_df["month"].values * 30
                X[:, i] = np.sin(2 * np.pi * doy / 365.25)
            elif fname == "season_cos" and "month" in cpue_df.columns:
                doy = cpue_df["month"].values * 30
                X[:, i] = np.cos(2 * np.pi * doy / 365.25)
            # else: stays at 0 (default)

        y = cpue_df["cpue"].values

        # Normalize CPUE to [0, 1]
        y_max = np.percentile(y, 99)
        if y_max > 0:
            y = np.clip(y / y_max, 0, 1)

        log.info(f"  Aligned: {X.shape} features, {len(y)} labels")
        return X, y

    def get_training_ready(
        self, csv_path: str, species: str = "yellowfin",
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        One-call method: load → clean → align → return (X, y, feature_names).
        """
        from engine.ml.stacking_ensemble import FeatureEngineer

        df = self.load(csv_path)

        # Filter to target species
        if "species" in df.columns:
            df = df[df["species"] == species].copy()
            if len(df) == 0:
                raise ValueError(f"No records for species '{species}'")

        feature_names = list(FeatureEngineer.FEATURE_NAMES)
        X, y = self.align_with_oceanmaster(df, feature_names)

        log.info(f"  Training ready: {species} — {X.shape[0]} samples, {X.shape[1]} features")
        return X, y, feature_names
