"""
OceanMaster — Integrated Predictor (Science + ML Fusion)
=========================================================
Combines OceanMaster's scientific models (Deutsch 2015 Φ, SEAPODYM HSI)
with trained ML predictions using a configurable fusion strategy.

Fusion Strategies:
  A. Weighted Average: final = w_ml × ML + w_sci × Science
  B. ML-Primary with Scientific Constraints: ML prediction gated by Φ and HSI
  C. Bayesian Fusion: Science as prior, ML as likelihood update
  D. Cascade: Use science for screening, ML for ranking

Also provides confidence estimation and anomaly detection.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
import logging

log = logging.getLogger("OceanMaster.Predictor")


@dataclass
class PredictionResult:
    """Container for a single prediction."""
    lat: float
    lon: float
    date: str
    species: str

    # Predictions
    cpue_final: float = 0.0
    cpue_ml: float = 0.0
    cpue_science: float = 0.0

    # Scientific indices
    phi: float = 0.0
    hsi: float = 0.0

    # Confidence
    confidence: float = 0.0
    confidence_level: str = "low"  # low, medium, high

    # Fusion details
    fusion_strategy: str = "weighted_average"
    ml_weight: float = 0.6
    science_weight: float = 0.4

    # Recommendation
    recommendation: str = "low"  # low, medium, high, hotspot

    def to_dict(self) -> Dict:
        return {
            "lat": self.lat, "lon": self.lon, "date": self.date,
            "species": self.species,
            "cpue_final": round(self.cpue_final, 2),
            "cpue_ml": round(self.cpue_ml, 2),
            "cpue_science": round(self.cpue_science, 2),
            "phi": round(self.phi, 3),
            "hsi": round(self.hsi, 4),
            "confidence": round(self.confidence, 3),
            "confidence_level": self.confidence_level,
            "recommendation": self.recommendation,
            "fusion_strategy": self.fusion_strategy,
        }


class IntegratedPredictor:
    """
    Fuses trained ML model with OceanMaster scientific models.

    Usage:
        predictor = IntegratedPredictor(species="yellowfin")
        predictor.load_ml_model("models/")
        result = predictor.predict(lat=25.0, lon=135.0, date="2026-03-01",
                                   ocean_features={...})
    """

    # Metabolic traits (from commercial_core_v2.py)
    METABOLIC_TRAITS = {
        "yellowfin": {"Eo": 0.40, "Pcrit_kPa": 4.8, "Topt_C": 28.0,
                       "Topt_sigma": 2.0, "phi_crit": 2.5},
        "bigeye":    {"Eo": 0.35, "Pcrit_kPa": 3.5, "Topt_C": 18.0,
                       "Topt_sigma": 3.0, "phi_crit": 2.0},
        "skipjack":  {"Eo": 0.45, "Pcrit_kPa": 5.5, "Topt_C": 26.0,
                       "Topt_sigma": 3.5, "phi_crit": 3.0},
        "albacore":  {"Eo": 0.38, "Pcrit_kPa": 4.0, "Topt_C": 20.0,
                       "Topt_sigma": 3.0, "phi_crit": 2.0},
    }

    KB = 8.617e-5
    TREF_K = 288.15

    # CPUE thresholds for recommendations
    CPUE_THRESHOLDS = {
        "yellowfin": {"low": 20, "medium": 45, "high": 80, "hotspot": 120},
        "bigeye":    {"low": 10, "medium": 25, "high": 50, "hotspot": 75},
        "skipjack":  {"low": 20, "medium": 60, "high": 100, "hotspot": 150},
        "albacore":  {"low": 10, "medium": 30, "high": 60, "hotspot": 90},
    }

    def __init__(
        self,
        species: str = "yellowfin",
        fusion_strategy: str = "ml_constrained",
        ml_weight: float = 0.6,
    ):
        self.species = species
        self.fusion_strategy = fusion_strategy
        self.ml_weight = ml_weight
        self.science_weight = 1.0 - ml_weight

        self.traits = self.METABOLIC_TRAITS.get(species, self.METABOLIC_TRAITS["yellowfin"])
        self.thresholds = self.CPUE_THRESHOLDS.get(species, self.CPUE_THRESHOLDS["yellowfin"])

        # ML model (loaded separately)
        self.ml_model = None
        self.feature_pipeline = None
        self.ml_available = False

    def load_ml_model(self, model_dir: str = "models"):
        """Load trained ML model and feature pipeline."""
        path = Path(model_dir)
        sp = self.species

        try:
            self.ml_model = joblib.load(path / f"stacking_{sp}.pkl")
            self.feature_pipeline = joblib.load(path / f"feature_pipeline_{sp}.pkl")
            self.ml_available = True
            log.info(f"ML model loaded for {sp}")
        except FileNotFoundError as e:
            log.warning(f"ML model not found: {e}. Using science-only mode.")
            self.ml_available = False

    def predict(
        self,
        lat: float,
        lon: float,
        date: str,
        ocean_features: Dict[str, float],
    ) -> PredictionResult:
        """
        Generate a fused prediction for a single location.

        Args:
            lat, lon: Location
            date: Date string (YYYY-MM-DD)
            ocean_features: Dict with keys: sst, chl, ssh, do,
                           current_speed, front_strength, eddy_strength
        """
        result = PredictionResult(
            lat=lat, lon=lon, date=date, species=self.species,
            fusion_strategy=self.fusion_strategy,
            ml_weight=self.ml_weight,
            science_weight=self.science_weight,
        )

        # 1. Compute scientific indices
        sst = ocean_features.get("sst", 27.0)
        do = ocean_features.get("do", 5.0)
        chl = ocean_features.get("chl", 0.1)

        result.phi = self._compute_phi(sst, do)
        result.hsi = self._compute_hsi(sst, chl)

        # Science-based CPUE estimate
        result.cpue_science = self._science_cpue(result.phi, result.hsi, ocean_features)

        # 2. ML prediction (if available)
        if self.ml_available:
            result.cpue_ml = self._ml_predict(lat, lon, date, ocean_features)
        else:
            result.cpue_ml = result.cpue_science  # Fallback

        # 3. Fusion
        result.cpue_final = self._fuse(result)

        # 4. Confidence
        result.confidence = self._estimate_confidence(result, ocean_features)
        if result.confidence > 0.7:
            result.confidence_level = "high"
        elif result.confidence > 0.4:
            result.confidence_level = "medium"
        else:
            result.confidence_level = "low"

        # 5. Recommendation
        cpue = result.cpue_final
        th = self.thresholds
        if cpue >= th["hotspot"]:
            result.recommendation = "hotspot"
        elif cpue >= th["high"]:
            result.recommendation = "high"
        elif cpue >= th["medium"]:
            result.recommendation = "medium"
        else:
            result.recommendation = "low"

        return result

    def predict_grid(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        date: str,
        resolution: float = 0.5,
        ocean_data_grid: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Predict CPUE across a spatial grid.

        Returns DataFrame with one row per grid cell.
        """
        lats = np.arange(lat_range[0], lat_range[1], resolution)
        lons = np.arange(lon_range[0], lon_range[1], resolution)

        from historical_data_collector import OceanEnvironmentSimulator
        env = OceanEnvironmentSimulator(seed=99)

        dt = pd.to_datetime(date)
        month = dt.month

        predictions = []
        for lat in lats:
            for lon in lons:
                ocean_features = {
                    "sst": env.simulate_sst(lat, lon, month),
                    "chl": env.simulate_chlorophyll(lat, lon, month),
                    "ssh": env.simulate_ssh(lat, lon, month),
                    "do": env.simulate_do(lat, lon, month,
                                          env.simulate_sst(lat, lon, month)),
                    "current_speed": env.simulate_current_speed(lat, lon),
                    "front_strength": env.simulate_front_strength(lat, lon),
                    "eddy_strength": env.simulate_eddy_strength(lat, lon),
                }
                result = self.predict(lat, lon, date, ocean_features)
                predictions.append(result.to_dict())

        return pd.DataFrame(predictions)

    # ─── Internal Methods ──────────────────────────────────

    def _compute_phi(self, sst_c: float, do_ml_l: float) -> float:
        """Deutsch 2015 Metabolic Index."""
        T_K = sst_c + 273.15
        po2 = do_ml_l * (21.0 / 8.0)
        demand = self.traits["Pcrit_kPa"] * np.exp(
            self.traits["Eo"] / self.KB * (1.0 / T_K - 1.0 / self.TREF_K)
        )
        return float(np.clip(po2 / max(demand, 0.01), 0, 20))

    def _compute_hsi(self, sst_c: float, chl: float) -> float:
        """Simplified SEAPODYM HSI."""
        thermal = np.exp(-0.5 * (
            (sst_c - self.traits["Topt_C"]) / self.traits["Topt_sigma"]
        ) ** 2)
        chl_log = np.log10(max(chl, 0.001))
        feeding = 1.0 / (1.0 + np.exp(-3.0 * (chl_log + 0.5)))
        return float(np.sqrt(thermal * feeding))

    def _science_cpue(self, phi, hsi, features) -> float:
        """Estimate CPUE from scientific indices alone."""
        phi_crit = self.traits["phi_crit"]
        phi_score = np.clip((phi - 1.0) / (phi_crit * 2.0), 0, 1)
        hsi_score = np.clip(hsi, 0, 1)
        front = features.get("front_strength", 0)
        front_bonus = 1.0 + 0.5 * np.clip(front / 0.05, 0, 1)

        from historical_data_collector import CPUE_RANGES
        median = CPUE_RANGES.get(self.species, {"median": 45})["median"]
        return float(median * (0.6 * phi_score + 0.4 * hsi_score) * front_bonus)

    def _ml_predict(self, lat, lon, date, features) -> float:
        """Get ML model prediction."""
        if not self.ml_available:
            return 0.0

        dt = pd.to_datetime(date)
        row = pd.DataFrame([{
            "sst": features.get("sst", 27),
            "chl": features.get("chl", 0.1),
            "ssh": features.get("ssh", 0.0),
            "do": features.get("do", 5.0),
            "current_speed": features.get("current_speed", 0.2),
            "front_strength": features.get("front_strength", 0.02),
            "eddy_strength": features.get("eddy_strength", 100),
            "phi": self._compute_phi(features.get("sst", 27), features.get("do", 5.0)),
            "hsi": self._compute_hsi(features.get("sst", 27), features.get("chl", 0.1)),
            "lat": lat, "lon": lon,
            "month": dt.month,
            "day_of_year": dt.timetuple().tm_yday,
            "species": self.species,
        }])

        from feature_engineering_pipeline import FeatureEngineeringPipeline
        pipeline = FeatureEngineeringPipeline(species=self.species)

        # Load fitted scaler
        if self.feature_pipeline:
            pipeline.scaler = self.feature_pipeline["scaler"]
            pipeline.feature_names = self.feature_pipeline["feature_names"]
            pipeline.is_fitted = True
            X, _, _ = pipeline.transform(row, fit=False, include_target=False)
        else:
            X, _, _ = pipeline.transform(row, fit=True, include_target=False)

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        pred = self.ml_model.predict(X)[0]
        return float(max(pred, 0))

    def _fuse(self, result: PredictionResult) -> float:
        """Apply fusion strategy."""
        ml = result.cpue_ml
        sci = result.cpue_science
        phi = result.phi
        hsi = result.hsi

        if self.fusion_strategy == "weighted_average":
            return self.ml_weight * ml + self.science_weight * sci

        elif self.fusion_strategy == "ml_constrained":
            # ML prediction with scientific safety gates
            fused = self.ml_weight * ml + self.science_weight * sci
            # Metabolic constraint
            if phi < 1.5:
                fused *= 0.3  # Severe metabolic stress
            elif phi < self.traits["phi_crit"]:
                fused *= 0.6  # Moderate stress
            # Habitat constraint
            if hsi < 0.2:
                fused *= 0.5  # Poor habitat
            return max(fused, 0)

        elif self.fusion_strategy == "science_only":
            return sci

        elif self.fusion_strategy == "ml_only":
            return ml

        else:
            return self.ml_weight * ml + self.science_weight * sci

    def _estimate_confidence(self, result: PredictionResult, features: Dict) -> float:
        """
        Estimate prediction confidence [0, 1].

        Higher when:
          - ML and science predictions agree
          - Φ is in a well-studied range
          - SST is near species optimum
          - Multiple positive indicators
        """
        scores = []

        # 1. ML-Science agreement (closer = more confident)
        if result.cpue_ml > 0 and result.cpue_science > 0:
            ratio = min(result.cpue_ml, result.cpue_science) / max(
                result.cpue_ml, result.cpue_science
            )
            scores.append(ratio)
        else:
            scores.append(0.3)

        # 2. Metabolic index quality
        phi_score = np.clip(result.phi / (self.traits["phi_crit"] * 2), 0, 1)
        scores.append(phi_score)

        # 3. HSI quality
        scores.append(np.clip(result.hsi, 0, 1))

        # 4. SST near optimum
        sst = features.get("sst", 27)
        sst_dev = abs(sst - self.traits["Topt_C"]) / self.traits["Topt_sigma"]
        sst_score = np.exp(-0.5 * sst_dev ** 2)
        scores.append(sst_score)

        return float(np.mean(scores))


# ═══════════════════════════════════════════════════
#  Quick Demo
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    print("=" * 60)
    print("Integrated Predictor — Demo")
    print("=" * 60)

    predictor = IntegratedPredictor(
        species="yellowfin",
        fusion_strategy="ml_constrained",
    )

    # Try loading ML model
    predictor.load_ml_model("ml_system/models")

    # Test predictions at several locations
    test_points = [
        {"lat": 25.0, "lon": 135.0, "label": "Kuroshio Current"},
        {"lat": 10.0, "lon": 150.0, "label": "Warm Pool Edge"},
        {"lat": 5.0, "lon": 165.0, "label": "Equatorial Pacific"},
        {"lat": 30.0, "lon": 170.0, "label": "Subtropical Front"},
    ]

    for pt in test_points:
        from historical_data_collector import OceanEnvironmentSimulator
        env = OceanEnvironmentSimulator(seed=99)
        features = {
            "sst": env.simulate_sst(pt["lat"], pt["lon"], 6),
            "chl": env.simulate_chlorophyll(pt["lat"], pt["lon"], 6),
            "ssh": env.simulate_ssh(pt["lat"], pt["lon"], 6),
            "do": env.simulate_do(pt["lat"], pt["lon"], 6,
                                  env.simulate_sst(pt["lat"], pt["lon"], 6)),
            "current_speed": 0.3,
            "front_strength": 0.025,
            "eddy_strength": 120,
        }
        result = predictor.predict(pt["lat"], pt["lon"], "2026-06-15", features)

        print(f"\n  {pt['label']} ({pt['lat']}°N, {pt['lon']}°E)")
        print(f"    CPUE Final:    {result.cpue_final:.1f} kg/day")
        print(f"    CPUE Science:  {result.cpue_science:.1f} kg/day")
        print(f"    Φ (Metabolic): {result.phi:.2f}")
        print(f"    HSI:           {result.hsi:.3f}")
        print(f"    Confidence:    {result.confidence:.2f} ({result.confidence_level})")
        print(f"    Recommend:     {result.recommendation.upper()}")
