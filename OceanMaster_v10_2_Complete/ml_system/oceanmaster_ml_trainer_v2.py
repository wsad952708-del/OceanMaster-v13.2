"""
OceanMaster — ML Model Trainer v2
==================================
Complete training pipeline that:
  1. Loads synthetic or real CPUE data
  2. Runs feature engineering
  3. Trains Stacking Ensemble (RF + GBR + Ridge → RidgeCV Meta)
  4. Cross-validates with multiple strategies
  5. Computes feature importance
  6. Saves trained model artifacts

Compatible with the existing engine/ml/stacking_ensemble.py architecture.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import joblib
import json
import logging
from typing import Dict, List, Optional, Tuple

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    median_absolute_error,
)

log = logging.getLogger("OceanMaster.Trainer")

# Try optional libs
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


class TrainingConfig:
    """Training hyperparameters and configuration."""

    def __init__(
        self,
        species: str = "yellowfin",
        test_size: float = 0.2,
        cv_folds: int = 5,
        random_state: int = 42,
        n_estimators_rf: int = 200,
        n_estimators_gbr: int = 150,
        max_depth_rf: int = 12,
        max_depth_gbr: int = 6,
        learning_rate_gbr: float = 0.05,
    ):
        self.species = species
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_estimators_rf = n_estimators_rf
        self.n_estimators_gbr = n_estimators_gbr
        self.max_depth_rf = max_depth_rf
        self.max_depth_gbr = max_depth_gbr
        self.learning_rate_gbr = learning_rate_gbr

    def to_dict(self) -> Dict:
        return self.__dict__


class OceanMasterMLTrainer:
    """
    Complete ML training pipeline for OceanMaster.

    Usage:
        trainer = OceanMasterMLTrainer(config)
        results = trainer.train(X_train, y_train, feature_names)
        metrics = trainer.evaluate(X_test, y_test)
        trainer.save("models/")
    """

    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.model: Optional[StackingRegressor] = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.training_metrics: Dict = {}
        self.is_trained = False

    def build_model(self) -> StackingRegressor:
        """
        Build the Stacking Ensemble model.

        Architecture:
          Level-0: RF + GBR + ExtraTrees + Ridge (+ XGBoost/LightGBM if available)
          Level-1: RidgeCV meta-learner
        """
        cfg = self.config

        estimators = [
            ("rf", RandomForestRegressor(
                n_estimators=cfg.n_estimators_rf,
                max_depth=cfg.max_depth_rf,
                min_samples_leaf=10,
                n_jobs=-1,
                random_state=cfg.random_state,
            )),
            ("gbr", GradientBoostingRegressor(
                n_estimators=cfg.n_estimators_gbr,
                max_depth=cfg.max_depth_gbr,
                learning_rate=cfg.learning_rate_gbr,
                subsample=0.8,
                random_state=cfg.random_state,
            )),
            ("et", ExtraTreesRegressor(
                n_estimators=cfg.n_estimators_rf,
                max_depth=cfg.max_depth_rf,
                min_samples_leaf=10,
                n_jobs=-1,
                random_state=cfg.random_state,
            )),
            ("ridge", Ridge(alpha=1.0)),
        ]

        if HAS_XGB:
            estimators.append(("xgb", xgb.XGBRegressor(
                n_estimators=200, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                n_jobs=-1, random_state=cfg.random_state,
            )))
            log.info("  + XGBoost added")

        if HAS_LGB:
            estimators.append(("lgb", lgb.LGBMRegressor(
                n_estimators=200, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                n_jobs=-1, random_state=cfg.random_state, verbose=-1,
            )))
            log.info("  + LightGBM added")

        self.model = StackingRegressor(
            estimators=estimators,
            final_estimator=RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]),
            cv=KFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state),
            n_jobs=-1,
        )

        log.info(f"Stacking model built: {len(estimators)} base learners + RidgeCV meta")
        return self.model

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None,
    ) -> Dict:
        """
        Train the stacking ensemble.

        Args:
            X: Feature matrix (n_samples, n_features), already scaled
            y: Target vector (CPUE in kg/day)
            feature_names: Names of feature columns

        Returns:
            Training metrics dictionary
        """
        log.info("=" * 60)
        log.info(f"Training {self.config.species} ML Model")
        log.info("=" * 60)
        log.info(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")

        self.feature_names = feature_names or [f"feat_{i}" for i in range(X.shape[1])]

        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Build model
        if self.model is None:
            self.build_model()

        # Train
        log.info("  Fitting stacking ensemble...")
        self.model.fit(X, y)
        self.is_trained = True
        log.info("  Model trained successfully.")

        # In-sample metrics
        y_pred_train = self.model.predict(X)
        train_r2 = r2_score(y, y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(y, y_pred_train))

        # Cross-validation
        log.info("  Running cross-validation...")
        kf = KFold(
            n_splits=self.config.cv_folds, shuffle=True,
            random_state=self.config.random_state
        )
        cv_scores = cross_val_score(
            self.model, X, y, cv=kf,
            scoring="neg_mean_absolute_error", n_jobs=-1
        )
        cv_mae = -cv_scores.mean()
        cv_mae_std = cv_scores.std()

        cv_r2_scores = cross_val_score(
            self.model, X, y, cv=kf,
            scoring="r2", n_jobs=-1
        )
        cv_r2 = cv_r2_scores.mean()

        # Feature importance (from Random Forest base learner)
        feature_importance = self._compute_feature_importance()

        self.training_metrics = {
            "species": self.config.species,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "train_r2": float(train_r2),
            "train_rmse": float(train_rmse),
            "cv_mae": float(cv_mae),
            "cv_mae_std": float(cv_mae_std),
            "cv_r2": float(cv_r2),
            "cv_r2_std": float(cv_r2_scores.std()),
            "feature_importance": feature_importance,
            "timestamp": datetime.utcnow().isoformat(),
            "config": self.config.to_dict(),
        }

        log.info(f"  Train R²: {train_r2:.4f}")
        log.info(f"  CV R²:    {cv_r2:.4f} ± {cv_r2_scores.std():.4f}")
        log.info(f"  CV MAE:   {cv_mae:.2f} ± {cv_mae_std:.2f} kg/day")

        return self.training_metrics

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        """
        Evaluate on held-out test set.

        Returns comprehensive metrics dictionary.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        y_pred = self.model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        med_ae = median_absolute_error(y_test, y_pred)

        # Accuracy within tolerance bands
        rel_errors = np.abs(y_pred - y_test) / np.maximum(y_test, 1.0)
        acc_10 = float(np.mean(rel_errors < 0.10) * 100)
        acc_20 = float(np.mean(rel_errors < 0.20) * 100)
        acc_30 = float(np.mean(rel_errors < 0.30) * 100)

        # Trend accuracy: does prediction get relative ordering right?
        # (Spearman-like)
        from scipy.stats import spearmanr
        spearman_r, spearman_p = spearmanr(y_test, y_pred)

        # Residual analysis
        residuals = y_pred - y_test
        residual_mean = float(np.mean(residuals))
        residual_std = float(np.std(residuals))

        metrics = {
            "test_rmse": float(rmse),
            "test_mae": float(mae),
            "test_r2": float(r2),
            "test_median_ae": float(med_ae),
            "accuracy_within_10pct": acc_10,
            "accuracy_within_20pct": acc_20,
            "accuracy_within_30pct": acc_30,
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "residual_mean": residual_mean,
            "residual_std": residual_std,
            "n_test_samples": int(len(y_test)),
            "predictions": y_pred.tolist(),
            "actuals": y_test.tolist(),
        }

        log.info("=" * 60)
        log.info("Test Set Evaluation")
        log.info("=" * 60)
        log.info(f"  RMSE:        {rmse:.2f} kg/day")
        log.info(f"  MAE:         {mae:.2f} kg/day")
        log.info(f"  R²:          {r2:.4f}")
        log.info(f"  Median AE:   {med_ae:.2f} kg/day")
        log.info(f"  Acc (±10%):  {acc_10:.1f}%")
        log.info(f"  Acc (±20%):  {acc_20:.1f}%")
        log.info(f"  Acc (±30%):  {acc_30:.1f}%")
        log.info(f"  Spearman r:  {spearman_r:.4f}")

        return metrics

    def evaluate_individual_models(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Dict]:
        """Evaluate each base learner individually for comparison."""
        if not self.is_trained:
            raise RuntimeError("Model not trained.")

        results = {}
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        for name, estimator in self.model.named_estimators_.items():
            y_pred = estimator.predict(X_test)
            results[name] = {
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "r2": float(r2_score(y_test, y_pred)),
            }
            log.info(f"  {name:8s}: R²={results[name]['r2']:.4f}, "
                     f"MAE={results[name]['mae']:.2f}")

        # Ensemble
        y_pred_ens = self.model.predict(X_test)
        results["ensemble"] = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_ens))),
            "mae": float(mean_absolute_error(y_test, y_pred_ens)),
            "r2": float(r2_score(y_test, y_pred_ens)),
        }
        log.info(f"  {'ENSEMBLE':8s}: R²={results['ensemble']['r2']:.4f}, "
                 f"MAE={results['ensemble']['mae']:.2f}")

        return results

    def _compute_feature_importance(self) -> List[Dict]:
        """Extract feature importance from RF base learner."""
        if not self.is_trained or self.model is None:
            return []

        try:
            rf_model = self.model.named_estimators_["rf"]
            importances = rf_model.feature_importances_
            pairs = sorted(
                zip(self.feature_names, importances),
                key=lambda x: x[1], reverse=True
            )
            result = [
                {"name": name, "importance": float(imp)}
                for name, imp in pairs
            ]
            log.info("  Top features:")
            for item in result[:10]:
                log.info(f"    {item['name']:30s} {item['importance']:.4f}")
            return result
        except Exception as e:
            log.warning(f"Could not extract feature importance: {e}")
            return []

    def save(self, model_dir: str = "models"):
        """Save trained model and all artifacts."""
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)

        sp = self.config.species

        # Model
        joblib.dump(self.model, path / f"stacking_{sp}.pkl")

        # Scaler (if separate from pipeline)
        joblib.dump(self.scaler, path / f"scaler_{sp}.pkl")

        # Feature names
        with open(path / f"feature_names_{sp}.txt", "w") as f:
            f.write("\n".join(self.feature_names))

        # Metrics
        with open(path / f"training_metrics_{sp}.json", "w") as f:
            # Remove non-serializable items
            safe_metrics = {
                k: v for k, v in self.training_metrics.items()
            }
            json.dump(safe_metrics, f, indent=2, default=str)

        log.info(f"Model artifacts saved to {path}/")

    def load(self, model_dir: str = "models"):
        """Load trained model from disk."""
        path = Path(model_dir)
        sp = self.config.species

        self.model = joblib.load(path / f"stacking_{sp}.pkl")
        self.is_trained = True

        feat_path = path / f"feature_names_{sp}.txt"
        if feat_path.exists():
            self.feature_names = feat_path.read_text().strip().split("\n")

        log.info(f"Model loaded from {path}/")


# ═══════════════════════════════════════════════════
#  Full Training Pipeline (end-to-end)
# ═══════════════════════════════════════════════════

def run_full_training(
    species: str = "yellowfin",
    n_samples: int = 2000,
    data_dir: str = "ml_system/data",
    model_dir: str = "ml_system/models",
    report_dir: str = "ml_system/reports",
) -> Dict:
    """
    End-to-end training: generate data → engineer features → train → evaluate.

    Returns dict with all metrics and paths.
    """
    from historical_data_collector import SyntheticCPUEGenerator
    from feature_engineering_pipeline import FeatureEngineeringPipeline

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"OceanMaster ML Training Pipeline — {species.upper()}")
    print("=" * 70)

    # Step 1: Generate synthetic data
    print("\n[Step 1/5] Generating training data...")
    gen = SyntheticCPUEGenerator(species=species, seed=42)
    df = gen.generate(n_samples)
    csv_path = Path(data_dir) / f"synthetic_cpue_{species}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved {len(df)} records to {csv_path}")

    # Step 2: Feature engineering
    print("\n[Step 2/5] Feature engineering...")
    pipeline = FeatureEngineeringPipeline(species=species)
    X, y, feature_names = pipeline.transform(df, fit=True, include_target=True)
    pipeline.save(f"{model_dir}/feature_pipeline_{species}.pkl")
    print(f"  Features: {X.shape[1]}, Samples: {X.shape[0]}")

    # Step 3: Train/test split
    print("\n[Step 3/5] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Step 4: Train
    print("\n[Step 4/5] Training stacking ensemble...")
    config = TrainingConfig(species=species)
    trainer = OceanMasterMLTrainer(config)
    train_metrics = trainer.train(X_train, y_train, feature_names)

    # Step 5: Evaluate
    print("\n[Step 5/5] Evaluating on test set...")
    test_metrics = trainer.evaluate(X_test, y_test)

    # Individual model comparison
    print("\n--- Individual Model Comparison ---")
    model_comparison = trainer.evaluate_individual_models(X_test, y_test)

    # Save everything
    trainer.save(model_dir)

    # Generate report
    report = _generate_training_report(
        species, train_metrics, test_metrics, model_comparison,
        len(df), X.shape[1]
    )
    report_path = Path(report_dir) / f"training_report_{species}.txt"
    report_path.write_text(report)
    print(f"\nReport saved: {report_path}")

    # Save results JSON
    results = {
        "species": species,
        "training": train_metrics,
        "testing": test_metrics,
        "model_comparison": model_comparison,
        "data_path": str(csv_path),
        "model_path": str(Path(model_dir) / f"stacking_{species}.pkl"),
    }
    results_path = Path(report_dir) / f"training_results_{species}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Train R²:    {train_metrics['train_r2']:.4f}")
    print(f"  CV R²:       {train_metrics['cv_r2']:.4f}")
    print(f"  Test R²:     {test_metrics['test_r2']:.4f}")
    print(f"  Test MAE:    {test_metrics['test_mae']:.2f} kg/day")
    print(f"  Acc (±20%):  {test_metrics['accuracy_within_20pct']:.1f}%")
    print(f"  Spearman:    {test_metrics['spearman_r']:.4f}")

    return results


def _generate_training_report(
    species, train_metrics, test_metrics, model_comparison,
    n_total, n_features
) -> str:
    """Generate human-readable training report."""
    report = f"""
╔══════════════════════════════════════════════════════════════╗
║  OceanMaster ML Training Report                             ║
╠══════════════════════════════════════════════════════════════╣

  Species:         {species}
  Date:            {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
  Total samples:   {n_total}
  Features:        {n_features}

═══════════════════════════════════════════════════════════════
  TRAINING METRICS
═══════════════════════════════════════════════════════════════
  Train R²:         {train_metrics['train_r2']:.4f}
  Train RMSE:       {train_metrics['train_rmse']:.2f} kg/day
  CV R²:            {train_metrics['cv_r2']:.4f} ± {train_metrics['cv_r2_std']:.4f}
  CV MAE:           {train_metrics['cv_mae']:.2f} ± {train_metrics['cv_mae_std']:.2f} kg/day

═══════════════════════════════════════════════════════════════
  TEST SET METRICS
═══════════════════════════════════════════════════════════════
  Test R²:          {test_metrics['test_r2']:.4f}
  Test RMSE:        {test_metrics['test_rmse']:.2f} kg/day
  Test MAE:         {test_metrics['test_mae']:.2f} kg/day
  Median AE:        {test_metrics['test_median_ae']:.2f} kg/day
  Accuracy (±10%):  {test_metrics['accuracy_within_10pct']:.1f}%
  Accuracy (±20%):  {test_metrics['accuracy_within_20pct']:.1f}%
  Accuracy (±30%):  {test_metrics['accuracy_within_30pct']:.1f}%
  Spearman r:       {test_metrics['spearman_r']:.4f}

═══════════════════════════════════════════════════════════════
  MODEL COMPARISON
═══════════════════════════════════════════════════════════════
"""
    for name, m in model_comparison.items():
        label = name.upper() if name == "ensemble" else name
        report += f"  {label:12s}  R²={m['r2']:.4f}  MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}\n"

    report += """
═══════════════════════════════════════════════════════════════
  TOP 10 FEATURES (by importance)
═══════════════════════════════════════════════════════════════
"""
    for i, feat in enumerate(train_metrics.get("feature_importance", [])[:10], 1):
        bar = "█" * int(feat["importance"] * 100)
        report += f"  {i:2d}. {feat['name']:30s} {feat['importance']:.4f} {bar}\n"

    # Quality assessment
    r2 = test_metrics["test_r2"]
    acc20 = test_metrics["accuracy_within_20pct"]
    if r2 > 0.7 and acc20 > 50:
        grade = "EXCELLENT — Ready for field validation"
    elif r2 > 0.5 and acc20 > 35:
        grade = "GOOD — Promising, continue development"
    elif r2 > 0.3:
        grade = "FAIR — Needs more data or feature engineering"
    else:
        grade = "POOR — Fundamental issues to address"

    report += f"""
═══════════════════════════════════════════════════════════════
  OVERALL ASSESSMENT: {grade}
═══════════════════════════════════════════════════════════════

╚══════════════════════════════════════════════════════════════╝
"""
    return report


# ═══════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    species = sys.argv[1] if len(sys.argv) > 1 else "yellowfin"
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

    results = run_full_training(species=species, n_samples=n_samples)
