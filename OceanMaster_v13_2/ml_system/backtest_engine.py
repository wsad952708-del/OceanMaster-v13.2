"""
OceanMaster — Backtest Engine
===============================
Walk-forward validation and rolling-window backtesting.

Validation Strategies:
  1. Walk-forward: Train on 2015-2019, test on 2020; train on 2015-2020, test on 2021; ...
  2. Rolling window: 24-month training window, 6-month test window, slide forward
  3. Spatial leave-one-out: Train excluding region X, test on region X
  4. Seasonal: Train on all-but-one season, test on held-out season

Each strategy produces accuracy metrics, trend consistency, and
confidence intervals.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import logging

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr

log = logging.getLogger("OceanMaster.Backtest")


class BacktestResult:
    """Container for a single backtest fold's results."""

    def __init__(
        self,
        fold_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        train_size: int,
        test_size: int,
    ):
        self.fold_name = fold_name
        self.y_true = y_true
        self.y_pred = y_pred
        self.train_size = train_size
        self.test_size = test_size

        # Compute metrics
        self.rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        self.mae = float(mean_absolute_error(y_true, y_pred))
        self.r2 = float(r2_score(y_true, y_pred))

        rel_err = np.abs(y_pred - y_true) / np.maximum(y_true, 1.0)
        self.acc_20 = float(np.mean(rel_err < 0.20) * 100)
        self.acc_30 = float(np.mean(rel_err < 0.30) * 100)

        sr, sp = spearmanr(y_true, y_pred)
        self.spearman_r = float(sr) if not np.isnan(sr) else 0.0

    def to_dict(self) -> Dict:
        return {
            "fold": self.fold_name,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "acc_20pct": self.acc_20,
            "acc_30pct": self.acc_30,
            "spearman_r": self.spearman_r,
        }


class BacktestEngine:
    """
    Walk-forward and rolling-window backtesting for OceanMaster ML.

    Usage:
        engine = BacktestEngine(species="yellowfin")
        results = engine.run_walk_forward(df, feature_pipeline)
        engine.print_report(results)
    """

    def __init__(
        self,
        species: str = "yellowfin",
        random_state: int = 42,
    ):
        self.species = species
        self.random_state = random_state

    def _build_model(self):
        """Build a fresh stacking model for each fold."""
        estimators = [
            ("rf", RandomForestRegressor(
                n_estimators=150, max_depth=10, min_samples_leaf=10,
                n_jobs=-1, random_state=self.random_state,
            )),
            ("gbr", GradientBoostingRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.05,
                subsample=0.8, random_state=self.random_state,
            )),
            ("et", ExtraTreesRegressor(
                n_estimators=150, max_depth=10, min_samples_leaf=10,
                n_jobs=-1, random_state=self.random_state,
            )),
            ("ridge", Ridge(alpha=1.0)),
        ]

        return StackingRegressor(
            estimators=estimators,
            final_estimator=RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]),
            cv=KFold(n_splits=3, shuffle=True, random_state=self.random_state),
            n_jobs=-1,
        )

    # ═══════════════════════════════════════════════════
    #  Strategy 1: Walk-Forward Validation
    # ═══════════════════════════════════════════════════

    def run_walk_forward(
        self,
        df: pd.DataFrame,
        feature_pipeline,
        train_years_start: int = 2015,
        first_test_year: int = 2020,
        last_test_year: int = 2024,
    ) -> List[BacktestResult]:
        """
        Walk-forward validation: expanding training window.

        For each test year:
          Train on [train_years_start, test_year-1]
          Test on [test_year]
        """
        print("\n" + "=" * 60)
        print("WALK-FORWARD VALIDATION")
        print("=" * 60)

        results = []
        for test_year in range(first_test_year, last_test_year + 1):
            train_mask = df["year"] < test_year
            test_mask = df["year"] == test_year

            if train_mask.sum() < 50 or test_mask.sum() < 10:
                log.warning(f"  Skipping year {test_year}: insufficient data")
                continue

            df_train = df[train_mask]
            df_test = df[test_mask]

            result = self._train_and_evaluate(
                df_train, df_test, feature_pipeline,
                fold_name=f"WF_{train_years_start}-{test_year-1}_test_{test_year}"
            )
            results.append(result)

            print(f"  {test_year}: "
                  f"R²={result.r2:.3f}, MAE={result.mae:.1f}, "
                  f"Acc±20%={result.acc_20:.0f}% "
                  f"(train={result.train_size}, test={result.test_size})")

        return results

    # ═══════════════════════════════════════════════════
    #  Strategy 2: Rolling Window Validation
    # ═══════════════════════════════════════════════════

    def run_rolling_window(
        self,
        df: pd.DataFrame,
        feature_pipeline,
        train_months: int = 24,
        test_months: int = 6,
        step_months: int = 6,
    ) -> List[BacktestResult]:
        """
        Rolling window: fixed-size training window slides forward.

        train_months: size of training window
        test_months: size of test window
        step_months: how far to slide forward each iteration
        """
        print("\n" + "=" * 60)
        print("ROLLING WINDOW VALIDATION")
        print(f"  Train window: {train_months}mo, Test window: {test_months}mo, "
              f"Step: {step_months}mo")
        print("=" * 60)

        df = df.sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])

        min_date = df["date"].min()
        max_date = df["date"].max()

        results = []
        window_start = min_date

        fold_idx = 0
        while True:
            train_end = window_start + pd.DateOffset(months=train_months)
            test_end = train_end + pd.DateOffset(months=test_months)

            if test_end > max_date:
                break

            train_mask = (df["date"] >= window_start) & (df["date"] < train_end)
            test_mask = (df["date"] >= train_end) & (df["date"] < test_end)

            if train_mask.sum() < 50 or test_mask.sum() < 10:
                window_start += pd.DateOffset(months=step_months)
                continue

            df_train = df[train_mask]
            df_test = df[test_mask]

            fold_name = (
                f"RW_{window_start.strftime('%Y%m')}-"
                f"{train_end.strftime('%Y%m')}_test_"
                f"{test_end.strftime('%Y%m')}"
            )

            result = self._train_and_evaluate(
                df_train, df_test, feature_pipeline, fold_name=fold_name
            )
            results.append(result)

            print(f"  Fold {fold_idx}: "
                  f"R²={result.r2:.3f}, MAE={result.mae:.1f}, "
                  f"Acc±20%={result.acc_20:.0f}%")

            window_start += pd.DateOffset(months=step_months)
            fold_idx += 1

        return results

    # ═══════════════════════════════════════════════════
    #  Strategy 3: Spatial Leave-One-Out
    # ═══════════════════════════════════════════════════

    def run_spatial_validation(
        self,
        df: pd.DataFrame,
        feature_pipeline,
    ) -> List[BacktestResult]:
        """
        Spatial leave-one-out: train excluding a region, test on that region.

        Regions:
          - Kuroshio (20-30°N, 125-145°E)
          - Warm Pool (0-15°N, 140-175°E)
          - Subtropical (25-35°N, 140-175°E)
          - South Pacific (0-10°S, 150-175°E)
        """
        print("\n" + "=" * 60)
        print("SPATIAL LEAVE-ONE-OUT VALIDATION")
        print("=" * 60)

        regions = {
            "Kuroshio": ((20, 30), (125, 145)),
            "Warm_Pool": ((0, 15), (140, 175)),
            "Subtropical": ((25, 35), (140, 175)),
            "Tropical_East": ((0, 15), (120, 140)),
        }

        results = []
        for region_name, ((lat_min, lat_max), (lon_min, lon_max)) in regions.items():
            in_region = (
                (df["lat"] >= lat_min) & (df["lat"] <= lat_max) &
                (df["lon"] >= lon_min) & (df["lon"] <= lon_max)
            )

            if in_region.sum() < 10 or (~in_region).sum() < 50:
                log.warning(f"  Skipping {region_name}: insufficient data")
                continue

            df_train = df[~in_region]
            df_test = df[in_region]

            result = self._train_and_evaluate(
                df_train, df_test, feature_pipeline,
                fold_name=f"Spatial_{region_name}"
            )
            results.append(result)

            print(f"  {region_name:20s}: "
                  f"R²={result.r2:.3f}, MAE={result.mae:.1f}, "
                  f"Acc±20%={result.acc_20:.0f}% (test={result.test_size})")

        return results

    # ═══════════════════════════════════════════════════
    #  Strategy 4: Seasonal Validation
    # ═══════════════════════════════════════════════════

    def run_seasonal_validation(
        self,
        df: pd.DataFrame,
        feature_pipeline,
    ) -> List[BacktestResult]:
        """
        Leave-one-season-out: train on 3 seasons, test on the held-out season.
        """
        print("\n" + "=" * 60)
        print("SEASONAL LEAVE-ONE-OUT VALIDATION")
        print("=" * 60)

        seasons = {
            "Winter": [12, 1, 2],
            "Spring": [3, 4, 5],
            "Summer": [6, 7, 8],
            "Fall": [9, 10, 11],
        }

        results = []
        for season_name, months in seasons.items():
            in_season = df["month"].isin(months)

            if in_season.sum() < 10 or (~in_season).sum() < 50:
                continue

            df_train = df[~in_season]
            df_test = df[in_season]

            result = self._train_and_evaluate(
                df_train, df_test, feature_pipeline,
                fold_name=f"Season_{season_name}"
            )
            results.append(result)

            print(f"  {season_name:10s}: "
                  f"R²={result.r2:.3f}, MAE={result.mae:.1f}, "
                  f"Acc±20%={result.acc_20:.0f}%")

        return results

    # ═══════════════════════════════════════════════════
    #  Internal helpers
    # ═══════════════════════════════════════════════════

    def _train_and_evaluate(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        feature_pipeline,
        fold_name: str,
    ) -> BacktestResult:
        """Train on df_train, evaluate on df_test."""
        # Feature engineering
        X_train, y_train, _ = feature_pipeline.transform(
            df_train, fit=True, include_target=True
        )
        X_test, y_test, _ = feature_pipeline.transform(
            df_test, fit=False, include_target=True
        )

        # Build & train fresh model
        model = self._build_model()
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return BacktestResult(
            fold_name=fold_name,
            y_true=y_test,
            y_pred=y_pred,
            train_size=len(df_train),
            test_size=len(df_test),
        )

    # ═══════════════════════════════════════════════════
    #  Reporting
    # ═══════════════════════════════════════════════════

    @staticmethod
    def summarize(results: List[BacktestResult], strategy_name: str = "") -> Dict:
        """Compute aggregate statistics across all folds."""
        if not results:
            return {"error": "No results"}

        r2s = [r.r2 for r in results]
        maes = [r.mae for r in results]
        accs = [r.acc_20 for r in results]
        spears = [r.spearman_r for r in results]

        summary = {
            "strategy": strategy_name,
            "n_folds": len(results),
            "r2_mean": float(np.mean(r2s)),
            "r2_std": float(np.std(r2s)),
            "r2_min": float(np.min(r2s)),
            "r2_max": float(np.max(r2s)),
            "mae_mean": float(np.mean(maes)),
            "mae_std": float(np.std(maes)),
            "acc20_mean": float(np.mean(accs)),
            "acc20_std": float(np.std(accs)),
            "spearman_mean": float(np.mean(spears)),
            "folds": [r.to_dict() for r in results],
        }
        return summary

    @staticmethod
    def print_summary(summary: Dict):
        """Print formatted summary."""
        print(f"\n{'─' * 60}")
        print(f"Summary: {summary.get('strategy', '')}")
        print(f"{'─' * 60}")
        print(f"  Folds:       {summary['n_folds']}")
        print(f"  R² mean:     {summary['r2_mean']:.4f} ± {summary['r2_std']:.4f}")
        print(f"  R² range:    [{summary['r2_min']:.4f}, {summary['r2_max']:.4f}]")
        print(f"  MAE mean:    {summary['mae_mean']:.2f} ± {summary['mae_std']:.2f} kg/day")
        print(f"  Acc (±20%):  {summary['acc20_mean']:.1f}% ± {summary['acc20_std']:.1f}%")
        print(f"  Spearman:    {summary['spearman_mean']:.4f}")


def run_full_backtest(
    species: str = "yellowfin",
    n_samples: int = 2000,
    report_dir: str = "ml_system/reports",
) -> Dict:
    """
    Run all four backtest strategies and produce a comprehensive report.
    """
    from historical_data_collector import SyntheticCPUEGenerator
    from feature_engineering_pipeline import FeatureEngineeringPipeline

    Path(report_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 70)
    print(f"  COMPLETE BACKTEST SUITE — {species.upper()}")
    print("═" * 70)

    # Generate data
    gen = SyntheticCPUEGenerator(species=species, seed=42)
    df = gen.generate(n_samples)
    pipeline = FeatureEngineeringPipeline(species=species)
    engine = BacktestEngine(species=species)

    all_summaries = {}

    # 1. Walk-forward
    wf_results = engine.run_walk_forward(df, pipeline)
    wf_summary = engine.summarize(wf_results, "Walk-Forward")
    engine.print_summary(wf_summary)
    all_summaries["walk_forward"] = wf_summary

    # 2. Rolling window
    rw_results = engine.run_rolling_window(df, pipeline)
    rw_summary = engine.summarize(rw_results, "Rolling Window (24mo/6mo)")
    engine.print_summary(rw_summary)
    all_summaries["rolling_window"] = rw_summary

    # 3. Spatial
    sp_results = engine.run_spatial_validation(df, pipeline)
    sp_summary = engine.summarize(sp_results, "Spatial Leave-One-Out")
    engine.print_summary(sp_summary)
    all_summaries["spatial"] = sp_summary

    # 4. Seasonal
    se_results = engine.run_seasonal_validation(df, pipeline)
    se_summary = engine.summarize(se_results, "Seasonal Leave-One-Out")
    engine.print_summary(se_summary)
    all_summaries["seasonal"] = se_summary

    # Overall assessment
    all_r2 = []
    for s in all_summaries.values():
        all_r2.append(s["r2_mean"])
    overall_r2 = np.mean(all_r2) if all_r2 else 0

    print("\n" + "═" * 70)
    print("  OVERALL BACKTEST ASSESSMENT")
    print("═" * 70)
    print(f"  Average R² across strategies: {overall_r2:.4f}")

    if overall_r2 > 0.6:
        print("  Grade: STRONG — Model generalizes well across time, space, and season")
    elif overall_r2 > 0.4:
        print("  Grade: MODERATE — Model shows reasonable generalization")
    elif overall_r2 > 0.2:
        print("  Grade: WEAK — Model may be overfitting or data is too noisy")
    else:
        print("  Grade: POOR — Fundamental issues need addressing")

    # Save
    report_path = Path(report_dir) / f"backtest_results_{species}.json"
    with open(report_path, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\n  Results saved: {report_path}")

    return all_summaries


# ═══════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    species = sys.argv[1] if len(sys.argv) > 1 else "yellowfin"
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

    run_full_backtest(species=species, n_samples=n_samples)
