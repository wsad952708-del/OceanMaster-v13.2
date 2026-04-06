"""
OceanMaster — Test Suite
==========================
Verifies all ML system components work correctly.

Run:
    python test_suite.py
"""

import sys
import os
import numpy as np
import pandas as pd
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "✅"
FAIL = "❌"
results = []


def test(name, func):
    """Run a test and record result."""
    try:
        func()
        results.append((name, True, ""))
        print(f"  {PASS} {name}")
    except Exception as e:
        results.append((name, False, str(e)))
        print(f"  {FAIL} {name}: {e}")
        traceback.print_exc()


# ═══════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════

def test_data_generator():
    from historical_data_collector import SyntheticCPUEGenerator
    gen = SyntheticCPUEGenerator(species="yellowfin", seed=42)
    df = gen.generate(100)
    assert len(df) == 100, f"Expected 100 rows, got {len(df)}"
    assert "cpue_kg_per_day" in df.columns
    assert "sst" in df.columns
    assert "phi" in df.columns
    assert df["cpue_kg_per_day"].min() >= 0
    assert df["sst"].min() > 10
    assert df["phi"].min() >= 0


def test_multi_species_data():
    from historical_data_collector import FAODataSimulator
    sim = FAODataSimulator(seed=42)
    data = sim.generate_fao_format(
        species_list=["yellowfin", "bigeye"],
        n_per_species=50,
        output_dir="ml_system/data/test"
    )
    assert len(data) == 2
    assert "yellowfin" in data
    assert len(data["yellowfin"]) == 50


def test_feature_pipeline():
    from historical_data_collector import SyntheticCPUEGenerator
    from feature_engineering_pipeline import FeatureEngineeringPipeline

    gen = SyntheticCPUEGenerator(species="yellowfin", seed=42)
    df = gen.generate(100)

    pipeline = FeatureEngineeringPipeline(species="yellowfin")
    X, y, names = pipeline.transform(df, fit=True, include_target=True)

    assert X.shape[0] == 100, f"Expected 100 samples, got {X.shape[0]}"
    assert X.shape[1] > 10, f"Expected >10 features, got {X.shape[1]}"
    assert len(y) == 100
    assert len(names) == X.shape[1]
    assert not np.any(np.isnan(X))
    assert not np.any(np.isinf(X))


def test_feature_pipeline_save_load():
    from historical_data_collector import SyntheticCPUEGenerator
    from feature_engineering_pipeline import FeatureEngineeringPipeline

    gen = SyntheticCPUEGenerator(species="yellowfin", seed=42)
    df = gen.generate(50)

    pipeline = FeatureEngineeringPipeline(species="yellowfin")
    X1, _, names1 = pipeline.transform(df, fit=True, include_target=False)

    pipeline.save("ml_system/models/test_pipeline.pkl")

    pipeline2 = FeatureEngineeringPipeline(species="yellowfin")
    pipeline2.load("ml_system/models/test_pipeline.pkl")
    X2, _, names2 = pipeline2.transform(df, fit=False, include_target=False)

    assert names1 == names2
    np.testing.assert_array_almost_equal(X1, X2, decimal=5)


def test_model_training():
    from historical_data_collector import SyntheticCPUEGenerator
    from feature_engineering_pipeline import FeatureEngineeringPipeline
    from oceanmaster_ml_trainer_v2 import OceanMasterMLTrainer, TrainingConfig

    gen = SyntheticCPUEGenerator(species="yellowfin", seed=42)
    df = gen.generate(200)

    pipeline = FeatureEngineeringPipeline(species="yellowfin")
    X, y, names = pipeline.transform(df, fit=True, include_target=True)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    config = TrainingConfig(species="yellowfin", n_estimators_rf=50, n_estimators_gbr=50)
    trainer = OceanMasterMLTrainer(config)
    metrics = trainer.train(X_train, y_train, names)

    assert metrics["train_r2"] > 0, f"Train R² should be > 0, got {metrics['train_r2']}"
    assert "feature_importance" in metrics

    test_metrics = trainer.evaluate(X_test, y_test)
    assert "test_r2" in test_metrics
    assert "test_mae" in test_metrics


def test_model_save_load():
    from oceanmaster_ml_trainer_v2 import OceanMasterMLTrainer, TrainingConfig
    from historical_data_collector import SyntheticCPUEGenerator
    from feature_engineering_pipeline import FeatureEngineeringPipeline

    gen = SyntheticCPUEGenerator(species="yellowfin", seed=42)
    df = gen.generate(100)
    pipeline = FeatureEngineeringPipeline(species="yellowfin")
    X, y, names = pipeline.transform(df, fit=True, include_target=True)

    config = TrainingConfig(species="yellowfin", n_estimators_rf=30, n_estimators_gbr=30)
    trainer = OceanMasterMLTrainer(config)
    trainer.train(X, y, names)
    trainer.save("ml_system/models")

    # Reload
    trainer2 = OceanMasterMLTrainer(config)
    trainer2.load("ml_system/models")
    assert trainer2.is_trained

    preds = trainer2.model.predict(X[:5])
    assert len(preds) == 5


def test_backtest_walk_forward():
    from historical_data_collector import SyntheticCPUEGenerator
    from feature_engineering_pipeline import FeatureEngineeringPipeline
    from backtest_engine import BacktestEngine

    gen = SyntheticCPUEGenerator(species="yellowfin", seed=42)
    df = gen.generate(500)
    pipeline = FeatureEngineeringPipeline(species="yellowfin")
    engine = BacktestEngine(species="yellowfin")

    results = engine.run_walk_forward(
        df, pipeline, first_test_year=2022, last_test_year=2024
    )
    assert len(results) > 0, "Walk-forward should produce results"
    for r in results:
        assert hasattr(r, "r2")
        assert hasattr(r, "mae")


def test_backtest_seasonal():
    from historical_data_collector import SyntheticCPUEGenerator
    from feature_engineering_pipeline import FeatureEngineeringPipeline
    from backtest_engine import BacktestEngine

    gen = SyntheticCPUEGenerator(species="yellowfin", seed=42)
    df = gen.generate(500)
    pipeline = FeatureEngineeringPipeline(species="yellowfin")
    engine = BacktestEngine(species="yellowfin")

    results = engine.run_seasonal_validation(df, pipeline)
    assert len(results) == 4, f"Expected 4 seasonal folds, got {len(results)}"


def test_integrated_predictor():
    from integrated_predictor import IntegratedPredictor

    pred = IntegratedPredictor(species="yellowfin", fusion_strategy="ml_constrained")

    features = {"sst": 28.0, "chl": 0.15, "ssh": 0.05, "do": 5.5,
                "current_speed": 0.25, "front_strength": 0.03, "eddy_strength": 120}
    result = pred.predict(25.0, 135.0, "2026-06-15", features)

    assert result.cpue_final >= 0
    assert result.phi > 0
    assert 0 <= result.hsi <= 1
    assert 0 <= result.confidence <= 1
    assert result.recommendation in ["low", "medium", "high", "hotspot"]


def test_metabolic_index_correctness():
    """Verify Φ computation matches Deutsch 2015."""
    from historical_data_collector import compute_metabolic_index

    # At Tref=15°C with ~5 mL/L DO → Φ should be moderate
    phi = compute_metabolic_index(15.0, 5.0, "yellowfin")
    assert phi > 0, f"Φ should be > 0 at 15°C"

    # Low DO should give low Φ (oxygen limitation)
    phi_low_do = compute_metabolic_index(28.0, 1.0, "yellowfin")
    phi_high_do = compute_metabolic_index(28.0, 7.0, "yellowfin")
    assert phi_high_do > phi_low_do, "Higher DO should give higher Φ"

    # Φ should be positive at optimal conditions
    phi_optimal = compute_metabolic_index(28.0, 6.0, "yellowfin")
    assert phi_optimal > 1.0, f"Φ should be > 1 at optimal conditions, got {phi_optimal}"


def test_seapodym_hsi():
    from historical_data_collector import compute_seapodym_hsi

    hsi = compute_seapodym_hsi(28.0, 0.3, "yellowfin")
    assert 0 <= hsi <= 1, f"HSI should be [0,1], got {hsi}"

    # Optimal conditions should give high HSI
    hsi_opt = compute_seapodym_hsi(28.0, 0.5, "yellowfin")
    # Far from optimal should give lower HSI
    hsi_cold = compute_seapodym_hsi(10.0, 0.01, "yellowfin")
    assert hsi_opt > hsi_cold


# ═══════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("OceanMaster ML System — Test Suite")
    print("=" * 60)

    tests = [
        ("Data Generator", test_data_generator),
        ("Multi-Species Data", test_multi_species_data),
        ("Feature Pipeline", test_feature_pipeline),
        ("Feature Save/Load", test_feature_pipeline_save_load),
        ("Model Training", test_model_training),
        ("Model Save/Load", test_model_save_load),
        ("Backtest Walk-Forward", test_backtest_walk_forward),
        ("Backtest Seasonal", test_backtest_seasonal),
        ("Integrated Predictor", test_integrated_predictor),
        ("Metabolic Index Φ", test_metabolic_index_correctness),
        ("SEAPODYM HSI", test_seapodym_hsi),
    ]

    print()
    for name, func in tests:
        test(name, func)

    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)

    print(f"\n{'─' * 60}")
    print(f"  Results: {passed}/{len(results)} passed")
    if failed:
        print(f"  {FAIL} {failed} test(s) failed:")
        for name, ok, err in results:
            if not ok:
                print(f"     - {name}: {err}")
    else:
        print(f"  {PASS} All tests passed!")
    print(f"{'─' * 60}")

    sys.exit(0 if failed == 0 else 1)
