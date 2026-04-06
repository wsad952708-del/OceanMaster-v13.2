#!/usr/bin/env python3
"""
OceanMaster — Master Pipeline Runner
======================================
One-click execution of the complete ML training & backtest pipeline.

Usage:
    python run_pipeline.py                     # Full pipeline (yellowfin)
    python run_pipeline.py --species bigeye    # Different species
    python run_pipeline.py --samples 3000      # More training data
    python run_pipeline.py --backtest-only     # Skip training, run backtest
    python run_pipeline.py --all-species       # Train all species

Output:
    ml_system/data/      — Training data CSVs
    ml_system/models/    — Trained model artifacts (.pkl)
    ml_system/reports/   — Training reports, backtest results
"""

import sys
import os
import time
import logging
import json
from pathlib import Path
from datetime import datetime

# Ensure ml_system is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("OceanMaster.Pipeline")


def parse_args():
    """Simple arg parsing without argparse dependency."""
    args = {
        "species": "yellowfin",
        "samples": 2000,
        "backtest_only": False,
        "all_species": False,
        "skip_backtest": False,
    }
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == "--species" and i + 1 < len(argv):
            args["species"] = argv[i + 1]
            i += 2
        elif argv[i] == "--samples" and i + 1 < len(argv):
            args["samples"] = int(argv[i + 1])
            i += 2
        elif argv[i] == "--backtest-only":
            args["backtest_only"] = True
            i += 1
        elif argv[i] == "--all-species":
            args["all_species"] = True
            i += 1
        elif argv[i] == "--skip-backtest":
            args["skip_backtest"] = True
            i += 1
        else:
            i += 1
    return args


def run_single_species(species: str, n_samples: int, skip_backtest: bool = False):
    """Run complete pipeline for one species."""
    data_dir = "ml_system/data"
    model_dir = "ml_system/models"
    report_dir = "ml_system/reports"

    for d in [data_dir, model_dir, report_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # ─── Phase 1: Training ──────────────────────
    print("\n" + "▓" * 70)
    print(f"  PHASE 1: ML TRAINING — {species.upper()}")
    print("▓" * 70)

    from oceanmaster_ml_trainer_v2 import run_full_training
    train_results = run_full_training(
        species=species,
        n_samples=n_samples,
        data_dir=data_dir,
        model_dir=model_dir,
        report_dir=report_dir,
    )

    # ─── Phase 2: Backtesting ──────────────────
    if not skip_backtest:
        print("\n" + "▓" * 70)
        print(f"  PHASE 2: BACKTESTING — {species.upper()}")
        print("▓" * 70)

        from backtest_engine import run_full_backtest
        backtest_results = run_full_backtest(
            species=species,
            n_samples=n_samples,
            report_dir=report_dir,
        )
    else:
        backtest_results = {"skipped": True}

    # ─── Phase 3: Integration Test ─────────────
    print("\n" + "▓" * 70)
    print(f"  PHASE 3: INTEGRATION TEST — {species.upper()}")
    print("▓" * 70)

    from integrated_predictor import IntegratedPredictor
    from historical_data_collector import OceanEnvironmentSimulator

    predictor = IntegratedPredictor(species=species, fusion_strategy="ml_constrained")
    predictor.load_ml_model(model_dir)

    env = OceanEnvironmentSimulator(seed=99)
    test_points = [
        (25.0, 135.0, "Kuroshio"),
        (10.0, 150.0, "Warm Pool"),
        (5.0, 165.0, "Equatorial"),
        (30.0, 170.0, "Subtropical"),
    ]

    print(f"\n  {'Location':25s} {'CPUE':>8s} {'Φ':>6s} {'HSI':>6s} "
          f"{'Conf':>6s} {'Recommend':>10s}")
    print("  " + "─" * 65)
    for lat, lon, label in test_points:
        features = {
            "sst": env.simulate_sst(lat, lon, 6),
            "chl": env.simulate_chlorophyll(lat, lon, 6),
            "ssh": env.simulate_ssh(lat, lon, 6),
            "do": env.simulate_do(lat, lon, 6, env.simulate_sst(lat, lon, 6)),
            "current_speed": env.simulate_current_speed(lat, lon),
            "front_strength": env.simulate_front_strength(lat, lon),
            "eddy_strength": env.simulate_eddy_strength(lat, lon),
        }
        result = predictor.predict(lat, lon, "2026-06-15", features)
        print(f"  {label:25s} {result.cpue_final:8.1f} {result.phi:6.2f} "
              f"{result.hsi:6.3f} {result.confidence:6.2f} "
              f"{result.recommendation:>10s}")

    elapsed = time.time() - t_start

    # ─── Summary ────────────────────────────────
    print("\n" + "═" * 70)
    print(f"  PIPELINE COMPLETE — {species.upper()}")
    print("═" * 70)
    print(f"  Time:          {elapsed:.1f}s")
    print(f"  Training R²:   {train_results['training']['train_r2']:.4f}")
    print(f"  CV R²:         {train_results['training']['cv_r2']:.4f}")
    print(f"  Test R²:       {train_results['testing']['test_r2']:.4f}")
    print(f"  Test MAE:      {train_results['testing']['test_mae']:.2f} kg/day")
    print(f"  Acc (±20%):    {train_results['testing']['accuracy_within_20pct']:.1f}%")

    if not skip_backtest and "walk_forward" in backtest_results:
        wf = backtest_results["walk_forward"]
        print(f"  Backtest R²:   {wf['r2_mean']:.4f} ± {wf['r2_std']:.4f}")

    print(f"\n  Artifacts:")
    print(f"    Models:  {model_dir}/stacking_{species}.pkl")
    print(f"    Data:    {data_dir}/synthetic_cpue_{species}.csv")
    print(f"    Reports: {report_dir}/training_report_{species}.txt")

    return {
        "species": species,
        "elapsed_seconds": elapsed,
        "training": train_results["training"],
        "testing": train_results["testing"],
        "backtest": backtest_results,
    }


def main():
    args = parse_args()

    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█  OceanMaster ML Training & Backtest System  v2.0" + " " * 19 + "█")
    print("█  Science + ML Fusion Pipeline" + " " * 38 + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    print(f"\n  Date:     {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Samples:  {args['samples']}")

    all_results = {}

    if args["all_species"]:
        species_list = ["yellowfin", "bigeye", "albacore"]
        print(f"  Species:  {', '.join(species_list)}")
        for sp in species_list:
            results = run_single_species(sp, args["samples"], args.get("skip_backtest", False))
            all_results[sp] = results
    else:
        sp = args["species"]
        print(f"  Species:  {sp}")
        if args["backtest_only"]:
            from backtest_engine import run_full_backtest
            all_results[sp] = run_full_backtest(species=sp, n_samples=args["samples"])
        else:
            results = run_single_species(sp, args["samples"], args.get("skip_backtest", False))
            all_results[sp] = results

    # Save master results
    master_path = Path("ml_system/reports/pipeline_results.json")
    with open(master_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n\n{'█' * 70}")
    print(f"  ALL DONE. Results: {master_path}")
    print(f"{'█' * 70}\n")


if __name__ == "__main__":
    main()
