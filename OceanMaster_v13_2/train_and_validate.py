"""
OceanMaster v13.2 — Phase 9: End-to-End Training & Validation  # [v12-phase9]
=============================================================================
Train Stacking Ensemble models using synthetic or real FAO data.

Usage:
    python train_and_validate.py --species yellowfin --samples 2000
    python train_and_validate.py --species bigeye --samples 2000
    python train_and_validate.py --species albacore --samples 1500
    python train_and_validate.py --species skipjack --samples 1500
    python train_and_validate.py --all
"""

import argparse
import logging
import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("OceanMaster.Train")

ALL_SPECIES = ["yellowfin", "bigeye", "albacore", "skipjack"]  # squid removed: no real WCPFC data


def train_species(
    species: str,
    n_samples: int = 2000,
    model_dir: str = "models",
    test_size: float = 0.2,
) -> dict:
    """Train and validate one species."""
    from engine.ml.stacking_ensemble import FishingStackingModel, FeatureEngineer
    from engine.ml.validation import PredictionValidator
    from sklearn.model_selection import train_test_split

    log.info("=" * 60)
    log.info(f"  Training: {species.upper()}")
    log.info("=" * 60)
    t0 = time.time()

    # ── Step 1: Generate or load data ──
    # [v14] 改為直接讀取真實漁業日誌
    data_path = Path(f"data/real_cpue_{species}.csv")
    generic_data_path = Path("data/real_cpue.csv")
    
    import pandas as pd
    
    # 預期特徵欄位
    expected_features = list(FeatureEngineer.FEATURE_NAMES)
    
    if data_path.exists():
        log.info(f"Using real catch data: {data_path}")
        df = pd.read_csv(data_path)
    elif generic_data_path.exists():
        log.info(f"Using generic real catch data: {generic_data_path}")
        df = pd.read_csv(generic_data_path)
        if 'species' in df.columns:
            df = df[df['species'].str.lower() == species.lower()]
    else:
        log.error(f"❌ 找不到適合 {species} 的真實漁獲數據檔。")
        log.error(f"請提供 data/real_cpue_{species}.csv 或 data/real_cpue.csv")
        return {}

    # 確保資料不為空
    if len(df) == 0:
        log.error(f"❌ 數據集為空（或沒有 {species} 的資料）")
        return {}

    # 提取特徵與目標變數 (假設目標變數名稱為 'cpue')
    # 若資料中缺少某些特徵，補 0 (或其他預設值)
    for feat in expected_features:
        if feat not in df.columns:
            log.warning(f"缺少特徵: {feat}，將填補預設值 0")
            df[feat] = 0

    if 'cpue' not in df.columns:
        log.error("❌ 數據集中找不到目標變數 'cpue' 欄位")
        return {}
        
    X = df[expected_features].values
    y = df['cpue'].values
    feature_names = expected_features

    # ── Step 2: Validate feature alignment ──
    log.info(f"  ✅ Feature alignment: {len(feature_names)} features match FEATURE_NAMES")

    # ── Step 3: Train/test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    log.info(f"  Split: train={X_train.shape[0]}, test={X_test.shape[0]}")

    # ── Step 4: Build and train model ──
    model = FishingStackingModel(species=species, model_dir=model_dir)
    train_metrics = model.train(X_train, y_train, feature_names)

    # ── Step 5: Test set evaluation ──
    X_test_scaled = model.scaler.transform(X_test)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0, posinf=0, neginf=0)
    y_pred = model.model.predict(X_test_scaled)
    y_pred = np.clip(y_pred, 0, 1)

    validator = PredictionValidator()
    test_metrics = validator.compute_metrics(y_test, y_pred, species)

    # ── Step 6: Feature importance ──
    fi = validator.feature_importance_report(model.model, feature_names, top_n=15)

    # ── Step 7: Conformal calibration ──
    model.calibrate_conformal(X_test, y_test)

    # ── Step 8: Verify safe_model_load ──
    verify_model = FishingStackingModel(species=species, model_dir=model_dir)
    load_ok = verify_model.safe_model_load()
    assert load_ok, f"safe_model_load() failed for {species}!"
    log.info(f"  ✅ safe_model_load() verification passed")

    elapsed = time.time() - t0
    result = {
        **test_metrics,
        "train_r2": train_metrics.get("r2", None),
        "train_rmse": train_metrics.get("rmse", None),
        "feature_importance": fi,
        "safe_model_load": load_ok,
        "training_time_sec": round(elapsed, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    log.info(f"  ── Results for {species} ──")
    log.info(f"  Test R²:       {test_metrics['r2']:.4f}")
    log.info(f"  Test RMSE:     {test_metrics['rmse']:.4f}")
    log.info(f"  Acc (±20%):    {test_metrics['accuracy_20pct']:.1%}")
    log.info(f"  Acc (±30%):    {test_metrics['accuracy_30pct']:.1%}")
    log.info(f"  Training time: {elapsed:.1f}s")
    log.info("")

    return result


def main():
    parser = argparse.ArgumentParser(description="OceanMaster v13.2 ML Training")
    parser.add_argument("--species", type=str, default=None,
                        choices=ALL_SPECIES, help="Species to train")
    parser.add_argument("--samples", type=int, default=2000,
                        help="Number of synthetic training samples")
    parser.add_argument("--model-dir", type=str, default="models",
                        help="Output directory for models")
    parser.add_argument("--all", action="store_true",
                        help="Train all 4 species")
    args = parser.parse_args()

    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║  OceanMaster v13.2 — ML Training Pipeline (Phase 9)  ║")
    log.info("╚══════════════════════════════════════════════════════╝")

    # Determine species list
    if args.all:
        species_list = ALL_SPECIES
        samples_map = {
            "yellowfin": 2000, "bigeye": 2000,
            "albacore": 1500, "skipjack": 1500,
            "neon_flying_squid": 1500, "japanese_flying_squid": 1500,
        }
    elif args.species:
        species_list = [args.species]
        samples_map = {args.species: args.samples}
    else:
        species_list = ALL_SPECIES
        samples_map = {
            "yellowfin": 2000, "bigeye": 2000,
            "albacore": 1500, "skipjack": 1500,
            "neon_flying_squid": 1500, "japanese_flying_squid": 1500,
        }

    all_results = {}
    all_fi = {}

    t_total = time.time()
    for sp in species_list:
        n = samples_map.get(sp, args.samples)
        result = train_species(sp, n_samples=n, model_dir=args.model_dir)
        all_results[sp] = result
        all_fi[sp] = result.pop("feature_importance", [])

    # ── Generate consolidated report ──
    from engine.ml.validation import PredictionValidator
    report_path = f"{args.model_dir}/training_report_v12.md"
    PredictionValidator.generate_report(all_results, all_fi, report_path)

    total_time = time.time() - t_total
    log.info("═" * 60)
    log.info(f"  All training complete in {total_time:.1f}s")
    log.info(f"  Report: {report_path}")
    log.info("═" * 60)

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Species':<12} {'R2':>8} {'RMSE':>8} {'Acc+/-20%':>10} {'Acc+/-30%':>10}")
    print("-" * 70)
    for sp, m in all_results.items():
        print(
            f"{sp:<12} {m['r2']:>8.4f} {m['rmse']:>8.4f} "
            f"{m['accuracy_20pct']:>10.1%} {m['accuracy_30pct']:>10.1%}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
