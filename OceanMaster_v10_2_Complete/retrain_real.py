#!/usr/bin/env python3
"""
OceanMaster — 用真實數據重新訓練 ML 模型
==========================================
用法:
  python retrain_real.py --data ml_system/data/simulated_cpue_yellowfin.csv
  python retrain_real.py --data my_data.csv --species yellowfin
  python retrain_real.py --all-species   ← 一次訓練全部魚種
"""

import sys
import os
import time
import json
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_system"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 抑制 LightGBM feature name warnings
warnings.filterwarnings("ignore", message=".*fitted with feature names.*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("OceanMaster.Retrain")


def safe_fmt(val, fmt=".4f", fallback="N/A"):
    """安全格式化數值"""
    try:
        return f"{float(val):{fmt}}"
    except (ValueError, TypeError):
        return fallback


def train_single_species(data_path, species="yellowfin"):
    """訓練單一魚種模型"""
    
    model_dir = "ml_system/models"
    report_dir = "ml_system/reports"
    data_dir = "ml_system/data"
    
    for d in [model_dir, report_dir, data_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    t_start = time.time()
    
    # ═══ Phase 1: 載入數據 ═══
    print(f"\n{'▓' * 70}")
    print(f"  PHASE 1: LOAD DATA — {species.upper()}")
    print(f"{'▓' * 70}")
    
    df = pd.read_csv(data_path)
    print(f"  載入: {data_path} ({len(df)} 條)")
    
    required = ["year", "month", "lat", "lon", "cpue_kg_per_day"]
    env_required = ["sst", "chl", "ssh", "do", "current_speed",
                    "front_strength", "eddy_strength", "phi", "hsi"]
    
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  ❌ 缺少欄位: {missing}")
        return None
    
    missing_env = [c for c in env_required if c not in df.columns]
    if missing_env:
        print(f"  自動補全環境數據...")
        try:
            from prepare_real_data import enrich_with_environment
            df = enrich_with_environment(df, species)
        except ImportError:
            print("  ❌ 需要 prepare_real_data.py")
            return None
    
    if "day_of_year" not in df.columns:
        df["day_of_year"] = df.apply(
            lambda r: datetime(int(r["year"]), int(r["month"]), 15).timetuple().tm_yday, axis=1)
    if "species" not in df.columns:
        df["species"] = species
    
    save_path = Path(data_dir) / f"simulated_cpue_{species}.csv"
    df.to_csv(save_path, index=False)
    
    print(f"  年份: {int(df['year'].min())}-{int(df['year'].max())}  "
          f"CPUE: {df['cpue_kg_per_day'].median():.1f} (中位數)")
    
    # ═══ Phase 2: 特徵工程 ═══
    print(f"\n{'▓' * 70}")
    print(f"  PHASE 2: FEATURE ENGINEERING")
    print(f"{'▓' * 70}")
    
    from feature_engineering_pipeline import FeatureEngineeringPipeline
    
    pipeline = FeatureEngineeringPipeline(species=species)
    X, y, feature_names = pipeline.transform(df, fit=True, include_target=True)
    pipeline.save(f"{model_dir}/feature_pipeline_{species}.pkl")
    print(f"  特徵: {X.shape[1]}, 樣本: {X.shape[0]}")
    
    # ═══ Phase 3: 訓練 ═══
    print(f"\n{'▓' * 70}")
    print(f"  PHASE 3: TRAIN STACKING ENSEMBLE")
    print(f"{'▓' * 70}")
    
    from sklearn.model_selection import train_test_split
    from oceanmaster_ml_trainer_v2 import OceanMasterMLTrainer, TrainingConfig
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print(f"  訓練: {X_train.shape[0]}, 測試: {X_test.shape[0]}")
    
    config = TrainingConfig(species=species)
    trainer = OceanMasterMLTrainer(config)
    train_metrics = trainer.train(X_train, y_train, feature_names)
    
    print(f"\n  Train R²: {safe_fmt(train_metrics.get('train_r2'))}")
    print(f"  CV R²:    {safe_fmt(train_metrics.get('cv_r2'))} ± "
          f"{safe_fmt(train_metrics.get('cv_r2_std'))}")
    
    # ═══ Phase 4: 評估 ═══
    print(f"\n{'▓' * 70}")
    print(f"  PHASE 4: EVALUATE")
    print(f"{'▓' * 70}")
    
    test_metrics = trainer.evaluate(X_test, y_test)
    
    test_r2 = test_metrics.get("test_r2", "N/A")
    test_mae = test_metrics.get("test_mae", "N/A")
    test_rmse = test_metrics.get("test_rmse", "N/A")
    acc_20 = test_metrics.get("accuracy_within_20pct", "N/A")
    spearman = test_metrics.get("spearman_r", "N/A")
    
    print(f"\n  ┌─────────────────────────────────┐")
    print(f"  │  Test R²:     {safe_fmt(test_r2):>10s}       │")
    print(f"  │  RMSE:      {safe_fmt(test_rmse, '.2f'):>8s} kg/day  │")
    print(f"  │  MAE:       {safe_fmt(test_mae, '.2f'):>8s} kg/day  │")
    print(f"  │  Acc (±20%): {safe_fmt(acc_20, '.1f'):>7s}%        │")
    print(f"  │  Spearman r: {safe_fmt(spearman):>10s}       │")
    print(f"  └─────────────────────────────────┘")
    
    # 模型比較
    comparison = trainer.evaluate_individual_models(X_test, y_test)
    
    # 儲存模型
    trainer.save(model_dir)
    
    # ═══ Phase 5: Backtest ═══
    print(f"\n{'▓' * 70}")
    print(f"  PHASE 5: BACKTEST")
    print(f"{'▓' * 70}")
    
    try:
        from backtest_engine import run_full_backtest
        backtest_results = run_full_backtest(
            species=species, n_samples=len(df), report_dir=report_dir)
    except Exception as e:
        print(f"  ⚠ Backtest 跳過: {e}")
        backtest_results = {"skipped": True}
    
    # ═══ Summary ═══
    elapsed = time.time() - t_start
    
    print(f"\n{'=' * 70}")
    print(f"  ✅ {species.upper()} 訓練完成! ({elapsed:.1f}s)")
    print(f"  🎯 Test R²: {safe_fmt(test_r2)}  |  MAE: {safe_fmt(test_mae, '.2f')} kg/day")
    print(f"  📁 模型: {model_dir}/stacking_{species}.pkl")
    print(f"{'=' * 70}")
    
    # JSON 報告
    results = {
        "species": species,
        "data_source": "REAL",
        "data_path": str(data_path),
        "n_records": len(df),
        "training": {k: v for k, v in train_metrics.items() 
                     if not isinstance(v, (list, np.ndarray))},
        "testing": {k: v for k, v in test_metrics.items() 
                    if not isinstance(v, (list, np.ndarray))},
        "comparison": comparison,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
    }
    
    results_path = Path(report_dir) / f"retrain_results_{species}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def main():
    data_path = None
    species = "yellowfin"
    all_species = False
    
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] in ("--data", "-d") and i + 1 < len(argv):
            data_path = argv[i + 1]; i += 2
        elif argv[i] in ("--species", "-s") and i + 1 < len(argv):
            species = argv[i + 1]; i += 2
        elif argv[i] == "--all-species":
            all_species = True; i += 1
        else:
            i += 1
    
    if all_species:
        # ── 全魚種批次訓練 ──
        print("\n" + "█" * 70)
        print("  🐟 OceanMaster — 全魚種 ML 批次訓練")
        print("█" * 70)
        
        species_list = ["yellowfin", "bigeye", "skipjack", "albacore"]
        all_results = {}
        
        for sp in species_list:
            sp_data = f"ml_system/data/simulated_cpue_{sp}.csv"
            if os.path.exists(sp_data):
                result = train_single_species(sp_data, sp)
                if result:
                    all_results[sp] = result
            else:
                print(f"\n  ⚠ 跳過 {sp}: 找不到 {sp_data}")
        
        # 總結表格
        print("\n\n" + "█" * 70)
        print("  📊 全魚種訓練總結")
        print("█" * 70)
        print(f"\n  {'魚種':<12s} {'R²':>8s} {'MAE':>10s} {'Acc±20%':>10s} {'記錄數':>8s}")
        print("  " + "─" * 52)
        
        for sp, res in all_results.items():
            if res:
                t = res.get("testing", {})
                print(f"  {sp:<12s} "
                      f"{safe_fmt(t.get('test_r2')):>8s} "
                      f"{safe_fmt(t.get('test_mae'), '.2f'):>8s}  "
                      f"{safe_fmt(t.get('accuracy_within_20pct'), '.1f'):>8s}% "
                      f"{res.get('n_records', 0):>7d}")
        
        print(f"\n  ✅ 共訓練 {len(all_results)} 個魚種模型")
        print(f"\n  下一步: python main_v10_2.py --run-now")
    
    else:
        if not data_path:
            print("用法:")
            print("  python retrain_real.py --data <csv> --species yellowfin")
            print("  python retrain_real.py --all-species")
            sys.exit(1)
        
        if not os.path.exists(data_path):
            print(f"❌ 找不到: {data_path}"); sys.exit(1)
        
        train_single_species(data_path, species)
        print(f"\n  下一步: python main_v10_2.py --run-now")


if __name__ == "__main__":
    main()
