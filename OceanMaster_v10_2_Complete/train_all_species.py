#!/usr/bin/env python3
"""
OceanMaster — 全魚種一鍵準備 + 訓練
=====================================
把 sample CSV 全部轉換並訓練。

用法:
  python train_all_species.py
"""

import subprocess
import sys
import os
from pathlib import Path

SPECIES_FILES = {
    "yellowfin": "sample_cpue_data.csv",
    "bigeye":    "sample_cpue_bigeye.csv",
    "skipjack":  "sample_cpue_skipjack.csv",
    "albacore":  "sample_cpue_albacore.csv",
}

def main():
    print("█" * 70)
    print("  🐟 OceanMaster — 全魚種一鍵準備 + 訓練")
    print("█" * 70)
    
    # Step 1: 轉換所有 sample data
    print("\n" + "=" * 70)
    print("  STEP 1: 轉換數據")
    print("=" * 70)
    
    for species, filename in SPECIES_FILES.items():
        if not os.path.exists(filename):
            print(f"  ⚠ 找不到 {filename}，跳過 {species}")
            continue
        
        print(f"\n  📦 轉換 {species}...")
        cmd = [
            sys.executable, "prepare_real_data.py",
            "--input", filename,
            "--format", "custom",
            "--species", species,
        ]
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"  ❌ {species} 轉換失敗")
    
    # Step 2: 批次訓練
    print("\n\n" + "=" * 70)
    print("  STEP 2: 批次訓練全魚種")
    print("=" * 70)
    
    cmd = [sys.executable, "retrain_real.py", "--all-species"]
    subprocess.run(cmd)
    
    # Step 3: 確認模型
    print("\n\n" + "=" * 70)
    print("  STEP 3: 確認模型檔案")
    print("=" * 70)
    
    model_dir = Path("ml_system/models")
    for species in SPECIES_FILES:
        model_path = model_dir / f"stacking_{species}.pkl"
        pipeline_path = model_dir / f"feature_pipeline_{species}.pkl"
        
        m_ok = "✅" if model_path.exists() else "❌"
        p_ok = "✅" if pipeline_path.exists() else "❌"
        m_size = f"{model_path.stat().st_size / 1024:.0f} KB" if model_path.exists() else "N/A"
        
        print(f"  {m_ok} stacking_{species}.pkl ({m_size})")
        print(f"  {p_ok} feature_pipeline_{species}.pkl")
    
    print(f"\n  下一步: python main_v10_2.py --run-now")


if __name__ == "__main__":
    main()
