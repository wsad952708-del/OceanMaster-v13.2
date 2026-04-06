"""
OceanMaster v13.2 — 外部資源一鍵設定腳本
==========================================
自動下載/設定八個開源資源中需要外部取得的部分:

1. tuna-prediction (印尼鮪魚數據 CSV)
2. DatLSTM (SST 預報模型, 南京工業大學)
3. vessel-classification (GFW 漁船分類 CNN)
4. GlobalFishingWatch/training-data (漁業行為 ML 訓練集)

使用方式:
  python setup_external_resources.py --all
  python setup_external_resources.py --tuna-data
  python setup_external_resources.py --datlstm
  python setup_external_resources.py --vessel-class
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

log = logging.getLogger("OceanMaster.Setup")

RESOURCES = {
    "tuna-prediction": {
        "url": "https://github.com/stevenalbert/tuna-prediction.git",
        "description": "印尼鮪魚預測 (Naive Bayes + GFW CSV 2012-2016)",
        "target_dir": "external/tuna-prediction",
        "useful_files": ["data/", "*.csv"],
        "priority": "P0",
    },
    "DatLSTM": {
        "url": "https://github.com/Nanjing-Tech-University-CSIC/DatLSTM.git",
        "description": "ConvLSTM + Deformable Attention SST 預報 (MDPI 2024)",
        "target_dir": "external/DatLSTM",
        "useful_files": ["model/", "*.py"],
        "priority": "P0",
    },
    "vessel-classification": {
        "url": "https://github.com/GlobalFishingWatch/vessel-classification.git",
        "description": "TF CNN 漁船分類 (延繩釣/圍網/拖網)",
        "target_dir": "external/vessel-classification",
        "useful_files": ["classification/", "*.py"],
        "priority": "P1",
    },
    "training-data": {
        "url": "https://github.com/GlobalFishingWatch/training-data.git",
        "description": "GFW 漁業行為 ML 訓練集 (Numpy)",
        "target_dir": "external/gfw-training-data",
        "useful_files": ["*.csv", "*.npy"],
        "priority": "P1",
    },
    "TunaForecaster": {
        "url": "https://github.com/jamesadhitthana/TunaForecaster.git",
        "description": "SVM 鮪魚預測 + Dash 前端 (SST+Chl-a)",
        "target_dir": "external/TunaForecaster",
        "useful_files": ["*.csv", "*.py"],
        "priority": "P1",
    },
}


def check_git():
    """Check if git is available."""
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def clone_repo(name: str, info: dict, base_dir: Path):
    """Clone a single repository."""
    target = base_dir / info["target_dir"]

    if target.exists():
        print(f"  ⚠️  {name}: already exists at {target}")
        print(f"      Delete the folder to re-download.")
        return True

    print(f"\n  📥 Cloning {name}...")
    print(f"     {info['description']}")
    print(f"     URL: {info['url']}")
    print(f"     → {target}")

    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", info["url"], str(target)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            # Count files
            n_files = sum(1 for _ in target.rglob("*") if _.is_file())
            size_mb = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1e6
            print(f"     ✅ Done: {n_files} files ({size_mb:.1f} MB)")
            return True
        else:
            print(f"     ❌ Failed: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"     ❌ Timeout (120s)")
        return False
    except Exception as e:
        print(f"     ❌ Error: {e}")
        return False


def extract_tuna_data(base_dir: Path):
    """Extract useful CSV data from tuna-prediction repo."""
    tuna_dir = base_dir / "external" / "tuna-prediction"
    if not tuna_dir.exists():
        return

    csv_files = list(tuna_dir.rglob("*.csv"))
    if csv_files:
        target_data_dir = base_dir / "data" / "external_tuna"
        target_data_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        for f in csv_files:
            dest = target_data_dir / f.name
            if not dest.exists():
                shutil.copy2(f, dest)
        print(f"\n  📋 Extracted {len(csv_files)} CSV files → data/external_tuna/")


def show_status(base_dir: Path):
    """Show status of all external resources."""
    print("\n" + "=" * 60)
    print("  OceanMaster v13.2 — External Resources Status")
    print("=" * 60)

    for name, info in RESOURCES.items():
        target = base_dir / info["target_dir"]
        if target.exists():
            n_files = sum(1 for _ in target.rglob("*") if _.is_file())
            status = f"✅ Installed ({n_files} files)"
        else:
            status = "❌ Not installed"
        print(f"  [{info['priority']}] {name:25s} {status}")
        print(f"       {info['description']}")

    # Check built-in resources
    print("\n  ── Built-in Resources ──")
    builtins = {
        "GFW Data Loader": "engine/gfw_data_loader.py",
        "GFW Heatmap": "engine/gfw_heatmap.py",
        "GFW Training Pipeline": "train_with_gfw.py",
        "AIS Shadow Fishing": "engine/ais_shadow_fishing.py",
        "SST Forecaster": "engine/sst_forecaster.py",
        "GFW Training CSV": "data/gfw_training_set.csv",
        "GFW Cache": "data/gfw_cache",
    }
    for name, path in builtins.items():
        full = base_dir / path
        exists = full.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {name:25s} {path}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="OceanMaster — Download external resources"
    )
    parser.add_argument("--all", action="store_true",
                        help="Download all resources")
    parser.add_argument("--tuna-data", action="store_true",
                        help="Download tuna-prediction data")
    parser.add_argument("--datlstm", action="store_true",
                        help="Download DatLSTM SST forecaster")
    parser.add_argument("--vessel-class", action="store_true",
                        help="Download vessel classification model")
    parser.add_argument("--gfw-training", action="store_true",
                        help="Download GFW training data")
    parser.add_argument("--tunaforecaster", action="store_true",
                        help="Download TunaForecaster (SVM + Dash)")
    parser.add_argument("--status", action="store_true",
                        help="Show current status")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    base_dir = Path(__file__).parent
    print("=" * 60)
    print("  OceanMaster v13.2 — External Resource Setup")
    print(f"  Base: {base_dir}")
    print("=" * 60)

    if args.status:
        show_status(base_dir)
        return

    if not check_git():
        print("❌ Git is not installed or not in PATH.")
        print("   Install: https://git-scm.com/downloads")
        return

    # Determine what to download
    tasks = []
    if args.all or args.tuna_data:
        tasks.append("tuna-prediction")
    if args.all or args.datlstm:
        tasks.append("DatLSTM")
    if args.all or args.vessel_class:
        tasks.append("vessel-classification")
    if args.all or args.gfw_training:
        tasks.append("training-data")
    if args.all or args.tunaforecaster:
        tasks.append("TunaForecaster")

    if not tasks:
        print("\n  No resources selected. Use --all or --status")
        print("  Example: python setup_external_resources.py --all")
        show_status(base_dir)
        return

    print(f"\n  Downloading {len(tasks)} resources...")

    results = {}
    for name in tasks:
        info = RESOURCES[name]
        ok = clone_repo(name, info, base_dir)
        results[name] = ok

    # Extract tuna data if cloned
    if "tuna-prediction" in tasks and results.get("tuna-prediction"):
        extract_tuna_data(base_dir)

    # Summary
    print("\n" + "=" * 60)
    print("  Results:")
    for name, ok in results.items():
        status = "✅ Success" if ok else "❌ Failed"
        print(f"    {name}: {status}")
    print("=" * 60)

    show_status(base_dir)


if __name__ == "__main__":
    main()
