"""
OceanMaster v13.2 — Incremental Data Collection
================================================
每次跑完後記錄 hotspot + 環境資料，持續蓄積供未來有監督重訓練使用。

[v15.3-audit] 自動微調已停用，僅資料蒐集。

原設計:
  - XGBoost 支持 `xgb_model` 參數（繼續訓練）
  - 累積 30 次 run 後自動微調
  - 已因缺乏驗證門檻而停用（見 maybe_finetune docstring）

Workflow:
  1. 每次 pipeline 完成 → append hotspot 到 CSV log
  2. 累積 ≥ 30 runs → 自動觸發微調
  3. 用 RandomForest / XGBoost 的 warm_start / continue_training
  4. 保存更新的模型權重
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger("OceanMaster.IncrementalLearner")

LOG_DIR = Path("data/run_logs")
LOG_FILE = LOG_DIR / "hotspot_history.csv"
FINETUNE_TRIGGER = 30  # 累積 N 次後觸發微調


def log_run_results(
    hotspots: List[Dict],
    run_metadata: Optional[Dict] = None,
) -> int:
    """
    記錄本次 pipeline 結果到 CSV。

    每個 hotspot 記錄:
    - timestamp, run_id
    - lat, lon, species, hsi, cpue_index
    - sst, chl, depth_m, mld_m, z20_m
    - gfw_density, hook_depth_optimal
    - 用於後續微調的 feature vector

    Returns:
        累積 run 次數
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 確定是否需要寫 header
    write_header = not LOG_FILE.exists()

    columns = [
        "timestamp", "run_id", "rank",
        "lat", "lon", "species", "hsi_pct", "phi",
        "sst", "chl", "npp", "do",
        "depth_m", "mld_m", "z20_m",
        "front_pct", "eddy_pct", "convergence_pct",
        "cpue_index", "cpue_ci_low", "cpue_ci_high",
        "gfw_density", "hook_depth_optimal",
        "distance_nm", "eez",
    ]

    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            if write_header:
                f.write(",".join(columns) + "\n")

            for h in hotspots:
                values = [
                    ts, run_id, str(h.get("rank", 0)),
                    f"{h.get('lat', 0):.4f}", f"{h.get('lon', 0):.4f}",
                    h.get("species", ""), str(int(round(h.get("score", 0) * 100))),
                    f"{h.get('phi', 0):.2f}",
                    f"{h.get('sst', 0):.1f}", f"{h.get('chl', 0):.4f}",
                    f"{h.get('npp', 0):.0f}", f"{h.get('do', 0):.1f}",
                    f"{h.get('depth_m', 0):.0f}",
                    f"{h.get('mld_m', 50):.0f}", f"{h.get('z20_m', 150):.0f}",
                    str(h.get("front_persistence", 0)),
                    str(h.get("eddy_edge", 0)),
                    str(h.get("convergence", 0)),
                    str(h.get("cpue_index", 0)),
                    str(h.get("cpue_ci_low", 0)),
                    str(h.get("cpue_ci_high", 0)),
                    f"{h.get('gfw_density', 0):.3f}",
                    str(h.get("hook_depth_optimal", 0)),
                    f"{h.get('distance_nm', 0):.0f}",
                    h.get("eez", ""),
                ]
                f.write(",".join(values) + "\n")

        # 統計累積 run 次數
        n_runs = _count_unique_runs()
        log.info(f"  📝 Run Log: {len(hotspots)} hotspots saved, "
                 f"cumulative runs={n_runs}")
        return n_runs

    except Exception as e:
        log.warning(f"  Run Log: {e}")
        return 0


def _count_unique_runs() -> int:
    """計算歷史 run 次數。"""
    if not LOG_FILE.exists():
        return 0
    try:
        import csv
        run_ids = set()
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                run_ids.add(row.get("run_id", ""))
        return len(run_ids)
    except Exception:
        return 0


def maybe_finetune(
    model_dir: str = "models",
    species_list: Optional[List[str]] = None,
) -> bool:
    """
    檢查累積資料狀態並記錄統計（僅資料蒐集，不自動微調）。

    [v15.3-audit] 安全設計決策:
      原版會在累積 30 runs 後自動微調並覆寫模型，但存在以下風險：
      1. 無 holdout 驗證門檻 — 無法確認新模型優於舊模型
      2. 劣質回報資料（壞天氣/設備故障）會直接污染模型
      3. 無人工確認步驟 — 船上自動執行，岸上無法干預
      因此改為：僅持續蒐集資料，供未來有監督的重新訓練使用。
      自動模型替換將在建立完整驗證管線後重新啟用。

    Returns:
        False (目前不執行微調)
    """
    n_runs = _count_unique_runs()
    log.info(f"  📝 Incremental data: {n_runs} runs accumulated "
             f"(auto-finetune disabled, data collection only)")

    if n_runs >= FINETUNE_TRIGGER:
        log.info(f"  ℹ️ {n_runs} runs available for supervised retraining "
                 f"(manual trigger required)")

    # Archive old log (keep last 100 runs) to prevent unbounded growth
    if n_runs > 100:
        _trim_log_file(keep_runs=100)

    return False


def _trim_log_file(keep_runs: int = 100):
    """保留最近 N 次 run 的記錄。"""
    try:
        import pandas as pd
        df = pd.read_csv(LOG_FILE)
        run_ids = df["run_id"].unique()
        if len(run_ids) > keep_runs:
            keep = run_ids[-keep_runs:]
            df = df[df["run_id"].isin(keep)]
            df.to_csv(LOG_FILE, index=False)
            log.info(f"  Log trimmed: kept {keep_runs} recent runs")
    except Exception as e:
        log.debug(f"[降級] engine/incremental_learner.py: {e}")
