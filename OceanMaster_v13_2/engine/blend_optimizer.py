"""
OceanMaster v13.2 — Blend Weight Optimizer
=========================================
[v17] 競品整合: 替換硬編碼混合權重 (15%/35%/50%)

核心概念 (逆向自 GreenFish):
  GreenFish 的核心優勢是用幾十年漁獲日誌讓 ML 自己學最優權重。
  這裡用 WCPFC 歷史 CPUE 數據做 held-out validation 找最優 w_v8, w_comm, w_gf。

使用方法:
  weights = load_optimal_weights("yellowfin")
  blended = weights["w_v8"]*v8 + weights["w_comm"]*comm + weights["w_gf"]*gf

如果無預計算權重，fallback 到原始 15/35/50。
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

log = logging.getLogger("OceanMaster.BlendOptimizer")

# 預設權重 (fallback)
_DEFAULT_HSI_WEIGHTS = {
    "yellowfin": {"w_v8": 0.15, "w_comm": 0.35, "w_gf": 0.50},
    "bigeye":    {"w_v8": 0.15, "w_comm": 0.35, "w_gf": 0.50},
    "skipjack":  {"w_v8": 0.15, "w_comm": 0.35, "w_gf": 0.50},
    "albacore":  {"w_v8": 0.15, "w_comm": 0.35, "w_gf": 0.50},
}

# ML fusion 預設權重
# [v15.3-audit] 調整為 80:20 (science:ML)
# 原因: 所有 WCPFC 模型的 spatial CV R² 為負 (yellowfin:-0.21, bigeye:-0.41)
# 代表 ML 無法泛化到未見過的海域。物理 HSI 應主導，直到取得真實漁獲回饋重新校準。
_DEFAULT_ML_WEIGHTS = {
    "yellowfin": {"w_science": 0.80, "w_ml": 0.20},
    "bigeye":    {"w_science": 0.80, "w_ml": 0.20},
    "skipjack":  {"w_science": 0.80, "w_ml": 0.20},
    "albacore":  {"w_science": 0.80, "w_ml": 0.20},
}

# 優化後權重快取檔
_WEIGHTS_FILE = Path(__file__).parent.parent / "data" / "optimized_weights.json"


def load_optimal_weights(
    species: str,
    weight_type: str = "hsi",
) -> Dict[str, float]:
    """
    載入最優混合權重。

    Args:
        species: 魚種名稱
        weight_type: "hsi" (v8/comm/gf blend) 或 "ml" (science/ml blend)

    Returns:
        權重 dict, e.g. {"w_v8": 0.12, "w_comm": 0.38, "w_gf": 0.50, "source": "optimized"}
    """
    defaults = (_DEFAULT_HSI_WEIGHTS if weight_type == "hsi"
                else _DEFAULT_ML_WEIGHTS)

    # 嘗試載入預計算權重
    if _WEIGHTS_FILE.exists():
        try:
            with open(_WEIGHTS_FILE, "r", encoding="utf-8") as f:
                all_weights = json.load(f)
            key = f"{weight_type}_{species}"
            if key in all_weights:
                w = all_weights[key]
                w["source"] = "optimized"
                log.info(f"  ⚙️ {species} {weight_type} weights: {w} (optimized)")
                return w
        except Exception as e:
            log.warning(f"  ⚠️ Failed to load optimized weights: {e}")

    # Fallback 到預設
    w = defaults.get(species, defaults.get("yellowfin", {})).copy()
    w["source"] = "default"
    return w


def optimize_blend_weights(
    v8_scores: np.ndarray,
    comm_scores: np.ndarray,
    gf_scores: np.ndarray,
    target_cpue: np.ndarray,
    species: str,
) -> Dict[str, float]:
    """
    用 WCPFC CPUE 數據找最優 HSI 混合權重。

    Uses scipy.optimize.minimize with constraint: w_v8 + w_comm + w_gf = 1

    Args:
        v8_scores: v8 HSI 分數 (flatten)
        comm_scores: commercial HSI 分數 (flatten)
        gf_scores: GreenFish HSI 分數 (flatten)
        target_cpue: 真實 CPUE (normalize 到 0-1)
        species: 魚種

    Returns:
        {"w_v8": float, "w_comm": float, "w_gf": float, "r2": float}
    """
    try:
        from scipy.optimize import minimize

        # 移除 NaN
        mask = (np.isfinite(v8_scores) & np.isfinite(comm_scores)
                & np.isfinite(gf_scores) & np.isfinite(target_cpue))
        v8 = v8_scores[mask]
        comm = comm_scores[mask]
        gf = gf_scores[mask]
        cpue = target_cpue[mask]

        if len(cpue) < 50:
            log.warning(f"  ⚠️ Not enough data for optimization ({len(cpue)} samples)")
            return _DEFAULT_HSI_WEIGHTS.get(species, {}).copy()

        def loss(w):
            w_norm = w / w.sum()  # enforce sum=1
            blended = w_norm[0] * v8 + w_norm[1] * comm + w_norm[2] * gf
            return np.mean((blended - cpue) ** 2)  # MSE

        # 初始猜測 = 均等
        x0 = np.array([0.33, 0.34, 0.33])
        bounds = [(0.05, 0.80)] * 3  # 每個權重至少 5%
        constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}

        result = minimize(loss, x0, bounds=bounds, constraints=constraints,
                         method="SLSQP", options={"maxiter": 200})

        if result.success:
            w_opt = result.x / result.x.sum()
            # 計算 R²
            blended_opt = w_opt[0] * v8 + w_opt[1] * comm + w_opt[2] * gf
            ss_res = np.sum((cpue - blended_opt) ** 2)
            ss_tot = np.sum((cpue - np.mean(cpue)) ** 2)
            r2 = 1 - ss_res / max(ss_tot, 1e-10)

            weights = {
                "w_v8": round(float(w_opt[0]), 4),
                "w_comm": round(float(w_opt[1]), 4),
                "w_gf": round(float(w_opt[2]), 4),
                "r2": round(float(r2), 4),
                "n_samples": int(mask.sum()),
            }
            log.info(f"  ✅ {species} optimal weights: {weights}")
            return weights
        else:
            log.warning(f"  ⚠️ Optimization failed: {result.message}")
            return _DEFAULT_HSI_WEIGHTS.get(species, {}).copy()

    except ImportError:
        log.warning("  ⚠️ scipy not available, using default weights")
        return _DEFAULT_HSI_WEIGHTS.get(species, {}).copy()


def save_optimized_weights(all_weights: Dict[str, Dict]) -> None:
    """存儲優化後的權重到 JSON 快取。"""
    _WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_WEIGHTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_weights, f, indent=2, ensure_ascii=False)
    log.info(f"  💾 Saved optimized weights to {_WEIGHTS_FILE}")
