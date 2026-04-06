"""
OceanMaster v13.2 — HSI 動態權重引擎 [v13-hsi-weights]
=====================================================
依當前黑潮入侵指數 + 季節（冬季黑潮強/夏季季風強）
自動調整三層 HSI 權重。

三層架構:
  Layer 1: 基礎 SSI (SST, CHL, SSH) — 基本棲地適合度
  Layer 2: 動態物理 (front, FTLE, 黑潮距離) — 聚集結構
  Layer 3: 生態時序 (food chain, DVM, 月相) — 覓食時機

設計原則:
  - 黑潮強入侵 (intrusion_index > 1.5) → Layer 2 權重上升
  - 冬季 (11-3月) → 黑潮主導, Layer 2 增強
  - 夏季 (6-8月) → 季風流/上升流主導, Layer 1:CHL 增強
  - ENSO 修正預留接口
"""

import numpy as np
import logging
from typing import Dict, Any, Optional

log = logging.getLogger("OceanMaster.HSI.Weights")


# ═══════════════════════════════════════════════════
# 預設權重 (baseline)
# ═══════════════════════════════════════════════════

# 三層 HSI 預設權重
BASELINE_LAYER_WEIGHTS = {
    "layer1_basic": 0.45,     # SST + CHL + SSH
    "layer2_physics": 0.30,   # front + FTLE + 黑潮距離
    "layer3_ecology": 0.25,   # food chain + DVM + 月相
}

# Layer 1 內部權重
BASELINE_L1_WEIGHTS = {
    "sst": 0.45,
    "chl": 0.30,
    "ssh": 0.15,
    "do":  0.10,
}

# Layer 2 內部權重
BASELINE_L2_WEIGHTS = {
    "front_strength": 0.30,
    "ftle": 0.25,
    "kuroshio_boost": 0.25,
    "eddy_proximity": 0.20,
}

# Layer 3 內部權重
BASELINE_L3_WEIGHTS = {
    "food_chain_score": 0.40,
    "dvm_alignment": 0.30,
    "moon_effect": 0.15,
    "phi_viability": 0.15,
}


def compute_dynamic_weights(
    month: int,
    kuroshio_intrusion: float = 1.0,
    enso_oni: float = 0.0,
    species: str = "yellowfin",
) -> Dict[str, Dict[str, float]]:
    """
    計算動態 HSI 權重。

    Parameters
    ----------
    month : int
        當前月份 (1-12)
    kuroshio_intrusion : float
        黑潮入侵指數 (KuroshioEngine.intrusion_index)
        < 0.8 = 弱, 0.8-1.2 = 正常, > 1.5 = 強入侵
    enso_oni : float
        ONI 指數 (-3 ~ +3)
        > +0.5 = El Niño, < -0.5 = La Niña
    species : str
        目標物種

    Returns
    -------
    dict with keys: layer_weights, l1, l2, l3, adjustments
    """
    # 複製基線
    lw = dict(BASELINE_LAYER_WEIGHTS)
    l1 = dict(BASELINE_L1_WEIGHTS)
    l2 = dict(BASELINE_L2_WEIGHTS)
    l3 = dict(BASELINE_L3_WEIGHTS)
    adjustments = []

    # ── 季節調整 ──
    if month in (11, 12, 1, 2, 3):
        # 冬季: 黑潮主導 → Layer 2 增強
        lw["layer2_physics"] += 0.08
        lw["layer1_basic"] -= 0.05
        lw["layer3_ecology"] -= 0.03
        l2["kuroshio_boost"] += 0.10
        l2["front_strength"] += 0.05
        l2["eddy_proximity"] -= 0.05
        l2["ftle"] -= 0.10
        adjustments.append("winter_kuroshio_dominant")
        log.debug(f"  冬季校正: L2 +0.08, kuroshio_boost +0.10")

    elif month in (6, 7, 8):
        # 夏季: 季風流/上升流主導 → CHL 更重要
        lw["layer1_basic"] += 0.05
        lw["layer2_physics"] -= 0.03
        lw["layer3_ecology"] -= 0.02
        l1["chl"] += 0.10
        l1["sst"] -= 0.05
        l1["ssh"] -= 0.05
        adjustments.append("summer_monsoon_upwelling")
        log.debug(f"  夏季校正: L1 CHL +0.10")

    # ── 黑潮入侵強度調整 ──
    if kuroshio_intrusion > 1.5:
        # 強入侵: Layer 2 大幅增強
        boost = min((kuroshio_intrusion - 1.5) * 0.15, 0.12)
        lw["layer2_physics"] += boost
        lw["layer1_basic"] -= boost * 0.6
        lw["layer3_ecology"] -= boost * 0.4
        l2["kuroshio_boost"] += 0.08
        l2["front_strength"] += 0.05
        adjustments.append(f"strong_kuroshio_intrusion({kuroshio_intrusion:.2f})")
        log.info(f"  黑潮強入侵 ({kuroshio_intrusion:.2f}): L2 +{boost:.3f}")

    elif kuroshio_intrusion < 0.6:
        # 黑潮極弱: 降低 kuroshio_boost
        l2["kuroshio_boost"] = max(0.05, l2["kuroshio_boost"] - 0.15)
        l2["front_strength"] += 0.10
        l2["ftle"] += 0.05
        adjustments.append(f"weak_kuroshio({kuroshio_intrusion:.2f})")

    # ── ENSO 修正 ──
    if enso_oni > 0.5:
        # El Niño: 溫躍層深化, SST 權重增加, 黑潮減弱
        l1["sst"] += 0.05
        l2["kuroshio_boost"] -= 0.05
        adjustments.append(f"el_nino(ONI={enso_oni:.1f})")
    elif enso_oni < -0.5:
        # La Niña: 黑潮增強, 上升流增強
        l2["kuroshio_boost"] += 0.05
        l1["chl"] += 0.05
        l1["sst"] -= 0.03
        adjustments.append(f"la_nina(ONI={enso_oni:.1f})")

    # ── 物種特化 ──
    if species in ("mahi_mahi", "blue_marlin"):
        # 近海表層魚: SST + front 更重要
        l1["sst"] += 0.05
        l2["front_strength"] += 0.05
        l1["ssh"] -= 0.05
        l2["ftle"] -= 0.05
        adjustments.append(f"species_nearshore_pelagic({species})")

    elif species in ("bigeye",):
        # 深水魚: 溫躍層更重要, 降低表層因子
        l1["sst"] -= 0.05
        l1["ssh"] += 0.05
        adjustments.append("species_deep_pelagic(bigeye)")

    elif species in ("mackerel_scad", "pacific_saury"):
        # 小型洄游魚: CHL (餌料) 更重要
        l1["chl"] += 0.10
        l1["sst"] -= 0.05
        l1["ssh"] -= 0.05
        adjustments.append(f"species_small_pelagic({species})")

    elif "squid" in species:
        # 魷魚: 月相 + DVM 很重要
        l3["moon_effect"] += 0.10
        l3["dvm_alignment"] += 0.05
        l3["food_chain_score"] -= 0.10
        l3["phi_viability"] -= 0.05
        adjustments.append("species_squid")

    # ── 正規化 ──
    lw = _normalize(lw)
    l1 = _normalize(l1)
    l2 = _normalize(l2)
    l3 = _normalize(l3)

    return {
        "layer_weights": lw,
        "l1": l1,
        "l2": l2,
        "l3": l3,
        "adjustments": adjustments,
    }


def apply_weighted_hsi(
    layer1_score: np.ndarray,
    layer2_score: np.ndarray,
    layer3_score: np.ndarray,
    weights: Dict[str, Dict[str, float]],
) -> np.ndarray:
    """
    將三層 HSI 分數用動態權重合併。

    Parameters
    ----------
    layer1_score, layer2_score, layer3_score : np.ndarray
        各層的 HSI 分數 (0-1)
    weights : dict
        compute_dynamic_weights() 的輸出

    Returns
    -------
    np.ndarray : 合併後的 HSI (0-1)
    """
    lw = weights["layer_weights"]
    w1 = lw["layer1_basic"]
    w2 = lw["layer2_physics"]
    w3 = lw["layer3_ecology"]

    # 加權幾何平均 (比算術更嚴格)
    s1 = np.clip(layer1_score, 1e-10, 1.0)
    s2 = np.clip(layer2_score, 1e-10, 1.0)
    s3 = np.clip(layer3_score, 1e-10, 1.0)

    hsi = np.power(s1, w1) * np.power(s2, w2) * np.power(s3, w3)
    return np.clip(hsi, 0.0, 1.0).astype(np.float32)


def _normalize(d: dict) -> dict:
    """正規化 dict values 使其總和 = 1.0"""
    total = sum(d.values())
    if total <= 0:
        return d
    return {k: max(v / total, 0.01) for k, v in d.items()}
