"""
OceanMaster v10.5 — Safety Filter
===================================
專職攔截: 颱風、波浪危險區、MPA/EEZ 禁漁區。

一票否決權: 安全與法規 凌駕 ML 任何分數。
"""
import logging
import numpy as np
from typing import Dict, List, Optional

log = logging.getLogger("OceanMaster.safety")

try:
    from engine.safety_checker import (
        filter_hotspots_by_safety, SafetyLevel,
    )
    SAFETY_OK = True
except ImportError:
    SAFETY_OK = False

try:
    from compliance.regulation_checker import filter_hotspots as legal_filter
    LEGAL_OK = True
except ImportError:
    LEGAL_OK = False

try:
    from engine.eez.eez_checker import EEZChecker
except ImportError:
    EEZChecker = None


class SafetyFilter:
    """
    三層過濾鏈: 氣象安全 → 法規合規 → EEZ 標註

    每一層都有絕對否決權: AVOID/MPA → 無條件剔除
    """

    def filter(
        self,
        hotspots: List[Dict],
        typhoon_alerts: List = None,
        wave_data: Optional[Dict] = None,
        lats: np.ndarray = None,
        lons: np.ndarray = None,
    ) -> List[Dict]:
        """
        Args:
            hotspots: fuse_and_rank 輸出的原始排名
        Returns:
            filtered: 通過安全/法規掃描的 hotspots
        """
        n_input = len(hotspots)

        # ── Layer 1: 氣象安全 (颱風/波浪/風速) ──
        if SAFETY_OK and typhoon_alerts and isinstance(typhoon_alerts, list):
            wh_grid = wave_data["wave_height"] if wave_data else None
            hotspots = filter_hotspots_by_safety(
                hotspots,
                typhoon_alerts=typhoon_alerts,
                wave_height_grid=wh_grid,
                lats=lats, lons=lons,
            )
            n_safety = n_input - len(hotspots)
            if n_safety > 0:
                log.warning(f"  🌊 Safety filter: {n_safety}/{n_input} removed (typhoon/wave)")

        # ── Layer 2: 法規合規 (MPA/RFMO) ──
        if LEGAL_OK:
            n_before = len(hotspots)
            hotspots = legal_filter(hotspots, remove_illegal=True)
            n_legal = n_before - len(hotspots)
            if n_legal > 0:
                log.warning(f"  ⚖️ Legal filter: {n_legal}/{n_before} in restricted zones")

        # ── Layer 3: EEZ 標註 (不剔除, 加資訊) ──
        if EEZChecker is not None:
            for h in hotspots:
                try:
                    eez_result = EEZChecker.check_point(h["lat"], h["lon"])
                    h["eez"] = eez_result.get("eez_name", "公海")
                    h["eez_high_seas"] = eez_result.get("is_high_seas", True)
                except Exception:
                    h["eez"] = "unknown"
                    h["eez_high_seas"] = True

        n_final = len(hotspots)
        total_removed = n_input - n_final
        if total_removed > 0:
            log.info(f"  🛡️ Total filtered: {n_input} → {n_final} ({total_removed} removed)")

        return hotspots

    @staticmethod
    def get_filter_summary(n_input: int, n_output: int) -> Dict:
        """監控面板用: 過濾摘要"""
        return {
            "input_count": n_input,
            "output_count": n_output,
            "removed_count": n_input - n_output,
            "removal_rate": round((n_input - n_output) / max(n_input, 1), 3),
        }
