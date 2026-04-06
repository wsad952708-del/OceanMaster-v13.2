"""
OceanMaster v13.2 — Safety Filter
===================================
專職攔截: 颱風 (含未來 72h 預測路徑)、波浪危險區、MPA/EEZ 禁漁區。

[v13.5] 新增: 時間維度颱風過濾 — 比對未來 72h 各 ForecastPoint 的暴風圈

一票否決權: 安全與法規 凌駕 ML 任何分數。
"""
import logging
import math
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


def _haversine_km(lat1, lon1, lat2, lon2):
    """大圓距離 (km)"""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# [v13.5] 危險半徑 (km) — 按時間類別
DANGER_RADIUS = {
    "core": 200,        # 暴風核心
    "moderate": 350,    # 中度危險
    "alert": 500,       # 邊緣警戒
}


class SafetyFilter:
    """
    四層過濾鏈: 氣象安全 → 未來颱風路徑 → 法規合規 → EEZ 標註

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
            typhoon_alerts: TyphoonAlert list (含 forecast_track)
        Returns:
            filtered: 通過安全/法規掃描的 hotspots
        """
        n_input = len(hotspots)

        # ── Layer 1: 氣象安全 (颱風當前位置/波浪/風速) ──
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

        # ── Layer 1.5: [v13.5] 未來颱風路徑時間維度過濾 ──
        if typhoon_alerts and isinstance(typhoon_alerts, list):
            n_before_future = len(hotspots)
            hotspots = self._filter_by_forecast_track(hotspots, typhoon_alerts)
            n_future = n_before_future - len(hotspots)
            if n_future > 0:
                log.warning(f"  🌀 Future track filter: {n_future}/{n_before_future} "
                            f"removed (72h forecast path)")

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

    def _filter_by_forecast_track(
        self,
        hotspots: List[Dict],
        typhoon_alerts: List,
    ) -> List[Dict]:
        """
        [v13.5] 時間維度過濾: 比對未來 72h 每個 ForecastPoint 的暴風圈

        規則:
        - 距離 <200km: 剔除 (暴風核心, 即時危險)
        - 距離 200-350km: HSI 降級 50% + 標記 "future_danger"
        - 距離 350-500km: 標記 "future_caution" (僅警示, 不剔除)
        """
        safe = []
        for h in hotspots:
            hlat, hlon = h.get("lat", 0), h.get("lon", 0)
            worst_level = None
            worst_hour = 0
            worst_typhoon = ""

            for alert in typhoon_alerts:
                if not hasattr(alert, 'forecast_track'):
                    continue
                for fp in getattr(alert, 'forecast_track', []):
                    if fp.hour == 0:
                        continue  # 當前位置已由 Layer 1 處理
                    dist = _haversine_km(hlat, hlon, fp.lat, fp.lon)

                    if dist < DANGER_RADIUS["core"]:
                        # 暴風核心 → 直接剔除
                        worst_level = "remove"
                        worst_hour = fp.hour
                        worst_typhoon = getattr(alert, 'name', '')
                        break
                    elif dist < DANGER_RADIUS["moderate"]:
                        if worst_level != "remove":
                            worst_level = "degrade"
                            worst_hour = fp.hour
                            worst_typhoon = getattr(alert, 'name', '')
                    elif dist < DANGER_RADIUS["alert"]:
                        if worst_level not in ("remove", "degrade"):
                            worst_level = "caution"
                            worst_hour = fp.hour
                            worst_typhoon = getattr(alert, 'name', '')

                if worst_level == "remove":
                    break

            if worst_level == "remove":
                log.debug(f"  ❌ Hotspot ({hlat:.1f},{hlon:.1f}) removed: "
                          f"typhoon {worst_typhoon} T+{worst_hour}h <200km")
                continue  # 不加入 safe list

            if worst_level == "degrade":
                # HSI 降級 50%
                h["hsi"] = round(h.get("hsi", 0) * 0.5, 3)
                h["future_danger"] = {
                    "typhoon": worst_typhoon,
                    "hour": worst_hour,
                    "level": "moderate",
                    "message": f"⚠️ 未來 {worst_hour} 小時內，{worst_typhoon} 颱風暴風圈將影響此海域",
                }
                log.debug(f"  ⚠️ Hotspot ({hlat:.1f},{hlon:.1f}) degraded: "
                          f"typhoon {worst_typhoon} T+{worst_hour}h 200-350km")

            elif worst_level == "caution":
                h["future_caution"] = {
                    "typhoon": worst_typhoon,
                    "hour": worst_hour,
                    "level": "alert",
                    "message": f"⚡ {worst_typhoon} 颱風可能於 {worst_hour} 小時後接近，請留意氣象預報",
                }

            safe.append(h)

        return safe

    @staticmethod
    def get_filter_summary(n_input: int, n_output: int) -> Dict:
        """監控面板用: 過濾摘要"""
        return {
            "input_count": n_input,
            "output_count": n_output,
            "removed_count": n_input - n_output,
            "removal_rate": round((n_input - n_output) / max(n_input, 1), 3),
        }
