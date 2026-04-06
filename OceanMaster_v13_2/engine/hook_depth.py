"""
OceanMaster v13.2 — Hook Depth Recommendation Engine
====================================================
模仿 SafetyNet Technologies 的原位環境感測器概念。

用衛星數據 + DVM 模型 + 溫躍層深度 → 建議最適下鉤深度 & 時段。
替代 SafetyNet 的 Enki 感測器 (溫/鹽/深度/濁度)。

數據來源: CMEMS 溫度剖面 + Argo 浮標氣候態
學術依據: Schaefer & Fuller 2010; Musyl et al. 2003; Bigelow et al. 2006
"""

import logging
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger("OceanMaster.HookDepth")

# DVM 行為參數 (from species_params.py)
_DVM_DEPTH_RANGES = {
    "yellowfin":            {"day": (50, 250),  "night": (0, 50),   "feed": (20, 150)},
    "bigeye":               {"day": (200, 600), "night": (0, 100),  "feed": (150, 400)},
    "skipjack":             {"day": (20, 200),  "night": (0, 30),   "feed": (10, 100)},
    "albacore":             {"day": (100, 300), "night": (0, 80),   "feed": (50, 250)},
    "japanese_flying_squid":{"day": (100, 400), "night": (0, 30),   "feed": (0, 30)},
    "pacific_saury":        {"day": (20, 100),  "night": (0, 15),   "feed": (0, 30)},
    "mahi_mahi":            {"day": (0, 40),    "night": (0, 20),   "feed": (0, 30)},
    "blue_marlin":          {"day": (50, 300),  "night": (0, 50),   "feed": (30, 200)},
    "mackerel_scad":        {"day": (30, 150),  "night": (0, 30),   "feed": (10, 80)},
}

# 最佳作業時段 (UTC+8 台灣時間)
_BEST_FISHING_TIMES = {
    "yellowfin":            {"best": "05:00-08:00", "alt": "16:00-19:00", "note": "晨昏覓食最活躍"},
    "bigeye":               {"best": "04:00-07:00", "alt": "17:00-20:00", "note": "深潛前/後回表層時段"},
    "skipjack":             {"best": "06:00-10:00", "alt": "15:00-18:00", "note": "日間表層活動"},
    "albacore":             {"best": "05:00-09:00", "alt": "16:00-19:00", "note": "晨間溫躍層覓食"},
    "japanese_flying_squid":{"best": "19:00-23:00", "alt": "02:00-05:00", "note": "夜間上浮表層 (集魚燈)"},
    "pacific_saury":        {"best": "19:00-02:00", "alt": "03:00-05:00", "note": "夜間棒受網作業"},
    "mahi_mahi":            {"best": "06:00-10:00", "alt": "15:00-18:00", "note": "日間表層拖釣"},
    "blue_marlin":          {"best": "06:00-10:00", "alt": "14:00-17:00", "note": "日間延繩釣"},
    "mackerel_scad":        {"best": "05:00-08:00", "alt": "16:00-19:00", "note": "晨昏圍網"},
}


def compute_hook_depth(
    species: str,
    sst: float,
    mld: float = 50.0,
    z20: float = 150.0,
    hour_utc: int = 6,
    month: int = 2,
    lat: float = 20.0,
    lunar_illumination: float = 0.5,
) -> Dict:
    """
    計算建議下鉤深度 + 最佳作業時段。

    Uses DVM behavior + thermocline + SST to recommend:
    - hook_depth_range: (min_m, max_m)
    - best_time: "HH:MM-HH:MM (UTC+8)"
    - confidence: High/Medium/Low

    Args:
        species: 魚種 ID
        sst: 海面溫度 °C
        mld: 混合層深度 m
        z20: 20°C 等溫線深度 m
        hour_utc: 當前 UTC 小時
        month: 月份
        lat: 緯度

    Returns:
        {
            "hook_depth_min": 120,
            "hook_depth_max": 250,
            "hook_depth_optimal": 180,
            "best_time": "05:00-08:00 (UTC+8)",
            "alt_time": "16:00-19:00 (UTC+8)",
            "time_note": "晨昏覓食最活躍",
            "method": "延繩釣",
            "confidence": "High",
        }
    """
    dvm = _DVM_DEPTH_RANGES.get(species, {"day": (50, 200), "night": (0, 50), "feed": (20, 150)})
    times = _BEST_FISHING_TIMES.get(species, {"best": "06:00-09:00", "alt": "16:00-19:00", "note": "一般"})

    # 判斷日夜 (簡化: UTC+8, 06:00-18:00 = 日間)
    hour_local = (hour_utc + 8) % 24
    is_daytime = 6 <= hour_local < 18

    if is_daytime:
        base_min, base_max = dvm["day"]
    else:
        base_min, base_max = dvm["night"]

    feed_min, feed_max = dvm["feed"]

    # 溫躍層修正: 魚通常在 MLD 附近活動
    # bigeye 在溫躍層以下覓食, 其他在溫躍層上方
    if species == "bigeye":
        # 大目鮪: 白天駐留溫躍層以下 250-400m (Schaefer & Fuller 2010)
        depth_min = max(mld * 0.8, 150)
        depth_max = min(z20 * 2.5, 500)
        depth_optimal = max(z20 * 1.5, 250)
    elif species in ("skipjack", "mahi_mahi", "pacific_saury"):
        # 表層魚種: hook depth = MLD 上方
        depth_min = max(0, base_min)
        depth_max = min(base_max, mld * 0.8)
        depth_optimal = mld * 0.4
    else:
        # 一般鮪魚: hook depth ≈ MLD ± 偏移
        depth_min = max(base_min, mld * 0.5)
        depth_max = min(base_max, mld * 2.5)
        depth_optimal = (feed_min + feed_max) / 2

    # SST 修正: 水溫越高 → 溫躍層越深 → hooks 要放更深
    sst_shift = (sst - 25.0) * 3.0  # +1°C SST → +3m depth
    depth_optimal += sst_shift
    depth_min += sst_shift * 0.5
    depth_max += sst_shift * 0.5

    # 季節修正: 冬季溫躍層較淺
    if month in (11, 12, 1, 2, 3):
        depth_optimal *= 0.85
        depth_min *= 0.85
        depth_max *= 0.85

    # [v18] 月相修正 (Tun-AI + Benoit-Bird 2009 MEPS 395:27-30)
    # 滿月時 micronekton 不完全上浮表層 → 夜間魚群留在較深處
    # lunar_illumination: 0=新月, 1=滿月
    if not is_daytime and lunar_illumination > 0.3:
        lunar_depth_factor = 1.0 + 0.4 * lunar_illumination  # 滿月: ×1.4
        depth_optimal *= lunar_depth_factor
        depth_min *= (1.0 + 0.2 * lunar_illumination)
        depth_max *= (1.0 + 0.3 * lunar_illumination)
        log.debug(f"  🌕 Lunar correction: illum={lunar_illumination:.2f}, "
                  f"depth ×{lunar_depth_factor:.2f}")

    # 確保合理範圍 (延繩釣最淺 30m，更淺是拖釣)
    depth_min = max(30, round(depth_min))
    depth_max = max(depth_min + 10, round(depth_max))
    depth_optimal = max(depth_min, min(depth_max, round(depth_optimal)))

    # 漁法推薦
    gear_map = {
        "yellowfin": "延繩釣", "bigeye": "深層延繩釣",
        "skipjack": "圍網/竿釣", "albacore": "延繩釣",
        "japanese_flying_squid": "魷釣/集魚燈", "pacific_saury": "棒受網/集魚燈",
        "mahi_mahi": "拖釣/曳繩", "blue_marlin": "延繩釣/拖釣",
        "mackerel_scad": "圍網/中層拖網",
    }

    # 信心度
    if abs(z20 - 150) < 50 and abs(mld - 50) < 30:
        confidence = "High"
    elif z20 > 0 and mld > 0:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "hook_depth_min": int(depth_min),
        "hook_depth_max": int(depth_max),
        "hook_depth_optimal": int(depth_optimal),
        "best_time": f"{times['best']} (UTC+8)",
        "alt_time": f"{times['alt']} (UTC+8)",
        "time_note": times["note"],
        "method": gear_map.get(species, "延繩釣"),
        "confidence": confidence,
    }


def enrich_hotspots_with_hook_depth(
    hotspots: List[Dict],
    mld_grid: Optional[np.ndarray] = None,
    z20_grid: Optional[np.ndarray] = None,
    lats: Optional[np.ndarray] = None,
    lons: Optional[np.ndarray] = None,
    month: int = 2,
) -> List[Dict]:
    """批量為 hotspot 加入下鉤深度建議。"""
    for h in hotspots:
        sp = h.get("species", "yellowfin")
        sst = h.get("sst", 25.0)

        # 從 grid 取 MLD/Z20
        mld_val = 50.0
        z20_val = 150.0
        if mld_grid is not None and lats is not None and lons is not None:
            li = int(np.argmin(np.abs(lats - h.get("lat", 0))))
            lj = int(np.argmin(np.abs(lons - h.get("lon", 0))))
            if 0 <= li < mld_grid.shape[0] and 0 <= lj < mld_grid.shape[1]:
                v = float(mld_grid[li, lj])
                if np.isfinite(v) and v > 0:
                    mld_val = v
        if z20_grid is not None and lats is not None and lons is not None:
            li = int(np.argmin(np.abs(lats - h.get("lat", 0))))
            lj = int(np.argmin(np.abs(lons - h.get("lon", 0))))
            if 0 <= li < z20_grid.shape[0] and 0 <= lj < z20_grid.shape[1]:
                v = float(z20_grid[li, lj])
                if np.isfinite(v) and v > 0:
                    z20_val = v

        result = compute_hook_depth(
            species=sp, sst=sst, mld=mld_val, z20=z20_val,
            month=month, lat=h.get("lat", 20),
            lunar_illumination=h.get("lunar_illumination", 0.5),
        )

        h["hook_depth_min"] = result["hook_depth_min"]
        h["hook_depth_max"] = result["hook_depth_max"]
        h["hook_depth_optimal"] = result["hook_depth_optimal"]
        h["hook_depth_label"] = f"{result['hook_depth_min']}-{result['hook_depth_max']}m (最佳{result['hook_depth_optimal']}m)"
        h["best_fishing_time"] = result["best_time"]
        h["alt_fishing_time"] = result["alt_time"]
        h["fishing_method"] = result["method"]
        h["dvm_depth_m"] = result["hook_depth_optimal"]  # for HTML popup

    log.info(
        f"  🪝 Hook Depth: {len(hotspots)} hotspots, "
        f"depth range={min(h.get('hook_depth_min', 0) for h in hotspots)}-"
        f"{max(h.get('hook_depth_max', 0) for h in hotspots)}m"
    )
    return hotspots
