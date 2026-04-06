"""
OceanMaster v13.2 — 安全區域檢查器
=====================================
Task #12: ML 預測流程前的強制安全過濾

判斷邏輯:
  🔴 AVOID:   颱風中心 500km 內
  🟡 CAUTION: 波高 > 2.5m  OR  風速 > 20m/s  OR  氣壓 < 990hPa
  ⭐ OPTIMAL: 颱風過境後 1~3 天（潛在高 CPUE 期）
  🟢 SAFE:    其他

整合點:
  在 main_v10_3.py 的 AI Fusion 前插入，
  標記每個 hotspot 的 safety_level 欄位。
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("OceanMaster.Safety")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  安全等級
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SafetyLevel(Enum):
    AVOID    = "AVOID"     # 🔴 絕對禁止
    CAUTION  = "CAUTION"   # 🟡 需特別注意
    SAFE     = "SAFE"      # 🟢 正常作業
    OPTIMAL  = "OPTIMAL"   # ⭐ 颱風後黃金漁場

    @property
    def emoji(self) -> str:
        return {
            "AVOID": "🔴",
            "CAUTION": "🟡",
            "SAFE": "🟢",
            "OPTIMAL": "⭐",
        }[self.value]


@dataclass
class SafetyResult:
    """安全評估結果"""
    level: SafetyLevel
    reason: str                         # 人類可讀原因
    details: Dict[str, Any]             # 細節 (距颱風距離, 波高, etc.)
    score: float = 1.0                  # 安全分數 0-1 (1=最安全)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  閾值常數
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TYPHOON_DANGER_RADIUS_KM = 500      # 颱風中心危險半徑
TYPHOON_CAUTION_RADIUS_KM = 800     # 颱風中心警告半徑
WAVE_DANGER_M = 4.0                 # 波高絕對危險
WAVE_CAUTION_M = 2.5                # 波高警告
WIND_DANGER_MS = 25.0               # 風速絕對危險 (m/s)
WIND_CAUTION_MS = 15.0              # 風速警告 (m/s)
PRESSURE_LOW_HPA = 990.0            # 低壓警告
POST_TYPHOON_OPTIMAL_DAYS = (1, 3)  # 颱風後黃金期 (天)
POST_TYPHOON_MAX_DIST_KM = 300      # 颱風後黃金期最大距離


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  核心函數
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def is_safe_for_fishing(
    lat: float,
    lon: float,
    date: Optional[datetime] = None,
    typhoon_alerts: Optional[List] = None,
    wave_height: Optional[float] = None,
    wind_speed: Optional[float] = None,
    pressure_msl: Optional[float] = None,
    recent_typhoon_passages: Optional[List[Dict]] = None,
) -> SafetyResult:
    """
    評估特定座標的作業安全性。

    Args:
        lat, lon: 評估座標
        date: 評估日期 (default: now UTC)
        typhoon_alerts: TyphoonAlert 列表 (來自 typhoon_tracker)
        wave_height: 有效波高 (m)
        wind_speed: 風速 (m/s)
        pressure_msl: 海平面氣壓 (hPa)
        recent_typhoon_passages: 近期颱風過境記錄 [{date, lat, lon, intensity}]

    Returns:
        SafetyResult(level, reason, details, score)
    """
    if date is None:
        date = datetime.now(timezone.utc)

    details: Dict[str, Any] = {
        "lat": lat, "lon": lon,
        "date": date.isoformat(),
    }
    reasons = []

    # ─── 1. 颱風距離檢查 (最高優先) ───
    min_typhoon_dist = float("inf")
    nearest_typhoon = None

    if typhoon_alerts:
        for alert in typhoon_alerts:
            alert_lat = alert.lat if hasattr(alert, 'lat') else alert.get("lat", 0)
            alert_lon = alert.lon if hasattr(alert, 'lon') else alert.get("lon", 0)
            dist = _haversine_km(lat, lon, alert_lat, alert_lon)

            if dist < min_typhoon_dist:
                min_typhoon_dist = dist
                nearest_typhoon = alert

        details["nearest_typhoon_km"] = round(min_typhoon_dist, 0)

        if min_typhoon_dist < TYPHOON_DANGER_RADIUS_KM:
            name = (nearest_typhoon.name if hasattr(nearest_typhoon, 'name')
                    else nearest_typhoon.get("name", "?"))
            return SafetyResult(
                level=SafetyLevel.AVOID,
                reason=f"颱風 {name} 中心 {min_typhoon_dist:.0f}km 內 "
                       f"(危險半徑 {TYPHOON_DANGER_RADIUS_KM}km)",
                details=details,
                score=0.0,
            )

        if min_typhoon_dist < TYPHOON_CAUTION_RADIUS_KM:
            reasons.append(
                f"距颱風 {min_typhoon_dist:.0f}km (警告半徑 {TYPHOON_CAUTION_RADIUS_KM}km)"
            )

    # ─── 2. 波高檢查 ───
    if wave_height is not None:
        details["wave_height_m"] = wave_height
        if wave_height >= WAVE_DANGER_M:
            return SafetyResult(
                level=SafetyLevel.AVOID,
                reason=f"有效波高 {wave_height:.1f}m ≥ {WAVE_DANGER_M}m (極危險)",
                details=details,
                score=0.0,
            )
        if wave_height >= WAVE_CAUTION_M:
            reasons.append(f"波高 {wave_height:.1f}m ≥ {WAVE_CAUTION_M}m")

    # ─── 3. 風速檢查 ───
    if wind_speed is not None:
        details["wind_speed_ms"] = wind_speed
        if wind_speed >= WIND_DANGER_MS:
            return SafetyResult(
                level=SafetyLevel.AVOID,
                reason=f"風速 {wind_speed:.1f}m/s ≥ {WIND_DANGER_MS}m/s (極危險)",
                details=details,
                score=0.1,
            )
        if wind_speed >= WIND_CAUTION_MS:
            reasons.append(f"風速 {wind_speed:.1f}m/s ≥ {WIND_CAUTION_MS}m/s")

    # ─── 4. 低壓檢查 ───
    if pressure_msl is not None:
        details["pressure_hpa"] = pressure_msl
        if pressure_msl < PRESSURE_LOW_HPA:
            reasons.append(f"氣壓 {pressure_msl:.0f}hPa < {PRESSURE_LOW_HPA}hPa (低壓系統)")

    # ─── 5. 颱風後黃金期檢測 ───
    if recent_typhoon_passages:
        for passage in recent_typhoon_passages:
            p_date = passage.get("date")
            if isinstance(p_date, str):
                try:
                    p_date = datetime.fromisoformat(p_date.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    continue

            if p_date is None:
                continue

            days_after = (date - p_date).total_seconds() / 86400
            p_lat = passage.get("lat", 0)
            p_lon = passage.get("lon", 0)
            dist = _haversine_km(lat, lon, p_lat, p_lon)

            if (POST_TYPHOON_OPTIMAL_DAYS[0] <= days_after <= POST_TYPHOON_OPTIMAL_DAYS[1]
                    and dist <= POST_TYPHOON_MAX_DIST_KM
                    and not reasons):  # 無其他危險
                details["post_typhoon_days"] = round(days_after, 1)
                details["post_typhoon_dist_km"] = round(dist, 0)
                return SafetyResult(
                    level=SafetyLevel.OPTIMAL,
                    reason=f"颱風過境後 {days_after:.1f} 天，距路徑 {dist:.0f}km "
                           f"(潛在高 CPUE 黃金期)",
                    details=details,
                    score=1.0,
                )

    # ─── 6. 總結判定 ───
    if reasons:
        # 計算安全分數 (0-1)
        score = 1.0
        if wave_height and wave_height >= WAVE_CAUTION_M:
            score -= 0.3 * min((wave_height - WAVE_CAUTION_M) / (WAVE_DANGER_M - WAVE_CAUTION_M), 1.0)
        if wind_speed and wind_speed >= WIND_CAUTION_MS:
            score -= 0.3 * min((wind_speed - WIND_CAUTION_MS) / (WIND_DANGER_MS - WIND_CAUTION_MS), 1.0)
        if min_typhoon_dist < TYPHOON_CAUTION_RADIUS_KM:
            score -= 0.2 * (1 - (min_typhoon_dist - TYPHOON_DANGER_RADIUS_KM)
                            / (TYPHOON_CAUTION_RADIUS_KM - TYPHOON_DANGER_RADIUS_KM))
        score = max(0.1, score)

        return SafetyResult(
            level=SafetyLevel.CAUTION,
            reason=" | ".join(reasons),
            details=details,
            score=round(score, 2),
        )

    return SafetyResult(
        level=SafetyLevel.SAFE,
        reason="海況正常，適合作業",
        details=details,
        score=1.0,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  批量評估 (用於 hotspot 過濾)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def assess_grid_safety(
    lats: np.ndarray,
    lons: np.ndarray,
    typhoon_alerts: Optional[List] = None,
    wave_height: Optional[np.ndarray] = None,
    wind_speed: Optional[np.ndarray] = None,
    pressure_msl: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    對整個 2D 網格進行安全評估。

    Returns:
        {
            'safety_level': 2D int array (0=AVOID, 1=CAUTION, 2=SAFE, 3=OPTIMAL)
            'safety_score': 2D float array (0-1)
            'typhoon_distance': 2D float array (km to nearest typhoon)
        }
    """
    ny, nx = len(lats), len(lons)
    level_grid = np.full((ny, nx), 2, dtype=np.int32)  # default SAFE
    score_grid = np.ones((ny, nx), dtype=np.float32)
    typhoon_dist = np.full((ny, nx), 9999.0, dtype=np.float32)

    # ── 颱風距離矩陣 ──
    if typhoon_alerts:
        for alert in typhoon_alerts:
            a_lat = alert.lat if hasattr(alert, 'lat') else alert.get("lat", 0)
            a_lon = alert.lon if hasattr(alert, 'lon') else alert.get("lon", 0)
            for iy, la in enumerate(lats):
                for ix, lo in enumerate(lons):
                    d = _haversine_km(la, lo, a_lat, a_lon)
                    if d < typhoon_dist[iy, ix]:
                        typhoon_dist[iy, ix] = d

        # AVOID: < 500km
        avoid_mask = typhoon_dist < TYPHOON_DANGER_RADIUS_KM
        level_grid[avoid_mask] = 0
        score_grid[avoid_mask] = 0.0

        # CAUTION: 500-800km
        caution_mask = (typhoon_dist >= TYPHOON_DANGER_RADIUS_KM) & \
                       (typhoon_dist < TYPHOON_CAUTION_RADIUS_KM)
        level_grid[caution_mask] = np.minimum(level_grid[caution_mask], 1)
        score_grid[caution_mask] = np.clip(
            (typhoon_dist[caution_mask] - TYPHOON_DANGER_RADIUS_KM) /
            (TYPHOON_CAUTION_RADIUS_KM - TYPHOON_DANGER_RADIUS_KM),
            0.2, 0.8,
        )

    # ── 波高 ──
    if wave_height is not None:
        wh = wave_height if wave_height.shape == (ny, nx) else np.full((ny, nx), 1.5)
        danger_mask = wh >= WAVE_DANGER_M
        level_grid[danger_mask] = 0
        score_grid[danger_mask] = 0.0

        caution_mask = (wh >= WAVE_CAUTION_M) & (wh < WAVE_DANGER_M)
        level_grid[caution_mask] = np.minimum(level_grid[caution_mask], 1)
        score_grid[caution_mask] *= np.clip(
            1.0 - (wh[caution_mask] - WAVE_CAUTION_M) / (WAVE_DANGER_M - WAVE_CAUTION_M),
            0.3, 0.9,
        )

    # ── 風速 ──
    if wind_speed is not None:
        ws = wind_speed if wind_speed.shape == (ny, nx) else np.full((ny, nx), 5.0)
        danger_mask = ws >= WIND_DANGER_MS
        level_grid[danger_mask] = 0
        score_grid[danger_mask] = 0.0

        caution_mask = (ws >= WIND_CAUTION_MS) & (ws < WIND_DANGER_MS)
        level_grid[caution_mask] = np.minimum(level_grid[caution_mask], 1)
        score_grid[caution_mask] *= np.clip(
            1.0 - (ws[caution_mask] - WIND_CAUTION_MS) / (WIND_DANGER_MS - WIND_CAUTION_MS),
            0.3, 0.9,
        )

    n_avoid = int(np.sum(level_grid == 0))
    n_caution = int(np.sum(level_grid == 1))
    n_safe = int(np.sum(level_grid >= 2))
    log.info(f"  Safety grid: 🔴AVOID={n_avoid}, 🟡CAUTION={n_caution}, 🟢SAFE={n_safe}")

    return {
        "safety_level": level_grid,
        "safety_score": score_grid,
        "typhoon_distance": typhoon_dist,
    }


def filter_hotspots_by_safety(
    hotspots: List[Dict],
    safety_result: Optional[Dict] = None,
    typhoon_alerts: Optional[List] = None,
    wave_height_grid: Optional[np.ndarray] = None,
    lats: Optional[np.ndarray] = None,
    lons: Optional[np.ndarray] = None,
) -> List[Dict]:
    """
    為每個 hotspot 添加 safety_level 欄位，並過濾 AVOID 的點。

    Returns:
        過濾後的 hotspot 列表 (移除 AVOID，保留 CAUTION/SAFE/OPTIMAL)
    """
    filtered = []
    for h in hotspots:
        h_lat, h_lon = h.get("lat", 0), h.get("lon", 0)

        # 點查安全性
        wh = None
        ws = None
        if lats is not None and lons is not None and wave_height_grid is not None:
            iy = int(np.argmin(np.abs(lats - h_lat)))
            ix = int(np.argmin(np.abs(lons - h_lon)))
            wh = float(wave_height_grid[iy, ix]) if iy < wave_height_grid.shape[0] and ix < wave_height_grid.shape[1] else None

        result = is_safe_for_fishing(
            lat=h_lat, lon=h_lon,
            typhoon_alerts=typhoon_alerts,
            wave_height=wh,
            wind_speed=ws,
        )

        h["safety_level"] = result.level.value
        h["safety_emoji"] = result.level.emoji
        h["safety_reason"] = result.reason
        h["safety_score"] = result.score

        if result.level != SafetyLevel.AVOID:
            filtered.append(h)
        else:
            log.info(f"  ⛔ 移除危險 hotspot: ({h_lat:.1f},{h_lon:.1f}) — {result.reason}")

    removed = len(hotspots) - len(filtered)
    if removed:
        log.info(f"  Safety filter: 移除 {removed} 個危險 hotspots")

    return filtered


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  工具函數
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """大圓距離"""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  [v16.0] 進階海況危險評估
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_rogue_wave_risk(
    wave_height: np.ndarray,
    wave_period: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    [v16.0 E1] 瘋狗浪風險指數。

    Rogue wave probability increases with wave steepness.
    Risk = Hs² × Tp (energy-period product).
    Ref: Dysthe et al. (2008) Annu. Rev. Fluid Mech.

    Thresholds:
      - Hs > 4m AND Tp > 12s → HIGH risk (0.7-1.0)
      - Hs > 3m AND Tp > 10s → MODERATE risk (0.3-0.7)
      - Otherwise → LOW risk (0-0.3)

    Returns:
        {
            "rogue_risk": 2D float (0-1),
            "rogue_level": 2D int (0=low, 1=moderate, 2=high),
        }
    """
    Hs = np.nan_to_num(wave_height, nan=0.0)
    Tp = np.nan_to_num(wave_period, nan=8.0)

    # Energy-period product (normalized)
    raw = Hs**2 * Tp
    # Normalize: typical danger threshold ~192 (4²×12)
    risk = np.clip(raw / 192.0, 0, 1).astype(np.float32)

    level = np.zeros_like(Hs, dtype=np.int32)
    level[(Hs > 3.0) & (Tp > 10.0)] = 1  # moderate
    level[(Hs > 4.0) & (Tp > 12.0)] = 2  # high

    n_high = int(np.sum(level == 2))
    if n_high > 0:
        log.warning(f"  ⚠️ Rogue wave HIGH risk: {n_high} grid cells")

    return {"rogue_risk": risk, "rogue_level": level}


def compute_parametric_roll_risk(
    wave_period: np.ndarray,
    vessel_roll_period: float = 12.0,
    vessel_speed_kn: float = 10.0,
    vessel_heading_deg: float = 0.0,
    wave_direction_deg: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    [v16.0 E2] 參數橫搖風險。

    Parametric rolling occurs when encounter wave period TE ≈ TR/2
    (ship natural roll period / 2).
    Ref: IMO MSC.1/Circ.1228 (2007)

    TE = Tp / |1 - Vs × cos(θ) / Vwave|
    where θ = relative angle between wave and vessel heading.

    Returns:
        {
            "roll_risk": 2D float (0-1),
            "encounter_period": 2D float (seconds),
        }
    """
    Tp = np.nan_to_num(wave_period, nan=8.0)

    # Wave celerity (deep water approximation): Vw = g*Tp / (2π)
    Vw = 9.81 * Tp / (2 * math.pi)  # m/s
    Vs = vessel_speed_kn * 0.5144    # knots → m/s

    # Relative angle (if wave direction available)
    if wave_direction_deg is not None:
        theta = np.radians(wave_direction_deg - vessel_heading_deg)
    else:
        # Assume head seas (worst case for parametric roll)
        theta = np.full_like(Tp, 0.0)

    cos_theta = np.cos(theta)

    # Encounter period
    denom = np.abs(1.0 - Vs * cos_theta / np.maximum(Vw, 0.1))
    denom = np.maximum(denom, 0.01)  # avoid division by zero
    TE = Tp / denom

    # Danger: TE ≈ TR/2 (within ±20%)
    target = vessel_roll_period / 2.0
    deviation = np.abs(TE - target) / target
    risk = np.clip(1.0 - deviation / 0.3, 0, 1).astype(np.float32)
    # Only relevant when deviation < 30%

    n_danger = int(np.sum(risk > 0.7))
    if n_danger > 0:
        log.warning(f"  ⚠️ Parametric roll danger: {n_danger} cells "
                    f"(TE≈{target:.1f}s resonance)")

    return {
        "roll_risk": risk,
        "encounter_period": TE.astype(np.float32),
    }


def compute_swell_warning(
    wave_period: np.ndarray,
    wave_height: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    [v16.0 E3] 長週期湧浪預警。

    Tp > 16s = long-period swell from distant storm (12-48h advance warning).
    Ref: WMO Guide to Wave Analysis and Forecasting (2018)

    Estimated storm distance = Tp × group_velocity × travel_time approximation.

    Returns:
        {
            "swell_warning": 2D bool mask (True = long-period swell detected),
            "swell_energy": 2D float (Hs² × Tp, normalized),
            "estimated_storm_dist_km": 2D float (rough estimate),
        }
    """
    Tp = np.nan_to_num(wave_period, nan=8.0)
    Hs = np.nan_to_num(wave_height, nan=0.0)

    # Long-period swell mask
    swell_mask = Tp > 16.0

    # Swell energy
    energy = (Hs**2 * Tp).astype(np.float32)
    max_e = max(float(np.max(energy)), 1.0)
    energy_norm = np.clip(energy / max_e, 0, 1).astype(np.float32)

    # Rough storm distance estimate (deep water group velocity = g*T / 4π)
    # Distance ≈ Vg × typical_travel_time (assume 24h)
    Vg = 9.81 * Tp / (4 * math.pi)  # m/s
    storm_dist = Vg * 86400 / 1000   # km (assuming 24h travel)
    storm_dist = np.where(swell_mask, storm_dist, 0).astype(np.float32)

    n_swell = int(np.sum(swell_mask))
    if n_swell > 0:
        log.warning(f"  🌊 Long-period swell detected: {n_swell} cells, "
                    f"Tp up to {float(np.max(Tp[swell_mask])):.1f}s, "
                    f"est. storm ~{float(np.max(storm_dist)):.0f}km away")

    return {
        "swell_warning": swell_mask,
        "swell_energy": energy_norm,
        "estimated_storm_dist_km": storm_dist,
    }


def compute_strong_current_zones(
    u_current: np.ndarray,
    v_current: np.ndarray,
    danger_threshold_kn: float = 3.0,
    caution_threshold_kn: float = 2.0,
) -> Dict[str, np.ndarray]:
    """
    [v16.0 E4] 急流危險區域標示。

    Current speed > 3 knots = danger zone (vessel control issues).
    Current speed > 2 knots = caution zone.

    Returns:
        {
            "current_speed_kn": 2D float (knots),
            "current_danger": 2D int (0=safe, 1=caution, 2=danger),
        }
    """
    u = np.nan_to_num(u_current, nan=0.0)
    v = np.nan_to_num(v_current, nan=0.0)

    # Speed in m/s → knots (1 m/s = 1.944 knots)
    speed_ms = np.sqrt(u**2 + v**2)
    speed_kn = (speed_ms * 1.944).astype(np.float32)

    danger = np.zeros_like(speed_kn, dtype=np.int32)
    danger[speed_kn >= caution_threshold_kn] = 1
    danger[speed_kn >= danger_threshold_kn] = 2

    n_danger = int(np.sum(danger == 2))
    n_caution = int(np.sum(danger == 1))
    if n_danger > 0:
        log.warning(f"  🌊 Strong current: {n_danger} danger cells "
                    f"(>{danger_threshold_kn}kn), {n_caution} caution")

    return {
        "current_speed_kn": speed_kn,
        "current_danger": danger,
    }


def assess_sea_hazards(
    wave_height: Optional[np.ndarray] = None,
    wave_period: Optional[np.ndarray] = None,
    u_current: Optional[np.ndarray] = None,
    v_current: Optional[np.ndarray] = None,
    vessel_roll_period: float = 12.0,
) -> Dict[str, Any]:
    """
    [v16.0] 綜合海況危險評估 — 整合 E1-E4 所有項目。

    Returns combined hazard dict with all E-batch results.
    """
    results: Dict[str, Any] = {}
    hazard_summary = []

    # E1: Rogue wave
    if wave_height is not None and wave_period is not None:
        rogue = compute_rogue_wave_risk(wave_height, wave_period)
        results.update(rogue)
        n_high = int(np.sum(rogue["rogue_level"] == 2))
        if n_high > 0:
            hazard_summary.append(f"瘋狗浪高風險 {n_high} 格")

    # E2: Parametric roll
    if wave_period is not None:
        roll = compute_parametric_roll_risk(wave_period, vessel_roll_period)
        results.update(roll)
        n_roll = int(np.sum(roll["roll_risk"] > 0.7))
        if n_roll > 0:
            hazard_summary.append(f"參數橫搖危險 {n_roll} 格")

    # E3: Swell warning
    if wave_height is not None and wave_period is not None:
        swell = compute_swell_warning(wave_period, wave_height)
        results.update(swell)
        n_swell = int(np.sum(swell["swell_warning"]))
        if n_swell > 0:
            hazard_summary.append(f"遠洋湧浪預警 {n_swell} 格")

    # E4: Strong current
    if u_current is not None and v_current is not None:
        current = compute_strong_current_zones(u_current, v_current)
        results.update(current)
        n_curr = int(np.sum(current["current_danger"] == 2))
        if n_curr > 0:
            hazard_summary.append(f"急流危險 {n_curr} 格")

    results["hazard_summary"] = hazard_summary
    results["has_hazards"] = len(hazard_summary) > 0

    if hazard_summary:
        log.warning(f"  🚨 Sea hazards: {' | '.join(hazard_summary)}")
    else:
        log.info("  ✅ No sea hazards detected")

    return results
