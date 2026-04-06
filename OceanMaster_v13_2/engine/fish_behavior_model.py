"""
OceanMaster v13.2 — 鮪魚行為模型 (Tuna Behavior Model)  # [v12-phase8]
====================================================================
模擬鮪魚的行為模式:
  1. 學校駐留時間估計 (基於環境條件)
  2. 最佳進食時段預測 (DVM 晨昏窗口)
  3. 遷移方向預測 (SST 梯度 + 海流)

科學參考:
  - 學校駐留時間: Precioso et al. 2022 (TUN-AI), Fisheries Research
  - DVM 進食窗口: Schaefer & Fuller 2007, Marine Biology
  - 遷移行為: Lehodey et al. 2008, Deep-Sea Res. (SEAPODYM)
  - 大目鮪深潛: Schaefer & Fuller 2010, Marine Biology
"""

import numpy as np
import math
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("OceanMaster.FishBehavior")


# [v12-phase8] 物種行為參數
BEHAVIOR_PARAMS = {
    "yellowfin": {
        "name_zh": "黃鰭鮪",
        "base_residence_days": 7,
        "eddy_bonus_days": (3, 5),
        "speed_km_day": (30, 50),
        "dawn_feeding": True,
        "dusk_feeding": True,
        "night_surface": True,     # 新月時
        "midday_deep": False,
        "max_dive_m": 300,
        "thermal_tolerance_min": 18.0,
    },
    "bigeye": {
        "name_zh": "大目鮪",
        "base_residence_days": 10,
        "eddy_bonus_days": (4, 6),
        "speed_km_day": (20, 40),
        "dawn_feeding": True,
        "dusk_feeding": True,
        "night_surface": True,
        "midday_deep": True,       # 體溫調節深潛 (Schaefer & Fuller 2007)
        "max_dive_m": 700,
        "thermal_tolerance_min": 9.0,
    },
    "skipjack": {
        "name_zh": "正鰹",
        "base_residence_days": 5,
        "eddy_bonus_days": (2, 3),
        "speed_km_day": (40, 60),
        "dawn_feeding": True,
        "dusk_feeding": True,
        "night_surface": True,
        "midday_deep": False,
        "max_dive_m": 250,
        "thermal_tolerance_min": 20.0,
    },
    "albacore": {
        "name_zh": "長鰭鮪",
        "base_residence_days": 8,
        "eddy_bonus_days": (3, 5),
        "speed_km_day": (50, 80),
        "dawn_feeding": True,
        "dusk_feeding": True,
        "night_surface": False,    # 長鰭不太做夜間表層覓食
        "midday_deep": False,
        "max_dive_m": 400,
        "thermal_tolerance_min": 12.0,
    },
}


class FishBehaviorModel:
    """
    [v12-phase8] 鮪魚行為預測模型

    提供:
    1. 學校駐留時間估計 — 鮪魚在某一覓食聚集點停留多久
    2. 最佳進食時段 — 一天中什麼時候捕獲率最高
    3. 遷移方向預測 — 鮪魚下一步去哪裡
    """

    def estimate_school_residence_time(
        self,
        species: str,
        sst: float,
        eddy_present: bool,
        dvm_depth: float,
    ) -> Dict:
        """
        [v12-phase8] 估計鮪魚學校在覓食聚集點的駐留時間

        典型駐留 5-10 天 (TUN-AI 研究: 中位數 10 天)
        渦旋出現: +3-5 天 (滯留效應)
        高 SST: 較短停留 (高新陳代謝，更快耗盡食物)

        Reference: Precioso et al. 2022 (TUN-AI paper)

        Parameters:
            species: 物種 key
            sst: 海表溫度 (°C)
            eddy_present: 是否有渦旋
            dvm_depth: DVM 深度 (m)

        Returns:
            dict with estimated_days, confidence, factors
        """
        bp = BEHAVIOR_PARAMS.get(species, BEHAVIOR_PARAMS["yellowfin"])
        base = bp["base_residence_days"]

        factors = []

        # 渦旋滯留效應
        eddy_bonus = 0
        if eddy_present:
            eddy_bonus = int(np.mean(bp["eddy_bonus_days"]))
            factors.append(f"渦旋滯留 +{eddy_bonus}天")

        # SST 效應 — 高溫 = 高代謝 = 更快消耗食物 = 較短停留
        sst_adj = 0
        tmin = bp["thermal_tolerance_min"]
        if sst > 28.0:
            sst_adj = -2  # 高溫縮短
            factors.append("高水溫 -2天 (代謝加速)")
        elif sst > 25.0:
            sst_adj = -1
            factors.append("溫暖水溫 -1天")
        elif sst < tmin + 3:
            sst_adj = 1  # 冷水延長 (新陳代謝慢)
            factors.append("低水溫 +1天 (代謝減緩)")

        # DVM 可及性 — 如果 DVM 深度在可達範圍內，停留時間增加
        dvm_adj = 0
        if dvm_depth <= bp["max_dive_m"]:
            dvm_adj = 1
            factors.append("DVM 可及 +1天")

        estimated_days = max(2, base + eddy_bonus + sst_adj + dvm_adj)

        return {
            "estimated_days": estimated_days,
            "min_days": max(2, estimated_days - 2),
            "max_days": estimated_days + 3,
            "species": species,
            "species_zh": bp["name_zh"],
            "factors": factors,
            "confidence": 0.65 if eddy_present else 0.5,
        }

    def estimate_feeding_windows(
        self,
        lat: float,
        lon: float,
        date: Optional[datetime],
        species: str,
        lunar_phase: Optional[Dict] = None,
        dvm_profile: Optional[List] = None,
    ) -> Dict:
        """
        [v12-phase8] 估計最佳進食時段

        進食窗口基於 DVM (垂直日遷移) 的晨昏模式:
        - 黎明窗口: 日出前 30 分鐘到日出後 2 小時 (DVM 上升)
        - 黃昏窗口: 日落前 2 小時到日落後 30 分鐘 (DVM 下降)
        - 夜間表層覓食: 僅新月時 (低光照 = DVM 上升更高)
        - 午間深潛覓食: 僅大目鮪 (體溫調節潛水)

        Reference: Schaefer & Fuller 2007 "Vertical movements of
                   bigeye tuna"

        Parameters:
            lat, lon: 位置 (用於計算日出/日落)
            date: 日期
            species: 物種 key
            lunar_phase: 月相資訊 (from LunarPhaseEngine)
            dvm_profile: 24h DVM 深度剖面

        Returns:
            dict with feeding_windows list, best_window, summary
        """
        if date is None:
            date = datetime.now(timezone.utc)

        bp = BEHAVIOR_PARAMS.get(species, BEHAVIOR_PARAMS["yellowfin"])

        # 簡化日出/日落計算 (基於緯度 + 經度 + 日期)
        sunrise, sunset = self._estimate_sun_times(lat, date, lon=lon)

        windows = []

        # 1. 黎明窗口 (DVM 上升期)
        if bp["dawn_feeding"]:
            dawn_start = sunrise - timedelta(minutes=30)
            dawn_end = sunrise + timedelta(hours=2)
            windows.append({
                "name": "黎明覓食",
                "name_en": "Dawn feeding",
                "start": dawn_start.strftime("%H:%M"),
                "end": dawn_end.strftime("%H:%M"),
                "type": "dawn",
                "quality": 0.9,
                "depth_range": "表層-中層",
                "mechanism": "DVM 上升，餌料從深層移至表層",
            })

        # 2. 黃昏窗口 (DVM 下降期)
        if bp["dusk_feeding"]:
            dusk_start = sunset - timedelta(hours=2)
            dusk_end = sunset + timedelta(minutes=30)
            windows.append({
                "name": "黃昏覓食",
                "name_en": "Dusk feeding",
                "start": dusk_start.strftime("%H:%M"),
                "end": dusk_end.strftime("%H:%M"),
                "type": "dusk",
                "quality": 0.85,
                "depth_range": "中層",
                "mechanism": "DVM 下降前的最後覓食機會",
            })

        # 3. 夜間表層覓食 (新月限定)
        if bp["night_surface"]:
            is_new_moon = False
            if lunar_phase is not None:
                illum = lunar_phase.get("illumination", 0.5)
                is_new_moon = illum < 0.15
            if is_new_moon:
                night_start = sunset + timedelta(hours=1)
                night_end = sunrise - timedelta(hours=1)
                # 處理跨午夜
                if night_end < night_start:
                    night_end += timedelta(days=1)
                windows.append({
                    "name": "新月夜間表層覓食",
                    "name_en": "New moon night surface",
                    "start": night_start.strftime("%H:%M"),
                    "end": night_end.strftime("%H:%M"),
                    "type": "night_surface",
                    "quality": 0.75,
                    "depth_range": "表層 (0-50m)",
                    "mechanism": "低光照 → DVM 上升更高 → 表層餌料密集",
                })

        # 4. 午間深潛覓食 (大目鮪限定)
        if bp["midday_deep"]:
            midday = sunrise + timedelta(hours=5)
            midday_end = sunset - timedelta(hours=3)
            windows.append({
                "name": "午間深潛覓食",
                "name_en": "Midday deep dive",
                "start": midday.strftime("%H:%M"),
                "end": midday_end.strftime("%H:%M"),
                "type": "midday_deep",
                "quality": 0.6,
                "depth_range": f"深層 (200-{bp['max_dive_m']}m)",
                "mechanism": "體溫調節潛水 — 表層升溫後潛入深層覓食",
            })

        # 最佳窗口
        best = max(windows, key=lambda w: w["quality"]) if windows else None

        return {
            "species": species,
            "species_zh": bp["name_zh"],
            "date": date.strftime("%Y-%m-%d"),
            "lat": lat,
            "lon": lon,
            "sunrise": sunrise.strftime("%H:%M"),
            "sunset": sunset.strftime("%H:%M"),
            "feeding_windows": windows,
            "best_window": best["name"] if best else "N/A",
            "total_windows": len(windows),
            "lunar_affected": lunar_phase.get("illumination", 0.5) < 0.15 if lunar_phase else False,
        }

    def predict_migration_direction(
        self,
        sst_gradient: Optional[Tuple[float, float]],
        current_vectors: Optional[Tuple[float, float]],
        historical_pattern: Optional[str],
        month: int,
        species: str,
        lat: float = 25.0,
        lon: float = 130.0,
    ) -> Dict:
        """
        [v12-phase8] 預測鮪魚遷移方向

        使用 SST 梯度、海流向量和歷史模式來預測遷移:
        - 方向: 羅盤方位
        - 速度: 公里/天
        - 使用拉格朗日平流來投射 3 天餌料場漂移

        Reference: Lehodey et al. 2008 SEAPODYM

        Parameters:
            sst_gradient: SST 梯度向量 (dT/dx, dT/dy) °C/degree
            current_vectors: 海流分量 (u, v) m/s
            historical_pattern: 歷史遷移模式 ("northward", "southward", etc.)
            month: 月份 (1-12)
            species: 物種 key
            lat, lon: 當前位置

        Returns:
            dict with direction, speed, projected_positions, confidence
        """
        bp = BEHAVIOR_PARAMS.get(species, BEHAVIOR_PARAMS["yellowfin"])
        speed_min, speed_max = bp["speed_km_day"]

        # ── 方向計算 ──
        dx, dy = 0.0, 0.0

        # 1. SST 梯度 (魚類傾向沿溫度鋒面移動)
        if sst_gradient is not None:
            # 鮪魚沿著等溫線移動 (垂直於梯度)
            grad_x, grad_y = sst_gradient
            grad_mag = math.sqrt(grad_x**2 + grad_y**2) + 1e-10
            # 垂直於梯度 (逆時針旋轉90度)
            dx += -grad_y / grad_mag * 0.4
            dy += grad_x / grad_mag * 0.4

        # 2. 海流向量 (被動被海流帶動)
        if current_vectors is not None:
            u, v = current_vectors
            # 轉換 m/s → 大略影響方向 (每天 ~86400秒)
            current_km_day = math.sqrt(u**2 + v**2) * 86.4
            if current_km_day > 0:
                dx += u / (math.sqrt(u**2 + v**2) + 1e-10) * 0.3
                dy += v / (math.sqrt(u**2 + v**2) + 1e-10) * 0.3

        # 3. 季節性遷移偏好 (歷史模式)
        seasonal_dir = self._get_seasonal_migration(species, month, lat)
        dx += seasonal_dir[0] * 0.3
        dy += seasonal_dir[1] * 0.3

        # 計算方位角 (以北為0°, 順時針)
        bearing = math.degrees(math.atan2(dx, dy)) % 360

        # 速度估計
        vec_mag = math.sqrt(dx**2 + dy**2)
        if vec_mag > 0:
            speed = speed_min + (speed_max - speed_min) * min(vec_mag, 1.0)
        else:
            speed = (speed_min + speed_max) / 2

        # 投射 3 天位置
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * math.cos(math.radians(lat))
        projected = []
        for day in range(1, 4):
            dlat = speed * day * math.cos(math.radians(bearing)) / km_per_deg_lat
            dlon = speed * day * math.sin(math.radians(bearing)) / max(km_per_deg_lon, 1)
            projected.append({
                "day": day,
                "lat": round(lat + dlat, 3),
                "lon": round(lon + dlon, 3),
            })

        # 信心度
        factors_available = sum([
            sst_gradient is not None,
            current_vectors is not None,
            historical_pattern is not None,
        ])
        confidence = 0.3 + factors_available * 0.2

        # 方位名稱
        compass = self._bearing_to_compass(bearing)

        return {
            "species": species,
            "species_zh": bp["name_zh"],
            "direction_degrees": round(bearing, 1),
            "direction_compass": compass,
            "speed_km_day": round(speed, 1),
            "projected_positions_3d": projected,
            "confidence": round(confidence, 2),
            "month": month,
            "factors": {
                "sst_gradient": sst_gradient is not None,
                "ocean_current": current_vectors is not None,
                "seasonal_pattern": seasonal_dir is not None,
            },
        }

    # ─── 內部工具方法 ───

    @staticmethod
    def _estimate_sun_times(lat: float, date: datetime,
                            lon: float = 130.0) -> Tuple[datetime, datetime]:
        """
        簡化日出/日落時間估計 (基於緯度 + 經度 + 年中日)

        使用 NOAA 簡化公式的近似版本。精度 ±15 分鐘。
        [v13.2-audit] 改用實際經度計算，不再硬編碼 130°E
        """
        doy = date.timetuple().tm_yday
        # 太陽赤緯
        decl = -23.45 * math.cos(math.radians(360 / 365 * (doy + 10)))
        # 時角
        lat_rad = math.radians(lat)
        decl_rad = math.radians(decl)

        cos_ha = -math.tan(lat_rad) * math.tan(decl_rad)
        cos_ha = max(-1.0, min(1.0, cos_ha))  # 極地安全
        ha = math.degrees(math.acos(cos_ha))

        # [v13.2-audit] 使用實際經度計算太陽正午 (UTC)
        # 太陽正午 UTC = 12:00 - longitude / 15
        solar_noon_h = 12.0 - lon / 15.0
        sunrise_h = solar_noon_h - ha / 15.0
        sunset_h = solar_noon_h + ha / 15.0

        sunrise_h = max(0, min(23.99, sunrise_h))
        sunset_h = max(0, min(23.99, sunset_h))

        sunrise = date.replace(
            hour=int(sunrise_h),
            minute=int((sunrise_h % 1) * 60),
            second=0, microsecond=0,
        )
        sunset = date.replace(
            hour=int(sunset_h),
            minute=int((sunset_h % 1) * 60),
            second=0, microsecond=0,
        )
        return sunrise, sunset

    @staticmethod
    def _get_seasonal_migration(species: str, month: int, lat: float) -> Tuple[float, float]:
        """
        季節性遷移偏好向量 (dx, dy)

        基於各物種的已知遷移模式:
        - 黃鰭/正鰹: 夏季北移，冬季南移 (追逐暖流)
        - 大目: 較少季節遷移，深層垂直移動為主
        - 長鰭: 追逐溫度鋒面，季節性南北移動明顯
        """
        # 北半球: 夏季 (5-9月) 北移，冬季 (11-3月) 南移
        if species in ("yellowfin", "skipjack"):
            if 5 <= month <= 9:
                return (0.0, 0.5)   # 北移
            elif month in (11, 12, 1, 2, 3):
                return (0.0, -0.5)  # 南移
            else:
                return (0.2, 0.0)   # 東移 (春秋過渡)

        elif species == "albacore":
            # 長鰭追逐北太平洋溫度鋒面 (黑潮延伸)
            if 4 <= month <= 8:
                return (0.3, 0.6)   # 東北移
            elif month in (10, 11, 12, 1):
                return (-0.2, -0.5) # 西南移
            else:
                return (0.1, 0.0)

        elif species == "bigeye":
            # 大目遷移較弱，偏好深層
            if lat > 25:
                return (0.0, -0.2)  # 微南移 (回暖水)
            else:
                return (0.0, 0.1)   # 微北移
        else:
            return (0.0, 0.0)

    @staticmethod
    def _bearing_to_compass(bearing: float) -> str:
        """方位角轉羅盤方位名稱"""
        dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        idx = round(bearing / 22.5) % 16
        return dirs[idx]
