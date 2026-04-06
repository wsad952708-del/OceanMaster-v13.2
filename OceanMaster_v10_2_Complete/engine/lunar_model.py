"""
OceanMaster v11 — 月相效應引擎 (Lunar Phase Engine)
====================================================
計算月相並估算對不同魚種×漁法的 CPUE 修正係數。

科學依據:
  - 新月夜晚: 最暗 → 集魚燈效果最佳 → 正鰹/魷魚 CPUE 提升 20-30%
  - 滿月夜晚: 最亮 → 魚群分散 → 延繩釣黃鰭鮪 CPUE 些微下降
  - 大目鮪: 月相影響夜間垂直遷移深度 (Musyl 2011)

文獻:
  - Skipjack/Squid purse seine: 月相效應顯著 (方向確認，係數保守估計)
  - Bigeye longline: Musyl et al. 2011 (MEPS)
  - Yellowfin CPUE: Hays et al. 2001

注意: 月相係數採保守值 (1.2x/0.8x)，待船長回報數據累積後可校準。
"""

import numpy as np
import logging
from datetime import datetime, timezone
from typing import Dict, Optional

log = logging.getLogger("OceanMaster.Lunar")


# 月相 CPUE 修正係數查找表
# key = (species, fishing_method)
# value = (new_moon_multiplier, full_moon_multiplier)
# 中間月相按 cosine 內插
LUNAR_CPUE_MODIFIERS = {
    # 圍網: 新月集魚燈效果好，滿月差
    ("skipjack", "purse_seine"):  (1.20, 0.80),
    ("squid", "squid_jigging"):   (1.25, 0.75),

    # 延繩釣: 月相影響較小
    ("yellowfin", "longline"):    (1.05, 0.95),
    ("bigeye", "longline"):       (1.08, 0.92),
    ("albacore", "longline"):     (1.03, 0.97),

    # 預設 (未指定漁法)
    ("skipjack", "longline"):     (1.03, 0.97),
    ("yellowfin", "purse_seine"): (1.10, 0.90),
}


class LunarPhaseEngine:
    """
    月相計算與漁業效應模型

    使用 Jean Meeus "Astronomical Algorithms" 的簡化公式，
    不需要 ephem 外部依賴。

    使用:
        lunar = LunarPhaseEngine()
        phase = lunar.compute_moon_phase(datetime(2025, 6, 15))
        # phase['illumination'] → 0.0~1.0 (0=新月, 1=滿月)
        modifier = lunar.compute_lunar_cpue_modifier('skipjack', 'purse_seine', phase['illumination'])
        # modifier → 0.8~1.2
    """

    # Meeus 新月參考曆元 (JDE)
    _KNOWN_NEW_MOON_JDE = 2451550.1  # 2000-01-06T18:14 UTC

    # 朔望月 (synodic month) = 29.53058886 天
    SYNODIC_MONTH = 29.53058886

    def compute_moon_phase(self, date: datetime) -> Dict:
        """
        計算月相 — 使用 Jean Meeus 簡化公式

        Parameters:
            date: 日期 (aware or naive, naive 假設 UTC)

        Returns:
            dict with:
              - illumination: 0.0-1.0 (0=新月, 1=滿月)
              - phase_angle: 0-360° (0=新月, 180=滿月)
              - phase_name: 新月/上弦/滿月/下弦
              - phase_emoji: 🌑🌓🌕🌗
              - days_since_new: 距上次新月天數
              - days_to_new: 距下次新月天數
        """
        # 轉換為 Julian Date
        if date.tzinfo is not None:
            date_utc = date.astimezone(timezone.utc)
        else:
            date_utc = date

        # Julian Date 計算
        y = date_utc.year
        m = date_utc.month
        d = date_utc.day + date_utc.hour / 24.0 + date_utc.minute / 1440.0

        if m <= 2:
            y -= 1
            m += 12

        A = int(y / 100)
        B = 2 - A + int(A / 4)
        jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + B - 1524.5

        # 月齡 (天)
        days_since_ref_new = jd - self._KNOWN_NEW_MOON_JDE
        lunations = days_since_ref_new / self.SYNODIC_MONTH
        phase_fraction = lunations - int(lunations)
        if phase_fraction < 0:
            phase_fraction += 1.0

        days_since_new = phase_fraction * self.SYNODIC_MONTH
        days_to_new = (1.0 - phase_fraction) * self.SYNODIC_MONTH

        # 月相角 (0-360°)
        phase_angle = phase_fraction * 360.0

        # 月面照亮率 (0-1)
        # illumination ≈ (1 - cos(phase_angle)) / 2
        illumination = (1.0 - np.cos(np.radians(phase_angle))) / 2.0

        # 月相名稱
        if phase_fraction < 0.0625 or phase_fraction >= 0.9375:
            phase_name = "新月"
            emoji = "🌑"
        elif phase_fraction < 0.1875:
            phase_name = "眉月"
            emoji = "🌒"
        elif phase_fraction < 0.3125:
            phase_name = "上弦"
            emoji = "🌓"
        elif phase_fraction < 0.4375:
            phase_name = "盈凸"
            emoji = "🌔"
        elif phase_fraction < 0.5625:
            phase_name = "滿月"
            emoji = "🌕"
        elif phase_fraction < 0.6875:
            phase_name = "虧凸"
            emoji = "🌖"
        elif phase_fraction < 0.8125:
            phase_name = "下弦"
            emoji = "🌗"
        else:
            phase_name = "殘月"
            emoji = "🌘"

        return {
            "illumination": float(illumination),
            "phase_angle": float(phase_angle),
            "phase_name": phase_name,
            "phase_emoji": emoji,
            "days_since_new": float(days_since_new),
            "days_to_new": float(days_to_new),
        }

    def compute_lunar_cpue_modifier(
        self,
        species: str,
        fishing_method: str = "longline",
        illumination: float = 0.5,
    ) -> float:
        """
        月相 CPUE 修正係數 (乘以基礎 HSI 分數)

        Parameters:
            species: 物種 ID
            fishing_method: 漁法 ('longline'|'purse_seine'|'squid_jigging')
            illumination: 月面照亮率 (0=新月, 1=滿月)

        Returns:
            modifier: CPUE 修正係數 (0.75~1.25)
        """
        key = (species, fishing_method)
        if key not in LUNAR_CPUE_MODIFIERS:
            # 嘗試匹配物種，使用預設漁法
            for k, v in LUNAR_CPUE_MODIFIERS.items():
                if k[0] == species:
                    key = k
                    break
            else:
                return 1.0  # 未知組合不修正

        new_mult, full_mult = LUNAR_CPUE_MODIFIERS[key]

        # illumination: 0=新月→new_mult, 1=滿月→full_mult
        # 使用 cosine 內插確保平滑過渡
        modifier = new_mult + (full_mult - new_mult) * illumination

        return float(np.clip(modifier, 0.5, 1.5))

    def get_best_fishing_days(
        self,
        start_date: datetime,
        n_days: int = 14,
        species: str = "skipjack",
        fishing_method: str = "purse_seine",
    ) -> list:
        """
        推薦未來 N 天的最佳出航日 (基於月相)

        Returns:
            list of dict: [{date, illumination, modifier, recommendation}]
        """
        from datetime import timedelta

        recommendations = []
        for i in range(n_days):
            d = start_date + timedelta(days=i)
            phase = self.compute_moon_phase(d)
            modifier = self.compute_lunar_cpue_modifier(
                species, fishing_method, phase["illumination"]
            )

            if modifier > 1.10:
                rec = "⭐ 極佳"
            elif modifier > 1.02:
                rec = "👍 良好"
            elif modifier > 0.95:
                rec = "— 普通"
            else:
                rec = "⚠️ 不佳"

            recommendations.append({
                "date": d.strftime("%Y-%m-%d"),
                "phase": phase["phase_name"],
                "emoji": phase["phase_emoji"],
                "illumination": round(phase["illumination"], 2),
                "modifier": round(modifier, 3),
                "recommendation": rec,
            })

        return recommendations
