"""
OceanMaster v13.2 — 戰術微觀特徵
==================================
[v13.5] 新增低成本+高商業價值的戰術特徵:

1. 月相指數 (Lunar Phase Index)
   - 農曆日期 → 0~1 滿月指數
   - 影響延繩釣下鉤深度 (滿月時表層亮, 魚群深潛)
   - 科學依據: Bigelow et al. (2000) — lunar phase affects catch rate

2. 渦旋強度指數 (Eddy Intensity Index)
   - 從 SSH anomaly 衍生
   - 正渦旋 (暖渦, anticyclonic) → 下沉流, 營養鹽少
   - 負渦旋 (冷渦, cyclonic) → 上升流, 營養鹽豐富
   - 科學依據: Gaube et al. (2014) — mesoscale eddies and tuna
"""

import math
import numpy as np
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

log = logging.getLogger("OceanMaster.Tactical")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  月相計算器 (Lunar Phase Calculator)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 已知新月日期 (UTC): 2000-01-06 18:14
_KNOWN_NEW_MOON = datetime(2000, 1, 6, 18, 14, 0, tzinfo=timezone.utc)
_SYNODIC_MONTH = 29.53058867  # 朔望月 (天)


def lunar_phase_index(dt: datetime = None) -> Dict[str, Any]:
    """
    計算指定日期的月相指數

    Returns:
        dict:
            phase: float, 0~1 (0=新月, 0.5=滿月)
            fullness: float, 0~1 (0=新月/全暗, 1=滿月/全亮)
            name: str, 月相名稱
            lunar_day: int, 農曆日 (1~30)
            fishing_impact: str, 對漁業的影響說明
            hook_depth_advice: str, 建議下鉤深度
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    # 計算自已知新月以來的天數
    delta_days = (dt - _KNOWN_NEW_MOON).total_seconds() / 86400

    # 月相 (0~1): 0=新月, 0.5=滿月
    phase = (delta_days % _SYNODIC_MONTH) / _SYNODIC_MONTH

    # 月亮亮度 (0~1): 使用 cos 函數, 0=新月(最暗), 1=滿月(最亮)
    fullness = 0.5 * (1 - math.cos(2 * math.pi * phase))

    # 農曆日 (近似)
    lunar_day = int(phase * _SYNODIC_MONTH) + 1

    # 月相名稱
    if phase < 0.0625:
        name = "新月 🌑"
    elif phase < 0.1875:
        name = "眉月 🌒"
    elif phase < 0.3125:
        name = "上弦月 🌓"
    elif phase < 0.4375:
        name = "盈凸月 🌔"
    elif phase < 0.5625:
        name = "滿月 🌕"
    elif phase < 0.6875:
        name = "虧凸月 🌖"
    elif phase < 0.8125:
        name = "下弦月 🌗"
    elif phase < 0.9375:
        name = "殘月 🌘"
    else:
        name = "新月 🌑"

    # 漁業影響
    if fullness > 0.8:
        fishing_impact = "滿月期 — 表層光線強，魚群傾向深潛，延繩釣效果可能下降"
        hook_depth_advice = "建議下鉤 200-300m (比平時深 50-100m)"
    elif fullness > 0.5:
        fishing_impact = "月光偏亮 — 魚群活動深度略深"
        hook_depth_advice = "建議下鉤 150-250m"
    elif fullness > 0.2:
        fishing_impact = "月光適中 — 正常作業深度"
        hook_depth_advice = "建議下鉤 100-200m (標準深度)"
    else:
        fishing_impact = "暗夜期 — 魚群較靠近表層，延繩釣黃金期"
        hook_depth_advice = "建議下鉤 80-150m (可淺放)"

    return {
        "phase": round(phase, 4),
        "fullness": round(fullness, 4),
        "name": name,
        "lunar_day": lunar_day,
        "fishing_impact": fishing_impact,
        "hook_depth_advice": hook_depth_advice,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  渦旋強度指數 (Eddy Intensity Index)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_eddy_index(ssh: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> Dict[str, np.ndarray]:
    """
    從 SSH (海面高度異常) 計算渦旋強度指數

    原理:
    - SSH anomaly > 0 → 暖渦 (anticyclonic): 海面隆起, 下沉流
    - SSH anomaly < 0 → 冷渦 (cyclonic): 海面凹陷, 上升流

    漁業影響:
    - 冷渦邊緣 = 營養鹽上湧 → 浮游生物 → 餌料 → 鮪魚聚集
    - 暖渦-冷渦交界 = 最佳漁場

    Args:
        ssh: 2D SSH anomaly (m)
        lats, lons: 1D coordinate arrays

    Returns:
        dict:
            eddy_intensity: 2D, 渦旋絕對強度 (0~1)
            eddy_type: 2D, +1=暖渦, -1=冷渦, 0=無明顯渦旋
            eddy_edge: 2D, 渦旋邊緣指數 (gradient magnitude, 高值=邊緣)
            cyclonic_index: 2D, 冷渦加分 (0~1, 冷渦邊緣最高)
    """
    if ssh is None or ssh.size == 0:
        log.warning("  SSH data missing — eddy index unavailable")
        dummy = np.zeros((len(lats), len(lons))) if lats.ndim == 1 else np.zeros_like(ssh)
        return {
            "eddy_intensity": dummy,
            "eddy_type": dummy.astype(int),
            "eddy_edge": dummy,
            "cyclonic_index": dummy,
        }

    ssh_clean = np.nan_to_num(ssh, nan=0.0)

    # === 渦旋強度: SSH anomaly 的絕對值 (標準化到 0~1) ===
    ssh_abs = np.abs(ssh_clean)
    # 分位數標準化 (避免極端值影響)
    p95 = np.percentile(ssh_abs[ssh_abs > 0], 95) if np.any(ssh_abs > 0) else 0.1
    eddy_intensity = np.clip(ssh_abs / max(p95, 0.01), 0, 1).astype(np.float32)

    # === 渦旋類型: +1=暖渦, -1=冷渦 ===
    threshold = 0.02  # 2cm SSH anomaly 閾值
    eddy_type = np.zeros_like(ssh_clean, dtype=np.int8)
    eddy_type[ssh_clean > threshold] = 1   # 暖渦
    eddy_type[ssh_clean < -threshold] = -1  # 冷渦

    # === 渦旋邊緣: SSH gradient magnitude ===
    # 高梯度 = 渦旋邊緣 → 最佳漁場位置
    dy = np.gradient(ssh_clean, axis=0)
    dx = np.gradient(ssh_clean, axis=1)
    gradient_mag = np.sqrt(dy ** 2 + dx ** 2)
    g95 = np.percentile(gradient_mag[gradient_mag > 0], 95) if np.any(gradient_mag > 0) else 0.01
    eddy_edge = np.clip(gradient_mag / max(g95, 0.001), 0, 1).astype(np.float32)

    # === 冷渦加分指數: 冷渦邊緣 → 上升流 → 營養鹽 → 魚群 ===
    # 只有冷渦邊緣才給高分
    cyclonic_mask = (eddy_type == -1).astype(float)
    # 擴展冷渦邊緣 (含邊界 1-2 格)
    from scipy.ndimage import maximum_filter
    cyclonic_expanded = maximum_filter(cyclonic_mask, size=3)
    cyclonic_index = (0.6 * eddy_edge + 0.4 * cyclonic_expanded * eddy_intensity).astype(np.float32)
    cyclonic_index = np.clip(cyclonic_index, 0, 1)

    n_warm = int(np.sum(eddy_type == 1))
    n_cold = int(np.sum(eddy_type == -1))
    log.info(f"  渦旋: {n_warm} 暖渦 + {n_cold} 冷渦, "
             f"邊緣強度 max={eddy_edge.max():.3f}")

    return {
        "eddy_intensity": eddy_intensity,
        "eddy_type": eddy_type,
        "eddy_edge": eddy_edge,
        "cyclonic_index": cyclonic_index,
    }
