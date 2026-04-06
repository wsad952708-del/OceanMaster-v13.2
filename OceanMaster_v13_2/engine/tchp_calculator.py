"""
OceanMaster v13.2 — 颱風潛熱與熱含量 (TCHP) 計算
===================================================
Task #16: Tropical Cyclone Heat Potential

基於 HYCOM temp_3d 計算:
  TCHP = ρ·Cp · ∫₀^D26 (T(z) - 26) dz

其中:
  ρ  = 1025 kg/m³ (海水密度)
  Cp = 3850 J/(kg·K) (海水比熱)
  D26 = 26°C 等溫線深度

業務邏輯:
  TCHP > 50 kJ/cm² → 颱風可能快速增強 (提前 2 天警報)
  TCHP > 80 kJ/cm² → 颱風幾乎確定增強 (Mainelli et al. 2008)

副產品:
  - D26 (26°C 等溫線深度) 本身也是好的漁場指標
  - TCHP 梯度 = 魚群聚集邊界
"""

import logging
import numpy as np
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger("OceanMaster.TCHP")

# 物理常數
RHO_SW = 1025.0      # kg/m³ 海水密度
CP_SW  = 3850.0      # J/(kg·K) 海水比熱
T_REF  = 26.0        # °C 參考溫度 (26°C 等溫線)

# 警報閾值
TCHP_RAPID_INTENSIFY = 50.0   # kJ/cm²
TCHP_CERTAIN_INTENSIFY = 80.0 # kJ/cm²


def compute_tchp(
    temp_3d: np.ndarray,
    depths: np.ndarray,
    lats: Optional[np.ndarray] = None,
    lons: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    計算颱風潛熱 (Tropical Cyclone Heat Potential)。

    Args:
        temp_3d: 3D (n_depths, ny, nx) 溫度剖面 (°C)
        depths: 1D (n_depths,) 深度值 (m, 正值向下)
        lats: 1D (ny,) 可選
        lons: 1D (nx,) 可選

    Returns:
        {
            'tchp': 2D (ny, nx) — TCHP (kJ/cm²)
            'd26': 2D (ny, nx) — 26°C 等溫線深度 (m)
            'upper_ocean_heat': 2D — 上層海洋熱含量
            'alert_level': 2D int — 0=正常, 1=可能增強, 2=確定增強
            'rapid_intensification_zones': int — 可能快速增強的格點數
        }
    """
    n_depths, ny, nx = temp_3d.shape
    depths = np.abs(depths)  # 確保深度為正值

    # ── 計算 D26 (26°C 等溫線深度) ──
    d26 = _compute_d26(temp_3d, depths)

    # ── 計算 TCHP ──
    tchp = _integrate_tchp(temp_3d, depths, d26)

    # ── 警報等級 ──
    alert = np.zeros((ny, nx), dtype=np.int32)
    alert[tchp >= TCHP_RAPID_INTENSIFY] = 1
    alert[tchp >= TCHP_CERTAIN_INTENSIFY] = 2

    n_rapid = int(np.sum(alert >= 1))

    # ── 上層海洋熱含量 (0-100m) ──
    uohc = _upper_ocean_heat(temp_3d, depths, max_depth=100)

    log.info(
        f"  TCHP: range=[{np.nanmin(tchp):.1f}, {np.nanmax(tchp):.1f}] kJ/cm², "
        f"D26=[{np.nanmin(d26):.0f}, {np.nanmax(d26):.0f}]m, "
        f"rapid_intensification_zones={n_rapid}"
    )

    if n_rapid > 0:
        pct = n_rapid / (ny * nx) * 100
        log.warning(f"  ⚠️ TCHP Alert: {n_rapid} zones ({pct:.1f}%) exceeding "
                    f"{TCHP_RAPID_INTENSIFY} kJ/cm² (typhoon rapid intensification risk)")

    return {
        "tchp": tchp.astype(np.float32),
        "d26": d26.astype(np.float32),
        "upper_ocean_heat": uohc.astype(np.float32),
        "alert_level": alert,
        "rapid_intensification_zones": n_rapid,
    }


def _compute_d26(
    temp_3d: np.ndarray,
    depths: np.ndarray,
) -> np.ndarray:
    """
    計算 26°C 等溫線深度 (D26)。
    使用線性插值找出溫度穿越 26°C 的深度。
    """
    n_depths, ny, nx = temp_3d.shape
    d26 = np.full((ny, nx), np.nan, dtype=np.float32)

    for iy in range(ny):
        for ix in range(nx):
            profile = temp_3d[:, iy, ix]
            if np.all(np.isnan(profile)):
                continue

            # 表層溫度 < 26°C → D26 = 0
            if profile[0] < T_REF:
                d26[iy, ix] = 0.0
                continue

            # 找 26°C 穿越點
            found = False
            for k in range(1, n_depths):
                t_above = profile[k - 1]
                t_below = profile[k]
                if np.isnan(t_above) or np.isnan(t_below):
                    continue

                if t_above >= T_REF and t_below < T_REF:
                    # 線性插值
                    frac = (T_REF - t_above) / (t_below - t_above)
                    d26[iy, ix] = depths[k - 1] + frac * (depths[k] - depths[k - 1])
                    found = True
                    break

            # 整個剖面都 > 26°C → D26 = 最深深度
            if not found and not np.isnan(profile[-1]) and profile[-1] >= T_REF:
                d26[iy, ix] = depths[-1]

    return d26


def _integrate_tchp(
    temp_3d: np.ndarray,
    depths: np.ndarray,
    d26: np.ndarray,
) -> np.ndarray:
    """
    垂直積分 TCHP:
      TCHP = ρ·Cp · ∫₀^D26 (T(z) - 26) dz
    單位: J/m² → kJ/cm² (÷ 1e7)
    """
    n_depths, ny, nx = temp_3d.shape
    tchp = np.zeros((ny, nx), dtype=np.float64)

    for iy in range(ny):
        for ix in range(nx):
            d26_val = d26[iy, ix]
            if np.isnan(d26_val) or d26_val <= 0:
                continue

            profile = temp_3d[:, iy, ix]
            heat = 0.0

            for k in range(1, n_depths):
                z_top = depths[k - 1]
                z_bot = depths[k]

                if z_top >= d26_val:
                    break

                z_bot_eff = min(z_bot, d26_val)
                dz = z_bot_eff - z_top

                t_top = profile[k - 1]
                t_bot = profile[k]
                if np.isnan(t_top) or np.isnan(t_bot):
                    continue

                # 梯形積分
                dt_top = max(t_top - T_REF, 0)
                dt_bot = max(t_bot - T_REF, 0)
                heat += 0.5 * (dt_top + dt_bot) * dz

            # J/m² → kJ/cm²
            tchp[iy, ix] = RHO_SW * CP_SW * heat / 1e7

    return tchp


def _upper_ocean_heat(
    temp_3d: np.ndarray,
    depths: np.ndarray,
    max_depth: float = 100.0,
) -> np.ndarray:
    """上層海洋熱含量 (0 - max_depth m)"""
    n_depths, ny, nx = temp_3d.shape
    uohc = np.zeros((ny, nx), dtype=np.float64)

    for k in range(1, n_depths):
        if depths[k - 1] >= max_depth:
            break
        z_bot = min(depths[k], max_depth)
        dz = z_bot - depths[k - 1]

        t_top = temp_3d[k - 1]
        t_bot = temp_3d[k]
        dt = 0.5 * (np.nan_to_num(t_top, nan=20) + np.nan_to_num(t_bot, nan=20))
        uohc += RHO_SW * CP_SW * dt * dz / 1e7  # kJ/cm²

    return uohc.astype(np.float32)


def tchp_alert_message(tchp_result: Dict) -> Optional[str]:
    """
    產生 TCHP 警報訊息 (供 Dashboard / 報告)。
    """
    n_rapid = tchp_result.get("rapid_intensification_zones", 0)
    if n_rapid == 0:
        return None

    tchp = tchp_result["tchp"]
    max_tchp = float(np.nanmax(tchp))

    if max_tchp >= TCHP_CERTAIN_INTENSIFY:
        return (f"🔴 TCHP 極高警報: 最大值 {max_tchp:.1f} kJ/cm² > {TCHP_CERTAIN_INTENSIFY}，"
                f"颱風幾乎確定會快速增強。建議提前 48 小時撤離相關海域。")
    elif max_tchp >= TCHP_RAPID_INTENSIFY:
        return (f"🟡 TCHP 警告: 最大值 {max_tchp:.1f} kJ/cm² > {TCHP_RAPID_INTENSIFY}，"
                f"颱風有可能快速增強。密切監視天氣預報。")

    return None
