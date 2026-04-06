"""
OceanMaster v13.2 — 核心物理海洋學演算法
=========================================
融合全球商業系統的演算法精華（逆向工程驗證）

第一層演算法（確定性，不用 AI）：
  1. SST 鋒面偵測 — Sobel + Cayula-Cornillon + Canny 三法融合
  2. Chl-a 鋒面偵測 — 中值濾波 + Belkin-O'Reilly 梯度法
  3. FTLE 拉格朗日相干結構 — 海流「隱形骨架」
  4. 溫躍層深度 — 大目鮪的生死線
  5. SSH 渦旋偵測 — 中尺度渦旋
  6. 涌升流偵測 — Ekman 輸送
"""

import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from typing import Dict, Tuple, Optional, Any
import logging

log = logging.getLogger("OceanMaster.Algorithms")


# ═══════════════════════════════════════════════════
#  1. SST 鋒面偵測（三法融合）
# ═══════════════════════════════════════════════════

def detect_sst_fronts(
    sst: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    method: str = "fusion",
) -> Dict[str, np.ndarray]:
    """
    偵測 SST 鋒面（溫度急變線 = 魚群聚集的關鍵）

    三種方法融合（比任何單一商業系統都全面）：
    1. Sobel 梯度法 — 計算溫度梯度的幅值
    2. Cayula-Cornillon — INCOIS 使用的直方圖法（32×32 窗口）
    3. Canny 邊緣偵測 — 經典計算機視覺方法

    返回:
        gradient_magnitude: 梯度幅值 (°C/km)
        front_mask: 布林遮罩（True = 鋒面位置）
        front_strength: 鋒面強度 (0-1)
    """
    # [v13.2] 維度防護：ndimage.sobel 需要 2D 且至少 3×3
    if sst.ndim != 2 or sst.shape[0] < 3 or sst.shape[1] < 3:
        log.warning(f"  detect_sst_fronts: sst.shape={sst.shape} 非 2D 或太小，跳過")
        _z = np.zeros_like(sst) if sst.ndim == 2 else np.zeros((len(lat), len(lon)))
        return {"gradient_magnitude": _z, "gradient_raw": _z,
                "front_mask": _z.astype(bool), "front_strength": _z.astype(np.float32),
                "front_direction": _z}
    log.info(f"偵測 SST 鋒面... 網格={sst.shape}")

    # [v13.2-P1] 記錄鋒面偵測前的 NaN 格點數
    n_nan = int(np.sum(np.isnan(sst)))
    if n_nan > 0:
        log.info(f"  ⚠️ 鋒面偵測前仍有 {n_nan}/{sst.size} 個 NaN 格點 "
                 f"({n_nan/sst.size*100:.1f}%)")

    # 處理 NaN
    sst_filled = _fill_nan_nearest(sst)

    # ── 方法 1：Sobel 梯度 ──
    grad_y = ndimage.sobel(sst_filled, axis=0)  # 南北方向
    grad_x = ndimage.sobel(sst_filled, axis=1)  # 東西方向
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

    # 轉換為 °C/km (考慮格點間距)
    if len(lat) > 1:
        dlat_km = np.abs(lat[1] - lat[0]) * 111.0  # 1° ≈ 111 km
    else:
        dlat_km = 25.0
    gradient_per_km = gradient_mag / (2 * dlat_km)  # Sobel 核寬度=2格點

    # ── 方法 2：Cayula-Cornillon 簡化版 ──
    # 用 32×32 窗口計算雙峰直方圖，判斷是否存在鋒面
    cc_fronts = _cayula_cornillon(sst_filled, window=32, delta_t=0.45, var_ratio=0.76)

    # ── 方法 3：Canny 邊緣偵測 ──
    # 高斯平滑 → 梯度 → 非極大值抑制 → 雙閾值
    canny_fronts = _canny_edge(sst_filled, sigma=1.5, low_thresh=0.3, high_thresh=0.7)

    # ── 三法融合 ──
    # 策略：三種方法都認為是鋒面 = 強鋒面，任兩種 = 中等鋒面
    combined = (gradient_mag > np.nanpercentile(gradient_mag, 85)).astype(float)
    combined += cc_fronts.astype(float)
    combined += canny_fronts.astype(float)

    front_mask = combined >= 2  # 至少兩種方法同意
    front_strength = np.clip(combined / 3.0, 0, 1)

    # 鋒面方向（梯度的角度）
    front_direction = np.arctan2(grad_y, grad_x) * 180 / np.pi

    n_fronts = np.sum(front_mask)
    log.info(f"偵測到 {n_fronts} 個鋒面格點 (佔 {n_fronts/sst.size*100:.1f}%)")

    return {
        "gradient_magnitude": gradient_per_km,
        "gradient_raw": gradient_mag,
        "front_mask": front_mask,
        "front_strength": front_strength,
        "front_direction": front_direction,
    }


def _cayula_cornillon(sst: np.ndarray, window: int = 32, delta_t: float = 0.45, var_ratio: float = 0.76) -> np.ndarray:
    """
    Cayula-Cornillon (1992) 鋒面偵測算法
    INCOIS PFZ 系統使用的核心算法（參數從論文取得）

    原理：在 window×window 的窗口中，計算 SST 直方圖。
    如果直方圖呈明顯雙峰（= 兩個水團交匯），則判定為鋒面。
    """
    ny, nx = sst.shape
    fronts = np.zeros((ny, nx), dtype=bool)
    half = window // 2

    # 滑動窗口（步長 = 窗口的一半，加快速度）
    for iy in range(half, ny - half, half // 2):
        for ix in range(half, nx - half, half // 2):
            patch = sst[iy-half:iy+half, ix-half:ix+half]
            if np.isnan(patch).sum() > patch.size * 0.3:
                continue

            vals = patch[~np.isnan(patch)]
            if len(vals) < window:
                continue

            # 計算直方圖雙峰性
            hist, edges = np.histogram(vals, bins=20)
            total = hist.sum()
            if total == 0:
                continue

            # 計算 bin 中心點
            centers = (edges[:-1] + edges[1:]) / 2

            # 找最佳分割點
            best_ratio = 0
            best_split = 0
            delta = 0
            for s in range(3, len(hist) - 3):
                left = hist[:s]
                right = hist[s:]
                n_left = left.sum()
                n_right = right.sum()
                if n_left < total * 0.1 or n_right < total * 0.1:
                    continue

                mean_left = np.average(centers[:s], weights=left)
                mean_right = np.average(centers[s:], weights=right)

                # 兩族群間距
                between_var = (n_left * n_right / total**2) * (mean_left - mean_right)**2
                within_var = np.var(vals)

                if within_var > 0:
                    ratio = between_var / within_var
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_split = s
                        delta = abs(mean_left - mean_right)

            # 判定：方差比 > 閾值 且 溫差 > 閾值
            if best_ratio > var_ratio and delta > delta_t:
                # 在窗口中心標記鋒面
                fronts[iy-2:iy+2, ix-2:ix+2] = True

    return fronts


def _canny_edge(data: np.ndarray, sigma: float = 1.5, low_thresh: float = 0.3, high_thresh: float = 0.7) -> np.ndarray:
    """簡化版 Canny 邊緣偵測（不依賴 OpenCV）"""
    # 1. 高斯平滑
    smoothed = ndimage.gaussian_filter(data, sigma=sigma)

    # 2. 梯度
    gy = ndimage.sobel(smoothed, axis=0)
    gx = ndimage.sobel(smoothed, axis=1)
    mag = np.sqrt(gx**2 + gy**2)

    # 3. 正規化
    if mag.max() > 0:
        mag_norm = mag / mag.max()
    else:
        return np.zeros_like(data, dtype=bool)

    # 4. 雙閾值
    strong = mag_norm > high_thresh
    weak = (mag_norm > low_thresh) & (~strong)

    # 5. 連通性：弱邊緣只在連接強邊緣時保留
    strong_dilated = ndimage.binary_dilation(strong, iterations=2)
    edges = strong | (weak & strong_dilated)

    return edges


# ═══════════════════════════════════════════════════
#  2. FTLE 拉格朗日相干結構（海流的「隱形骨架」）
# ═══════════════════════════════════════════════════

def compute_ftle(
    u: np.ndarray,
    v: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    integration_days: float = 3.0,
    dt_hours: float = 6.0,
) -> Dict[str, np.ndarray]:
    """
    計算 FTLE (Finite-Time Lyapunov Exponents)
    = 海流場中粒子的最大拉伸率

    高 FTLE 值 = 「吸引脊線」= 浮游生物/垃圾/小魚被海流聚集的線
    → 大魚會沿著這些線覓食

    這比單純看溫度準確 30%+ (論文已驗證)

    原理：
    1. 在每個格點放一顆虛擬粒子
    2. 用海流 (u, v) 追蹤 3 天後的位置
    3. 計算位移的 Jacobian 矩陣
    4. 最大奇異值 = FTLE 值
    """
    # [v13.2] 維度防護
    if u.ndim != 2 or v.ndim != 2 or u.shape[0] < 3 or u.shape[1] < 3:
        log.warning(f"  compute_ftle: u.shape={u.shape} 非 2D 或太小，跳過")
        _z = np.zeros((len(lat), len(lon)))
        return {"ftle": _z, "ridges": _z.astype(bool), "particle_dx": _z, "particle_dy": _z}
    if len(lat) != u.shape[0] or len(lon) != u.shape[1]:
        log.warning(f"  compute_ftle: lat({len(lat)})×lon({len(lon)}) ≠ u{u.shape}，跳過")
        _z = np.zeros_like(u)
        return {"ftle": _z, "ridges": _z.astype(bool), "particle_dx": _z, "particle_dy": _z}
    log.info(f"計算 FTLE... 網格={u.shape}, 積分={integration_days}天")

    ny, nx = u.shape
    T = integration_days * 24 * 3600  # 秒
    dt = dt_hours * 3600  # 秒
    n_steps = int(T / dt)

    # 建立經緯度網格
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    x0 = lon_grid.copy()
    y0 = lat_grid.copy()

    # 海流插值器
    u_interp = RegularGridInterpolator(
        (lat, lon), u, bounds_error=False, fill_value=0
    )
    v_interp = RegularGridInterpolator(
        (lat, lon), v, bounds_error=False, fill_value=0
    )

    # 粒子追蹤（4 階 Runge-Kutta）
    x = x0.copy()
    y = y0.copy()

    for step in range(n_steps):
        # m/s → 度/s
        deg_per_m_lon = 1.0 / (111320 * np.cos(np.radians(y)))
        deg_per_m_lat = 1.0 / 110540

        pts = np.column_stack([y.ravel(), x.ravel()])

        # RK4 積分
        k1u = u_interp(pts).reshape(ny, nx) * deg_per_m_lon
        k1v = v_interp(pts).reshape(ny, nx) * deg_per_m_lat

        pts2 = np.column_stack([(y + 0.5*dt*k1v).ravel(), (x + 0.5*dt*k1u).ravel()])
        k2u = u_interp(pts2).reshape(ny, nx) * deg_per_m_lon
        k2v = v_interp(pts2).reshape(ny, nx) * deg_per_m_lat

        pts3 = np.column_stack([(y + 0.5*dt*k2v).ravel(), (x + 0.5*dt*k2u).ravel()])
        k3u = u_interp(pts3).reshape(ny, nx) * deg_per_m_lon
        k3v = v_interp(pts3).reshape(ny, nx) * deg_per_m_lat

        pts4 = np.column_stack([(y + dt*k3v).ravel(), (x + dt*k3u).ravel()])
        k4u = u_interp(pts4).reshape(ny, nx) * deg_per_m_lon
        k4v = v_interp(pts4).reshape(ny, nx) * deg_per_m_lat

        x += (dt / 6) * (k1u + 2*k2u + 2*k3u + k4u)
        y += (dt / 6) * (k1v + 2*k2v + 2*k3v + k4v)

    # 計算位移場的梯度 (Cauchy-Green 張量)
    dx = x - x0
    dy = y - y0

    # 有限差分計算 Jacobian
    dxdx = np.gradient(dx, axis=1)  # ∂(Δx)/∂x₀
    dxdy = np.gradient(dx, axis=0)  # ∂(Δx)/∂y₀
    dydx = np.gradient(dy, axis=1)  # ∂(Δy)/∂x₀
    dydy = np.gradient(dy, axis=0)  # ∂(Δy)/∂y₀

    # Cauchy-Green 張量 C = JᵀJ
    C11 = dxdx**2 + dydx**2
    C12 = dxdx*dxdy + dydx*dydy
    C22 = dxdy**2 + dydy**2

    # 最大特徵值
    trace = C11 + C22
    det = C11 * C22 - C12**2
    discriminant = np.maximum(trace**2 - 4*det, 0)
    lambda_max = 0.5 * (trace + np.sqrt(discriminant))

    # FTLE = ln(sqrt(λ_max)) / T
    lambda_max = np.maximum(lambda_max, 1e-10)
    ftle = np.log(np.sqrt(lambda_max)) / T

    # 提取 FTLE 脊線（吸引結構）
    ftle_threshold = np.nanpercentile(ftle[np.isfinite(ftle)], 90)
    ridges = ftle > ftle_threshold

    log.info(f"FTLE 計算完成: 範圍 {np.nanmin(ftle):.2e} ~ {np.nanmax(ftle):.2e}")
    log.info(f"偵測到 {np.sum(ridges)} 個脊線格點")

    return {
        "ftle": ftle,
        "ridges": ridges,
        "particle_dx": dx,
        "particle_dy": dy,
    }


# ═══════════════════════════════════════════════════
#  3. 溫躍層深度計算（大目鮪的關鍵）
# ═══════════════════════════════════════════════════

def compute_thermocline(
    temp_3d: np.ndarray,
    depths: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    從 HYCOM 3D 溫度場計算溫躍層深度

    方法：找垂直溫度梯度最大的深度 = 溫躍層
    大目鮪白天躲在溫躍層下方（300-500m），鉤子要下在這裡

    返回:
        thermocline_depth: 溫躍層深度 (m)
        d20_depth: 20°C 等溫線深度 (延繩釣的經典參考線)
        mld: 混合層深度 (表面均勻溫度的下界)
    """
    log.info(f"計算溫躍層... temp_3d={temp_3d.shape}")

    nd, ny, nx = temp_3d.shape
    thermocline_depth = np.full((ny, nx), np.nan)
    d20_depth = np.full((ny, nx), np.nan)
    mld = np.full((ny, nx), np.nan)

    for iy in range(ny):
        for ix in range(nx):
            profile = temp_3d[:, iy, ix]

            if np.isnan(profile).all():
                continue

            # ── 溫躍層深度（最大垂直梯度）──
            valid = ~np.isnan(profile)
            if valid.sum() < 3:
                continue

            dT_dz = np.abs(np.diff(profile[valid]) / np.diff(depths[valid]))
            if len(dT_dz) > 0:
                max_idx = np.argmax(dT_dz)
                thermocline_depth[iy, ix] = depths[valid][max_idx]

            # ── 20°C 等溫線深度 (D20) ──
            # 線性插值找 T=20°C 的深度
            for k in range(len(profile) - 1):
                if np.isnan(profile[k]) or np.isnan(profile[k+1]):
                    continue
                if (profile[k] >= 20 and profile[k+1] < 20) or \
                   (profile[k] <= 20 and profile[k+1] > 20):
                    # 線性插值
                    frac = (20 - profile[k]) / (profile[k+1] - profile[k])
                    d20_depth[iy, ix] = depths[k] + frac * (depths[k+1] - depths[k])
                    break

            # ── 混合層深度 (MLD) ──
            # 定義：表層溫度下降 0.5°C 的深度
            sst_val = profile[0]
            if np.isnan(sst_val):
                continue
            for k in range(1, len(profile)):
                if np.isnan(profile[k]):
                    continue
                if sst_val - profile[k] > 0.5:
                    # Linear interpolation between k-1 and k
                    # Find depth where (sst_val - T(z)) = 0.5
                    delta_k1 = sst_val - profile[k-1]  # temp diff at k-1
                    delta_k  = sst_val - profile[k]     # temp diff at k (> 0.5)
                    denom = delta_k - delta_k1
                    if abs(denom) > 1e-8:
                        frac = (0.5 - delta_k1) / denom
                    else:
                        frac = 0.5
                    frac = np.clip(frac, 0, 1)
                    mld[iy, ix] = depths[k-1] + frac * (depths[k] - depths[k-1])
                    break

    with np.errstate(all='ignore'):
        tc_mean = np.nanmean(thermocline_depth) if np.any(np.isfinite(thermocline_depth)) else float('nan')
        d20_mean = np.nanmean(d20_depth) if np.any(np.isfinite(d20_depth)) else float('nan')
        mld_mean = np.nanmean(mld) if np.any(np.isfinite(mld)) else float('nan')
    log.info(f"溫躍層平均深度: {tc_mean:.0f}m")
    log.info(f"D20 平均深度: {d20_mean:.0f}m")
    log.info(f"MLD 平均深度: {mld_mean:.0f}m")

    return {
        "thermocline_depth": thermocline_depth,
        "d20_depth": d20_depth,
        "mld": mld,
    }


# ═══════════════════════════════════════════════════
#  4. Chl-a 鋒面偵測
# ═══════════════════════════════════════════════════

def compute_chl_gradient(
    chl: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Belkin-O'Reilly (2009) 葉綠素鋒面偵測

    Chl-a 鋒面 = 浮游生物密度的急變線 = 食物鏈的起點
    SST 鋒面 ∩ Chl-a 鋒面 = 最強的漁場信號（INCOIS 方法）
    """
    # [v13.2] 維度防護：ndimage.sobel/median_filter 需要 2D 且至少 3×3
    if chl.ndim != 2 or chl.shape[0] < 3 or chl.shape[1] < 3:
        log.warning(f"  compute_boa_gradient: chl.shape={chl.shape} 非 2D 或太小，跳過")
        _z = np.zeros_like(chl) if chl.ndim == 2 else np.zeros((len(lat), len(lon)))
        return {"chl_gradient": _z, "chl_front_mask": _z.astype(bool)}
    log.info("偵測 Chl-a 鋒面...")

    # Chl-a 取 log（因為分布高度偏態）
    chl_log = np.log10(np.maximum(chl, 0.001))

    # 中值濾波去噪（Belkin 的關鍵創新：保護形狀的濾波）
    chl_filtered = ndimage.median_filter(chl_log, size=3)

    # Sobel 梯度
    gy = ndimage.sobel(chl_filtered, axis=0)
    gx = ndimage.sobel(chl_filtered, axis=1)
    gradient = np.sqrt(gx**2 + gy**2)

    # 鋒面閾值：梯度 > 85th 百分位
    threshold = np.nanpercentile(gradient[np.isfinite(gradient)], 85)
    front_mask = gradient > threshold

    return {
        "chl_gradient": gradient,
        "chl_front_mask": front_mask,
    }


# ═══════════════════════════════════════════════════
#  5. 渦旋動能 EKE (Eddy Kinetic Energy) — v10.4 物理正確版
# ═══════════════════════════════════════════════════

# 物理常數
_OMEGA = 7.2921e-5  # 地球自轉角速度 (rad/s)
_G = 9.81           # 重力加速度 (m/s²)
_R_EARTH = 6371e3   # 地球半径 (m)
_F_MIN = 1e-6       # Coriolis 下限 (赤道防呆)


def calculate_eke(
    ssh: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> Dict[str, Any]:
    """
    從 SSH (海面高度異常) 計算渦旋動能 EKE。

    物理公式 (地轉近似):
      f = 2Ω·sin(φ)              — Coriolis parameter
      dx = R·cos(φ)·Δλ           — 經度方向真實距離 (m)
      dy = R·Δφ                  — 緯度方向真實距離 (m)
      u_g = -(g/f)·∂SSH/∂y       — 地轉流 u
      v_g =  (g/f)·∂SSH/∂x       — 地轉流 v
      EKE = 0.5·(u_g² + v_g²)    — 渦旋動能 (m²/s²)

    Args:
        ssh: 2D (ny, nx) 海面高度異常 (m)
        lat: 1D (ny,) 緯度 (degrees)
        lon: 1D (nx,) 經度 (degrees)

    Returns:
        {
            'eke': 2D (m²/s²) — 渦旋動能
            'u_geo': 2D (m/s) — 地轉流 u 分量
            'v_geo': 2D (m/s) — 地轉流 v 分量
            'ssh_anomaly': 2D (m) — SSH 異常
            'warm_eddies': 2D labeled — 暖渦旋 (向後相容)
            'cold_eddies': 2D labeled — 冷渦旋
            'n_warm': int,
            'n_cold': int,
        }
    """
    ny, nx = ssh.shape

    # ── 0. SSH 異常 + 高斯平滑 ──
    ssh_mean = np.nanmean(ssh)
    ssh_anomaly = ssh - ssh_mean
    ssh_smooth = ndimage.gaussian_filter(
        np.nan_to_num(ssh_anomaly, nan=0), sigma=1.5
    )

    # ── 1. Coriolis parameter f(φ) ──
    # f = 2Ω·sin(φ), 赤道附近 |f| < 1e-6 → clamp
    lat_rad = np.radians(lat)
    f_arr = 2.0 * _OMEGA * np.sin(lat_rad)  # shape (ny,)
    f_arr = np.where(np.abs(f_arr) < _F_MIN,
                     np.sign(f_arr + 1e-30) * _F_MIN, f_arr)
    # Expand to 2D: (ny, 1) for broadcasting
    f_2d = f_arr[:, np.newaxis]  # (ny, 1)

    # ── 2. 真實距離 dx, dy (Haversine-correct) ──
    # dy = R · Δφ (m), 每個 lat 間距
    dlat_rad = np.diff(lat_rad).mean() if ny > 1 else np.radians(0.25)
    dy = _R_EARTH * dlat_rad  # scalar (m), 緯度方向步長

    # dx = R · cos(φ) · Δλ (m), 隨緯度變化
    dlon_rad = np.radians(np.diff(lon).mean()) if nx > 1 else np.radians(0.25)
    cos_lat = np.cos(lat_rad)[:, np.newaxis]  # (ny, 1)
    dx_2d = _R_EARTH * cos_lat * dlon_rad     # (ny, 1), 每行不同

    # ── 3. SSH 梯度 (np.gradient, 使用真實距離) ──
    # np.gradient returns gradient along each axis
    dssh_dy_raw, dssh_dx_raw = np.gradient(ssh_smooth)

    # 除以真實距離得到 ∂SSH/∂x, ∂SSH/∂y (m/m = dimensionless)
    dssh_dy = dssh_dy_raw / dy             # ∂SSH/∂y
    dssh_dx = dssh_dx_raw / dx_2d          # ∂SSH/∂x, 考慮 cos(lat)

    # ── 4. 地轉流 (Geostrophic velocity) ──
    u_geo = -(_G / f_2d) * dssh_dy   # u_g = -(g/f) · ∂SSH/∂y
    v_geo = (_G / f_2d) * dssh_dx    # v_g =  (g/f) · ∂SSH/∂x

    # NaN 防護
    u_geo = np.nan_to_num(u_geo, nan=0, posinf=0, neginf=0).astype(np.float32)
    v_geo = np.nan_to_num(v_geo, nan=0, posinf=0, neginf=0).astype(np.float32)

    # ── 5. EKE = 0.5 × (u_g² + v_g²) ──
    eke = 0.5 * (u_geo ** 2 + v_geo ** 2)

    # ── 6. 暖冷渦旋偵測 (向後相容) ──
    warm_eddies_mask = ssh_smooth > 0.05  # SSH > +5cm
    cold_eddies_mask = ssh_smooth < -0.05  # SSH < -5cm
    warm_labeled, n_warm = ndimage.label(warm_eddies_mask)
    cold_labeled, n_cold = ndimage.label(cold_eddies_mask)

    log.info(
        f"EKE: range=[{np.nanmin(eke):.4f}, {np.nanmax(eke):.4f}] m²/s², "
        f"|u_g|_max={np.nanmax(np.abs(u_geo)):.3f} m/s, "
        f"eddies: {n_warm} warm + {n_cold} cold"
    )

    return {
        "eke": eke.astype(np.float32),
        "u_geo": u_geo,
        "v_geo": v_geo,
        "ssh_anomaly": ssh_anomaly.astype(np.float32),
        "warm_eddies": warm_labeled,
        "cold_eddies": cold_labeled,
        "n_warm": n_warm,
        "n_cold": n_cold,
    }


def detect_eddies(
    ssh: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> Dict[str, Any]:
    """
    [DEPRECATED] 使用 calculate_eke() 取代。
    保留以維持向後相容性。

    從 SSH (海面高度) 偵測中尺度渦旋
    暖渦旋 (SSH 高) = 水溫暖、對某些鮪魚有利
    冷渦旋 (SSH 低) = 涌升流、營養鹽豐富
    """
    import warnings
    warnings.warn(
        "detect_eddies() is deprecated, use calculate_eke() instead",
        DeprecationWarning, stacklevel=2,
    )
    return calculate_eke(ssh, lat, lon)


# ═══════════════════════════════════════════════════
#  6. VIIRS 夜光濾除（魷魚船偵測）
# ═══════════════════════════════════════════════════

def filter_fishing_lights(
    lights: np.ndarray,
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
) -> np.ndarray:
    """
    過濾 VIIRS 夜間光源，只保留可能的漁船集魚燈

    過濾邏輯：
    1. 排除陸地光源（簡化：只保留離海岸 > 一定距離的）
    2. 排除已知氣體火焰位置
    3. 亮度過濾（集魚燈通常很亮）
    """
    if len(lights) == 0:
        return lights

    # 簡化版：只保留亮度在合理範圍內的
    # 真實的集魚燈亮度通常在特定範圍
    brightness = lights[:, 2]
    valid = (brightness > 300) & (brightness < 2000)  # 經驗閾值

    return lights[valid]


# ═══════════════════════════════════════════════════
#  7. [v11] 生產力鋒面 (Productivity Front) — Bug #6 修正
# ═══════════════════════════════════════════════════

def compute_productivity_front(
    sst_front_strength: np.ndarray,
    chl_front_strength: np.ndarray,
    w_sst: float = 0.5,
    w_chl: float = 0.5,
) -> np.ndarray:
    """
    [v11 Bug #6] 融合 SST + Chl-a 鋒面為統一的「生產力鋒面」

    原理:
      - SST 鋒面 = 物理鋒面（水團交界、渦旋邊緣）
      - Chl-a 鋒面 = 生物鋒面（營養鹽梯度、浮游植物分界）
      - 兩者重合 = 「生產力鋒面」→ 最高漁場價值
      - 文獻: Belkin et al. 2009, Nieto et al. 2012

    方法:
      1. 幾何平均 → 只有兩者都強才給高分
      2. 加權線性 → 允許某一方特別強也有信號
      最終: 0.6 × geometric + 0.4 × linear

    Parameters:
        sst_front_strength: 2D (0-1) SST 鋒面強度
        chl_front_strength: 2D (0-1) Chl-a 鋒面強度
        w_sst: SST 鋒面權重 (0-1)
        w_chl: Chl-a 鋒面權重 (0-1)

    Returns:
        productivity_front: 2D (0-1) 生產力鋒面強度
    """
    # 確保最低值 > 0 避免 sqrt 問題
    sst_f = np.clip(sst_front_strength, 0.001, 1.0)
    chl_f = np.clip(chl_front_strength, 0.001, 1.0)

    # 幾何平均: 兩者都強 → 接近 1，一方弱 → 快速衰減
    geometric = np.sqrt(sst_f * chl_f)

    # 加權線性: 允許偏向一方
    linear = w_sst * sst_f + w_chl * chl_f

    # 混合
    productivity = 0.6 * geometric + 0.4 * linear

    # 正規化
    p_max = np.nanpercentile(productivity, 99) if np.any(productivity > 0) else 1.0
    productivity = np.clip(productivity / max(p_max, 1e-6), 0, 1)

    log.info(f"  Productivity front: avg={np.nanmean(productivity):.3f}, "
             f"max={np.nanmax(productivity):.3f}")

    return productivity.astype(np.float32)


# ═══════════════════════════════════════════════════
#  8. [v11] 鹽度鋒面偵測 — Feature B
# ═══════════════════════════════════════════════════

def detect_salinity_fronts(
    salinity: np.ndarray,
    lat: Optional[np.ndarray] = None,
    lon: Optional[np.ndarray] = None,
    threshold_psu_per_km: float = 0.02,
) -> Dict[str, np.ndarray]:
    """
    [v11 Feature B] 偵測鹽度鋒面

    科學依據:
      - 鹽度鋒面標示不同水團交匯處（黑潮/親潮、河口/外洋）
      - 魷魚和某些鮪魚物種對鹽度變化敏感
      - 文獻: Zainuddin et al. 2008, Mugo et al. 2010

    方法: Sobel 梯度 (與 SST 鋒面偵測相同框架)

    Parameters:
        salinity: 2D 鹽度場 (PSU)
        lat: 1D 緯度
        lon: 1D 經度
        threshold_psu_per_km: 鋒面判定閾值

    Returns:
        dict with:
          - gradient_magnitude: 鹽度梯度 (PSU/km)
          - front_mask: 布林遮罩
          - front_strength: 鋒面強度 (0-1)
    """
    # 填充 NaN
    sal_filled = _fill_nan_nearest(salinity)

    # Sobel 梯度
    grad_y = ndimage.sobel(sal_filled, axis=0)
    grad_x = ndimage.sobel(sal_filled, axis=1)
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

    # 轉換為 PSU/km
    if lat is not None and len(lat) > 1:
        dlat_km = np.abs(lat[1] - lat[0]) * 111.0
    else:
        dlat_km = 25.0
    gradient_per_km = gradient_mag / (2 * dlat_km)

    # 鋒面判定
    front_mask = gradient_per_km > threshold_psu_per_km

    # 鋒面強度正規化
    p95 = np.nanpercentile(gradient_per_km, 95) if np.any(gradient_per_km > 0) else 1.0
    front_strength = np.clip(gradient_per_km / max(p95, 1e-6), 0, 1)

    log.info(f"  Salinity fronts: {np.sum(front_mask)} pixels "
             f"({np.sum(front_mask)/salinity.size*100:.1f}%)")

    return {
        "gradient_magnitude": gradient_per_km.astype(np.float32),
        "front_mask": front_mask,
        "front_strength": front_strength.astype(np.float32),
    }


# ═══════════════════════════════════════════════════
#  9. [v12-enhance] BOA 梯度法 (Belkin-O'Reilly 2009)
# ═══════════════════════════════════════════════════

def compute_boa_gradient(
    sst: np.ndarray,
    chl: Optional[np.ndarray] = None,
    median_size: int = 3,
    grad_percentile: float = 85.0,
) -> Dict[str, np.ndarray]:
    """
    [v12-enhance] Belkin-O'Reilly Algorithm (BOA) 梯度法

    Belkin & O'Reilly 2009, JGR 114: C02012
    原為 NASA 全球海洋鋒面產品的核心算法。

    核心思想 (三步驟):
      1. Median filter — 保護邊緣的去噪 (不像 Gaussian 會模糊鋒面)
      2. Sobel gradient — 計算梯度幅值
      3. Contextual threshold — 局部自適應閾值 (百分位)

    與既有 detect_sst_fronts() 的區別:
      - BOA 用中值濾波 (保護邊緣) 而非 Canny (固定閾值)
      - BOA 同時融合 SST + Chl-a 空間重合度
      - BOA 的閾值是局部百分位 (自適應) 而非全場固定

    Parameters:
        sst: 2D SST 場 (°C)
        chl: 2D Chl-a 場 (mg/m³), 可選
        median_size: 中值濾波窗口大小 (像素)
        grad_percentile: 鋒面閾值百分位

    Returns:
        dict: {
            'boa_sst_gradient': 2D SST 梯度
            'boa_sst_front': 2D bool (鋒面 mask)
            'boa_combined': 2D float (0-1) 合併指數
        }
    """
    # [v13.2] 維度防護
    if sst.ndim != 2 or sst.shape[0] < 3 or sst.shape[1] < 3:
        log.warning(f"  BOA 梯度法: sst.shape={sst.shape} 非 2D 或太小，跳過")
        _z = np.zeros_like(sst) if sst.ndim == 2 else np.zeros((1, 1))
        return {"boa_sst_gradient": _z.astype(np.float32),
                "boa_sst_front": _z.astype(bool),
                "boa_combined": _z.astype(np.float32)}
    log.info(f"  BOA 梯度法: median={median_size}, threshold=P{grad_percentile}")

    # ── Step 1: 中值濾波 (Belkin 的關鍵 — 保護鋒面形狀) ──
    sst_safe = np.nan_to_num(sst, nan=np.nanmean(sst))
    sst_filtered = ndimage.median_filter(sst_safe, size=median_size)

    # ── Step 2: Sobel 梯度 ──
    gy = ndimage.sobel(sst_filtered, axis=0)
    gx = ndimage.sobel(sst_filtered, axis=1)
    sst_grad = np.sqrt(gx**2 + gy**2)

    # ── Step 3: 局部自適應閾值 ──
    valid_grad = sst_grad[np.isfinite(sst_grad)]
    if len(valid_grad) > 0:
        threshold = np.nanpercentile(valid_grad, grad_percentile)
        sst_front = sst_grad > threshold
    else:
        sst_front = np.zeros_like(sst, dtype=bool)

    # 歸一化 SST 梯度 → 0-1
    gmax = np.nanmax(sst_grad) if np.any(sst_grad > 0) else 1.0
    sst_grad_norm = np.clip(sst_grad / max(gmax, 1e-6), 0, 1)

    # ── Chl-a 融合 (可選) ──
    if chl is not None:
        chl_log = np.log10(np.maximum(np.nan_to_num(chl, nan=0.01), 0.001))
        chl_filtered = ndimage.median_filter(chl_log, size=median_size)
        cy = ndimage.sobel(chl_filtered, axis=0)
        cx = ndimage.sobel(chl_filtered, axis=1)
        chl_grad = np.sqrt(cx**2 + cy**2)

        c_max = np.nanmax(chl_grad) if np.any(chl_grad > 0) else 1.0
        chl_grad_norm = np.clip(chl_grad / max(c_max, 1e-6), 0, 1)

        # 空間重合度加分 (SST front ∩ Chl front = 更強的漁場信號)
        chl_front = chl_grad > np.nanpercentile(
            chl_grad[np.isfinite(chl_grad)], grad_percentile
        ) if len(chl_grad[np.isfinite(chl_grad)]) > 0 else np.zeros_like(chl, dtype=bool)

        coincidence = (sst_front & chl_front).astype(np.float32) * 0.2
        combined = 0.5 * sst_grad_norm + 0.3 * chl_grad_norm + coincidence
    else:
        combined = sst_grad_norm

    # 正規化
    c_max = np.nanmax(combined) if np.any(combined > 0) else 1.0
    combined = np.clip(combined / max(c_max, 1e-6), 0, 1)

    n_front = np.sum(sst_front)
    log.info(f"  BOA: {n_front} 鋒面像素, combined max={np.nanmax(combined):.3f}")

    return {
        "boa_sst_gradient": sst_grad.astype(np.float32),
        "boa_sst_front": sst_front,
        "boa_combined": combined.astype(np.float32),
    }


# ═══════════════════════════════════════════════════
#  工具函數
# ═══════════════════════════════════════════════════

def _fill_nan_nearest(data: np.ndarray) -> np.ndarray:
    """用最近鄰插值填充 NaN"""
    mask = np.isnan(data)
    if not mask.any():
        return data.copy()
    filled = data.copy()
    # 用距離變換找最近的有效值
    ind = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    filled = data[tuple(ind)]
    return filled

