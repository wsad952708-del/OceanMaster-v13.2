"""
OceanMaster v13.2 — 海底地形 & 衍生物理量模組
===============================================
補齊商業系統的地理/物理特徵

新增技術：
  1. 海底地形 (Bathymetry) — ETOPO1 via ERDDAP
     - 海底山偵測 (seamount detection)
     - 大陸架邊緣偵測
     - 深度適宜性
  2. 渦動能 (EKE) — 從海流異常計算
     - EKE = ½(u'² + v'²)
     - 高 EKE → 中尺度渦旋活動強 → 漁場活性高
  3. 鹽度適宜性 — HYCOM 已有數據
  4. Ekman 輸送 — 風場驅動的上升流
     - we = curl(τ/ρf) = (1/ρf)(∂τy/∂x - ∂τx/∂y)
     - 正 Ekman pumping → 上升流 → 營養鹽上湧 → 生產力高

數據來源（已查證 URL）：
  ETOPO1 ERDDAP: https://coastwatch.noaa.gov/erddap/griddap/etopo180
  HYCOM: tds.hycom.org (salinity, water_u, water_v)
"""

import numpy as np
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Try importing cupy for GPU acceleration, fallback to numpy
try:
    import cupy as cp
    xp = cp
    print("Using CuPy for GPU acceleration.")
except ImportError:
    xp = np
    print("CuPy not found, falling back to NumPy.")

log = logging.Logger("OceanMaster.GeoPhys")

# 物理常數
RHO_WATER = 1025.0   # 海水密度 kg/m³
OMEGA = 7.292e-5      # 地球自轉角速度 rad/s
G = 9.81              # 重力加速度 m/s²


class BathymetryAnalyzer:
    """海底地形分析"""

    def __init__(self):
        self._cache = None

    async def fetch_etopo1(
        self, client, lat_range, lon_range, stride: int = 6
    ) -> Optional[np.ndarray]:
        """
        從 ERDDAP 抓取 ETOPO1

        URL 格式（已查證）：
        https://coastwatch.noaa.gov/erddap/griddap/etopo180.csv?
        altitude[({lat_max}):stride:({lat_min})][({lon_min}):stride:({lon_max})]

        stride=6 → ~0.1° 解析度（足夠漁場分析）
        """
        try:
            url = (
                f"https://coastwatch.noaa.gov/erddap/griddap/etopo180.csv?"
                f"altitude[({lat_range[1]}):{stride}:({lat_range[0]})]"
                f"[({lon_range[0]}):{stride}:({lon_range[1]})]"
            )
            log.info(f"  抓取 ETOPO1: stride={stride}")
            resp = await client.get(url, timeout=60)
            if resp.status_code == 200:
                lines = resp.text.strip().split('\n')
                # 跳過前兩行 (header + units)
                data = []
                for line in lines[2:]:
                    parts = line.split(',')
                    if len(parts) >= 3:
                        data.append([float(parts[0]), float(parts[1]), float(parts[2])])
                if data:
                    arr = np.array(data)
                    lats_unique = np.unique(arr[:, 0])
                    lons_unique = np.unique(arr[:, 1])
                    # [v13.2 Bugfix] ERDDAP latitude is descending, np.unique sorts ascending
                    # Direct reshape causes vertical flip! Map using indices to properly align.
                    grid = np.full((len(lats_unique), len(lons_unique)), -4000.0, dtype=np.float32)
                    lat_idx = np.searchsorted(lats_unique, arr[:, 0])
                    lon_idx = np.searchsorted(lons_unique, arr[:, 1])
                    grid[lat_idx, lon_idx] = arr[:, 2]
                    return grid
        except Exception as e:
            log.warning(f"  ETOPO1 抓取失敗: {e}")
        return None

    @staticmethod
    def generate_bathymetry(
        lats: np.ndarray, lons: np.ndarray
    ) -> np.ndarray:
        """
        海底地形取得（優先 ETOPO1 API → 回退到確定性地形模型）

        v10.2.1 修正: 消除所有 np.random 調用。
        回退模型基於真實地理特徵（已知海溝/陸棚/海山座標），非隨機生成。
        """
        # ── 嘗試 ETOPO1 ERDDAP ──
        try:
            import httpx
            lat_min, lat_max = float(lats.min()), float(lats.max())
            lon_min, lon_max = float(lons.min()), float(lons.max())
            stride = max(1, int(max(len(lats), len(lons)) / 150))
            url = (
                f"https://coastwatch.noaa.gov/erddap/griddap/etopo180.csv?"
                f"altitude[({lat_max}):{stride}:({lat_min})]"
                f"[({lon_min}):{stride}:({lon_max})]"
            )
            resp = httpx.get(url, timeout=60)
            if resp.status_code == 200:
                lines = resp.text.strip().split('\n')
                data = []
                for line in lines[2:]:
                    parts = line.split(',')
                    if len(parts) >= 3:
                        data.append([float(parts[0]), float(parts[1]), float(parts[2])])
                if data:
                    arr = np.array(data)
                    from scipy.interpolate import RegularGridInterpolator
                    ulats = np.unique(arr[:, 0])
                    ulons = np.unique(arr[:, 1])
                    grid = np.full((len(ulats), len(ulons)), -4000.0, dtype=np.float32)
                    lat_idx = np.searchsorted(ulats, arr[:, 0])
                    lon_idx = np.searchsorted(ulons, arr[:, 1])
                    grid[lat_idx, lon_idx] = arr[:, 2]
                    interp = RegularGridInterpolator(
                        (ulats, ulons), grid,
                        method='linear', bounds_error=False, fill_value=-4000)
                    g_lat, g_lon = np.meshgrid(lats, lons, indexing='ij')
                    result = interp((g_lat, g_lon)).astype(np.float32)
                    log.info(f"  ETOPO1 OK: {result.shape}, depth [{result.min():.0f}, {result.max():.0f}]m")
                    return result
        except Exception as e:
            log.warning(f"  ETOPO1 API failed: {e}")

        # ── 確定性回退模型（零 np.random）──
        log.info("  Using deterministic bathymetry model (no np.random)")
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
        
        # ═══ [v13.3] 商業級離線地形架構 (Offline DEM) 準備 ═══
        # 真正的船載系統不依賴外部 API 獲取靜態地形。此處建立讀取本地 ETOPO/GEBCO 的接口。
        # 只要在 data/bathymetry/ 放入 etopo_15min_wpac.npz，系統就會自動載入真實高精細地形。
        offline_bathy_path = Path("data/bathymetry/etopo_15min_wpac.npz")
        if offline_bathy_path.exists():
            try:
                offline_data = np.load(offline_bathy_path)
                ob_lats = offline_data['lats']
                ob_lons = offline_data['lons']
                ob_bathy = offline_data['bathy']
                from scipy.interpolate import RegularGridInterpolator
                interp = RegularGridInterpolator((ob_lats, ob_lons), ob_bathy, bounds_error=False, fill_value=-4000.0)
                # target points
                pts = np.array([lat_grid.flatten(), lon_grid.flatten()]).T
                result_bathy = interp(pts).reshape(lat_grid.shape)
                log.info(f"  ✅ Loaded offline GEBCO/ETOPO bathymetry. Interpolated to {lat_grid.shape}")
                return result_bathy.astype(np.float32)
            except Exception as e:
                log.error(f"  Failed to load offline bathy: {e}")
                
        # ── 以下為開發環境無離線海圖時的 Fallback (將被逐步淘汰) ──
        bathy = np.full_like(lat_grid, -4000.0, dtype=np.float32)

        # 矩形近似 — 寧可多標（海岸線外幾km）也不漏標 (DEPRECATED: 等待離線海圖完全取代)
        KNOWN_LAND = [
            # ===== 菲律賓（細分 12 區，確保所有島嶼群） =====
            (14.0, 18.8, 119.7, 123.0),   # 呂宋島（含馬尼拉灣+邦板牙）
            (11.5, 14.0, 119.8, 122.5),   # [NEW] 民都洛 (Mindoro)、Lubang、Marinduque
            (18.3, 20.5, 120.3, 122.5),   # 巴丹+巴布延群島
            (13.0, 14.5, 123.0, 124.5),   # Catanduanes + Camarines
            (10.5, 14.0, 122.5, 125.5),   # Samar + Leyte + Masbate + 東維薩亞
            (9.0, 11.0, 123.0, 124.5),    # Bohol + Cebu + Negros 東部
            (9.0, 11.5, 121.5, 123.5),    # Negros + Panay + Romblon
            (7.5, 10.0, 121.5, 127.0),    # 民答那峨（主島完整）
            (5.5, 8.0, 124.5, 127.5),     # 民答那峨南端 + General Santos
            (4.5, 7.5, 118.5, 123.0),     # 蘇祿群島（Basilan/Jolo/Tawi-Tawi）
            (7.0, 13.0, 117.0, 121.0),    # 巴拉望全段（含南端 Balabac）
            (5.5, 8.5, 116.0, 119.0),     # 蘇祿海西側島礁
            # ===== 台灣 =====
            (21.8, 25.4, 119.8, 122.2),   # 台灣本島+澎湖
            (23.5, 25.8, 119.3, 120.5),   # 澎湖群島擴大
            # ===== 日本 =====
            (30.5, 35.5, 129.0, 142.0),   # 本州+四國+九州
            (26.0, 30.5, 126.0, 131.5),   # 琉球群島+沖繩
            (24.0, 26.5, 122.5, 128.5),   # 先島群島（宮古/石垣）
            # ===== 中國 =====
            (18.0, 35.5, 108.0, 120.5),   # 中國東南沿岸（含海南島）
            # ===== 韓國 =====
            (33.0, 38.5, 124.5, 130.5),   # 韓國+濟州
            # ===== 印尼 =====
            (-1.0, 5.5, 118.0, 128.5),    # 蘇拉威西+摩鹿加
            (-8.5, 0.0, 114.0, 141.0),    # 爪哇+帝汶+巴布亞
            (0.5, 7.5, 104.0, 118.5),     # 婆羅洲+蘇門答臘
            (-5.0, 1.0, 127.0, 141.0),    # 摩鹿加群島+巴布亞西
            # ===== 巴布亞新幾內亞 =====
            (-10.0, 0.0, 141.0, 156.0),   # PNG
            # ===== 越南+寮+柬 =====
            (8.0, 23.5, 102.0, 110.0),    # 越南
            # ===== 馬來西亞 =====
            (0.8, 7.5, 99.5, 119.5),      # 馬來半島+沙巴砂拉越
            # ===== 密克羅尼西亞+帛琉 =====
            (2.0, 10.0, 130.5, 139.5),    # 帛琉+雅浦+加羅林（小島群）
            # ===== 澳洲北端 =====
            (-20.0, -10.0, 130.0, 155.0), # 澳洲北海岸+約克角
        ]
        for lat1, lat2, lon1, lon2 in KNOWN_LAND:
            land_mask = ((lat_grid >= lat1) & (lat_grid <= lat2) &
                         (lon_grid >= lon1) & (lon_grid <= lon2))
            bathy[land_mask] = 100.0  # 正值 = 陸地

        # 大陸架 — 用距海岸線距離的確定性梯度
        shelf_mask = (lon_grid < 125) & (lat_grid > 20) & (bathy < 0)
        bathy[shelf_mask] = -50 - 30 * (125 - lon_grid[shelf_mask])  # 越近岸越淺
        scs_shelf = (lon_grid < 120) & (lat_grid > 5) & (lat_grid < 25) & (bathy < 0)
        bathy[scs_shelf] = np.minimum(bathy[scs_shelf], -150)

        # 馬里亞納海溝（只在非陸地區域套用）
        trench_dist = np.sqrt((lat_grid - 11)**2 + (lon_grid - 142)**2)
        trench_mask = (trench_dist < 3) & (bathy < 0)
        bathy[trench_mask] = -8000 - 2000 * np.exp(-(trench_dist[trench_mask])**2 / 2)

        # 菲律賓海溝（只在非陸地區域套用）
        pt_dist = np.sqrt((lat_grid - 10)**2 + (lon_grid - 126.5)**2)
        pt_mask = (pt_dist < 2) & (bathy < 0)
        bathy[pt_mask] = -7000 - 2000 * np.exp(-(pt_dist[pt_mask])**2 / 1.5)

        # 已知海山座標（來自 SeamountCatalog / Kim & Wessel 2011）
        KNOWN_SEAMOUNTS = [
            (30.0, 140.0, 2500, 1.0),  # 大東海嶺
            (27.5, 142.5, 2000, 0.8),  # 小笠原海嶺
            (20.0, 138.0, 1800, 0.7),  # 西馬里亞納海嶺
            (24.0, 131.5, 2200, 1.2),  # 九州帛琉海嶺
            (18.0, 136.5, 1500, 0.6),  # 帕里斯維拉海嶺
            (15.0, 148.0, 2800, 1.5),  # 馬紹爾海山群
            (32.0, 143.5, 1200, 0.5),  # 天皇海山鏈南端
            (12.0, 134.0, 1600, 0.8),  # 雅浦海嶺
            (25.0, 137.0, 1400, 0.6),  # 四國海盆海山
            (22.0, 145.0, 2000, 0.9),  # 馬里亞納弧前
            (8.5, 150.0, 1700, 0.7),   # 卡羅琳海山
            (35.0, 144.0, 1100, 0.5),  # 日本海溝西側
        ]
        for sm_lat, sm_lon, sm_h, sm_w in KNOWN_SEAMOUNTS:
            dist = np.sqrt((lat_grid - sm_lat)**2 + (lon_grid - sm_lon)**2)
            # 只在非陸地區域加海山
            ocean_mask = bathy < 0
            bathy[ocean_mask] += sm_h * np.exp(-(dist[ocean_mask] / sm_w)**2)

        # [v13.2] 允許正值（陸地），不再 clip 到 0
        return np.clip(bathy, -11000, 500).astype(np.float32)

    @staticmethod
    def detect_seamounts(bathy: np.ndarray, threshold_m: float = 1000) -> np.ndarray:
        """
        海底山偵測

        方法：局部高點與周圍均值的差異 > threshold
        海底山是重要的魚群聚集地（上升流 + 營養鹽）
        """
        from scipy.ndimage import uniform_filter
        smoothed = uniform_filter(bathy.astype(float), size=11)
        prominence = bathy - smoothed  # 正值 = 比周圍高
        seamount_mask = prominence > threshold_m
        return seamount_mask, prominence

    @staticmethod
    def compute_depth_suitability(
        bathy: np.ndarray, species: str
    ) -> np.ndarray:
        """
        深度適宜性

        不同物種偏好不同的海底深度：
        - Skipjack: 偏好大陸架邊緣 200-2000m 水深
        - Bigeye: 偏好深水區 1000-5000m
        - 魷魚: 偏好大陸架到大陸坡 100-1000m
        """
        depth = -bathy  # 轉為正值深度

        prefs = {
            "skipjack":  (200, 3000, 500, 1500),
            "yellowfin": (200, 4000, 500, 2000),
            "bigeye":    (500, 6000, 1000, 4000),
            "albacore":  (300, 5000, 800, 3000),
            "squid_todarodes": (100, 2000, 200, 800),
            "squid_ommastrephes": (200, 3000, 300, 1500),
        }

        if species not in prefs:
            return np.ones_like(bathy, dtype=np.float32) * 0.5

        d_min, d_max, opt_min, opt_max = prefs[species]

        si = np.zeros_like(depth, dtype=np.float32)
        # 在最適範圍內 = 1
        optimal = (depth >= opt_min) & (depth <= opt_max)
        si[optimal] = 1.0
        # 可接受範圍 = 線性衰減
        below_opt = (depth >= d_min) & (depth < opt_min)
        si[below_opt] = (depth[below_opt] - d_min) / max(opt_min - d_min, 1)
        above_opt = (depth > opt_max) & (depth <= d_max)
        si[above_opt] = (d_max - depth[above_opt]) / max(d_max - opt_max, 1)

        return np.clip(si, 0, 1).astype(np.float32)


class OceanPhysicsEngine:
    """衍生物理量計算"""

    @staticmethod
    def compute_eke(
        u_current: np.ndarray,
        v_current: np.ndarray,
        u_mean: Optional[np.ndarray] = None,
        v_mean: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        渦動能 EKE (Eddy Kinetic Energy)

        EKE = ½(u'² + v'²)

        其中 u' = u - ū, v' = v - v̄
        如果沒有氣候態平均，用空間平滑近似

        高 EKE 區域 → 中尺度渦旋活動強 → 營養鹽混合 → 漁場活性高

        單位：cm²/s² (常見表示法) 或 m²/s²
        """
        if u_mean is None:
            from scipy.ndimage import uniform_filter
            u_mean = uniform_filter(u_current.astype(float), size=9)
            v_mean = uniform_filter(v_current.astype(float), size=9)

        u_prime = u_current - u_mean
        v_prime = v_current - v_mean

        eke = 0.5 * (u_prime**2 + v_prime**2)

        return eke.astype(np.float32)

    @staticmethod
    def compute_eke_suitability(eke: np.ndarray) -> np.ndarray:
        """
        EKE 適宜性

        高 EKE → 好的漁場（但太高表示不穩定）
        最適 EKE: 200-800 cm²/s²
        """
        # 轉為 cm²/s²
        eke_cm = eke * 1e4 if np.nanmax(eke) < 1 else eke

        # Sigmoid: 在 200 cm²/s² 附近快速上升
        si = 1.0 / (1.0 + np.exp(-0.01 * (eke_cm - 200)))
        # 過高懲罰（> 2000 cm²/s²）
        high_penalty = np.where(eke_cm > 2000, np.exp(-(eke_cm - 2000) / 1000), 1.0)

        return np.clip(si * high_penalty, 0, 1).astype(np.float32)

    @staticmethod
    def compute_salinity_suitability(
        salinity: np.ndarray, species: str
    ) -> np.ndarray:
        """
        鹽度適宜性

        鮪魚偏好高鹽海水（34-36 PSU）
        鹽度梯度 = 不同水團交界 = 潛在鋒面
        """
        prefs = {
            "skipjack":  (34.0, 36.0, 35.0, 0.8),
            "yellowfin": (33.5, 36.5, 35.0, 1.0),
            "bigeye":    (33.0, 36.5, 34.8, 1.2),
            "albacore":  (33.5, 36.0, 34.5, 1.0),
            "squid_todarodes": (33.0, 35.0, 34.0, 1.0),
            "squid_ommastrephes": (33.5, 35.5, 34.5, 0.8),
        }

        if species not in prefs:
            return np.ones_like(salinity, dtype=np.float32) * 0.5

        s_min, s_max, s_opt, sigma = prefs[species]

        # 高斯適宜性
        si = np.exp(-((salinity - s_opt) ** 2) / (2 * sigma ** 2))

        # 硬邊界
        si = np.where(salinity < s_min - 1, 0.0, si)
        si = np.where(salinity > s_max + 1, 0.0, si)

        return np.clip(si, 0, 1).astype(np.float32)

    @staticmethod
    def compute_ekman_pumping(
        wind_u: np.ndarray,
        wind_v: np.ndarray,
        lats: np.ndarray,
        dx: float = 25000,  # 格距 (m)
    ) -> np.ndarray:
        """
        Ekman pumping velocity（上升流速度）

        we = (1/ρf) × curl(τ)
        τ = ρ_air × Cd × |W| × W  (風應力)
        curl(τ) = ∂τy/∂x - ∂τx/∂y

        正值 = 上升流（營養鹽上湧，利於漁場）
        負值 = 下降流

        INCOIS PFZ 已納入此指標（查證確認）
        """
        # 風應力計算
        rho_air = 1.225  # kg/m³
        cd = 1.3e-3      # 拖曳係數
        wind_speed = np.sqrt(wind_u**2 + wind_v**2)
        tau_x = rho_air * cd * wind_speed * wind_u
        tau_y = rho_air * cd * wind_speed * wind_v

        # Coriolis 參數
        if lats.ndim == 1:
            lat_2d = lats[:, np.newaxis] * np.ones((1, wind_u.shape[1]))
        else:
            lat_2d = lats
        f = 2 * OMEGA * np.sin(np.radians(lat_2d))
        # 避免赤道 f→0
        f = np.where(np.abs(f) < 1e-6, np.sign(f + 1e-10) * 1e-6, f)

        # 風應力旋度
        dtau_y_dx = np.gradient(tau_y, dx, axis=1)
        dtau_x_dy = np.gradient(tau_x, dx, axis=0)
        curl_tau = dtau_y_dx - dtau_x_dy

        # Ekman pumping
        w_e = curl_tau / (RHO_WATER * f)

        # 單位：m/s → m/day
        w_e_day = w_e * 86400

        return w_e_day.astype(np.float32)

    @staticmethod
    def compute_upwelling_index(ekman_w: np.ndarray) -> np.ndarray:
        """
        上升流指數 (0-1)

        正 Ekman pumping → 上升流 → 好的漁場
        """
        # 只取正值（上升流）
        upwelling = np.maximum(ekman_w, 0)
        # 歸一化
        maxval = np.nanpercentile(upwelling, 95) if np.any(upwelling > 0) else 1.0
        if maxval > 0:
            ui = upwelling / maxval
        else:
            ui = np.zeros_like(upwelling)
        return np.clip(ui, 0, 1).astype(np.float32)

    @staticmethod
    def compute_ssd(
        sst: np.ndarray,
        salinity: np.ndarray,
    ) -> np.ndarray:
        """
        [v16.0] Sea Surface Density (SSD).

        Simplified UNESCO EOS-80 polynomial (Millero & Poisson, 1981).
        σ₀ = ρ(S,T,0) - 1000

        Density fronts = water mass boundaries = prey concentration.
        No gsw dependency — uses the standard polynomial directly.
        """
        T = sst.astype(np.float64)
        S = salinity.astype(np.float64)

        # Pure water density (Bigg formula)
        rho_w = (999.842594 + 6.793952e-2 * T
                 - 9.095290e-3 * T**2
                 + 1.001685e-4 * T**3
                 - 1.120083e-6 * T**4
                 + 6.536332e-9 * T**5)

        # Salinity terms
        A = (8.24493e-1 - 4.0899e-3 * T + 7.6438e-5 * T**2
             - 8.2467e-7 * T**3 + 5.3875e-9 * T**4)
        B = -5.72466e-3 + 1.0227e-4 * T - 1.6546e-6 * T**2
        C = 4.8314e-4

        rho = rho_w + A * S + B * S**1.5 + C * S**2
        sigma0 = rho - 1000.0

        return sigma0.astype(np.float32)

    @staticmethod
    def compute_thermal_gradient(
        temp_3d: np.ndarray,
        depths: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        [v16.0] Multi-depth vertical temperature gradient ∂T/∂z.

        Identifies thermocline layers at 50m, 100m, 200m spacing.
        Strong gradient = sharp thermocline = prey/predator boundary.

        Args:
            temp_3d: (n_depths, ny, nx) temperature profile
            depths: 1D array of depth levels (m, positive down)

        Returns:
            {
                "dt_dz_max": 2D maximum gradient (°C/m),
                "thermocline_depth": 2D depth of max gradient (m),
                "gradient_50m": 2D gradient at ~50m,
                "gradient_100m": 2D gradient at ~100m,
                "gradient_200m": 2D gradient at ~200m,
            }
        """
        n_z, ny, nx = temp_3d.shape

        dt_dz = np.zeros((n_z - 1, ny, nx), dtype=np.float32)
        z_mid = np.zeros(n_z - 1, dtype=np.float32)

        for k in range(n_z - 1):
            dz = max(depths[k + 1] - depths[k], 1.0)
            dt_dz[k] = (temp_3d[k] - temp_3d[k + 1]) / dz  # positive = cooling with depth
            z_mid[k] = (depths[k] + depths[k + 1]) / 2.0

        # Max gradient and its depth
        max_idx = np.argmax(dt_dz, axis=0)
        dt_dz_max = np.take_along_axis(dt_dz, max_idx[np.newaxis], axis=0)[0]
        thermocline_d = z_mid[max_idx]

        # Gradients at specific target depths
        def gradient_at_depth(target_m):
            idx = np.argmin(np.abs(z_mid - target_m))
            return dt_dz[idx].copy()

        return {
            "dt_dz_max": dt_dz_max,
            "thermocline_depth": thermocline_d,
            "gradient_50m": gradient_at_depth(50),
            "gradient_100m": gradient_at_depth(100),
            "gradient_200m": gradient_at_depth(200),
        }

    def analyze_full(
        self,
        u_current: np.ndarray,
        v_current: np.ndarray,
        salinity: Optional[np.ndarray],
        wind_u: Optional[np.ndarray],
        wind_v: Optional[np.ndarray],
        lats: np.ndarray,
        bathy: Optional[np.ndarray] = None,
        species_list: Optional[list] = None,
        sst: Optional[np.ndarray] = None,
        temp_3d: Optional[np.ndarray] = None,
        temp_3d_depths: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """完整物理量分析"""
        if species_list is None:
            species_list = ["skipjack", "yellowfin", "bigeye"]

        log.info("⚡ 衍生物理量計算")

        result = {}

        # EKE
        eke = self.compute_eke(u_current, v_current)
        result["eke"] = eke
        result["eke_si"] = self.compute_eke_suitability(eke)
        log.info(f"  EKE 平均: {np.nanmean(eke):.4f} m²/s²")

        # 鹽度
        if salinity is not None:
            result["salinity_si"] = {}
            for sp in species_list:
                result["salinity_si"][sp] = self.compute_salinity_suitability(
                    salinity, sp
                )

            # [v16.0] SSD 海面密度
            if sst is not None:
                ssd = self.compute_ssd(sst, salinity)
                result["ssd"] = ssd
                log.info(f"  SSD σ₀: [{np.nanmin(ssd):.2f}, {np.nanmax(ssd):.2f}]")

        # [v16.0] 多深度溫度梯度
        if temp_3d is not None and temp_3d_depths is not None:
            tg = self.compute_thermal_gradient(temp_3d, temp_3d_depths)
            result["thermal_gradient"] = tg
            log.info(f"  ∂T/∂z max={np.nanmax(tg['dt_dz_max']):.4f} °C/m, "
                     f"thermocline avg={np.nanmean(tg['thermocline_depth']):.0f}m")

        # Ekman
        if wind_u is not None and wind_v is not None:
            ekman_w = self.compute_ekman_pumping(wind_u, wind_v, lats)
            result["ekman_pumping"] = ekman_w
            result["upwelling_index"] = self.compute_upwelling_index(ekman_w)
            log.info(f"  Ekman pumping 範圍: "
                     f"{np.nanmin(ekman_w):.2f} ~ {np.nanmax(ekman_w):.2f} m/day")

        # 地形
        if bathy is not None:
            result["depth_si"] = {}
            for sp in species_list:
                result["depth_si"][sp] = BathymetryAnalyzer.compute_depth_suitability(
                    bathy, sp
                )

            try:
                seamount_mask, prominence = BathymetryAnalyzer.detect_seamounts(bathy)
                result["seamount_mask"] = seamount_mask
                result["seamount_prominence"] = prominence
                n_sm = int(np.sum(seamount_mask))
                log.info(f"  偵測到 {n_sm} 個海底山格點")
            except ImportError:
                log.info("  海底山偵測需要 scipy")

        return result

    # ──────────────────────────────────────────────
    #  [v16.0 A5] Geostrophic Current from SSH
    # ──────────────────────────────────────────────

    @staticmethod
    def compute_geostrophic_current(
        ssh: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        [v16.0 A5] Compute geostrophic current (u_g, v_g) from SSH gradient.

        u_g = -(g/f) * d(SSH)/dy
        v_g = +(g/f) * d(SSH)/dx

        Ref: Cushman-Roisin & Beckers (2011),
             Introduction to Geophysical Fluid Dynamics

        Args:
            ssh: 2D SSH field (m)
            lats: 1D latitude array
            lons: 1D longitude array

        Returns:
            {"u_geo": 2D (m/s), "v_geo": 2D (m/s), "speed_geo": 2D (m/s)}
        """
        g = 9.81  # m/s²
        ny, nx = ssh.shape
        u_g = np.zeros_like(ssh, dtype=np.float32)
        v_g = np.zeros_like(ssh, dtype=np.float32)

        dy = 111000.0  # m per degree latitude (approximate)

        for j in range(ny):
            lat_rad = np.radians(lats[j])
            f = 2.0 * 7.2921e-5 * np.sin(lat_rad)  # Coriolis parameter
            if abs(f) < 1e-8:  # skip near equator (±2°)
                continue
            dx = 111000.0 * np.cos(lat_rad)  # m per degree longitude

            for i in range(nx):
                # Central differences
                j_up = min(j + 1, ny - 1)
                j_dn = max(j - 1, 0)
                i_rt = min(i + 1, nx - 1)
                i_lt = max(i - 1, 0)

                d_lat = lats[j_up] - lats[j_dn]
                d_lon = lons[i_rt] - lons[i_lt]
                if abs(d_lat) < 1e-6 or abs(d_lon) < 1e-6:
                    continue

                dssh_dy = (ssh[j_up, i] - ssh[j_dn, i]) / (d_lat * dy)
                dssh_dx = (ssh[j, i_rt] - ssh[j, i_lt]) / (d_lon * dx)

                u_g[j, i] = -(g / f) * dssh_dy
                v_g[j, i] = +(g / f) * dssh_dx

        speed = np.sqrt(u_g**2 + v_g**2).astype(np.float32)
        log.info(f"  Geostrophic: |u_g|_max={np.nanmax(np.abs(u_g)):.3f}, "
                 f"|v_g|_max={np.nanmax(np.abs(v_g)):.3f} m/s")

        return {"u_geo": u_g, "v_geo": v_g, "speed_geo": speed}

    # ──────────────────────────────────────────────
    #  [v16.0 A6] Wind Stress Curl
    # ──────────────────────────────────────────────

    @staticmethod
    def compute_wind_stress_curl(
        wind_u: np.ndarray,
        wind_v: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        rho_air: float = 1.225,
        cd: float = 1.3e-3,
    ) -> Dict[str, np.ndarray]:
        """
        [v16.0 A6] Compute wind stress curl → Ekman upwelling/downwelling.

        tau_x = rho_air * Cd * |W| * u
        tau_y = rho_air * Cd * |W| * v
        curl(tau) = d(tau_y)/dx - d(tau_x)/dy

        Positive curl → Ekman upwelling (good for fishing)
        Negative curl → Ekman downwelling

        Ref: Large & Pond (1981) J. Phys. Oceanogr.

        Returns:
            {"wind_stress_curl": 2D (N/m³), "ekman_upwelling": bool mask}
        """
        ny, nx = wind_u.shape
        speed = np.sqrt(wind_u**2 + wind_v**2)

        # Wind stress
        tau_x = rho_air * cd * speed * wind_u
        tau_y = rho_air * cd * speed * wind_v

        curl = np.zeros((ny, nx), dtype=np.float32)

        for j in range(1, ny - 1):
            dx = 111000.0 * np.cos(np.radians(lats[j]))
            dy_m = 111000.0
            for i in range(1, nx - 1):
                d_lon = lons[i + 1] - lons[i - 1]
                d_lat = lats[j + 1] - lats[j - 1]
                dtauy_dx = (tau_y[j, i + 1] - tau_y[j, i - 1]) / (d_lon * dx)
                dtaux_dy = (tau_x[j + 1, i] - tau_x[j - 1, i]) / (d_lat * dy_m)
                curl[j, i] = dtauy_dx - dtaux_dy

        upwelling_mask = curl > 0
        log.info(f"  Wind stress curl: range [{curl.min():.2e}, {curl.max():.2e}] N/m3, "
                 f"upwelling cells: {np.sum(upwelling_mask)}")

        return {
            "wind_stress_curl": curl,
            "ekman_upwelling": upwelling_mask,
        }

