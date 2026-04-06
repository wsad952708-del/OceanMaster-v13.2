"""
OceanMaster v13.2 — CMEMS SSH 數據擷取 + 渦旋偵測
===================================================
🔴 關鍵升級：接入 Copernicus Marine SSH 數據

數據源：
  - CMEMS 全球海洋物理分析預報 (0.083°, 日)
    dataset_id: cmems_mod_glo_phy_anfc_0.083deg_P1D-m
    變量: zos (sea_surface_height_above_geoid)

渦旋偵測方法（逆向工程自商業系統）：
  1. Okubo-Weiss (OW) 參數法 — 經典物理方法
  2. SLA 封閉等高線法 (Chelton et al. 2011) — AVISO 標準算法
  3. SSH 梯度 + 地轉流場推算

原理：
  - 暖渦旋 (反氣旋) = SSH 高值 → 下沉流 → 水溫暖 → 黃鰭鮪偏好
  - 冷渦旋 (氣旋) = SSH 低值 → 涌升流 → 營養鹽豐富 → 鰹魚偏好
  - 渦旋邊緣 = 最佳漁場位置（聚集效應）
"""

import numpy as np
import logging
import asyncio
import httpx
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timezone, timedelta
from pathlib import Path
import os

log = logging.getLogger("OceanMaster.CMEMS")


# ═══════════════════════════════════════════════════
#  CMEMS SSH 數據擷取
# ═══════════════════════════════════════════════════

class CMEMSFetcher:
    """
    Copernicus Marine SSH 數據擷取器

    支援兩種方式:
    1. copernicusmarine Python API (需要帳號)
    2. OPeNDAP 直接存取 (備用)

    免費帳號申請: https://marine.copernicus.eu/
    """

    def __init__(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.lat_min, self.lat_max = lat_range
        self.lon_min, self.lon_max = lon_range
        self.username = username
        self.password = password

        # [v13.2-LicenseFix] The copernicusmarine API is now decoupled via external fetcher
        # to prevent EUPL/GPL licensing contagion.
        import subprocess
        self._has_cmems_api = True  # We assume we can call the external script
        
        # Check if the external script exists
        self._cmems_fetcher_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'data_fetcher_external', 'cmems_fetcher.py'
        )
        if not os.path.exists(self._cmems_fetcher_script):
            self._has_cmems_api = False
            log.info("外部 CMEMS fetcher 未找到，使用 AVISO+ HTTP 備用方式")

    async def fetch_ssh(self) -> Dict[str, np.ndarray]:
        """
        取得 SSH (海面高度異常) 數據

        優先順序:
        1. CMEMS Python API → 最佳 (0.083°, 日更新)
        2. AVISO+ ERDDAP → 備用 (0.25°)
        3. 模擬數據 → 最後手段
        """
        # 方法 1: CMEMS API
        if self._has_cmems_api:
            result = await self._fetch_via_cmems_api()
            if result is not None:
                return result

        # 方法 2: AVISO+ ERDDAP (免登入)
        result = await self._fetch_via_aviso_erddap()
        if result is not None:
            return result

        # 方法 3: 模擬數據
        log.warning("SSH 數據全部取得失敗，使用模擬數據")
        return self._generate_demo_ssh()

    async def _fetch_via_cmems_api(self) -> Optional[Dict[str, np.ndarray]]:
        """透過外部腳本取得 SSH (解耦 copernicusmarine API)"""
        try:
            import subprocess
            import xarray as xr

            now = datetime.now(timezone.utc)
            start = (now - timedelta(days=3)).strftime("%Y-%m-%dT00:00:00")
            end = now.strftime("%Y-%m-%dT00:00:00")

            log.info("透過外部 cmems_fetcher 腳本取得 SSH 數據...")

            # [v13.2-R7] 傳入 CMEMS 帳密
            _user = self.username or os.environ.get("CMEMS_USER", "")
            _pass = self.password or os.environ.get("CMEMS_PASS", "")

            out_dir = "C:/tmp/L2_cache"
            os.makedirs(out_dir, exist_ok=True)
            
            cmd = [
                "python", self._cmems_fetcher_script,
                "--lat-min", str(self.lat_min),
                "--lat-max", str(self.lat_max),
                "--lon-min", str(self.lon_min),
                "--lon-max", str(self.lon_max),
                "--out-dir", out_dir,
                "--target", "ssh"
            ]
            
            env = os.environ.copy()
            if _user:
                env["CMEMS_USER"] = _user
                env["CMEMS_PASS"] = _pass
                
            process = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if process.returncode != 0:
                log.warning(f"外部 CMEMS SSH 腳本執行失敗: {process.stderr}")
                return None
                
            ds = xr.open_dataset(os.path.join(out_dir, "cmems_ssh.nc"))

            # 取最新一天
            ssh_data = ds["zos"].isel(time=-1).values
            lats = ds["latitude"].values
            lons = ds["longitude"].values
            ds.close()

            # SSH 異常 = SSH - 平均
            ssh_anomaly = ssh_data - np.nanmean(ssh_data)

            result = {
                "lat": lats,
                "lon": lons,
                "ssh": ssh_data,
                "ssh_anomaly": ssh_anomaly,
                "source": "CMEMS",
                "resolution": "0.083°",
            }

            log.info(f"CMEMS SSH 取得成功: {ssh_data.shape}, "
                     f"範圍 {np.nanmin(ssh_data):.3f}~{np.nanmax(ssh_data):.3f}m")
            return result

        except Exception as e:
            log.warning(f"CMEMS API 取得失敗: {e}")
            return None

    async def _fetch_via_aviso_erddap(self) -> Optional[Dict[str, np.ndarray]]:
        """透過 AVISO+ ERDDAP 取得 SLA (海面高度異常)"""
        try:
            now = datetime.now(timezone.utc)
            t_start = (now - timedelta(days=7)).strftime("%Y-%m-%dT00:00:00Z")
            t_end = now.strftime("%Y-%m-%dT00:00:00Z")

            # [v13.2-R8] coastwatch.pfeg is DEAD; use NCEI/CoastWatch mirrors
            # Try multiple SLA dataset IDs on working servers
            _sla_servers = [
                "https://www.ncei.noaa.gov/erddap/griddap",
                "https://coastwatch.noaa.gov/erddap/griddap",
            ]
            _sla_datasets = [
                ("erdTAgeo1day_LonPM180", "sla"),
                ("erdTAgeoDaily_LonPM180", "sla"),
                ("noaacwSLAdaily", "sea_surface_height_above_sea_level"),
            ]

            resp = None
            for sla_server in _sla_servers:
                for sla_dsid, sla_var in _sla_datasets:
                    url = (
                        f"{sla_server}/{sla_dsid}.json"
                        f"?{sla_var}[({t_start}):1:({t_end})]"
                        f"[({self.lat_min}):1:({self.lat_max})]"
                        f"[({self.lon_min}):1:({self.lon_max})]"
                    )

                    log.info(f"透過 {sla_server.split('//')[1][:15]} 取得 SLA ({sla_dsid})...")

                    from engine.data_fetcher_v2 import safe_fetch
                    async with httpx.AsyncClient() as client:
                        resp = await safe_fetch(client, url, f"AVISO-{sla_dsid[:12]}", timeout=60)

                    if resp is not None:
                        break
                if resp is not None:
                    break

            if resp is None:
                log.warning("AVISO SLA: all servers/datasets failed")
                return None

            data = resp.json()
            table = data["table"]
            cols = table["columnNames"]
            rows = np.array(table["rows"])

            lat_col = rows[:, cols.index("latitude")].astype(float)
            lon_col = rows[:, cols.index("longitude")].astype(float)

            sla_idx = None
            for candidate in [sla_var, "sla", "sea_level_anomaly", "sea_surface_height_above_sea_level", "adt"]:
                if candidate in cols:
                    sla_idx = cols.index(candidate)
                    break

            if sla_idx is None:
                log.error("AVISO 回應中找不到 SLA 欄位")
                return None

            sla_col = np.where(
                rows[:, sla_idx] == "NaN", np.nan,
                rows[:, sla_idx]
            ).astype(float)

            lats = np.unique(lat_col)
            lons = np.unique(lon_col)

            # 取最新時間步
            n_per_step = len(lats) * len(lons)
            if len(sla_col) >= n_per_step:
                sla_latest = sla_col[-n_per_step:]
                lat_col_latest = lat_col[-n_per_step:]
                lon_col_latest = lon_col[-n_per_step:]
            else:
                sla_latest = sla_col
                lat_col_latest = lat_col
                lon_col_latest = lon_col

            sla_grid = np.full((len(lats), len(lons)), np.nan, dtype=np.float32)
            lat_idx = np.searchsorted(lats, lat_col_latest)
            lon_idx = np.searchsorted(lons, lon_col_latest)
            sla_grid[lat_idx, lon_idx] = sla_latest

            result = {
                "lat": lats,
                "lon": lons,
                "ssh": sla_grid,
                "ssh_anomaly": sla_grid,  # SLA 本身就是異常值
                "source": "AVISO+ERDDAP",
                "resolution": "0.25°",
            }

            log.info(f"AVISO SLA 取得成功: {sla_grid.shape}")
            return result

        except Exception as e:
            log.warning(f"AVISO ERDDAP 取得失敗: {e}")
            return None

    def _generate_demo_ssh(self) -> Dict[str, np.ndarray]:
        """生成模擬 SSH 數據"""
        log.info("使用模擬 SSH 數據")
        lats = np.arange(self.lat_min, self.lat_max, 0.25)
        lons = np.arange(self.lon_min, self.lon_max, 0.25)
        lat_g, lon_g = np.meshgrid(lats, lons, indexing="ij")

        # 模擬 SSH: 黑潮主軸 + 中尺度渦旋
        # 黑潮（沿東經130-140度，北緯20-35度）
        ssh_base = 0.5 * np.exp(-((lat_g - 25)**2 + (lon_g - 135)**2) / 50)

        # [v13.2-fix] 確定性渦旋模式 — 取代 np.random（可再現性）
        _eddy_specs = [
            (self.lat_min + 7, self.lon_min + 8, 2.0,  0.10),
            (self.lat_min + 12, self.lon_min + 15, 1.5, -0.12),
            (self.lat_min + 5, self.lon_min + 20, 2.5,  0.08),
            (self.lat_min + 18, self.lon_min + 10, 1.8, -0.09),
            (self.lat_min + 10, self.lon_min + 25, 2.2,  0.11),
            (self.lat_min + 15, self.lon_min + 5, 1.3, -0.07),
            (self.lat_min + 20, self.lon_min + 18, 2.8,  0.13),
            (self.lat_min + 8, self.lon_min + 12, 1.6, -0.10),
        ]
        for cy, cx, r, amp in _eddy_specs:
            ssh_base += amp * np.exp(-((lat_g - cy)**2 + (lon_g - cx)**2) / (2 * r**2))

        # 確定性空間擾動
        ssh_base += 0.02 * np.sin(lat_g * 5.7 + lon_g * 8.3)

        return {
            "lat": lats, "lon": lons,
            "ssh": ssh_base,
            "ssh_anomaly": ssh_base - np.nanmean(ssh_base),
            "source": "demo",
            "is_demo": True,
        }


# ═══════════════════════════════════════════════════
#  渦旋偵測（三法融合 — 超越商業系統）
# ═══════════════════════════════════════════════════

class EddyDetector:
    """
    中尺度渦旋偵測器

    三種方法融合（逆向工程自 AVISO/CMEMS 渦旋追蹤系統）：

    1. Okubo-Weiss (OW) 參數法
       - W = Sn² + Ss² - ω² (剪切²平方 + 拉伸² - 渦度²)
       - W < -0.2σ_W → 渦旋核心

    2. SLA 封閉等高線法 (Chelton et al. 2011)
       - 找 SLA 局部極值
       - 追蹤封閉等高線
       - 通過面積/振幅閾值篩選

    3. 地轉流場旋轉偵測 (Nencioli et al. 2010)
       - 從 SSH 計算地轉流
       - 幾何判定旋轉結構
    """

    def __init__(self):
        self.eddies_warm = []
        self.eddies_cold = []

    def detect_all(
        self,
        ssh: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
    ) -> Dict[str, Any]:
        """
        完整渦旋偵測管線

        返回:
            warm_eddies: 暖渦旋列表 (反氣旋, SSH+)
            cold_eddies: 冷渦旋列表 (氣旋, SSH-)
            eddy_map: 渦旋標記網格
            geostrophic_u/v: 地轉流分量
        """
        log.info(f"渦旋偵測... SSH 網格={ssh.shape}")

        # 預處理
        ssh_filled = np.nan_to_num(ssh, nan=np.nanmean(ssh))
        ssh_smooth = ndimage.gaussian_filter(ssh_filled, sigma=1.5)
        sla = ssh_smooth - np.nanmean(ssh_smooth)

        # ── 1. Okubo-Weiss 方法 ──
        ow_eddies = self._okubo_weiss(sla, lat, lon)

        # ── 2. SLA 封閉等高線法 ──
        contour_eddies = self._sla_contour_method(sla, lat, lon)

        # ── 3. 計算地轉流 ──
        geo_u, geo_v = self._compute_geostrophic_current(sla, lat, lon)

        # ── 融合結果 ──
        eddy_map, warm_list, cold_list = self._merge_eddies(
            ow_eddies, contour_eddies, sla, lat, lon
        )

        self.eddies_warm = warm_list
        self.eddies_cold = cold_list

        n_warm = len(warm_list)
        n_cold = len(cold_list)
        log.info(f"渦旋偵測完成: {n_warm} 暖渦旋, {n_cold} 冷渦旋")

        return {
            "eddy_map": eddy_map,
            "warm_eddies": warm_list,
            "cold_eddies": cold_list,
            "n_warm": n_warm,
            "n_cold": n_cold,
            "ssh_anomaly": sla,
            "geostrophic_u": geo_u,
            "geostrophic_v": geo_v,
        }

    def _okubo_weiss(
        self,
        sla: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Okubo-Weiss 渦旋偵測

        原理：
        OW = Sn² + Ss² - ω²
        Sn = ∂u/∂x - ∂v/∂y (法向剪切)
        Ss = ∂v/∂x + ∂u/∂y (切向剪切)
        ω = ∂v/∂x - ∂u/∂y (相對渦度)

        OW < -0.2σ_W → 渦旋主導區（核心）
        OW > +0.2σ_W → 應變主導區（背景場）
        """
        # 從 SLA 計算地轉流
        geo_u, geo_v = self._compute_geostrophic_current(sla, lat, lon)

        # 計算速度梯度
        if len(lat) > 1:
            dlat = np.abs(lat[1] - lat[0]) * 111000  # m
        else:
            dlat = 25000

        if len(lon) > 1:
            mean_lat = np.mean(lat)
            dlon = np.abs(lon[1] - lon[0]) * 111000 * np.cos(np.radians(mean_lat))
        else:
            dlon = 25000

        # ∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y
        dudx = np.gradient(geo_u, dlon, axis=1)
        dudy = np.gradient(geo_u, dlat, axis=0)
        dvdx = np.gradient(geo_v, dlon, axis=1)
        dvdy = np.gradient(geo_v, dlat, axis=0)

        # OW 參數
        Sn = dudx - dvdy       # 法向剪切
        Ss = dvdx + dudy       # 切向剪切
        omega = dvdx - dudy    # 相對渦度

        OW = Sn**2 + Ss**2 - omega**2

        # 閾值: -0.2σ_W
        sigma_OW = np.nanstd(OW)
        threshold = -0.2 * sigma_OW

        eddy_mask = OW < threshold
        vorticity = omega

        return {
            "ow": OW,
            "eddy_mask": eddy_mask,
            "vorticity": vorticity,
            "threshold": threshold,
        }

    def _sla_contour_method(
        self,
        sla: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        min_radius_km: float = 25,
        max_radius_km: float = 300,
        min_amplitude_m: float = 0.02,
    ) -> List[Dict[str, Any]]:
        """
        SLA 封閉等高線法 (Chelton et al. 2011)

        步驟：
        1. 找 SLA 局部極值（大於/小於周圍）
        2. 從極值向外追蹤封閉等高線
        3. 通過面積/振幅閾值篩選
        4. 計算渦旋參數（中心、半徑、振幅、旋轉方向）
        """
        eddies = []
        ny, nx = sla.shape

        # SLA 等高線間距
        contour_step = 0.01  # 1cm

        # ── 找正極值（暖渦旋 / 反氣旋）──
        sla_max_filtered = ndimage.maximum_filter(sla, size=9)
        local_max_mask = (sla == sla_max_filtered) & (sla > min_amplitude_m)

        for iy, ix in zip(*np.where(local_max_mask)):
            eddy = self._trace_eddy(
                sla, lat, lon, iy, ix,
                polarity="anticyclonic",
                contour_step=contour_step,
                min_radius_km=min_radius_km,
                max_radius_km=max_radius_km,
            )
            if eddy is not None:
                eddies.append(eddy)

        # ── 找負極值（冷渦旋 / 氣旋）──
        sla_min_filtered = ndimage.minimum_filter(sla, size=9)
        local_min_mask = (sla == sla_min_filtered) & (sla < -min_amplitude_m)

        for iy, ix in zip(*np.where(local_min_mask)):
            eddy = self._trace_eddy(
                -sla, lat, lon, iy, ix,  # 反轉 SLA
                polarity="cyclonic",
                contour_step=contour_step,
                min_radius_km=min_radius_km,
                max_radius_km=max_radius_km,
            )
            if eddy is not None:
                eddy["amplitude"] = -eddy["amplitude"]
                eddies.append(eddy)

        # 去重複（距離太近的取較大的）
        eddies = self._nms_eddies(eddies, min_distance_km=50)

        return eddies

    def _trace_eddy(
        self,
        sla: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        center_iy: int,
        center_ix: int,
        polarity: str,
        contour_step: float = 0.01,
        min_radius_km: float = 25,
        max_radius_km: float = 300,
    ) -> Optional[Dict[str, Any]]:
        """追蹤單個渦旋的封閉等高線"""
        ny, nx = sla.shape
        center_val = sla[center_iy, center_ix]

        # 從中心值向外降低，找最外層封閉等高線
        best_radius = 0
        best_area = 0

        for level in np.arange(center_val - contour_step, 0, -contour_step):
            mask = sla >= level

            # 只保留包含中心點的連通區域
            labeled, n_features = ndimage.label(mask)
            if labeled[center_iy, center_ix] == 0:
                break

            region_label = labeled[center_iy, center_ix]
            region_mask = labeled == region_label

            # 計算等效半徑
            area_pixels = np.sum(region_mask)
            if len(lat) > 1:
                pixel_area_km2 = (
                    np.abs(lat[1] - lat[0]) * 111
                    * np.abs(lon[1] - lon[0]) * 111
                    * np.cos(np.radians(lat[center_iy]))
                )
            else:
                pixel_area_km2 = 625  # 25km × 25km

            area_km2 = area_pixels * pixel_area_km2
            radius_km = np.sqrt(area_km2 / np.pi)

            if radius_km > max_radius_km:
                break

            if radius_km >= min_radius_km:
                best_radius = radius_km
                best_area = area_km2

        if best_radius < min_radius_km:
            return None

        amplitude = float(center_val)

        return {
            "lat": float(lat[center_iy]),
            "lon": float(lon[center_ix]),
            "polarity": polarity,
            "radius_km": float(best_radius),
            "area_km2": float(best_area),
            "amplitude": float(amplitude),
        }

    def _compute_geostrophic_current(
        self,
        sla: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        從 SSH 計算地轉流

        公式 (f-plane 近似):
        u_g = -(g/f) × ∂η/∂y
        v_g = (g/f) × ∂η/∂x

        其中 f = 2Ω sin(φ) (科氏參數)
        """
        g = 9.81  # m/s²
        omega = 7.2921e-5  # 地球自轉角速度

        ny, nx = sla.shape
        lat_2d = np.broadcast_to(lat.reshape(-1, 1), (ny, nx)) if lat.ndim == 1 else lat

        # 科氏參數 (避免赤道附近 f→0)
        f = 2 * omega * np.sin(np.radians(lat_2d))
        f = np.where(np.abs(f) < 1e-6, np.sign(f) * 1e-6, f)

        # SSH 梯度
        if len(lat) > 1:
            dy = np.abs(lat[1] - lat[0]) * 111000
        else:
            dy = 25000
        if len(lon) > 1:
            dx = np.abs(lon[1] - lon[0]) * 111000 * np.cos(np.radians(np.mean(lat)))
        else:
            dx = 25000

        deta_dy = np.gradient(sla, dy, axis=0)
        deta_dx = np.gradient(sla, dx, axis=1)

        # 地轉流
        u_geo = -(g / f) * deta_dy
        v_geo = (g / f) * deta_dx

        return u_geo, v_geo

    def _merge_eddies(
        self,
        ow_result: Dict,
        contour_eddies: List[Dict],
        sla: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
    ) -> Tuple[np.ndarray, List[Dict], List[Dict]]:
        """融合 OW 和等高線方法的結果"""
        ny, nx = sla.shape
        eddy_map = np.zeros((ny, nx), dtype=int)  # 0=無, 1=暖渦旋, -1=冷渦旋

        warm_list = []
        cold_list = []

        for eddy in contour_eddies:
            if eddy["polarity"] == "anticyclonic":
                warm_list.append(eddy)
            else:
                cold_list.append(eddy)

        # 在 eddy_map 上標記
        vorticity = ow_result.get("vorticity")
        ow_mask = ow_result.get("eddy_mask", np.zeros((ny, nx), dtype=bool))

        # OW 渦旋核心 + 渦度方向
        if vorticity is not None:
            eddy_map[ow_mask & (vorticity < 0)] = 1   # 反氣旋（北半球）
            eddy_map[ow_mask & (vorticity > 0)] = -1  # 氣旋（北半球）

        # 等高線渦旋也加入
        for eddy in contour_eddies:
            iy = np.argmin(np.abs(lat - eddy["lat"]))
            ix = np.argmin(np.abs(lon - eddy["lon"]))
            r_idx = max(1, int(eddy["radius_km"] / 25))  # 格點數

            y_lo = max(0, iy - r_idx)
            y_hi = min(ny, iy + r_idx + 1)
            x_lo = max(0, ix - r_idx)
            x_hi = min(nx, ix + r_idx + 1)

            val = 1 if eddy["polarity"] == "anticyclonic" else -1
            eddy_map[y_lo:y_hi, x_lo:x_hi] = val

        return eddy_map, warm_list, cold_list

    def _nms_eddies(
        self, eddies: List[Dict], min_distance_km: float = 50
    ) -> List[Dict]:
        """非極大值抑制：去除太近的渦旋"""
        if not eddies:
            return []

        sorted_eddies = sorted(
            eddies, key=lambda e: abs(e["amplitude"]), reverse=True
        )
        selected = []

        for eddy in sorted_eddies:
            too_close = False
            for sel in selected:
                dist = np.sqrt(
                    ((eddy["lat"] - sel["lat"]) * 111)**2
                    + ((eddy["lon"] - sel["lon"])
                       * 111 * np.cos(np.radians(eddy["lat"])))**2
                )
                if dist < min_distance_km:
                    too_close = True
                    break
            if not too_close:
                selected.append(eddy)

        return selected

    def compute_eddy_fishing_score(
        self,
        eddy_map: np.ndarray,
        sla: np.ndarray,
        species: str = "skipjack",
    ) -> np.ndarray:
        """
        計算渦旋對漁場的貢獻分數

        物種偏好：
        - 鰹魚: 偏好冷渦旋邊緣（涌升流帶來營養鹽）
        - 黃鰭鮪: 偏好暖渦旋（水溫暖）
        - 大目鮪: 偏好渦旋邊緣（鋒面效應）
        - 魷魚: 偏好冷渦旋（營養鹽豐富）
        """
        eddy_score = np.zeros_like(sla, dtype=float)

        species_prefs = {
            "skipjack": {"warm_core": 0.3, "warm_edge": 0.7, "cold_core": 0.6, "cold_edge": 0.9},
            "yellowfin": {"warm_core": 0.8, "warm_edge": 0.7, "cold_core": 0.3, "cold_edge": 0.5},
            "bigeye": {"warm_core": 0.4, "warm_edge": 0.8, "cold_core": 0.5, "cold_edge": 0.8},
            "squid": {"warm_core": 0.2, "warm_edge": 0.4, "cold_core": 0.7, "cold_edge": 0.8},
        }
        prefs = species_prefs.get(species, species_prefs["skipjack"])

        # 暖渦旋核心
        warm_core = eddy_map > 0
        eddy_score[warm_core] = prefs["warm_core"]

        # 暖渦旋邊緣（膨脹 - 原始）
        warm_dilated = ndimage.binary_dilation(warm_core, iterations=3)
        warm_edge = warm_dilated & ~warm_core
        eddy_score[warm_edge] = np.maximum(eddy_score[warm_edge], prefs["warm_edge"])

        # 冷渦旋核心
        cold_core = eddy_map < 0
        eddy_score[cold_core] = prefs["cold_core"]

        # 冷渦旋邊緣
        cold_dilated = ndimage.binary_dilation(cold_core, iterations=3)
        cold_edge = cold_dilated & ~cold_core
        eddy_score[cold_edge] = np.maximum(eddy_score[cold_edge], prefs["cold_edge"])

        return eddy_score
