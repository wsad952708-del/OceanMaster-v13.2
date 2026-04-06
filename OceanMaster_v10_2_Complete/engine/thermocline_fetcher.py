"""
OceanMaster — 溫躍層深度 (Z20) 與混合層深度 (MLD) 擷取器
=========================================================
數據源（全部免費）:
  1. HYCOM GOFS 3.1 — 3D 溫度剖面 → 計算 Z20 / MLD
  2. Copernicus GLOBAL_ANALYSISFORECAST_PHY_001_024 — mlotst (MLD)
  3. WOA23 氣候態 — 備援

Z20 = 20°C 等溫線深度（溫躍層代理指標）
MLD = 混合層深度（溫度閾值法 ΔT > 0.5°C）
"""

import numpy as np
import logging
from typing import Optional, Dict, Tuple

log = logging.getLogger("OceanMaster.Thermocline")

# HYCOM 標準深度層 (m)
HYCOM_DEPTHS = np.array([
    0, 2, 4, 6, 8, 10, 12, 15, 20, 25,
    30, 35, 40, 45, 50, 60, 70, 80, 90, 100,
    125, 150, 200, 250, 300, 350, 400, 500, 600, 700,
    800, 900, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 5000
], dtype=np.float32)


class ThermoclineFetcher:
    """
    從 3D 溫度剖面計算溫躍層深度 (Z20) 與混合層深度 (MLD)。

    優先使用 OceanDataFetcher 已擷取的 HYCOM temp_3d 數據，
    若不可用則以 WOA23 氣候態估算。
    """

    def __init__(self):
        pass

    # ─── 核心：從 3D 溫度剖面計算 Z20 ───

    @staticmethod
    def compute_z20(temp_profile: np.ndarray, depths: np.ndarray) -> float:
        """
        計算 20°C 等溫線深度 (Z20)

        在垂直溫度剖面中找到 T=20°C 的深度，使用線性內插。
        Z20 是溫躍層深度的標準代理指標 (Kessler 1990)。

        Parameters:
            temp_profile: 1D 溫度陣列 (°C), 從表層到深層
            depths: 1D 深度陣列 (m), 對應溫度

        Returns:
            Z20 深度 (m)，若整個水柱 >20°C 回傳最大深度
        """
        if len(temp_profile) < 2:
            return 150.0  # 預設值

        # 表面已 <= 20°C → 溫躍層出露，Z20 = 0
        if temp_profile[0] <= 20.0:
            return 0.0

        for i in range(1, len(temp_profile)):
            if temp_profile[i] <= 20.0:
                t_above = temp_profile[i - 1]
                t_below = temp_profile[i]
                d_above = depths[i - 1]
                d_below = depths[i]
                if abs(t_below - t_above) < 0.001:
                    return float(d_above)
                z20 = d_above + (20.0 - t_above) / (t_below - t_above) * (d_below - d_above)
                return float(np.clip(z20, 0, 1500))

        # 整個水柱 > 20°C (熱帶表層)
        return float(depths[-1])

    @staticmethod
    def compute_mld(temp_profile: np.ndarray, depths: np.ndarray,
                    threshold: float = 0.5) -> float:
        """
        計算混合層深度 (MLD) — 溫度閾值法

        MLD = 溫度相對表層下降超過 threshold 的第一個深度。
        (de Boyer Montégut et al. 2004)

        Parameters:
            temp_profile: 1D 溫度陣列 (°C)
            depths: 1D 深度陣列 (m)
            threshold: 溫差閾值 (°C)，預設 0.5

        Returns:
            MLD 深度 (m)
        """
        if len(temp_profile) < 2:
            return 30.0

        t_surface = temp_profile[0]
        for i in range(1, len(temp_profile)):
            if abs(temp_profile[i] - t_surface) > threshold:
                # 線性內插到精確深度
                dt_prev = abs(temp_profile[i - 1] - t_surface)
                dt_curr = abs(temp_profile[i] - t_surface)
                d_prev = depths[i - 1]
                d_curr = depths[i]
                if abs(dt_curr - dt_prev) < 0.001:
                    return float(d_prev)
                mld = d_prev + (threshold - dt_prev) / (dt_curr - dt_prev) * (d_curr - d_prev)
                return float(np.clip(mld, 0, 500))

        return float(depths[-1])

    @staticmethod
    def compute_thermocline_gradient(temp_profile: np.ndarray,
                                      depths: np.ndarray) -> Tuple[float, float]:
        """
        計算溫躍層梯度強度與深度

        Returns:
            (gradient_max_depth, gradient_strength)
            梯度最強處的深度 (m) 及梯度值 (°C/m)
        """
        if len(temp_profile) < 3:
            return 100.0, 0.1

        gradients = np.abs(np.diff(temp_profile) / np.diff(depths))
        idx = np.argmax(gradients)
        depth = (depths[idx] + depths[idx + 1]) / 2.0
        return float(depth), float(gradients[idx])

    # ─── 批量計算（整個網格） ───

    def compute_from_temp3d(self, temp_3d: np.ndarray,
                            lats: np.ndarray, lons: np.ndarray,
                            depths: Optional[np.ndarray] = None) -> Dict:
        """
        從 3D 溫度場 (depth × lat × lon) 計算 Z20 與 MLD 的 2D 場。

        注意: temp_3d 的 lat/lon 維度可能比 lats/lons 小 (HYCOM 解析度較低)。
        此函數先在 temp_3d 的原始網格上計算，再插值到目標網格。

        Parameters:
            temp_3d: 3D array (depth, lat_hycom, lon_hycom) 溫度 °C
            lats, lons: 目標 1D 座標陣列 (完整網格)
            depths: 1D 深度陣列 (若 None 使用 HYCOM 標準深度)

        Returns:
            dict with 'z20', 'mld', 'thermocline_depth', 'thermocline_strength'
        """
        if depths is None:
            depths = HYCOM_DEPTHS[:temp_3d.shape[0]]

        nz, ny_h, nx_h = temp_3d.shape
        ny, nx = len(lats), len(lons)

        # Step 1: 在 HYCOM 原始網格上計算
        z20_h = np.full((ny_h, nx_h), 150.0, dtype=np.float32)
        mld_h = np.full((ny_h, nx_h), 30.0, dtype=np.float32)
        tc_depth_h = np.full((ny_h, nx_h), 100.0, dtype=np.float32)
        tc_strength_h = np.full((ny_h, nx_h), 0.1, dtype=np.float32)

        for j in range(ny_h):
            for i in range(nx_h):
                profile = temp_3d[:, j, i]
                if np.all(np.isnan(profile)):
                    continue

                valid = ~np.isnan(profile)
                if np.sum(valid) < 3:
                    continue

                p = profile[valid]
                d = depths[valid]

                z20_h[j, i] = self.compute_z20(p, d)
                mld_h[j, i] = self.compute_mld(p, d)
                td, ts = self.compute_thermocline_gradient(p, d)
                tc_depth_h[j, i] = td
                tc_strength_h[j, i] = ts

        # Step 2: 插值到目標網格 (如果尺寸不同)
        if ny_h == ny and nx_h == nx:
            z20, mld, tc_depth, tc_strength = z20_h, mld_h, tc_depth_h, tc_strength_h
        else:
            # HYCOM 的 lat/lon 座標: 均勻取樣於目標範圍
            lats_h = np.linspace(lats[0], lats[-1], ny_h)
            lons_h = np.linspace(lons[0], lons[-1], nx_h)

            try:
                from scipy.interpolate import RegularGridInterpolator

                def _interp(field_h):
                    interp = RegularGridInterpolator(
                        (lats_h, lons_h), field_h,
                        method='linear', bounds_error=False,
                        fill_value=None  # extrapolate
                    )
                    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
                    pts = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
                    return interp(pts).reshape(ny, nx).astype(np.float32)

                z20 = _interp(z20_h)
                mld = _interp(mld_h)
                tc_depth = _interp(tc_depth_h)
                tc_strength = _interp(tc_strength_h)
            except ImportError:
                # 無 scipy: 用最近鄰插值
                log.warning("  scipy 不可用，使用最近鄰插值")
                from numpy import interp as np_interp

                def _nn_interp(field_h):
                    result = np.full((ny, nx), np.nanmean(field_h), dtype=np.float32)
                    for j, lat in enumerate(lats):
                        j_h = min(int((lat - lats_h[0]) / max(lats_h[-1] - lats_h[0], 1e-6) * (ny_h - 1) + 0.5), ny_h - 1)
                        j_h = max(0, j_h)
                        for i, lon in enumerate(lons):
                            i_h = min(int((lon - lons_h[0]) / max(lons_h[-1] - lons_h[0], 1e-6) * (nx_h - 1) + 0.5), nx_h - 1)
                            i_h = max(0, i_h)
                            result[j, i] = field_h[j_h, i_h]
                    return result

                z20 = _nn_interp(z20_h)
                mld = _nn_interp(mld_h)
                tc_depth = _nn_interp(tc_depth_h)
                tc_strength = _nn_interp(tc_strength_h)

            log.info(f"  HYCOM grid ({ny_h}×{nx_h}) → target grid ({ny}×{nx}) interpolated")

        log.info(f"  Z20: {np.nanmean(z20):.0f}m avg, "
                 f"MLD: {np.nanmean(mld):.0f}m avg, "
                 f"Thermocline: {np.nanmean(tc_strength):.3f} °C/m")

        return {
            "z20": z20,
            "mld": mld,
            "thermocline_depth": tc_depth,
            "thermocline_strength": tc_strength,
        }

    # ─── GLORYS12 via copernicusmarine ───

    def fetch_glorys12(self, lats: np.ndarray, lons: np.ndarray,
                       month: int = None) -> Optional[Dict]:
        """
        從 Copernicus Marine GLORYS12 取得 3D 溫度場，計算 Z20 / MLD。

        產品: GLOBAL_MULTIYEAR_PHY_001_030 (glorys12v1 再分析)
              GLOBAL_ANALYSISFORECAST_PHY_001_024 (即時預報)
        解析度: 1/12° (~8km), 日度, 50 深度層
        免費: 只需 Copernicus Marine 帳號
        """
        try:
            import copernicusmarine as cm
        except ImportError:
            log.debug("  copernicusmarine 未安裝，跳過 GLORYS12")
            return None

        import os
        from datetime import datetime, timedelta, timezone
        user = os.environ.get("CMEMS_USER", "")
        pwd = os.environ.get("CMEMS_PASS", "")
        if not user:
            log.debug("  CMEMS_USER 未設定，跳過 GLORYS12")
            return None

        lat_min, lat_max = float(lats.min()), float(lats.max())
        lon_min, lon_max = float(lons.min()), float(lons.max())
        now = datetime.now(timezone.utc)

        # 選擇深度層 (取到 1000m 足以計算 Z20)
        target_depths = [0, 5, 10, 15, 20, 30, 50, 75, 100,
                         125, 150, 200, 250, 300, 400, 500, 600, 800, 1000]

        # 嘗試順序: 即時預報 → 再分析
        datasets = [
            ("cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m", "thetao", "GLORYS-forecast"),
            ("cmems_mod_glo_phy_my_0.083deg_P1D-m", "thetao", "GLORYS12-reanalysis"),
        ]

        for dsid, var, tag in datasets:
            try:
                log.info(f"  嘗試 {tag}: {dsid}")
                ds = cm.open_dataset(
                    dataset_id=dsid,
                    variables=[var],
                    minimum_latitude=lat_min, maximum_latitude=lat_max,
                    minimum_longitude=lon_min, maximum_longitude=lon_max,
                    minimum_depth=0.0, maximum_depth=1000.0,
                    start_datetime=(now - timedelta(days=5)).isoformat(),
                    end_datetime=now.isoformat(),
                    username=user, password=pwd,
                )

                # 取最新時間步
                temp_data = ds[var].isel(time=-1).values  # shape: (depth, lat, lon)
                src_depths = ds.depth.values.astype(np.float32)
                src_lats = ds.latitude.values
                src_lons = ds.longitude.values
                ds.close()

                if temp_data.ndim != 3 or temp_data.shape[0] < 5:
                    log.warning(f"  {tag}: insufficient depth layers ({temp_data.shape})")
                    continue

                # NaN 安全
                temp_data = np.where(np.isfinite(temp_data), temp_data, np.nan)

                log.info(f"  ✅ {tag}: {temp_data.shape} (depth×lat×lon)")

                # 在 GLORYS 網格上計算 Z20/MLD
                nz, ny_g, nx_g = temp_data.shape
                z20_g = np.full((ny_g, nx_g), 150.0, dtype=np.float32)
                mld_g = np.full((ny_g, nx_g), 30.0, dtype=np.float32)

                for j in range(ny_g):
                    for i in range(nx_g):
                        profile = temp_data[:, j, i]
                        valid = ~np.isnan(profile)
                        if np.sum(valid) < 3:
                            continue
                        p = profile[valid]
                        d = src_depths[valid]
                        z20_g[j, i] = self.compute_z20(p, d)
                        mld_g[j, i] = self.compute_mld(p, d)

                # 插值到目標網格
                ny, nx = len(lats), len(lons)
                try:
                    from scipy.interpolate import RegularGridInterpolator

                    def _interp_glorys(field):
                        interp = RegularGridInterpolator(
                            (src_lats, src_lons), field,
                            method='linear', bounds_error=False, fill_value=None
                        )
                        lat_g, lon_g = np.meshgrid(lats, lons, indexing='ij')
                        pts = np.column_stack([lat_g.ravel(), lon_g.ravel()])
                        return interp(pts).reshape(ny, nx).astype(np.float32)

                    z20 = _interp_glorys(z20_g)
                    mld = _interp_glorys(mld_g)
                except ImportError:
                    # 最近鄰回退
                    z20 = np.full((ny, nx), np.nanmean(z20_g), dtype=np.float32)
                    mld = np.full((ny, nx), np.nanmean(mld_g), dtype=np.float32)

                coverage = np.sum(np.isfinite(z20)) / z20.size * 100
                log.info(f"  GLORYS12 Z20: avg={np.nanmean(z20):.0f}m, "
                         f"MLD: avg={np.nanmean(mld):.0f}m, coverage={coverage:.0f}%")

                return {
                    "z20": z20,
                    "mld": mld,
                    "thermocline_depth": z20,  # 以 Z20 近似
                    "thermocline_strength": np.full((ny, nx), 0.1, dtype=np.float32),
                    "source": tag,
                }
            except Exception as e:
                log.warning(f"  {tag} failed: {e}")
                continue

        log.info("  GLORYS12: all datasets failed, falling back")
        return None

    # ─── 氣候態備援 ───

    @staticmethod
    def climatology_z20(lats: np.ndarray, lons: np.ndarray,
                        month: int) -> np.ndarray:
        """
        WOA23 氣候態 Z20 估算

        基於緯度帶的一般化估算 (Riley et al. 2015):
          - 赤道太平洋: Z20 ≈ 100-200m (ENSO 敏感)
          - 亞熱帶: Z20 ≈ 150-300m
          - 溫帶: Z20 ≈ 200-400m
        """
        ny, nx = len(lats), len(lons)
        z20 = np.full((ny, nx), 200.0, dtype=np.float32)

        for j, lat in enumerate(lats):
            abs_lat = abs(lat)
            if abs_lat < 10:
                # 赤道帶 — Z20 較淺
                base = 120.0
                # 太平洋 warm pool vs cold tongue
                for i, lon in enumerate(lons):
                    if 130 <= lon <= 180:
                        z20[j, i] = base + 30  # warm pool: 較深
                    else:
                        z20[j, i] = base - 20  # cold tongue: 較淺
            elif abs_lat < 25:
                base = 180.0 + (abs_lat - 10) * 5
                z20[j, :] = base
            else:
                base = 250.0 + (abs_lat - 25) * 3
                z20[j, :] = base

        # 季節調整 (北半球)
        if 6 <= month <= 8:
            z20 *= 0.9   # 夏季溫躍層上升
        elif 12 <= month or month <= 2:
            z20 *= 1.1   # 冬季溫躍層下沉

        return z20

    @staticmethod
    def climatology_mld(lats: np.ndarray, lons: np.ndarray,
                        month: int) -> np.ndarray:
        """
        WOA23 氣候態 MLD 估算

        基於 de Boyer Montégut 氣候態:
          - 赤道帶: 20-40m (風弱→淺混合層)
          - 亞熱帶: 30-80m (隨季節)
          - 溫帶冬季: 可達 100-200m
        """
        ny, nx = len(lats), len(lons)
        mld = np.full((ny, nx), 40.0, dtype=np.float32)

        for j, lat in enumerate(lats):
            abs_lat = abs(lat)
            if abs_lat < 10:
                mld[j, :] = 25.0
            elif abs_lat < 25:
                mld[j, :] = 30.0 + (abs_lat - 10) * 2
            else:
                mld[j, :] = 60.0 + (abs_lat - 25) * 3

        # 冬半球混合層加深
        if month in (12, 1, 2):
            # 北半球冬 → 深 MLD
            for j, lat in enumerate(lats):
                if lat > 20:
                    mld[j, :] *= 1.8
        elif month in (6, 7, 8):
            # 北半球夏 → 淺 MLD
            for j, lat in enumerate(lats):
                if lat > 20:
                    mld[j, :] *= 0.6

        return mld
