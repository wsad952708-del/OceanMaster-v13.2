"""
OceanMaster — GEBCO 海底地形特徵提取器
========================================
從 GEBCO 2024 / ETOPO 海底地形提取 ML 特徵。

學術依據:
  - Druon et al. (2012) PLOS ONE 7:e51581 — 鮪魚棲息地適宜度
  - Scales et al. (2014) J. Appl. Ecol. 51:1514 — 中尺度鋒面與覓食

靜態特徵 (只需下載一次):
  - depth: 海底深度 (m)
  - slope: 海底坡度 (°)
  - roughness: 粗糙度 (地形變化)
  - dist_to_seamount: 到最近海底山的距離 (km)
  - dist_to_shelf_break: 到大陸棚邊緣的距離 (km)

數據源: GEBCO 2024 (免費), 或系統已有的 ETOPO 回退
"""

import numpy as np
import logging
from typing import Optional, Dict, Tuple
from pathlib import Path
from scipy.ndimage import generic_filter, sobel

log = logging.getLogger("OceanMaster.GEBCO")


class GEBCOFeatures:
    """
    海底地形特徵提取器

    優先使用本地 GEBCO netCDF 檔案,
    回退使用 ETOPO1 via ERDDAP 或系統已有的 bathymetry 函數。
    """

    # 大陸棚邊緣深度定義
    SHELF_BREAK_DEPTH = -200   # m (大陸棚邊緣)
    SEAMOUNT_DEPTH = -1500     # m (海底山頂部閾值)
    SEAMOUNT_RISE = 1000       # m (海底山必須高出周圍至少 1000m)

    def __init__(self, gebco_path: Optional[str] = None):
        """
        Parameters:
            gebco_path: GEBCO netCDF 檔案路徑 (可選)
        """
        self.gebco_path = gebco_path
        self._bathy_cache = {}

    def compute_features(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        bathy: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        計算所有地形特徵。

        Parameters:
            lats: 1D 緯度陣列
            lons: 1D 經度陣列
            bathy: 2D 海底深度 (m, 負值), 若 None 則從 GEBCO/ETOPO 取得

        Returns:
            dict with: depth, slope, roughness,
                       dist_to_seamount, dist_to_shelf_break,
                       bpi_fine, bpi_broad
        """
        ny, nx = len(lats), len(lons)

        if bathy is None:
            bathy = self._get_bathymetry(lats, lons)

        if bathy is None or bathy.size == 0:
            log.warning("  GEBCO: No bathymetry data available")
            return self._empty_features(ny, nx)

        # 確保 bathy 是負值 (海底深度)
        if np.nanmean(bathy) > 0:
            bathy = -bathy

        # 1. 深度 (直接使用)
        depth = bathy.astype(np.float32)

        # 2. 坡度 (Sobel gradient)
        slope = self._compute_slope(bathy, lats, lons)

        # 3. 粗糙度 (局部標準差)
        roughness = self._compute_roughness(bathy)

        # 4. 距離最近海底山
        dist_seamount = self._dist_to_seamount(bathy, lats, lons)

        # 5. 距離大陸棚邊緣
        dist_shelf = self._dist_to_shelf_break(bathy, lats, lons)

        # 6. [v16.0] BPI 海底地形位置指數 (雙尺度)
        bpi_fine, bpi_broad = self._compute_bpi(bathy)

        log.info(f"  GEBCO features: depth=[{np.nanmin(depth):.0f}, "
                 f"{np.nanmax(depth):.0f}]m, "
                 f"slope avg={np.nanmean(slope):.2f}°, "
                 f"BPI_fine=[{np.nanmin(bpi_fine):.0f},{np.nanmax(bpi_fine):.0f}], "
                 f"seamount_dist avg={np.nanmean(dist_seamount):.0f}km")

        return {
            "depth": depth,
            "slope": slope,
            "roughness": roughness,
            "dist_to_seamount": dist_seamount,
            "dist_to_shelf_break": dist_shelf,
            "bpi_fine": bpi_fine,
            "bpi_broad": bpi_broad,
        }

    def _get_bathymetry(self, lats: np.ndarray, lons: np.ndarray) -> Optional[np.ndarray]:
        """取得海底深度資料"""

        # 嘗試 1: 本地 GEBCO netCDF
        if self.gebco_path and Path(self.gebco_path).exists():
            try:
                return self._load_gebco_nc(lats, lons)
            except Exception as e:
                log.debug(f"  GEBCO netCDF: {e}")

        # 嘗試 2: 使用系統已有的 BathymetryAnalyzer
        try:
            from engine.ocean_physics import BathymetryAnalyzer
            bathy = BathymetryAnalyzer.generate_bathymetry(lats, lons)
            if bathy is not None and np.any(bathy != 0):
                log.info("  GEBCO: using BathymetryAnalyzer (ETOPO1)")
                return bathy
        except ImportError as e:
            log.debug(f"[降級] engine/gebco_features.py: {e}")

        # 嘗試 3: 簡易估算 (基於距海岸距離)
        return self._estimate_bathymetry(lats, lons)

    def _load_gebco_nc(self, lats, lons):
        """從 GEBCO netCDF 檔案載入"""
        import xarray as xr
        ds = xr.open_dataset(self.gebco_path)
        bathy = ds["elevation"].sel(
            lat=slice(lats.min(), lats.max()),
            lon=slice(lons.min(), lons.max()),
        ).values
        ds.close()
        return bathy

    def _estimate_bathymetry(self, lats, lons) -> np.ndarray:
        """粗略估算海底深度 (當無真實數據時)"""
        ny, nx = len(lats), len(lons)
        depth = np.full((ny, nx), -4000.0, dtype=np.float32)

        for j, lat in enumerate(lats):
            abs_lat = abs(lat)
            for i, lon in enumerate(lons):
                # 西太平洋海底地形粗略模型
                if 120 <= lon <= 130:
                    # 大陸棚區
                    if abs_lat < 5:
                        depth[j, i] = -3000.0
                    else:
                        depth[j, i] = -200.0 - abs_lat * 50
                elif 126 <= lon <= 136 and 25 <= lat <= 35:
                    # 琉球海溝
                    depth[j, i] = -5000.0
                elif 140 <= lon <= 145 and 20 <= lat <= 35:
                    # 伊豆-小笠原海溝
                    depth[j, i] = -6000.0
                else:
                    # 一般深海
                    depth[j, i] = -4000.0 - np.sin(lon * 0.1) * 500

        return depth

    def _compute_bpi(self, bathy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        [v16.0] Bathymetric Position Index (BPI).

        BPI = depth[i,j] - mean(depth in annular neighborhood)
        Positive BPI = seamount/ridge crest (upwelling, fish aggregation)
        Negative BPI = trough/channel
        Zero BPI = flat plain or slope

        Returns fine-scale (3×3) and broad-scale (9×9) BPI.
        Ref: Wilson et al. (2007) Marine Geodesy 30:3-35
        """
        from scipy.ndimage import uniform_filter

        bathy_f = bathy.astype(np.float64)

        # Fine-scale BPI (inner radius ~1 cell, outer radius ~3 cells)
        mean_fine = uniform_filter(bathy_f, size=3, mode='nearest')
        bpi_fine = (bathy_f - mean_fine).astype(np.float32)

        # Broad-scale BPI (outer radius ~9 cells)
        mean_broad = uniform_filter(bathy_f, size=9, mode='nearest')
        bpi_broad = (bathy_f - mean_broad).astype(np.float32)

        return bpi_fine, bpi_broad

    def _compute_slope(self, bathy: np.ndarray,
                       lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """計算海底坡度 (度)"""
        # Sobel 梯度
        dx = sobel(bathy, axis=1)  # 東西方向
        dy = sobel(bathy, axis=0)  # 南北方向

        # 轉換為實際距離 (m)
        dlat = np.mean(np.abs(np.diff(lats))) * 111000  # m/格
        dlon = np.mean(np.abs(np.diff(lons))) * 111000 * np.cos(np.radians(np.mean(lats)))

        # 坡度 (度)
        gradient = np.sqrt((dx / max(dlon, 1))**2 + (dy / max(dlat, 1))**2)
        slope = np.degrees(np.arctan(gradient))

        return np.nan_to_num(slope, nan=0.0).astype(np.float32)

    def _compute_roughness(self, bathy: np.ndarray) -> np.ndarray:
        """計算地形粗糙度 (局部 3×3 標準差)"""
        try:
            roughness = generic_filter(
                bathy.astype(np.float64),
                np.std, size=3,
                mode='nearest',
            )
            return roughness.astype(np.float32)
        except Exception:
            return np.zeros_like(bathy, dtype=np.float32)

    def _dist_to_seamount(self, bathy: np.ndarray,
                          lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """
        計算每個網格點到最近海底山的距離 (km)。

        海底山定義: 深度 < SEAMOUNT_DEPTH 且高出周圍 > SEAMOUNT_RISE
        """
        ny, nx = bathy.shape
        dist = np.full((ny, nx), 500.0, dtype=np.float32)  # 預設 500km

        # 偵測海底山 (局部突起)
        try:
            # 局部平均深度 (5×5 窗口)
            mean_depth = generic_filter(
                bathy.astype(np.float64),
                np.mean, size=5,
                mode='nearest',
            )
            # 海底山 = 比周圍淺超過 1000m, 且本身深度 > 1500m
            rise = bathy - mean_depth  # 正值 = 比周圓淺
            seamount_mask = (rise > self.SEAMOUNT_RISE) & (bathy > self.SEAMOUNT_DEPTH)

            seamount_points = np.argwhere(seamount_mask)
        except Exception:
            seamount_points = np.array([])

        if len(seamount_points) == 0:
            # 沒偵測到海底山 → 用已知位置
            seamount_points = self._known_seamount_indices(lats, lons)

        if len(seamount_points) == 0:
            return dist

        # 計算距離 (Haversine 近似)
        for j in range(ny):
            for i in range(nx):
                min_d = 500.0
                lat1 = lats[j] if j < len(lats) else lats[-1]
                lon1 = lons[i] if i < len(lons) else lons[-1]
                for sj, si in seamount_points:
                    if sj < len(lats) and si < len(lons):
                        lat2 = lats[sj]
                        lon2 = lons[si]
                        d = self._haversine(lat1, lon1, lat2, lon2)
                        if d < min_d:
                            min_d = d
                dist[j, i] = min_d

        return dist

    def _dist_to_shelf_break(self, bathy: np.ndarray,
                             lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """計算到大陸棚邊緣 (-200m 等深線) 的距離 (km)"""
        ny, nx = bathy.shape
        dist = np.full((ny, nx), 300.0, dtype=np.float32)  # 預設 300km

        # 找 -200m 等深線
        shelf_mask = (bathy > self.SHELF_BREAK_DEPTH - 50) & \
                     (bathy < self.SHELF_BREAK_DEPTH + 50)
        shelf_points = np.argwhere(shelf_mask)

        if len(shelf_points) == 0:
            return dist

        # 計算距離 (向量化近似)
        for j in range(ny):
            lat1 = lats[j] if j < len(lats) else lats[-1]
            for i in range(nx):
                lon1 = lons[i] if i < len(lons) else lons[-1]
                min_d = 300.0
                # 取最近 N 個候選點
                dists_approx = np.abs(shelf_points[:, 0] - j) + \
                               np.abs(shelf_points[:, 1] - i)
                nearest_idx = np.argsort(dists_approx)[:5]
                for idx in nearest_idx:
                    sj, si = shelf_points[idx]
                    if sj < len(lats) and si < len(lons):
                        d = self._haversine(lat1, lon1, lats[sj], lons[si])
                        if d < min_d:
                            min_d = d
                dist[j, i] = min_d

        return dist

    def _known_seamount_indices(self, lats, lons) -> np.ndarray:
        """已知西太平洋海底山位置 (硬編碼)"""
        # 主要西太平洋海底山
        known = [
            (25.0, 131.0),   # 沖之鳥島附近
            (20.0, 136.0),   # 小笠原群島
            (15.0, 142.0),   # 馬里亞納
            (12.0, 145.0),   # 關島附近
            (22.0, 128.0),   # 琉球弧
            (19.0, 125.0),   # 巴士海峽
        ]
        indices = []
        for lat_k, lon_k in known:
            if (lats.min() <= lat_k <= lats.max() and
                    lons.min() <= lon_k <= lons.max()):
                j_idx = np.argmin(np.abs(lats - lat_k))
                i_idx = np.argmin(np.abs(lons - lon_k))
                indices.append([j_idx, i_idx])

        return np.array(indices) if indices else np.array([])

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2) -> float:
        """Haversine 距離 (km)"""
        R = 6371.0
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat / 2)**2 +
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
             np.sin(dlon / 2)**2)
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    def _empty_features(self, ny, nx) -> Dict:
        """空特徵 (回退)"""
        return {
            "depth": np.full((ny, nx), -4000.0, dtype=np.float32),
            "slope": np.zeros((ny, nx), dtype=np.float32),
            "roughness": np.zeros((ny, nx), dtype=np.float32),
            "dist_to_seamount": np.full((ny, nx), 300.0, dtype=np.float32),
            "dist_to_shelf_break": np.full((ny, nx), 200.0, dtype=np.float32),
            "bpi_fine": np.zeros((ny, nx), dtype=np.float32),
            "bpi_broad": np.zeros((ny, nx), dtype=np.float32),
        }
