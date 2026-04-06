"""
OceanMaster — 餌料場推算引擎 (Forage Engine)
=============================================
核心理念: 食物在哪 = 魚在哪。

食物鏈追蹤:
  太陽光 → 浮游植物 (CHL) → 淨初級生產力 (NPP)
  → 浮游動物 → 微型浮游動物 (micronekton) → 鮪魚

模型:
  1. VGPM (Behrenfeld & Falkowski 1997): CHL + SST + PAR → NPP
  2. Iverson 1990 營養轉換: NPP → 餌料生物量 (ε = 0.04)
  3. 垂直再分配: 表層 vs 深層餌料 (基於 Z20 / MLD)

數據源:
  - CHL: OceanDataFetcher 已擷取 (VIIRS)
  - SST: OceanDataFetcher 已擷取 (OISST)
  - PAR: MODIS erdMH1par0mday (ERDDAP) 或日照長度估算
"""

import numpy as np
import logging
from typing import Dict, Optional

log = logging.getLogger("OceanMaster.ForageEngine")


class ForageEngine:
    """
    餌料場推算引擎

    使用方法:
        engine = ForageEngine()
        result = engine.compute(chl, sst, lats, par=par_data, z20=z20, mld=mld)
        # result['npp']             → 淨初級生產力 (mg C/m²/day)
        # result['forage_total']    → 總餌料指數
        # result['forage_surface']  → 表層餌料 (0-MLD)
        # result['forage_deep']     → 深層餌料 (MLD-Z20)
    """

    # VGPM Pb_opt 多項式係數 (Behrenfeld & Falkowski 1997)
    VGPM_COEFFS = np.array([
        -3.27e-8, 3.4132e-6, -1.348e-4, 2.46e-3,
        -2.05e-2, 6.17e-2, 2.749e-1, 1.2956
    ])

    # 營養轉換效率 (Iverson 1990)
    EPSILON = 0.04  # 初級生產 → 中層生物量

    # [v11 Bug #2] 季節性雲量查找表 (cloud transmittance)
    # 行 = 月份(1-12), 列 = 緯度帶 (60S, 30S, EQ, 10N_ITCZ, 30N, 60N)
    # 值 = 雲量透過率 (0-1), 1=無雲
    # 基於 ISCCP + CERES 衛星數據 (Rossow & Schiffer 1999, Loeb 2018)
    SEASONAL_CLOUD_LUT = {
        #    60S   30S    EQ   10N   30N   60N
        1:  [0.42, 0.62, 0.50, 0.55, 0.58, 0.38],  # Jan
        2:  [0.40, 0.60, 0.48, 0.52, 0.55, 0.35],
        3:  [0.43, 0.58, 0.50, 0.50, 0.58, 0.40],
        4:  [0.45, 0.55, 0.52, 0.48, 0.60, 0.42],
        5:  [0.42, 0.52, 0.48, 0.45, 0.62, 0.45],
        6:  [0.38, 0.50, 0.45, 0.42, 0.65, 0.48],  # Jun: ITCZ north
        7:  [0.35, 0.48, 0.42, 0.40, 0.60, 0.50],  # Jul: monsoon
        8:  [0.38, 0.50, 0.44, 0.42, 0.58, 0.48],
        9:  [0.42, 0.55, 0.48, 0.45, 0.55, 0.42],
        10: [0.45, 0.58, 0.50, 0.50, 0.52, 0.38],
        11: [0.43, 0.60, 0.52, 0.55, 0.50, 0.35],
        12: [0.42, 0.62, 0.50, 0.55, 0.55, 0.36],  # Dec
    }
    CLOUD_LAT_BINS = [-60, -30, 0, 10, 30, 60]

    def compute_pb_opt(self, sst: np.ndarray) -> np.ndarray:
        """
        VGPM 最大碳固定率 Pb_opt (mg C / mg Chl / hr)

        七次多項式，SST 的函數 (Behrenfeld & Falkowski 1997)
        """
        sst_clipped = np.clip(sst, -2, 35)
        pb_opt = np.polyval(self.VGPM_COEFFS, sst_clipped)
        return np.clip(pb_opt, 0.1, 20.0)

    @staticmethod
    def compute_euphotic_depth(chl: np.ndarray) -> np.ndarray:
        """
        真光層深度 Z_eu (m) — Morel & Berthon 1989

        Z_eu = 568.2 × CHL^(-0.746)  (for open ocean, CHL < 10)
        """
        chl_safe = np.clip(chl, 0.01, 30.0)
        z_eu = 568.2 * np.power(chl_safe, -0.746)
        return np.clip(z_eu, 5.0, 300.0)

    @staticmethod
    def estimate_par(
        lats: np.ndarray,
        month: int,
        lons: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        [v11 Bug #2] 改進版 PAR 估算

        改進:
        1. 季節性雲量 LUT (ITCZ 修正)
        2. 經度依賴的雲量修正 (西太平洋暖池多雲)
        3. 更精確的天頂角計算

        Parameters:
            lats: 1D 緯度
            month: 月份 (1-12)
            lons: 1D 經度 (可選，用於區域雲量修正)

        Returns:
            PAR: 1D (Einstein/m²/day)
        """
        day_of_year = (month - 1) * 30 + 15
        solar_dec = 23.45 * np.sin(np.radians((360 / 365) * (day_of_year - 81)))

        # 取得本月雲量 LUT
        cloud_row = ForageEngine.SEASONAL_CLOUD_LUT.get(month, [0.50]*6)
        lat_bins = ForageEngine.CLOUD_LAT_BINS

        ny = len(lats)
        par = np.full(ny, 40.0)
        for j, lat in enumerate(lats):
            cos_zen = np.sin(np.radians(lat)) * np.sin(np.radians(solar_dec)) + \
                      np.cos(np.radians(lat)) * np.cos(np.radians(solar_dec))
            cos_zen = np.clip(cos_zen, 0.1, 1.0)

            # [v11] 從 LUT 插值雲量透過率
            cloud_trans = np.interp(lat, lat_bins, cloud_row)

            # [v11] 經度修正: 西太平洋暖池 (120-170E) 多雲
            if lons is not None and j < len(lons):
                lon_j = lons[j] if lons.ndim == 1 else 150.0
                # 暖池區域額外減少透過率
                warm_pool_penalty = 0.08 * np.exp(-((lon_j - 150)**2 / 1500))
                cloud_trans = max(cloud_trans - warm_pool_penalty, 0.25)

            # TOA PAR ≈ 55 E/m²/day, 經大氣衰減
            par[j] = 55.0 * cos_zen * cloud_trans

        return np.clip(par, 5.0, 60.0)

    @staticmethod
    def compute_day_length(lat: float, month: int) -> float:
        """日照長度 (hours)"""
        day_of_year = (month - 1) * 30 + 15
        solar_dec = 23.45 * np.sin(np.radians((360 / 365) * (day_of_year - 81)))

        cos_ha = -np.tan(np.radians(lat)) * np.tan(np.radians(solar_dec))
        cos_ha = np.clip(cos_ha, -1, 1)
        hour_angle = np.degrees(np.arccos(cos_ha))
        return 2.0 * hour_angle / 15.0

    def compute_npp_vgpm(self, chl: np.ndarray, sst: np.ndarray,
                         lats: np.ndarray,
                         par: Optional[np.ndarray] = None,
                         month: int = 6,
                         lons: Optional[np.ndarray] = None,
                         npp_model: str = 'auto') -> np.ndarray:
        """
        [v11 Bug #3] NPP 計算 — 支持多模型

        模型選擇:
          - 'vgpm':   標準 VGPM (Behrenfeld & Falkowski 1997)
          - 'eppley': Eppley-VGPM 變體 (溫度依賴性更強)
          - 'auto':   根據 Chl-a 自動切換
                       Chl < 0.15 → Eppley (寡營養海域)
                       Chl > 0.5  → 標準 VGPM (沿岸高 Chl)
                       中間 → 兩者加權混合

        Parameters:
            chl: 2D CHL-a (mg/m³)
            sst: 2D SST (°C)
            lats: 1D 緯度
            par: 2D PAR (Einstein/m²/day) — 可選
            month: 月份
            lons: 1D 經度 — 可選 (用於改進 PAR 估算)
            npp_model: 'vgpm' | 'eppley' | 'auto'

        Returns:
            NPP: 2D (mg C / m² / day)
        """
        ny, nx = chl.shape

        # PAR
        if par is None:
            par_1d = self.estimate_par(lats, month, lons=lons)
            par_2d = np.tile(par_1d.reshape(-1, 1), (1, nx))
        elif par.ndim == 1:
            par_2d = np.tile(par.reshape(-1, 1), (1, nx))
        else:
            par_2d = par

        # 光限制因子
        f_light = par_2d / (par_2d + 4.1)

        # 真光層深度
        z_eu = self.compute_euphotic_depth(chl)

        # 日照長度
        day_lengths = np.array([self.compute_day_length(lat, month) for lat in lats])
        dl_2d = np.tile(day_lengths.reshape(-1, 1), (1, nx))

        # [v11] 根據模型選擇計算 Pb_opt
        if npp_model == 'eppley':
            pb_opt = self._compute_pb_opt_eppley(sst)
        elif npp_model == 'vgpm':
            pb_opt = self.compute_pb_opt(sst)
        else:  # 'auto'
            pb_standard = self.compute_pb_opt(sst)
            pb_eppley = self._compute_pb_opt_eppley(sst)

            # 根據 Chl-a 濃度混合
            chl_median = np.nanmedian(chl)
            if chl_median < 0.15:
                # 寡營養海域: 純 Eppley
                pb_opt = pb_eppley
                log.info("  NPP model: Eppley-VGPM (low Chl)")
            elif chl_median > 0.5:
                # 沿岸/高營養: 純標準 VGPM
                pb_opt = pb_standard
                log.info("  NPP model: Standard VGPM (high Chl)")
            else:
                # 中間: 加權混合
                alpha = (chl_median - 0.15) / (0.5 - 0.15)  # 0→1
                pb_opt = (1 - alpha) * pb_eppley + alpha * pb_standard
                log.info(f"  NPP model: Blended (α={alpha:.2f})")

        # NPP 公式
        npp = 0.66125 * pb_opt * f_light * np.maximum(chl, 0.01) * z_eu * dl_2d
        npp = np.clip(npp, 0, 5000)

        return npp.astype(np.float32), z_eu.astype(np.float32)

    def _compute_pb_opt_eppley(self, sst: np.ndarray) -> np.ndarray:
        """
        [v11 Bug #3] Eppley-VGPM: 溫度依賴性更強的 Pb_opt

        Eppley (1972) 公式: Pb_opt = 1.54 × 10^(0.0275 × SST)

        相比標準 VGPM:
        - 熱帶 (SST>28°C) 產生更高 Pb_opt → 更高 NPP
        - 冷水 (SST<15°C) 產生更低 Pb_opt → 更低 NPP
        - 在寡營養海域 (低 Chl) 表現更佳

        文獻: Carr et al. 2006 比較，Eppley 在熱帶改善 ~20%
        """
        sst_clipped = np.clip(sst, -2, 35)
        pb_opt = 1.54 * np.power(10.0, 0.0275 * sst_clipped)
        return np.clip(pb_opt, 0.1, 20.0)

    def compute(self, chl: np.ndarray, sst: np.ndarray,
                lats: np.ndarray,
                par: Optional[np.ndarray] = None,
                z20: Optional[np.ndarray] = None,
                mld: Optional[np.ndarray] = None,
                month: int = 6,
                lons: Optional[np.ndarray] = None,
                npp_model: str = 'auto') -> Dict:
        """
        完整餌料場計算

        Returns:
            dict with:
              - npp:            淨初級生產力 (mg C/m²/day)
              - z_euphotic:     真光層深度 (m)
              - forage_total:   總餌料指數 (0-1)
              - forage_surface: 表層餌料 (0-1)
              - forage_deep:    深層餌料 (0-1)
              - pb_opt:         最大碳固定率
        """
        ny, nx = chl.shape

        # Step 1: NPP + Z_eu  [v11: 支持多模型]
        npp, z_eu = self.compute_npp_vgpm(
            chl, sst, lats, par, month,
            lons=lons, npp_model=npp_model
        )

        # Step 2: 餌料生物量 (Iverson 1990)
        forage = self.EPSILON * npp  # mg C/m²/day 到中層轉換

        # Step 3: 歸一化
        f_max = np.nanpercentile(forage, 99) if np.any(forage > 0) else 1.0
        forage_norm = np.clip(forage / max(f_max, 0.01), 0, 1).astype(np.float32)

        # Step 4: 垂直再分配
        if z20 is not None and mld is not None:
            deep_fraction = np.clip((z20 - mld) / np.maximum(z20, 10), 0.2, 0.8)
            forage_surface = forage_norm * (1 - deep_fraction)
            forage_deep = forage_norm * deep_fraction
        else:
            forage_surface = forage_norm * 0.5
            forage_deep = forage_norm * 0.5

        log.info(f"  NPP: {np.nanmean(npp):.0f} mg C/m²/day (avg), "
                 f"max={np.nanmax(npp):.0f}")
        log.info(f"  Forage: surface={np.nanmean(forage_surface):.2f}, "
                 f"deep={np.nanmean(forage_deep):.2f}")

        return {
            "npp": npp,
            "z_euphotic": z_eu,
            "forage_total": forage_norm,
            "forage_surface": forage_surface.astype(np.float32),
            "forage_deep": forage_deep.astype(np.float32),
            "pb_opt": self.compute_pb_opt(sst).astype(np.float32),
        }
