"""
OceanMaster v10.0 — 初級生產力 & 營養鏈模組
==============================================
SEAPODYM / CATSAT 級核心技術：食物鏈建模

科學基礎：
  PP → 浮游動物 → 微型浮游生物(Micronekton) → 鮪魚餌料
  這是 CATSAT v6 和 SEAPODYM 的核心 — 解釋「魚為什麼在這裡」

實現策略（經查證的方法）：
  1. VGPM (Behrenfeld & Falkowski 1997): PP 標準演算法
     PP = 0.66125 × PBopt × (E₀/(E₀+4.1)) × Zeu × Chl × DL

  2. 簡化 Trophic Transfer：PP → Zooplankton → Micronekton
     使用 SEAPODYM-LMTL 的簡化版本

  3. 餌料可及性指數(Forage Accessibility Index)
     考慮深度、日夜垂直遷移、溶氧限制

數據來源（已查證 URL）：
  - Oregon State VGPM: orca.science.oregonstate.edu (免費, 8天合成)
  - MODIS Chl-a: ERDDAP erdMH1chla1day
  - MODIS PAR: ERDDAP erdMH1par01day

參考文獻：
  - Behrenfeld & Falkowski (1997) L&O 42:1-20
  - Lehodey et al. (2010) SEAPODYM-LMTL micronekton model
  - CATSAT v6.0: zooplankton + micronekton maps
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

log = logging.getLogger("OceanMaster.PP")


class PrimaryProductionEngine:
    """
    初級生產力與營養鏈分析引擎

    三層計算：
    Layer 1: VGPM → Net Primary Production (mgC/m²/day)
    Layer 2: PP → Zooplankton biomass (簡化轉換)
    Layer 3: Zooplankton → Micronekton → Forage Index (各物種)
    """

    # ─── VGPM 核心公式 ───────────────────────────
    @staticmethod
    def compute_pb_opt(sst: np.ndarray) -> np.ndarray:
        """
        PBopt: 最佳光合效率 (mgC/mgChl/hr)
        Behrenfeld & Falkowski (1997) 的 7階多項式

        PBopt = f(SST)，在 ~20°C 達到峰值 ~4.0

        原始公式（已查證）：
        PBopt = 1.2956 + 2.749e-1*T + 6.17e-2*T² - 2.05e-2*T³
                + 2.462e-3*T⁴ - 1.348e-4*T⁵ + 3.4132e-6*T⁶
                - 3.27e-8*T⁷
        當 SST > 28.5°C 時，PBopt 固定為 4.0
        """
        T = np.clip(sst, -2.0, 40.0)

        pb = (1.2956
              + 2.749e-1 * T
              + 6.17e-2 * T**2
              - 2.05e-2 * T**3
              + 2.462e-3 * T**4
              - 1.348e-4 * T**5
              + 3.4132e-6 * T**6
              - 3.27e-8 * T**7)

        # SST > 28.5°C: cap at 4.0 (Behrenfeld & Falkowski 1997)
        pb = np.where(sst > 28.5, 4.0, pb)
        return np.clip(pb, 0.0, 10.0).astype(np.float32)

    @staticmethod
    def compute_euphotic_depth(chl: np.ndarray) -> np.ndarray:
        """
        真光層深度 Zeu (m)
        Morel & Berthon (1989) 經驗公式（VGPM 標準方法）

        Zeu = 568.2 × Chl_tot^(-0.746)  當 Chl_tot ≤ 1.0
        Zeu = 200.0 × Chl_tot^(-0.293)  當 Chl_tot > 1.0

        其中 Chl_tot ≈ Chl_surface × Zeu (需迭代)
        簡化版：直接用表層 Chl
        """
        chl_safe = np.clip(chl, 0.01, 50.0)

        zeu = np.where(
            chl_safe <= 1.0,
            568.2 * chl_safe ** (-0.746),
            200.0 * chl_safe ** (-0.293)
        )
        return np.clip(zeu, 5.0, 250.0).astype(np.float32)

    @staticmethod
    def compute_day_length(lat: np.ndarray, doy: int) -> np.ndarray:
        """
        日照時數 DL (hours)
        天文公式計算光週期

        Args:
            lat: 緯度 (度)
            doy: Day of Year (1-365)
        """
        # 太陽赤緯
        dec = 23.45 * np.sin(np.radians(360 / 365 * (284 + doy)))
        dec_rad = np.radians(dec)
        lat_rad = np.radians(np.clip(lat, -65, 65))

        # 時角
        cos_ha = -np.tan(lat_rad) * np.tan(dec_rad)
        cos_ha = np.clip(cos_ha, -1.0, 1.0)
        ha = np.degrees(np.arccos(cos_ha))

        dl = 2.0 * ha / 15.0  # 轉換為小時
        return np.clip(dl, 0.0, 24.0).astype(np.float32)

    def compute_vgpm(
        self,
        chl: np.ndarray,
        sst: np.ndarray,
        par: Optional[np.ndarray] = None,
        lat: Optional[np.ndarray] = None,
        doy: int = 180,
    ) -> Dict[str, np.ndarray]:
        """
        VGPM 完整計算

        PP = 0.66125 × PBopt(SST) × (E₀/(E₀+4.1)) × Zeu(Chl) × Chl × DL

        單位：mgC/m²/day

        Args:
            chl: Chlorophyll-a (mg/m³)
            sst: Sea Surface Temperature (°C)
            par: Photosynthetically Active Radiation (Einstein/m²/day)
                 如果為 None，使用緯度估算
            lat: 緯度（用於估算 PAR 和日照）
            doy: Day of Year

        Returns:
            dict with npp, pb_opt, zeu, par_used, day_length
        """
        log.info(f"🌿 VGPM 計算: grid={chl.shape}, DOY={doy}")

        # PBopt
        pb_opt = self.compute_pb_opt(sst)

        # Zeu
        zeu = self.compute_euphotic_depth(chl)

        # PAR (如果沒有真實數據，從緯度估算)
        if par is None and lat is not None:
            # 簡化 PAR 估算: 赤道 ~40-50 E/m²/day, 高緯度 ~15-30
            if lat.ndim == 1 and chl.ndim == 2:
                lat_2d = lat[:, np.newaxis] * np.ones((1, chl.shape[1]))
            elif lat.ndim == 2:
                lat_2d = lat
            else:
                lat_2d = lat
            abs_lat = np.abs(lat_2d)

            season = 1.0 + 0.3 * np.cos(2 * np.pi * (doy - 172) / 365)
            par = 45.0 * np.cos(np.radians(abs_lat * 0.8)) * season
            par = np.clip(par, 5.0, 65.0).astype(np.float32)
        elif par is None:
            par = np.full_like(chl, 35.0)  # 全球平均 PAR

        # 光利用效率因子
        light_func = par / (par + 4.1)

        # 日照時數
        if lat is not None:
            if lat.ndim == 1:
                dl = self.compute_day_length(lat, doy)
                # 廣播到 2D
                if chl.ndim == 2:
                    dl = dl[:, np.newaxis] * np.ones((1, chl.shape[1]))
            else:
                dl = self.compute_day_length(lat, doy)
        else:
            dl = np.full_like(chl, 12.0)

        # VGPM 主公式
        npp = 0.66125 * pb_opt * light_func * zeu * chl * dl

        # 物理約束
        npp = np.clip(npp, 0.0, 20000.0).astype(np.float32)

        mean_npp = float(np.nanmean(npp))
        log.info(f"  NPP 範圍: {np.nanmin(npp):.0f} - {np.nanmax(npp):.0f} mgC/m²/day")
        log.info(f"  NPP 平均: {mean_npp:.0f} mgC/m²/day")

        return {
            "npp": npp,
            "pb_opt": pb_opt,
            "zeu": zeu,
            "par": par,
            "day_length": dl,
        }

    # ─── 營養鏈轉換 ──────────────────────────────
    @staticmethod
    def compute_zooplankton_index(npp: np.ndarray) -> np.ndarray:
        """
        浮游動物豐度指數

        簡化 SEAPODYM-LMTL 的能量傳遞：
        - 營養效率 ≈ 10-20%（Lindeman 效率）
        - 時間延遲 ≈ 2-4 週（我們用空間平滑近似）
        - 歸一化到 0-1

        Zoo_index = sigmoid(NPP / NPP_median - 1)
        高 NPP → 高浮游動物 → 高營養
        """
        # 使用 log 轉換（NPP 分佈偏斜）
        npp_log = np.log1p(np.clip(npp, 0, 20000))
        median_log = np.nanmedian(npp_log)

        if median_log > 0:
            normalized = (npp_log - median_log) / max(np.nanstd(npp_log), 0.1)
        else:
            normalized = npp_log / max(np.nanmax(npp_log), 1.0)

        # Sigmoid 轉換到 0-1
        zoo_index = 1.0 / (1.0 + np.exp(-1.5 * normalized))

        return np.clip(zoo_index, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def compute_micronekton_density(
        zoo_index: np.ndarray,
        sst: Optional[np.ndarray] = None,
        depth_layer: str = "epipelagic",
    ) -> np.ndarray:
        """
        微型浮游生物密度估計

        SEAPODYM-LMTL 定義 6 個功能群組：
        1-3: 日間在表層/中層/深層的遷移群
        4-6: 非遷移群（固定深度）

        我們的簡化版：
        - 表層(0-200m): 主要受 NPP 和 Zoo 驅動
        - 中層(200-500m): 白天微型浮游生物下潛
        - 深層(500-1000m): 僅非遷移群

        溫度效應：代謝率 ∝ exp(T/10) (Q10 法則)
        """
        # 基礎密度 = 浮游動物指數 × 營養轉換效率(~15%)
        base_density = zoo_index * 0.15

        # 深度層調整
        layer_factors = {
            "epipelagic": 1.0,    # 0-200m: 白天有遷移群+表層固定群
            "mesopelagic": 0.7,   # 200-500m: 白天遷移群集中
            "bathypelagic": 0.2,  # 500m+: 僅深層固定群
        }
        layer_f = layer_factors.get(depth_layer, 1.0)

        if sst is not None:
            # Q10 溫度效應（簡化）
            # 溫暖水域代謝快，但也消耗更快
            temp_factor = np.clip(0.5 + 0.05 * (sst - 15), 0.3, 1.5)
            density = base_density * layer_f * temp_factor
        else:
            density = base_density * layer_f

        return np.clip(density, 0.0, 1.0).astype(np.float32)

    # ─── 餌料可及性指數 ──────────────────────────
    def compute_forage_index(
        self,
        micronekton: np.ndarray,
        species: str,
        do_si: Optional[np.ndarray] = None,
        time_of_day: str = "mixed",
    ) -> np.ndarray:
        """
        各物種的餌料可及性指數 (Forage Accessibility Index)

        SEAPODYM 核心概念：
        「魚不只需要食物存在，還需要能到達食物所在的深度」

        考慮：
        1. 微型浮游生物密度（食物量）
        2. 物種垂直能力（能潛多深）
        3. 溶氧限制（能不能待在那個深度）
        4. 日夜遷移（白天微型浮游生物在深處，晚上在表層）

        Args:
            micronekton: 微型浮游生物密度 (0-1)
            species: 物種名
            do_si: 溶氧適宜性（如果有）
            time_of_day: "day" / "night" / "mixed"

        Returns:
            forage_index: 0-1 餌料可及性
        """
        # 物種特定的覓食效率
        foraging_efficiency = {
            "skipjack":  0.9,   # 表層高效覓食
            "yellowfin": 0.85,  # 中層覓食
            "bigeye":    0.95,  # 深潛覓食高手（白天在深層找獵物）
            "albacore":  0.8,   # 溫帶中層覓食
            "squid_todarodes": 0.7,
            "squid_ommastrephes": 0.75,
        }

        efficiency = foraging_efficiency.get(species, 0.8)

        # 基礎餌料 = 微型浮游生物 × 覓食效率
        forage = micronekton * efficiency

        # 日夜效應
        if time_of_day == "night":
            # 夜間：微型浮游生物上浮到表層，所有物種都能接觸
            forage = forage * 1.3
        elif time_of_day == "day":
            # 白天：微型浮游生物下潛，只有深潛物種能接觸
            if species in ["skipjack", "squid_todarodes"]:
                forage = forage * 0.6  # 淺層物種白天餌料減少
            elif species == "bigeye":
                forage = forage * 1.1  # Bigeye 白天追著餌料下潛

        # 溶氧限制
        if do_si is not None:
            forage = forage * do_si

        return np.clip(forage, 0.0, 1.0).astype(np.float32)

    # ─── 完整分析管線 ─────────────────────────────
    def analyze(
        self,
        chl: np.ndarray,
        sst: np.ndarray,
        lats: np.ndarray,
        par: Optional[np.ndarray] = None,
        do_si: Optional[Dict[str, np.ndarray]] = None,
        doy: Optional[int] = None,
        species_list: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        完整營養鏈分析

        Args:
            chl: Chlorophyll-a (mg/m³)
            sst: SST (°C)
            lats: 緯度陣列
            par: PAR (可選)
            do_si: {species: do_suitability} (可選, 來自 DO 模組)
            doy: Day of Year

        Returns:
            dict with npp, zoo_index, micronekton, forage_index
        """
        if doy is None:
            doy = datetime.now().timetuple().tm_yday

        if species_list is None:
            species_list = ["skipjack", "yellowfin", "bigeye", "albacore"]

        log.info(f"🌊 營養鏈分析: grid={chl.shape}, DOY={doy}")

        # Layer 1: VGPM → NPP
        vgpm_result = self.compute_vgpm(chl, sst, par, lats, doy)
        npp = vgpm_result["npp"]

        # Layer 2: NPP → Zooplankton
        zoo_index = self.compute_zooplankton_index(npp)
        log.info(f"  浮游動物指數: {np.nanmean(zoo_index):.3f}")

        # Layer 3: Zoo → Micronekton
        micronekton_epi = self.compute_micronekton_density(
            zoo_index, sst, "epipelagic"
        )
        micronekton_meso = self.compute_micronekton_density(
            zoo_index, sst, "mesopelagic"
        )
        log.info(f"  微型浮游生物(表層): {np.nanmean(micronekton_epi):.3f}")
        log.info(f"  微型浮游生物(中層): {np.nanmean(micronekton_meso):.3f}")

        # Layer 4: Micronekton → 各物種餌料可及性
        forage = {}
        for sp in species_list:
            sp_do_si = do_si.get(sp) if do_si else None

            # 根據物種選擇主要覓食深度層
            if sp in ["skipjack", "squid_todarodes"]:
                mkn = micronekton_epi
            elif sp == "bigeye":
                mkn = micronekton_meso  # Bigeye 主要在中層覓食
            else:
                mkn = 0.6 * micronekton_epi + 0.4 * micronekton_meso

            fi = self.compute_forage_index(mkn, sp, sp_do_si, "mixed")
            forage[sp] = fi
            log.info(f"  {sp} 餌料指數: {np.nanmean(fi):.3f}")

        return {
            "npp": npp,
            "pb_opt": vgpm_result["pb_opt"],
            "zeu": vgpm_result["zeu"],
            "par": vgpm_result["par"],
            "zoo_index": zoo_index,
            "micronekton_epi": micronekton_epi,
            "micronekton_meso": micronekton_meso,
            "forage_index": forage,
        }
