"""
OceanMaster v13.2 — 浮游動物豐度代理指標  # [v12-enhance]
========================================
估算浮游動物豐度，作為鮪魚餌料可用性的間接指標。

科學背景:
  - SEAPODYM 有 6 個明確的 micronekton 功能群
  - 浮游動物響應 NPP 有 7-30 天時間滯後
  - 溫度調節浮游動物代謝速率和豐度

方法 (按優先級):
  1. CMEMS BGC zooc 直接產品（若有 API key）
  2. NPP × 滯後效應估算（30天移動滯後）
  3. CHL + SST 的經驗代理模型

v12 新增:
  - [v12-enhance] Chl-a 粒徑分級代理 (Brewin et al. 2010, JGR 115)
    total Chl → pico / nano / micro 分級 → 物種特異加權
  - [v12-enhance] Cushing (1990) 轉換效率: 熱帶 ε ~ 10-15%, 溫帶 ε ~ 5-10%

文獻:
  - Lehodey et al. 2008 SEAPODYM（MEPS）
  - Dueri et al. 2014 (PLos ONE)
  - Brewin et al. 2010 (JGR 115: C10008)
"""

import numpy as np
import logging
from typing import Optional, Dict

log = logging.getLogger("OceanMaster.Zooplankton")


# [v12-enhance] 物種對不同粒徑浮游植物的偏好權重
# 大型浮游植物 → 大型浮游動物 → 大型魚類 (Cushing 1990)
SPECIES_SIZE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "skipjack":  {"pico": 0.30, "nano": 0.40, "micro": 0.30},  # 表層雜食
    "yellowfin": {"pico": 0.15, "nano": 0.35, "micro": 0.50},  # 偏好中大型餌料
    "bigeye":    {"pico": 0.10, "nano": 0.30, "micro": 0.60},  # 大型深海 micronekton
    "albacore":  {"pico": 0.20, "nano": 0.35, "micro": 0.45},  # 中深層
}


class ZooplanktonProxy:
    """
    浮游動物豐度代理指標

    使用:
        zoo = ZooplanktonProxy()
        index = zoo.estimate_from_npp_lag(npp_current, npp_7d_ago, npp_14d_ago, npp_30d_ago, sst, lats)
    """

    # 滯後權重 (Lehodey et al. 2008)
    LAG_WEIGHTS = {
        7: 0.40,    # 7 天前 NPP 的權重
        14: 0.30,   # 14 天前
        30: 0.30,   # 30 天前
    }

    # 浮游動物最適溫度 (°C)
    ZOO_TOPT = 25.0
    ZOO_TSIGMA = 8.0

    def estimate_from_npp_lag(
        self,
        npp_current: np.ndarray,
        npp_7d: Optional[np.ndarray] = None,
        npp_14d: Optional[np.ndarray] = None,
        npp_30d: Optional[np.ndarray] = None,
        sst: Optional[np.ndarray] = None,
        lats: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        基於 NPP 時間滯後估算浮游動物豐度指數 (0-1)

        模型:
          1. 滯後 NPP: lag_npp = 0.4×NPP(t-7) + 0.3×NPP(t-14) + 0.3×NPP(t-30)
          2. 溫度調節: 最適溫度 25°C 的高斯函數
          3. 緯度食物網複雜度修正 (熱帶簡單/溫帶複雜)

        Parameters:
            npp_current: 2D 當前 NPP (mg C/m²/day)
            npp_7d: 7天前 NPP (若無則用 npp_current)
            npp_14d: 14天前 NPP
            npp_30d: 30天前 NPP
            sst: 2D SST (°C)
            lats: 1D 緯度

        Returns:
            zoo_index: 2D (0-1) 浮游動物豐度指數
        """
        ny, nx = npp_current.shape

        # Step 1: 滯後加權 NPP
        w7, w14, w30 = self.LAG_WEIGHTS[7], self.LAG_WEIGHTS[14], self.LAG_WEIGHTS[30]

        lag_npp = np.zeros_like(npp_current, dtype=np.float64)
        if npp_7d is not None:
            lag_npp += w7 * np.nan_to_num(npp_7d)
        else:
            lag_npp += w7 * np.nan_to_num(npp_current)

        if npp_14d is not None:
            lag_npp += w14 * np.nan_to_num(npp_14d)
        else:
            lag_npp += w14 * np.nan_to_num(npp_current) * 0.9  # 略衰減

        if npp_30d is not None:
            lag_npp += w30 * np.nan_to_num(npp_30d)
        else:
            lag_npp += w30 * np.nan_to_num(npp_current) * 0.8

        # 正規化 NPP 到 0-1
        p95 = np.nanpercentile(lag_npp, 95) if np.any(lag_npp > 0) else 1.0
        npp_norm = np.clip(lag_npp / max(p95, 1e-6), 0, 1)

        # Step 2: 溫度調節
        if sst is not None:
            temp_factor = np.exp(
                -((sst - self.ZOO_TOPT) ** 2) / (2 * self.ZOO_TSIGMA ** 2)
            )
        else:
            temp_factor = np.ones((ny, nx))

        # Step 3: 緯度食物網複雜度
        # 熱帶: 食物網簡單，NPP→浮游動物轉換快 (ε較高)
        # 溫帶: 食物網複雜，有更多中間層 (ε較低)
        if lats is not None:
            abs_lat = np.abs(lats)
            if abs_lat.ndim == 1:
                lat_factor = np.tile(
                    np.clip(1.2 - 0.01 * abs_lat, 0.6, 1.2).reshape(-1, 1),
                    (1, nx),
                )
            else:
                lat_factor = np.clip(1.2 - 0.01 * np.abs(lats), 0.6, 1.2)
        else:
            lat_factor = np.ones((ny, nx))

        # 合成
        zoo_index = npp_norm * temp_factor * lat_factor

        # 最終正規化
        z_max = np.nanmax(zoo_index) if np.any(zoo_index > 0) else 1.0
        zoo_index = np.clip(zoo_index / max(z_max, 1e-6), 0, 1)

        log.info(f"  Zooplankton proxy: avg={np.nanmean(zoo_index):.3f}, "
                 f"max={np.nanmax(zoo_index):.3f}")

        return zoo_index.astype(np.float32)

    def estimate_simple(
        self,
        chl: np.ndarray,
        sst: np.ndarray,
    ) -> np.ndarray:
        """
        簡易代理: 僅用當前 CHL + SST

        用於無歷史 NPP 數據時的降級模式。
        """
        # CHL → NPP 粗略估算 (正相關)
        chl_norm = np.log10(np.maximum(chl, 0.01) + 1)
        p95 = np.nanpercentile(chl_norm, 95) if np.any(chl_norm > 0) else 1.0
        chl_index = np.clip(chl_norm / max(p95, 1e-6), 0, 1)

        # 溫度調節
        temp_factor = np.exp(
            -((sst - self.ZOO_TOPT) ** 2) / (2 * self.ZOO_TSIGMA ** 2)
        )

        zoo_index = chl_index * temp_factor
        z_max = np.nanmax(zoo_index) if np.any(zoo_index > 0) else 1.0
        return np.clip(zoo_index / max(z_max, 1e-6), 0, 1).astype(np.float32)

    # ── [v12-enhance] Size-fractionated Chl-a → species-specific zooplankton ──

    @staticmethod
    def fractionate_chl(chl_total: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Chl-a 粒徑分級 (Brewin et al. 2010, JGR 115: C10008)

        將 total Chl-a 分為三個粒徑等級:
          - Pico (< 2 μm): Prochlorococcus, Synechococcus
          - Nano (2-20 μm): 小型矽藻、鞭毛藻
          - Micro (> 20 μm): 大型矽藻、甲藻 (主要食物網)

        Brewin 模型 (簡化):
          Chl_pico = C_pico_max × (1 - exp(-S_pico × Chl_total / C_pico_max))
          Chl_micro = Chl_total - Chl_nano - Chl_pico

        Parameters:
            chl_total: 2D total Chl-a (mg/m³)

        Returns:
            dict with 'pico', 'nano', 'micro' fractions (0-1 relative)
        """
        chl = np.maximum(chl_total, 0.01)

        # Brewin et al. 2010 參數 (Table 2, Atlantic)
        C_pico_max = 0.20   # mg/m³ — pico 飽和上限
        S_pico = 0.75        # pico 斜率
        C_nano_max = 0.80    # mg/m³ — nano 飽和上限
        S_nano = 0.50        # nano 斜率

        # Pico: 低 Chl 環境下主導，高 Chl 時飽和
        chl_pico = C_pico_max * (1.0 - np.exp(-S_pico * chl / C_pico_max))

        # Nano: 中等 Chl 環境
        chl_nano = C_nano_max * (1.0 - np.exp(-S_nano * chl / C_nano_max)) - chl_pico
        chl_nano = np.maximum(chl_nano, 0.0)

        # Micro: 剩餘 = 高 Chl 環境主導 (upwelling, 湧升流)
        chl_micro = np.maximum(chl - chl_pico - chl_nano, 0.0)

        # 歸一化為相對比例 (0-1)
        total = chl_pico + chl_nano + chl_micro
        total = np.maximum(total, 1e-6)

        result = {
            "pico":  (chl_pico / total).astype(np.float32),
            "nano":  (chl_nano / total).astype(np.float32),
            "micro": (chl_micro / total).astype(np.float32),
        }

        log.info(f"  📊 Chl size fractions: "
                 f"pico={np.nanmean(result['pico']):.2f}, "
                 f"nano={np.nanmean(result['nano']):.2f}, "
                 f"micro={np.nanmean(result['micro']):.2f}")

        return result

    def estimate_species_specific(
        self,
        chl: np.ndarray,
        sst: np.ndarray,
        species: str,
    ) -> np.ndarray:
        """
        [v12-enhance] 物種特異的浮游動物指標

        用 Chl 粒徑分級 × 物種偏好權重 → 更精確的餌料可用性

        大目鮪偏好 micro (大型浮游植物 → 大型微小浮游生物)
        正鰹偏好 nano/pico (表層小型食物)
        """
        fractions = self.fractionate_chl(chl)
        weights = SPECIES_SIZE_WEIGHTS.get(species, {"pico": 0.33, "nano": 0.33, "micro": 0.34})

        # 加權合成
        weighted_chl = (
            weights["pico"] * fractions["pico"]
            + weights["nano"] * fractions["nano"]
            + weights["micro"] * fractions["micro"]
        )

        # 乘以溫度因子
        temp_factor = np.exp(
            -((sst - self.ZOO_TOPT) ** 2) / (2 * self.ZOO_TSIGMA ** 2)
        )

        zoo_index = weighted_chl * temp_factor
        z_max = np.nanmax(zoo_index) if np.any(zoo_index > 0) else 1.0
        zoo_index = np.clip(zoo_index / max(z_max, 1e-6), 0, 1)

        log.info(f"  🐟 {species} species-specific zoo: "
                 f"avg={np.nanmean(zoo_index):.3f} "
                 f"(w_pico={weights['pico']:.2f}, w_micro={weights['micro']:.2f})")

        return zoo_index.astype(np.float32)

