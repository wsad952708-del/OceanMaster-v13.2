"""
OceanMaster — NPZ 生態動力模型 (Nutrient-Phytoplankton-Zooplankton)
===================================================================
簡化版 NPZ 模型，作為 VGPM 靜態估算的動態補充。

三個狀態變數:
  N — 營養鹽 (mmol N/m³)
  P — 浮游植物 (mmol N/m³)
  Z — 浮游動物 (mmol N/m³)

方程式 (Fasham 1990 / Franks 2002 簡化版):
  dP/dt = μ_max × f(N) × f(light) × P  −  g_max × f(P) × Z  −  m_P × P
  dZ/dt = γ × g_max × f(P) × Z  −  m_Z × Z²
  dN/dt = −dP/dt_growth + recycling

f(N) = N / (K_N + N)              Michaelis-Menten 營養限制
f(light) = I / (K_I + I)          光限制
f(P) = P² / (K_P² + P²)          Holling Type III 攝食

參數來源:
  - μ_max: Eppley 1972 溫度依賴
  - g_max, K_P: Franks 2002
  - 物種獵物列表: species_params.py

文獻:
  Fasham et al. (1990) J Marine Res
  Franks (2002) J Marine Systems
  Eppley (1972) Fishery Bulletin
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple

log = logging.getLogger("OceanMaster.NPZ")


# ═══════════════════════════════════════════════════════
# NPZ 預設參數
# ═══════════════════════════════════════════════════════

DEFAULT_PARAMS = {
    # 浮游植物
    "mu_max": 2.0,          # 最大比生長率 (day⁻¹) @ 20°C
    "K_N": 0.5,             # 營養鹽半飽和常數 (mmol N/m³)
    "K_I": 15.0,            # 光半飽和 (Einstein/m²/day)
    "m_P": 0.1,             # 浮游植物自然死亡率 (day⁻¹)

    # 浮游動物
    "g_max": 1.0,           # 最大攝食率 (day⁻¹)
    "K_P": 1.0,             # 攝食半飽和 (mmol N/m³)
    "gamma": 0.3,           # 攝食同化效率
    "m_Z": 0.2,             # 浮游動物死亡率 (day⁻¹·(mmol N/m³)⁻¹)

    # 營養鹽
    "recycling": 0.1,       # 回收率 (day⁻¹)
    "N_deep": 10.0,         # 深層營養鹽補充 (mmol N/m³)

    # 積分
    "dt": 0.1,              # 時間步長 (天)
    "n_days": 30,           # 積分天數
}

# 物種特定的浮游動物偏好權重
# key = 獵物名稱 (from species_params.prey), value = Z 權重加成
PREY_ZOO_WEIGHT = {
    "zooplankton": 1.0,
    "copepods": 0.9,
    "euphausiids": 0.85,
    "small_crustaceans": 0.7,
    "squid": 0.3,            # 高營養級, 間接依賴 Z
    "small_pelagics": 0.4,
    "flying_fish": 0.3,
    "skipjack": 0.1,         # 頂級掠食者, 弱依賴
    "mahi_mahi": 0.1,
}


class NPZModel:
    """
    簡化 NPZ 生態動力模型

    用法:
        model = NPZModel()
        result = model.run(sst, chl, par, nutrients=no3)
        # result["phytoplankton"]  → 2D (0-1) 浮游植物豐度
        # result["zooplankton"]    → 2D (0-1) 浮游動物豐度
        # result["nutrients"]      → 2D 殘餘營養鹽
    """

    def __init__(self, params: Optional[Dict] = None):
        self.p = {**DEFAULT_PARAMS, **(params or {})}

    def _growth_rate_temp(self, sst: np.ndarray) -> np.ndarray:
        """溫度依賴的浮游植物生長率 (Eppley 1972)

        μ(T) = μ_max × 1.066^T   (Eppley curve)
        歸一化到 μ_max @ 20°C
        """
        sst_c = np.clip(sst, 0, 35)
        return self.p["mu_max"] * np.power(1.066, sst_c) / np.power(1.066, 20.0)

    def _nutrient_limitation(self, N: np.ndarray) -> np.ndarray:
        """Michaelis-Menten 營養限制"""
        return N / (self.p["K_N"] + N)

    def _light_limitation(self, par: np.ndarray) -> np.ndarray:
        """光限制因子"""
        return par / (self.p["K_I"] + par)

    def _grazing(self, P: np.ndarray) -> np.ndarray:
        """Holling Type III 攝食 (sigmoidal)"""
        kp2 = self.p["K_P"] ** 2
        return self.p["g_max"] * P ** 2 / (kp2 + P ** 2)

    def run(
        self,
        sst: np.ndarray,
        chl: np.ndarray,
        par: Optional[np.ndarray] = None,
        nutrients: Optional[np.ndarray] = None,
        n_days: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        運行 NPZ 模型

        Parameters
        ----------
        sst : 2D SST (°C)
        chl : 2D Chl-a (mg/m³), 用於初始化 P
        par : 2D PAR (Einstein/m²/day), 可選
        nutrients : 2D 營養鹽 (mmol N/m³), 可選
        n_days : 積分天數, 預設 30

        Returns
        -------
        dict:
            phytoplankton : 2D (0-1) 歸一化浮游植物豐度
            zooplankton   : 2D (0-1) 歸一化浮游動物豐度
            nutrients     : 2D 殘餘營養鹽
            P_raw         : 2D 原始 P (mmol N/m³)
            Z_raw         : 2D 原始 Z (mmol N/m³)
        """
        ny, nx = sst.shape
        dt = self.p["dt"]
        steps = int((n_days or self.p["n_days"]) / dt)

        # ── 初始條件 ──
        # P: 從 Chl-a 估算 (Chl:N ≈ 1.59 mg Chl / mmol N, Geider 1997)
        chl_safe = np.clip(np.nan_to_num(chl, nan=0.1), 0.01, 30.0)
        P = (chl_safe / 1.59).astype(np.float64)

        # Z: 初始為 P 的 10% (典型海洋比例)
        Z = (P * 0.10).astype(np.float64)

        # N: 從外部營養鹽, 或用 SST 估算 (冷水=高營養)
        if nutrients is not None:
            N = np.clip(np.nan_to_num(nutrients, nan=5.0), 0, 40).astype(np.float64)
        else:
            # 反比 SST: 冷水 → 高 N (湧升, 混合)
            sst_safe = np.clip(np.nan_to_num(sst, nan=25.0), 0, 35)
            N = np.clip(15.0 - 0.4 * sst_safe, 0.5, 15.0).astype(np.float64)

        # PAR
        if par is None:
            par_2d = np.full((ny, nx), 35.0, dtype=np.float64)
        else:
            par_2d = np.clip(np.nan_to_num(par, nan=35.0), 1, 60).astype(np.float64)

        # ── 溫度依賴生長率 ──
        mu_T = self._growth_rate_temp(sst)

        # ── Euler 前進積分 ──
        log.debug(f"  NPZ: integrating {steps} steps ({n_days or self.p['n_days']} days)")

        for _ in range(steps):
            # 限制因子
            f_N = self._nutrient_limitation(N)
            f_I = self._light_limitation(par_2d)

            # 浮游植物生長
            growth = mu_T * f_N * f_I * P

            # 攝食
            graze = self._grazing(P) * Z

            # 死亡
            mort_P = self.p["m_P"] * P
            mort_Z = self.p["m_Z"] * Z * Z  # 密度依賴

            # 更新
            dP = (growth - graze - mort_P) * dt
            dZ = (self.p["gamma"] * graze - mort_Z) * dt
            dN = (-growth + self.p["recycling"] * (mort_P + mort_Z)
                  + self.p["recycling"] * (1 - self.p["gamma"]) * graze) * dt

            P = np.maximum(P + dP, 1e-6)
            Z = np.maximum(Z + dZ, 1e-6)
            N = np.maximum(N + dN, 0)

        # ── 歸一化到 0-1 (與 HSI 格式一致) ──
        P_f32 = P.astype(np.float32)
        Z_f32 = Z.astype(np.float32)

        p99 = np.nanpercentile(P_f32, 99) if P_f32.size > 0 else 1.0
        z99 = np.nanpercentile(Z_f32, 99) if Z_f32.size > 0 else 1.0

        P_norm = np.clip(P_f32 / max(p99, 1e-6), 0, 1).astype(np.float32)
        Z_norm = np.clip(Z_f32 / max(z99, 1e-6), 0, 1).astype(np.float32)

        log.info(f"  NPZ: P_mean={np.nanmean(P_f32):.2f}, Z_mean={np.nanmean(Z_f32):.2f} mmol N/m³")

        return {
            "phytoplankton": P_norm,
            "zooplankton": Z_norm,
            "nutrients": N.astype(np.float32),
            "P_raw": P_f32,
            "Z_raw": Z_f32,
        }

    def get_species_zoo_weight(self, species: str) -> float:
        """
        根據物種獵物列表計算浮游動物依賴權重

        Parameters
        ----------
        species : 物種 key (from species_params.SPECIES)

        Returns
        -------
        float : 0-1 浮游動物依賴程度
        """
        try:
            from engine.species_params import SPECIES
            sp = SPECIES.get(species, {})
            prey_list = sp.get("prey", [])
            if not prey_list:
                # 沒有 prey 欄位 → 用營養級估算
                tl = sp.get("trophic_level", 4.0)
                return max(0, 1.0 - (tl - 2.0) * 0.25)

            total_w = sum(PREY_ZOO_WEIGHT.get(p, 0.3) for p in prey_list)
            return min(total_w / max(len(prey_list), 1), 1.0)

        except ImportError:
            return 0.5

    def compute_species_forage_npz(
        self,
        species: str,
        npz_result: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        將 NPZ 結果轉為物種特定的餌料指數

        浮游動物依賴高的魚種 → 權重偏 Z
        浮游動物依賴低的魚種 → 權重偏 P (間接: P→小魚→大魚)
        """
        w_z = self.get_species_zoo_weight(species)
        w_p = 1.0 - w_z

        forage = (w_p * npz_result["phytoplankton"]
                  + w_z * npz_result["zooplankton"])

        return np.clip(forage, 0, 1).astype(np.float32)


# ═══════════════════════════════════════════════════════
# 模組級快捷函數
# ═══════════════════════════════════════════════════════

def compute_npz(
    sst: np.ndarray,
    chl: np.ndarray,
    par: Optional[np.ndarray] = None,
    nutrients: Optional[np.ndarray] = None,
    n_days: int = 30,
) -> Dict[str, np.ndarray]:
    """模組級快捷: 執行 NPZ 模型並回傳結果"""
    model = NPZModel()
    return model.run(sst, chl, par=par, nutrients=nutrients, n_days=n_days)
