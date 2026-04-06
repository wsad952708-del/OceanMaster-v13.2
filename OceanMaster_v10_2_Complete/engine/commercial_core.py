"""
OceanMaster v10.1 — 商業核心引擎 (Commercial Core Engine)
============================================================
⚠️ [v12-fix] DEPRECATED — 此檔案已被 commercial_core_v2.py 取代。
所有新代碼和 import 應使用 commercial_core_v2.py。
保留此檔案僅為向後相容性參考。

整合從全球付費系統逆向推理出的 7 大核心秘密

🔑 逆向推理的商業秘密（CATSAT / SEAPODYM / PFZ 未公開但可推導）：

秘密 1: 代謝指數 Φ (Metabolic Index)
  來源: Deutsch et al. 2015 Science + Chen et al. 2024 Nature Comms
  公式: Φ = pO₂ / [Pcrit × exp(Eo/kB × (1/T - 1/Tref))]
  意義: 溫度+溶氧的聯合約束 → 比單獨用 SST 或 DO 準確度高 50%+
  商業系統怎麼用: SEAPODYM 用 DO tolerance mask，但 Φ 是更先進的統一框架
  Φ > 2-5 = 可棲息，Φ < 1 = 致死

秘密 2: 多深度 3D 環境場 (Multi-Depth 3D)
  來源: Su et al. 2024 Fisheries Research
  發現: 用 0-300m 多深度的 T/S/DO/Chl 做 CNN 輸入，準確度大幅提升
  商業系統怎麼用: CATSAT 用 SEAPODYM 的 3D 微型浮游生物場
  我們用: HYCOM 3D + WOA2023 → 在 0/50/100/200/300m 取環境變數

秘密 3: SEAPODYM 真實棲息地公式
  Feeding Habitat = Σ(accessible_micronekton_k × oxygen_mask_k)
  Thermal Habitat = Gaussian(T, T_opt(age), σ_T(age))
  Spawning Habitat = T_spawning × larvae_food × predator_avoidance
  oxygen_mask = 1 if DO > DO_crit, smooth_transition otherwise

秘密 4: 渦旋邊緣追蹤 + 時間滯後
  來源: American Samoa 長鰭鮪研究
  發現: CPUE 在渦旋「邊緣」最高（不是中心），且滯後 EKE 峰值 ~2 個月
  商業系統怎麼用: CATSAT 追蹤渦旋邊緣而非面積

秘密 5: 鋒面時間持續性 (Front Persistence)
  來源: Cayula-Cornillon + INCOIS PFZ
  發現: 持續 3-7 天的穩定鋒面比瞬時鋒面重要 10 倍
  商業系統怎麼用: PFZ 用多日合成追蹤鋒面穩定性

秘密 6: CatBoost / LightGBM 取代 XGBoost
  來源: Shi et al. 2024, Xu et al. 2024
  發現: LightGBM 和 CatBoost 在漁場預測中準確率和速度都優於 XGBoost/RF

秘密 7: Argo 浮標即時校準
  來源: argopy (Python library)
  用途: 用近實時 Argo 剖面數據校準衛星反演的次表層值

數據來源（全部免費，已查證 URL）：
  NOAA ERDDAP | HYCOM 3D | CMEMS | WOA2023 | Oregon State VGPM
  ETOPO1 | Argo GDAC | GFW AIS | VIIRS | Open-Meteo
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

log = logging.getLogger("OceanMaster.Core")

# ═══════════════════════════════════════════════════════
# 物理常數
# ═══════════════════════════════════════════════════════
KB = 8.617e-5  # Boltzmann 常數 (eV/K)
TREF_K = 288.15  # 參考溫度 15°C (K)


# ═══════════════════════════════════════════════════════
# 秘密 1: 代謝指數 Φ (Metabolic Index)
# ═══════════════════════════════════════════════════════

# 物種代謝參數 (從 Deutsch 2015 + 文獻整理)
# Eo = 溫度敏感係數 (eV), Pcrit = 臨界 pO2 (kPa at Tref)
METABOLIC_TRAITS = {
    "skipjack": {
        "Eo": 0.45,          # 高溫敏感（熱帶表層物種）
        "Pcrit_kPa": 5.5,    # 相對高的氧需求
        "Topt_C": 26.0,      # 最適溫度
        "Topt_sigma": 3.5,   # 溫度容忍寬度
        "phi_crit": 3.0,     # 臨界 Φ（需要高於靜息的 3 倍才能活動覓食）
        "depth_pref_m": [0, 50, 100, 150],
    },
    "yellowfin": {
        "Eo": 0.40,
        "Pcrit_kPa": 4.5,
        "Topt_C": 24.0,
        "Topt_sigma": 4.0,
        "phi_crit": 2.8,
        "depth_pref_m": [0, 50, 100, 200],
    },
    "bigeye": {
        "Eo": 0.30,          # 低溫敏感（深潛物種，rete mirabile）
        "Pcrit_kPa": 2.5,    # 低氧需求 → 能在 OMZ 邊緣覓食
        "Topt_C": 18.0,      # 偏好較冷水
        "Topt_sigma": 6.0,   # 寬溫度範圍
        "phi_crit": 2.0,
        "depth_pref_m": [50, 100, 200, 300, 400],
    },
    "albacore": {
        "Eo": 0.38,
        "Pcrit_kPa": 4.0,
        "Topt_C": 19.0,
        "Topt_sigma": 3.0,
        "phi_crit": 2.5,
        "depth_pref_m": [0, 50, 100, 200, 300],
    },
    "squid_todarodes": {
        "Eo": 0.50,          # 魷魚代謝率極高
        "Pcrit_kPa": 5.0,
        "Topt_C": 16.0,
        "Topt_sigma": 4.0,
        "phi_crit": 3.5,
        "depth_pref_m": [0, 50, 100, 200],
    },
    "squid_ommastrephes": {
        "Eo": 0.48,
        "Pcrit_kPa": 4.8,
        "Topt_C": 22.0,
        "Topt_sigma": 5.0,
        "phi_crit": 3.2,
        "depth_pref_m": [0, 50, 100, 200, 300],
    },
}


class MetabolicIndexEngine:
    """
    代謝指數 Φ 計算引擎

    Φ = pO₂ / [Pcrit × exp(Eo/kB × (1/T - 1/Tref))]

    其中:
      pO₂ = 環境中的氧分壓 (kPa)
      Pcrit = 物種在參考溫度下的臨界 pO₂
      Eo = 溫度對代謝率的敏感係數 (eV)
      T = 環境溫度 (K)
      Tref = 參考溫度 (288.15 K = 15°C)

    Φ 的物理意義：
      Φ = 環境供氧能力 / 生物需氧量
      Φ > phi_crit (2-5) → 可持續活動和覓食
      1 < Φ < phi_crit → 可存活但活動受限
      Φ < 1 → 致死區域
    """

    @staticmethod
    def do_to_po2(do_ml_l: np.ndarray, temp_c: np.ndarray,
                  salinity: Optional[np.ndarray] = None) -> np.ndarray:
        """
        溶解氧 (ml/L) → 氧分壓 pO₂ (kPa)

        使用 Garcia & Gordon (1992) 的簡化逆算:
        pO₂ ≈ DO / DO_sat × 0.2095 × (Patm - pH₂O)

        其中 DO_sat 用 Weiss (1970) 公式:
        ln(DO_sat) = A1 + A2×(100/T) + A3×ln(T/100) + A4×(T/100)
                     + S×[B1 + B2×(T/100) + B3×(T/100)²]
        """
        T_K = temp_c + 273.15
        T_100 = T_K / 100.0
        S = salinity if salinity is not None else np.full_like(temp_c, 35.0)

        # Weiss (1970) 氧溶解度係數
        A1, A2, A3, A4 = -173.4292, 249.6339, 143.3483, -21.8492
        B1, B2, B3 = -0.033096, 0.014259, -0.0017000

        ln_do_sat = (A1 + A2 * (100.0 / T_K) + A3 * np.log(T_100)
                     + A4 * T_100
                     + S * (B1 + B2 * T_100 + B3 * T_100**2))
        do_sat = np.exp(ln_do_sat)  # ml/L

        # 避免除零
        do_sat = np.maximum(do_sat, 0.1)

        # 氧飽和度
        oxygen_saturation = np.clip(do_ml_l / do_sat, 0.0, 1.5)

        # pO₂ = saturation × 0.2095 × (101.325 - 水蒸氣壓)
        # 簡化：水蒸氣壓 ≈ 0.5-3 kPa (隨溫度)
        pvap = 0.61078 * np.exp(17.269 * temp_c / (temp_c + 237.3))  # kPa
        po2 = oxygen_saturation * 0.2095 * (101.325 - pvap)

        return np.clip(po2, 0.0, 25.0).astype(np.float32)

    @staticmethod
    def compute_phi(
        po2: np.ndarray,
        temp_c: np.ndarray,
        species: str,
    ) -> np.ndarray:
        """
        計算代謝指數 Φ

        Φ = pO₂ / [Pcrit × exp(Eo/kB × (1/Tref - 1/T))]

        這是 Nature Communications 2024 證實的最強棲息地預測指標
        """
        traits = METABOLIC_TRAITS.get(species, METABOLIC_TRAITS["yellowfin"])
        T_K = temp_c + 273.15

        # 代謝需氧量隨溫度的指數增長
        metabolic_demand = traits["Pcrit_kPa"] * np.exp(
            traits["Eo"] / KB * (1.0 / TREF_K - 1.0 / T_K)
        )

        # Φ = 供氧 / 需氧
        phi = po2 / np.maximum(metabolic_demand, 0.01)

        return phi.astype(np.float32)

    @staticmethod
    def phi_to_habitat_viability(
        phi: np.ndarray, species: str
    ) -> np.ndarray:
        """
        Φ → 棲息地可行性指數 (0-1)

        使用 sigmoid 轉換:
          viability = 1 / (1 + exp(-k × (Φ - Φ_crit)))

        Φ >> Φ_crit → viability ≈ 1
        Φ = Φ_crit → viability = 0.5 (邊界)
        Φ << Φ_crit → viability → 0
        """
        traits = METABOLIC_TRAITS.get(species, METABOLIC_TRAITS["yellowfin"])
        phi_crit = traits["phi_crit"]
        k = 3.0 / max(phi_crit, 1.0)  # 斜率

        viability = 1.0 / (1.0 + np.exp(-k * (phi - phi_crit)))
        return np.clip(viability, 0.0, 1.0).astype(np.float32)


# ═══════════════════════════════════════════════════════
# 秘密 3: SEAPODYM 式棲息地指數（完整重建）
# ═══════════════════════════════════════════════════════

class SEAPODYMHabitatEngine:
    """
    SEAPODYM 棲息地指數完整重建

    三個獨立棲息地指數的加權乘積:
      H_total = H_thermal × H_feeding × H_oxygen

    其中:
      H_thermal: 高斯溫度偏好（隨年齡/體型變化）
      H_feeding: 可及餌料密度（微型浮游生物 × 深度可及性）
      H_oxygen: 溶氧耐受面具（sigmoid 閾值）
    """

    @staticmethod
    def compute_thermal_habitat(
        temp: np.ndarray,
        species: str,
    ) -> np.ndarray:
        """
        熱棲息地指數 H_thermal

        SEAPODYM 公式（已查證）:
          H_T = exp(-(T - T_opt)² / (2σ²))

        T_opt 和 σ 隨物種和年齡變化
        """
        traits = METABOLIC_TRAITS.get(species, METABOLIC_TRAITS["yellowfin"])
        t_opt = traits["Topt_C"]
        sigma = traits["Topt_sigma"]

        h_thermal = np.exp(-((temp - t_opt) ** 2) / (2 * sigma ** 2))
        return np.clip(h_thermal, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def compute_feeding_habitat(
        forage_index: np.ndarray,
        phi_viability: np.ndarray,
    ) -> np.ndarray:
        """
        覓食棲息地 H_feeding

        SEAPODYM 公式（逆向推理）:
          H_F = Σ_k (M_k × A_k × O_k) / Σ_k (M_k)

        其中:
          M_k = 第 k 功能群的微型浮游生物密度
          A_k = 深度可及性（物種能否到達該深度）
          O_k = 氧耐受面具

        我們的簡化版：forage_index 已包含 M×A，乘上 Φ 的氧面具
        """
        h_feeding = forage_index * phi_viability
        return np.clip(h_feeding, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def compute_combined_habitat(
        h_thermal: np.ndarray,
        h_feeding: np.ndarray,
        phi_viability: np.ndarray,
        front_persistence: Optional[np.ndarray] = None,
        eke_si: Optional[np.ndarray] = None,
        eddy_edge: Optional[np.ndarray] = None,
        depth_si: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        SEAPODYM 式綜合棲息地指數

        H = H_thermal^w1 × H_feeding^w2 × Φ_viability^w3
            × (1 + bonus_factors)

        加權指數（幾何平均比算術平均更合理，因為任一因子為 0 就致命）
        """
        # 核心三因子（幾何平均）
        w_thermal = 0.25
        w_feeding = 0.30
        w_phi = 0.25

        h = (np.maximum(h_thermal, 0.001) ** w_thermal
             * np.maximum(h_feeding, 0.001) ** w_feeding
             * np.maximum(phi_viability, 0.001) ** w_phi)

        # 加分因子（加法，不是致命因子）
        remaining_weight = 0.20  # 分配給以下輔助因子
        bonus = np.zeros_like(h)
        n_bonus = 0

        if front_persistence is not None:
            bonus += front_persistence
            n_bonus += 1
        if eke_si is not None:
            bonus += eke_si
            n_bonus += 1
        if eddy_edge is not None:
            bonus += eddy_edge
            n_bonus += 1
        if depth_si is not None:
            bonus += depth_si
            n_bonus += 1

        if n_bonus > 0:
            bonus_avg = bonus / n_bonus
            h = h * (1.0 + remaining_weight * bonus_avg)

        return np.clip(h, 0.0, 1.0).astype(np.float32)


# ═══════════════════════════════════════════════════════
# 秘密 4: 渦旋邊緣偵測 + 時間滯後
# ═══════════════════════════════════════════════════════

class EddyEdgeDetector:
    """
    渦旋邊緣魚群聚集效應

    商業系統的秘密：魚不在渦旋中心，而在邊緣
    原因：渦旋邊緣 = 強烈混合 + 營養鹽上湧 + 鋒面形成

    方法：
    1. 從 OW 參數識別渦旋
    2. 計算渦旋邊緣（OW 梯度最大處）
    3. 邊緣 + 2 個月時間滯後 = 最佳漁場
    """

    @staticmethod
    def compute_eddy_edge_index(
        ow: np.ndarray,
        sla: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        渦旋邊緣指數

        邊緣定義：|∇OW| 最大的區域
        漁場品質：邊緣 >> 中心 >> 外部
        """
        # OW 梯度
        grad_y, grad_x = np.gradient(ow)
        grad_mag = np.sqrt(grad_y**2 + grad_x**2)

        # 歸一化
        p95 = np.nanpercentile(grad_mag, 95)
        if p95 > 0:
            edge_index = grad_mag / p95
        else:
            edge_index = np.zeros_like(ow)

        # 額外：SLA 梯度也標記邊緣
        if sla is not None:
            sla_gy, sla_gx = np.gradient(sla)
            sla_grad = np.sqrt(sla_gy**2 + sla_gx**2)
            sla_p95 = np.nanpercentile(sla_grad, 95)
            if sla_p95 > 0:
                edge_index = 0.6 * edge_index + 0.4 * (sla_grad / sla_p95)

        return np.clip(edge_index, 0.0, 1.0).astype(np.float32)


# ═══════════════════════════════════════════════════════
# 秘密 5: 鋒面時間持續性
# ═══════════════════════════════════════════════════════

class FrontPersistenceTracker:
    """
    鋒面持續性追蹤

    INCOIS PFZ 的核心秘密：持續的鋒面 >> 瞬時的鋒面

    方法：
    1. 每日鋒面偵測 → 二值化
    2. 多日疊加（3-7天）
    3. 持續出現的位置 = 穩定鋒面 = 優質漁場
    4. 持續性分數 = 出現天數 / 總天數

    在沒有多日數據時，用空間平滑近似時間穩定性
    """

    @staticmethod
    def compute_persistence_proxy(
        front_strength: np.ndarray,
        n_days_simulated: int = 5,
    ) -> np.ndarray:
        """
        鋒面持續性代理指標

        原理：空間上連續的鋒面更可能是穩定的
        用空間自相關作為時間持續性的代理

        真正的實現需要多日數據，這裡用統計方法近似
        """
        from scipy.ndimage import uniform_filter, maximum_filter

        # 鋒面二值化
        threshold = np.nanpercentile(front_strength, 70)
        front_binary = (front_strength > threshold).astype(np.float32)

        # 空間一致性 = 周圍格點也是鋒面的比例
        spatial_consistency = uniform_filter(front_binary, size=5)

        # 鋒面強度的穩定性（用局部方差的倒數代理）
        local_mean = uniform_filter(front_strength, size=5)
        local_sq = uniform_filter(front_strength**2, size=5)
        local_var = np.maximum(local_sq - local_mean**2, 1e-6)
        stability = 1.0 / (1.0 + local_var / np.maximum(local_mean**2, 1e-6))

        # 綜合持續性 = 空間一致性 × 穩定性 × 強度
        persistence = spatial_consistency * stability * np.clip(
            front_strength / max(np.nanpercentile(front_strength, 90), 0.01), 0, 1
        )

        return np.clip(persistence, 0.0, 1.0).astype(np.float32)


# ═══════════════════════════════════════════════════════
# 秘密 2: 多深度 3D 特徵提取
# ═══════════════════════════════════════════════════════

class MultiDepthFeatureExtractor:
    """
    多深度 3D 環境場特徵提取

    CNN 研究(Su et al. 2024)證實：
    用 0-300m 多深度的 T/S/DO/Chl 準確度 >> 僅用表層

    HYCOM 提供 40 個標準深度，我們提取 5 個關鍵深度：
    0m (表層) / 50m (次表層) / 100m (溫躍層上) /
    200m (溫躍層下) / 300m (中層)

    對每個深度計算 Φ → 取垂直平均或最小值
    """

    ANALYSIS_DEPTHS = [0, 50, 100, 200, 300]

    @staticmethod
    def compute_depth_integrated_phi(
        temp_3d: np.ndarray,
        do_3d: np.ndarray,
        depths: list,
        species: str,
        salinity_3d: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        在多個深度計算 Φ，取垂直整合值

        Returns:
            phi_surface: 表層 Φ
            phi_min: 棲息深度範圍內的最小 Φ（限制因子）
            phi_mean: 棲息深度範圍內的平均 Φ
            phi_profile: 各深度的 Φ
        """
        traits = METABOLIC_TRAITS.get(species, METABOLIC_TRAITS["yellowfin"])
        pref_depths = traits["depth_pref_m"]
        max_depth = max(pref_depths)

        mi_engine = MetabolicIndexEngine()
        phi_stack = []

        for i, d in enumerate(depths):
            if d > max_depth * 1.5:
                continue
            if i >= temp_3d.shape[0] or i >= do_3d.shape[0]:
                continue

            temp_layer = temp_3d[i]
            do_layer = do_3d[i]
            sal_layer = salinity_3d[i] if salinity_3d is not None else None

            po2 = mi_engine.do_to_po2(do_layer, temp_layer, sal_layer)
            phi = mi_engine.compute_phi(po2, temp_layer, species)
            phi_stack.append(phi)

        if not phi_stack:
            return {
                "phi_surface": np.ones(temp_3d[0].shape, dtype=np.float32) * 3.0,
                "phi_min": np.ones(temp_3d[0].shape, dtype=np.float32) * 3.0,
                "phi_mean": np.ones(temp_3d[0].shape, dtype=np.float32) * 3.0,
            }

        phi_array = np.stack(phi_stack, axis=0)

        return {
            "phi_surface": phi_stack[0],
            "phi_min": np.nanmin(phi_array, axis=0),
            "phi_mean": np.nanmean(phi_array, axis=0),
        }


# ═══════════════════════════════════════════════════════
# 完整 v10.1 商業級 HSI 管線
# ═══════════════════════════════════════════════════════

class CommercialGradeHSI:
    """
    商業級棲息地適宜性指數

    整合所有 7 大秘密的最終 HSI 計算

    管線：
    1. Metabolic Index Φ (秘密1) → 溫度+氧聯合約束
    2. Multi-depth 3D (秘密2) → 次表層環境場
    3. SEAPODYM 棲息地 (秘密3) → H_thermal × H_feeding × H_oxygen
    4. Eddy edge (秘密4) → 渦旋邊緣加分
    5. Front persistence (秘密5) → 穩定鋒面加分
    6. 輸出 → LightGBM/CatBoost 融合 (秘密6)
    7. Argo 校準接口 (秘密7)
    """

    def __init__(self):
        self.mi_engine = MetabolicIndexEngine()
        self.seapodym = SEAPODYMHabitatEngine()
        self.eddy_edge = EddyEdgeDetector()
        self.front_tracker = FrontPersistenceTracker()
        self.depth_extractor = MultiDepthFeatureExtractor()

    def compute_ultimate_hsi(
        self,
        sst: np.ndarray,
        chl: np.ndarray,
        do_surface: np.ndarray,
        forage_index: np.ndarray,
        front_strength: np.ndarray,
        species: str,
        ow: Optional[np.ndarray] = None,
        sla: Optional[np.ndarray] = None,
        eke_si: Optional[np.ndarray] = None,
        depth_si: Optional[np.ndarray] = None,
        salinity: Optional[np.ndarray] = None,
        temp_3d: Optional[np.ndarray] = None,
        do_3d: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        計算終極 HSI

        Returns dict with:
          hsi: 0-1 最終棲息地指數
          phi: 代謝指數
          phi_viability: Φ 的棲息地可行性
          h_thermal: 熱棲息地
          h_feeding: 覓食棲息地
          front_persistence: 鋒面持續性
          eddy_edge: 渦旋邊緣指數
          confidence: 置信度（數據品質指標）
        """
        log.info(f"  🧬 {species}: 商業級 HSI 計算中...")

        # ── 秘密 1: Metabolic Index Φ ──
        po2 = self.mi_engine.do_to_po2(do_surface, sst, salinity)
        phi = self.mi_engine.compute_phi(po2, sst, species)
        phi_viability = self.mi_engine.phi_to_habitat_viability(phi, species)

        # ── 秘密 2: 多深度 (如果有 3D 數據) ──
        if temp_3d is not None and do_3d is not None:
            depths = [0, 50, 100, 200, 300][:min(temp_3d.shape[0], 5)]
            phi_3d = self.depth_extractor.compute_depth_integrated_phi(
                temp_3d, do_3d, depths, species
            )
            # 用最小 Φ 作為限制因子（木桶效應）
            phi_viability = self.mi_engine.phi_to_habitat_viability(
                phi_3d["phi_min"], species
            )
            log.info(f"    3D Φ_min={np.nanmean(phi_3d['phi_min']):.2f}")

        # ── 秘密 3: SEAPODYM 棲息地 ──
        h_thermal = self.seapodym.compute_thermal_habitat(sst, species)
        h_feeding = self.seapodym.compute_feeding_habitat(forage_index, phi_viability)

        # ── 秘密 4: 渦旋邊緣 ──
        eddy_edge_idx = None
        if ow is not None:
            eddy_edge_idx = self.eddy_edge.compute_eddy_edge_index(ow, sla)

        # ── 秘密 5: 鋒面持續性 ──
        front_persist = None
        try:
            front_persist = self.front_tracker.compute_persistence_proxy(front_strength)
        except ImportError:
            front_persist = np.clip(front_strength / max(np.nanmax(front_strength), 0.01),
                                    0, 1).astype(np.float32)

        # ── 秘密 3 (完整): 綜合棲息地 ──
        hsi = self.seapodym.compute_combined_habitat(
            h_thermal=h_thermal,
            h_feeding=h_feeding,
            phi_viability=phi_viability,
            front_persistence=front_persist,
            eke_si=eke_si,
            eddy_edge=eddy_edge_idx,
            depth_si=depth_si,
        )

        # 置信度（基於數據完整性）
        confidence = 0.5  # 基礎
        if temp_3d is not None:
            confidence += 0.15  # 有 3D 數據
        if ow is not None:
            confidence += 0.1   # 有渦旋數據
        if salinity is not None:
            confidence += 0.1   # 有鹽度
        if do_surface is not None and np.any(do_surface > 0):
            confidence += 0.15  # 有 DO

        mean_hsi = float(np.nanmean(hsi))
        mean_phi = float(np.nanmean(phi))
        log.info(f"    HSI={mean_hsi:.3f} | Φ={mean_phi:.2f} | "
                 f"H_T={np.nanmean(h_thermal):.3f} | "
                 f"H_F={np.nanmean(h_feeding):.3f} | "
                 f"conf={confidence:.0%}")

        return {
            "hsi": hsi,
            "phi": phi,
            "phi_viability": phi_viability,
            "h_thermal": h_thermal,
            "h_feeding": h_feeding,
            "front_persistence": front_persist,
            "eddy_edge": eddy_edge_idx,
            "confidence": confidence,
        }
