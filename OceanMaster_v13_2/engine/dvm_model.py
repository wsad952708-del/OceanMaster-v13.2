"""
OceanMaster v13.2 — 魚群垂直遷移 (DVM) 與深度可及性模型  # [v12-enhance]
==================================================
模擬鮪魚的日間垂直遷移行為，計算不同物種對不同深度餌料的可及性。

科學背景:
  - 黃鰭鮪: 白天 50-250m，夜晚表層 (Schaefer 2007)
  - 大目鮪: 白天可潛至 600m, 夜晚 0-100m (Musyl 2003)
  - 正鰹: 白天 20-200m, 嚴格在 MLD 上方 (Brill 1999)
  - 長鰭鮪: 白天 100-300m, 夜晚 0-80m (Williams 2014)

  大目鮪可以「吃到」深層餌料 (= Z20 附近的 micronekton)
  黃鰭鮪只能在表層覓食 → 需要餌料上浮到 MLD 附近
  這就是為什麼大目鮪和黃鰭鮪的熱點位置不同!

v12 新增:
  - [v12-enhance] 月光→DVM 深度壓制 (Benoit-Bird et al. 2009, MEPS 395:27-30)
    滿月時 micronekton 抑制上浮 → 表層餌料減少
  - [v12-enhance] 連續日夜過渡 (cosine weighting)，替代二元 day/night 切換
  - [v12-enhance] 若有 HYCOM 溫度剖面，優先使用真實深層溫度
"""

import numpy as np
import logging
from typing import Dict, Optional
from engine.species_params import DVM_PARAMS, Z20_PREFS

log = logging.getLogger("OceanMaster.DVM")


class DVMModel:
    """
    日間垂直遷移模型

    計算某物種在某海域能接觸到多少比例的餌料生物。

    使用:
        dvm = DVMModel()
        access = dvm.compute_accessibility('bigeye', z20, mld, sst)
        # access → 0-1 的深度可及性矩陣
    """

    # [v12-enhance] 月光對 DVM 的影響係數
    # Benoit-Bird et al. 2009: 滿月時 micronekton 上浮幅度減少 ~30-50%
    MOON_DVM_SUPPRESSION = 0.40  # 滿月最大抑制表層餌料比例

    def compute_accessibility(self, species: str,
                               z20: np.ndarray,
                               mld: np.ndarray,
                               sst: Optional[np.ndarray] = None,
                               is_daytime: bool = True,
                               temp_profile: Optional[np.ndarray] = None,
                               ) -> np.ndarray:
        """
        計算某物種的餌料深度可及性

        可及性 = overlap(魚的活動深度, 餌料的分布深度) × 溫度耐受

        Parameters:
            species: 物種 ID
            z20: 2D 溫躍層深度 (m)
            mld: 2D 混合層深度 (m)
            sst: 2D 表面溫度 — 用於粗略估算深層溫度
            is_daytime: 白天 or 夜晚
            temp_profile: [v12-enhance] 3D 溫度 (n_depths, ny, nx)，若提供則用真實深層溫度

        Returns:
            accessibility: 2D (0-1)
        """
        if species not in DVM_PARAMS:
            log.warning(f"  物種 {species} 無 DVM 參數，使用預設")
            return np.full_like(z20, 0.5)

        params = DVM_PARAMS[species]

        if is_daytime:
            fish_min, fish_max = params["day_depth_range"]
        else:
            fish_min, fish_max = params["night_depth_range"]

        # 餌料主要分布在兩個層次:
        #   1. 表層: 0 ~ MLD (白天少，夜晚多 — micronekton DVM 上浮)
        #   2. 深層: MLD ~ Z20 及更深 (白天多)
        if is_daytime:
            # 白天 micronekton 主要在深層 (200-700m)
            forage_top = mld
            forage_bottom = z20 * 1.5  # 餌料可延伸到 Z20 以下
        else:
            # 夜晚 micronekton 上浮到表層
            forage_top = np.zeros_like(mld)
            forage_bottom = mld

        # 計算重疊區間
        overlap_top = np.maximum(fish_min, forage_top)
        overlap_bottom = np.minimum(fish_max, forage_bottom)
        overlap = np.maximum(overlap_bottom - overlap_top, 0)

        forage_range = np.maximum(forage_bottom - forage_top, 1)
        accessibility = overlap / forage_range

        # 溫度耐受校正
        if sst is not None:
            temp_min = params["temp_min_tolerance"]
            depth_mid = (fish_min + fish_max) / 2.0

            # [v12-enhance] 若有 HYCOM 溫度剖面 → 直接從 profile 內插
            if temp_profile is not None and temp_profile.ndim == 3:
                # 簡化: 用 depth_mid/100 對應的 layer index
                layer_idx = int(np.clip(depth_mid / 100.0, 0, temp_profile.shape[0] - 1))
                deep_temp = temp_profile[layer_idx]  # [v12-enhance] 真實深層溫度
            else:
                # Fallback: 線性估算 (每 100m 降 3°C)
                deep_temp = sst - (depth_mid / 100.0) * 3.0

            temp_mask = np.where(deep_temp >= temp_min, 1.0,
                                 np.clip((deep_temp - temp_min + 3) / 3, 0, 1))
            accessibility *= temp_mask

        return np.clip(accessibility, 0, 1).astype(np.float32)

    # [v12-enhance] 月光對 DVM 的影響
    @staticmethod
    def moonlight_dvm_factor(lunar_illumination: "np.ndarray | float") -> "np.ndarray | float":
        """
        月光抑制 micronekton 上浮的因子

        Benoit-Bird et al. 2009, MEPS 395:27-30:
          滿月時 micronekton 不完全上浮到表層 → 表層餌料減少 30-50%
          新月時 micronekton 正常上浮 → 不影響

        Parameters:
            lunar_illumination: 0-1 (0=新月, 1=滿月)

        Returns:
            dvm_factor: 0.6-1.0 (1=正常, 0.6=滿月抑制)
        """
        # 線性模型: factor = 1.0 - suppression * illumination
        factor = 1.0 - DVMModel.MOON_DVM_SUPPRESSION * lunar_illumination
        if isinstance(factor, np.ndarray):
            return np.clip(factor, 0.5, 1.0).astype(np.float32)
        return max(0.5, min(1.0, factor))

    def compute_feeding_index(self, species: str,
                               forage_surface: np.ndarray,
                               forage_deep: np.ndarray,
                               z20: np.ndarray,
                               mld: np.ndarray,
                               sst: Optional[np.ndarray] = None,
                               hour_utc: int = 12,
                               lunar_illumination: Optional[float] = None,
                               temp_profile: Optional[np.ndarray] = None,
                               ) -> np.ndarray:
        """
        綜合覓食指數 — 結合 DVM 行為與餌料分布

        H_feeding = Σ(餌料密度 × 可及性) / 最大值

        Parameters:
            species: 物種 ID
            forage_surface: 表層餌料 (0-1)
            forage_deep: 深層餌料 (0-1)
            z20, mld: 溫躍層/混合層深度
            sst: 表面溫度
            hour_utc: UTC 小時 (判斷日夜)
            lunar_illumination: [v12-enhance] 0-1 月光照度 (0=新月, 1=滿月)
            temp_profile: [v12-enhance] 3D 溫度剖面

        Returns:
            feeding_index: 2D (0-1)
        """
        # 日夜能力計算
        access_day = self.compute_accessibility(
            species, z20, mld, sst, is_daytime=True, temp_profile=temp_profile
        )
        access_night = self.compute_accessibility(
            species, z20, mld, sst, is_daytime=False, temp_profile=temp_profile
        )

        # 白天: 魚吃深層餌料（micronekton 在深層）
        # 夜晚: 魚吃表層餌料（micronekton 上浮）
        feeding_day = access_day * forage_deep
        feeding_night = access_night * forage_surface

        # [v12-enhance] 月光抑制: 滿月時表層餌料減少
        if lunar_illumination is not None:
            moon_factor = self.moonlight_dvm_factor(lunar_illumination)
            feeding_night = feeding_night * moon_factor  # [v12-enhance]
            log.info(f"  🌙 月光 DVM 抑制: illumination={lunar_illumination:.2f}, "
                     f"factor={moon_factor:.2f}")

        # [v12-enhance] 連續日夜加權 (cosine transition 取代 50/50 固定比)
        # hour_utc=6 → 日出, 12 → 正午, 18 → 日落, 0 → 午夜
        # day_weight: 正午=1.0, 午夜=0.0, 日出/日落=0.5
        day_weight = 0.5 * (1.0 + np.cos(2.0 * np.pi * (hour_utc - 12.0) / 24.0))
        night_weight = 1.0 - day_weight  # [v12-enhance]

        feeding = day_weight * feeding_day + night_weight * feeding_night

        # 正規化
        f_max = np.nanpercentile(feeding, 99) if np.any(feeding > 0) else 1.0
        feeding_norm = np.clip(feeding / max(f_max, 1e-10), 0, 1)

        log.info(f"  {species} feeding index: avg={np.nanmean(feeding_norm):.3f}, "
                 f"max={np.nanmax(feeding_norm):.3f} "
                 f"(day_w={day_weight:.2f}, night_w={night_weight:.2f})")

        return feeding_norm.astype(np.float32)

    @staticmethod
    def compute_z20_preference(z20: np.ndarray, species: str) -> np.ndarray:
        """
        Z20 偏好評分

        不同物種偏好不同深度的溫躍層:
          - 正鰹: 淺溫躍層 (Z20 ~ 120m) → 表層餌料多
          - 大目鮪: 深溫躍層 (Z20 ~ 300m) → 深層覓食空間大
        """
        if species not in Z20_PREFS:
            return np.ones_like(z20) * 0.5

        pref = Z20_PREFS[species]
        optimal = pref["optimal"]
        sigma = pref["sigma"]

        score = np.exp(-((z20 - optimal) ** 2) / (2 * sigma ** 2))
        return score.astype(np.float32)

