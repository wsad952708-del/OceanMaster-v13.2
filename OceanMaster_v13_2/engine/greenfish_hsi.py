"""
OceanMaster — GreenFish Lite HSI 引擎
=====================================
商業級棲息地適合度指數 — SEAPODYM 風格加權幾何平均。

HSI = (H_thermal^w1 × H_feeding^w2 × H_oxygen^w3 × H_dynamic^w4 × H_z20^w5)^(1/Σw)

與現有 CommercialGradeHSI 的區別:
  - 現有: 線性加權平均，~8 個因子
  - 此版: 幾何加權平均（乘法），7 個因子
  - 幾何平均的優勢: 任何一個因子為 0 → HSI = 0 (生態真實性)
  - 例如: 即使 SST 完美，但溶氧太低 → 魚不會去 → HSI 直接為 0
"""

import numpy as np
import logging
from typing import Dict, Optional
from engine.species_params import SPECIES, Z20_PREFS

log = logging.getLogger("OceanMaster.GreenFishHSI")


class GreenFishLiteHSI:
    """
    GreenFish Lite 棲息地適合度指數引擎

    整合所有物理+生物+動態因子，以 SEAPODYM 風格加權幾何平均產生
    最終 HSI 分數 (0-100)。

    使用:
        hsi_engine = GreenFishLiteHSI()
        result = hsi_engine.compute('yellowfin',
            sst=sst, z20=z20, mld=mld,
            phi_viability=phi_v,
            feeding_index=feed,
            eddy_edge=edge, eke=eke,
            front_strength=front,
            chl=chl,
        )
        # result['hsi']  → 0-100
        # result['components'] → 各子指數
    """

    # ═══════════════════════════════════════════════════
    # 權重配置 (基於 SEAPODYM + GreenFish 逆向推理)
    # [v11] 新增 salinity_front，重新分配權重
    # ═══════════════════════════════════════════════════
    WEIGHTS = {
        "thermal":         0.24,   # SST 與 T_opt 偏差 (最重要)
        "feeding":         0.21,   # 餌場可及性 + 餌料量
        "oxygen":          0.14,   # 代謝指數 Φ
        "z20":             0.12,   # 溫躍層深度偏好
        "dynamic":         0.11,   # 渦旋邊緣 + EKE
        "front":           0.08,   # SST/CHL 鋒面（或生產力鋒面）
        "chl":             0.05,   # CHL-a 濃度
        "salinity_front":  0.05,   # [v11] 鹽度鋒面
    }

    @staticmethod
    def _estimate_deep_temp(sst, depth_m, deep_temp=None):
        """Estimate deep-layer temperature with optional real data override."""
        if deep_temp is not None:
            return deep_temp
        # Conservative lapse: 0.04°C/m shallow (<200m), 0.06°C/m deep (>200m)
        # Better matches WOA2023 tropical thermocline profile
        lapse = 0.06 if depth_m > 200 else 0.04
        est = sst - (depth_m / 100.0) * lapse * 100.0
        return np.clip(est, 4.0, 30.0)

    def compute(self, species: str,
                sst: np.ndarray,
                phi_viability: np.ndarray,
                feeding_index: Optional[np.ndarray] = None,
                z20: Optional[np.ndarray] = None,
                mld: Optional[np.ndarray] = None,
                eddy_edge: Optional[np.ndarray] = None,
                eke: Optional[np.ndarray] = None,
                front_strength: Optional[np.ndarray] = None,
                chl: Optional[np.ndarray] = None,
                salinity_front_strength: Optional[np.ndarray] = None,
                deep_temp: Optional[np.ndarray] = None,
                **kwargs) -> Dict:
        """
        計算 GreenFish Lite HSI

        Returns:
            dict with:
              - hsi: 2D (0-100) 最終分數
              - components: dict of sub-indices (0-1)
              - weights_used: dict of actual weights
        """
        ny, nx = sst.shape
        sp = SPECIES.get(species, SPECIES["yellowfin"])

        # ─── Sub-index 1: 熱棲息地 H_thermal ───
        # [v11 Bug #4] 若物種有分離的表層/深層溫度偏好，使用 24h 加權
        if "Topt_surface_C" in sp and "Topt_deep_C" in sp:
            # 表層偏好 (夜間上浮, SST 為參考)
            t_opt_s = sp["Topt_surface_C"]
            sigma_s = sp["Topt_sigma"]
            h_surface = np.exp(-((sst - t_opt_s) ** 2) / (2 * sigma_s ** 2))

            # 深層偏好 (日間深潛, 估算深層溫度)
            t_opt_d = sp["Topt_deep_C"]
            sigma_d = sp.get("Topt_deep_sigma", 4.0)
            # 深層溫度: 用真實 3D 數據或保守 lapse rate
            dvm_depth_est = 300.0 if species == "bigeye" else 150.0
            deep_temp_est = self._estimate_deep_temp(sst, dvm_depth_est, deep_temp)
            h_deep = np.exp(-((deep_temp_est - t_opt_d) ** 2) / (2 * sigma_d ** 2))

            # 24h 加權: 日間(深) 12h + 夜間(淺) 12h
            h_thermal = 0.5 * h_surface + 0.5 * h_deep
        else:
            # 傳統: 單一 Topt (skipjack 等淺層物種)
            t_opt = sp["Topt_C"]
            sigma = sp["Topt_sigma"]
            h_thermal = np.exp(-((sst - t_opt) ** 2) / (2 * sigma ** 2))

        # ─── Sub-index 2: 覓食棲息地 H_feeding ───
        if feeding_index is not None:
            h_feeding = np.clip(feeding_index, 0.01, 1)
        else:
            # fallback: CHL-based proxy (log-Gaussian)
            if chl is not None:
                chl_opt_lo, chl_opt_hi = sp.get("chl_optimal", (0.1, 0.5))
                chl_mid = (chl_opt_lo + chl_opt_hi) / 2
                # sigma 用整個最適範圍的寬度，避免過於敏感
                log_sigma = max(np.log10(chl_opt_hi / chl_opt_lo) / 2.0, 0.3)
                h_feeding = np.exp(-((np.log10(np.maximum(chl, 0.01)) -
                                       np.log10(chl_mid)) ** 2) / (2 * log_sigma ** 2))
                h_feeding = np.clip(h_feeding, 0.01, 1)
            else:
                h_feeding = np.full((ny, nx), 0.5)

        # ─── Sub-index 3: 氧棲息地 H_oxygen ───
        h_oxygen = np.clip(phi_viability, 0.01, 1)

        # ─── Sub-index 4: Z20 偏好 ───
        if z20 is not None and species in Z20_PREFS:
            pref = Z20_PREFS[species]
            h_z20 = np.exp(-((z20 - pref["optimal"]) ** 2) / (2 * pref["sigma"] ** 2))
            h_z20 = np.clip(h_z20, 0.01, 1)
        else:
            h_z20 = np.full((ny, nx), 0.5)

        # ─── Sub-index 5: 動態棲息地 H_dynamic ───
        if eddy_edge is not None:
            # [商用] 防禦性 shape 處理 — 確保 eddy_edge 是 2D (ny, nx)
            if eddy_edge.ndim == 1:
                eddy_edge = np.broadcast_to(eddy_edge.reshape(-1, 1) if eddy_edge.shape[0] == ny
                             else eddy_edge.reshape(1, -1), (ny, nx)).copy()
            elif eddy_edge.shape != (ny, nx):
                eddy_edge = np.full((ny, nx), float(np.nanmean(eddy_edge)))
            eke_norm = np.zeros((ny, nx))
            if eke is not None:
                # [商用] 防禦性 shape 處理 — 確保 eke 是 2D (ny, nx)
                if eke.ndim == 1:
                    eke = np.broadcast_to(eke.reshape(-1, 1) if eke.shape[0] == ny
                                 else eke.reshape(1, -1), (ny, nx)).copy()
                elif eke.shape != (ny, nx):
                    eke = np.full((ny, nx), float(np.nanmean(eke)))
                eke_p95 = np.nanpercentile(eke, 95) if np.any(eke > 0) else 1.0
                eke_norm = np.clip(eke / max(eke_p95, 1e-10), 0, 1)
            h_dynamic = np.clip(0.6 * eddy_edge + 0.4 * eke_norm, 0.01, 1)
        else:
            h_dynamic = np.full((ny, nx), 0.3)

        # ─── Sub-index 6: 鋒面 H_front ───
        if front_strength is not None:
            f_p95 = np.nanpercentile(front_strength, 95) if np.any(front_strength > 0) else 1.0
            h_front = np.clip(front_strength / max(f_p95, 1e-10), 0.01, 1)
        else:
            h_front = np.full((ny, nx), 0.3)

        # ─── Sub-index 7: CHL 適宜度 ───
        if chl is not None:
            chl_opt_lo, chl_opt_hi = sp.get("chl_optimal", (0.1, 0.5))
            # log-Gaussian: 在最適範圍內為 1，範圍外按距離衍減
            chl_log = np.log10(np.maximum(chl, 0.01))
            chl_lo_log = np.log10(chl_opt_lo)
            chl_hi_log = np.log10(chl_opt_hi)
            chl_mid_log = (chl_lo_log + chl_hi_log) / 2.0
            chl_range_log = max(chl_hi_log - chl_lo_log, 0.2)  # 避免零寬
            # 範圍內 = 1，範圍外用 距離/範圍寬度 做平滑衍減
            dist_from_range = np.where(
                chl_log < chl_lo_log, chl_lo_log - chl_log,
                np.where(chl_log > chl_hi_log, chl_log - chl_hi_log, 0.0)
            )
            h_chl = np.exp(-(dist_from_range ** 2) / (2 * (chl_range_log * 0.8) ** 2))
            h_chl = np.clip(h_chl, 0.01, 1)
        else:
            h_chl = np.full((ny, nx), 0.5)

        # ─── Sub-index 8: [v11] 鹽度鋒面 H_salinity_front ───
        if salinity_front_strength is not None:
            sf_p95 = np.nanpercentile(salinity_front_strength, 95) if np.any(salinity_front_strength > 0) else 1.0
            h_salinity_front = np.clip(salinity_front_strength / max(sf_p95, 1e-10), 0.01, 1)
        else:
            h_salinity_front = np.full((ny, nx), 0.3)

        # ═══════════════════════════════════════════════
        # 加權幾何平均 (SEAPODYM 風格)
        # [v11] 新增 salinity_front 子指數
        # ═══════════════════════════════════════════════
        w = self.WEIGHTS
        total_w = sum(w.values())

        # log-space 計算避免下溢
        log_hsi = (
            w["thermal"]        * np.log(h_thermal) +
            w["feeding"]        * np.log(h_feeding) +
            w["oxygen"]         * np.log(h_oxygen) +
            w["z20"]            * np.log(h_z20) +
            w["dynamic"]        * np.log(h_dynamic) +
            w["front"]          * np.log(h_front) +
            w["chl"]            * np.log(h_chl) +
            w["salinity_front"] * np.log(h_salinity_front)
        ) / total_w

        hsi = np.exp(log_hsi) * 100  # 0-100

        # 安全裁剪
        hsi = np.clip(hsi, 0, 100).astype(np.float32)

        # NaN 處理
        hsi = np.where(np.isnan(hsi), 0, hsi)

        log.info(f"  GreenFish HSI ({sp['name_zh']}): "
                 f"avg={np.nanmean(hsi):.1f}, max={np.nanmax(hsi):.1f}")

        components = {
            "H_thermal": h_thermal.astype(np.float32),
            "H_feeding": h_feeding.astype(np.float32),
            "H_oxygen": h_oxygen.astype(np.float32),
            "H_z20": h_z20.astype(np.float32),
            "H_dynamic": h_dynamic.astype(np.float32),
            "H_front": h_front.astype(np.float32),
            "H_chl": h_chl.astype(np.float32),
            "H_salinity_front": h_salinity_front.astype(np.float32),
        }

        return {
            "hsi": hsi,
            "components": components,
            "weights_used": dict(w),
        }

    @staticmethod
    def explain_hsi(components: Dict, weights: Dict, species_zh: str) -> str:
        """
        產生中文可解釋性文字

        Returns:
            說明文字，例如:
            「黃鰭鮪 HSI 由溫度棲息地(25%)主導，本區域 SST 偏高，
             餌場指數(22%)中等，溫躍層深度適中(Z20=180m)。」
        """
        # 各子指數平均值
        means = {k: float(np.nanmean(v)) for k, v in components.items()}

        # 排序：貢獻最大的因子
        contributions = []
        for key, w_val in weights.items():
            h_key = f"H_{key}"
            if h_key in means:
                contrib = w_val * means[h_key]
                contributions.append((key, contrib, means[h_key]))

        contributions.sort(key=lambda x: x[1], reverse=True)

        names_zh = {
            "thermal": "水溫適合度",
            "feeding": "餌場指數",
            "oxygen": "代謝指數(Φ)",
            "z20": "溫躍層偏好",
            "dynamic": "渦旋邊緣",
            "front": "鋒面強度",
            "chl": "葉綠素",
            "salinity_front": "鹽度鋒面",
        }

        lines = [f"{species_zh} 棲息地評分:"]
        for key, contrib, h_val in contributions[:4]:
            name = names_zh.get(key, key)
            pct = int(weights.get(key, 0) * 100)
            level = "優" if h_val > 0.7 else "中" if h_val > 0.4 else "差"
            lines.append(f"  {name}({pct}%): {level} ({h_val:.2f})")

        return " ".join(lines)
