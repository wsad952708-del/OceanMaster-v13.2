"""
OceanMaster v12 — SSH 渦旋偵測 + 渦旋邊緣追蹤  # [v12-enhance]
==========================================
基於衛星 SSH (海面高度) 異常偵測中尺度渦旋。

科學背景:
  - 渦旋邊緣的 CPUE 比中心高 3-5 倍 (American Samoa 研究)
  - 反氣旋渦旋的下沉流困住餌料，吸引鮪魚

演算法:
  1. Okubo-Weiss 參數: W = Sn² + Ss² - ω²
  2. W < -2σ_W → 渦旋核心
  3. |∇W| 最大 → 渦旋邊緣 (漁場黃金區!)
  4. 速度異常動能 = 0.5 × (u'² + v'²)

v12 新增:
  - [v12-enhance] Okubo-Weiss 正式分類 (Isern-Fontanet 2006)
    渦度主導 / 應變主導 / 背景 三區域分類

數據源:
  - OceanDataFetcher 已擷取的 SSH + 地轉流 (AVISO/CMEMS)
"""

import numpy as np
import logging
from typing import Dict

log = logging.getLogger("OceanMaster.EddyDetector")


class EddyDetector:
    """
    渦旋偵測引擎 — Okubo-Weiss + 渦旋邊緣追蹤

    使用:
        detector = EddyDetector()
        result = detector.detect(ssh, u_geo, v_geo, lats, lons)
        # result['eddy_edge'] → 漁場黃金區
    """

    def __init__(self, ow_threshold: float = -2.0):
        """
        Parameters:
            ow_threshold: Okubo-Weiss 閾值倍數 (W < ow_threshold × σ)
        """
        self.ow_threshold = ow_threshold

    def detect(self, ssh: np.ndarray,
               u_geo: np.ndarray, v_geo: np.ndarray,
               lats: np.ndarray, lons: np.ndarray) -> Dict:
        """
        執行渦旋偵測

        Parameters:
            ssh: 2D SSH 異常場 (m)
            u_geo: 2D 地轉 U 流速 (m/s)
            v_geo: 2D 地轉 V 流速 (m/s)
            lats, lons: 1D 座標陣列

        Returns:
            dict with:
              - eddy_core:   bool mask (渦旋核心)
              - eddy_edge:   float 0-1 (渦旋邊緣強度)
              - eke:         float (速度異常動能 m²/s²)
              - vorticity:   float (渦度 s⁻¹)
              - ssh_gradient: float (SSH 梯度)
              - eddy_type:   +1=反氣旋(暖), -1=氣旋(冷)
        """
        ny, nx = ssh.shape

        if ny < 3 or nx < 3:
            log.warning("SSH 網格太小，無法偵測渦旋")
            return self._empty_result(ny, nx)

        # ─── Step 1: 計算平均格點間距 (m) ───
        mean_dy = np.mean(np.abs(np.diff(lats))) * 111000.0  # deg → m
        if mean_dy < 1:
            mean_dy = 111000.0
        mean_dx = np.mean(np.abs(np.diff(lons))) * 111000.0 * np.cos(np.radians(np.mean(lats)))
        if mean_dx < 1:
            mean_dx = 111000.0

        # ─── Step 2: 速度梯度 (s⁻¹) ───
        # 使用標量間距避免 NaN (uniform grid assumption)
        du_dy, du_dx = np.gradient(u_geo, mean_dy, mean_dx)
        dv_dy, dv_dx = np.gradient(v_geo, mean_dy, mean_dx)

        # NaN 安全處理
        du_dy = np.nan_to_num(du_dy, 0.0)
        du_dx = np.nan_to_num(du_dx, 0.0)
        dv_dy = np.nan_to_num(dv_dy, 0.0)
        dv_dx = np.nan_to_num(dv_dx, 0.0)

        # ─── Step 3: Okubo-Weiss 參數 ───
        sn = du_dx - dv_dy      # 法向應變
        ss = dv_dx + du_dy      # 剪應變
        omega = dv_dx - du_dy   # 渦度

        # 渦度 clipping 防止數值雜訊
        omega = np.clip(omega, -0.01, 0.01)

        W = sn**2 + ss**2 - omega**2

        # ─── Step 4: 渦旋核心偵測 ───
        sigma_W = np.nanstd(W)
        if sigma_W < 1e-15:
            log.info("  SSH 場均勻，無渦旋特徵")
            return self._empty_result(ny, nx)

        eddy_core = W < (self.ow_threshold * sigma_W)
        n_core = np.sum(eddy_core)
        log.info(f"  Okubo-Weiss: σ={sigma_W:.2e}, 核心網格={n_core}")

        # ─── Step 5: 渦旋邊緣 (OW 梯度最大處) ───
        grad_W_y, grad_W_x = np.gradient(W)
        grad_W_mag = np.sqrt(grad_W_y**2 + grad_W_x**2)

        # 歸一化到 0-1
        gmax = np.nanmax(grad_W_mag)
        if gmax > 0:
            eddy_edge = grad_W_mag / gmax
        else:
            eddy_edge = np.zeros((ny, nx))

        # 只保留核心附近的邊緣 (膨脹核心 mask)
        try:
            from scipy.ndimage import binary_dilation
            expanded_core = binary_dilation(eddy_core, iterations=3)
            eddy_edge *= expanded_core
        except ImportError:
            pass  # scipy 不可用時不做限制

        # ─── Step 6: 速度異常動能 ───
        # 注意: 單一快照無法計算真正的 EKE (需時間序列)。
        # 這裡用空間異常動能 (SAE) 作為渦旋活躍度代理。
        u_mean = np.nanmean(u_geo)
        v_mean = np.nanmean(v_geo)
        u_prime = u_geo - u_mean
        v_prime = v_geo - v_mean
        eke = 0.5 * (u_prime**2 + v_prime**2)

        # ─── Step 7: SSH 梯度 ───
        ssh_grad_y, ssh_grad_x = np.gradient(ssh)
        ssh_gradient = np.sqrt(ssh_grad_y**2 + ssh_grad_x**2)

        # ─── Step 8: 渦旋類型 ───
        mean_lat = np.mean(lats)
        if mean_lat > 0:
            eddy_type = np.where(omega < 0, 1.0, -1.0)  # 北半球反氣旋=順時針=ω<0
        else:
            eddy_type = np.where(omega > 0, 1.0, -1.0)

        eddy_type *= eddy_core.astype(float)

        n_anti = np.sum(eddy_type > 0)
        n_cycl = np.sum(eddy_type < 0)
        log.info(f"  渦旋: {n_anti} 反氣旋(暖) + {n_cycl} 氣旋(冷)")

        return {
            "eddy_core": eddy_core,
            "eddy_edge": eddy_edge.astype(np.float32),
            "eke": eke.astype(np.float32),
            "vorticity": omega.astype(np.float32),
            "ssh_gradient": ssh_gradient.astype(np.float32),
            "eddy_type": eddy_type.astype(np.float32),
        }

    # ── [v12-enhance] Okubo-Weiss 正式分類 ──

    def classify_okubo_weiss(
        self,
        u_geo: np.ndarray,
        v_geo: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> Dict:
        """
        [v12-enhance] Okubo-Weiss 正式分類 (Isern-Fontanet et al. 2006)

        將海域分為三個動力學區域:
          - 渦度主導 (W < -W₀): 旋轉佔優 → 渦旋核心 → 困住浮游生物
          - 應變主導 (W > +W₀): 拉伸佔優 → 渦旋間細絲 → 物質輸送通道
          - 背景 (|W| < W₀): 幾乎均勻流 → 非漁區

        漁業應用:
          - 渦度主導 → 渦旋核心: 大目鮪偏好 (冷核上升流)
          - 應變主導 → 渦旋邊緣/細絲: 正鰹/黃鰭鮪偏好 (物質聚集)

        Parameters:
            u_geo: 2D 地轉 U 流速 (m/s)
            v_geo: 2D 地轉 V 流速 (m/s)
            lats, lons: 1D 座標陣列

        Returns:
            dict: {
                'ow_class':  2D int (-1=渦度主導, 0=背景, +1=應變主導)
                'ow_field':  2D float (W 值)
                'sigma_w':   float (W 標準差)
                'pct_vorticity': float (渦度主導面積比)
                'pct_strain':    float (應變主導面積比)
            }
        """
        ny, nx = u_geo.shape

        # 格距 (m)
        mean_dy = max(np.mean(np.abs(np.diff(lats))) * 111000.0, 1.0)
        mean_dx = max(
            np.mean(np.abs(np.diff(lons))) * 111000.0 * np.cos(np.radians(np.mean(lats))),
            1.0,
        )

        # 速度梯度
        du_dy, du_dx = np.gradient(np.nan_to_num(u_geo), mean_dy, mean_dx)
        dv_dy, dv_dx = np.gradient(np.nan_to_num(v_geo), mean_dy, mean_dx)

        sn = du_dx - dv_dy      # 法向應變
        ss = dv_dx + du_dy      # 剪應變
        omega = dv_dx - du_dy   # 渦度
        omega = np.clip(omega, -0.01, 0.01)

        W = sn**2 + ss**2 - omega**2

        sigma_W = np.nanstd(W)
        if sigma_W < 1e-15:
            return {
                "ow_class": np.zeros((ny, nx), dtype=np.int8),
                "ow_field": W.astype(np.float32),
                "sigma_w": 0.0,
                "pct_vorticity": 0.0,
                "pct_strain": 0.0,
            }

        # 閾值: W₀ = |ow_threshold| × σ_W
        W0 = abs(self.ow_threshold) * sigma_W

        # 分類
        ow_class = np.zeros((ny, nx), dtype=np.int8)
        ow_class[W < -W0] = -1  # 渦度主導 (vorticity-dominant)
        ow_class[W > +W0] = +1  # 應變主導 (strain-dominant)
        # 0 = 背景

        total = ny * nx
        pct_vort = float(np.sum(ow_class == -1)) / total * 100
        pct_strain = float(np.sum(ow_class == +1)) / total * 100

        log.info(f"  🌀 OW 分類: 渦度主導={pct_vort:.1f}%, "
                 f"應變主導={pct_strain:.1f}%, "
                 f"背景={100 - pct_vort - pct_strain:.1f}%")

        return {
            "ow_class": ow_class,
            "ow_field": W.astype(np.float32),
            "sigma_w": float(sigma_W),
            "pct_vorticity": pct_vort,
            "pct_strain": pct_strain,
        }

    @staticmethod
    def compute_eddy_biological_enrichment(
        eddy_core: np.ndarray,
        eddy_type: np.ndarray,
        chl: np.ndarray,
        sst: np.ndarray,
        eddy_edge: np.ndarray,
        species: str = "yellowfin",
    ) -> np.ndarray:
        """
        [v11 Feature D] 渦旋生物增益

        科學依據:
          - 氣旋渦旋 (冷核) → 上升流 → 營養鹽 → 高 Chl → 大目鮪偏好
          - 反氣旋渦旋 (暖核) → 下沉 → 暖核邊緣聚集餌料 → 黃鰭鮪偏好
          - 邊緣匯聚帶 → 高 EKE → 正鰹偏好
          - 文獻: Gaube et al. 2014, Braun et al. 2019

        方法:
          1. Chl-a 異常: 渦旋內均值 vs 背景
          2. 冷核增益: tanh(chl_anom / 0.5) → 上升流帶來營養鹽
          3. 暖核邊緣增益: 邊緣處鋒面強度
          4. 物種偏好: 加權結合

        Parameters:
            eddy_core: 2D bool (渦旋核心)
            eddy_type: 2D float (+1=反氣旋, -1=氣旋)
            chl: 2D Chl-a (mg/m³)
            sst: 2D SST (°C)
            eddy_edge: 2D float (0-1, 渦旋邊緣強度)
            species: 物種名稱

        Returns:
            enrichment: 2D float (0-1) 生物增益指數
        """
        ny, nx = chl.shape
        enrichment = np.zeros((ny, nx), dtype=np.float32)

        # 背景 Chl-a (渦旋外均值)
        non_eddy_mask = ~eddy_core & np.isfinite(chl)
        if np.any(non_eddy_mask):
            chl_bg = np.nanmean(chl[non_eddy_mask])
        else:
            chl_bg = np.nanmean(chl) if np.any(np.isfinite(chl)) else 0.1

        chl_bg = max(chl_bg, 0.01)  # 避免除零

        # Chl-a 異常
        chl_anom = (chl - chl_bg) / chl_bg

        # 1. 冷核增益: 氣旋渦旋的上升流帶來高 Chl
        cold_core_mask = eddy_type < -0.5  # 氣旋 (冷核)
        cold_enrichment = np.tanh(np.maximum(chl_anom, 0) / 0.5) * cold_core_mask

        # 2. 暖核邊緣增益: 反氣旋邊緣的匯聚帶
        warm_core_mask = eddy_type > 0.5   # 反氣旋 (暖核)
        # 用邊緣強度作為邊緣增益
        warm_rim = eddy_edge * warm_core_mask

        # 3. SST 異常增益: 溫差越大 → 渦旋越強
        sst_bg = np.nanmean(sst[non_eddy_mask]) if np.any(non_eddy_mask) else np.nanmean(sst)
        sst_anom = np.abs(sst - sst_bg)
        sst_factor = np.clip(sst_anom / 2.0, 0, 1)  # 2°C = 飽和

        # 物種偏好加權
        if species == "bigeye":
            # 大目鮪: 強烈偏好冷核 (上升流), 中等偏好暖核邊緣
            enrichment = (0.7 * cold_enrichment + 0.2 * warm_rim + 0.1 * sst_factor)
        elif species in ("yellowfin", "skipjack"):
            # 黃鰭鮪/正鰹: 偏好暖核邊緣 (下沉流困住餌料)
            enrichment = (0.3 * cold_enrichment + 0.5 * warm_rim + 0.2 * sst_factor)
        elif species == "albacore":
            # 長鰭鮪: 偏好冷核和溫度鋒面
            enrichment = (0.5 * cold_enrichment + 0.3 * warm_rim + 0.2 * sst_factor)
        else:
            enrichment = (0.4 * cold_enrichment + 0.4 * warm_rim + 0.2 * sst_factor)

        # 正規化到 0-1
        e_max = np.nanmax(enrichment) if np.any(enrichment > 0) else 1.0
        enrichment = np.clip(enrichment / max(e_max, 1e-6), 0, 1)

        n_enriched = np.sum(enrichment > 0.3)
        log.info(f"  Eddy bio enrichment ({species}): "
                 f"{n_enriched} pixels > 0.3, max={np.nanmax(enrichment):.3f}")

        return enrichment.astype(np.float32)

    @staticmethod
    def _empty_result(ny: int, nx: int) -> Dict:
        return {
            "eddy_core": np.zeros((ny, nx), dtype=bool),
            "eddy_edge": np.zeros((ny, nx), dtype=np.float32),
            "eke": np.zeros((ny, nx), dtype=np.float32),
            "vorticity": np.zeros((ny, nx), dtype=np.float32),
            "ssh_gradient": np.zeros((ny, nx), dtype=np.float32),
            "eddy_type": np.zeros((ny, nx), dtype=np.float32),
        }

