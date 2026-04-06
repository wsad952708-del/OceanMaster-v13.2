"""
OceanMaster v13.2 — Ocean Color Front Detection [D3]
=====================================================
Detect CHL fronts (sharp chlorophyll gradients) from satellite data.
CHL fronts indicate convergence zones that concentrate prey.

[v13.2-GH5] Upgraded: Cayula-Cornillon statistical front detection
Source: CoLAB-ATLANTIC/JUNO + Descanonge/fronts-toolbox (github.com)
Original method: Sobel gradient only
Added: Cayula-Cornillon histogram bimodality test (Cayula & Cornillon 1992)
Fusion: Cayula-Cornillon × Sobel → higher precision, fewer false positives

Ref: Belkin & O'Reilly (2009) J. Geophys. Res. 114:C04003
Ref: Cayula & Cornillon (1992) J. Atmos. Ocean. Tech. 9:67-80
"""

import numpy as np
import logging
from typing import Dict, Any, Optional

log = logging.getLogger("OceanMaster.ChlFront")


class OceanColorFrontDetector:
    """
    [v16.0 D3] Detect chlorophyll concentration fronts.

    Methods:
      1. Sobel gradient: fast, catches all gradients (including noise)
      2. Cayula-Cornillon: statistical, tests histogram bimodality (fewer false alarms)
      3. Fusion: both agree → high-confidence front

    Usage:
        detector = OceanColorFrontDetector()
        result = detector.detect(chl_2d, method='fusion')
    """

    def __init__(
        self,
        gradient_threshold: float = 0.15,  # log(mg/m³)/pixel
        min_chl: float = 0.05,  # minimum CHL to consider
        cc_window_size: int = 16,  # Cayula-Cornillon window size (pixels)
        cc_min_theta: float = 0.7,  # CC bimodality threshold
    ):
        self.gradient_threshold = gradient_threshold
        self.min_chl = min_chl
        self.cc_window_size = cc_window_size
        self.cc_min_theta = cc_min_theta

    def detect(
        self,
        chl: np.ndarray,
        lats: np.ndarray = None,
        lons: np.ndarray = None,
        method: str = 'fusion',
    ) -> Dict[str, Any]:
        """
        Detect CHL fronts.

        Args:
            chl: 2D chlorophyll-a (mg/m³)
            method: 'sobel' | 'cayula_cornillon' | 'fusion'

        Returns:
            {
                "chl_gradient": 2D gradient magnitude,
                "front_mask": 2D bool (strong fronts),
                "cc_front_mask": 2D bool (Cayula-Cornillon fronts, if applicable),
                "front_density_pct": float,
                "n_front_cells": int,
                "mean_gradient": float,
                "method_used": str,
                "advisory": str,
            }
        """
        ny, nx = chl.shape

        # Log transform (CHL is log-normal distributed)
        chl_safe = np.clip(chl, self.min_chl, None)
        log_chl = np.log10(chl_safe).astype(np.float64)

        # ── Sobel gradient (always computed) ──
        grad = self._sobel_gradient(log_chl).astype(np.float32)
        sobel_mask = grad > self.gradient_threshold

        # ── Method selection ──
        cc_mask = None
        if method in ('cayula_cornillon', 'fusion'):
            cc_mask = self._cayula_cornillon(log_chl)

        if method == 'sobel' or cc_mask is None:
            front_mask = sobel_mask
            method_used = 'sobel'
        elif method == 'cayula_cornillon':
            front_mask = cc_mask
            method_used = 'cayula_cornillon'
        else:  # fusion
            # Fusion: both methods agree → high confidence
            # CC detects statistically significant fronts
            # Sobel provides gradient magnitude
            front_mask = sobel_mask & cc_mask
            method_used = 'fusion(sobel+CC)'

        n_fronts = int(np.sum(front_mask))
        density = n_fronts / max(ny * nx, 1) * 100

        if density > 20:
            advisory = "CHL front density HIGH - strong convergence zones"
        elif density > 5:
            advisory = "CHL fronts present - moderate convergence"
        else:
            advisory = "Few CHL fronts - weak convergence"

        log.info(f"  CHL fronts [{method_used}]: {n_fronts} cells ({density:.1f}%), "
                 f"max gradient={np.max(grad):.3f}")

        result = {
            "chl_gradient": grad,
            "front_mask": front_mask,
            "front_density_pct": round(density, 1),
            "n_front_cells": n_fronts,
            "mean_gradient": round(float(np.mean(grad)), 4),
            "method_used": method_used,
            "advisory": advisory,
        }
        if cc_mask is not None:
            result["cc_front_mask"] = cc_mask
        return result

    def _cayula_cornillon(self, field: np.ndarray) -> np.ndarray:
        """
        Cayula & Cornillon (1992) 鋒面偵測

        核心思想: 在滑動視窗內做 SST/CHL 直方圖
        → 如果直方圖呈雙模態 (bimodal) → 說明視窗內有兩團不同水體
        → 兩團的分界 = 鋒面

        步驟:
          1. 滑動視窗掃描
          2. 每個視窗做直方圖 (n_bins=32)
          3. 檢測雙模態: theta = (N_left × N_right) / (N_total²)
          4. theta > threshold → 鋒面存在
          5. 鋒面位置 = 直方圖谷值處

        Source: Cayula & Cornillon 1992, CoLAB-ATLANTIC/JUNO implementation
        """
        ny, nx = field.shape
        w = self.cc_window_size
        front = np.zeros((ny, nx), dtype=bool)

        # 半視窗
        hw = w // 2

        for j in range(hw, ny - hw, hw):
            for i in range(hw, nx - hw, hw):
                window = field[j - hw:j + hw, i - hw:i + hw]
                valid = window[~np.isnan(window)]

                if len(valid) < w * w * 0.5:
                    continue  # 太多 NaN

                # ── 直方圖雙模態檢定 ──
                n_bins = min(32, max(8, len(valid) // 4))
                hist, bin_edges = np.histogram(valid, bins=n_bins)

                # 找谷值 (兩個峰之間的最小值)
                if len(hist) < 3:
                    continue

                # 平滑直方圖 (3-point moving average)
                hist_smooth = np.convolve(hist, [0.25, 0.5, 0.25], mode='same')

                # 尋找所有局部最大值
                peaks = []
                for k in range(1, len(hist_smooth) - 1):
                    if hist_smooth[k] > hist_smooth[k-1] and hist_smooth[k] > hist_smooth[k+1]:
                        peaks.append(k)

                if len(peaks) < 2:
                    continue  # 不是雙模態

                # 取最大的兩個峰
                peaks.sort(key=lambda k: -hist_smooth[k])
                p1, p2 = sorted(peaks[:2])

                # 計算 theta (bimodality coefficient)
                n_left = np.sum(hist[p1:p2])
                valley_idx = p1 + np.argmin(hist_smooth[p1:p2+1])
                n_valley = hist_smooth[valley_idx]
                n_peak_avg = (hist_smooth[p1] + hist_smooth[p2]) / 2

                if n_peak_avg < 1:
                    continue

                # theta = 1 - (valley_depth / peak_height)
                # theta 越高 → 雙模態越明顯 → 鋒面越確定
                theta = 1.0 - (n_valley / n_peak_avg)

                if theta > self.cc_min_theta:
                    # 鋒面存在！標記視窗中央區域
                    # 鋒面位置 = 值落在谷值附近的像素
                    threshold = bin_edges[valley_idx]
                    left_vals = window < threshold
                    right_vals = window >= threshold

                    # 鋒面 = 相鄰像素跨越閾值的位置
                    from scipy.ndimage import binary_dilation
                    boundary = binary_dilation(left_vals) & binary_dilation(right_vals)
                    front[j - hw:j + hw, i - hw:i + hw] |= boundary

        return front

    @staticmethod
    def _sobel_gradient(field: np.ndarray) -> np.ndarray:
        """Compute gradient magnitude using Sobel operators."""
        try:
            from scipy.ndimage import sobel
            gx = sobel(field, axis=1)
            gy = sobel(field, axis=0)
            return np.sqrt(gx**2 + gy**2)
        except ImportError:
            # Manual 3x3 Sobel
            ny, nx = field.shape
            grad = np.zeros_like(field)
            for j in range(1, ny - 1):
                for i in range(1, nx - 1):
                    gx = (field[j-1, i+1] + 2*field[j, i+1] + field[j+1, i+1]
                          - field[j-1, i-1] - 2*field[j, i-1] - field[j+1, i-1])
                    gy = (field[j+1, i-1] + 2*field[j+1, i] + field[j+1, i+1]
                          - field[j-1, i-1] - 2*field[j-1, i] - field[j-1, i+1])
                    grad[j, i] = np.sqrt(gx**2 + gy**2)
            return grad
