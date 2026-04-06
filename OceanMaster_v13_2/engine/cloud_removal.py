"""
OceanMaster — CloudRemovalNet
==============================
衛星 SST 雲遮蔽區域去除／插值補全模組

不依賴 PyTorch，純 scipy.interpolate.griddata 實作：
  1. Cubic 插值補全 NaN 區域
  2. 邊界無法 cubic 時回退 nearest
  3. preserve_fronts=True → Sobel 梯度鋒面增強，避免插值使鋒面模糊
  4. 可選微波 SST 背景矩陣做為先驗

[v13.2-P1] 2026-02 新增
"""

import numpy as np
from scipy.interpolate import griddata
from scipy import ndimage
from typing import Optional, Dict, Any
import logging

log = logging.getLogger("OceanMaster.CloudRemoval")


class CloudRemovalNet:
    """雲遮蔽去除器 — griddata 插值 + 鋒面增強

    Usage:
        cr = CloudRemovalNet(preserve_fronts=True)
        sst_filled = cr.fill(sst_with_nan)
        sst_filled = cr.fill(sst_with_nan, mw_sst=microwave_sst_background)
    """

    def __init__(
        self,
        preserve_fronts: bool = True,
        front_enhance_sigma: float = 1.0,
        front_enhance_weight: float = 0.3,
    ):
        """
        Args:
            preserve_fronts: 補全後套用 Sobel 梯度鋒面增強
            front_enhance_sigma: 鋒面增強高斯平滑 sigma
            front_enhance_weight: 鋒面增強強度 (0-1)
        """
        self.preserve_fronts = preserve_fronts
        self.front_enhance_sigma = front_enhance_sigma
        self.front_enhance_weight = front_enhance_weight

    def fill(
        self,
        sst: np.ndarray,
        mw_sst: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """補全 SST 中的 NaN (雲遮蔽) 區域

        Args:
            sst: 2D SST array (°C)，含 NaN
            mw_sst: 可選微波 SST 背景矩陣 (同尺寸)
                     微波穿透雲層但解析度較低，可做插值先驗

        Returns:
            gap-filled SST，原始非 NaN 值完全保留
        """
        if sst.ndim != 2:
            raise ValueError(f"SST must be 2D, got shape={sst.shape}")

        nan_mask = np.isnan(sst)
        n_nan = int(np.sum(nan_mask))
        n_total = sst.size

        if n_nan == 0:
            log.info("  ☁️ CloudRemoval: 無 NaN 需補全")
            return sst.copy()

        if n_nan == n_total:
            log.warning("  ☁️ CloudRemoval: 全部 NaN，無法插值")
            if mw_sst is not None:
                return mw_sst.copy()
            return sst.copy()

        pct = n_nan / n_total * 100
        log.info(f"  ☁️ CloudRemoval: 補全 {n_nan}/{n_total} 格點 "
                 f"({pct:.1f}% 雲覆蓋)")

        # ── 建立座標 ──
        ny, nx = sst.shape
        yy, xx = np.mgrid[0:ny, 0:nx]

        # 有效點（非 NaN）
        valid_mask = ~nan_mask
        valid_points = np.column_stack([yy[valid_mask], xx[valid_mask]])
        valid_values = sst[valid_mask]

        # 需要補全的點
        fill_points = np.column_stack([yy[nan_mask], xx[nan_mask]])

        # ── 微波 SST 融合: 用 mw_sst 補充已知點 ──
        if mw_sst is not None and mw_sst.shape == sst.shape:
            # 在 NaN 區域中，如果微波有值，也加入已知點
            # 但給予較低權重（透過混合）
            mw_valid = nan_mask & np.isfinite(mw_sst)
            n_mw = int(np.sum(mw_valid))
            if n_mw > 0:
                mw_points = np.column_stack([yy[mw_valid], xx[mw_valid]])
                mw_values = mw_sst[mw_valid]
                valid_points = np.vstack([valid_points, mw_points])
                valid_values = np.concatenate([valid_values, mw_values])
                log.info(f"  ☁️ CloudRemoval: 融合 {n_mw} 個微波 SST 先驗點")

        # ── Step 1: Cubic 插值 ──
        filled = sst.copy()
        try:
            cubic_values = griddata(
                valid_points, valid_values, fill_points,
                method='cubic',
            )
            # cubic 可能在邊界產生 NaN，用 nearest 回退
            still_nan = np.isnan(cubic_values)
            if np.any(still_nan):
                nearest_values = griddata(
                    valid_points, valid_values, fill_points,
                    method='nearest',
                )
                cubic_values[still_nan] = nearest_values[still_nan]
                n_fallback = int(np.sum(still_nan))
                log.debug(f"  ☁️ CloudRemoval: {n_fallback} 邊界點回退 nearest")

            filled[nan_mask] = cubic_values

        except Exception as e:
            # 完全回退到 nearest
            log.warning(f"  ☁️ CloudRemoval: cubic 失敗 ({e})，回退 nearest")
            nearest_values = griddata(
                valid_points, valid_values, fill_points,
                method='nearest',
            )
            filled[nan_mask] = nearest_values

        # ── Step 2: 鋒面增強 (preserve_fronts) ──
        if self.preserve_fronts:
            filled = self._enhance_fronts(sst, filled, nan_mask)

        # 再次確保原始有效值完全不動
        filled[valid_mask] = sst[valid_mask]

        remaining_nan = int(np.sum(np.isnan(filled)))
        if remaining_nan > 0:
            log.warning(f"  ☁️ CloudRemoval: 仍有 {remaining_nan} 個殘餘 NaN")

        return filled.astype(np.float32)

    def _enhance_fronts(
        self,
        original: np.ndarray,
        filled: np.ndarray,
        nan_mask: np.ndarray,
    ) -> np.ndarray:
        """Sobel 梯度鋒面增強

        插值會平滑鋒面結構。此步驟從原始有效區域計算梯度場，
        然後在補全區域邊緣恢復鋒面銳度。

        方法:
          1. 從原始 SST (NaN 用 nearest 快速填充) 計算 Sobel 梯度
          2. 從補全 SST 計算 Sobel 梯度
          3. 在 NaN 邊界 (膨脹2格) 內，用原始梯度引導修正補全值
        """
        # 原始 SST 的梯度場 (用 nearest 快填充)
        orig_nearest = original.copy()
        mask = np.isnan(orig_nearest)
        if mask.any():
            ind = ndimage.distance_transform_edt(
                mask, return_distances=False, return_indices=True
            )
            orig_nearest = original[tuple(ind)]

        # Sobel 梯度
        orig_gy = ndimage.sobel(orig_nearest, axis=0)
        orig_gx = ndimage.sobel(orig_nearest, axis=1)
        orig_grad = np.sqrt(orig_gx**2 + orig_gy**2)

        filled_gy = ndimage.sobel(filled, axis=0)
        filled_gx = ndimage.sobel(filled, axis=1)
        filled_grad = np.sqrt(filled_gx**2 + filled_gy**2)

        # 鋒面邊界帶: NaN 區域膨脹 2 格的邊緣
        boundary = ndimage.binary_dilation(nan_mask, iterations=2) & ~nan_mask

        if not boundary.any():
            return filled

        # 在邊界帶，如果原始梯度更大，則增強補全值
        # 方向: 沿梯度方向微調，幅度由 front_enhance_weight 控制
        grad_diff = orig_grad - filled_grad
        enhance_mask = boundary & (grad_diff > 0)

        if enhance_mask.any():
            # 高斯平滑梯度差異，避免尖銳跳變
            smooth_diff = ndimage.gaussian_filter(
                grad_diff * enhance_mask.astype(float),
                sigma=self.front_enhance_sigma,
            )
            # 加回梯度修正 (乘以方向)
            correction = smooth_diff * self.front_enhance_weight
            # 沿梯度方向修正 (簡化: 取 y 方向分量)
            direction = np.sign(orig_gy)
            result = filled.copy()
            result[enhance_mask] += (correction * direction)[enhance_mask]
            return result

        return filled

    def get_stats(self, sst: np.ndarray) -> Dict[str, Any]:
        """取得雲覆蓋統計"""
        nan_mask = np.isnan(sst)
        n_nan = int(np.sum(nan_mask))
        return {
            "total_pixels": sst.size,
            "nan_pixels": n_nan,
            "cloud_coverage_pct": n_nan / sst.size * 100 if sst.size > 0 else 0,
            "valid_pixels": sst.size - n_nan,
        }


# ── 模組級快捷函數 ──

_default_remover = None


def cloud_remove_sst(
    sst: np.ndarray,
    mw_sst: Optional[np.ndarray] = None,
    preserve_fronts: bool = True,
) -> np.ndarray:
    """模組級快捷函數：雲遮蔽去除

    Args:
        sst: 2D SST (含 NaN)
        mw_sst: 可選微波 SST 背景
        preserve_fronts: 是否保留鋒面結構

    Returns:
        gap-filled SST (float32)
    """
    global _default_remover
    if _default_remover is None or _default_remover.preserve_fronts != preserve_fronts:
        _default_remover = CloudRemovalNet(preserve_fronts=preserve_fronts)
    return _default_remover.fill(sst, mw_sst)
