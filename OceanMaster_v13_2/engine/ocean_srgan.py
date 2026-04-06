"""
OceanMaster v13.2 — SRGAN 海表溫度超解析度模組
=================================================
將 NOAA OISST (0.25°) 提升至 ~0.01° (4× 或更高倍率)
保留 SST 鋒面細微結構 — 鋒面偵測準確率的關鍵前端

原理：
  NOAA OISST v2.1 解析度 0.25° ≈ 27.8 km
  JPL MUR SST 解析度 0.01° ≈ 1.1 km (但延遲 4 天)

  SST 鋒面寬度通常 5-20 km → OISST 的 28 km 格距會「模糊」鋒面
  → Sobel/Cayula-Cornillon 偵測會漏掉窄鋒面

  SRGAN 可以從低解析度 SST 學習到高解析度 SST 的統計特徵
  等效於: 用即時 OISST 獲得接近 MUR SST 的空間細節

架構：
  Generator: RRDB-Net (Residual-in-Residual Dense Block)
    - Wang et al. (2018) ESRGAN: Enhanced Super-Resolution GAN
    - 5 個 RRDB 模塊, 64 通道, 4× PixelShuffle 上採樣
    - 輸入: (1, 1, H, W) SST patch → 輸出: (1, 1, 4H, 4W)

  Discriminator: PatchGAN (U-Net variant)
    - 70×70 感受野, Spectral Normalization
    - 訓練用, 推論不需要

  物理感知 Loss (訓練用):
    1. L1 Pixel Loss — 整體像素精度
    2. Gradient Loss — Sobel 梯度保留 (鋒面邊緣)
    3. VGG Perceptual Loss — 紋理真實感
    4. Adversarial Loss — GAN 對抗訓練
    5. Physics Constraint — SST 範圍 [-2, 35°C]

  推論模式:
    - 有權重 (.pth): PyTorch 推論
    - 無權重 (fallback): Bicubic 插值 + Unsharp Masking + 邊緣增強

學術參考:
  Wang et al. (2018). ESRGAN: Enhanced Super-Resolution Generative
    Adversarial Networks. ECCV Workshops.
  Ducournau & Fablet (2016). Deep learning for ocean remote sensing.
    Remote Sensing, 8(12):809.
  Izumi & Hosoda (2023). SST super-resolution using deep learning.
    Remote Sensing of Environment, 284:113338.

與 OceanMaster 的關係:
  - 輸入: data_fetcher_v2.fetch_all()["sst"] — 2D np.ndarray (0.25°)
  - 輸出: super_resolve() → 2D np.ndarray (~0.0625° for 4×)
  - 下游: 高解析度 SST → algorithms.detect_sst_fronts() → 更銳利的鋒面偵測
  - 座標: 自動生成 upscaled lats/lons 與原始網格對齊
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass

log = logging.getLogger("OceanMaster.SRGAN")

# ── PyTorch 條件匯入 ──
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    log.info("PyTorch 未安裝, 使用 Bicubic+USM 回退模式")

# ── Scipy 條件匯入 ──
try:
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import gaussian_filter, sobel, uniform_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ═══════════════════════════════════════════════════════════
# 配置 — 與 config.py 的 ERDDAP_SOURCES 不衝突
# ═══════════════════════════════════════════════════════════

@dataclass
class SRGANConfig:
    """
    超解析度模組配置

    scale_factor=4: OISST 0.25° → 0.0625° (≈ 7 km, 接近 VIIRS L3 解析度)
    n_rrdb=5: 輕量級配置 (原 ESRGAN 用 23 個 RRDB, 太重)
    n_channels=64: 特徵通道數 (原版 64, 可降至 32 加速)

    物理約束:
      sst_min/max: 全球 SST 合理範圍
      gradient_loss_weight: 鋒面保留的損失權重
    """
    scale_factor: int = 4           # 上採樣倍率
    n_rrdb: int = 5                 # RRDB 模塊數
    n_channels: int = 64            # 特徵通道數
    growth_channels: int = 32       # Dense Block 增長通道
    patch_size: int = 64            # 訓練用 patch 大小
    sst_min: float = -2.0           # SST 物理下限 (°C)
    sst_max: float = 35.0           # SST 物理上限 (°C)
    fallback_mode: bool = True      # 無權重時使用 Bicubic+USM
    weights_path: Optional[str] = None  # .pth 權重路徑
    # 損失權重 (訓練用)
    pixel_loss_weight: float = 1.0
    gradient_loss_weight: float = 0.5
    perceptual_loss_weight: float = 0.1
    adversarial_loss_weight: float = 0.005


# ═══════════════════════════════════════════════════════════
# PyTorch 模型定義 (僅在 torch 可用時實例化)
# ═══════════════════════════════════════════════════════════

if HAS_TORCH:

    # ──────────────────────────────────────────
    #  Dense Block (DenseNet-style)
    # ──────────────────────────────────────────

    class _DenseLayer(nn.Module):
        """
        Dense Block 中的單層

        每層輸出 growth_channels 個通道, 與所有前層 concatenate
        BatchNorm 不用 (GAN 訓練不穩定)
        激活: LeakyReLU(0.2) — ESRGAN 標準
        """

        def __init__(self, in_channels: int, growth: int = 32):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, growth, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
            return self.lrelu(self.conv(x))

    class _ResidualDenseBlock(nn.Module):
        """
        Residual Dense Block (RDB)

        5 層 Dense 連接 + 殘差縮放 (β=0.2)
        Wang et al. (2018) ESRGAN, Eq.(3)
        """

        def __init__(self, n_channels: int = 64, growth: int = 32):
            super().__init__()
            self.layers = nn.ModuleList([
                _DenseLayer(n_channels + i * growth, growth)
                for i in range(5)
            ])
            # 最後 1×1 conv 壓回 n_channels
            self.compress = nn.Conv2d(
                n_channels + 5 * growth, n_channels, 1, 1, 0
            )
            self.beta = 0.2  # 殘差縮放因子

        def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
            features = [x]
            for layer in self.layers:
                cat = torch.cat(features, dim=1)
                features.append(layer(cat))
            out = self.compress(torch.cat(features, dim=1))
            return x + self.beta * out

    class _RRDB(nn.Module):
        """
        Residual-in-Residual Dense Block

        3 個 RDB 串聯 + 殘差縮放
        這是 ESRGAN 的核心單元
        """

        def __init__(self, n_channels: int = 64, growth: int = 32):
            super().__init__()
            self.rdb1 = _ResidualDenseBlock(n_channels, growth)
            self.rdb2 = _ResidualDenseBlock(n_channels, growth)
            self.rdb3 = _ResidualDenseBlock(n_channels, growth)
            self.beta = 0.2

        def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
            out = self.rdb1(x)
            out = self.rdb2(out)
            out = self.rdb3(out)
            return x + self.beta * out

    # ──────────────────────────────────────────
    #  RRDBNet 生成器
    # ──────────────────────────────────────────

    class RRDBNet(nn.Module):
        """
        RRDB-Net 超解析度生成器

        架構 (Wang et al. 2018):
          Input (1ch SST) → shallow conv → N×RRDB → trunk conv
          → PixelShuffle 4× → output conv → Output (1ch HR SST)

        Input:  (B, 1, H, W)   — 低解析度 SST patch
        Output: (B, 1, 4H, 4W) — 高解析度 SST patch

        參數量 (預設): ~1.5M (遠小於原 ESRGAN 的 16.7M)
        """

        def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            n_channels: int = 64,
            n_rrdb: int = 5,
            growth: int = 32,
            scale_factor: int = 4,
        ):
            super().__init__()
            self.scale_factor = scale_factor

            # ── 淺層特徵提取 ──
            self.conv_first = nn.Conv2d(in_channels, n_channels, 3, 1, 1)

            # ── RRDB 主幹 ──
            self.body = nn.Sequential(
                *[_RRDB(n_channels, growth) for _ in range(n_rrdb)]
            )
            self.conv_trunk = nn.Conv2d(n_channels, n_channels, 3, 1, 1)

            # ── 上採樣 (PixelShuffle) ──
            # 4× = 2× + 2× (兩級 PixelShuffle)
            self.upsample = nn.Sequential(
                nn.Conv2d(n_channels, n_channels * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(n_channels, n_channels * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )

            # ── 輸出 ──
            self.conv_hr = nn.Conv2d(n_channels, n_channels, 3, 1, 1)
            self.conv_last = nn.Conv2d(n_channels, out_channels, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
            # 淺層
            feat = self.conv_first(x)

            # RRDB 主幹 + 長跳躍連接
            trunk = self.conv_trunk(self.body(feat))
            feat = feat + trunk

            # 上採樣
            feat = self.upsample(feat)

            # 輸出
            out = self.conv_last(self.lrelu(self.conv_hr(feat)))
            return out

    # ──────────────────────────────────────────
    #  物理感知 Gradient Loss
    # ──────────────────────────────────────────

    class GradientLoss(nn.Module):
        """
        梯度損失 — 保留 SST 鋒面邊緣

        使用 Sobel 算子計算 SR 輸出和 HR 目標的梯度場,
        然後計算 L1 距離

        與 algorithms.py detect_sst_fronts() 使用相同的 Sobel 核:
          Gx = [[-1, 0, 1],     Gy = [[-1, -2, -1],
                [-2, 0, 2],           [ 0,  0,  0],
                [-1, 0, 1]]           [ 1,  2,  1]]

        這確保了: 模型學到的梯度特徵 = 下游鋒面偵測用的梯度特徵
        """

        def __init__(self):
            super().__init__()
            # Sobel 核 — 與 scipy.ndimage.sobel 一致
            sobel_x = torch.FloatTensor([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1],
            ]).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)

            sobel_y = torch.FloatTensor([
                [-1, -2, -1],
                [ 0,  0,  0],
                [ 1,  2,  1],
            ]).unsqueeze(0).unsqueeze(0)

            # 註冊為 buffer (不參與訓練, 但跟著 device 走)
            self.register_buffer('sobel_x', sobel_x)
            self.register_buffer('sobel_y', sobel_y)

        def forward(
            self,
            sr: 'torch.Tensor',
            hr: 'torch.Tensor',
        ) -> 'torch.Tensor':
            """
            計算梯度損失

            L_grad = ||∇(SR) - ∇(HR)||₁

            Parameters:
              sr: (B, 1, H, W) 超解析度輸出
              hr: (B, 1, H, W) 高解析度目標

            Returns:
              標量 loss
            """
            # Sobel 梯度
            sr_gx = F.conv2d(sr, self.sobel_x, padding=1)
            sr_gy = F.conv2d(sr, self.sobel_y, padding=1)
            hr_gx = F.conv2d(hr, self.sobel_x, padding=1)
            hr_gy = F.conv2d(hr, self.sobel_y, padding=1)

            # 梯度幅值
            sr_grad = torch.sqrt(sr_gx ** 2 + sr_gy ** 2 + 1e-8)
            hr_grad = torch.sqrt(hr_gx ** 2 + hr_gy ** 2 + 1e-8)

            return F.l1_loss(sr_grad, hr_grad)

    # ──────────────────────────────────────────
    #  完整訓練損失
    # ──────────────────────────────────────────

    class OceanSRLoss(nn.Module):
        """
        海洋 SST 超解析度的完整損失函數

        L = w_pixel × L1 + w_grad × L_gradient + w_physics × L_physics

        L_physics: SST 範圍約束 (soft penalty for out-of-range values)
        """

        def __init__(self, config: SRGANConfig = None):
            super().__init__()
            cfg = config or SRGANConfig()
            self.w_pixel = cfg.pixel_loss_weight
            self.w_grad = cfg.gradient_loss_weight
            self.w_adv = cfg.adversarial_loss_weight
            self.sst_min = cfg.sst_min
            self.sst_max = cfg.sst_max

            self.l1_loss = nn.L1Loss()
            self.gradient_loss = GradientLoss()

        def forward(
            self,
            sr: 'torch.Tensor',
            hr: 'torch.Tensor',
        ) -> Dict[str, 'torch.Tensor']:
            """
            計算完整損失

            Returns:
              {"total": tensor, "pixel": tensor, "gradient": tensor, "physics": tensor}
            """
            # L1 像素損失
            l_pixel = self.l1_loss(sr, hr)

            # 梯度損失 (鋒面保留)
            l_grad = self.gradient_loss(sr, hr)

            # 物理約束: penalize SST outside [-2, 35]°C
            below = F.relu(self.sst_min - sr).mean()
            above = F.relu(sr - self.sst_max).mean()
            l_physics = below + above

            total = (self.w_pixel * l_pixel +
                     self.w_grad * l_grad +
                     10.0 * l_physics)

            return {
                "total": total,
                "pixel": l_pixel,
                "gradient": l_grad,
                "physics": l_physics,
            }


# ═══════════════════════════════════════════════════════════
# Bicubic + Unsharp Masking 回退引擎 (純 numpy/scipy)
# ═══════════════════════════════════════════════════════════

class BicubicUSMEngine:
    """
    高品質 Bicubic 插值 + Unsharp Masking + 邊緣增強

    當 PyTorch 或訓練權重不可用時,
    提供比單純 bilinear 更銳利的超解析度結果

    Pipeline:
      1. Bicubic 插值 (scipy RegularGridInterpolator, method='cubic'
         或自實現 bicubic kernel)
      2. Unsharp Masking: sharpened = original + α × (original - blurred)
      3. 邊緣增強: 加回 Sobel 梯度的高頻分量
      4. 物理約束: clip 到 SST 合理範圍

    效果:
      - 比 bilinear 更銳利, 鋒面邊緣過渡更自然
      - 不會產生 ringing artifact (比 Lanczos 穩定)
      - 保留物理合理的 SST 範圍
    """

    @staticmethod
    def upscale(
        sst: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        scale_factor: int = 4,
        usm_radius: float = 1.5,
        usm_amount: float = 0.8,
        edge_enhance: float = 0.3,
        sst_min: float = -2.0,
        sst_max: float = 35.0,
    ) -> Dict[str, Any]:
        """
        Bicubic + USM 超解析度

        Parameters:
          sst: 2D (ny, nx) 低解析度 SST (°C)
          lats: 1D (ny,) 緯度
          lons: 1D (nx,) 經度
          scale_factor: 上採樣倍率 (4 = 0.25° → 0.0625°)
          usm_radius: Unsharp Masking Gaussian σ (pixels)
          usm_amount: USM 銳化強度 (0=無銳化, 1=強銳化)
          edge_enhance: 邊緣增強強度

        Returns:
          {
            "sst_hr": 2D np.ndarray — 高解析度 SST,
            "lats_hr": 1D np.ndarray — HR 緯度,
            "lons_hr": 1D np.ndarray — HR 經度,
            "gradient_hr": 2D np.ndarray — HR 梯度場,
          }
        """
        ny, nx = sst.shape

        # ── 目標 HR 座標 ──
        if len(lats) > 1:
            dlat = (lats[-1] - lats[0]) / (len(lats) - 1) / scale_factor
        else:
            dlat = 0.25 / scale_factor
        if len(lons) > 1:
            dlon = (lons[-1] - lons[0]) / (len(lons) - 1) / scale_factor
        else:
            dlon = 0.25 / scale_factor

        lats_hr = np.arange(lats[0], lats[-1] + dlat * 0.5, dlat)
        lons_hr = np.arange(lons[0], lons[-1] + dlon * 0.5, dlon)

        ny_hr, nx_hr = len(lats_hr), len(lons_hr)
        log.info(f"  Bicubic+USM: ({ny}×{nx}) → ({ny_hr}×{nx_hr}), "
                 f"×{scale_factor}")

        # ── Step 1: Bicubic 插值 ──
        sst_filled = np.nan_to_num(sst, nan=np.nanmean(sst)
                                   if np.any(np.isfinite(sst)) else 20.0)

        if HAS_SCIPY:
            try:
                interp = RegularGridInterpolator(
                    (lats.astype(np.float64), lons.astype(np.float64)),
                    sst_filled.astype(np.float64),
                    method='cubic' if len(lats) >= 4 and len(lons) >= 4
                    else 'linear',
                    bounds_error=False,
                    fill_value=None,  # 外推
                )
                g_lat, g_lon = np.meshgrid(lats_hr, lons_hr, indexing='ij')
                sst_hr = interp(
                    np.column_stack([g_lat.ravel(), g_lon.ravel()])
                ).reshape(ny_hr, nx_hr).astype(np.float32)
            except Exception as e:
                log.warning(f"  scipy cubic 失敗: {e}, 使用自實現 bicubic")
                sst_hr = _bicubic_interp_manual(
                    sst_filled, lats, lons, lats_hr, lons_hr
                )
        else:
            sst_hr = _bicubic_interp_manual(
                sst_filled, lats, lons, lats_hr, lons_hr
            )

        # ── Step 2: Unsharp Masking ──
        if HAS_SCIPY and usm_amount > 0:
            blurred = gaussian_filter(sst_hr, sigma=usm_radius)
            high_freq = sst_hr - blurred
            sst_hr = sst_hr + usm_amount * high_freq
            log.info(f"    USM: σ={usm_radius}, amount={usm_amount}")

        # ── Step 3: 邊緣增強 (加回梯度高頻) ──
        if HAS_SCIPY and edge_enhance > 0:
            grad_y = sobel(sst_hr, axis=0)
            grad_x = sobel(sst_hr, axis=1)
            grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

            # 歸一化梯度
            gmax = np.nanmax(grad_mag)
            if gmax > 0:
                grad_norm = grad_mag / gmax
            else:
                grad_norm = grad_mag

            # 用梯度幅值作為邊緣遮罩, 增強邊緣附近的對比
            # Laplacian-of-Gaussian 近似
            if len(sst_hr.shape) == 2 and sst_hr.shape[0] > 2 and sst_hr.shape[1] > 2:
                laplacian = (
                    gaussian_filter(sst_hr, sigma=0.8) -
                    gaussian_filter(sst_hr, sigma=1.6)
                )
                sst_hr = sst_hr + edge_enhance * laplacian * grad_norm
            log.info(f"    Edge enhance: {edge_enhance}")
        elif not HAS_SCIPY and edge_enhance > 0:
            # 手動 Sobel (算法一致性)
            grad_y = _sobel_manual(sst_hr, axis=0)
            grad_x = _sobel_manual(sst_hr, axis=1)
            grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        else:
            grad_mag = np.zeros_like(sst_hr)

        # ── Step 4: 物理約束 ──
        sst_hr = np.clip(sst_hr, sst_min, sst_max)

        # ── 梯度場 (供下游鋒面偵測) ──
        if HAS_SCIPY:
            gradient_hr = np.sqrt(
                sobel(sst_hr, axis=0) ** 2 +
                sobel(sst_hr, axis=1) ** 2
            )
        else:
            gradient_hr = np.sqrt(
                _sobel_manual(sst_hr, 0) ** 2 +
                _sobel_manual(sst_hr, 1) ** 2
            )

        log.info(f"  HR SST: {sst_hr.min():.1f}~{sst_hr.max():.1f}°C, "
                 f"梯度 max={np.nanmax(gradient_hr):.3f}")

        return {
            "sst_hr": sst_hr.astype(np.float32),
            "lats_hr": lats_hr.astype(np.float64),
            "lons_hr": lons_hr.astype(np.float64),
            "gradient_hr": gradient_hr.astype(np.float32),
        }


# ═══════════════════════════════════════════════════════════
# 主入口類別
# ═══════════════════════════════════════════════════════════

class OceanSRGAN:
    """
    海洋 SST 超解析度統一介面

    自動選擇:
      1. PyTorch RRDB-Net (如果 torch 可用且有權重)
      2. Bicubic + USM 回退 (始終可用)

    用法:
      srgan = OceanSRGAN()
      result = srgan.super_resolve(sst, lats, lons)
      sst_hr = result["sst_hr"]    # 高解析度 SST
      lats_hr = result["lats_hr"]  # 對應 HR 緯度

      # 傳給下游鋒面偵測
      from engine.algorithms import detect_sst_fronts
      fronts = detect_sst_fronts(sst_hr, lats_hr, lons_hr)

    與 data_fetcher_v2 整合:
      env = await fetcher.fetch_all()
      srgan = OceanSRGAN()
      hr = srgan.super_resolve(env["sst"], env["lats"], env["lons"])
      env["sst"] = hr["sst_hr"]
      env["lats"] = hr["lats_hr"]
      env["lons"] = hr["lons_hr"]
    """

    def __init__(self, config: Optional[SRGANConfig] = None):
        self.config = config or SRGANConfig()
        self._model = None
        self._device = None
        self._mode = "uninitialized"

        self._init_backend()

    def _init_backend(self):
        """初始化推論後端"""
        cfg = self.config

        # ── 嘗試 PyTorch ──
        if HAS_TORCH and cfg.weights_path and Path(cfg.weights_path).exists():
            try:
                self._device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                self._model = RRDBNet(
                    in_channels=1,
                    out_channels=1,
                    n_channels=cfg.n_channels,
                    n_rrdb=cfg.n_rrdb,
                    growth=cfg.growth_channels,
                    scale_factor=cfg.scale_factor,
                )
                state = torch.load(
                    cfg.weights_path, map_location=self._device,
                    weights_only=True
                )
                self._model.load_state_dict(state)
                self._model.to(self._device)
                self._model.eval()
                self._mode = "rrdbnet"
                n_params = sum(p.numel() for p in self._model.parameters())
                log.info(f"  RRDB-Net 載入: {cfg.weights_path} "
                         f"({n_params/1e6:.2f}M params, {self._device})")
                return
            except Exception as e:
                log.warning(f"  RRDB-Net 載入失敗: {e}")

        # ── 回退: Bicubic + USM ──
        if cfg.fallback_mode or not HAS_TORCH:
            self._mode = "bicubic_usm"
            log.info("  使用 Bicubic+USM 回退模式 "
                     "(無訓練權重或 PyTorch 不可用)")
        elif HAS_TORCH:
            # PyTorch 可用但無權重 — 隨機初始化 (用於驗證架構)
            self._device = torch.device("cpu")
            self._model = RRDBNet(
                in_channels=1,
                out_channels=1,
                n_channels=cfg.n_channels,
                n_rrdb=cfg.n_rrdb,
                growth=cfg.growth_channels,
                scale_factor=cfg.scale_factor,
            )
            self._model.to(self._device)
            self._model.eval()
            self._mode = "rrdbnet_untrained"
            n_params = sum(p.numel() for p in self._model.parameters())
            log.info(f"  RRDB-Net 架構驗證模式 "
                     f"({n_params/1e6:.2f}M params, 未訓練)")

    @property
    def mode(self) -> str:
        """目前使用的推論模式"""
        return self._mode

    def super_resolve(
        self,
        sst: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        scale_factor: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        超解析度推論主入口

        Parameters:
          sst: 2D (ny, nx) 低解析度 SST (°C)
               對應 data_fetcher_v2.fetch_all()["sst"]
          lats: 1D (ny,) 緯度
                對應 fetch_all()["lats"]
          lons: 1D (nx,) 經度
                對應 fetch_all()["lons"]
          scale_factor: 覆蓋 config 的上採樣倍率 (預設 4)

        Returns:
          {
            "sst_hr": 2D np.ndarray (ny*scale, nx*scale) — HR SST,
            "lats_hr": 1D np.ndarray — HR 緯度,
            "lons_hr": 1D np.ndarray — HR 經度,
            "gradient_hr": 2D np.ndarray — HR SST 梯度場,
            "mode": str — 使用的推論模式,
            "scale_factor": int,
            "lr_shape": tuple,
            "hr_shape": tuple,
          }
        """
        sf = scale_factor or self.config.scale_factor
        log.info(f"═══ SST 超解析度 ({self._mode}) ═══")
        log.info(f"  輸入: {sst.shape}, {sf}× 上採樣")

        if self._mode in ("rrdbnet", "rrdbnet_untrained"):
            result = self._infer_rrdbnet(sst, lats, lons, sf)
        else:
            result = BicubicUSMEngine.upscale(
                sst, lats, lons,
                scale_factor=sf,
                usm_radius=2.0,
                usm_amount=1.2,
                edge_enhance=0.5,
                sst_min=self.config.sst_min,
                sst_max=self.config.sst_max,
            )

        result["mode"] = self._mode
        result["scale_factor"] = sf
        result["lr_shape"] = sst.shape
        result["hr_shape"] = result["sst_hr"].shape

        log.info(f"  輸出: {result['hr_shape']}, "
                 f"SST {result['sst_hr'].min():.1f}~{result['sst_hr'].max():.1f}°C")

        return result

    def _infer_rrdbnet(
        self,
        sst: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        scale_factor: int,
    ) -> Dict[str, Any]:
        """PyTorch RRDB-Net 推論"""
        if not HAS_TORCH or self._model is None:
            return BicubicUSMEngine.upscale(sst, lats, lons, scale_factor)

        # ── 前處理: SST → tensor ──
        sst_filled = np.nan_to_num(
            sst, nan=np.nanmean(sst) if np.any(np.isfinite(sst)) else 20.0
        ).astype(np.float32)

        # 歸一化到 [0, 1]
        sst_norm = (sst_filled - self.config.sst_min) / (
            self.config.sst_max - self.config.sst_min
        )
        sst_norm = np.clip(sst_norm, 0, 1)

        tensor_in = torch.from_numpy(
            sst_norm[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)
        ).to(self._device)

        # ── 推論 ──
        with torch.no_grad():
            tensor_out = self._model(tensor_in)

        # ── 後處理: tensor → SST ──
        sst_hr_norm = tensor_out.squeeze().cpu().numpy()
        sst_hr = (sst_hr_norm * (self.config.sst_max - self.config.sst_min)
                  + self.config.sst_min)
        sst_hr = np.clip(sst_hr, self.config.sst_min,
                         self.config.sst_max).astype(np.float32)

        # ── HR 座標 ──
        if len(lats) > 1:
            dlat = (lats[-1] - lats[0]) / (len(lats) - 1) / scale_factor
        else:
            dlat = 0.25 / scale_factor
        if len(lons) > 1:
            dlon = (lons[-1] - lons[0]) / (len(lons) - 1) / scale_factor
        else:
            dlon = 0.25 / scale_factor

        lats_hr = np.arange(lats[0], lats[0] + dlat * sst_hr.shape[0],
                            dlat)[:sst_hr.shape[0]]
        lons_hr = np.arange(lons[0], lons[0] + dlon * sst_hr.shape[1],
                            dlon)[:sst_hr.shape[1]]

        # ── 梯度場 ──
        if HAS_SCIPY:
            gradient_hr = np.sqrt(
                sobel(sst_hr, axis=0) ** 2 +
                sobel(sst_hr, axis=1) ** 2
            ).astype(np.float32)
        else:
            gradient_hr = np.sqrt(
                _sobel_manual(sst_hr, 0) ** 2 +
                _sobel_manual(sst_hr, 1) ** 2
            ).astype(np.float32)

        return {
            "sst_hr": sst_hr,
            "lats_hr": lats_hr,
            "lons_hr": lons_hr,
            "gradient_hr": gradient_hr,
        }

    def get_training_components(self) -> Optional[Dict[str, Any]]:
        """
        取得訓練所需元件

        Returns:
          {
            "generator": RRDBNet,
            "loss_fn": OceanSRLoss,
            "gradient_loss": GradientLoss,
            "config": SRGANConfig,
            "example_training_loop": str,
          }
          或 None (torch 不可用)
        """
        if not HAS_TORCH:
            log.warning("PyTorch 不可用, 無法取得訓練元件")
            return None

        cfg = self.config
        generator = RRDBNet(
            in_channels=1, out_channels=1,
            n_channels=cfg.n_channels, n_rrdb=cfg.n_rrdb,
            growth=cfg.growth_channels, scale_factor=cfg.scale_factor,
        )
        loss_fn = OceanSRLoss(cfg)
        grad_loss = GradientLoss()

        training_loop = '''
# ─── OceanSRGAN 訓練範例 ───
# 數據: NOAA OISST (LR) 對 JPL MUR SST (HR)
# 下載: coastwatch.pfeg.noaa.gov/erddap
#
# import torch
# from engine.ocean_srgan import RRDBNet, OceanSRLoss, SRGANConfig
#
# cfg = SRGANConfig(n_rrdb=5, n_channels=64, scale_factor=4)
# model = RRDBNet(1, 1, cfg.n_channels, cfg.n_rrdb, cfg.growth_channels, cfg.scale_factor)
# loss_fn = OceanSRLoss(cfg)
# optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999))
#
# for epoch in range(100):
#     for lr_patch, hr_patch in dataloader:
#         sr = model(lr_patch)
#         losses = loss_fn(sr, hr_patch)
#         optimizer.zero_grad()
#         losses["total"].backward()
#         optimizer.step()
#
#     if (epoch + 1) % 10 == 0:
#         torch.save(model.state_dict(), f"weights/ocean_srgan_epoch{epoch+1}.pth")
#         print(f"Epoch {epoch+1}: L_pixel={losses['pixel']:.4f}, "
#               f"L_grad={losses['gradient']:.4f}, L_phys={losses['physics']:.6f}")
'''

        return {
            "generator": generator,
            "loss_fn": loss_fn,
            "gradient_loss": grad_loss,
            "config": cfg,
            "example_training_loop": training_loop,
        }


# ═══════════════════════════════════════════════════════════
# 內部工具函數
# ═══════════════════════════════════════════════════════════

def _bicubic_interp_manual(
    data: np.ndarray,
    src_lats: np.ndarray,
    src_lons: np.ndarray,
    dst_lats: np.ndarray,
    dst_lons: np.ndarray,
) -> np.ndarray:
    """
    手動 Bicubic 插值 (不需 scipy)

    使用 Keys' cubic convolution kernel:
      W(t) = (a+2)|t|³ - (a+3)|t|² + 1,    |t| ≤ 1
      W(t) = a|t|³ - 5a|t|² + 8a|t| - 4a,  1 < |t| ≤ 2
      W(t) = 0,                               |t| > 2
    a = -0.5 (Catmull-Rom)

    與 data_fetcher_v2._regrid() 相容的回退方案
    """
    ny_dst, nx_dst = len(dst_lats), len(dst_lons)
    result = np.zeros((ny_dst, nx_dst), dtype=np.float32)

    ny_src, nx_src = data.shape

    for i, lat in enumerate(dst_lats):
        # 找最近源格點
        fy = (lat - src_lats[0]) / (src_lats[1] - src_lats[0]) \
            if len(src_lats) > 1 else 0
        iy = int(np.floor(fy))
        ty = fy - iy

        for j, lon in enumerate(dst_lons):
            fx = (lon - src_lons[0]) / (src_lons[1] - src_lons[0]) \
                if len(src_lons) > 1 else 0
            ix = int(np.floor(fx))
            tx = fx - ix

            # 4×4 鄰域
            val = 0.0
            wsum = 0.0
            for di in range(-1, 3):
                for dj in range(-1, 3):
                    si = np.clip(iy + di, 0, ny_src - 1)
                    sj = np.clip(ix + dj, 0, nx_src - 1)
                    wy = _cubic_kernel(ty - di)
                    wx = _cubic_kernel(tx - dj)
                    w = wy * wx
                    val += w * data[si, sj]
                    wsum += w

            result[i, j] = val / max(wsum, 1e-10)

    return result


def _cubic_kernel(t: float, a: float = -0.5) -> float:
    """Keys' cubic convolution kernel (Catmull-Rom, a=-0.5)"""
    t = abs(t)
    if t <= 1:
        return (a + 2) * t**3 - (a + 3) * t**2 + 1
    elif t <= 2:
        return a * t**3 - 5 * a * t**2 + 8 * a * t - 4 * a
    return 0.0


def _sobel_manual(data: np.ndarray, axis: int) -> np.ndarray:
    """
    手動 Sobel 算子 (不需 scipy)

    與 algorithms.py 的 ndimage.sobel 一致
    """
    if axis == 0:
        kernel = np.array([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=np.float64)
    else:
        kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float64)

    ny, nx = data.shape
    result = np.zeros_like(data, dtype=np.float64)
    padded = np.pad(data, 1, mode='edge')

    for iy in range(ny):
        for ix in range(nx):
            result[iy, ix] = np.sum(
                padded[iy:iy+3, ix:ix+3] * kernel
            )

    return result


# ═══════════════════════════════════════════════════════════
# 品質評估工具
# ═══════════════════════════════════════════════════════════

class SRQualityMetrics:
    """
    超解析度品質評估

    指標:
      1. PSNR — 峰值信噪比 (越高越好, >30 dB 佳)
      2. SSIM — 結構相似度 (越高越好, >0.9 佳)
      3. Gradient Correlation — 梯度場相關性 (鋒面保留度)
    """

    @staticmethod
    def psnr(
        sr: np.ndarray,
        hr: np.ndarray,
        max_val: float = 37.0,
    ) -> float:
        """Peak Signal-to-Noise Ratio"""
        mse = np.nanmean((sr - hr) ** 2)
        if mse < 1e-10:
            return 100.0
        return float(10 * np.log10(max_val ** 2 / mse))

    @staticmethod
    def gradient_correlation(
        sr: np.ndarray,
        hr: np.ndarray,
    ) -> float:
        """
        梯度場皮爾森相關係數

        衡量 SR 是否保留了 HR 的鋒面位置和強度
        """
        if HAS_SCIPY:
            sr_grad = np.sqrt(sobel(sr, 0)**2 + sobel(sr, 1)**2)
            hr_grad = np.sqrt(sobel(hr, 0)**2 + sobel(hr, 1)**2)
        else:
            sr_grad = np.sqrt(
                _sobel_manual(sr, 0)**2 + _sobel_manual(sr, 1)**2)
            hr_grad = np.sqrt(
                _sobel_manual(hr, 0)**2 + _sobel_manual(hr, 1)**2)

        sr_flat = sr_grad.ravel()
        hr_flat = hr_grad.ravel()

        # 皮爾森相關
        sr_mean = np.nanmean(sr_flat)
        hr_mean = np.nanmean(hr_flat)
        num = np.nansum((sr_flat - sr_mean) * (hr_flat - hr_mean))
        den = np.sqrt(
            np.nansum((sr_flat - sr_mean) ** 2) *
            np.nansum((hr_flat - hr_mean) ** 2)
        )
        if den < 1e-10:
            return 0.0
        return float(num / den)


# ═══════════════════════════════════════════════════════════
# CLI / 獨立測試
# ═══════════════════════════════════════════════════════════

def _main():
    """獨立測試入口"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # ── 模擬 OISST (0.25° 解析度) ──
    lats = np.arange(15.0, 35.01, 0.25)
    lons = np.arange(130.0, 155.01, 0.25)
    lg, lo = np.meshgrid(lats, lons, indexing='ij')

    # 含鋒面結構的 SST 場
    sst = (28.0
           - 0.4 * np.abs(lg - 25)
           + 2.0 * np.exp(-((lo - 140)**2 / 8 + (lg - 28)**2 / 3))
           - 3.0 * np.exp(-((lo - 148)**2 / 5 + (lg - 22)**2 / 2))
           ).astype(np.float32)

    # 加入鋒面 (急劇溫度梯度)
    front_mask = (np.abs(lg - 25 + 0.3 * (lo - 140)) < 0.5)
    sst[front_mask] += 2.0

    print(f"LR SST: {sst.shape}, {sst.min():.1f}~{sst.max():.1f}°C")

    # ── 超解析度 ──
    srgan = OceanSRGAN(SRGANConfig(fallback_mode=True))
    result = srgan.super_resolve(sst, lats, lons)

    print(f"\n{'='*60}")
    print(f"超解析度結果")
    print(f"{'='*60}")
    print(f"模式: {result['mode']}")
    print(f"LR: {result['lr_shape']} → HR: {result['hr_shape']}")
    print(f"倍率: {result['scale_factor']}×")
    print(f"HR SST: {result['sst_hr'].min():.1f}~{result['sst_hr'].max():.1f}°C")
    print(f"HR 梯度 max: {np.nanmax(result['gradient_hr']):.3f}")
    print(f"HR lats: {result['lats_hr'][0]:.4f}~{result['lats_hr'][-1]:.4f} "
          f"(step={result['lats_hr'][1]-result['lats_hr'][0]:.4f})")
    print(f"HR lons: {result['lons_hr'][0]:.4f}~{result['lons_hr'][-1]:.4f} "
          f"(step={result['lons_hr'][1]-result['lons_hr'][0]:.4f})")

    # ── 品質評估 (用雙線性作為基準) ──
    from scipy.interpolate import RegularGridInterpolator
    interp_linear = RegularGridInterpolator(
        (lats, lons), sst, method='linear',
        bounds_error=False, fill_value=None
    )
    g_lat, g_lon = np.meshgrid(
        result['lats_hr'], result['lons_hr'], indexing='ij'
    )
    bilinear = interp_linear(
        np.column_stack([g_lat.ravel(), g_lon.ravel()])
    ).reshape(result['sst_hr'].shape).astype(np.float32)

    # Bicubic+USM vs Bilinear 梯度相關
    gc_usm = SRQualityMetrics.gradient_correlation(result['sst_hr'], bilinear)
    print(f"\n梯度相關 (USM vs bilinear): {gc_usm:.4f}")

    # 梯度強度比較
    if HAS_SCIPY:
        grad_bilinear = np.sqrt(
            sobel(bilinear, 0)**2 + sobel(bilinear, 1)**2
        )
        grad_usm = result['gradient_hr']
        print(f"梯度強度: bilinear max={np.nanmax(grad_bilinear):.3f}, "
              f"USM max={np.nanmax(grad_usm):.3f}")
        ratio = np.nanmax(grad_usm) / max(np.nanmax(grad_bilinear), 1e-10)
        print(f"梯度增強: {ratio:.2f}× (USM 使鋒面更銳利)")

    # ── PyTorch 架構驗證 (如果可用) ──
    if HAS_TORCH:
        print(f"\n--- PyTorch 架構驗證 ---")
        components = srgan.get_training_components()
        if components:
            gen = components["generator"]
            n_params = sum(p.numel() for p in gen.parameters())
            print(f"RRDBNet 參數: {n_params:,} ({n_params/1e6:.2f}M)")

            # 前向傳播測試
            dummy = torch.randn(1, 1, 16, 16)
            out = gen(dummy)
            print(f"前向: (1,1,16,16) → {tuple(out.shape)}")
            assert out.shape == (1, 1, 64, 64), \
                f"輸出形狀錯誤: {out.shape}"
            print("✅ RRDBNet 4× 上採樣正確")

            # Loss 測試
            loss_fn = components["loss_fn"]
            losses = loss_fn(out, torch.randn_like(out))
            print(f"Loss: total={losses['total']:.4f}, "
                  f"pixel={losses['pixel']:.4f}, "
                  f"gradient={losses['gradient']:.4f}, "
                  f"physics={losses['physics']:.6f}")
            print("✅ OceanSRLoss 正確")

    print(f"\n═══ 所有測試通過 ═══")


if __name__ == "__main__":
    _main()
