"""
OceanMaster v13.2 — U-Net Fishing Ground Spatial Prediction (PyTorch)
=====================================================================
蒼鷺 AI 核心架構復現：Modified U-Net + CBAM Attention

Architecture:
  - Encoder: 4 down-sampling blocks (Conv-BN-ReLU ×2 + MaxPool)
  - Bottleneck: 512 → 1024
  - Decoder: 4 up-sampling blocks (ConvTranspose + skip concat + Conv-BN-ReLU ×2)
  - CBAM Attention after each decoder block
  - Output: Conv2d(64, 1, 1) → Sigmoid → fishing probability map

References:
  - Ronneberger et al. (2015) U-Net: Convolutional Networks for Biomedical Image Segmentation
  - Woo et al. (2018) CBAM: Convolutional Block Attention Module
  - 蒼鷺 AI: Modified U-Net for squid fishing ground prediction (水產學報 2024)

Note: Requires PyTorch. Requires real CPUE data for meaningful training.
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("OceanMaster.UNet")

# ── Lazy PyTorch import ──
_torch = None
_nn = None
_F = None


def _ensure_torch():
    global _torch, _nn, _F
    if _torch is None:
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            _torch = torch
            _nn = nn
            _F = F
        except ImportError:
            raise ImportError(
                "PyTorch is required for U-Net. "
                "Install: pip install torch"
            )
    return _torch, _nn, _F


# ═══════════════════════════════════════════════════════════
#  Building Blocks
# ═══════════════════════════════════════════════════════════

class DoubleConv:
    """Two sequential Conv2d-BatchNorm-ReLU blocks."""

    def __new__(cls, in_ch: int, out_ch: int):
        torch, nn, _ = _ensure_torch()

        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DownBlock:
    """Encoder block: MaxPool2d → DoubleConv."""

    def __new__(cls, in_ch: int, out_ch: int):
        torch, nn, _ = _ensure_torch()

        return nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )


def _make_cbam(channels: int, reduction: int = 16):
    """
    CBAM: Convolutional Block Attention Module (Woo et al. 2018).

    Channel Attention: GAP + GMP → SharedMLP → sigmoid
    Spatial Attention: AvgPool + MaxPool along channel → Conv7×7 → sigmoid
    """
    torch, nn, F = _ensure_torch()

    class ChannelAttention(nn.Module):
        def __init__(self, ch, r):
            super().__init__()
            mid = max(ch // r, 1)
            self.mlp = nn.Sequential(
                nn.Linear(ch, mid, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(mid, ch, bias=False),
            )

        def forward(self, x):
            B, C, H, W = x.shape
            # Global average pool
            avg = x.mean(dim=[2, 3])  # (B, C)
            # Global max pool
            mx = x.amax(dim=[2, 3])   # (B, C)
            # Shared MLP
            att = torch.sigmoid(self.mlp(avg) + self.mlp(mx))  # (B, C)
            return x * att.unsqueeze(-1).unsqueeze(-1)

    class SpatialAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

        def forward(self, x):
            avg = x.mean(dim=1, keepdim=True)   # (B, 1, H, W)
            mx = x.amax(dim=1, keepdim=True)     # (B, 1, H, W)
            cat = torch.cat([avg, mx], dim=1)    # (B, 2, H, W)
            att = torch.sigmoid(self.conv(cat))  # (B, 1, H, W)
            return x * att

    class CBAM(nn.Module):
        def __init__(self, ch, r):
            super().__init__()
            self.ca = ChannelAttention(ch, r)
            self.sa = SpatialAttention()

        def forward(self, x):
            x = self.ca(x)
            x = self.sa(x)
            return x

    return CBAM(channels, reduction)


# ═══════════════════════════════════════════════════════════
#  U-Net Model
# ═══════════════════════════════════════════════════════════

def _build_unet_model(in_channels: int, use_cbam: bool = True):
    """Build the full U-Net model as an nn.Module."""
    torch, nn, F = _ensure_torch()

    class UNet(nn.Module):
        """
        Modified U-Net with optional CBAM Attention.

        Architecture:
          Encoder: in → 64 → 128 → 256 → 512
          Bottleneck: 512 → 1024
          Decoder: 1024 → 512 → 256 → 128 → 64
          Output: 64 → 1 (sigmoid)
        """

        def __init__(self, n_channels, cbam=True):
            super().__init__()
            self.n_channels = n_channels

            # Encoder
            self.inc = DoubleConv(n_channels, 64)
            self.down1 = DownBlock(64, 128)
            self.down2 = DownBlock(128, 256)
            self.down3 = DownBlock(256, 512)

            # Bottleneck
            self.bottleneck = DownBlock(512, 1024)

            # Decoder
            self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
            self.dec4 = DoubleConv(1024, 512)  # 512 (up) + 512 (skip) = 1024

            self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.dec3 = DoubleConv(512, 256)

            self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.dec2 = DoubleConv(256, 128)

            self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.dec1 = DoubleConv(128, 64)

            # CBAM on decoder blocks
            self.use_cbam = cbam
            if cbam:
                self.cbam4 = _make_cbam(512)
                self.cbam3 = _make_cbam(256)
                self.cbam2 = _make_cbam(128)
                self.cbam1 = _make_cbam(64)

            # Output
            self.outc = nn.Conv2d(64, 1, kernel_size=1)

        def forward(self, x):
            # Ensure minimum spatial dimensions for 4 pooling layers
            # Bottleneck needs >= 2×2 spatial → input needs >= 32×32
            _, _, H, W = x.shape
            min_size = 32
            target_h = max(H, min_size)
            target_w = max(W, min_size)
            # Round up to multiple of 16
            target_h = target_h + (16 - target_h % 16) % 16
            target_w = target_w + (16 - target_w % 16) % 16
            pad_h = target_h - H
            pad_w = target_w - W
            if pad_h > 0 or pad_w > 0:
                # Use constant padding (0) — reflect fails when pad >= dim
                x = F.pad(x, [0, pad_w, 0, pad_h], mode='constant', value=0)

            # Encoder
            x1 = self.inc(x)      # (B, 64, H, W)
            x2 = self.down1(x1)   # (B, 128, H/2, W/2)
            x3 = self.down2(x2)   # (B, 256, H/4, W/4)
            x4 = self.down3(x3)   # (B, 512, H/8, W/8)

            # Bottleneck
            x5 = self.bottleneck(x4)  # (B, 1024, H/16, W/16)

            # Decoder with skip connections
            d4 = self.up4(x5)  # (B, 512, H/8, W/8)
            # Handle size mismatch from odd dimensions
            if d4.shape != x4.shape:
                d4 = F.interpolate(d4, size=x4.shape[2:], mode='bilinear', align_corners=False)
            d4 = torch.cat([d4, x4], dim=1)  # (B, 1024, H/8, W/8)
            d4 = self.dec4(d4)  # (B, 512, H/8, W/8)
            if self.use_cbam:
                d4 = self.cbam4(d4)

            d3 = self.up3(d4)
            if d3.shape != x3.shape:
                d3 = F.interpolate(d3, size=x3.shape[2:], mode='bilinear', align_corners=False)
            d3 = torch.cat([d3, x3], dim=1)
            d3 = self.dec3(d3)
            if self.use_cbam:
                d3 = self.cbam3(d3)

            d2 = self.up2(d3)
            if d2.shape != x2.shape:
                d2 = F.interpolate(d2, size=x2.shape[2:], mode='bilinear', align_corners=False)
            d2 = torch.cat([d2, x2], dim=1)
            d2 = self.dec2(d2)
            if self.use_cbam:
                d2 = self.cbam2(d2)

            d1 = self.up1(d2)
            if d1.shape != x1.shape:
                d1 = F.interpolate(d1, size=x1.shape[2:], mode='bilinear', align_corners=False)
            d1 = torch.cat([d1, x1], dim=1)
            d1 = self.dec1(d1)
            if self.use_cbam:
                d1 = self.cbam1(d1)

            # Output
            out = self.outc(d1)  # (B, 1, H_padded, W_padded)

            # Remove padding
            if pad_h > 0 or pad_w > 0:
                out = out[:, :, :H, :W]

            return torch.sigmoid(out)

    return UNet(in_channels, use_cbam)


# ═══════════════════════════════════════════════════════════
#  Public API — compatible with existing OceanMaster interface
# ═══════════════════════════════════════════════════════════

class UNetFishingPredictor:
    """
    U-Net for full-map fishing prediction.

    蒼鷺 AI 架構復現 — Modified U-Net + CBAM Attention.

    Channels (default):
      0: SST, 1: CHL, 2: SSH, 3: U_current, 4: V_current,
      5: Depth, 6: NPP

    Output: fishing probability map (0-1, same spatial resolution as input)

    Usage:
        predictor = UNetFishingPredictor()
        result = predictor.predict(features, lats, lons)
        prob_map = result["fishing_probability"]  # 2D numpy array
    """

    FEATURE_NAMES = ["sst", "chl", "ssh", "u_current", "v_current", "depth", "npp"]
    N_FEATURES = len(FEATURE_NAMES)

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_cbam: bool = True,
        device: Optional[str] = None,
    ):
        self.is_trained = False
        self.use_cbam = use_cbam
        self._model = None
        self._device = None

        try:
            torch, nn, _ = _ensure_torch()
            self._device = torch.device(
                device if device else ("cuda" if torch.cuda.is_available() else "cpu")
            )
            self._model = _build_unet_model(self.N_FEATURES, use_cbam=use_cbam)
            self._model = self._model.to(self._device)
            self._model.eval()

            n_params = sum(p.numel() for p in self._model.parameters())
            log.info(
                f"  U-Net: {n_params:,} parameters, CBAM={'ON' if use_cbam else 'OFF'}, "
                f"device={self._device}"
            )

            if model_path:
                self.load_weights(model_path)

        except ImportError:
            log.warning(
                "  U-Net: PyTorch not available. "
                "Install: pip install torch. "
                "Falling back to random output."
            )

    def load_weights(self, path: str):
        """Load trained model weights from a .pt file."""
        torch, _, _ = _ensure_torch()
        try:
            state_dict = torch.load(path, map_location=self._device, weights_only=True)
            self._model.load_state_dict(state_dict)
            self._model.eval()
            self.is_trained = True
            log.info(f"  U-Net: loaded weights from {path}")
        except Exception as e:
            log.warning(f"  U-Net: failed to load weights: {e}")

    def save_weights(self, path: str):
        """Save model weights to a .pt file."""
        torch, _, _ = _ensure_torch()
        if self._model is None:
            raise RuntimeError("No model to save")
        from pathlib import Path as P
        P(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), path)
        log.info(f"  U-Net: saved weights to {path}")

    def get_model(self):
        """Get the raw PyTorch nn.Module for training."""
        if self._model is None:
            raise RuntimeError("PyTorch model not initialized")
        return self._model

    def predict(
        self,
        features: Dict[str, np.ndarray],
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Run U-Net inference on feature grids.

        Args:
            features: dict with keys from FEATURE_NAMES, values are 2D grids (H, W)
            lats: 1D latitude array
            lons: 1D longitude array

        Returns:
            {
                "fishing_probability": 2D grid (H, W), values 0-1,
                "is_trained": bool,
            }
        """
        first_key = next(
            (k for k in self.FEATURE_NAMES if k in features), None
        )
        if first_key is None:
            raise ValueError(f"features must contain at least one of {self.FEATURE_NAMES}")

        H, W = features[first_key].shape

        if self._model is None:
            # No PyTorch — return zeros
            log.warning("  U-Net: no PyTorch model, returning zeros")
            return {
                "fishing_probability": np.zeros((H, W), dtype=np.float32),
                "is_trained": False,
            }

        torch, _, _ = _ensure_torch()

        # Stack and normalize input features: (1, C, H, W)
        x = np.zeros((self.N_FEATURES, H, W), dtype=np.float32)
        for i, fname in enumerate(self.FEATURE_NAMES):
            grid = features.get(fname)
            if grid is not None:
                arr = np.asarray(grid, dtype=np.float32)
                mean = float(np.nanmean(arr))
                std = float(np.nanstd(arr)) + 1e-8
                x[i] = np.nan_to_num((arr - mean) / std, nan=0.0)

        x_tensor = torch.from_numpy(x).unsqueeze(0).to(self._device)  # (1, C, H, W)

        # Inference
        with torch.no_grad():
            prob = self._model(x_tensor)  # (1, 1, H, W)

        prob_np = prob[0, 0].cpu().numpy()  # (H, W) — safe for any spatial dims

        if not self.is_trained:
            log.warning(
                "  U-Net: untrained model — output is random noise. "
                "Train with real CPUE data for meaningful predictions."
            )

        log.info(f"  🗺️ U-Net: grid {H}×{W}, mean_prob={float(np.mean(prob_np)):.3f}")
        return {"fishing_probability": prob_np, "is_trained": self.is_trained}
