"""
OceanMaster v13.2 — SST 預報模組
==================================
用 ConvLSTM 結構預測未來 1-7 天的 SST 場。

架構:
  - 輸入: 過去 N 天的 SST 2D 場 (time_steps, H, W)
  - 模型: ConvLSTM + Deformable-style attention (受 DatLSTM 啟發)
  - 輸出: 未來 D+1 ~ D+7 的 SST 2D 場

整合方式:
  1. 獨立使用: python sst_forecaster.py --days 7
  2. Pipeline 內使用: forecaster.predict(sst_history) → sst_future
  3. 搭配外部 DatLSTM 權重: forecaster.load_datlstm_weights("path/to/weights.pth")

學術依據:
  - DatLSTM: Nanjing Tech, MDPI 2024 (ConvLSTM + Deformable Attention)
  - ST-UNet: ConvLSTM + U-Net for SST prediction
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("OceanMaster.SSTForecaster")

# ═══════════════════════════════════════════════════
# PyTorch model (optional — graceful fallback to climatology)
# ═══════════════════════════════════════════════════

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class ConvLSTMCell(nn.Module):
        """Single ConvLSTM cell for SST spatiotemporal prediction."""

        def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
            super().__init__()
            self.hidden_channels = hidden_channels
            padding = kernel_size // 2
            self.gates = nn.Conv2d(
                in_channels + hidden_channels, 4 * hidden_channels,
                kernel_size=kernel_size, padding=padding, bias=True,
            )

        def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
            combined = torch.cat([x, h], dim=1)
            gates = self.gates(combined)
            i, f, o, g = gates.chunk(4, dim=1)
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            o = torch.sigmoid(o)
            g = torch.tanh(g)
            c_next = f * c + i * g
            h_next = o * torch.tanh(c_next)
            return h_next, c_next

    class SpatialAttention(nn.Module):
        """Spatial attention (deformable-inspired, no deformable conv dependency)."""

        def __init__(self, channels: int):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(channels, channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, 1, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            attn = self.conv(x)
            return x * attn

    class SSTForecastNet(nn.Module):
        """
        SST 預報網路 (DatLSTM 風格).

        Architecture:
          Encoder: 2-layer ConvLSTM (captures temporal dynamics)
          Attention: Spatial attention after encoding
          Decoder: ConvLSTM + Conv head (generates D+1 ~ D+K predictions)
        """

        def __init__(
            self,
            in_channels: int = 1,
            hidden_channels: int = 32,
            forecast_days: int = 7,
        ):
            super().__init__()
            self.forecast_days = forecast_days
            self.hidden_channels = hidden_channels

            # Encoder
            self.enc1 = ConvLSTMCell(in_channels, hidden_channels)
            self.enc2 = ConvLSTMCell(hidden_channels, hidden_channels)
            self.attention = SpatialAttention(hidden_channels)

            # Decoder
            self.dec = ConvLSTMCell(hidden_channels, hidden_channels)
            self.output_conv = nn.Conv2d(hidden_channels, forecast_days, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (B, T, 1, H, W) — past T days of SST

            Returns:
                (B, forecast_days, H, W) — predicted SST for D+1 ~ D+K
            """
            B, T, C, H, W = x.shape
            device = x.device

            # Init hidden states
            h1 = torch.zeros(B, self.hidden_channels, H, W, device=device)
            c1 = torch.zeros_like(h1)
            h2 = torch.zeros_like(h1)
            c2 = torch.zeros_like(h1)

            # Encode
            for t in range(T):
                h1, c1 = self.enc1(x[:, t], h1, c1)
                h2, c2 = self.enc2(h1, h2, c2)

            # Attention
            encoded = self.attention(h2)

            # Decode (all days at once via conv)
            h_dec = encoded
            c_dec = torch.zeros_like(h_dec)
            h_dec, c_dec = self.dec(encoded, h_dec, c_dec)

            out = self.output_conv(h_dec)  # (B, forecast_days, H, W)
            return out


class SSTForecaster:
    """
    SST 預報器 — 封裝 DatLSTM 風格模型 + 氣候態回退。

    使用方式:
        forecaster = SSTForecaster(forecast_days=7)
        sst_future = forecaster.predict(sst_history_3d)
        # sst_future: dict {1: sst_d1, 2: sst_d2, ..., 7: sst_d7}
    """

    def __init__(
        self,
        forecast_days: int = 7,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.forecast_days = forecast_days
        self.device = device
        self.model = None

        if HAS_TORCH and model_path:
            self._load_model(model_path)
        elif HAS_TORCH:
            log.info("  SST Forecaster: no trained weights, will use persistence + climatology")

    def _load_model(self, path: str):
        """Load trained DatLSTM or SSTForecastNet weights."""
        try:
            state = torch.load(path, map_location=self.device, weights_only=True)
            # Infer model config from state dict
            hidden = state.get("config", {}).get("hidden_channels", 32)
            self.model = SSTForecastNet(
                hidden_channels=hidden,
                forecast_days=self.forecast_days,
            )
            self.model.load_state_dict(state["model_state_dict"])
            self.model.eval()
            self.model.to(self.device)
            log.info(f"  SST Forecaster: loaded weights from {path}")
        except Exception as e:
            log.warning(f"  SST Forecaster: failed to load {path}: {e}")
            self.model = None

    def predict(
        self,
        sst_history: np.ndarray,
        lats: Optional[np.ndarray] = None,
        lons: Optional[np.ndarray] = None,
        current_month: int = 6,
    ) -> Dict[int, np.ndarray]:
        """
        預測未來 N 天 SST。

        Args:
            sst_history: (T, H, W) 過去 T 天的 SST 場
                         如果是 (H, W) 單張圖，自動擴展為 T=1
            lats: 緯度陣列 (用於氣候態回退)
            lons: 經度陣列 (用於氣候態回退)
            current_month: 當前月份

        Returns:
            {day_offset: sst_grid} e.g. {1: array(H,W), 2: array(H,W), ...}
        """
        if sst_history.ndim == 2:
            sst_history = sst_history[np.newaxis, ...]  # (1, H, W)

        T, H, W = sst_history.shape

        # Method 1: Use trained model (if available)
        if self.model is not None and HAS_TORCH:
            return self._predict_dl(sst_history)

        # Method 2: Persistence + damped climatology (no model needed)
        return self._predict_persistence_clim(sst_history, current_month)

    def _predict_dl(self, sst_history: np.ndarray) -> Dict[int, np.ndarray]:
        """用深度學習模型預測。"""
        T, H, W = sst_history.shape
        x = torch.from_numpy(sst_history).float()
        x = x.unsqueeze(0).unsqueeze(2)  # (1, T, 1, H, W)
        x = x.to(self.device)

        with torch.no_grad():
            out = self.model(x)  # (1, forecast_days, H, W)

        out_np = out.cpu().numpy()[0]  # (forecast_days, H, W)

        result = {}
        for d in range(self.forecast_days):
            result[d + 1] = out_np[d]

        log.info(f"  SST Forecast (DL): {self.forecast_days} days, "
                 f"shape={H}x{W}")
        return result

    def _predict_persistence_clim(
        self,
        sst_history: np.ndarray,
        month: int = 6,
    ) -> Dict[int, np.ndarray]:
        """
        Persistence + damped climatology 預報。

        原理:
          - D+1: 主要靠 persistence (今天的 SST ≈ 明天的 SST)
          - D+3: persistence 權重下降，climatology 權重上升
          - D+7: 主要靠季節性趨勢

        這不是隨便的方法 — 這是 NOAA 官方 SST 預報的 baseline。
        任何深度學習模型如果打不贏 persistence，就不值得用。
        """
        T, H, W = sst_history.shape

        # 當前 SST (最後一張)
        sst_current = sst_history[-1]

        # 趨勢估算 (如果有多天歷史)
        if T >= 3:
            # 用最近 3 天的線性趨勢
            trend = (sst_history[-1] - sst_history[-3]) / 2.0
        elif T >= 2:
            trend = sst_history[-1] - sst_history[-2]
        else:
            trend = np.zeros_like(sst_current)

        # 限制趨勢幅度 (SST 一天變化不會超過 ±0.5°C)
        trend = np.clip(trend, -0.5, 0.5)

        # 季節性日變化 (北半球, 簡化)
        # 夏天 SST 每天約 +0.02°C, 冬天約 -0.02°C
        if month in (4, 5, 6, 7, 8):
            seasonal_drift = 0.02
        elif month in (10, 11, 12, 1, 2):
            seasonal_drift = -0.02
        else:
            seasonal_drift = 0.0

        result = {}
        for d in range(1, self.forecast_days + 1):
            # Persistence 權重衰減 (指數衰減)
            persistence_weight = np.exp(-d / 5.0)  # D+1: 0.82, D+3: 0.55, D+7: 0.25
            clim_weight = 1.0 - persistence_weight

            # 預測 = persistence × (current + trend×d) + clim × seasonal_drift
            sst_pred = (
                persistence_weight * (sst_current + trend * d * 0.5) +
                clim_weight * (sst_current + seasonal_drift * d)
            )

            # 物理約束: SST 不會低於 -2°C (結冰點) 或高於 35°C
            sst_pred = np.clip(sst_pred, -2.0, 35.0)

            result[d] = sst_pred.astype(np.float32)

        log.info(f"  SST Forecast (persistence+clim): {self.forecast_days} days, "
                 f"D+1 Δ={float(np.mean(np.abs(result[1] - sst_current))):.3f}°C, "
                 f"D+7 Δ={float(np.mean(np.abs(result[self.forecast_days] - sst_current))):.3f}°C")
        return result

    def forecast_hsi(
        self,
        sst_future: Dict[int, np.ndarray],
        hsi_model_fn=None,
        current_features: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[int, np.ndarray]:
        """
        用預報 SST 生成未來漁場預測。

        Args:
            sst_future: {day_offset: sst_grid} (from predict())
            hsi_model_fn: callable(features_dict) → hsi_grid
            current_features: 當前環境特徵 (除 SST 外保持不變)

        Returns:
            {day_offset: hsi_grid}
        """
        if hsi_model_fn is None:
            log.warning("  SST Forecast → HSI: no model function provided, "
                        "returning SST-only proxy")
            # 簡易 proxy: SST 在 25-30°C 時 HSI 高
            result = {}
            for d, sst in sst_future.items():
                hsi = np.exp(-0.5 * ((sst - 27.5) / 3.0) ** 2)
                result[d] = hsi.astype(np.float32)
            return result

        result = {}
        for d, sst in sst_future.items():
            features = dict(current_features) if current_features else {}
            features["sst"] = sst
            hsi = hsi_model_fn(features)
            result[d] = hsi
            log.info(f"  D+{d} HSI: mean={float(hsi.mean()):.3f}, "
                     f"max={float(hsi.max()):.3f}")

        return result


# ═══════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="SST Forecast (DatLSTM-style)")
    parser.add_argument("--days", type=int, default=7, help="Forecast days")
    parser.add_argument("--weights", default=None, help="Path to trained weights")
    parser.add_argument("--demo", action="store_true", help="Run demo with synthetic data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    forecaster = SSTForecaster(
        forecast_days=args.days,
        model_path=args.weights,
    )

    if args.demo:
        print("=" * 50)
        print("  SST Forecaster Demo (Persistence + Climatology)")
        print("=" * 50)

        # Generate synthetic SST history (10 days, 20x20 grid)
        H, W = 20, 20
        lats = np.linspace(15, 30, H)
        lons = np.linspace(120, 135, W)
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

        # Base SST: warmer near equator
        base_sst = 30 - (lat_grid - 15) * 0.3
        history = np.stack([
            base_sst + np.random.normal(0, 0.3, (H, W)) + 0.02 * t
            for t in range(10)
        ])  # (10, H, W)

        print(f"\n  Input: {history.shape[0]} days × {H}×{W} grid")
        print(f"  Current SST: mean={history[-1].mean():.2f}°C")

        result = forecaster.predict(history, lats, lons, current_month=8)

        print(f"\n  Forecast:")
        for d, sst in result.items():
            delta = float(np.mean(np.abs(sst - history[-1])))
            print(f"    D+{d}: mean={sst.mean():.2f}°C, Δ={delta:.3f}°C")

        # Generate HSI proxy
        hsi_future = forecaster.forecast_hsi(result)
        print(f"\n  HSI Proxy (SST-based):")
        for d, hsi in hsi_future.items():
            print(f"    D+{d}: mean HSI={hsi.mean():.3f}, "
                  f"hotspot cells (>0.8)={int((hsi > 0.8).sum())}")

        print(f"\n  ✅ Done. Replace persistence with DatLSTM weights for better accuracy.")


if __name__ == "__main__":
    main()
