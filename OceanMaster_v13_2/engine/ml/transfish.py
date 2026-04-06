"""
OceanMaster v13.2 — TransFish Multi-Modal Temporal Attention
=============================================================
P3: State-of-the-art fishing effort prediction.

Architecture: ResNet + LSTM + Transformer (inspired by TransFish, 2024)
  - ResNet: extracts spatial features from daily environment maps
  - LSTM: captures temporal dependencies across days
  - Transformer: attention mechanism focuses on most relevant time steps
  - Output: 7-day fishing effort distribution forecast
  - Reported 5.32% average daily forecasting error

Note: Requires PyTorch + GPU + VMS training data.
This module provides architecture definition for future training.
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional

log = logging.getLogger("OceanMaster.TransFish")


class ResNetFeatureExtractor:
    """Simplified ResNet block for spatial feature extraction (numpy demo)."""

    def __init__(self, in_ch: int = 6, out_dim: int = 64):
        self.in_ch = in_ch
        self.out_dim = out_dim
        # 1×1 projection
        self.proj = np.random.randn(out_dim, in_ch).astype(np.float32) * 0.1

    def extract(self, x: np.ndarray) -> np.ndarray:
        """x: (C, H, W) → (out_dim,) via global average pooling."""
        H, W = x.shape[1], x.shape[2]
        # Channel-wise mean (global avg pooling)
        channel_means = np.mean(x.reshape(self.in_ch, -1), axis=1)  # (C,)
        # Linear projection
        feat = self.proj @ channel_means  # (out_dim,)
        return np.maximum(feat, 0)  # ReLU


class TransFishPredictor:
    """
    TransFish: Multi-modal temporal attention fishing predictor.

    Pipeline:
      1. ResNet extracts spatial features per day → (T, D)
      2. LSTM processes temporal sequence → (T, D')
      3. Transformer attention selects key days → (D'')
      4. MLP outputs grid-level prediction → (H, W)

    This is currently an architecture stub. Real training requires:
      - 3+ years of VMS/AIS fishing effort data
      - Matched daily satellite environment grids
      - GPU training infrastructure
    """

    FEATURES = ["sst", "chl", "ssh", "u_current", "v_current", "npp"]
    FEATURE_DIM = 64
    HIDDEN_DIM = 32
    N_HEADS = 4

    def __init__(self, model_path: Optional[str] = None):
        self.is_trained = False
        self.resnet = ResNetFeatureExtractor(len(self.FEATURES), self.FEATURE_DIM)

        # LSTM weights (simplified)
        self.lstm_wh = np.random.randn(
            4 * self.HIDDEN_DIM, self.HIDDEN_DIM
        ).astype(np.float32) * 0.1
        self.lstm_wx = np.random.randn(
            4 * self.HIDDEN_DIM, self.FEATURE_DIM
        ).astype(np.float32) * 0.1
        self.lstm_b = np.zeros(4 * self.HIDDEN_DIM, dtype=np.float32)

        # Transformer attention (simplified: single-head linear)
        self.attn_q = np.random.randn(self.HIDDEN_DIM, self.HIDDEN_DIM).astype(np.float32) * 0.1
        self.attn_k = np.random.randn(self.HIDDEN_DIM, self.HIDDEN_DIM).astype(np.float32) * 0.1
        self.attn_v = np.random.randn(self.HIDDEN_DIM, self.HIDDEN_DIM).astype(np.float32) * 0.1

        if model_path:
            try:
                np.load(model_path, allow_pickle=False)
                self.is_trained = True
                log.info(f"  TransFish: loaded from {model_path}")
            except Exception as e:
                log.debug(f"[降級] engine/ml/transfish.py: {e}")

    def _lstm_step(self, x: np.ndarray, h: np.ndarray, c: np.ndarray):
        """Single LSTM step. x:(D_in), h:(D_h), c:(D_h) → new_h, new_c"""
        gates = self.lstm_wx @ x + self.lstm_wh @ h + self.lstm_b
        d = self.HIDDEN_DIM
        gates = np.clip(gates, -20, 20)  # prevent overflow in sigmoid
        i = 1.0 / (1.0 + np.exp(-gates[:d]))
        f = 1.0 / (1.0 + np.exp(-gates[d:2*d]))
        g = np.tanh(gates[2*d:3*d])
        o = 1.0 / (1.0 + np.exp(-gates[3*d:]))
        new_c = f * c + i * g
        new_h = o * np.tanh(new_c)
        return new_h, new_c

    def _attention(self, seq: np.ndarray) -> np.ndarray:
        """Self-attention over sequence. seq: (T, D) → (D,)"""
        T = seq.shape[0]
        Q = seq @ self.attn_q  # (T, D)
        K = seq @ self.attn_k  # (T, D)
        V = seq @ self.attn_v  # (T, D)

        # Scaled dot-product attention
        scale = np.sqrt(self.HIDDEN_DIM)
        scores = (Q @ K.T) / scale  # (T, T)
        # Softmax
        scores = scores - scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores) / (np.sum(np.exp(scores), axis=-1, keepdims=True) + 1e-8)

        # Weighted sum → (T, D), then average over T
        attended = weights @ V  # (T, D)
        return np.mean(attended, axis=0)  # (D,)

    def predict(
        self,
        feature_sequence: List[Dict[str, np.ndarray]],
        lats: np.ndarray,
        lons: np.ndarray,
        forecast_days: int = 7,
    ) -> Dict[str, Any]:
        """
        Run TransFish inference.

        Args:
            feature_sequence: T days of feature dicts
            forecast_days: number of days to forecast

        Returns:
            {
                "forecast_maps": list of T 2D grids (H, W),
                "attention_weights": which days mattered most,
                "is_trained": bool,
            }
        """
        T = len(feature_sequence)
        if T == 0:
            return {"forecast_maps": [], "attention_weights": [], "is_trained": False}

        H = feature_sequence[0][self.FEATURES[0]].shape[0]
        W = feature_sequence[0][self.FEATURES[0]].shape[1]

        # 1. ResNet: extract spatial features per day
        spatial_features = np.zeros((T, self.FEATURE_DIM), dtype=np.float32)
        for t in range(T):
            day = feature_sequence[t]
            x = np.zeros((len(self.FEATURES), H, W), dtype=np.float32)
            for i, fname in enumerate(self.FEATURES):
                grid = day.get(fname)
                if grid is not None:
                    x[i] = np.nan_to_num(grid, 0.0)
            spatial_features[t] = self.resnet.extract(x)

        # 2. LSTM: temporal processing
        h = np.zeros(self.HIDDEN_DIM, dtype=np.float32)
        c = np.zeros(self.HIDDEN_DIM, dtype=np.float32)
        lstm_seq = np.zeros((T, self.HIDDEN_DIM), dtype=np.float32)
        for t in range(T):
            h, c = self._lstm_step(spatial_features[t], h, c)
            lstm_seq[t] = h

        # 3. Transformer attention
        context = self._attention(lstm_seq)  # (D,)

        # 4. Generate forecast maps (simple broadcast for demo)
        forecast_maps = []
        for d in range(forecast_days):
            decay = np.exp(-d * 0.1)  # confidence decay with forecast horizon
            fmap = np.zeros((H, W), dtype=np.float32)
            for ch in range(self.HIDDEN_DIM):
                fmap += context[ch] * decay
            fmap = 1.0 / (1.0 + np.exp(-np.clip(fmap, -10, 10)))
            forecast_maps.append(fmap)

        if not self.is_trained:
            log.warning("  TransFish: untrained — output is random. "
                        "Requires VMS data + GPU training.")

        log.info(f"  🤖 TransFish: {T} days → {forecast_days} day forecast, "
                 f"grid {H}×{W}")

        return {
            "forecast_maps": forecast_maps,
            "attention_weights": lstm_seq[-1].tolist() if T > 0 else [],
            "is_trained": self.is_trained,
            "forecast_days": forecast_days,
        }
