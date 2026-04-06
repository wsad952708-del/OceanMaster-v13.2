"""
OceanMaster v13.2 — ConvLSTM2D Spatiotemporal CPUE Predictor (PyTorch)
======================================================================
CATCH 核心架構復現：多層 ConvLSTM + 輸出卷積 + ReLU

Architecture:
  - Input: sequence of daily environment grids (T, B, C, H, W)
  - Multi-layer ConvLSTM (configurable depth, default 2 layers)
  - Output: Conv2d → ReLU → predicted CPUE density map

References:
  - Shi et al. (2015) Convolutional LSTM Network: A Machine Learning Approach
    for Precipitation Nowcasting
  - CATCH (Blue Economy AI): Multi-layer ConvLSTM for Icelandic fisheries
    (Biology Methods & Protocols, 2024)

Note: Requires PyTorch. Requires real CPUE data for meaningful training.
"""

import logging
import os
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("OceanMaster.ConvLSTM")

# ── Lazy PyTorch import ──
_torch = None
_nn = None


def _ensure_torch():
    global _torch, _nn
    if _torch is None:
        try:
            import torch
            import torch.nn as nn
            _torch = torch
            _nn = nn
        except ImportError:
            raise ImportError(
                "PyTorch is required for ConvLSTM. "
                "Install: pip install torch"
            )
    return _torch, _nn


# ═══════════════════════════════════════════════════════════
#  ConvLSTM Cell — Real Conv2d gates
# ═══════════════════════════════════════════════════════════

def _build_convlstm_cell(input_channels: int, hidden_channels: int, kernel_size: int = 3):
    """Build a single ConvLSTM cell as an nn.Module."""
    torch, nn = _ensure_torch()

    class ConvLSTMCell(nn.Module):
        """
        Convolutional LSTM Cell (Shi et al. 2015).

        Uses real nn.Conv2d for all gates.
        Input gate, Forget gate, Cell candidate, Output gate
        all computed via a single fused convolution for efficiency.
        """

        def __init__(self, in_ch, hid_ch, ks):
            super().__init__()
            self.hidden_channels = hid_ch
            pad = ks // 2

            # Single fused conv for all 4 gates: i, f, g, o
            self.gates_conv = nn.Conv2d(
                in_ch + hid_ch, 4 * hid_ch,
                kernel_size=ks, padding=pad, bias=True
            )

            # Proper LSTM initialization (Jozefowicz et al. 2015)
            # Xavier for weights, forget gate bias = 1.0 (start by remembering)
            nn.init.xavier_uniform_(self.gates_conv.weight)
            nn.init.zeros_(self.gates_conv.bias)
            # Set forget gate bias to 1.0: bias indices [hid_ch : 2*hid_ch]
            self.gates_conv.bias.data[hid_ch:2*hid_ch] = 1.0

        def forward(self, x, h_prev, c_prev):
            """
            Args:
                x: (B, C_in, H, W)
                h_prev: (B, C_hid, H, W)
                c_prev: (B, C_hid, H, W)

            Returns:
                h_next: (B, C_hid, H, W)
                c_next: (B, C_hid, H, W)
            """
            combined = torch.cat([x, h_prev], dim=1)  # (B, C_in+C_hid, H, W)
            gates = self.gates_conv(combined)  # (B, 4*C_hid, H, W)

            hc = self.hidden_channels
            i = torch.sigmoid(gates[:, 0*hc:1*hc])     # input gate
            f = torch.sigmoid(gates[:, 1*hc:2*hc])     # forget gate
            g = torch.tanh(gates[:, 2*hc:3*hc])        # cell candidate
            o = torch.sigmoid(gates[:, 3*hc:4*hc])     # output gate

            c_next = f * c_prev + i * g
            h_next = o * torch.tanh(c_next)

            return h_next, c_next

        def init_hidden(self, batch_size, height, width, device):
            """Initialize zero hidden and cell states."""
            h = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
            c = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
            return h, c

    return ConvLSTMCell(input_channels, hidden_channels, kernel_size)


# ═══════════════════════════════════════════════════════════
#  Multi-Layer ConvLSTM Model
# ═══════════════════════════════════════════════════════════

def _build_convlstm_model(
    in_channels: int,
    hidden_channels: List[int],
    kernel_sizes: List[int],
    output_activation: str = "relu",
):
    """
    Build a multi-layer ConvLSTM model.

    Args:
        in_channels: number of input feature channels
        hidden_channels: list of hidden dims per layer, e.g. [32, 16]
        kernel_sizes: list of kernel sizes per layer, e.g. [3, 3]
        output_activation: "relu" (for CPUE regression) or "sigmoid" (for probability)
    """
    torch, nn = _ensure_torch()

    class MultiLayerConvLSTM(nn.Module):
        """
        Multi-layer ConvLSTM (CATCH architecture).

        Processes a temporal sequence of 2D fields and outputs
        a spatial prediction map.

        CATCH 架構：
          ConvLSTM(in→32) → ConvLSTM(32→16) → Conv2d(16→1) → ReLU
        """

        def __init__(self, n_in, hid_chs, ksizes, out_act):
            super().__init__()
            self.n_layers = len(hid_chs)
            self.hidden_channels = hid_chs

            # Build ConvLSTM layers
            self.cells = nn.ModuleList()
            for i in range(self.n_layers):
                cur_in = n_in if i == 0 else hid_chs[i - 1]
                cell = _build_convlstm_cell(cur_in, hid_chs[i], ksizes[i])
                self.cells.append(cell)

            # Output projection: last hidden → 1 channel
            self.output_conv = nn.Conv2d(hid_chs[-1], 1, kernel_size=1)
            self.out_act = out_act

        def forward(self, x):
            """
            Args:
                x: (B, T, C, H, W) — temporal sequence of 2D fields

            Returns:
                out: (B, 1, H, W) — predicted map
            """
            B, T, C, H, W = x.shape
            device = x.device

            # Initialize hidden states for all layers
            h_states = []
            c_states = []
            for i in range(self.n_layers):
                h, c = self.cells[i].init_hidden(B, H, W, device)
                h_states.append(h)
                c_states.append(c)

            # Process temporal sequence
            for t in range(T):
                input_t = x[:, t]  # (B, C, H, W)

                for i in range(self.n_layers):
                    if i == 0:
                        layer_input = input_t
                    else:
                        layer_input = h_states[i - 1]

                    h_states[i], c_states[i] = self.cells[i](
                        layer_input, h_states[i], c_states[i]
                    )

            # Use final hidden state of last layer → output
            out = self.output_conv(h_states[-1])  # (B, 1, H, W)

            if self.out_act == "relu":
                out = torch.relu(out)
            elif self.out_act == "sigmoid":
                out = torch.sigmoid(out)

            return out

    return MultiLayerConvLSTM(in_channels, hidden_channels, kernel_sizes, output_activation)


# ═══════════════════════════════════════════════════════════
#  Public API — compatible with existing OceanMaster interface
# ═══════════════════════════════════════════════════════════

class ConvLSTMPredictor:
    """
    ConvLSTM2D fishing ground predictor.

    CATCH 架構復現 — Multi-layer ConvLSTM for spatiotemporal prediction.

    Input: sequence of daily environment grids
    Output: predicted CPUE density map

    Architecture (default, matching CATCH):
      ConvLSTM(6→32) → ConvLSTM(32→16) → Conv2d(16→1) → ReLU

    Usage:
        predictor = ConvLSTMPredictor()
        result = predictor.predict(feature_sequence, lats, lons)
        hsi_map = result["predicted_hsi"]  # 2D numpy array
    """

    FEATURES = ["sst", "chl", "ssh", "u_current", "v_current", "npp"]
    N_FEATURES = len(FEATURES)

    def __init__(
        self,
        model_path: Optional[str] = None,
        hidden_channels: Optional[List[int]] = None,
        kernel_sizes: Optional[List[int]] = None,
        output_activation: str = "relu",
        device: Optional[str] = None,
    ):
        self.is_trained = False
        self.model_path = model_path
        self._model = None
        self._device = None

        hid_chs = hidden_channels or [32, 16]
        ksizes = kernel_sizes or [3, 3]

        try:
            torch, nn = _ensure_torch()
            self._device = torch.device(
                device if device else ("cuda" if torch.cuda.is_available() else "cpu")
            )
            self._model = _build_convlstm_model(
                self.N_FEATURES, hid_chs, ksizes, output_activation
            )
            self._model = self._model.to(self._device)
            self._model.eval()

            n_params = sum(p.numel() for p in self._model.parameters())
            log.info(
                f"  ConvLSTM: {n_params:,} params, "
                f"layers={hid_chs}, device={self._device}"
            )

            if model_path and os.path.exists(model_path):
                self.load_weights(model_path)

        except ImportError:
            log.warning(
                "  ConvLSTM: PyTorch not available. "
                "Install: pip install torch"
            )

    def load_weights(self, path: str):
        """Load trained model weights."""
        torch, _ = _ensure_torch()
        try:
            state_dict = torch.load(path, map_location=self._device, weights_only=True)
            self._model.load_state_dict(state_dict)
            self._model.eval()
            self.is_trained = True
            log.info(f"  ConvLSTM: loaded weights from {path}")
        except Exception as e:
            log.warning(f"  ConvLSTM: failed to load weights: {e}")

    def save_weights(self, path: str):
        """Save model weights."""
        torch, _ = _ensure_torch()
        if self._model is None:
            raise RuntimeError("No model to save")
        from pathlib import Path as P
        P(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), path)
        log.info(f"  ConvLSTM: saved weights to {path}")

    def get_model(self):
        """Get raw PyTorch nn.Module for training."""
        if self._model is None:
            raise RuntimeError("PyTorch model not initialized")
        return self._model

    def predict(
        self,
        feature_sequence: List[Dict[str, np.ndarray]],
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Run ConvLSTM inference on a sequence of daily grids.

        Args:
            feature_sequence: list of dicts, each with keys from FEATURES,
                              values are 2D grids (H, W). Length = T (days)
            lats, lons: coordinate arrays

        Returns:
            {
                "predicted_hsi": 2D grid (H, W), values >= 0
                "confidence": 2D grid
                "forecast_days": int
                "is_trained": bool
            }
        """
        T = len(feature_sequence)
        if T == 0:
            return {"predicted_hsi": None, "confidence": None,
                    "forecast_days": 0, "is_trained": False}

        first = feature_sequence[0]
        # Find spatial dims from first available feature key
        first_key = next(
            (k for k in self.FEATURES if k in first), None
        )
        if first_key is None:
            raise ValueError(f"feature_sequence[0] must contain at least one of {self.FEATURES}")
        H = first[first_key].shape[0]
        W = first[first_key].shape[1]

        if self._model is None:
            log.warning("  ConvLSTM: no PyTorch model, returning zeros")
            return {
                "predicted_hsi": np.zeros((H, W), dtype=np.float32),
                "confidence": np.full((H, W), 0.5, dtype=np.float32),
                "forecast_days": T,
                "is_trained": False,
            }

        torch, _ = _ensure_torch()

        # Build input tensor: (1, T, C, H, W)
        x = np.zeros((T, self.N_FEATURES, H, W), dtype=np.float32)
        for t in range(T):
            day = feature_sequence[t]
            for i, fname in enumerate(self.FEATURES):
                grid = day.get(fname)
                if grid is not None:
                    arr = np.asarray(grid, dtype=np.float32)
                    mean = float(np.nanmean(arr))
                    std = float(np.nanstd(arr)) + 1e-8
                    x[t, i] = np.nan_to_num((arr - mean) / std, nan=0.0)

        x_tensor = torch.from_numpy(x).unsqueeze(0).to(self._device)  # (1, T, C, H, W)

        # Inference
        with torch.no_grad():
            out = self._model(x_tensor)  # (1, 1, H, W)

        hsi = out[0, 0].cpu().numpy()  # (H, W) — safe for any spatial dims

        # Confidence: inverse of spatial variance in last hidden state
        confidence = np.clip(1.0 - np.std(hsi) / (np.mean(hsi) + 1e-8), 0.1, 0.95)
        confidence_map = np.full((H, W), confidence, dtype=np.float32)

        if not self.is_trained:
            log.warning(
                "  ConvLSTM: untrained model — predictions are random noise. "
                "Train with historical data for meaningful results."
            )

        log.info(
            f"  🧠 ConvLSTM: processed {T} days, "
            f"grid {H}×{W}, "
            f"mean_HSI={float(np.mean(hsi)):.3f}"
        )

        return {
            "predicted_hsi": hsi,
            "confidence": confidence_map,
            "forecast_days": T,
            "is_trained": self.is_trained,
        }

    def generate_training_data(
        self,
        historical_grids: List[Dict[str, np.ndarray]],
        historical_cpue: List[np.ndarray],
        window: int = 30,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training samples from historical data.

        Args:
            historical_grids: list of daily feature dicts (length N days)
            historical_cpue: list of daily CPUE grids (length N days)
            window: lookback window (days)

        Returns:
            X: (N-window, window, N_FEATURES, H, W) — float32
            Y: (N-window, H, W) — float32
        """
        N = len(historical_grids)
        if N <= window:
            return np.array([]), np.array([])

        H = historical_grids[0][self.FEATURES[0]].shape[0]
        W = historical_grids[0][self.FEATURES[0]].shape[1]

        X = np.zeros((N - window, window, self.N_FEATURES, H, W), dtype=np.float32)
        Y = np.zeros((N - window, H, W), dtype=np.float32)

        for i in range(N - window):
            for t in range(window):
                day = historical_grids[i + t]
                for f, fname in enumerate(self.FEATURES):
                    grid = day.get(fname, np.zeros((H, W)))
                    X[i, t, f] = grid
            Y[i] = historical_cpue[i + window]

        log.info(f"  ConvLSTM training data: {X.shape[0]} samples, "
                 f"window={window}, features={self.N_FEATURES}")
        return X, Y
