"""
OceanMaster v13.2 — DL Data Pipeline
=====================================
將 CMEMS/ERDDAP 環境場資料轉換為 PyTorch DL 模型可用的 2D 張量。

Pipeline:
  data_fetcher_v2.fetch_all() → dict of 2D numpy arrays
  → OceanTensorDataset → (input_tensor, target_tensor) pairs
  → DataLoader → batched training

Supports:
  - Single-frame mode: (C, H, W) for U-Net
  - Sequence mode: (T, C, H, W) for ConvLSTM
  - 時間統計特徵: 30 天 → mean/max/trend/std = 4 stats × N_features
  - CPUE density target: point CPUE → 2D KDE field
  - Binary fishing ground target: CPUE > threshold → 0/1

References:
  蒼鷺: SST+Chl-a at 30-day timescale → time statistics as channels
  CATCH: Bottom temp + salinity + DO + CPUE density → ConvLSTM sequence
"""

import logging
import warnings
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

log = logging.getLogger("OceanMaster.DLPipeline")

_torch = None


def _ensure_torch():
    global _torch
    if _torch is None:
        try:
            import torch
            _torch = torch
        except ImportError:
            raise ImportError("PyTorch required. Install: pip install torch")
    return _torch


# ═══════════════════════════════════════════════════════════
#  Feature normalization stats
# ═══════════════════════════════════════════════════════════

# Climatological mean/std for common ocean variables.
# Source: WOA 2023 + GlobColour + CMEMS GLORYS12V1 global statistics.
# Used when per-sample normalization is not desired.
CLIM_STATS = {
    #           (mean,    std)
    "sst":     (22.0,    6.0),    # °C, global ocean 0-30°C
    "chl":     (0.3,     0.5),    # mg/m³, log-normal, heavy-tailed
    "ssh":     (0.0,     0.15),   # m, relative to geoid
    "u_current": (0.0,   0.2),    # m/s
    "v_current": (0.0,   0.2),    # m/s
    "depth":   (-2000.0, 2000.0), # m (negative = below sea level)
    "npp":     (500.0,   400.0),  # mgC/m²/day
    "salinity": (34.5,   1.0),    # PSU
    "do":      (5.0,     2.0),    # mL/L
    "bottom_temp": (5.0, 4.0),    # °C, CMEMS GLORYS bottom layer
}


def normalize_field(
    arr: np.ndarray,
    feature_name: str,
    mode: str = "per_sample",
) -> np.ndarray:
    """
    Normalize a 2D ocean field.

    Args:
        arr: 2D array (H, W)
        feature_name: key in CLIM_STATS
        mode: "per_sample" (zero-mean unit-var per image) or "climatology"

    Returns:
        normalized array, NaN → 0.0
    """
    arr = np.asarray(arr, dtype=np.float32)

    if mode == "climatology" and feature_name in CLIM_STATS:
        mean, std = CLIM_STATS[feature_name]
    else:
        # Fast-path: if all NaN, skip nanmean/nanstd entirely
        if np.all(np.isnan(arr)):
            mean = 0.0
            std = 0.0
        else:
            mean = float(np.nanmean(arr))
            std = float(np.nanstd(arr))

    std = max(std, 1e-8)
    out = (arr - mean) / std
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ═══════════════════════════════════════════════════════════
#  Time statistics (蒼鷺 30-day approach)
# ═══════════════════════════════════════════════════════════

def compute_time_statistics(
    daily_grids: List[np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Compress a sequence of daily 2D fields into 4 statistical channels.

    蒼鷺 approach: 30 days of SST/Chl-a → mean/max/trend/std
    → 4 channels per variable instead of 30.

    Args:
        daily_grids: list of T 2D arrays (H, W)

    Returns:
        {
            "mean": (H, W),
            "max": (H, W),
            "std": (H, W),
            "trend": (H, W),  # linear slope per pixel
        }
    """
    if len(daily_grids) == 0:
        raise ValueError("Empty daily grid list")

    stack = np.stack([np.nan_to_num(g, nan=0.0) for g in daily_grids], axis=0)  # (T, H, W)
    T = stack.shape[0]

    result = {
        "mean": np.mean(stack, axis=0).astype(np.float32),
        "max": np.max(stack, axis=0).astype(np.float32),
        "std": np.std(stack, axis=0).astype(np.float32),
    }

    # Linear trend: slope of linear regression per pixel
    # y = a*t + b → a = (T*Σ(t*y) - Σt*Σy) / (T*Σ(t²) - (Σt)²)
    t_idx = np.arange(T, dtype=np.float32)
    sum_t = t_idx.sum()
    sum_t2 = (t_idx ** 2).sum()
    denom = T * sum_t2 - sum_t ** 2

    if abs(denom) < 1e-8:
        result["trend"] = np.zeros_like(result["mean"])
    else:
        # (T, H, W) × (T,) → sum along T
        sum_ty = np.tensordot(stack, t_idx, axes=([0], [0]))  # (H, W)
        sum_y = stack.sum(axis=0)
        trend = (T * sum_ty - sum_t * sum_y) / denom
        result["trend"] = trend.astype(np.float32)

    return result


# ═══════════════════════════════════════════════════════════
#  CPUE Density Target Generation
# ═══════════════════════════════════════════════════════════

def cpue_to_density_field(
    cpue_points: List[Dict[str, float]],
    lats: np.ndarray,
    lons: np.ndarray,
    bandwidth_deg: float = 0.5,
    binary_threshold: Optional[float] = None,
) -> np.ndarray:
    """
    Convert point CPUE observations to a 2D density field.

    Args:
        cpue_points: list of {"lat": float, "lon": float, "cpue": float}
        lats: 1D target latitude grid
        lons: 1D target longitude grid
        bandwidth_deg: KDE bandwidth in degrees
        binary_threshold: if set, return binary (CPUE > threshold → 1)

    Returns:
        2D array (len(lats), len(lons)), values 0-1 normalized
    """
    ny, nx = len(lats), len(lons)
    density = np.zeros((ny, nx), dtype=np.float64)

    if len(cpue_points) == 0:
        return density.astype(np.float32)

    # Gaussian kernel for each point
    sigma = bandwidth_deg
    for pt in cpue_points:
        lat_diff = lats - pt["lat"]
        lon_diff = lons - pt["lon"]

        # 2D Gaussian kernel
        lat_kern = np.exp(-0.5 * (lat_diff / sigma) ** 2)  # (ny,)
        lon_kern = np.exp(-0.5 * (lon_diff / sigma) ** 2)  # (nx,)

        weight = pt.get("cpue", 1.0)
        density += weight * np.outer(lat_kern, lon_kern)

    # Normalize
    dmax = density.max()
    if dmax > 0:
        density /= dmax

    if binary_threshold is not None:
        density = (density > binary_threshold).astype(np.float32)

    return density.astype(np.float32)


# ═══════════════════════════════════════════════════════════
#  PyTorch Dataset
# ═══════════════════════════════════════════════════════════

def create_unet_dataset(
    samples: List[Tuple[Dict[str, np.ndarray], np.ndarray]],
    feature_names: List[str],
    norm_mode: str = "per_sample",
):
    """
    Create a PyTorch Dataset for U-Net training.

    Args:
        samples: list of (features_dict, target_2d) tuples
                 features_dict has keys from feature_names
                 target_2d is (H, W) CPUE density or binary label
        feature_names: ordered list of feature keys to stack
        norm_mode: "per_sample" or "climatology"

    Returns:
        torch.utils.data.TensorDataset
    """
    torch = _ensure_torch()

    if len(samples) == 0:
        raise ValueError("No samples provided")

    H, W = samples[0][1].shape
    C = len(feature_names)
    N = len(samples)

    X = np.zeros((N, C, H, W), dtype=np.float32)
    Y = np.zeros((N, 1, H, W), dtype=np.float32)

    for idx, (feat_dict, target) in enumerate(samples):
        for c, fname in enumerate(feature_names):
            grid = feat_dict.get(fname)
            if grid is not None:
                X[idx, c] = normalize_field(grid, fname, mode=norm_mode)
        Y[idx, 0] = target

    X_tensor = torch.tensor(X)
    Y_tensor = torch.tensor(Y)

    return torch.utils.data.TensorDataset(X_tensor, Y_tensor)


def create_convlstm_dataset(
    sequences: List[Tuple[List[Dict[str, np.ndarray]], np.ndarray]],
    feature_names: List[str],
    norm_mode: str = "per_sample",
):
    """
    Create a PyTorch Dataset for ConvLSTM training.

    Args:
        sequences: list of (feature_sequence, target_2d) tuples
                   feature_sequence is list of T dicts
                   target_2d is (H, W) CPUE density
        feature_names: ordered list of feature keys
        norm_mode: "per_sample" or "climatology"

    Returns:
        torch.utils.data.TensorDataset
    """
    torch = _ensure_torch()

    if len(sequences) == 0:
        raise ValueError("No sequences provided")

    T = len(sequences[0][0])
    H, W = sequences[0][1].shape
    C = len(feature_names)
    N = len(sequences)

    X = np.zeros((N, T, C, H, W), dtype=np.float32)
    Y = np.zeros((N, 1, H, W), dtype=np.float32)

    for idx, (seq, target) in enumerate(sequences):
        for t, day_dict in enumerate(seq):
            for c, fname in enumerate(feature_names):
                grid = day_dict.get(fname)
                if grid is not None:
                    X[idx, t, c] = normalize_field(grid, fname, mode=norm_mode)
        Y[idx, 0] = target

    X_tensor = torch.tensor(X)
    Y_tensor = torch.tensor(Y)

    return torch.utils.data.TensorDataset(X_tensor, Y_tensor)


def create_dataloader(dataset, batch_size: int = 4, shuffle: bool = True, num_workers: int = 0):
    """Wrap a Dataset into a DataLoader."""
    torch = _ensure_torch()
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if _torch.cuda.is_available() else False,
    )
