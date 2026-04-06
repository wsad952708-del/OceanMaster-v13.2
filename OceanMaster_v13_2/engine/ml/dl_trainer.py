"""
OceanMaster v13.2 — DL Trainer
================================
通用 DL 訓練 loop，U-Net 和 ConvLSTM 共用。

Features:
  - GPU/CPU auto-detect
  - Early stopping with patience
  - Learning rate scheduling (ReduceLROnPlateau)
  - Checkpointing (save best + periodic)
  - Loss functions: BCE, MSE, PinballLoss (Quantile Regression)
  - Metrics: F1 (蒼鷺 benchmark), SSIM (CATCH benchmark), IoU

References:
  蒼鷺 benchmark: F1 = 0.87 (annual mean, squid fishing ground)
  CATCH benchmark: SSIM = 0.955 (cod CPUE density)
"""

import logging
import time
import numpy as np
from typing import Any, Dict, List, Optional, Callable, Tuple
from pathlib import Path

log = logging.getLogger("OceanMaster.DLTrainer")

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
            raise ImportError("PyTorch required. Install: pip install torch")
    return _torch, _nn


# ═══════════════════════════════════════════════════════════
#  Loss Functions
# ═══════════════════════════════════════════════════════════

def get_bce_loss():
    """Binary Cross Entropy — U-Net 二分類漁場 (有魚/沒魚). Note: use BCELoss (not WithLogits) because U-Net output already has sigmoid."""
    _, nn = _ensure_torch()
    return nn.BCELoss()


def get_mse_loss():
    """MSE — ConvLSTM CPUE 回歸."""
    _, nn = _ensure_torch()
    return nn.MSELoss()


def build_pinball_loss(quantile: float = 0.5):
    """
    Pinball (Quantile) Loss — 信賴帶估計.

    GreenFish 推測做法 — 用不同 quantile (0.1, 0.5, 0.9) 訓練
    三個頭，得到 10th/50th/90th percentile 預測。

    L_q(y, ŷ) = max(q*(y-ŷ), (q-1)*(y-ŷ))
    """
    torch, _ = _ensure_torch()

    class PinballLoss(torch.nn.Module):
        def __init__(self, q):
            super().__init__()
            self.q = q

        def forward(self, pred, target):
            error = target - pred
            return torch.mean(
                torch.max(self.q * error, (self.q - 1.0) * error)
            )

    return PinballLoss(quantile)


# ═══════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════

def compute_f1(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """
    Compute F1 score for binary fishing ground prediction.
    蒼鷺 benchmark: F1 = 0.87.

    Args:
        pred: 2D probability map (H, W), values 0-1
        target: 2D binary labels (H, W), values 0 or 1
        threshold: binarize prediction at this value

    Returns:
        F1 score (0-1). Returns 0 if no positive predictions or targets.
    """
    pred_bin = (pred > threshold).astype(np.float32).ravel()
    target_bin = (target > threshold).astype(np.float32).ravel()

    tp = np.sum(pred_bin * target_bin)
    fp = np.sum(pred_bin * (1 - target_bin))
    fn = np.sum((1 - pred_bin) * target_bin)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return float(f1)


def compute_ssim(
    pred: np.ndarray,
    target: np.ndarray,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> float:
    """
    Compute Structural Similarity Index (SSIM).
    CATCH benchmark: SSIM = 0.955.

    Simplified SSIM without Gaussian weighting.
    Works on 2D arrays.

    Args:
        pred: 2D prediction (H, W)
        target: 2D ground truth (H, W)

    Returns:
        SSIM value (-1 to 1, higher is better)
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    mu_p = np.mean(pred)
    mu_t = np.mean(target)
    sigma_p = np.std(pred)
    sigma_t = np.std(target)
    sigma_pt = np.mean((pred - mu_p) * (target - mu_t))

    ssim = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / \
           ((mu_p ** 2 + mu_t ** 2 + C1) * (sigma_p ** 2 + sigma_t ** 2 + C2))

    return float(ssim)


def compute_iou(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """
    Compute Intersection-over-Union for fishing ground overlap.
    More stringent than SSIM for hotspot localization.

    Args:
        pred: 2D probability map (H, W)
        target: 2D binary labels (H, W)

    Returns:
        IoU (0-1)
    """
    pred_bin = pred > threshold
    target_bin = target > threshold

    intersection = np.sum(pred_bin & target_bin)
    union = np.sum(pred_bin | target_bin)

    return float(intersection / (union + 1e-8))


# ═══════════════════════════════════════════════════════════
#  DL Trainer
# ═══════════════════════════════════════════════════════════

class DLTrainer:
    """
    Generic DL training loop for U-Net and ConvLSTM.

    Usage:
        from engine.ml.unet_fishing import UNetFishingPredictor
        from engine.ml.dl_data_pipeline import create_unet_dataset, create_dataloader
        from engine.ml.dl_trainer import DLTrainer, get_bce_loss

        predictor = UNetFishingPredictor(use_cbam=True)
        model = predictor.get_model()

        dataset = create_unet_dataset(samples, feature_names)
        train_loader = create_dataloader(dataset, batch_size=4)

        trainer = DLTrainer(model, loss_fn=get_bce_loss(), lr=1e-3)
        history = trainer.fit(train_loader, val_loader, epochs=50)

        predictor.save_weights("models/unet_squid.pt")
    """

    def __init__(
        self,
        model,
        loss_fn=None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
        checkpoint_dir: str = "models/checkpoints",
    ):
        torch, nn = _ensure_torch()

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.loss_fn = loss_fn or nn.MSELoss()

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")

        log.info(
            f"  DLTrainer: device={self.device}, "
            f"lr={lr}, weight_decay={weight_decay}"
        )

    def train_epoch(self, dataloader) -> float:
        """One training epoch. Returns mean loss."""
        torch, _ = _ensure_torch()
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            if len(batch) == 2:
                X, Y = batch
            else:
                raise ValueError(f"Expected 2 items per batch, got {len(batch)}")

            X = X.to(self.device)
            Y = Y.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(X)

            # Handle shape mismatch: pred might be (B,1,H,W), Y might be (B,1,H,W) or (B,H,W)
            if pred.shape != Y.shape:
                if Y.dim() == 3 and pred.dim() == 4:
                    Y = Y.unsqueeze(1)

            loss = self.loss_fn(pred, Y)
            loss.backward()

            # Gradient clipping (prevent exploding gradients in ConvLSTM)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def validate(self, dataloader) -> Tuple[float, Dict[str, float]]:
        """Validation pass. Returns (loss, metrics_dict)."""
        torch, _ = _ensure_torch()
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                X, Y = batch
                X = X.to(self.device)
                Y = Y.to(self.device)

                pred = self.model(X)

                if pred.shape != Y.shape:
                    if Y.dim() == 3 and pred.dim() == 4:
                        Y = Y.unsqueeze(1)

                loss = self.loss_fn(pred, Y)
                total_loss += loss.item()
                n_batches += 1

                all_preds.append(pred.cpu().numpy())
                all_targets.append(Y.cpu().numpy())

        avg_loss = total_loss / max(n_batches, 1)

        # Compute metrics on concatenated results
        preds_cat = np.concatenate(all_preds, axis=0)  # (N, 1, H, W)
        targets_cat = np.concatenate(all_targets, axis=0)

        # Average metrics across samples
        f1_scores = []
        ssim_scores = []
        iou_scores = []
        for i in range(preds_cat.shape[0]):
            p = preds_cat[i, 0]  # (H, W)
            t = targets_cat[i, 0]
            f1_scores.append(compute_f1(p, t))
            ssim_scores.append(compute_ssim(p, t))
            iou_scores.append(compute_iou(p, t))

        metrics = {
            "f1": float(np.mean(f1_scores)),
            "ssim": float(np.mean(ssim_scores)),
            "iou": float(np.mean(iou_scores)),
        }

        return avg_loss, metrics

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 50,
        patience: int = 10,
        save_every: int = 10,
        model_name: str = "model",
    ) -> Dict[str, List]:
        """
        Full training loop with early stopping.

        Args:
            train_loader: training DataLoader
            val_loader: optional validation DataLoader
            epochs: max epochs
            patience: early stopping patience
            save_every: save checkpoint every N epochs
            model_name: prefix for checkpoint files

        Returns:
            history dict with train_loss, val_loss, val_f1, val_ssim, val_iou per epoch
        """
        torch, _ = _ensure_torch()

        history = {
            "train_loss": [],
            "val_loss": [],
            "val_f1": [],
            "val_ssim": [],
            "val_iou": [],
            "lr": [],
        }

        best_val_loss = float("inf")
        no_improve = 0

        log.info(f"  Training: {epochs} epochs, patience={patience}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # Train
            train_loss = self.train_epoch(train_loader)
            history["train_loss"].append(train_loss)
            history["lr"].append(self.optimizer.param_groups[0]["lr"])

            # Validate
            if val_loader is not None:
                val_loss, metrics = self.validate(val_loader)
                history["val_loss"].append(val_loss)
                history["val_f1"].append(metrics["f1"])
                history["val_ssim"].append(metrics["ssim"])
                history["val_iou"].append(metrics["iou"])

                # LR scheduler
                self.scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve = 0
                    # Save best model
                    best_path = self.checkpoint_dir / f"{model_name}_best.pt"
                    torch.save(self.model.state_dict(), best_path)
                else:
                    no_improve += 1

                dt = time.time() - t0
                log.info(
                    f"  Epoch {epoch}/{epochs} ({dt:.1f}s): "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                    f"F1={metrics['f1']:.3f}, SSIM={metrics['ssim']:.3f}, "
                    f"IoU={metrics['iou']:.3f}, "
                    f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
                )

                if no_improve >= patience:
                    log.info(f"  Early stopping at epoch {epoch} (patience={patience})")
                    break
            else:
                dt = time.time() - t0
                log.info(
                    f"  Epoch {epoch}/{epochs} ({dt:.1f}s): "
                    f"train_loss={train_loss:.4f}"
                )

            # Periodic checkpoint
            if save_every > 0 and epoch % save_every == 0:
                ckpt_path = self.checkpoint_dir / f"{model_name}_epoch{epoch}.pt"
                torch.save(self.model.state_dict(), ckpt_path)

        # Load best model if we have validation
        if val_loader is not None:
            best_path = self.checkpoint_dir / f"{model_name}_best.pt"
            if best_path.exists():
                self.model.load_state_dict(
                    torch.load(best_path, map_location=self.device, weights_only=True)
                )
                log.info(f"  Loaded best model (val_loss={best_val_loss:.4f})")

        return history
