"""
OceanMaster v13.2 — GP / Ensemble Uncertainty Estimator [C6]
==============================================================
Provide confidence intervals for each hotspot HSI prediction.
Ref: Rasmussen & Williams (2006) "Gaussian Processes for ML"
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any

log = logging.getLogger("OceanMaster.Uncertainty")


class UncertaintyEstimator:
    """Estimate prediction uncertainty via ensemble variance or GP regression."""

    def __init__(self, method: str = "ensemble"):
        self.method = method

    def estimate(
        self,
        hotspots: List[Dict[str, Any]],
        feature_keys: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Add hsi_std, hsi_lower, hsi_upper, confidence_level to each hotspot."""
        if not hotspots:
            return hotspots

        keys = feature_keys or ["sst", "chl", "ssh", "eke", "npp", "do_ml_per_l", "depth", "slope"]
        X = np.array([[float(h.get(k, 0)) for k in keys] for h in hotspots], dtype=np.float64)
        y = np.array([float(h.get("score", 0.5)) for h in hotspots], dtype=np.float64)

        if len(X) < 3:
            for h in hotspots:
                s = h.get("score", 0.5)
                h.update(hsi_std=0.15, hsi_lower=max(0, s - 0.15), hsi_upper=min(1, s + 0.15),
                         confidence_level="low")
            return hotspots

        stds = (self._gp_uncertainty(X, y) if self.method == "gp" and len(X) <= 200
                else self._ensemble_uncertainty(X, y))

        for i, h in enumerate(hotspots):
            sigma = float(stds[i])
            s = float(h.get("score", 0.5))
            h["hsi_std"] = round(sigma, 4)
            h["hsi_lower"] = round(max(0, s - 1.96 * sigma), 4)
            h["hsi_upper"] = round(min(1, s + 1.96 * sigma), 4)
            h["confidence_level"] = "high" if sigma < 0.08 else "medium" if sigma < 0.15 else "low"

        log.info(f"  Uncertainty: mean σ={np.mean(stds):.4f}")
        return hotspots

    def _gp_uncertainty(self, X, y):
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel
            from sklearn.preprocessing import StandardScaler
            X_s = StandardScaler().fit_transform(X)
            gp = GaussianProcessRegressor(kernel=RBF() + WhiteKernel(0.01), n_restarts_optimizer=2)
            gp.fit(X_s, y)
            _, stds = gp.predict(X_s, return_std=True)
            return stds
        except Exception as e:
            log.warning(f"  GP failed ({e}), falling back to ensemble")
            return self._ensemble_uncertainty(X, y)

    def _ensemble_uncertainty(self, X, y):
        rng = np.random.default_rng(42)
        preds = np.zeros((len(X), 50))
        for t in range(50):
            noise = rng.normal(0, 0.05, X.shape)
            X_n = X + noise
            c = X_n - np.mean(X_n, axis=0, keepdims=True)
            w = np.linalg.lstsq(c, y - np.mean(y), rcond=None)[0]
            preds[:, t] = c @ w + np.mean(y)
        return np.clip(np.std(preds, axis=1), 0.02, 0.3)
