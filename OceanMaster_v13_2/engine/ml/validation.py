"""
OceanMaster v13.2 — Phase 9: Prediction Validation Framework  # [v12-phase9]
============================================================================
Validate ML predictions against known data with comprehensive metrics.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

log = logging.getLogger("OceanMaster.Validation")


class PredictionValidator:
    """
    Validate predictions against known data.

    Metrics:
      - R², RMSE, MAE
      - Accuracy within ±20% (fisheries standard)
      - Spatial bias analysis
      - SHAP feature importance
    """

    def __init__(self):
        self.results: Dict[str, Dict] = {}

    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        species: str = "unknown",
    ) -> Dict:
        """Compute comprehensive regression metrics."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        # Filter valid
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_t = y_true[mask]
        y_p = y_pred[mask]
        n = len(y_t)

        if n < 5:
            return {"error": "insufficient_data", "n": n}

        # Core metrics
        residuals = y_t - y_p
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        r2 = 1 - (ss_res / max(ss_tot, 1e-10))
        rmse = np.sqrt(np.mean(residuals ** 2))
        mae = np.mean(np.abs(residuals))

        # Accuracy within ±20% (fisheries standard)
        y_t_safe = np.where(y_t > 0.01, y_t, 0.01)
        relative_error = np.abs(residuals) / y_t_safe
        acc_20 = float(np.mean(relative_error < 0.20))
        acc_30 = float(np.mean(relative_error < 0.30))

        # Bias
        mean_bias = float(np.mean(residuals))
        median_bias = float(np.median(residuals))

        # Percentiles
        p10, p50, p90 = np.percentile(y_p, [10, 50, 90])

        return {
            "species": species,
            "n_samples": n,
            "r2": float(r2),
            "rmse": float(rmse),
            "mae": float(mae),
            "accuracy_20pct": acc_20,
            "accuracy_30pct": acc_30,
            "mean_bias": mean_bias,
            "median_bias": median_bias,
            "pred_p10": float(p10),
            "pred_p50": float(p50),
            "pred_p90": float(p90),
            "y_true_mean": float(np.mean(y_t)),
            "y_pred_mean": float(np.mean(y_p)),
        }

    def backtest(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        species: str = "unknown",
        n_folds: int = 5,
        lats: Optional[np.ndarray] = None,
        lons: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Cross-validation backtest.
        Uses SpatialBlockCV if lat/lon provided, else KFold fallback.
        """
        all_y_true = []
        all_y_pred = []
        fold_metrics = []

        if lats is not None and lons is not None:
            from engine.ml.stacking_ensemble import SpatialBlockCV
            spatial_cv = SpatialBlockCV(n_blocks=5, buffer_km=50)
            fold_iter = enumerate(spatial_cv.split(X, lats, lons))
            log.info(f"  Using SpatialBlockCV for backtest ({species})")
        else:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            fold_iter = enumerate(kf.split(X))
            log.warning(f"  No lat/lon — using random KFold for backtest ({species})")

        for fold, (train_idx, test_idx) in fold_iter:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Clone model for each fold
            from sklearn.base import clone

            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            y_pred = fold_model.predict(X_test)

            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

            fm = self.compute_metrics(y_test, y_pred, species)
            fm["fold"] = fold
            fold_metrics.append(fm)

        overall = self.compute_metrics(
            np.array(all_y_true), np.array(all_y_pred), species
        )
        overall["fold_metrics"] = fold_metrics
        overall["n_folds"] = len(fold_metrics)

        self.results[species] = overall
        return overall

    def spatial_validation(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        n_bins: int = 5,
    ) -> Dict:
        """Per-region accuracy analysis."""
        lat_bins = np.linspace(lats.min(), lats.max(), n_bins + 1)
        lon_bins = np.linspace(lons.min(), lons.max(), n_bins + 1)

        spatial_results = []
        for i in range(n_bins):
            for j in range(n_bins):
                mask = (
                    (lats >= lat_bins[i]) & (lats < lat_bins[i + 1])
                    & (lons >= lon_bins[j]) & (lons < lon_bins[j + 1])
                )
                if mask.sum() < 5:
                    continue
                m = self.compute_metrics(y_true[mask], y_pred[mask])
                m["lat_range"] = (float(lat_bins[i]), float(lat_bins[i + 1]))
                m["lon_range"] = (float(lon_bins[j]), float(lon_bins[j + 1]))
                spatial_results.append(m)

        return {"spatial_cells": spatial_results, "n_bins": n_bins}

    def temporal_validation(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        months: np.ndarray,
    ) -> Dict:
        """Per-month accuracy analysis."""
        monthly = []
        for m in range(1, 13):
            mask = months == m
            if mask.sum() < 5:
                continue
            metrics = self.compute_metrics(y_true[mask], y_pred[mask])
            metrics["month"] = m
            monthly.append(metrics)

        return {"monthly_metrics": monthly}

    # ── [逆向工程成果] 漁場預測專用嚴格指標 ──────────────────

    @staticmethod
    def hotspot_iou(
        predicted: np.ndarray,
        observed: np.ndarray,
        percentile: float = 90.0,
    ) -> Dict:
        """
        Hotspot IoU — 高 CPUE 區域的 Intersection over Union.

        為什麼需要這個指標 (逆向工程發現):
          CATCH 報告 SSI=0.955，但 80% 得分來自「正確預測空白海域」。
          SSIM 對空間稀疏資料（大面積 CPUE≈0）偏寬鬆。
          Hotspot IoU 只看高值區域，直接衡量「有沒有找對漁場」。

        Args:
            predicted: 2D array (H, W), 預測 CPUE 或 HSI
            observed: 2D array (H, W), 觀測 CPUE
            percentile: 閾值百分位 (default 90 = 只看前10%高值區域)

        Returns:
            {
                "hotspot_iou": float (0-1, 1=完美重疊),
                "hotspot_precision": float (預測熱點中多少是真熱點),
                "hotspot_recall": float (真熱點中多少被預測到),
                "threshold_pred": float,
                "threshold_obs": float,
                "n_hotspot_pred": int,
                "n_hotspot_obs": int,
            }
        """
        p = np.nan_to_num(predicted, nan=0.0)
        o = np.nan_to_num(observed, nan=0.0)

        # 閾值：各自的 P90
        thresh_p = np.percentile(p[p > 0], percentile) if np.any(p > 0) else 0
        thresh_o = np.percentile(o[o > 0], percentile) if np.any(o > 0) else 0

        mask_p = p >= thresh_p if thresh_p > 0 else p > 0
        mask_o = o >= thresh_o if thresh_o > 0 else o > 0

        intersection = np.sum(mask_p & mask_o)
        union = np.sum(mask_p | mask_o)

        iou = float(intersection / max(union, 1))
        precision = float(intersection / max(np.sum(mask_p), 1))
        recall = float(intersection / max(np.sum(mask_o), 1))

        log.info(f"  Hotspot IoU (P{percentile:.0f}): {iou:.3f} "
                 f"(prec={precision:.3f}, rec={recall:.3f})")

        return {
            "hotspot_iou": iou,
            "hotspot_precision": precision,
            "hotspot_recall": recall,
            "threshold_pred": float(thresh_p),
            "threshold_obs": float(thresh_o),
            "n_hotspot_pred": int(np.sum(mask_p)),
            "n_hotspot_obs": int(np.sum(mask_o)),
        }

    @staticmethod
    def centroid_distance_km(
        predicted: np.ndarray,
        observed: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> Dict:
        """
        質心偏移距離 — 預測漁場中心 vs 觀測中心的 km 偏差.

        為什麼需要 (逆向工程發現):
          四間競爭對手都不報告這個指標。漁民最關心的是
          「AI 說去那裡，到底偏了多少公里」。這比 R² 更直觀。

        Args:
            predicted: 2D array (H, W)
            observed: 2D array (H, W)
            lats: 1D latitude array (length H)
            lons: 1D longitude array (length W)

        Returns:
            {
                "distance_km": float,
                "pred_centroid": (lat, lon),
                "obs_centroid": (lat, lon),
            }
        """
        p = np.nan_to_num(predicted, nan=0.0)
        o = np.nan_to_num(observed, nan=0.0)

        # 權重質心
        def _weighted_centroid(field, lat_arr, lon_arr):
            w = np.maximum(field, 0)
            total = w.sum()
            if total < 1e-10:
                return np.mean(lat_arr), np.mean(lon_arr)
            # 建立 2D 經緯度網格
            lat_grid, lon_grid = np.meshgrid(lat_arr, lon_arr, indexing="ij")
            c_lat = float(np.sum(lat_grid * w) / total)
            c_lon = float(np.sum(lon_grid * w) / total)
            return c_lat, c_lon

        pred_c = _weighted_centroid(p, lats, lons)
        obs_c = _weighted_centroid(o, lats, lons)

        # Haversine distance
        def _haversine(lat1, lon1, lat2, lon2):
            R = 6371.0  # km
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = (np.sin(dlat / 2) ** 2
                 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
                 * np.sin(dlon / 2) ** 2)
            return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        dist = _haversine(pred_c[0], pred_c[1], obs_c[0], obs_c[1])

        log.info(f"  Centroid distance: {dist:.1f} km "
                 f"(pred={pred_c[0]:.2f}°N,{pred_c[1]:.2f}°E → "
                 f"obs={obs_c[0]:.2f}°N,{obs_c[1]:.2f}°E)")

        return {
            "distance_km": float(dist),
            "pred_centroid": pred_c,
            "obs_centroid": obs_c,
        }

    def feature_importance_report(
        self,
        model,
        feature_names: List[str],
        top_n: int = 15,
    ) -> List[Dict]:
        """Extract and rank feature importance."""
        importances = None

        # Try different model types
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "estimators_"):
            # StackingRegressor — estimators_ are fitted instances
            for est in model.estimators_:
                if hasattr(est, "feature_importances_"):
                    importances = est.feature_importances_
                    break

        if importances is None:
            return []

        ranked = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1], reverse=True,
        )[:top_n]

        return [{"name": n, "importance": float(v)} for n, v in ranked]

    @staticmethod
    def generate_report(
        all_metrics: Dict[str, Dict],
        feature_importance: Dict[str, List] = None,
        output_path: str = "models/training_report_v12.md",
    ) -> str:
        """Generate markdown training report."""
        lines = [
            "# OceanMaster v13.2 — ML Training Report",
            "",
            f"> Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "---",
            "",
            "## Summary",
            "",
            "| Species | R² | RMSE | MAE | Acc ±20% | Acc ±30% | N Samples |",
            "|---------|-----|------|-----|----------|----------|-----------|",
        ]

        for sp, m in all_metrics.items():
            if "error" in m:
                continue
            lines.append(
                f"| {sp} | {m['r2']:.4f} | {m['rmse']:.4f} | {m['mae']:.4f} "
                f"| {m['accuracy_20pct']:.1%} | {m['accuracy_30pct']:.1%} "
                f"| {m['n_samples']} |"
            )

        lines.extend(["", "---", ""])

        # Per-species detail
        for sp, m in all_metrics.items():
            if "error" in m:
                continue
            lines.extend([
                f"## {sp.capitalize()}",
                "",
                f"- **R²**: {m['r2']:.4f}",
                f"- **RMSE**: {m['rmse']:.4f}",
                f"- **MAE**: {m['mae']:.4f}",
                f"- **Accuracy (±20%)**: {m['accuracy_20pct']:.1%}",
                f"- **Accuracy (±30%)**: {m['accuracy_30pct']:.1%}",
                f"- **Mean Bias**: {m['mean_bias']:.4f}",
                f"- **Prediction range**: P10={m['pred_p10']:.3f}, P50={m['pred_p50']:.3f}, P90={m['pred_p90']:.3f}",
                "",
            ])

            # Feature importance
            if feature_importance and sp in feature_importance:
                fi = feature_importance[sp]
                if fi:
                    lines.extend([
                        "### Top Features",
                        "",
                        "| Rank | Feature | Importance |",
                        "|------|---------|------------|",
                    ])
                    for i, f in enumerate(fi[:10], 1):
                        lines.append(f"| {i} | {f['name']} | {f['importance']:.4f} |")
                    lines.append("")

            lines.extend(["---", ""])

        # Data source note
        lines.extend([
            "## Data Source",
            "",
            "> **Synthetic data** — trained on scientifically realistic generated data.",
            "> Swap for real FAO/RFMO CPUE data for production accuracy.",
            "> Expected R² with real data: 0.40-0.55 (higher noise, more realistic).",
            "",
        ])

        report = "\n".join(lines)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        log.info(f"Report saved: {output_path}")

        return report
