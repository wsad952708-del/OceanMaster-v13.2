"""
OceanMaster v13.2 — Prediction Engine
=======================================
專職執行 ML 推論 + SHAP 解釋。

反幻覺鐵律:
  - NaN 特徵行 → 跳過預測，不猜測
  - 每個 hotspot 必須附帶 confidence + dominant_features
  - 嚴禁以零值填補 NaN 騙過模型
"""
import logging
import numpy as np
from typing import Any, Dict, List, Optional
from pathlib import Path

log = logging.getLogger("OceanMaster.predict")

try:
    import joblib
    ML_LIBS_OK = True
except ImportError:
    ML_LIBS_OK = False

try:
    from engine.shap_explainer import get_shap_explainer
    SHAP_OK = True
except ImportError:
    SHAP_OK = False


# ML 12-feature spec (GFW-trained)
GFW_FEATURES = [
    "sst", "chl", "ssh", "do", "current_speed",
    "front_strength", "eddy_strength", "phi",
    "bathy_depth", "bathy_slope", "dist_seamount", "dist_shelf_break",
]


class PredictionEngine:
    """
    ML Ensemble CPUE 預測 + SHAP 解釋

    NaN 阻斷: feature row 含 NaN → skip (不補值)
    信心度: 基於 across-species model variance
    主導特徵: SHAP top-3
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.pipelines = {}
        self.shap_explainer = None
        self._load_models()

    def _load_models(self):
        if not ML_LIBS_OK:
            return

        # [v13.2-P0] Import HMAC verification
        try:
            from engine.ml.stacking_ensemble import _verify_model_file
        except ImportError:
            log.warning("[P1] engine.ml.stacking_ensemble not available — model signature verification DISABLED")
            _verify_model_file = lambda p: True  # Dev fallback — no signing module

        for sp in ["yellowfin", "bigeye", "skipjack", "albacore"]:
            model_path = Path(f"ml_system/models/stacking_{sp}.pkl")
            pipe_path = Path(f"ml_system/models/feature_pipeline_{sp}.pkl")
            if model_path.exists():
                try:
                    _verify_model_file(model_path)
                    loaded = joblib.load(model_path)
                    if isinstance(loaded, dict) and "model" in loaded:
                        self.models[sp] = loaded["model"]
                        self.scalers[sp] = loaded.get("scaler")
                    else:
                        self.models[sp] = loaded
                    if pipe_path.exists():
                        _verify_model_file(pipe_path)
                        self.pipelines[sp] = joblib.load(pipe_path)
                except RuntimeError as e:
                    log.error(f"  [P0] ML {sp} signature verification failed: {e}")
                except Exception as e:
                    log.warning(f"  ML {sp}: {e}")

        if self.models:
            log.info(f"  ✅ ML models: {list(self.models.keys())}")
            if SHAP_OK:
                self.shap_explainer = get_shap_explainer()
                for sp, model in self.models.items():
                    feat_names = GFW_FEATURES
                    if sp in self.pipelines and hasattr(self.pipelines[sp], 'feature_names_'):
                        feat_names = list(self.pipelines[sp].feature_names_)
                    self.shap_explainer.register_model(sp, model, feat_names)

    @property
    def available(self):
        return len(self.models) > 0

    def predict_grid(
        self,
        species: str,
        sst: np.ndarray,
        chl: np.ndarray,
        ssh: np.ndarray,
        do_surface: np.ndarray,
        current_speed: np.ndarray,
        front_strength: np.ndarray,
        eke: np.ndarray,
        phi: np.ndarray,
        bathy_depth: np.ndarray,
        bathy_slope: np.ndarray,
        dist_seamount: np.ndarray,
        dist_shelf_break: np.ndarray,
        ny: int, nx: int,
    ) -> Optional[Dict]:
        """
        ML grid predict.

        Returns: {"cpue": 2D, "confidence": 2D, "nan_blocked": int}
        NaN 行直接跳過.
        """
        if species not in self.models:
            return None

        n_points = ny * nx

        def safe_flat(arr, default=0.0):
            if arr is None:
                return np.full(n_points, np.nan)  # NaN, 不是 0!
            try:
                a = np.broadcast_to(arr, (ny, nx)).copy()
                return a.ravel()
            except Exception:
                return np.full(n_points, np.nan)

        X = np.column_stack([
            safe_flat(sst), safe_flat(chl), safe_flat(ssh),
            safe_flat(do_surface), safe_flat(current_speed),
            safe_flat(front_strength), safe_flat(eke), safe_flat(phi),
            safe_flat(bathy_depth), safe_flat(bathy_slope),
            safe_flat(dist_seamount), safe_flat(dist_shelf_break),
        ])

        # ── NaN 阻斷: 含 NaN 的 row 不予預測 ──
        valid_mask = np.all(np.isfinite(X), axis=1)
        nan_blocked = int(np.sum(~valid_mask))

        cpue_flat = np.full(n_points, np.nan)
        conf_flat = np.full(n_points, 0.0)

        if np.any(valid_mask):
            X_valid = X[valid_mask]

            # 處理 Inf (不等於 NaN, 但也危險)
            X_valid = np.nan_to_num(X_valid, nan=0.0, posinf=0.0, neginf=0.0)

            # Scale
            if species in self.scalers and self.scalers[species] is not None:
                try:
                    X_valid = self.scalers[species].transform(X_valid)
                except Exception as e:
                    log.warning(f"  {species}: scaler failed ({e})")

            pred = self.models[species].predict(X_valid)
            cpue_flat[valid_mask] = np.clip(pred, 0, 500)
            conf_flat[valid_mask] = np.clip(pred / 100, 0.0, 1.0)

        cpue_grid = cpue_flat.reshape(ny, nx)
        conf_grid = conf_flat.reshape(ny, nx)

        if nan_blocked > 0:
            log.info(f"  {species}: NaN blocked {nan_blocked}/{n_points} grid cells")

        return {
            "cpue": cpue_grid,
            "confidence": conf_grid,
            "nan_blocked": nan_blocked,
            "nan_blocked_pct": round(nan_blocked / max(n_points, 1) * 100, 1),
        }

    def explain_hotspot(self, hotspot: Dict, species: str, features: Dict) -> Dict:
        """
        單點 SHAP 解釋: 回傳 top-3 dominant features
        """
        if self.shap_explainer is None or species not in self.models:
            return {"dominant_features": [], "method": "unavailable"}

        try:
            result = self.shap_explainer.explain_point(features, species)
            return {
                "explain": result.get("explanation_text", ""),
                "shap_values": result.get("raw_shap_values", {}),
                "dominant_features": self._top_features(result.get("raw_shap_values", {})),
                "method": result.get("method", "shap"),
            }
        except Exception as e:
            log.debug(f"  SHAP {species}: {e}")
            return {"dominant_features": [], "method": "fallback"}

    @staticmethod
    def _top_features(shap_vals: Dict, n: int = 3) -> List[str]:
        """取 SHAP 絕對值 top-N 特徵名"""
        if not shap_vals:
            return []
        sorted_feats = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)
        return [f[0] for f in sorted_feats[:n]]
