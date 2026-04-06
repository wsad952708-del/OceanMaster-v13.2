"""
OceanMaster v13.2 — 真實 SHAP 可解釋性模組
============================================
用 shap.TreeExplainer 對 ML base learners (RF/LightGBM)
計算真正的 Shapley values。

如果 shap 沒有安裝，graceful fallback 到現有的
ExplainableHSI 靜態權重方法。
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any

log = logging.getLogger("OceanMaster.SHAP")

# ─── 中文因子名稱對照表 ─────────────────────────
FEATURE_NAMES_ZH = {
    "sst":              "🌡️海表溫度",
    "chl":              "🌿葉綠素",
    "ftle":             "🌀FTLE聚集線",
    "phi":              "🫧代謝指數",
    "front_strength":   "🔥溫度鋒面",
    "npp":              "🌱初級生產力",
    "do_surface":       "💨溶解氧",
    "current_speed":    "🌊海流速度",
    "eke":              "⚡渦動能",
    "prey_field_index": "🍤餌料場",
    "ssh":              "📐海面高度",
    "sst_gradient":     "🌡️溫度梯度",
    "chl_gradient":     "🌿葉綠素梯度",
    "sst_anomaly":      "📊SST異常",
    "lat":              "📍緯度",
    "lon":              "📍經度",
    "month":            "📅月份",
    "day_of_year":      "📅年日數",
    "hsi":              "🎯HSI評分",
    "eddy_strength":    "🌀渦旋強度",
    "salinity":         "🧂鹽度",
    "depth":            "⛰️水深",
    "bathy_roughness":  "🏔️地形崎嶇度",
    "wind_speed":       "💨風速",
    "lunar_factor":     "🌙月相",
    "oni":              "🌍ENSO指數",
    "h_thermal":        "🌡️溫度適宜度",
    "phi_viability":    "🫧代謝可行性",
    "h_feeding":        "🍤餌料可及性",
    "front_persistence":"🌊穩定鋒面",
    "eddy_edge":        "🌀渦旋邊緣",
}


def _get_zh_name(feature: str) -> str:
    """取得特徵的中文名稱"""
    # 嘗試完全匹配
    if feature in FEATURE_NAMES_ZH:
        return FEATURE_NAMES_ZH[feature]
    # 嘗試部分匹配
    for key, name in FEATURE_NAMES_ZH.items():
        if key in feature:
            return name
    return feature


class SHAPExplainer:
    """
    真正的 SHAP 解釋器。

    使用 shap.TreeExplainer 計算 Shapley values。
    如果 shap 不可用，退回到靜態權重方法。
    """

    def __init__(self):
        self._shap_available = False
        self._explainers: Dict[str, Any] = {}  # species → TreeExplainer
        self._feature_names: Dict[str, List[str]] = {}
        self._fallback_mode = True

        try:
            import shap
            self._shap_available = True
            log.info("  ✅ SHAP library available — TreeExplainer enabled")
        except ImportError:
            log.info("  ⚠️ SHAP library not installed — using weight-based fallback")

    def register_model(self, species: str, model: Any, feature_names: List[str] = None):
        """
        註冊一個已訓練的 ML 模型用於 SHAP 分析。

        Args:
            species: 魚種名稱
            model: 已訓練的模型 (需要有 .predict 方法)
            feature_names: 特徵名稱列表
        """
        if not self._shap_available:
            return

        import shap

        try:
            # 嘗試取得 base estimator（如果是 Stacking 模型）
            base_model = self._extract_tree_model(model)

            if base_model is not None:
                self._explainers[species] = shap.TreeExplainer(base_model)
                self._feature_names[species] = feature_names or []
                self._fallback_mode = False
                log.info(f"  SHAP: registered TreeExplainer for {species} "
                         f"({type(base_model).__name__}, "
                         f"{len(feature_names or [])} features)")
            else:
                log.info(f"  SHAP: no tree-based model found for {species}, "
                         f"using KernelExplainer fallback")
                # KernelExplainer is slow but works with any model
                # We'll only use it if explicitly needed
                self._feature_names[species] = feature_names or []

        except Exception as e:
            log.warning(f"  SHAP: failed to register {species}: {e}")

    @staticmethod
    def _extract_tree_model(model) -> Optional[Any]:
        """
        從 Stacking 或 Pipeline 中提取 tree-based base model。

        支援:
        - RandomForestRegressor / Classifier
        - LightGBM
        - XGBoost
        - StackingRegressor (取第一個 tree-based estimator)
        - Pipeline (取最後一步)
        """
        # 直接是 tree-based model
        model_type = type(model).__name__
        tree_types = (
            "RandomForestRegressor", "RandomForestClassifier",
            "GradientBoostingRegressor", "GradientBoostingClassifier",
            "ExtraTreesRegressor", "ExtraTreesClassifier",
            "LGBMRegressor", "LGBMClassifier",
            "XGBRegressor", "XGBClassifier",
            "DecisionTreeRegressor", "DecisionTreeClassifier",
        )

        if model_type in tree_types:
            return model

        # Pipeline: unwrap the last step
        if hasattr(model, 'steps'):
            return SHAPExplainer._extract_tree_model(model.steps[-1][1])

        # Stacking: get the first tree-based estimator
        if hasattr(model, 'estimators_'):
            for est in model.estimators_:
                result = SHAPExplainer._extract_tree_model(est)
                if result is not None:
                    return result

        # StackingRegressor with named_estimators_
        if hasattr(model, 'named_estimators_'):
            for name, est in model.named_estimators_.items():
                result = SHAPExplainer._extract_tree_model(est)
                if result is not None:
                    log.info(f"  SHAP: using '{name}' from stacking as tree model")
                    return result

        # Custom model with .base_models or .models
        for attr in ('base_models', 'models', 'best_model', 'rf_model',
                      'lgbm_model', 'xgb_model', 'model'):
            if hasattr(model, attr):
                obj = getattr(model, attr)
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        result = SHAPExplainer._extract_tree_model(v)
                        if result is not None:
                            log.info(f"  SHAP: extracted '{k}' as tree model")
                            return result
                elif isinstance(obj, list):
                    for v in obj:
                        result = SHAPExplainer._extract_tree_model(v)
                        if result is not None:
                            return result
                else:
                    result = SHAPExplainer._extract_tree_model(obj)
                    if result is not None:
                        return result

        return None

    def explain_point(
        self,
        features_dict: Dict[str, float],
        species: str = "yellowfin",
        feature_names: List[str] = None,
    ) -> Dict[str, Any]:
        """
        對單一格點計算 SHAP 值。

        Args:
            features_dict: {feature_name: value} 格式
            species: 魚種
            feature_names: 如果跟 register 時不同

        Returns:
            {
                "method": "shap" | "weight_based",
                "top_factors": [
                    {"feature": "sst", "name": "🌡️海表溫度",
                     "value": 28.5, "shap_value": 0.15,
                     "direction": "positive", "importance_pct": 35.2},
                    ...
                ],
                "explanation_text": "此漁場 SST 28.5°C 最有利...",
                "raw_shap_values": {feature: shap_value, ...}
            }
        """
        # 嘗試 TreeExplainer
        if species in self._explainers:
            try:
                return self._explain_with_shap(features_dict, species, feature_names)
            except Exception as e:
                log.debug(f"  SHAP explain failed: {e}, using fallback")

        # Fallback: 靜態權重方法
        return self._explain_with_weights(features_dict, species)

    def _explain_with_shap(
        self, features_dict: Dict[str, float], species: str,
        feature_names: List[str] = None,
    ) -> Dict[str, Any]:
        """用真正的 TreeExplainer 計算 SHAP"""
        explainer = self._explainers[species]
        names = feature_names or self._feature_names.get(species, [])

        # 建構特徵向量（確保順序與訓練時一致）
        if names:
            x = np.array([features_dict.get(n, 0.0) for n in names]).reshape(1, -1)
        else:
            names = list(features_dict.keys())
            x = np.array(list(features_dict.values())).reshape(1, -1)

        shap_values = explainer.shap_values(x)[0]  # shape: (n_features,)

        # 組合結果
        raw_shap = {}
        factors = []
        total_abs = sum(abs(sv) for sv in shap_values)

        for i, (name, sv) in enumerate(zip(names, shap_values)):
            raw_shap[name] = float(sv)
            if abs(sv) > 1e-6:  # 過濾極小值
                factors.append({
                    "feature": name,
                    "name": _get_zh_name(name),
                    "value": features_dict.get(name, 0.0),
                    "shap_value": float(sv),
                    "direction": "positive" if sv > 0 else "negative",
                    "importance_pct": abs(sv) / max(total_abs, 1e-6) * 100,
                })

        # 按 |SHAP| 排序
        factors.sort(key=lambda x: -abs(x["shap_value"]))
        top_factors = factors[:6]

        # 生成解釋文字
        text = self._make_explanation_text(top_factors, species)

        return {
            "method": "shap",
            "top_factors": top_factors,
            "explanation_text": text,
            "raw_shap_values": raw_shap,
        }

    def _explain_with_weights(
        self, features_dict: Dict[str, float], species: str,
    ) -> Dict[str, Any]:
        """靜態權重 fallback（類似現有 ExplainableHSI）"""
        # 權重表
        WEIGHTS = {
            "sst": 0.18, "chl": 0.12, "phi": 0.15,
            "front_strength": 0.10, "ftle": 0.10,
            "npp": 0.08, "do_surface": 0.07,
            "current_speed": 0.05, "eke": 0.05,
            "prey_field_index": 0.04,
            "h_thermal": 0.20, "phi_viability": 0.20,
            "h_feeding": 0.25, "front_persistence": 0.05,
            "eddy_edge": 0.05, "ssh": 0.03,
        }

        # [v13.2-R6] 特徵物理範圍 — 用於歸一化，防止大單位特徵壓爆小單位
        # 格式: (min, max) — 歸一化後所有特徵值落在 0-1
        FEATURE_RANGES = {
            "sst": (15.0, 32.0),         # °C
            "chl": (0.01, 5.0),          # mg/m³
            "phi": (0.0, 5.0),           # metabolic index
            "front_strength": (0.0, 1.0),
            "ftle": (0.0, 1.0),
            "npp": (0.0, 2000.0),        # mgC/m²/day
            "do_surface": (2.0, 8.0),    # mL/L
            "current_speed": (0.0, 2.0), # m/s
            "eke": (0.0, 0.5),           # m²/s²
            "prey_field_index": (0.0, 1.0),
            "h_thermal": (0.0, 1.0),
            "phi_viability": (0.0, 1.0),
            "h_feeding": (0.0, 1.0),
            "front_persistence": (0.0, 1.0),
            "eddy_edge": (0.0, 1.0),
            "ssh": (-1.0, 1.0),          # m
            "sst_gradient": (0.0, 2.0),  # °C/deg
            "chl_gradient": (0.0, 1.0),
            "sst_anomaly": (-5.0, 5.0),  # °C
            "salinity": (30.0, 37.0),    # PSU
            "depth": (0.0, 5000.0),      # m
            "bathy_roughness": (0.0, 500.0),  # m (TRI)
            "wind_speed": (0.0, 25.0),   # m/s
            "lunar_factor": (0.0, 1.0),
            "oni": (-3.0, 3.0),          # ENSO index
        }

        factors = []
        total = 0.0

        for key, val in features_dict.items():
            w = WEIGHTS.get(key, 0.02)
            if val is None or not np.isfinite(val):
                continue
            # [v13.2-R6] 歸一化到 0-1 再乘權重，避免單位差異壓爆
            lo, hi = FEATURE_RANGES.get(key, (0.0, 1.0))
            normalized = (float(val) - lo) / max(hi - lo, 1e-6)
            normalized = max(0.0, min(1.0, normalized))  # clamp
            contrib = normalized * w
            total += abs(contrib)
            factors.append({
                "feature": key,
                "name": _get_zh_name(key),
                "value": float(val),
                "shap_value": contrib,  # pseudo-SHAP (normalized)
                "direction": "positive" if contrib > 0 else "negative",
                "importance_pct": 0,  # 後面填
            })

        # 正規化
        for f in factors:
            f["importance_pct"] = abs(f["shap_value"]) / max(total, 1e-6) * 100

        factors.sort(key=lambda x: -abs(x["shap_value"]))
        top_factors = factors[:6]

        text = self._make_explanation_text(top_factors, species)

        return {
            "method": "weight_based",
            "top_factors": top_factors,
            "explanation_text": text,
            "raw_shap_values": {f["feature"]: f["shap_value"] for f in factors},
        }

    @staticmethod
    def _make_explanation_text(
        top_factors: List[Dict], species: str, top_n: int = 4,
    ) -> str:
        """生成漁民友好的中文解釋"""
        SPECIES_ZH = {
            "yellowfin": "黃鰭鮪", "bigeye": "大目鮪",
            "skipjack": "正鰹", "albacore": "長鰭鮪",
        }
        sp_zh = SPECIES_ZH.get(species, species)

        lines = [f"【{sp_zh}】預測依據："]
        for i, f in enumerate(top_factors[:top_n]):
            arrow = "↑" if f["direction"] == "positive" else "↓"
            pct = f["importance_pct"]
            lines.append(
                f"  {i+1}. {f['name']} {arrow} "
                f"(貢獻 {pct:.0f}%)"
            )

        return "\n".join(lines)


# ─── 全域 singleton ──────────────────────────────
_global_explainer: Optional[SHAPExplainer] = None


def get_shap_explainer() -> SHAPExplainer:
    """取得全域 SHAP 解釋器實例"""
    global _global_explainer
    if _global_explainer is None:
        _global_explainer = SHAPExplainer()
    return _global_explainer
