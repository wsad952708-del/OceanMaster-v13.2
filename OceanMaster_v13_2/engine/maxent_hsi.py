"""
OceanMaster v13.2 — MaxEnt HSI (棲地適宜性指數)
================================================
[v13.2-GH3] 數據驅動的棲地適宜性預測

來源: 受 earth-chris/elapid (github.com) 啟發
方法: MaxEnt 核心數學 (feature transforms + L1 logistic regression)
依賴: 僅 numpy + sklearn (無新依賴)

與現有 hsi_models.py 的關係:
  hsi_models.py = 物理公式法 (基於文獻的 SST/CHL 偏好函數)
  maxent_hsi.py = 數據驅動法 (從漁獲出現點自動學習偏好)
  兩者互補 → 可在 stacking_ensemble 中同時使用
"""

import numpy as np
import logging
from typing import Optional, Dict, List, Tuple

log = logging.getLogger("OceanMaster.MaxentHSI")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    log.warning("sklearn not available, MaxentHSI will use numpy fallback")


class MaxentFeatureTransformer:
    """
    MaxEnt 特徵轉換器

    MaxEnt 的核心不是模型本身（就是 logistic regression），
    而是它的 **feature transforms** — 把原始環境變數轉換為:
      - Linear:    x
      - Quadratic: x²
      - Hinge:     max(0, x - threshold)  (分段線性)
      - Threshold: 1 if x > t else 0      (階梯函數)

    這些轉換讓線性模型也能捕捉非線性棲地偏好。

    Source: Phillips et al. 2006, 2017 (MaxEnt 經典論文)
    Implementation inspired by: elapid.MaxentFeatureTransformer
    """

    def __init__(self, n_hinge: int = 10, n_threshold: int = 10):
        """
        Args:
            n_hinge: 每個特徵的 hinge knot 數量
            n_threshold: 每個特徵的 threshold 數量
        """
        self.n_hinge = n_hinge
        self.n_threshold = n_threshold
        self._knots = {}
        self._feature_names = []
        self._is_fitted = False

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        計算 hinge/threshold 的 knot 位置 (基於訓練資料分位數)

        Args:
            X: (n_samples, n_features) 環境變數矩陣
            feature_names: 特徵名稱列表
        """
        n_features = X.shape[1]
        self._feature_names = feature_names or [f"f{i}" for i in range(n_features)]
        self._knots = {}

        for i in range(n_features):
            col = X[:, i]
            valid = col[~np.isnan(col)]
            if len(valid) < 10:
                self._knots[i] = {"hinge": [], "threshold": []}
                continue

            # Hinge knots: 均勻分布在 5% ~ 95% 分位數
            h_quantiles = np.linspace(0.05, 0.95, self.n_hinge)
            h_knots = np.percentile(valid, h_quantiles * 100)

            # Threshold knots: 同上
            t_quantiles = np.linspace(0.05, 0.95, self.n_threshold)
            t_knots = np.percentile(valid, t_quantiles * 100)

            self._knots[i] = {
                "hinge": h_knots.tolist(),
                "threshold": t_knots.tolist(),
            }

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        轉換原始特徵 → MaxEnt 衍生特徵

        Returns:
            (n_samples, n_derived_features) 衍生特徵矩陣
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before transform()")

        n_samples, n_features = X.shape
        features = []

        for i in range(n_features):
            col = X[:, i]

            # Linear
            features.append(col)

            # Quadratic
            features.append(col ** 2)

            # Hinge features: max(0, x - knot)
            for knot in self._knots.get(i, {}).get("hinge", []):
                features.append(np.maximum(0, col - knot))

            # Threshold features: 1 if x > knot else 0
            for knot in self._knots.get(i, {}).get("threshold", []):
                features.append((col > knot).astype(np.float64))

        return np.column_stack(features) if features else X

    def fit_transform(self, X: np.ndarray, feature_names=None) -> np.ndarray:
        self.fit(X, feature_names)
        return self.transform(X)


class MaxentHSI:
    """
    MaxEnt 棲地適宜性模型

    使用方式:
        1. 準備漁獲出現點 (presence) 的環境變數
        2. 準備背景點 (background) 的環境變數
        3. fit(X_presence, X_background)
        4. predict(X_grid) → 棲地適宜性 0-1

    與 elapid 的差異:
        - 不依賴 rasterio/geopandas (避免安裝負擔)
        - 直接接受 numpy 陣列
        - sklearn LogisticRegression 替代自定義優化器
        - 保留 MaxEnt 核心數學 (feature transforms)

    如果日後安裝 elapid:
        pip install elapid
        model = elapid.MaxentModel()  # 更完整的實作
    """

    def __init__(self, C: float = 1.0, n_hinge: int = 10, n_threshold: int = 10):
        """
        Args:
            C: L1 正則化強度 (越小越強)
            n_hinge: Hinge knot 數量
            n_threshold: Threshold knot 數量
        """
        self.C = C
        self.transformer = MaxentFeatureTransformer(n_hinge, n_threshold)
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.model = None
        self._is_fitted = False

    def fit(
        self,
        X_presence: np.ndarray,
        X_background: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "MaxentHSI":
        """
        訓練 MaxEnt HSI 模型

        Args:
            X_presence: (n_presence, n_features) 漁獲出現點的環境變數
            X_background: (n_background, n_features) 背景點的環境變數
            feature_names: 特徵名稱列表

        Returns:
            self
        """
        if not HAS_SKLEARN:
            log.error("sklearn required for MaxentHSI.fit()")
            return self

        # 合併 presence + background
        X_all = np.vstack([X_presence, X_background])
        y_all = np.concatenate([
            np.ones(len(X_presence)),
            np.zeros(len(X_background)),
        ])

        # 處理 NaN
        nan_mask = ~np.any(np.isnan(X_all), axis=1)
        X_all = X_all[nan_mask]
        y_all = y_all[nan_mask]

        if len(X_all) < 10:
            log.error(f"Too few valid samples: {len(X_all)}")
            return self

        # Feature transforms
        X_transformed = self.transformer.fit_transform(X_all, feature_names)

        # 標準化
        X_scaled = self.scaler.fit_transform(X_transformed)

        # L1 Logistic Regression (MaxEnt 等價)
        self.model = LogisticRegression(
            penalty="l1",
            C=self.C,
            solver="liblinear",
            max_iter=1000,
            class_weight="balanced",  # 處理 presence << background 的不平衡
        )
        self.model.fit(X_scaled, y_all)
        self._is_fitted = True

        n_nonzero = np.sum(self.model.coef_ != 0)
        n_total = self.model.coef_.shape[1]
        log.info(f"MaxentHSI fitted: {n_nonzero}/{n_total} non-zero features, "
                 f"presence={int(y_all.sum())}, background={int((1-y_all).sum())}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        預測棲地適宜性

        Args:
            X: (n_samples, n_features) 環境變數

        Returns:
            (n_samples,) 棲地適宜性 0.0-1.0
        """
        if not self._is_fitted:
            log.warning("MaxentHSI not fitted, returning uniform 0.5")
            return np.full(len(X), 0.5, dtype=np.float32)

        # 處理 NaN
        nan_mask = np.any(np.isnan(X), axis=1)
        result = np.full(len(X), np.nan, dtype=np.float32)

        X_valid = X[~nan_mask]
        if len(X_valid) == 0:
            return result

        X_transformed = self.transformer.transform(X_valid)
        X_scaled = self.scaler.transform(X_transformed)
        proba = self.model.predict_proba(X_scaled)[:, 1]
        result[~nan_mask] = proba.astype(np.float32)

        return result

    def get_feature_importance(self) -> Dict[str, float]:
        """取得特徵重要性 (L1 係數絕對值)"""
        if not self._is_fitted:
            return {}
        importance = np.abs(self.model.coef_[0])
        # 只回報原始特徵的 linear 項 importance
        n_orig = len(self.transformer._feature_names)
        # 每個特徵產生 2 + n_hinge + n_threshold 個衍生特徵
        result = {}
        for i, name in enumerate(self.transformer._feature_names):
            # 加總這個特徵所有衍生項的 importance
            n_derived = 2 + self.transformer.n_hinge + self.transformer.n_threshold
            start = i * n_derived
            end = start + n_derived
            if end <= len(importance):
                result[name] = float(np.sum(importance[start:end]))
        return dict(sorted(result.items(), key=lambda x: -x[1]))
