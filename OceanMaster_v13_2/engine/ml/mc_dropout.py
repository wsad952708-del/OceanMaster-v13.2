"""
OceanMaster v13.2 — MC Dropout 不確定性量化  [逆向工程成果]
==========================================================
四間競爭對手（蒼鷺/海鷹/CATCH/GreenFish）全部只給「最佳預測值」，
不提供任何信心區間。這是它們的共同盲點。

本模組用 Monte Carlo Dropout 在推論時啟用 dropout，跑 N 次前向傳播，
取 mean 做最佳預測、std 做不確定性。漁民看到
「HSI = 0.85 ± 0.03」比「HSI = 0.85」更有信心。

用法:
    from engine.ml.mc_dropout import MCDropoutWrapper
    wrapper = MCDropoutWrapper(n_forward=30)
    result = wrapper.predict_with_uncertainty(model, input_tensor)
    print(result["mean"])     # 最佳預測
    print(result["std"])      # 不確定性（越大越不確定）
    print(result["ci_low"])   # 95% 信心下界
    print(result["ci_high"])  # 95% 信心上界

科學依據:
    Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation"
    ICML 2016 — 證明 MC Dropout ≈ 變分貝葉斯推論

適用模型:
    - UNetFishingPredictor (unet_fishing.py)
    - ConvLSTMPredictor (convlstm_predictor.py)
    - 任何含 nn.Dropout 層的 PyTorch 模型
"""

import numpy as np
import logging
from typing import Any, Dict, Optional

log = logging.getLogger("OceanMaster.MCDropout")


class MCDropoutWrapper:
    """
    Monte Carlo Dropout 不確定性量化包裝器。

    原理：
      1. 推論時保持 dropout 啟用（model.train() 但不更新梯度）
      2. 對同一輸入跑 N 次前向傳播，每次 dropout 隨機遮蔽不同神經元
      3. N 次結果取 mean = 最佳預測，std = 不確定性
      4. 95% CI = mean ± 1.96 × std

    優勢 vs 其他方法：
      - vs Ensemble: 不需訓練多個模型（成本低 N 倍）
      - vs Quantile Regression: 不需修改模型架構
      - 缺點: 需要模型有 dropout 層（你的 U-Net 和 ConvLSTM 都有）
    """

    def __init__(self, n_forward: int = 30, confidence: float = 0.95):
        """
        Args:
            n_forward: MC 前向傳播次數（30 通常足夠，越多越精確但越慢）
            confidence: 信心水準（default 0.95 = 95% CI）
        """
        self.n_forward = n_forward
        self.confidence = confidence
        # z-score for confidence interval
        if confidence == 0.95:
            self.z = 1.96
        elif confidence == 0.90:
            self.z = 1.645
        elif confidence == 0.99:
            self.z = 2.576
        else:
            # General case (normal approx)
            from scipy.stats import norm
            self.z = norm.ppf((1 + confidence) / 2)

    def predict_with_uncertainty(
        self,
        model: Any,
        input_tensor: Any,
        return_all: bool = False,
    ) -> Dict[str, Any]:
        """
        用 MC Dropout 預測並量化不確定性。

        Args:
            model: PyTorch nn.Module（必須含 dropout 層）
            input_tensor: 模型輸入 tensor (已在正確 device 上)
            return_all: 是否返回全部 N 組原始預測

        Returns:
            {
                "mean": np.ndarray — 最佳預測（N 次平均）
                "std": np.ndarray — 不確定性（N 次標準差）
                "ci_low": np.ndarray — 信心下界
                "ci_high": np.ndarray — 信心上界
                "coefficient_of_variation": np.ndarray — std/mean（相對不確定性）
                "confidence": float — 信心水準
                "n_forward": int — MC 次數
                "all_predictions": Optional[np.ndarray] — 全部 N 組預測（如果 return_all=True）
            }
        """
        try:
            import torch
        except ImportError:
            log.error("PyTorch not installed — MC Dropout unavailable")
            return {"error": "pytorch_not_installed"}

        # 檢查模型有沒有 dropout 層
        has_dropout = any(
            isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d))
            for m in model.modules()
        )
        if not has_dropout:
            log.warning("⚠️ 模型沒有 Dropout 層 — MC Dropout 退化為普通推論，"
                        "不確定性全為 0")

        # 啟用 dropout（但不計算梯度）
        model.train()  # 保持 dropout 啟用
        predictions = []

        with torch.no_grad():
            for i in range(self.n_forward):
                output = model(input_tensor)
                if isinstance(output, torch.Tensor):
                    predictions.append(output.cpu().numpy())
                else:
                    predictions.append(np.asarray(output))

        # 恢復模型到 eval 模式
        model.eval()

        # Stack: (N, *output_shape)
        all_preds = np.stack(predictions, axis=0)

        # 統計
        mean = np.mean(all_preds, axis=0)
        std = np.std(all_preds, axis=0)
        ci_low = mean - self.z * std
        ci_high = mean + self.z * std

        # 變異係數（相對不確定性）
        mean_safe = np.where(np.abs(mean) > 1e-10, mean, 1e-10)
        cv = std / np.abs(mean_safe)

        log.info(f"  MC Dropout ({self.n_forward}x): "
                 f"mean={np.mean(mean):.4f}, "
                 f"avg_std={np.mean(std):.4f}, "
                 f"avg_CV={np.mean(cv):.2%}")

        result = {
            "mean": mean,
            "std": std,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "coefficient_of_variation": cv,
            "confidence": self.confidence,
            "n_forward": self.n_forward,
        }

        if return_all:
            result["all_predictions"] = all_preds

        return result

    def compute_spatial_uncertainty_map(
        self,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        從 MC Dropout 結果生成空間不確定性指標。

        用於 dashboard 顯示：
          - 🟢 低不確定性區域 (CV < 10%): 高信心漁場
          - 🟡 中不確定性區域 (10-30%): 參考用
          - 🔴 高不確定性區域 (CV > 30%): 需謹慎

        Args:
            result: predict_with_uncertainty 的返回值

        Returns:
            {
                "confidence_mask": np.ndarray — 3 級信心遮罩 (1=高,2=中,3=低)
                "pct_high_confidence": float — 高信心區域佔比
                "pct_medium_confidence": float
                "pct_low_confidence": float
                "recommendation": str — 中文建議
            }
        """
        cv = result.get("coefficient_of_variation")
        if cv is None:
            return {"error": "no_cv_data"}

        # 分級
        high = cv < 0.10      # CV < 10%
        medium = (cv >= 0.10) & (cv < 0.30)
        low = cv >= 0.30

        total = cv.size
        pct_high = float(np.sum(high)) / max(total, 1)
        pct_med = float(np.sum(medium)) / max(total, 1)
        pct_low = float(np.sum(low)) / max(total, 1)

        # 信心遮罩
        mask = np.ones_like(cv, dtype=np.int8) * 2  # default = 中
        mask[high] = 1   # 高信心
        mask[low] = 3     # 低信心

        # 中文建議
        if pct_high > 0.7:
            rec = "🟢 整體預測信心高，可放心參考漁場建議"
        elif pct_high > 0.4:
            rec = "🟡 部分區域預測不確定，建議優先前往高信心（綠色）區域"
        else:
            rec = "🔴 預測不確定性偏高，建議結合個人經驗謹慎判斷"

        return {
            "confidence_mask": mask,
            "pct_high_confidence": pct_high,
            "pct_medium_confidence": pct_med,
            "pct_low_confidence": pct_low,
            "recommendation": rec,
        }
