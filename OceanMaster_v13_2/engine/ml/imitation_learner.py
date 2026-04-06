"""
IL — Imitation Learning 模仿學習模組

============================================================
🎯 功能說明：
    從資深船長的歷史決策記錄中學習，
    讓 AI 模仿「30 年經驗老船長」的直覺判斷。

    原理：
    - 收集老船長過去 N 年的決策：「在什麼環境下 → 去了哪裡 → 抓到多少」
    - 把環境條件作為 Input，船長決策作為 Output
    - 訓練神經網路模仿這個 Input → Output 的映射

    效果：AI 學會了船長「看到海水偏綠就知道有魚」的直覺。

📌 架構狀態：✅ 完整架構  |  ❌ 尚未訓練
📌 缺少什麼：資深船長的歷史決策記錄 (至少 500 次出海記錄)
📌 資料格式要求：
    每筆記錄需包含：
    - 出海日期
    - 環境條件快照 (SST, Chl-a, SSH, 浪高, 風速等)
    - 船長的決策 (去了哪裡、用了什麼漁法)
    - 結果 (漁獲量)
============================================================
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class CaptainDecisionRecord:
    """
    船長決策記錄 — 模仿學習的訓練資料格式

    ⚠️ 買家需要收集至少 500 筆這種格式的記錄
    """
    date: str                          # 出海日期
    environment_features: Dict         # 環境條件 (SST, Chl-a 等)
    decision_lat: float                # 船長決定去的緯度
    decision_lon: float                # 船長決定去的經度
    fishing_method: str                # 使用的漁法
    actual_catch_kg: float             # 實際漁獲量 (kg)
    captain_id: str = ""               # 船長識別碼
    captain_experience_years: int = 0  # 船長經驗年數


class CaptainPolicyNetwork(nn.Module):
    """
    船長策略網路 — 模仿船長的決策模型

    輸入：環境條件向量 (SST, Chl-a, SSH, 浪高, 風速, 月相, ...)
    輸出：預測船長會去的座標 (lat, lon) 和使用的漁法

    ⚠️ 訓練完成後，這個網路就是「虛擬老船長」
    """

    def __init__(self, input_dim: int = 44, hidden_dim: int = 256, num_fishing_methods: int = 5):
        super().__init__()

        # 共享特徵提取
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # 座標預測頭 (回歸)
        self.location_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # (lat, lon)
        )

        # 漁法預測頭 (分類)
        self.method_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_fishing_methods),
        )

    def forward(self, x: 'torch.Tensor') -> Tuple['torch.Tensor', 'torch.Tensor']:
        """
        Args:
            x: 環境條件向量, shape: (B, input_dim)

        Returns:
            location: 預測座標 (B, 2)
            method_logits: 漁法機率 (B, num_methods)
        """
        features = self.feature_extractor(x)
        location = self.location_head(features)
        method_logits = self.method_head(features)
        return location, method_logits


class ImitationLearner:
    """
    模仿學習器

    ⚠️ 架構狀態：骨架已完成，需要船長決策資料才能訓練
    ⚠️ 買家使用流程：
        1. 收集資深船長的出海記錄 (CaptainDecisionRecord)
        2. 呼叫 train() 開始模仿學習
        3. 呼叫 predict_like_captain() 讓 AI 模仿老船長決策
    """

    def __init__(self, input_dim: int = 44, device: str = 'auto'):
        self.input_dim = input_dim
        self.device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu' if device == 'auto' else device
        self.model = None

    def train(
        self,
        records: List[CaptainDecisionRecord],
        num_epochs: int = 100
    ) -> Dict:
        """
        訓練模仿學習模型

        ⚠️ TODO: 買家需要提供船長決策記錄
        """
        raise NotImplementedError(
            "⚠️ 模仿學習需要船長決策記錄。\n"
            f"買家請提供至少 500 筆 CaptainDecisionRecord。\n"
            f"目前收到 {len(records)} 筆。"
        )

    def predict_like_captain(self, environment: Dict) -> Dict:
        """
        像老船長一樣做決策

        Args:
            environment: 當前環境條件 {"sst": 28.5, "chl_a": 0.3, ...}

        Returns:
            {"predicted_lat": ..., "predicted_lon": ..., "fishing_method": ...}
        """
        raise NotImplementedError("⚠️ 需要先完成訓練。")
