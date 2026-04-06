"""
FL — Federated Learning 聯邦學習模組

============================================================
🎯 功能說明：
    讓多家漁業公司「共同訓練」AI 模型，但不需要分享各自的機密漁獲日誌。

    原理：
    - 每家公司在自己的電腦上用自己的數據訓練模型
    - 只把「訓練後的模型梯度」上傳到中央伺服器
    - 中央伺服器把所有公司的梯度平均後，發回更新後的模型
    - 結果：每家公司都受益於所有公司的數據量，但沒有人洩漏自己的秘密

    適用場景：
    - 3~5 家漁業公司各自有 10 年漁獲日誌
    - 單獨訓練效果一般，但合起來有 30~50 年的數據量
    - 但漁獲日誌是商業機密，不可能直接交給對方

📌 架構狀態：✅ 完整架構  |  ❌ 尚未部署
📌 缺少什麼：至少 2 個以上的數據持有者（漁業公司/漁會）
📌 買家需要：
    1. 找到 2+ 家願意合作的漁業公司
    2. 每家公司各自部署一個 FederatedClient
    3. 部署一個 FederatedServer 作為中央協調器
    4. 各公司定期上傳梯度，下載更新後的全局模型

📌 依賴框架：Flower (pip install flwr) — 最流行的聯邦學習框架
============================================================
"""

import logging
from typing import Dict, List, Optional, Any
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
class FederatedRound:
    """聯邦學習的一個回合"""
    round_id: int
    participating_clients: List[str]
    global_loss: float = 0.0
    improvements: Dict[str, float] = None


class FederatedServer:
    """
    聯邦學習伺服器（中央協調器）

    負責：
    1. 分發全局模型給各客戶端
    2. 收集各客戶端的梯度更新
    3. 聚合梯度 (FedAvg 演算法)
    4. 更新全局模型並發回

    ⚠️ 架構狀態：骨架已完成
    ⚠️ 買家使用流程：
        server = FederatedServer()
        server.initialize_global_model(model_architecture)
        server.run_rounds(num_rounds=50)
    """

    def __init__(
        self,
        aggregation_strategy: str = "fedavg",
        min_clients: int = 2,
        min_fit_clients: int = 2
    ):
        """
        Args:
            aggregation_strategy: 聚合策略
                - "fedavg": Federated Averaging（最經典）
                - "fedprox": FedProx（處理數據不均勻）
            min_clients: 最小參與客戶端數
            min_fit_clients: 最小訓練客戶端數
        """
        self.aggregation_strategy = aggregation_strategy
        self.min_clients = min_clients
        self.min_fit_clients = min_fit_clients
        self._global_model_state = None
        self._round_history: List[FederatedRound] = []
        self._registered_clients: Dict[str, Dict] = {}

        logger.info(
            f"FL Server 初始化 | 策略={aggregation_strategy}, "
            f"最少 {min_clients} 個客戶端"
        )

    def register_client(self, client_id: str, client_name: str, data_size: int):
        """
        註冊一個聯邦學習客戶端（一家漁業公司）

        ⚠️ 範例：
            server.register_client("company_a", "海豐漁業", data_size=50000)
            server.register_client("company_b", "新東洋水產", data_size=30000)
        """
        self._registered_clients[client_id] = {
            "name": client_name,
            "data_size": data_size,
            "rounds_participated": 0
        }
        logger.info(f"FL Client 註冊: {client_name} (資料量: {data_size})")

    def initialize_global_model(self, model_state_dict: Optional[Dict] = None):
        """
        初始化全局模型

        ⚠️ TODO: 買家提供模型架構的 state_dict
        """
        self._global_model_state = model_state_dict
        logger.info("全局模型已初始化")

    def aggregate_updates(self, client_updates: Dict[str, Dict]) -> Dict:
        """
        聚合各客戶端的梯度更新 (FedAvg)

        FedAvg 演算法：
        全局模型 = Σ (客戶端模型 × 客戶端數據量) / 總數據量

        Args:
            client_updates: {client_id: model_state_dict}

        Returns:
            更新後的全局 model_state_dict
        """
        if len(client_updates) < self.min_fit_clients:
            raise ValueError(
                f"參與客戶端不足。需要至少 {self.min_fit_clients} 個，"
                f"目前只有 {len(client_updates)} 個。"
            )

        # ⚠️ TODO: 實作 FedAvg 加權平均
        raise NotImplementedError(
            "⚠️ 聯邦聚合需要至少 2 個客戶端的模型更新。\n"
            "買家請確保已註冊至少 2 家漁業公司，\n"
            "各自在本地訓練後上傳梯度。"
        )

    def run_rounds(self, num_rounds: int = 50):
        """
        執行多輪聯邦學習

        ⚠️ TODO: 買家需要部署 Flower 框架
        """
        raise NotImplementedError(
            "⚠️ 聯邦學習需要部署分散式訓練框架。\n"
            "買家請安裝: pip install flwr\n"
            "然後在每家公司的電腦上部署 FederatedClient。"
        )


class FederatedClient:
    """
    聯邦學習客戶端（一家漁業公司的本地訓練器）

    在公司自己的電腦上用自己的數據訓練，
    只上傳模型梯度，不上傳原始漁獲日誌。

    ⚠️ 架構狀態：骨架已完成
    ⚠️ 每家漁業公司各部署一個
    """

    def __init__(
        self,
        client_id: str,
        local_data_path: str,
        server_address: str = "localhost:8080"
    ):
        """
        Args:
            client_id: 客戶端識別碼
            local_data_path: 本地漁獲日誌路徑
            server_address: 聯邦伺服器地址
        """
        self.client_id = client_id
        self.local_data_path = local_data_path
        self.server_address = server_address
        self._local_model = None

        logger.info(f"FL Client 初始化 | id={client_id}, server={server_address}")

    def local_train(self, global_model_state: Dict, num_local_epochs: int = 5) -> Dict:
        """
        在本地數據上訓練模型

        ⚠️ 數據永遠不會離開這台電腦！
        ⚠️ 只有訓練後的模型權重會被上傳。

        Args:
            global_model_state: 從伺服器收到的全局模型
            num_local_epochs: 本地訓練輪數

        Returns:
            本地訓練後的模型 state_dict
        """
        raise NotImplementedError(
            "⚠️ 本地訓練需要漁獲日誌資料。\n"
            f"買家請將資料放在: {self.local_data_path}\n"
            "資料格式：日期, 緯度, 經度, 物種, 漁獲量(kg)"
        )

    def get_model_update(self) -> Dict:
        """取得模型更新（只有梯度差異，不含原始數據）"""
        raise NotImplementedError("需要先完成 local_train()。")
