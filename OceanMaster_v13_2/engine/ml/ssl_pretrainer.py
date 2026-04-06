"""
SSL — Self-Supervised Learning 自監督學習預訓練模組

============================================================
🎯 功能說明：
    讓模型在「沒有漁獲標註」的情況下，
    從大量無標註的衛星圖（SST、Chl-a、SSH）中
    自動學會海洋的基礎特徵（鋒面形狀、渦旋結構、溫度梯度）。

    當未來拿到真實漁獲日誌時，這些已經學會的「海洋常識」
    可以作為 Fine-tuning 的起點，加速訓練 5~10 倍。

📌 架構狀態：✅ 完整架構  |  ❌ 尚未訓練
📌 缺少什麼：大量無標註衛星圖（NOAA OISST、CMEMS Chl-a）
📌 買家需要：
    1. 下載 10 年份的 SST / Chl-a / SSH NetCDF 檔案
    2. 執行 ssl_pretrainer.pretrain() 進行自監督預訓練
    3. 預訓練完成後，將權重載入 U-Net / ConvLSTM 做 Fine-tuning

📌 方法論：
    使用 MAE (Masked Autoencoder) 策略：
    - 隨機遮蔽衛星圖的 75% 區域
    - 訓練模型「猜出被遮蔽的區域長什麼樣」
    - 模型因此學會海洋的空間結構與物理規律

📌 如何啟用：
    pretrainer = SSLPretrainer(backbone='unet')
    pretrainer.pretrain(satellite_data_dir='/path/to/netcdf/')
    pretrainer.export_pretrained_weights('pretrained_ocean_encoder.pt')
============================================================
"""

import logging
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MaskedOceanEncoder(nn.Module):
    """
    海洋遮罩自編碼器 (Masked Ocean Autoencoder)

    將衛星圖的 75% 區域遮蔽，訓練 Encoder 學會
    從剩餘 25% 的資訊中重建完整的海洋場。

    ⚠️ 這個 Encoder 訓練完成後，可以直接移植到
       U-Net / ConvLSTM 的 Encoder 部分，
       作為「已經學會海洋常識」的預訓練權重。
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 256):
        """
        Args:
            in_channels: 輸入通道數 (例如 3 = SST + Chl-a + SSH)
            embed_dim: 編碼器的嵌入維度
        """
        super().__init__()

        # === Encoder: 學習海洋特徵 ===
        # 這個 Encoder 的結構要跟你現有的 U-Net Encoder 相容
        # 這樣訓練完之後可以直接把權重複製過去
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, embed_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
        )

        # === Decoder: 重建被遮蔽的區域 ===
        # 訓練完成後，Decoder 會被丟棄，只保留 Encoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, in_channels, 3, padding=1),
        )

    def forward(
        self,
        x: 'torch.Tensor',
        mask: 'torch.Tensor'
    ) -> Tuple['torch.Tensor', 'torch.Tensor']:
        """
        Args:
            x: 輸入衛星圖, shape: (B, C, H, W)
            mask: 遮罩 (1=保留, 0=遮蔽), shape: (B, 1, H, W)

        Returns:
            reconstruction: 重建的完整圖, shape: (B, C, H, W)
            latent: 編碼器的潛在表示, shape: (B, embed_dim, H/4, W/4)
        """
        masked_input = x * mask  # 把被遮蔽的區域填零
        latent = self.encoder(masked_input)
        reconstruction = self.decoder(latent)
        return reconstruction, latent


class SSLPretrainer:
    """
    自監督學習預訓練器

    管理整個「無標註衛星圖預訓練」的流程。

    ⚠️ 架構狀態：骨架已完成，尚未訓練
    ⚠️ 買家使用流程：
        1. 準備衛星 NetCDF 資料
        2. 呼叫 pretrain() 開始自監督預訓練
        3. 呼叫 export_pretrained_weights() 導出權重
        4. 在 dl_trainer.py 中載入這些權重作為 DL 模型的初始值
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 256,
        mask_ratio: float = 0.75,
        learning_rate: float = 1e-4,
        device: str = 'auto'
    ):
        """
        Args:
            in_channels: 衛星資料的通道數
                         3 = SST + Chl-a + SSH
                         如果有更多資料 (DO, MLD 等)，可以增加
            embed_dim: Encoder 的嵌入維度
            mask_ratio: 遮蔽比例 (預設 75%，跟 MAE 論文一致)
            learning_rate: 學習率
            device: 'cuda' / 'cpu' / 'auto'
        """
        self.in_channels = in_channels
        self.mask_ratio = mask_ratio
        self.learning_rate = learning_rate

        if device == 'auto':
            self.device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model = None
        self.optimizer = None
        self._is_pretrained = False

        logger.info(
            f"SSL 預訓練器初始化 | channels={in_channels}, "
            f"mask_ratio={mask_ratio}, device={self.device}"
        )

    def _build_model(self):
        """建立 Masked Autoencoder 模型"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("需要安裝 PyTorch: pip install torch")

        self.model = MaskedOceanEncoder(
            in_channels=self.in_channels,
            embed_dim=256
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.05
        )

    def _generate_mask(
        self,
        batch_size: int,
        height: int,
        width: int
    ) -> 'torch.Tensor':
        """
        生成隨機遮罩

        以 patch 為單位遮蔽（而非逐像素），
        確保模型學到的是大尺度空間結構。

        Returns:
            mask: (B, 1, H, W), 1=保留, 0=遮蔽
        """
        patch_size = 16  # 每個 patch 16x16 像素
        h_patches = height // patch_size
        w_patches = width // patch_size
        total_patches = h_patches * w_patches
        keep_patches = int(total_patches * (1 - self.mask_ratio))

        mask = torch.zeros(batch_size, 1, height, width, device=self.device)

        for b in range(batch_size):
            # 隨機選擇要保留的 patch
            indices = torch.randperm(total_patches)[:keep_patches]
            for idx in indices:
                row = (idx // w_patches) * patch_size
                col = (idx % w_patches) * patch_size
                mask[b, 0, row:row+patch_size, col:col+patch_size] = 1.0

        return mask

    def pretrain(
        self,
        satellite_data_dir: Optional[str] = None,
        num_epochs: int = 100,
        batch_size: int = 16
    ) -> Dict[str, List[float]]:
        """
        執行自監督預訓練

        ⚠️ TODO: 買家需要實作以下步驟：
            1. 將 satellite_data_dir 指向存放 NetCDF 衛星圖的資料夾
            2. 實作 _load_satellite_batch() 方法來讀取 .nc 檔案
            3. 確保 GPU VRAM 至少 12GB (單卡 RTX 3060 以上)

        ⚠️ 預估訓練時間：
            - 10 年份衛星資料 (約 3,650 張圖)
            - 100 Epochs
            - 單卡 RTX 4090: 約 6~12 小時
            - 雙卡 RTX 4090: 約 3~6 小時

        Args:
            satellite_data_dir: 衛星 NetCDF 資料夾路徑
            num_epochs: 預訓練輪數
            batch_size: 批次大小

        Returns:
            訓練歷史 {'loss': [...]}
        """
        if self.model is None:
            self._build_model()

        logger.info(f"開始 SSL 自監督預訓練 | epochs={num_epochs}")

        # ⚠️ TODO: 買家在此處實作真實衛星資料的載入
        # 以下為架構示意，不含實際資料讀取
        raise NotImplementedError(
            "⚠️ SSL 預訓練需要真實衛星資料。\n"
            "買家請實作 _load_satellite_batch() 方法，\n"
            "從 satellite_data_dir 讀取 SST / Chl-a / SSH 的 NetCDF 檔案。\n"
            "預計需要的資料量：10 年份，每天一張，共約 3,650 張衛星圖。\n"
            "資料來源：NOAA OISST (免費) + CMEMS (免費註冊)"
        )

    def export_pretrained_weights(self, save_path: str):
        """
        導出預訓練好的 Encoder 權重

        ⚠️ 只導出 Encoder 部分，Decoder 會被丟棄。
        ⚠️ 買家在 DL 訓練時要這樣載入：
            state_dict = torch.load('pretrained_ocean_encoder.pt')
            unet_model.encoder.load_state_dict(state_dict, strict=False)

        Args:
            save_path: 權重儲存路徑
        """
        if self.model is None:
            raise RuntimeError("模型尚未建立或訓練")

        encoder_state = self.model.encoder.state_dict()
        torch.save(encoder_state, save_path)
        logger.info(f"Encoder 權重已導出至: {save_path}")

    def load_pretrained_weights(self, load_path: str):
        """
        載入預訓練權重

        Args:
            load_path: 預訓練權重檔案路徑
        """
        if self.model is None:
            self._build_model()

        state_dict = torch.load(load_path, map_location=self.device)
        self.model.encoder.load_state_dict(state_dict)
        self._is_pretrained = True
        logger.info(f"預訓練權重已從 {load_path} 載入")

    @property
    def is_pretrained(self) -> bool:
        """是否已完成預訓練"""
        return self._is_pretrained

    def get_encoder(self) -> nn.Module:
        """
        取得預訓練好的 Encoder

        買家可以直接把這個 Encoder 嫁接到 U-Net 或其他 DL 模型上。

        Returns:
            預訓練好的 Encoder nn.Module
        """
        if self.model is None:
            self._build_model()
        return self.model.encoder
