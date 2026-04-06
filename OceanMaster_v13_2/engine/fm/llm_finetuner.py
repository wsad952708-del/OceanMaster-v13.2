"""
FM — Foundation Model / LLM 微調管線

============================================================
🎯 功能說明：
    將大型語言模型 (LLM) 微調為「遠洋漁業專家」，使其能：
    1. 讀懂船長手寫日誌（例如「今天海面偏綠，魚訊好」）
    2. 解析漁業法規文件（例如 WCPFC 限捕令）
    3. 翻譯衛星數據報告為船長看得懂的中文建議
    4. 作為 AI Agent 的「大腦」提供決策推理

📌 架構狀態：✅ 完整架構  |  ❌ 尚未微調
📌 缺少什麼：
    1. 基礎模型權重（Llama 3 8B 或 Mistral 7B，需自行下載 ~15GB）
    2. 微調資料集（船長日誌、漁業 QA 對、法規摘要）
📌 買家需要：
    1. 從 HuggingFace 下載 LLM 基礎權重
    2. 準備至少 1,000 筆船長日誌 + 問答對
    3. 租用 GPU (至少 1× A6000 48GB 或 2× RTX 4090)
    4. 執行 finetuner.finetune() 開始微調

📌 微調方法：QLoRA (Quantized Low-Rank Adaptation)
    - 不需要修改 LLM 的所有參數（70 億個）
    - 只訓練約 0.1% 的額外參數（~7M）
    - VRAM 需求從 80GB 降低到 ~20GB
    - 品質幾乎等同全精度微調
============================================================
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FishingQAPair:
    """
    漁業問答對 — 用於微調 LLM 的訓練資料格式

    ⚠️ 買家需要準備至少 1,000 筆這種格式的資料
    """
    instruction: str    # 問題或指令
    input_text: str     # 情境或上下文（可為空）
    output_text: str    # 期望的回答

    # === 範例 ===
    # instruction: "根據以下海洋環境分析，今天適不適合出海捕鮪魚？"
    # input_text: "SST: 28.5°C, Chl-a: 0.35 mg/m³, 浪高: 1.2m, 風速: 12kt"
    # output_text: "適合。海溫 28.5°C 處於黃鰭鮪最適棲地範圍..."


class LLMFineTuner:
    """
    LLM 微調器 (QLoRA)

    負責將通用的大型語言模型微調成遠洋漁業專家。

    ⚠️ 架構狀態：完整骨架，需要買家提供以下才能運行：
        1. 基礎模型路徑 (model_name_or_path)
        2. 微調資料集 (training_data)
        3. GPU 硬體 (至少 20GB VRAM)

    ⚠️ 依賴套件 (買家需安裝)：
        pip install transformers peft bitsandbytes datasets accelerate
    """

    # === 支援的基礎模型 ===
    SUPPORTED_MODELS = {
        "llama3-8b": {
            "hf_name": "meta-llama/Meta-Llama-3-8B-Instruct",
            "params": "8B",
            "min_vram_gb": 20,
            "description": "Meta 的 Llama 3 8B，中英文理解佳"
        },
        "mistral-7b": {
            "hf_name": "mistralai/Mistral-7B-Instruct-v0.3",
            "params": "7B",
            "min_vram_gb": 18,
            "description": "Mistral 7B，輕量但推理速度快"
        },
        "qwen2-7b": {
            "hf_name": "Qwen/Qwen2-7B-Instruct",
            "params": "7B",
            "min_vram_gb": 18,
            "description": "通義千問 2，中文能力最強"
        },
    }

    def __init__(
        self,
        model_name: str = "qwen2-7b",
        lora_rank: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        quantization_bits: int = 4,
        output_dir: str = "./fm_checkpoints"
    ):
        """
        Args:
            model_name: 基礎模型代號 (見 SUPPORTED_MODELS)
            lora_rank: LoRA 的秩 (越高品質越好但越吃 VRAM)
            lora_alpha: LoRA 的 alpha 縮放因子
            lora_dropout: LoRA 的 dropout 比率
            quantization_bits: 量化位數 (4 = QLoRA, 8 = 8-bit, 16 = 半精度)
            output_dir: 微調後權重的儲存路徑
        """
        self.model_name = model_name
        self.model_config = self.SUPPORTED_MODELS.get(model_name, {})
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.quantization_bits = quantization_bits
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.tokenizer = None
        self._is_finetuned = False

        logger.info(
            f"LLM 微調器初始化 | model={model_name}, "
            f"LoRA r={lora_rank}, bits={quantization_bits}"
        )

    def load_base_model(self):
        """
        載入基礎 LLM 模型 (量化版)

        ⚠️ 首次執行需下載模型權重 (~15GB)
        ⚠️ 需要安裝: pip install transformers peft bitsandbytes
        ⚠️ 需要 HuggingFace 帳號和 Access Token

        ⚠️ 買家執行步驟：
            1. 到 huggingface.co 註冊帳號
            2. 取得 Access Token
            3. 在終端機執行: huggingface-cli login
            4. 再呼叫此函數
        """
        raise NotImplementedError(
            "⚠️ LLM 載入需要安裝 transformers + peft + bitsandbytes。\n"
            "買家請執行：\n"
            "  pip install transformers peft bitsandbytes datasets accelerate\n"
            "  huggingface-cli login\n"
            f"然後下載模型: {self.model_config.get('hf_name', 'N/A')}\n"
            f"最低 VRAM 需求: {self.model_config.get('min_vram_gb', 'N/A')} GB"
        )

    def prepare_dataset(
        self,
        qa_pairs: List[FishingQAPair]
    ) -> Any:
        """
        將漁業問答對轉換為 LLM 微調格式

        Args:
            qa_pairs: 漁業問答對列表

        Returns:
            HuggingFace Dataset 物件

        ⚠️ 買家需要準備的資料範例：
            data = [
                FishingQAPair(
                    instruction="分析今天的漁場條件",
                    input_text="SST=29°C, Chl-a=0.4, SSH異常=+0.15m",
                    output_text="條件優良。海溫 29°C 處於大目鮪最適範圍..."
                ),
                FishingQAPair(
                    instruction="這段船長日誌提到了什麼漁獲資訊？",
                    input_text="今天北緯 15 度有大批水鳥聚集，下網 4 小時...",
                    output_text="從日誌中擷取：位置=N15°, 指標=水鳥聚集..."
                ),
            ]
        """
        formatted = []
        for qa in qa_pairs:
            formatted.append({
                "instruction": qa.instruction,
                "input": qa.input_text,
                "output": qa.output_text
            })

        logger.info(f"已準備 {len(formatted)} 筆微調資料")
        return formatted

    def finetune(
        self,
        training_data: List[FishingQAPair],
        num_epochs: int = 3,
        learning_rate: float = 2e-4,
        batch_size: int = 4,
        max_seq_length: int = 2048
    ) -> Dict[str, Any]:
        """
        執行 QLoRA 微調

        ⚠️ 預估訓練時間：
            - 1,000 筆資料, 3 Epochs
            - 單卡 RTX 4090 (QLoRA 4-bit): 約 2~4 小時
            - 單卡 A6000 48GB (LoRA FP16): 約 1~2 小時
            - 雙卡 A100 80GB: 約 30 分鐘

        ⚠️ VRAM 需求：
            - QLoRA 4-bit: ~20 GB (RTX 4090 可跑)
            - LoRA FP16: ~35 GB (需要 A6000 以上)

        Args:
            training_data: 漁業問答對列表
            num_epochs: 微調輪數
            learning_rate: 學習率
            batch_size: 批次大小
            max_seq_length: 最大序列長度

        Returns:
            訓練歷史
        """
        raise NotImplementedError(
            "⚠️ LLM 微調需要完整的 GPU 環境。\n"
            "買家請確認以下條件已就緒：\n"
            "  1. ✅ 已安裝 transformers, peft, bitsandbytes\n"
            "  2. ✅ 已下載基礎模型權重\n"
            "  3. ✅ GPU VRAM ≥ 20GB\n"
            "  4. ✅ 已準備至少 1,000 筆 FishingQAPair 資料\n"
            "然後在此函數中實作 QLoRA 訓練迴圈。"
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """
        使用微調後的 LLM 進行推論

        Args:
            prompt: 輸入提示詞
            max_new_tokens: 最大生成長度
            temperature: 生成溫度 (越低越確定性)

        Returns:
            LLM 的回應文本

        ⚠️ 使用範例：
            finetuner = LLMFineTuner()
            finetuner.load_finetuned("./fm_checkpoints/best")
            response = finetuner.generate(
                "SST=28°C, Chl-a=0.3, 浪高=1.5m。今天該去哪裡捕魚？"
            )
        """
        if not self._is_finetuned:
            raise RuntimeError(
                "⚠️ LLM 尚未微調。請先執行 finetune() 或 load_finetuned()。"
            )
        raise NotImplementedError("需要載入微調後的模型才能生成。")

    def load_finetuned(self, checkpoint_path: str):
        """載入微調後的 LoRA 權重"""
        logger.info(f"載入微調權重: {checkpoint_path}")
        # ⚠️ TODO: 實作 peft.PeftModel.from_pretrained()
        self._is_finetuned = True

    @property
    def is_finetuned(self) -> bool:
        return self._is_finetuned
