"""
AI Agent — 遠洋漁業 AI 指揮官（中央大腦）

============================================================
🎯 功能說明：
    OceanMaster 的最高層決策架構。
    AI Agent 是整個系統的「大腦」，它：
    1. 接收船長的問題或指令
    2. 決定要呼叫哪些模型（ML/DL/RL/FM/RAG/PINN/GA...）
    3. 串接各模型的輸出，進行綜合推理
    4. 產出最終的漁場建議、航線規劃、風險評估

    ┌─────────────────────────────────────┐
    │           AI Agent (本模組)          │
    │          🧠 中央決策引擎             │
    ├─────────────────────────────────────┤
    │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐  │
    │  │  ML │ │  DL │ │  RL │ │  FM │  │
    │  │XGBst│ │U-Net│ │Route│ │ LLM │  │
    │  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘  │
    │     │       │       │       │      │
    │  ┌──┴──┐ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐  │
    │  │PINN │ │ SSL │ │ GA  │ │ RAG │  │
    │  └─────┘ └─────┘ └─────┘ └─────┘  │
    │                                     │
    │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐  │
    │  │GenAI│ │Edge │ │ MAS │ │ IL  │  │
    │  └─────┘ └─────┘ └─────┘ └─────┘  │
    └─────────────────────────────────────┘

📌 架構狀態：✅ 完整架構  |  ❌ 尚未連接 FM (LLM)
📌 缺少什麼：
    1. FM 微調完成的 LLM 權重
    2. RAG 知識庫（歷史漁獲文件）
    3. 各衛星資料的 API Key
📌 買家需要：
    1. 完成 FM 微調
    2. 建立 RAG 知識庫
    3. 呼叫 agent.process_request() 開始使用

📌 Agent 的決策流程：
    船長輸入 → 意圖識別 → 工具選擇 → 工具執行 → 結果整合 → 輸出建議
============================================================
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """船長意圖分類"""
    HOTSPOT_QUERY = "hotspot_query"           # 「今天哪裡有魚？」
    ROUTE_PLANNING = "route_planning"         # 「幫我規劃航線」
    WEATHER_CHECK = "weather_check"           # 「明天的天氣安全嗎？」
    HISTORICAL_QUERY = "historical_query"     # 「去年同期帛琉怎麼樣？」
    SPECIES_ANALYSIS = "species_analysis"     # 「大目鮪現在在哪裡聚集？」
    REGULATION_CHECK = "regulation_check"     # 「帛琉 EEZ 可以捕什麼魚？」
    FUEL_OPTIMIZATION = "fuel_optimization"   # 「最省油的回港路線？」
    RISK_ASSESSMENT = "risk_assessment"       # 「這個區域安全嗎？」
    GENERAL_QUESTION = "general_question"     # 其他自由問答


@dataclass
class ToolResult:
    """工具（模型）的執行結果"""
    tool_name: str              # 工具名稱（例如 "unet_predictor"）
    success: bool               # 是否成功
    data: Any = None            # 返回的數據
    error: str = ""             # 錯誤訊息
    execution_time_ms: float = 0.0  # 執行時間 (毫秒)


@dataclass
class AgentResponse:
    """Agent 的最終回應"""
    answer: str                         # 給船長看的中文回答
    confidence: float                   # 信心度 0~1
    tools_used: List[str]               # 使用了哪些工具
    data_attachments: Dict = field(default_factory=dict)  # 附帶的數據（熱點座標、航線等）
    warnings: List[str] = field(default_factory=list)     # 安全警告
    timestamp: str = ""


class FishingAIAgent:
    """
    遠洋漁業 AI 指揮官

    串接所有 ML/DL/RL/FM/RAG/PINN/GA/GenAI/Edge/MAS/IL/FL 模組，
    作為統一的決策入口。

    ⚠️ 架構狀態：骨架已完成
    ⚠️ 買家使用流程：
        1. 初始化 Agent 並註冊所有可用工具
        2. 呼叫 agent.process_request("今天哪裡有魚？")
        3. Agent 自動決定呼叫哪些模型，返回綜合建議
    """

    def __init__(self):
        """
        初始化 AI Agent

        Agent 本身不需要訓練。
        它的「智慧」來自它串接的各個模型。
        """
        # === 已註冊的工具（模型） ===
        self._tools: Dict[str, Callable] = {}
        self._tool_descriptions: Dict[str, str] = {}

        # === 系統狀態 ===
        self._fm_available = False          # FM (LLM) 是否已微調就緒
        self._rag_available = False         # RAG 知識庫是否已建立
        self._models_loaded = False         # DL/ML 模型是否已載入

        # === 對話歷史 ===
        self._conversation_history: List[Dict] = []

        logger.info("AI Agent 初始化完成 | 等待工具註冊")

    # ============================================================
    # 工具註冊 — 把所有模型「報到」
    # ============================================================
    def register_tool(
        self,
        name: str,
        function: Callable,
        description: str
    ):
        """
        向 Agent 註冊一個工具（模型）

        ⚠️ 買家在 main 入口處這樣做：
            agent = FishingAIAgent()

            # 註冊現有模型
            agent.register_tool(
                name="unet_predictor",
                function=unet.predict,
                description="U-Net 空間熱點預測模型"
            )
            agent.register_tool(
                name="convlstm_predictor",
                function=convlstm.predict,
                description="ConvLSTM 時空序列預測模型"
            )
            agent.register_tool(
                name="route_planner",
                function=route_planner.compute_all_routes,
                description="RL 航線規劃器"
            )
            agent.register_tool(
                name="safety_checker",
                function=safety.check,
                description="安全檢查模組"
            )
            # ... 註冊所有其他模型 ...

        Args:
            name: 工具名稱（唯一識別碼）
            function: 工具的可呼叫函數
            description: 工具功能描述（給 Agent 決策用）
        """
        self._tools[name] = function
        self._tool_descriptions[name] = description
        logger.info(f"工具已註冊: {name} — {description}")

    def register_fm(self, llm_generate_fn: Callable):
        """
        註冊 FM (LLM) 推論函數

        ⚠️ 買家在 FM 微調完成後：
            from engine.fm.llm_finetuner import LLMFineTuner
            finetuner = LLMFineTuner()
            finetuner.load_finetuned("./fm_checkpoints/best")
            agent.register_fm(finetuner.generate)
        """
        self._tools["fm_llm"] = llm_generate_fn
        self._fm_available = True
        logger.info("FM (LLM) 已註冊至 Agent")

    def register_rag(self, rag_query_fn: Callable):
        """
        註冊 RAG 檢索函數

        ⚠️ 買家在 RAG 知識庫建立後：
            from engine.fm.rag_engine import RAGEngine
            rag = RAGEngine()
            rag.ingest_documents(my_documents)
            agent.register_rag(rag.query)
        """
        self._tools["rag_search"] = rag_query_fn
        self._rag_available = True
        logger.info("RAG 檢索引擎已註冊至 Agent")

    # ============================================================
    # 意圖識別 — 理解船長想做什麼
    # ============================================================
    def _classify_intent(self, user_input: str) -> IntentType:
        """
        分類船長的意圖

        ⚠️ 當 FM (LLM) 可用時：使用 LLM 進行語意理解
        ⚠️ 當 FM 不可用時：使用關鍵字比對（降級模式）
        """
        # === 降級模式：關鍵字比對 ===
        keyword_map = {
            IntentType.HOTSPOT_QUERY: ["熱點", "哪裡有魚", "漁場", "去哪", "捕魚"],
            IntentType.ROUTE_PLANNING: ["航線", "路線", "怎麼走", "規劃"],
            IntentType.WEATHER_CHECK: ["天氣", "浪高", "颱風", "風速", "安全"],
            IntentType.HISTORICAL_QUERY: ["去年", "歷史", "以前", "上次", "同期"],
            IntentType.SPECIES_ANALYSIS: ["鮪魚", "旗魚", "黃鰭", "大目", "物種"],
            IntentType.REGULATION_CHECK: ["法規", "EEZ", "限捕", "禁漁"],
            IntentType.FUEL_OPTIMIZATION: ["油耗", "燃油", "省油", "回港"],
            IntentType.RISK_ASSESSMENT: ["危險", "風險", "瘋狗浪", "安不安全"],
        }

        for intent, keywords in keyword_map.items():
            if any(kw in user_input for kw in keywords):
                return intent

        return IntentType.GENERAL_QUESTION

    # ============================================================
    # 工具編排 — 決定要呼叫哪些模型
    # ============================================================
    def _plan_tool_chain(self, intent: IntentType) -> List[str]:
        """
        根據意圖規劃工具呼叫鏈

        不同的問題需要不同的模型組合：
        「今天哪裡有魚？」→ DL 預測 → 安全過濾 → 航線規劃
        「去年同期怎麼樣？」→ RAG 檢索 → LLM 摘要
        """
        # === 工具鏈映射 ===
        tool_chains = {
            IntentType.HOTSPOT_QUERY: [
                "data_orchestrator",        # 抓衛星數據
                "feature_builder",          # 建構特徵
                "unet_predictor",           # U-Net 空間預測
                "convlstm_predictor",       # ConvLSTM 時序預測
                "xgboost_predictor",        # XGBoost 即時預測
                "stacking_ensemble",        # 模型融合
                "safety_checker",           # 安全過濾
                "species_probability",      # 物種機率
            ],
            IntentType.ROUTE_PLANNING: [
                "hotspot_query_chain",      # 先找熱點
                "route_planner",            # RL 航線規劃
                "fuel_predictor",           # 燃油估算
                "genetic_optimizer",        # GA 全局最佳化
            ],
            IntentType.WEATHER_CHECK: [
                "weather_fetcher",          # 天氣資料
                "wave_fetcher",             # 浪高資料
                "typhoon_tracker",          # 颱風追蹤
                "safety_checker",           # 安全評估
            ],
            IntentType.HISTORICAL_QUERY: [
                "rag_search",               # RAG 檢索歷史記錄
                "fm_llm",                   # LLM 摘要回答
            ],
            IntentType.SPECIES_ANALYSIS: [
                "species_probability",
                "fish_behavior_model",
                "migration_corridor",
                "dvm_model",
            ],
            IntentType.REGULATION_CHECK: [
                "rag_search",               # 搜尋法規
                "fm_llm",                   # LLM 解讀法規
            ],
            IntentType.FUEL_OPTIMIZATION: [
                "route_planner",
                "fuel_predictor",
                "genetic_optimizer",
            ],
            IntentType.RISK_ASSESSMENT: [
                "safety_checker",
                "typhoon_tracker",
                "wave_fetcher",
            ],
            IntentType.GENERAL_QUESTION: [
                "rag_search",
                "fm_llm",
            ],
        }

        planned = tool_chains.get(intent, ["fm_llm"])

        # 只返回已註冊的工具
        available = [t for t in planned if t in self._tools]
        unavailable = [t for t in planned if t not in self._tools]

        if unavailable:
            logger.warning(
                f"以下工具尚未註冊，將跳過: {unavailable}\n"
                f"⚠️ 買家需要註冊這些工具才能完整運作。"
            )

        return available

    # ============================================================
    # 執行工具鏈
    # ============================================================
    def _execute_tool_chain(
        self,
        tool_chain: List[str],
        context: Dict
    ) -> List[ToolResult]:
        """
        依序執行工具鏈中的每個工具

        每個工具的輸出會作為下一個工具的輸入（管線模式）。
        """
        results = []

        for tool_name in tool_chain:
            tool_fn = self._tools.get(tool_name)
            if tool_fn is None:
                results.append(ToolResult(
                    tool_name=tool_name,
                    success=False,
                    error=f"工具 '{tool_name}' 未註冊"
                ))
                continue

            try:
                start_time = datetime.now()
                # ⚠️ TODO: 買家需要確保每個工具的輸入/輸出格式一致
                output = tool_fn(context)
                elapsed = (datetime.now() - start_time).total_seconds() * 1000

                results.append(ToolResult(
                    tool_name=tool_name,
                    success=True,
                    data=output,
                    execution_time_ms=elapsed
                ))

                # 把輸出加入 context 給下一個工具
                context[f"{tool_name}_result"] = output

            except NotImplementedError as e:
                results.append(ToolResult(
                    tool_name=tool_name,
                    success=False,
                    error=f"⚠️ 工具 '{tool_name}' 架構已就緒但尚未實作: {str(e)}"
                ))
            except Exception as e:
                results.append(ToolResult(
                    tool_name=tool_name,
                    success=False,
                    error=str(e)
                ))

        return results

    # ============================================================
    # 結果整合 — 把所有模型的輸出合成最終回答
    # ============================================================
    def _synthesize_response(
        self,
        intent: IntentType,
        tool_results: List[ToolResult],
        user_input: str
    ) -> AgentResponse:
        """
        將多個模型的輸出整合成一個連貫的回答

        ⚠️ 當 FM (LLM) 可用時：
            把所有模型輸出交給 LLM，讓它產生自然語言回答
        ⚠️ 當 FM 不可用時：
            使用模板引擎產生結構化回答（降級模式）
        """
        successful_tools = [r for r in tool_results if r.success]
        failed_tools = [r for r in tool_results if not r.success]

        # 收集數據
        data_attachments = {}
        for r in successful_tools:
            if r.data is not None:
                data_attachments[r.tool_name] = r.data

        # 收集警告
        warnings = [r.error for r in failed_tools if r.error]

        # === 降級模式：模板回答 ===
        if not self._fm_available:
            answer = self._template_response(intent, successful_tools)
        else:
            # === FM 模式：LLM 自然語言生成 ===
            # ⚠️ TODO: 買家在 FM 微調完成後，此處會呼叫 LLM 生成回答
            answer = self._template_response(intent, successful_tools)

        return AgentResponse(
            answer=answer,
            confidence=len(successful_tools) / max(len(tool_results), 1),
            tools_used=[r.tool_name for r in successful_tools],
            data_attachments=data_attachments,
            warnings=warnings,
            timestamp=datetime.now().isoformat()
        )

    def _template_response(
        self,
        intent: IntentType,
        results: List[ToolResult]
    ) -> str:
        """降級模式的模板回答（不需要 LLM）"""
        tool_names = [r.tool_name for r in results]

        templates = {
            IntentType.HOTSPOT_QUERY:
                f"已使用 {len(results)} 個模型分析漁場熱點。"
                f"使用的模型：{', '.join(tool_names)}。"
                f"詳細座標請查看 data_attachments。",
            IntentType.ROUTE_PLANNING:
                f"航線規劃完成。使用了 {', '.join(tool_names)}。",
            IntentType.WEATHER_CHECK:
                f"天氣分析完成。使用了 {', '.join(tool_names)}。",
        }

        return templates.get(
            intent,
            f"分析完成。使用了 {len(results)} 個工具: {', '.join(tool_names)}。"
        )

    # ============================================================
    # 主入口 — 處理船長的請求
    # ============================================================
    def process_request(
        self,
        user_input: str,
        context: Optional[Dict] = None
    ) -> AgentResponse:
        """
        處理船長的請求（Agent 的主入口）

        ⚠️ 這是整個 OceanMaster 系統的最高層 API。
        ⚠️ 所有的 ML/DL/RL/FM/RAG/PINN/GA 都在這裡被統一調度。

        流程：
            1. 意圖識別：船長想做什麼？
            2. 工具規劃：需要呼叫哪些模型？
            3. 工具執行：依序跑完所有模型
            4. 結果整合：合成最終回答

        Args:
            user_input: 船長的自然語言輸入
            context: 額外的情境資訊（日期、位置、物種等）

        Returns:
            AgentResponse 包含回答、信心度、使用的工具等

        ⚠️ 使用範例：
            agent = FishingAIAgent()
            # ... 註冊所有工具 ...
            response = agent.process_request("今天北緯 15 度有沒有黃鰭鮪？")
            print(response.answer)
            print(response.data_attachments)  # 熱點座標
        """
        context = context or {}
        context["user_input"] = user_input
        context["timestamp"] = datetime.now().isoformat()

        # Step 1: 意圖識別
        intent = self._classify_intent(user_input)
        logger.info(f"意圖識別: {intent.value}")

        # Step 2: 規劃工具鏈
        tool_chain = self._plan_tool_chain(intent)
        logger.info(f"工具鏈規劃: {tool_chain}")

        # Step 3: 執行工具鏈
        tool_results = self._execute_tool_chain(tool_chain, context)

        # Step 4: 整合結果
        response = self._synthesize_response(intent, tool_results, user_input)

        # 記錄對話歷史
        self._conversation_history.append({
            "input": user_input,
            "intent": intent.value,
            "response": response.answer,
            "tools": response.tools_used,
            "timestamp": response.timestamp
        })

        return response

    # ============================================================
    # 系統狀態報告
    # ============================================================
    def status_report(self) -> Dict:
        """
        輸出 Agent 的系統狀態報告

        買家可以呼叫此函數檢查哪些模組已就緒、哪些還缺。
        """
        return {
            "registered_tools": list(self._tools.keys()),
            "registered_tool_count": len(self._tools),
            "tool_descriptions": self._tool_descriptions,
            "fm_available": self._fm_available,
            "rag_available": self._rag_available,
            "conversation_count": len(self._conversation_history),
            "missing_critical": self._get_missing_critical(),
        }

    def _get_missing_critical(self) -> List[str]:
        """列出缺少的關鍵模組"""
        critical = {
            "unet_predictor": "U-Net 空間預測 (DL)",
            "convlstm_predictor": "ConvLSTM 時序預測 (DL)",
            "xgboost_predictor": "XGBoost 即時預測 (ML)",
            "route_planner": "RL 航線規劃",
            "safety_checker": "安全檢查模組",
            "fm_llm": "FM 語言模型 (LLM)",
            "rag_search": "RAG 知識檢索",
        }
        missing = []
        for tool_id, desc in critical.items():
            if tool_id not in self._tools:
                missing.append(f"{tool_id}: {desc}")
        return missing
