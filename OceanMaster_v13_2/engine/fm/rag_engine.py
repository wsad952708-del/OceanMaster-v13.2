"""
RAG — Retrieval-Augmented Generation 檢索增強生成引擎

============================================================
🎯 功能說明：
    讓 AI Agent 的 LLM 大腦擁有「長期記憶」。
    當船長問 AI：「上次聖嬰年我們在帛琉抓到多少？」
    RAG 引擎會：
    1. 把問題轉成向量 (Embedding)
    2. 在向量資料庫中搜尋最相關的歷史記錄
    3. 把搜尋到的資料餵給 LLM
    4. LLM 結合搜尋結果給出精準回答

    沒有 RAG，LLM 就是一個「失憶的天才」。
    有了 RAG，LLM 變成「能翻資料庫的專家」。

📌 架構狀態：✅ 完整架構  |  ❌ 尚未建立知識庫
📌 缺少什麼：
    1. 歷史漁獲記錄文件 (CSV / Excel / PDF)
    2. 漁業法規文件 (WCPFC 規範等)
    3. 過去的衛星分析報告
📌 買家需要：
    1. 安裝向量資料庫: pip install chromadb sentence-transformers
    2. 將文件倒入 rag_engine.ingest_documents()
    3. 在 AI Agent 中呼叫 rag_engine.query() 進行檢索

📌 向量資料庫選擇：
    - ChromaDB: 輕量級，適合本地部署（推薦）
    - FAISS: Facebook 的高效向量搜尋（適合大規模）
    - Qdrant: 生產級向量資料庫（適合雲端）
============================================================
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """
    文件物件 — 存入向量資料庫的最小單位

    每個 Document 代表一段可被檢索的知識片段。
    例如：一條歷史漁獲記錄、一段法規條文、一份衛星報告摘要。
    """
    content: str                    # 文件內容（純文字）
    metadata: Dict[str, Any]        # 附加資訊（日期、來源、物種等）
    source: str = ""                # 資料來源（檔案名稱）
    doc_id: str = ""                # 唯一識別碼

    # === 範例 ===
    # content: "2023年8月15日，船隊A在北緯15.3度、東經135.7度捕獲黃鰭鮪3.2噸..."
    # metadata: {"date": "2023-08-15", "species": "yellowfin", "catch_tons": 3.2}
    # source: "catch_log_2023.csv"


@dataclass
class RetrievalResult:
    """檢索結果"""
    document: Document              # 找到的文件
    relevance_score: float          # 相關性分數 (0~1)
    rank: int                       # 排名


class RAGEngine:
    """
    RAG 檢索增強生成引擎

    ⚠️ 架構狀態：骨架已完成，尚未建立知識庫
    ⚠️ 買家使用流程：
        1. 初始化 RAGEngine
        2. 呼叫 ingest_documents() 倒入所有歷史文件
        3. 在 AI Agent 中呼叫 query() 進行檢索
        4. 把檢索結果餵給 LLM 生成回答

    ⚠️ 依賴套件：
        pip install chromadb sentence-transformers
    """

    # === 支援的嵌入模型 ===
    EMBEDDING_MODELS = {
        "bge-base-zh": {
            "hf_name": "BAAI/bge-base-zh-v1.5",
            "dim": 768,
            "description": "中文最強嵌入模型，適合漁業中文文件"
        },
        "multilingual-e5": {
            "hf_name": "intfloat/multilingual-e5-large",
            "dim": 1024,
            "description": "多語言嵌入模型，中英日文通吃"
        },
    }

    def __init__(
        self,
        collection_name: str = "oceanmaster_knowledge",
        embedding_model: str = "bge-base-zh",
        persist_directory: str = "./rag_database",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 5
    ):
        """
        Args:
            collection_name: 向量集合名稱
            embedding_model: 嵌入模型代號 (見 EMBEDDING_MODELS)
            persist_directory: 向量資料庫的本地儲存路徑
            chunk_size: 文件切割大小 (字元數)
            chunk_overlap: 文件切割重疊 (字元數)
            top_k: 預設檢索返回的文件數量
        """
        self.collection_name = collection_name
        self.embedding_config = self.EMBEDDING_MODELS.get(embedding_model, {})
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        self._db = None
        self._embedder = None
        self._document_count = 0

        logger.info(
            f"RAG 引擎初始化 | collection={collection_name}, "
            f"embedding={embedding_model}, top_k={top_k}"
        )

    def _init_database(self):
        """
        初始化向量資料庫

        ⚠️ 需要安裝: pip install chromadb
        """
        raise NotImplementedError(
            "⚠️ 向量資料庫初始化需要安裝 chromadb。\n"
            "買家請執行: pip install chromadb sentence-transformers\n"
            "然後在此函數中實作 ChromaDB 的初始化。"
        )

    def _chunk_text(self, text: str) -> List[str]:
        """
        將長文件切割成小片段

        切割策略：固定大小 + 重疊，確保語意完整性。
        例如：一份 2000 字的報告被切成 4 個 500 字的片段，
        每個片段跟下一個有 50 字的重疊。
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def ingest_documents(
        self,
        documents: List[Document]
    ) -> int:
        """
        將文件倒入向量資料庫

        ⚠️ 買家使用流程：
            1. 把歷史漁獲日誌轉成 Document 物件列表
            2. 呼叫此函數建立知識庫

        ⚠️ 範例：
            docs = [
                Document(
                    content="2023年8月，黃鰭鮪主要分布在...",
                    metadata={"year": 2023, "species": "yellowfin"},
                    source="historical_analysis.pdf"
                ),
            ]
            count = rag.ingest_documents(docs)

        Args:
            documents: 文件列表

        Returns:
            成功匯入的文件片段數量
        """
        total_chunks = 0
        for doc in documents:
            chunks = self._chunk_text(doc.content)
            total_chunks += len(chunks)
            # ⚠️ TODO: 將每個 chunk 嵌入向量空間並存入 ChromaDB
            logger.info(
                f"文件 '{doc.source}' 切割為 {len(chunks)} 個片段"
            )

        self._document_count += total_chunks
        logger.info(f"共匯入 {total_chunks} 個文件片段，總計 {self._document_count} 個")
        return total_chunks

    def ingest_csv(
        self,
        csv_path: str,
        content_columns: List[str],
        metadata_columns: Optional[List[str]] = None
    ) -> int:
        """
        從 CSV 檔案匯入文件

        ⚠️ 專為漁獲日誌設計：
            rag.ingest_csv(
                csv_path="catch_logs.csv",
                content_columns=["日期", "位置", "漁獲量", "備註"],
                metadata_columns=["物種", "船名"]
            )

        Args:
            csv_path: CSV 檔案路徑
            content_columns: 要合併為文件內容的欄位名稱
            metadata_columns: 要作為 metadata 的欄位名稱

        Returns:
            匯入的文件數量
        """
        raise NotImplementedError(
            "⚠️ CSV 匯入需要 pandas。\n"
            "買家請執行: pip install pandas\n"
            "然後將漁獲日誌 CSV 檔案路徑傳入此函數。"
        )

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """
        檢索與問題最相關的文件片段

        ⚠️ 這是 AI Agent 呼叫的核心函數。
        ⚠️ Agent 會把檢索結果塞進 LLM 的 Prompt 中：
            results = rag.query("上次聖嬰年帛琉的漁獲量？")
            context = "\\n".join([r.document.content for r in results])
            llm_prompt = f"根據以下資料回答問題：\\n{context}\\n\\n問題：..."

        Args:
            question: 使用者的問題（自然語言）
            top_k: 返回的文件數量（預設使用初始化時的設定）
            filter_metadata: 過濾條件（例如 {"species": "yellowfin"}）

        Returns:
            排序後的檢索結果列表
        """
        k = top_k or self.top_k

        raise NotImplementedError(
            "⚠️ 查詢需要已建立的向量資料庫。\n"
            "買家請先執行 ingest_documents() 匯入文件，\n"
            "然後再呼叫此函數進行檢索。\n"
            f"目前資料庫中有 {self._document_count} 個文件片段。"
        )

    def build_context_prompt(
        self,
        question: str,
        results: List[RetrievalResult],
        max_context_length: int = 3000
    ) -> str:
        """
        將檢索結果組裝成 LLM 可用的 Context Prompt

        Args:
            question: 原始問題
            results: 檢索結果
            max_context_length: 最大上下文長度

        Returns:
            組裝好的 Prompt 字串
        """
        context_parts = []
        current_length = 0

        for r in results:
            if current_length + len(r.document.content) > max_context_length:
                break
            context_parts.append(
                f"[來源: {r.document.source} | 相關度: {r.relevance_score:.2f}]\n"
                f"{r.document.content}"
            )
            current_length += len(r.document.content)

        context = "\n\n---\n\n".join(context_parts)

        prompt = (
            f"你是一位資深的遠洋漁業 AI 專家。\n"
            f"請根據以下檢索到的歷史資料，回答使用者的問題。\n"
            f"如果資料不足以回答，請明確說明。\n\n"
            f"=== 檢索到的相關資料 ===\n{context}\n\n"
            f"=== 使用者問題 ===\n{question}\n\n"
            f"請回答："
        )

        return prompt

    @property
    def document_count(self) -> int:
        return self._document_count
