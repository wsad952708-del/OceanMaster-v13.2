"""
OceanMaster v13.2 — BaseFetcher 統一容錯基類
=============================================
Task #19: 所有外部 API 擷取均須繼承此類，透過
fetch_with_cascade() 實現瀑布式兜底 (Cascade Fallback)。

核心保證：
  1. 任何單一數據源超時 (Timeout) 絕不崩潰主執行緒
  2. 自動 exponential backoff + jitter
  3. 完整 cascade 日誌供監控面板使用
  4. httpx 連線池重用
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore

log = logging.getLogger("OceanMaster.BaseFetcher")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DataSource: 描述單一外部 API 數據來源
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class DataSource:
    """單一數據源定義"""
    name: str                                          # 顯示名稱 (e.g. "NOAA-WW3")
    url_template: Optional[str] = None                 # URL 模板 (可含 {lat_min} 等)
    parser: Optional[Callable] = None                  # 解析回應的 callable
    timeout: int = 30                                  # 單次請求超時 (秒)
    mirrors: Optional[List[str]] = None                # 鏡像 URL 列表
    is_climatology: bool = False                       # 是否為氣候態兜底
    headers: Optional[Dict[str, str]] = None           # 額外 HTTP headers


@dataclass
class CascadeAttempt:
    """單次 cascade 嘗試的紀錄"""
    source_name: str
    url: str = ""
    status: str = "pending"        # success | failed | timeout | skipped
    latency_ms: float = 0.0
    error: str = ""
    timestamp: str = ""
    retry_count: int = 0


@dataclass
class CascadeResult:
    """fetch_with_cascade 的回傳結果"""
    data: Any = None                    # 解析後的數據 (dict / np.ndarray)
    source_used: str = "NONE"           # 實際命中的數據源名稱
    attempts: List[CascadeAttempt] = field(default_factory=list)
    total_latency_ms: float = 0.0
    fallback_used: bool = False         # 是否使用了非主源

    @property
    def success(self) -> bool:
        return self.data is not None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  BaseFetcher: 統一容錯基類
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BaseFetcher:
    """
    所有外部 API 擷取的基類。
    
    使用方式:
        class MyFetcher(BaseFetcher):
            async def fetch_data(self, lat_range, lon_range):
                sources = [
                    DataSource("主源", url_template="...", parser=self._parse_main),
                    DataSource("備源", url_template="...", parser=self._parse_backup),
                    DataSource("氣候態", is_climatology=True, parser=self._clim),
                ]
                result = await self.fetch_with_cascade(sources)
                return result.data
    """

    def __init__(self):
        self._cascade_log: List[CascadeAttempt] = []   # 全域日誌 (供 Task #20)
        self._stats: Dict[str, Dict] = {}              # 每個 source 的統計
        # [v13.2-R6] 共用連線池 — 避免每次請求都重建 TLS/DNS
        self._shared_client: Optional[Any] = None

    async def fetch_with_cascade(
        self,
        sources: List[DataSource],
        global_timeout: int = 90,
        max_retries: int = 2,
        context: Optional[Dict[str, Any]] = None,
    ) -> CascadeResult:
        """
        瀑布式兜底機制:
          1. 依序嘗試 sources 列表
          2. 每個 source 內有 max_retries 次重試 + exponential backoff
          3. 全域 timeout 保護 — 超過 global_timeout 秒強制中斷
          4. 全部失敗 → CascadeResult(data=None)
          5. 所有嘗試結果記錄到 _cascade_log

        Args:
            sources: 依優先順序排列的數據源列表
            global_timeout: 整體最大等待時間 (秒)
            max_retries: 每個 source 的最大重試次數
            context: URL 模板的填充變數 (lat_min, lon_min, etc.)

        Returns:
            CascadeResult with .data, .source_used, .attempts
        """
        ctx = context or {}
        result = CascadeResult()
        t0_global = time.monotonic()

        for src in sources:
            # 全域 timeout 檢查
            elapsed = (time.monotonic() - t0_global) * 1000
            if elapsed > global_timeout * 1000:
                log.warning(f"  Cascade global timeout ({global_timeout}s) reached "
                            f"after trying {len(result.attempts)} sources")
                break

            # 氣候態兜底 — 直接呼叫 parser，不做 HTTP
            if src.is_climatology:
                attempt = CascadeAttempt(
                    source_name=src.name,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                try:
                    if src.parser:
                        data = src.parser(ctx) if not asyncio.iscoroutinefunction(src.parser) \
                            else await src.parser(ctx)
                        if data is not None:
                            attempt.status = "success"
                            attempt.latency_ms = (time.monotonic() - t0_global) * 1000
                            result.data = data
                            result.source_used = src.name
                            result.fallback_used = True
                            result.attempts.append(attempt)
                            self._record_attempt(attempt)
                            log.info(f"  ✅ Cascade: {src.name} (climatology fallback)")
                            break
                except Exception as e:
                    attempt.status = "failed"
                    attempt.error = str(e)
                result.attempts.append(attempt)
                self._record_attempt(attempt)
                continue

            # 構建 URL 列表 (mirrors + url_template)
            urls = self._build_urls(src, ctx)
            if not urls:
                attempt = CascadeAttempt(
                    source_name=src.name, status="skipped",
                    error="no URL available",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                result.attempts.append(attempt)
                self._record_attempt(attempt)
                continue

            # 重試迴圈
            for url in urls:
                for retry in range(max_retries + 1):
                    # 全域 timeout 再次檢查
                    elapsed_ms = (time.monotonic() - t0_global) * 1000
                    if elapsed_ms > global_timeout * 1000:
                        break

                    remaining_s = max(5, global_timeout - elapsed_ms / 1000)
                    effective_timeout = min(src.timeout, remaining_s)

                    attempt = CascadeAttempt(
                        source_name=src.name,
                        url=url[:120],  # 截斷日誌
                        retry_count=retry,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                    t0 = time.monotonic()

                    try:
                        data = await asyncio.wait_for(
                            self._do_fetch(url, src, ctx, effective_timeout),
                            timeout=effective_timeout + 2,
                        )
                        latency = (time.monotonic() - t0) * 1000
                        attempt.latency_ms = latency

                        if data is not None:
                            attempt.status = "success"
                            result.data = data
                            result.source_used = src.name
                            result.fallback_used = (src != sources[0])
                            result.attempts.append(attempt)
                            result.total_latency_ms = (time.monotonic() - t0_global) * 1000
                            self._record_attempt(attempt)
                            log.info(f"  ✅ Cascade: {src.name} ({latency:.0f}ms)")
                            return result
                        else:
                            attempt.status = "failed"
                            attempt.error = "parser returned None"

                    except asyncio.TimeoutError:
                        attempt.status = "timeout"
                        attempt.error = f"timeout after {effective_timeout:.0f}s"
                        attempt.latency_ms = (time.monotonic() - t0) * 1000
                        log.debug(f"  ⏱️ {src.name} timeout ({effective_timeout:.0f}s)")

                    except Exception as e:
                        attempt.status = "failed"
                        attempt.error = str(e)[:200]
                        attempt.latency_ms = (time.monotonic() - t0) * 1000
                        log.debug(f"  ❌ {src.name}: {e}")

                    result.attempts.append(attempt)
                    self._record_attempt(attempt)

                    # Exponential backoff + jitter (僅在非最後一次重試)
                    if retry < max_retries:
                        backoff = min(2 ** retry * 1.0 + random.uniform(0, 1), 10.0)
                        await asyncio.sleep(backoff)

        result.total_latency_ms = (time.monotonic() - t0_global) * 1000
        if result.data is None:
            log.warning(f"  ⚠️ Cascade: ALL {len(sources)} sources failed "
                        f"({result.total_latency_ms:.0f}ms)")
        return result

    async def _do_fetch(
        self,
        url: str,
        source: DataSource,
        context: Dict[str, Any],
        timeout: float,
    ) -> Any:
        """
        執行單次 HTTP GET + 解析。子類可覆寫。
        [v13.2-R6] 使用共用連線池 — 避免每次請求都 DNS + TLS
        """
        if httpx is None:
            import requests
            resp = requests.get(
                url,
                timeout=min(timeout, 60),
                headers=source.headers or {},
            )
            if resp.status_code != 200:
                return None
            text = resp.text
        else:
            # 延遲建立共用 client（首次請求時）
            if self._shared_client is None:
                self._shared_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(timeout, connect=10),
                    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                    follow_redirects=True,
                )
            resp = await self._shared_client.get(
                url,
                headers=source.headers or {},
                timeout=timeout,
            )
            if resp.status_code != 200:
                return None
            text = resp.text

        # 呼叫解析器
        if source.parser:
            if asyncio.iscoroutinefunction(source.parser):
                return await source.parser(text, context)
            return source.parser(text, context)

        return text

    def _build_urls(
        self, source: DataSource, context: Dict[str, Any]
    ) -> List[str]:
        """
        從 DataSource 構建所有可嘗試的 URL。
        支援 mirrors + url_template 變數替換。
        """
        urls = []
        if source.mirrors:
            for mirror in source.mirrors:
                if source.url_template:
                    try:
                        url = source.url_template.format(base=mirror, **context)
                        urls.append(url)
                    except KeyError:
                        urls.append(mirror)
                else:
                    urls.append(mirror)
        elif source.url_template:
            try:
                urls.append(source.url_template.format(**context))
            except KeyError as e:
                log.debug(f"  URL template missing key: {e}")
        return urls

    def _record_attempt(self, attempt: CascadeAttempt):
        """記錄到全域日誌 + 更新統計"""
        self._cascade_log.append(attempt)
        # 保留最近 500 筆
        if len(self._cascade_log) > 500:
            self._cascade_log = self._cascade_log[-500:]

        # 統計
        name = attempt.source_name
        if name not in self._stats:
            self._stats[name] = {
                "total": 0, "success": 0, "failed": 0,
                "timeout": 0, "avg_latency_ms": 0.0,
            }
        s = self._stats[name]
        s["total"] += 1
        s[attempt.status] = s.get(attempt.status, 0) + 1
        # Running average latency
        n = s["total"]
        s["avg_latency_ms"] = (s["avg_latency_ms"] * (n - 1) + attempt.latency_ms) / n

    def get_health_report(self) -> Dict[str, Any]:
        """
        回傳所有數據源的健康狀況 (供 Task #20 Dashboard)。
        """
        report = {}
        for name, s in self._stats.items():
            total = max(s["total"], 1)
            report[name] = {
                "success_rate": round(s.get("success", 0) / total * 100, 1),
                "timeout_rate": round(s.get("timeout", 0) / total * 100, 1),
                "avg_latency_ms": round(s["avg_latency_ms"], 0),
                "total_calls": total,
            }
        return report

    def get_recent_log(self, n: int = 50) -> List[Dict]:
        """回傳最近 N 筆 cascade 日誌"""
        return [
            {
                "source": a.source_name,
                "status": a.status,
                "latency_ms": round(a.latency_ms, 0),
                "error": a.error,
                "url": a.url,
                "time": a.timestamp,
            }
            for a in self._cascade_log[-n:]
        ]
