"""
OceanMaster v8.0 — 排程器
===========================
APScheduler 自動排程，每天凌晨 2 點執行完整分析管線

排程邏輯：
  02:00 → 抓取全部數據
  02:10 → 物理演算法（鋒面、FTLE、溫躍層）
  02:20 → HSI 棲地適合度
  02:25 → AI 融合 + 熱點排名
  02:30 → 生成 KML → 準備傳送

另外每 6 小時快速更新一次 SST + 海流
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("OceanMaster.Scheduler")


class OceanMasterScheduler:
    """排程器：管理數據抓取和分析的自動執行"""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self._scheduler = None

    def start(self):
        """啟動排程器"""
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from apscheduler.triggers.cron import CronTrigger
            from apscheduler.triggers.interval import IntervalTrigger

            self._scheduler = AsyncIOScheduler(timezone="UTC")

            # 每天凌晨 02:00 UTC 完整分析
            self._scheduler.add_job(
                self._run_full_pipeline,
                CronTrigger(hour=2, minute=0),
                id="daily_full",
                name="每日完整分析",
                misfire_grace_time=3600,
            )

            # 每 6 小時快速更新
            self._scheduler.add_job(
                self._run_quick_update,
                IntervalTrigger(hours=6),
                id="quick_update",
                name="快速更新 (SST+海流)",
                misfire_grace_time=1800,
            )

            self._scheduler.start()
            log.info("✅ 排程器已啟動")
            log.info("  📅 每日 02:00 UTC: 完整分析")
            log.info("  🔄 每 6 小時: 快速更新")

        except ImportError:
            log.warning("APScheduler 未安裝，排程功能停用")
            log.warning("安裝方式: pip install apscheduler")

    def stop(self):
        if self._scheduler:
            self._scheduler.shutdown()
            log.info("排程器已停止")

    async def _run_full_pipeline(self):
        """完整管線：抓取 → 運算 → 生成 KML"""
        log.info("=" * 60)
        log.info("⏰ 排程觸發：完整分析管線")
        log.info("=" * 60)
        try:
            result = await self.pipeline.run_full()
            log.info(f"✅ 完整分析完成: {result.get('kml_path', 'N/A')}")
        except Exception as e:
            log.error(f"❌ 完整分析失敗: {e}", exc_info=True)

    async def _run_quick_update(self):
        """快速更新：只抓 SST + 海流，重算鋒面"""
        log.info("🔄 排程觸發：快速更新")
        try:
            result = await self.pipeline.run_quick()
            log.info(f"✅ 快速更新完成")
        except Exception as e:
            log.error(f"❌ 快速更新失敗: {e}", exc_info=True)
