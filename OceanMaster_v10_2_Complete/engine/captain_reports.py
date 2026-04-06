"""
OceanMaster v9.0 — AIS 漁船活動 + 船長回報系統
================================================
"""

import numpy as np
import logging
import json
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

log = logging.getLogger("OceanMaster.AIS")


class CaptainReportSystem:
    """船長回報學習系統"""

    def __init__(self, db_path: str = "data/captain_reports.json"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.reports: List[Dict] = []
        self._load()

    def add_report(self, report: Dict[str, Any]) -> Dict:
        report["timestamp"] = datetime.now(timezone.utc).isoformat()
        report["id"] = len(self.reports)
        self.reports.append(report)
        self._save()
        return {"status": "success", "report_id": report["id"]}

    def get_reports(self, species: Optional[str] = None, days_back: int = 90) -> List[Dict]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
        filtered = [r for r in self.reports if r.get("timestamp", "") >= cutoff]
        if species:
            filtered = [r for r in filtered if r.get("species") == species]
        return filtered

    def _save(self):
        try:
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(self.reports, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.error(f"回報保存失敗: {e}")

    def _load(self):
        if self.db_path.exists():
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    self.reports = json.load(f)
            except Exception:
                self.reports = []
