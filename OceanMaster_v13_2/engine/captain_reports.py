"""
OceanMaster v13.2 — AIS 漁船活動 + 船長回報系統
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

    # [v13.2-audit] 允許的欄位白名單
    ALLOWED_FIELDS = {
        "lat", "lon", "species", "catch_kg", "catch_mt",
        "sst_observed", "notes", "gear_type", "vessel_name",
        "date", "wind_speed", "wave_height", "bycatch",
    }
    MAX_REPORT_SIZE = 50  # 最多 50 個欄位

    def add_report(self, report: Dict[str, Any]) -> Dict:
        # [v13.2-audit] 輸入驗證
        if not isinstance(report, dict):
            return {"status": "error", "message": "report must be a dict"}
        if len(report) > self.MAX_REPORT_SIZE:
            return {"status": "error", "message": f"too many fields (max {self.MAX_REPORT_SIZE})"}

        # 過濾未知欄位
        filtered = {k: v for k, v in report.items() if k in self.ALLOWED_FIELDS}
        if not filtered:
            return {"status": "error", "message": "no valid fields provided"}

        # 基本型別檢查
        if "lat" in filtered and not isinstance(filtered["lat"], (int, float)):
            return {"status": "error", "message": "lat must be numeric"}
        if "lon" in filtered and not isinstance(filtered["lon"], (int, float)):
            return {"status": "error", "message": "lon must be numeric"}
        if "catch_kg" in filtered and not isinstance(filtered["catch_kg"], (int, float)):
            return {"status": "error", "message": "catch_kg must be numeric"}

        filtered["timestamp"] = datetime.now(timezone.utc).isoformat()
        filtered["id"] = len(self.reports)
        self.reports.append(filtered)
        self._save()
        return {"status": "success", "report_id": filtered["id"]}

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
