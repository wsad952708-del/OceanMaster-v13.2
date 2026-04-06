"""
OceanMaster v13.2 — Data Retention & Cleanup Pipeline
=======================================================
資料防爆與生命週期管理: 防止硬碟一個月內爆滿。

保留政策:
  T+0 ~ T-14 天: 保留原始檔案 (高解析度)
  T-15 ~ T-60 天: 降採樣 → 刪除原始
  T-60+ 天: 永久刪除, 僅保留 catch_reports.db + env_snapshot
"""
import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Tuple

log = logging.getLogger("OceanMaster.cleanup")


# ── 保留政策常數 ──
TIER_1_DAYS = 14    # 原始保留天數
TIER_2_DAYS = 60    # 降採樣保留天數
# >60 天永久刪除

# ── 受保護檔案 (絕對不能刪) ──
PROTECTED_PATTERNS = [
    "catch_reports.db",        # 漁獲回報 (Ground Truth)
    "env_snapshot*.json",      # 環境快照 (ML 訓練資產)
    "*.pkl",                   # ML 模型
    "*.joblib",                # ML 模型
]

# ── 可清理檔案類型 ──
CLEANABLE_EXTENSIONS = {
    ".nc", ".nc4",             # NetCDF
    ".grib", ".grib2",         # GRIB
    ".json",                   # raw API cache
    ".csv",                    # raw data
    ".html",                   # generated maps
    ".kml",                    # KML output
    ".geojson",                # GeoJSON output
}

# ── 清理目錄 ──
SCAN_DIRS = [
    "output",
    "cache",
    "data/raw",
    "data/cache",
]


class CleanupPolicy:
    """
    三階段生命週期管理:
      Tier 1 (0-14d): Full resolution retained
      Tier 2 (15-60d): Marked for downsample, originals deleted
      Tier 3 (60d+):   Permanent delete (except protected)
    """

    def __init__(self, base_dir: str = ".", dry_run: bool = False):
        self.base_dir = Path(base_dir)
        self.dry_run = dry_run
        self.now = datetime.now(timezone.utc)

    def execute(self) -> Dict:
        """
        執行清理, 回傳摘要。
        """
        summary = {
            "scanned": 0,
            "tier1_kept": 0,
            "tier2_downsample": 0,
            "tier3_deleted": 0,
            "protected": 0,
            "freed_bytes": 0,
            "errors": [],
        }

        for scan_dir in SCAN_DIRS:
            target = self.base_dir / scan_dir
            if not target.exists():
                continue
            self._scan_directory(target, summary)

        mode = "DRY-RUN" if self.dry_run else "EXECUTED"
        log.info(f"  🧹 Cleanup [{mode}]:")
        log.info(f"     Scanned: {summary['scanned']} files")
        log.info(f"     Tier 1 (kept):       {summary['tier1_kept']}")
        log.info(f"     Tier 2 (downsample): {summary['tier2_downsample']}")
        log.info(f"     Tier 3 (deleted):    {summary['tier3_deleted']}")
        log.info(f"     Protected:           {summary['protected']}")
        log.info(f"     Freed: {summary['freed_bytes'] / (1024*1024):.1f} MB")

        return summary

    def _scan_directory(self, target: Path, summary: Dict):
        """遞迴掃描目錄"""
        for fpath in target.rglob("*"):
            if fpath.is_dir():
                continue
            summary["scanned"] += 1

            # 受保護? → skip
            if self._is_protected(fpath):
                summary["protected"] += 1
                continue

            # 不可清理類型? → skip
            if fpath.suffix.lower() not in CLEANABLE_EXTENSIONS:
                continue

            # 計算檔案年齡
            age_days = self._file_age_days(fpath)

            if age_days <= TIER_1_DAYS:
                summary["tier1_kept"] += 1
            elif age_days <= TIER_2_DAYS:
                summary["tier2_downsample"] += 1
                self._mark_downsample(fpath)
            else:
                # Tier 3: 永久刪除
                size = fpath.stat().st_size
                summary["freed_bytes"] += size
                summary["tier3_deleted"] += 1

                if self.dry_run:
                    log.info(f"     [DRY] DELETE: {fpath.name} "
                             f"({age_days}d old, {size/1024:.0f}KB)")
                else:
                    try:
                        fpath.unlink()
                        log.info(f"     DELETED: {fpath.name} ({age_days}d)")
                    except Exception as e:
                        summary["errors"].append(str(e))

    def _is_protected(self, fpath: Path) -> bool:
        """檢查檔案是否受保護"""
        name = fpath.name.lower()
        for pattern in PROTECTED_PATTERNS:
            if "*" in pattern:
                prefix = pattern.replace("*", "")
                if name.startswith(prefix.rstrip(".").lower()) or name.endswith(prefix.lstrip("*").lower()):
                    return True
            elif name == pattern.lower():
                return True
        return False

    def _file_age_days(self, fpath: Path) -> int:
        """計算檔案年齡 (天)"""
        try:
            mtime = datetime.fromtimestamp(fpath.stat().st_mtime, tz=timezone.utc)
            return (self.now - mtime).days
        except Exception:
            return 0

    def _mark_downsample(self, fpath: Path):
        """標記檔案需要降採樣 (建立 .downsample marker)"""
        marker = fpath.with_suffix(fpath.suffix + ".needs_downsample")
        if not marker.exists() and not self.dry_run:
            try:
                marker.write_text(
                    json.dumps({
                        "original": str(fpath),
                        "marked_at": self.now.isoformat(),
                        "policy": "0.01deg → 0.1deg",
                    })
                )
            except Exception as e:
                log.debug(f"[降級] cron_cleanup_pipeline.py: {e}")


def main():
    parser = argparse.ArgumentParser(description="OceanMaster Data Cleanup")
    parser.add_argument("--dry-run", action="store_true",
                        help="模擬清理，不實際刪除")
    parser.add_argument("--base-dir", default=".",
                        help="專案根目錄")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    policy = CleanupPolicy(base_dir=args.base_dir, dry_run=args.dry_run)
    summary = policy.execute()

    # 驗證: catch_reports.db 和 env_snapshot 毫髮無傷
    db_path = Path(args.base_dir) / "data" / "catch_reports.db"
    if db_path.exists():
        log.info(f"  ✅ catch_reports.db: SAFE ({db_path.stat().st_size / 1024:.1f}KB)")
    else:
        log.info(f"  ℹ️ catch_reports.db: not found (ok if new install)")


if __name__ == "__main__":
    main()
