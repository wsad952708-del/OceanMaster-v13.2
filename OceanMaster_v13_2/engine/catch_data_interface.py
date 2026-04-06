"""
OceanMaster v13.2 — 漁業歷史捕獲數據接口 (Catch Data Interface)
===============================================================
預留與漁業公司對接的標準化接口。

支援格式:
  - CSV: 日期,經度,緯度,魚種,漁獲量(kg),漁法,漁船ID
  - 台灣漁業署 VDR 格式 (未來擴充)
  - 手動回報 (船長日誌)

用途:
  1. 回測 (backtest) HSI 預測 vs 實際漁獲
  2. 校準物種參數 (SST_optimal, CHL_optimal)
  3. 訓練 ML 模型的真實標籤
  4. 計算歷史漁場先驗 (historical_prior)

Schema:
  catch_record = {
    "date": "2025-06-15",
    "lat": 22.5,
    "lon": 121.3,
    "species": "yellowfin",
    "catch_kg": 150.0,
    "fishing_method": "longline",
    "vessel_id": "CT3-1234",
    "sst_observed": 28.5,       # optional
    "notes": "黑潮邊緣, 鋒面附近",  # optional
  }
"""

import json
import csv
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

log = logging.getLogger("OceanMaster.CatchData")

# ─── 標準 Schema ─────────────────────────────────────
REQUIRED_FIELDS = ["date", "lat", "lon", "species", "catch_kg"]
OPTIONAL_FIELDS = [
    "fishing_method", "vessel_id", "sst_observed", "chl_observed",
    "depth_m", "hook_depth_m", "effort_hours", "notes",
]
SUPPORTED_SPECIES = [
    "yellowfin", "bigeye", "skipjack", "albacore",
    "mahi_mahi", "blue_marlin", "mackerel_scad", "pacific_saury",
    "neon_flying_squid", "japanese_flying_squid",
    "squid_todarodes", "squid_ommastrephes",
    "swordfish", "striped_marlin", "black_marlin",
]
SUPPORTED_METHODS = [
    "longline", "purse_seine", "trolling", "squid_jigging",
    "stick_held_dip_net", "gillnet", "pole_and_line", "driftnet", "unknown",
]

# ─── WCPFC 欄位映射 ─────────────────────────────────────
_WCPFC_DIR = Path("data/wcpfc")
_WCPFC_SPECIES_MAP = {
    # LONGLINE
    "yft_c": "yellowfin", "bet_c": "bigeye", "alb_c": "albacore",
    "swo_c": "swordfish", "mls_c": "striped_marlin",
    "blm_c": "black_marlin", "bum_c": "blue_marlin",
    # PURSE_SEINE (aggregate all set types)
    "skj_c_una": "skipjack", "yft_c_una": "yellowfin", "bet_c_una": "bigeye",
    "skj_c_log": "skipjack", "yft_c_log": "yellowfin", "bet_c_log": "bigeye",
    "skj_c_dfad": "skipjack", "yft_c_dfad": "yellowfin", "bet_c_dfad": "bigeye",
    "skj_c_afad": "skipjack", "yft_c_afad": "yellowfin", "bet_c_afad": "bigeye",
    "skj_c_oth": "skipjack", "yft_c_oth": "yellowfin", "bet_c_oth": "bigeye",
    # POLE_AND_LINE
    "skj_c": "skipjack", "yft_c": "yellowfin",
}


def _parse_coord(s: str) -> float:
    """Convert WCPFC lat5/lon5 (e.g. '05N', '140E', '30S', '170W') to decimal."""
    s = s.strip().strip('"')
    if not s:
        return 0.0
    direction = s[-1].upper()
    value = float(s[:-1])
    if direction in ('S', 'W'):
        value = -value
    return value


@dataclass
class CatchRecord:
    """單筆漁獲記錄"""
    date: str
    lat: float
    lon: float
    species: str
    catch_kg: float
    fishing_method: str = "unknown"
    vessel_id: str = ""
    sst_observed: Optional[float] = None
    chl_observed: Optional[float] = None
    depth_m: Optional[float] = None
    hook_depth_m: Optional[float] = None
    effort_hours: Optional[float] = None
    notes: str = ""

    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None and v != ""}

    def validate(self) -> List[str]:
        """驗證記錄完整性, 回傳錯誤列表"""
        errors = []
        try:
            datetime.strptime(self.date, "%Y-%m-%d")
        except ValueError:
            errors.append(f"Invalid date format: {self.date} (expected YYYY-MM-DD)")
        if not (-90 <= self.lat <= 90):
            errors.append(f"Invalid lat: {self.lat}")
        if not (-180 <= self.lon <= 180):
            errors.append(f"Invalid lon: {self.lon}")
        if self.species not in SUPPORTED_SPECIES:
            errors.append(f"Unknown species: {self.species}")
        if self.catch_kg < 0:
            errors.append(f"Negative catch: {self.catch_kg}")
        if self.fishing_method not in SUPPORTED_METHODS:
            errors.append(f"Unknown method: {self.fishing_method}")
        return errors


class CatchDataInterface:
    """
    漁業歷史數據管理器

    Usage:
        cdi = CatchDataInterface(data_dir="data/catch_history")
        # 匯入 CSV
        result = cdi.import_csv("high_sea_2024.csv")
        # 查詢
        records = cdi.query(species="yellowfin", lat_range=(20, 27))
        # 計算歷史先驗
        prior = cdi.compute_historical_prior(lats, lons, "yellowfin", month=6)
    """

    def __init__(self, data_dir: str = "data/catch_history"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._records: List[CatchRecord] = []
        self._wcpfc_record_count = 0  # WCPFC records (lazy, not materialized)
        self._prior_cache = None      # Precomputed LONGLINE prior JSON
        self._load_existing()
        self._load_wcpfc_csvs()

    def _load_existing(self):
        """載入已有的 JSON 紀錄"""
        json_file = self.data_dir / "catch_records.json"
        if json_file.exists():
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._records = [CatchRecord(**r) for r in data]
                log.info(f"  CatchData: loaded {len(self._records)} existing records")
            except Exception as e:
                log.warning(f"  CatchData: failed to load existing records: {e}")

    def _load_wcpfc_csvs(self):
        """[v13.2] 載入 WCPFC 官方 CSV — 只計算 record count，不展開為 CatchRecord（太大）"""
        wcpfc_dir = _WCPFC_DIR
        if not wcpfc_dir.exists():
            return

        total = 0
        files = {
            "LONGLINE.CSV": "longline",
            "PURSE_SEINE.CSV": "purse_seine",
            "POLE_AND_LINE.CSV": "pole_and_line",
            "DRIFTNET.CSV": "driftnet",
        }
        for fname, method in files.items():
            fpath = wcpfc_dir / fname
            if fpath.exists():
                n = sum(1 for _ in open(fpath, encoding="utf-8-sig")) - 1  # minus header
                total += n
                log.info(f"  CatchData: WCPFC {fname}: {n:,} rows")

        self._wcpfc_record_count = total
        if total > 0:
            log.info(f"  CatchData: WCPFC total: {total:,} aggregate records")

        # Load precomputed prior JSON
        prior_path = wcpfc_dir / "historical_prior_longline.json"
        if prior_path.exists():
            try:
                with open(prior_path, "r", encoding="utf-8") as f:
                    self._prior_cache = json.load(f)
                n_cells = sum(len(v) for month_data in self._prior_cache.get("prior", {}).values()
                              for v in month_data.values())
                log.info(f"  CatchData: loaded WCPFC prior ({n_cells:,} grid-month cells)")
            except Exception as e:
                log.warning(f"  CatchData: failed to load prior: {e}")

    def _save(self):
        """持久化到 JSON"""
        json_file = self.data_dir / "catch_records.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in self._records], f,
                      ensure_ascii=False, indent=2, default=str)

    def import_csv(self, csv_path: str, delimiter: str = ",") -> Dict:
        """
        匯入 CSV 漁獲數據

        CSV 格式:
          date,lat,lon,species,catch_kg[,fishing_method,vessel_id,...]

        Returns:
            dict with imported, errors, total
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            return {"imported": 0, "errors": [f"File not found: {csv_path}"], "total": 0}

        imported = 0
        errors = []
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row_num, row in enumerate(reader, start=2):
                try:
                    record = CatchRecord(
                        date=row.get("date", "").strip(),
                        lat=float(row.get("lat", 0)),
                        lon=float(row.get("lon", 0)),
                        species=row.get("species", "").strip().lower(),
                        catch_kg=float(row.get("catch_kg", 0)),
                        fishing_method=row.get("fishing_method", "unknown").strip().lower(),
                        vessel_id=row.get("vessel_id", "").strip(),
                        sst_observed=float(row["sst_observed"]) if "sst_observed" in row and row["sst_observed"] else None,
                        notes=row.get("notes", ""),
                    )
                    validation_errors = record.validate()
                    if validation_errors:
                        errors.append(f"Row {row_num}: {'; '.join(validation_errors)}")
                        continue
                    self._records.append(record)
                    imported += 1
                except Exception as e:
                    errors.append(f"Row {row_num}: {e}")

        if imported > 0:
            self._save()

        log.info(f"  CatchData: imported {imported} records from {csv_path.name}")
        if errors:
            log.warning(f"  CatchData: {len(errors)} errors during import")

        return {
            "imported": imported,
            "errors": errors[:20],  # 只回傳前 20 個錯誤
            "total": len(self._records),
        }

    def import_json(self, records: List[Dict]) -> Dict:
        """
        匯入 JSON 漁獲數據 (API 端點用)

        Parameters:
            records: [{date, lat, lon, species, catch_kg, ...}, ...]

        Returns:
            dict with imported, errors, total
        """
        imported = 0
        errors = []
        for i, row in enumerate(records):
            try:
                record = CatchRecord(**{k: v for k, v in row.items()
                                        if k in CatchRecord.__dataclass_fields__})
                validation_errors = record.validate()
                if validation_errors:
                    errors.append(f"Record {i}: {'; '.join(validation_errors)}")
                    continue
                self._records.append(record)
                imported += 1
            except Exception as e:
                errors.append(f"Record {i}: {e}")

        if imported > 0:
            self._save()

        return {"imported": imported, "errors": errors[:20], "total": len(self._records)}

    def query(
        self,
        species: Optional[str] = None,
        lat_range: Optional[Tuple[float, float]] = None,
        lon_range: Optional[Tuple[float, float]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict]:
        """查詢漁獲記錄"""
        results = []
        for r in self._records:
            if species and r.species != species:
                continue
            if lat_range and not (lat_range[0] <= r.lat <= lat_range[1]):
                continue
            if lon_range and not (lon_range[0] <= r.lon <= lon_range[1]):
                continue
            if date_from and r.date < date_from:
                continue
            if date_to and r.date > date_to:
                continue
            results.append(r.to_dict())
            if len(results) >= limit:
                break
        return results

    def compute_historical_prior(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        species: str,
        month: Optional[int] = None,
        sigma_deg: float = 2.5,
    ) -> np.ndarray:
        """
        [v13.2] 計算歷史漁場先驗分布 — 優先使用 WCPFC 真實資料

        Uses precomputed historical_prior_longline.json when available.
        Falls back to KDE on CatchRecord list.
        """
        ny, nx = len(lats), len(lons)
        prior = np.zeros((ny, nx), dtype=np.float32)

        # ── [v13.2] 優先用 WCPFC precomputed prior ──
        if self._prior_cache is not None:
            prior_data = self._prior_cache.get("prior", {})
            sp_data = prior_data.get(species, {})

            # Collect points for target month (or all months)
            points = []
            if month is not None:
                points = sp_data.get(str(month), [])
            else:
                for m_data in sp_data.values():
                    points.extend(m_data)

            if points:
                lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
                for pt in points:
                    p_lat, p_lon, cpue = pt["lat"], pt["lon"], pt["cpue"]
                    dist_sq = (lat_grid - p_lat) ** 2 + (lon_grid - p_lon) ** 2
                    prior += cpue * np.exp(-dist_sq / (2 * sigma_deg ** 2))

                if prior.max() > 0:
                    prior /= prior.max()

                log.info(f"  CatchData: WCPFC prior for {species} month={month}: "
                         f"{len(points)} grid cells, max={prior.max():.3f}")
                return prior

        # ── Fallback: KDE on CatchRecord list ──
        records = [r for r in self._records if r.species == species]
        if month is not None:
            records = [r for r in records
                       if datetime.strptime(r.date, "%Y-%m-%d").month == month]

        if not records:
            log.info(f"  CatchData: no records for {species} month={month}")
            return prior

        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
        # [v13.2-audit] normalize by global max catch, not self (was always 1.0)
        max_catch = max((r.catch_kg for r in records), default=1.0)
        max_catch = max(max_catch, 1.0)
        for r in records:
            weight = r.catch_kg / max_catch
            dist_sq = (lat_grid - r.lat) ** 2 + (lon_grid - r.lon) ** 2
            prior += weight * np.exp(-dist_sq / (2 * sigma_deg ** 2))

        if prior.max() > 0:
            prior /= prior.max()

        log.info(f"  CatchData: KDE prior for {species}: "
                 f"{len(records)} records, max={prior.max():.3f}")
        return prior

    def generate_mock_data(self, n_records: int = 200) -> Dict:
        """
        生成模擬漁獲數據 (開發/測試用)

        基於台灣漁業實際作業模式生成:
          - 高雄港出發, 台灣周邊+西太平洋
          - 季節×魚種的合理分布
          - 漁獲量符合統計分布
        """
        np.random.seed(42)  # 可重複
        mock_records = []

        species_configs = {
            "yellowfin":  {"lat": (15, 30), "lon": (120, 160), "catch": (30, 200), "months": [3,4,5,6,7,8,9]},
            "bigeye":     {"lat": (10, 28), "lon": (120, 165), "catch": (20, 150), "months": [1,2,3,4,5,10,11,12]},
            "skipjack":   {"lat": (5, 25),  "lon": (125, 170), "catch": (50, 500), "months": list(range(1,13))},
            "mahi_mahi":  {"lat": (22, 27), "lon": (119, 123), "catch": (10, 80),  "months": [4,5,6,7,8,9,10]},
            "blue_marlin": {"lat": (21, 26), "lon": (120, 124), "catch": (50, 300), "months": [5,6,7,8,9]},
            "mackerel_scad": {"lat": (22, 26), "lon": (118, 122), "catch": (100, 2000), "months": [3,4,5,6,7]},
            "pacific_saury": {"lat": (25, 35), "lon": (130, 155), "catch": (200, 5000), "months": [8,9,10,11,12]},
        }

        for sp, cfg in species_configs.items():
            n_sp = n_records // len(species_configs)
            for _ in range(n_sp):
                month = np.random.choice(cfg["months"])
                year = np.random.choice([2023, 2024, 2025])
                day = np.random.randint(1, 28)
                record = CatchRecord(
                    date=f"{year}-{month:02d}-{day:02d}",
                    lat=round(np.random.uniform(*cfg["lat"]), 2),
                    lon=round(np.random.uniform(*cfg["lon"]), 2),
                    species=sp,
                    catch_kg=round(np.random.uniform(*cfg["catch"]), 1),
                    # [v13.2-audit] safe access — SPECIES may be empty dict
                    fishing_method=SPECIES.get(sp, {}).get("gear_type", "unknown"),
                    vessel_id=f"CT3-{np.random.randint(1000, 9999)}",
                )
                mock_records.append(record)

        self._records.extend(mock_records)
        self._save()

        log.info(f"  CatchData: generated {len(mock_records)} mock records")
        return {"generated": len(mock_records), "total": len(self._records)}

    @property
    def record_count(self) -> int:
        return len(self._records) + self._wcpfc_record_count

    def get_species_summary(self) -> Dict:
        """按物種統計漁獲記錄"""
        summary = {}
        for r in self._records:
            if r.species not in summary:
                summary[r.species] = {"count": 0, "total_kg": 0.0}
            summary[r.species]["count"] += 1
            summary[r.species]["total_kg"] += r.catch_kg
        return summary


# ─── 便利函數 ─────────────────────────────────────
try:
    from engine.species_params import SPECIES
except ImportError:
    SPECIES = {}
