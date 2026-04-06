"""
OceanMaster v10.3 — 颱風影響 ML 特徵工程
==========================================
Task #18: IBTrACS 歷史颱風路徑 + ML 特徵

訓練資料: IBTrACS 2015-2024 (NOAA NCEI)
新增特徵:
  - days_before_typhoon (-7 ~ 0)
  - days_after_typhoon (0 ~ 7)
  - typhoon_intensity_cat (1-5, Saffir-Simpson)
  - distance_to_typhoon_track (km)

資料來源:
  https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.WP.list.v04r01.csv
"""

import csv
import io
import logging
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("OceanMaster.TyphoonML")

# IBTrACS 下載 URL (西太平洋)
IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.WP.list.v04r01.csv"
)

CACHE_DIR = Path("data/ibtracs")


class TyphoonFeatureEngineer:
    """
    颱風影響 ML 特徵工程。
    基於 IBTrACS 歷史颱風路徑計算空間-時間特徵。
    """

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.tracks: List[Dict] = []  # 已載入的歷史颱風軌跡

    def load_ibtracs(self, years: range = range(2015, 2025)) -> int:
        """
        載入 IBTrACS 歷史颱風軌跡。
        優先從本機快取讀取，如無則回傳空列表。

        Returns:
            載入的颱風數量
        """
        csv_path = self.cache_dir / "ibtracs_wp.csv"

        if csv_path.exists():
            self.tracks = self._parse_ibtracs_csv(csv_path, years)
            log.info(f"  IBTrACS: loaded {len(self.tracks)} tracks "
                      f"({years.start}-{years.stop-1})")
            return len(self.tracks)

        # 嘗試從內建小型資料集載入
        embedded = self._get_embedded_tracks(years)
        if embedded:
            self.tracks = embedded
            log.info(f"  IBTrACS: {len(embedded)} embedded tracks "
                      f"({years.start}-{years.stop-1})")
            return len(embedded)

        log.warning("  IBTrACS: no data available. "
                    f"Download from {IBTRACS_URL} to {csv_path}")
        return 0

    async def download_ibtracs(self) -> bool:
        """
        下載 IBTrACS CSV (約 50MB)。
        """
        try:
            import httpx
        except ImportError:
            import requests as httpx

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.cache_dir / "ibtracs_wp.csv"

        log.info(f"  IBTrACS: downloading from NCEI...")
        try:
            if hasattr(httpx, 'AsyncClient'):
                async with httpx.AsyncClient(timeout=120) as client:
                    resp = await client.get(IBTRACS_URL)
                    if resp.status_code == 200:
                        csv_path.write_bytes(resp.content)
                        log.info(f"  IBTrACS: saved {len(resp.content)/1e6:.1f}MB")
                        return True
            else:
                resp = httpx.get(IBTRACS_URL, timeout=120)
                if resp.status_code == 200:
                    csv_path.write_bytes(resp.content)
                    return True
        except Exception as e:
            log.warning(f"  IBTrACS download failed: {e}")

        return False

    def compute_features(
        self,
        lat: float,
        lon: float,
        date: datetime,
        search_radius_km: float = 1000.0,
        time_window_days: int = 7,
    ) -> Dict[str, float]:
        """
        對特定座標和日期計算颱風 ML 特徵。

        Returns:
            {
                'days_before_typhoon': float (-7~0, 999 if none)
                'days_after_typhoon': float (0~7, 999 if none)
                'typhoon_intensity_cat': int (0-5)
                'distance_to_typhoon_track': float (km, 9999 if none)
            }
        """
        result = {
            "days_before_typhoon": 999.0,
            "days_after_typhoon": 999.0,
            "typhoon_intensity_cat": 0,
            "distance_to_typhoon_track": 9999.0,
        }

        if not self.tracks:
            return result

        best_before = 999.0
        best_after = 999.0
        min_dist = 9999.0
        best_cat = 0

        for track in self.tracks:
            for pt in track.get("points", []):
                pt_date = pt.get("date")
                if pt_date is None:
                    continue

                # 計算時間差 (天)
                dt_days = (date - pt_date).total_seconds() / 86400.0
                dist_km = _haversine_km(lat, lon, pt["lat"], pt["lon"])

                # 僅考慮距離 < search_radius_km 的軌跡點
                if dist_km > search_radius_km:
                    continue

                if dist_km < min_dist:
                    min_dist = dist_km
                    best_cat = pt.get("category", 0)

                # 颱風尚未到達 (days_before: -7~0)
                if 0 < dt_days <= time_window_days:
                    # 正值 dt_days = 颱風在過去 = 已過境
                    if dt_days < best_after:
                        best_after = dt_days

                # 颱風即將到達 (days_before: -7~0)
                if -time_window_days <= dt_days < 0:
                    if abs(dt_days) < abs(best_before):
                        best_before = dt_days

        result["days_before_typhoon"] = best_before
        result["days_after_typhoon"] = best_after
        result["typhoon_intensity_cat"] = best_cat
        result["distance_to_typhoon_track"] = round(min_dist, 1)

        return result

    def compute_features_grid(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        date: datetime,
    ) -> Dict[str, np.ndarray]:
        """
        對整個 2D 網格計算颱風 ML 特徵。
        """
        ny, nx = len(lats), len(lons)
        days_before = np.full((ny, nx), 999.0, dtype=np.float32)
        days_after = np.full((ny, nx), 999.0, dtype=np.float32)
        intensity = np.zeros((ny, nx), dtype=np.int32)
        distance = np.full((ny, nx), 9999.0, dtype=np.float32)

        for iy in range(ny):
            for ix in range(nx):
                feat = self.compute_features(lats[iy], lons[ix], date)
                days_before[iy, ix] = feat["days_before_typhoon"]
                days_after[iy, ix] = feat["days_after_typhoon"]
                intensity[iy, ix] = feat["typhoon_intensity_cat"]
                distance[iy, ix] = feat["distance_to_typhoon_track"]

        return {
            "days_before_typhoon": days_before,
            "days_after_typhoon": days_after,
            "typhoon_intensity_cat": intensity,
            "distance_to_typhoon_track": distance,
        }

    # ─── 私有方法 ───

    def _parse_ibtracs_csv(
        self, csv_path: Path, years: range
    ) -> List[Dict]:
        """解析 IBTrACS CSV 檔案"""
        tracks = {}
        try:
            with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
                # 跳過前兩行 (header + units)
                lines = f.readlines()
                if len(lines) < 3:
                    return []

                reader = csv.DictReader(io.StringIO(''.join(lines[0:1] + lines[2:])))
                for row in reader:
                    try:
                        sid = row.get("SID", "")
                        year = int(row.get("SEASON", "0"))
                        if year not in years:
                            continue

                        iso_time = row.get("ISO_TIME", "")
                        lat = float(row.get("LAT", "999"))
                        lon = float(row.get("LON", "999"))
                        wind = row.get("USA_WIND", "")
                        wind_kt = int(float(wind)) if wind and wind.strip() else 0

                        if abs(lat) > 90 or abs(lon) > 360:
                            continue

                        try:
                            dt = datetime.strptime(iso_time, "%Y-%m-%d %H:%M:%S")
                            dt = dt.replace(tzinfo=timezone.utc)
                        except ValueError:
                            continue

                        if sid not in tracks:
                            tracks[sid] = {
                                "sid": sid,
                                "name": row.get("NAME", "UNNAMED"),
                                "season": year,
                                "points": [],
                            }

                        tracks[sid]["points"].append({
                            "date": dt,
                            "lat": lat,
                            "lon": lon,
                            "wind_kt": wind_kt,
                            "category": _kt_to_category(wind_kt),
                        })
                    except (ValueError, KeyError):
                        continue

        except Exception as e:
            log.warning(f"  IBTrACS parse error: {e}")

        return list(tracks.values())

    @staticmethod
    def _get_embedded_tracks(years: range) -> List[Dict]:
        """
        內建近年重大颱風資料 (用於無法下載 IBTrACS 時)。
        僅包含西太平洋主要颱風的簡化軌跡。
        """
        embedded = [
            {
                "sid": "2024239N12154",
                "name": "SHANSHAN",
                "season": 2024,
                "points": [
                    {"date": datetime(2024, 8, 22, tzinfo=timezone.utc),
                     "lat": 21.0, "lon": 135.0, "wind_kt": 55, "category": 0},
                    {"date": datetime(2024, 8, 24, tzinfo=timezone.utc),
                     "lat": 24.5, "lon": 131.0, "wind_kt": 95, "category": 2},
                    {"date": datetime(2024, 8, 26, tzinfo=timezone.utc),
                     "lat": 28.0, "lon": 128.5, "wind_kt": 120, "category": 4},
                    {"date": datetime(2024, 8, 29, tzinfo=timezone.utc),
                     "lat": 32.0, "lon": 130.5, "wind_kt": 80, "category": 1},
                ],
            },
            {
                "sid": "2024190N05140",
                "name": "GAEMI",
                "season": 2024,
                "points": [
                    {"date": datetime(2024, 7, 20, tzinfo=timezone.utc),
                     "lat": 15.5, "lon": 131.0, "wind_kt": 45, "category": 0},
                    {"date": datetime(2024, 7, 22, tzinfo=timezone.utc),
                     "lat": 20.0, "lon": 126.0, "wind_kt": 85, "category": 2},
                    {"date": datetime(2024, 7, 24, tzinfo=timezone.utc),
                     "lat": 23.5, "lon": 121.5, "wind_kt": 100, "category": 3},
                    {"date": datetime(2024, 7, 26, tzinfo=timezone.utc),
                     "lat": 27.0, "lon": 119.0, "wind_kt": 60, "category": 0},
                ],
            },
            {
                "sid": "2023258N12128",
                "name": "KOINU",
                "season": 2023,
                "points": [
                    {"date": datetime(2023, 10, 1, tzinfo=timezone.utc),
                     "lat": 17.5, "lon": 134.0, "wind_kt": 50, "category": 0},
                    {"date": datetime(2023, 10, 3, tzinfo=timezone.utc),
                     "lat": 20.0, "lon": 127.0, "wind_kt": 115, "category": 4},
                    {"date": datetime(2023, 10, 5, tzinfo=timezone.utc),
                     "lat": 22.0, "lon": 120.5, "wind_kt": 95, "category": 2},
                ],
            },
        ]

        return [t for t in embedded if t["season"] in years]


def _kt_to_category(kt: int) -> int:
    """風速 (kt) → Saffir-Simpson"""
    if kt >= 137: return 5
    if kt >= 113: return 4
    if kt >= 96:  return 3
    if kt >= 83:  return 2
    if kt >= 64:  return 1
    return 0


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
