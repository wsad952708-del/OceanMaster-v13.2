"""
OceanMaster v13.2 — 颱風追蹤引擎 + 72h 路徑預測
===================================================
Task #11: JTWC 颱風警報 RSS 爬蟲 + GeoJSON 輸出
[v13.5] 新增 72h Persistence+Beta-Advection 預報模型

數據源 (Cascade):
  1. JTWC RSS: https://www.metoc.navy.mil/jtwc/rss/jtwc.rss
  2. GDACS RSS: https://www.gdacs.org/xml/rss_tc.xml (Fallback)
  3. 本機快取 (< 1 小時)
  4. 空列表 (所有 RSS 不可用時)

預測模型:
  Persistence + Beta-Advection (CLIPER 簡化版)
  - 基於統計平均移動速度 + 科氏力偏轉 + 遞迴轉向
  - 72h 每 12h 輸出一個 ForecastPoint
  - 強度衰減: 每 12h 5-8% (高緯度加速衰減)

輸出:
  - TyphoonAlert dataclass (含 forecast_track)
  - GeoJSON FeatureCollection (含分層危險圈 200/350/500km)
  - data/typhoon_alerts.json 快取
"""

import asyncio
import json
import logging
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from engine.base_fetcher import BaseFetcher, DataSource

log = logging.getLogger("OceanMaster.Typhoon")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Data Models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ForecastPoint:
    """颱風預測路徑上的一點"""
    hour: int           # 預報小時 (0, 12, 24, 36, 48, 60, 72)
    lat: float
    lon: float
    intensity_kt: int = 0
    category: int = 0   # Saffir-Simpson 1-5

@dataclass
class TyphoonAlert:
    """單一活躍颱風警報"""
    typhoon_id: str             # e.g. "WP072026"
    name: str                   # e.g. "GAEMI"
    lat: float                  # 目前中心緯度
    lon: float                  # 目前中心經度
    intensity_kt: int = 0       # 最大持續風速 (kt)
    category: int = 0           # Saffir-Simpson 等級
    timestamp: str = ""         # ISO 時間戳
    basin: str = "WP"           # WP/EP/NA/SI/SP
    forecast_track: List[ForecastPoint] = field(default_factory=list)
    source: str = ""            # JTWC / GDACS


def kt_to_category(kt: int) -> int:
    """風速 (kt) → Saffir-Simpson 颶風等級"""
    if kt >= 137:
        return 5
    elif kt >= 113:
        return 4
    elif kt >= 96:
        return 3
    elif kt >= 83:
        return 2
    elif kt >= 64:
        return 1
    return 0  # 熱帶風暴或以下


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """大圓距離 (km)"""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TyphoonTracker
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TyphoonTracker(BaseFetcher):
    """
    颱風追蹤系統 (繼承 BaseFetcher cascade 機制)
    [v13.5] 含 72h 路徑預測
    """

    RSS_SOURCES = [
        {
            "name": "JTWC",
            "url": "https://www.metoc.navy.mil/jtwc/rss/jtwc.rss",
            "parser": "_parse_jtwc",
        },
        {
            "name": "GDACS",
            "url": "https://www.gdacs.org/xml/rss_tc.xml",
            "parser": "_parse_gdacs",
        },
    ]

    CACHE_PATH = Path("data/typhoon_alerts.json")
    CACHE_MAX_AGE_HOURS = 1

    def __init__(self):
        super().__init__()

    async def fetch_active_typhoons(self) -> List[TyphoonAlert]:
        """
        取得目前活躍颱風列表 + 72h 預測路徑。

        Cascade: JTWC RSS → GDACS RSS → 快取 → 空列表
        """
        cached = self._load_cache()

        for src_cfg in self.RSS_SOURCES:
            try:
                alerts = await self._fetch_rss(src_cfg)
                if alerts:
                    # [v13.5] 為每個颱風生成 72h 預測路徑
                    for alert in alerts:
                        if not alert.forecast_track:
                            alert.forecast_track = self._extrapolate_track_72h(alert)
                    self._save_cache(alerts)
                    log.info(f"  颱風: {len(alerts)} 個活躍警報 ({src_cfg['name']})")
                    return alerts
            except Exception as e:
                log.debug(f"  颱風 {src_cfg['name']}: {e}")
                continue

        if cached:
            log.info(f"  颱風: 使用快取 ({len(cached)} 個)")
            return cached

        log.info("  颱風: 無活躍警報 (或所有 RSS 不可用)")
        return []

    async def _fetch_rss(self, src_cfg: Dict) -> List[TyphoonAlert]:
        """抓取並解析單一 RSS 來源"""
        try:
            import httpx
        except ImportError:
            import requests as httpx

        url = src_cfg["url"]
        try:
            if hasattr(httpx, 'AsyncClient'):
                async with httpx.AsyncClient(timeout=20) as client:
                    resp = await client.get(url)
                    if resp.status_code != 200:
                        return []
                    xml_text = resp.text
            else:
                resp = httpx.get(url, timeout=20)
                if resp.status_code != 200:
                    return []
                xml_text = resp.text
        except Exception as e:
            log.debug(f"  RSS fetch failed: {e}")
            return []

        parser_name = src_cfg["parser"]
        if parser_name == "_parse_jtwc":
            return self._parse_jtwc(xml_text)
        elif parser_name == "_parse_gdacs":
            return self._parse_gdacs(xml_text)
        return []

    # ─── RSS 解析器 ───

    def _parse_jtwc(self, xml_text: str) -> List[TyphoonAlert]:
        """解析 JTWC RSS XML"""
        alerts = []
        try:
            root = ET.fromstring(xml_text)
            for item in root.iter("item"):
                title = item.findtext("title", "")
                desc = item.findtext("description", "")
                if not any(kw in title.upper() for kw in
                           ["TROPICAL", "TYPHOON", "HURRICANE", "CYCLONE", "WARNING"]):
                    continue
                alert = self._parse_jtwc_item(title, desc)
                if alert:
                    alerts.append(alert)
        except ET.ParseError as e:
            log.debug(f"  JTWC XML parse error: {e}")
        return alerts

    def _parse_jtwc_item(self, title: str, desc: str) -> Optional[TyphoonAlert]:
        """從 JTWC item 中提取颱風資訊"""
        id_match = re.search(r'(\d{2}[WESCPAB])', title)
        name_match = re.search(r'\((\w+)\)', title)
        typhoon_id = id_match.group(1) if id_match else "UNKNOWN"
        name = name_match.group(1) if name_match else "UNNAMED"

        lat, lon = self._extract_coords(desc)
        if lat is None:
            return None

        wind_match = re.search(r'(\d+)\s*(?:KT|KNOTS|kt)', desc)
        intensity_kt = int(wind_match.group(1)) if wind_match else 50

        year = datetime.now(timezone.utc).strftime("%Y")
        return TyphoonAlert(
            typhoon_id=f"WP{typhoon_id}{year[-2:]}",
            name=name, lat=lat, lon=lon,
            intensity_kt=intensity_kt,
            category=kt_to_category(intensity_kt),
            timestamp=datetime.now(timezone.utc).isoformat(),
            basin="WP", source="JTWC",
        )

    def _parse_gdacs(self, xml_text: str) -> List[TyphoonAlert]:
        """解析 GDACS RSS XML"""
        alerts = []
        try:
            root = ET.fromstring(xml_text)
            ns = {
                "gdacs": "http://www.gdacs.org",
                "geo": "http://www.w3.org/2003/01/geo/wgs84_pos#",
            }
            for item in root.iter("item"):
                title = item.findtext("title", "")
                if "TC" not in title.upper() and "TROPICAL" not in title.upper():
                    continue

                lat_el = item.find(".//geo:lat", ns)
                lon_el = item.find(".//geo:long", ns)
                if lat_el is not None and lon_el is not None:
                    lat = float(lat_el.text)
                    lon = float(lon_el.text)
                else:
                    point = item.find(".//geo:Point/geo:pos", ns)
                    if point is not None and point.text:
                        parts = point.text.strip().split()
                        lat, lon = float(parts[0]), float(parts[1])
                    else:
                        continue

                name_match = re.search(r'"([^"]+)"', title)
                name = name_match.group(1) if name_match else title[:20]
                severity = item.findtext("{http://www.gdacs.org}severity", "")
                wind_match = re.search(r'(\d+)', severity)
                intensity_kt = int(wind_match.group(1)) if wind_match else 50

                alerts.append(TyphoonAlert(
                    typhoon_id=f"GDACS-{len(alerts)+1}",
                    name=name, lat=lat, lon=lon,
                    intensity_kt=intensity_kt,
                    category=kt_to_category(intensity_kt),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    source="GDACS",
                ))
        except ET.ParseError as e:
            log.debug(f"  GDACS XML parse error: {e}")
        return alerts

    @staticmethod
    def _extract_coords(text: str) -> tuple:
        """從文字中提取經緯度"""
        lat_match = re.search(r'(\d+\.?\d*)\s*([NS])', text, re.IGNORECASE)
        lon_match = re.search(r'(\d+\.?\d*)\s*([EW])', text, re.IGNORECASE)
        if lat_match and lon_match:
            lat = float(lat_match.group(1))
            if lat_match.group(2).upper() == 'S':
                lat = -lat
            lon = float(lon_match.group(1))
            if lon_match.group(2).upper() == 'W':
                lon = -lon
            return lat, lon
        return None, None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  [v13.5] 72h 路徑預測 — CLIPER 簡化版
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _extrapolate_track_72h(self, alert: TyphoonAlert) -> List[ForecastPoint]:
        """
        Persistence + Beta-Advection 72h 預報模型

        原理:
        1. 基礎速度: 西北太平洋颱風統計平均值 (分緯度帶)
           - <15N: 偏西移動 (信風帶)
           - 15-25N: 西北移動, 逐漸北轉
           - 25-35N: 轉向帶 (recurvature)
           - >35N: 東北加速 (西風帶)
        2. Beta-drift: 行星渦度梯度 → 偏北西 ~1.5 m/s
        3. Recurvature: 接近副高邊緣時向東北轉
        4. 強度衰減: 每 12h 5-8% (高緯度加速)
        """
        points = []
        lat0, lon0 = alert.lat, alert.lon
        kt0 = alert.intensity_kt or 50

        # T+0: 當前位置
        points.append(ForecastPoint(
            hour=0, lat=round(lat0, 2), lon=round(lon0, 2),
            intensity_kt=kt0, category=kt_to_category(kt0),
        ))

        lat, lon, kt = lat0, lon0, float(kt0)

        for h in [12, 24, 36, 48, 60, 72]:
            # === 移動速度 (度/12h) ===
            if lat < 15:
                dlat, dlon = 0.8, -1.2       # 偏西移動
            elif lat < 25:
                f = (lat - 15) / 10
                dlat = 1.0 + f * 0.3
                dlon = -0.8 + f * 0.5
            elif lat < 35:
                rf = (lat - 25) / 10          # recurvature 0→1
                dlat = 1.2 + rf * 0.5
                dlon = -0.3 + rf * 1.5
            else:
                dlat, dlon = 1.5, 1.5         # 東北加速

            # Beta-drift (北半球偏北西)
            dlat += 0.15
            dlon -= 0.10

            # 經度修正 (高緯度每度距離更短)
            cos_lat = math.cos(math.radians(lat))
            if cos_lat > 0.1:
                dlon *= math.cos(math.radians(25)) / cos_lat

            lat += dlat
            lon += dlon

            # 強度衰減
            decay = 0.05 if lat < 30 else 0.08
            kt *= (1.0 - decay)
            kt = max(kt, 25)  # 最低 25kt

            points.append(ForecastPoint(
                hour=h, lat=round(lat, 2), lon=round(lon, 2),
                intensity_kt=int(kt), category=kt_to_category(int(kt)),
            ))

        log.info(f"  颱風 {alert.name}: 72h 預測 "
                 f"{lat0:.1f}N,{lon0:.1f}E → {lat:.1f}N,{lon:.1f}E, "
                 f"{kt0}kt→{int(kt)}kt")
        return points

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  GeoJSON 輸出 (含分層危險圈)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def to_geojson(alerts: List[TyphoonAlert]) -> Dict:
        """轉出 GeoJSON FeatureCollection (含預測路徑 + 分層危險圈)"""
        features = []
        for a in alerts:
            # ── 颱風中心 ──
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [a.lon, a.lat]},
                "properties": {
                    "type": "typhoon_center",
                    "typhoon_id": a.typhoon_id,
                    "name": a.name,
                    "intensity_kt": a.intensity_kt,
                    "category": a.category,
                    "source": a.source,
                    "timestamp": a.timestamp,
                },
            })

            # ── [v13.5] 分層危險圈: 200/350/500km ──
            for radius_km, level, color in [
                (200, "core", "#ef4444"),
                (350, "moderate", "#f97316"),
                (500, "alert", "#eab308"),
            ]:
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [_circle_coords(a.lat, a.lon, radius_km)],
                    },
                    "properties": {
                        "type": "danger_zone",
                        "level": level,
                        "color": color,
                        "typhoon_id": a.typhoon_id,
                        "radius_km": radius_km,
                    },
                })

            # ── 預測路徑 ──
            if a.forecast_track:
                track_coords = [[fp.lon, fp.lat] for fp in a.forecast_track]
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": track_coords},
                    "properties": {
                        "type": "forecast_track",
                        "typhoon_id": a.typhoon_id,
                        "name": a.name,
                    },
                })

                # 每 12h 預測點 + 未來危險圈
                for fp in a.forecast_track:
                    if fp.hour == 0:
                        continue
                    features.append({
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [fp.lon, fp.lat]},
                        "properties": {
                            "type": "forecast_point",
                            "typhoon_id": a.typhoon_id,
                            "hour": fp.hour,
                            "intensity_kt": fp.intensity_kt,
                            "category": fp.category,
                        },
                    })
                    # 未來位置 350km 危險圈
                    features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [_circle_coords(fp.lat, fp.lon, 350)],
                        },
                        "properties": {
                            "type": "future_danger_zone",
                            "typhoon_id": a.typhoon_id,
                            "hour": fp.hour,
                            "radius_km": 350,
                        },
                    })

        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "count": len(alerts),
                "generated": datetime.now(timezone.utc).isoformat(),
            },
        }

    # ─── 快取 ───

    def _save_cache(self, alerts: List[TyphoonAlert]):
        """寫入本機 JSON 快取"""
        self.CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "alerts": [asdict(a) for a in alerts],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.CACHE_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_cache(self) -> List[TyphoonAlert]:
        """讀取本機快取 (含 forecast_track 還原)"""
        if not self.CACHE_PATH.exists():
            return []
        try:
            data = json.loads(self.CACHE_PATH.read_text(encoding="utf-8"))
            ts = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
            age_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
            if age_hours > self.CACHE_MAX_AGE_HOURS:
                return []
            return [
                TyphoonAlert(
                    typhoon_id=a["typhoon_id"],
                    name=a["name"],
                    lat=a["lat"], lon=a["lon"],
                    intensity_kt=a.get("intensity_kt", 0),
                    category=a.get("category", 0),
                    timestamp=a.get("timestamp", ""),
                    source=a.get("source", "cache"),
                    forecast_track=[
                        ForecastPoint(**fp) for fp in a.get("forecast_track", [])
                    ],
                )
                for a in data.get("alerts", [])
            ]
        except Exception:
            return []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  工具函數
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _circle_coords(lat: float, lon: float, radius_km: float, n: int = 36) -> List:
    """產生圓形座標 (近似, for GeoJSON polygon)"""
    coords = []
    for i in range(n + 1):
        angle = 2 * math.pi * i / n
        dlat = radius_km / 111.0 * math.cos(angle)
        dlon = radius_km / (111.0 * math.cos(math.radians(lat))) * math.sin(angle)
        coords.append([round(lon + dlon, 4), round(lat + dlat, 4)])
    return coords
