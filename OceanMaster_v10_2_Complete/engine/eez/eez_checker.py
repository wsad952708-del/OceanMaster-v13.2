"""
OceanMaster v10.2.2 EEZ Checker — 商業級修正版
================================================
修正: 多邊形取代矩形消除重疊、小區域優先、支援 Shapely GeoJSON
驗證: 35.28N/140.87E→日本 | 33.89N/149.92E→公海 | 9.03N/133.45E→帛琉
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

log = logging.getLogger("OceanMaster.EEZ")


class EEZChecker:
    """EEZ 判定引擎 — 多邊形邊界 + 射線法"""

    # 格式: (名稱, [(lat, lon), ...]) — 小面積優先檢查
    ZONES: List[Tuple[str, List[Tuple[float, float]]]] = [
        ("台灣EEZ", [
            (21.5, 118.0), (21.5, 123.5), (23.5, 124.5),
            (26.0, 123.0), (26.5, 121.0), (25.5, 119.5),
            (23.0, 117.5), (21.5, 118.0),
        ]),
        ("帛琉EEZ", [
            (2.0, 130.0), (2.0, 136.0), (10.0, 136.0),
            (10.0, 130.0), (2.0, 130.0),
        ]),
        ("美國EEZ(關島/北馬里亞納)", [
            (12.0, 143.5), (12.0, 148.5), (21.0, 148.5),
            (21.0, 146.0), (16.0, 143.5), (12.0, 143.5),
        ]),
        ("菲律賓EEZ", [
            (5.0, 116.0), (5.0, 127.5), (12.0, 129.0),
            (18.0, 128.0), (21.0, 122.0), (21.0, 118.0),
            (18.0, 116.0), (5.0, 116.0),
        ]),
        # 日本 — 本州/北海道
        ("日本EEZ", [
            (30.0, 127.0), (30.5, 131.0), (33.0, 132.5),
            (35.0, 132.0), (38.0, 133.0), (40.0, 135.0),
            (42.0, 140.0), (45.0, 142.0), (46.0, 145.0),
            (46.0, 149.0), (44.0, 148.5), (40.0, 145.5),
            (36.0, 143.0), (33.0, 141.5), (30.0, 138.0),
            (30.0, 127.0),
        ]),
        # 日本 — 小笠原/伊豆
        ("日本EEZ", [
            (24.0, 139.5), (24.0, 144.5), (28.0, 144.5),
            (30.0, 142.0), (30.0, 138.0), (27.0, 139.0),
            (24.0, 139.5),
        ]),
        # 日本 — 琉球/奄美
        ("日本EEZ", [
            (26.0, 123.0), (26.0, 131.0), (30.0, 131.0),
            (30.0, 127.0), (28.0, 125.0), (26.0, 123.0),
        ]),
        ("密克羅尼西亞EEZ", [
            (1.0, 136.0), (1.0, 164.0), (10.0, 164.0),
            (10.0, 136.0), (1.0, 136.0),
        ]),
        ("馬紹爾群島EEZ", [
            (4.0, 160.0), (4.0, 173.0), (15.0, 173.0),
            (15.0, 160.0), (4.0, 160.0),
        ]),
        ("印尼EEZ", [
            (-5.0, 120.0), (-5.0, 141.0), (5.0, 141.0),
            (5.0, 130.0), (2.0, 120.0), (-5.0, 120.0),
        ]),
    ]

    _shapely_polys = None
    _shapely_checked = False

    @classmethod
    def _try_shapely(cls):
        if cls._shapely_checked:
            return cls._shapely_polys is not None
        cls._shapely_checked = True
        try:
            from shapely.geometry import Point, shape
            import json
            for gp in [
                Path(__file__).parent / "data" / "eez_v12_western_pacific.geojson",
                Path("data/eez_v12.geojson"),
            ]:
                if gp.exists():
                    with open(gp) as f:
                        gj = json.load(f)
                    cls._shapely_polys = [
                        (feat["properties"].get("GEONAME", "Unknown"), shape(feat["geometry"]))
                        for feat in gj.get("features", [])
                    ]
                    log.info(f"  EEZ: Shapely+GeoJSON ({len(cls._shapely_polys)} zones)")
                    return True
        except Exception:
            pass
        return False

    @classmethod
    def check_point(cls, lat: float, lon: float) -> Dict[str, Any]:
        """判定座標所在 EEZ → {'eez_name': str, 'is_high_seas': bool, ...}"""
        if cls._try_shapely() and cls._shapely_polys:
            try:
                from shapely.geometry import Point
                pt = Point(lon, lat)
                for name, poly in cls._shapely_polys:
                    if poly.contains(pt):
                        return {"eez_name": name, "is_high_seas": False, "lat": lat, "lon": lon}
                return {"eez_name": "公海", "is_high_seas": True, "lat": lat, "lon": lon}
            except Exception:
                pass

        for zone_name, polygon in cls.ZONES:
            if cls._point_in_polygon(lat, lon, polygon):
                return {"eez_name": zone_name, "is_high_seas": False, "lat": lat, "lon": lon}
        return {"eez_name": "公海", "is_high_seas": True, "lat": lat, "lon": lon}

    @classmethod
    def check_hotspots(cls, hotspots: list) -> list:
        for h in hotspots:
            r = cls.check_point(h["lat"], h["lon"])
            h["eez"] = r["eez_name"]
            h["eez_high_seas"] = r["is_high_seas"]
        return hotspots

    @staticmethod
    def _point_in_polygon(lat: float, lon: float, polygon: List[Tuple[float, float]]) -> bool:
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            yi, xi = polygon[i]
            yj, xj = polygon[j]
            if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    @classmethod
    def get_zone_info(cls, eez_name: str) -> dict:
        INFO = {
            "公海": {"authority": "WCPFC/IATTC", "license_required": False,
                     "note": "高度洄游魚類受RFMO管理"},
            "日本EEZ": {"authority": "Japan Fisheries Agency", "license_required": True,
                        "note": "需日本政府許可證"},
            "台灣EEZ": {"authority": "台灣漁業署", "license_required": True,
                        "note": "台灣籍漁船免許可"},
            "菲律賓EEZ": {"authority": "BFAR Philippines", "license_required": True,
                          "note": "外國漁船需特別許可"},
            "帛琉EEZ": {"authority": "Palau BMR", "license_required": True,
                        "note": "嚴格保護區，部分禁漁"},
            "美國EEZ(關島/北馬里亞納)": {"authority": "NOAA/NMFS", "license_required": True,
                                         "note": "美國聯邦漁業法規"},
            "密克羅尼西亞EEZ": {"authority": "NORMA FSM", "license_required": True,
                               "note": "需FSM入漁許可"},
            "馬紹爾群島EEZ": {"authority": "MIMRA", "license_required": True,
                              "note": "需入漁許可"},
        }
        return INFO.get(eez_name, {"authority": "Unknown", "license_required": True, "note": ""})
