"""
OceanMaster v10.4 — 法規掃雷艦 (Legal Compliance Checker)
==========================================================
Task #1: MPA / RFMO 禁漁區即時碰撞偵測

Architecture:
  - 啟動時: 載入多邊形 → Shapely STRtree (R-Tree 空間索引)
  - 查詢時: Point-in-Polygon O(log n), 毫秒級回應
  - 攔截: 非法座標強制 {"legal": False, "violation": "MPA_ZONE"}

Data sources (priority):
  1. data/global_mpa_rfmo.geojson (使用者自行下載 GFW/WDPA)
  2. 內建嵌入式多邊形 (WCPFC 公海禁漁區 + 重要 MPA)

嚴禁每次請求重新讀取檔案 — STRtree 在 module level 初始化一次。
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("OceanMaster.Legal")

# ═══════════════════════════════════════════════════
# 嵌入式禁漁區多邊形 (不依賴外部 GeoJSON)
# ═══════════════════════════════════════════════════
# 來源: WCPFC CMM (Conservation and Management Measures),
#        IATTC, ICCAT, GFW public MPA boundaries
#
# 格式: [(name, type, polygon_coords), ...]
# polygon_coords = [(lon, lat), ...] (GeoJSON 順序)

EMBEDDED_ZONES = [
    # ── WCPFC 高海鰹魚禁漁區 (CMM 2008-01, Pocket #1 & #2) ──
    (
        "WCPFC_HSP1_FAD_closure",
        "RFMO_SEASONAL",
        "每年 7-9 月禁止 FAD 作業",
        [(155, 20), (155, -1), (175, -1), (175, 20), (155, 20)],
    ),
    (
        "WCPFC_HSP2_FAD_closure",
        "RFMO_SEASONAL",
        "每年 7-9 月禁止 FAD 作業",
        [(175, 20), (175, -1), (195, -1), (195, 20), (175, 20)],
    ),
    # ── 帛琉國家海洋保護區 (PNMS) ──
    (
        "Palau_National_Marine_Sanctuary",
        "MPA_ZONE",
        "帛琉 EEZ 80% 全面禁漁",
        [(131, 2), (131, 8), (136, 8), (136, 2), (131, 2)],
    ),
    # ── 馬里亞納海溝國家海洋紀念碑 ──
    (
        "Marianas_Trench_MNM",
        "MPA_ZONE",
        "美國海洋保護區，禁止商業捕魚",
        [(142, 11), (142, 21), (148, 21), (148, 11), (142, 11)],
    ),
    # ── 鳳凰群島保護區 (PIPA) ──
    (
        "Phoenix_Islands_Protected_Area",
        "MPA_ZONE",
        "基里巴斯全面禁漁海洋保護區",
        [(-176, -5), (-176, -1), (-170, -1), (-170, -5), (-176, -5)],
    ),
    # ── 帕帕哈瑙莫夸基亞國家海洋紀念碑 ──
    (
        "Papahanaumokuakea_MNM",
        "MPA_ZONE",
        "夏威夷西北部群島全面保護",
        [(-180, 22), (-180, 30), (-154, 30), (-154, 22), (-180, 22)],
    ),
    # ── 加拉帕戈斯海洋保護區 ──
    (
        "Galapagos_Marine_Reserve",
        "MPA_ZONE",
        "厄瓜多爾加拉帕戈斯禁漁區",
        [(-92, -2), (-92, 1), (-89, 1), (-89, -2), (-92, -2)],
    ),
    # ── IATTC 東太平洋禁漁期 (72 天封閉) ──
    (
        "IATTC_Closure_Zone",
        "RFMO_SEASONAL",
        "東太平洋圍網 72 天封閉區",
        [(-150, -5), (-150, 5), (-80, 5), (-80, -5), (-150, -5)],
    ),
    # ── 查戈斯群島 MPA (英國) ──
    (
        "Chagos_MPA",
        "MPA_ZONE",
        "印度洋最大禁漁 MPA",
        [(70, -8), (70, -4), (74, -4), (74, -8), (70, -8)],
    ),
    # ── 南極海洋保護區 (Ross Sea) ──
    (
        "Ross_Sea_MPA",
        "MPA_ZONE",
        "CCAMLR 南極最大海洋保護區",
        [(150, -78), (150, -60), (-150, -60), (-150, -78), (150, -78)],
    ),
]


class RegulationChecker:
    """
    MPA / RFMO 禁漁區碰撞偵測器。

    使用 Shapely STRtree 空間索引，啟動時建立一次，
    後續每次查詢 O(log n)。
    """

    def __init__(self, geojson_path: Optional[str] = None):
        """
        初始化空間索引。

        Args:
            geojson_path: 可選外部 GeoJSON 路徑
                          (e.g. "data/global_mpa_rfmo.geojson")
        """
        self._zones: List[Dict] = []
        self._tree = None
        self._shapely_available = False

        self._load_zones(geojson_path)

    def _load_zones(self, geojson_path: Optional[str] = None):
        """載入禁漁區多邊形並建立 STRtree"""
        try:
            from shapely.geometry import Polygon, shape
            from shapely import STRtree
            self._shapely_available = True
        except ImportError:
            log.warning("Shapely not installed — using fallback point-in-polygon")
            self._load_zones_fallback()
            return

        from shapely.geometry import Polygon, shape
        from shapely import STRtree

        polygons = []
        zone_meta = []

        # 1. 載入嵌入式多邊形
        for name, violation_type, description, coords in EMBEDDED_ZONES:
            try:
                poly = Polygon(coords)
                if poly.is_valid:
                    polygons.append(poly)
                    zone_meta.append({
                        "name": name,
                        "violation": violation_type,
                        "description": description,
                    })
            except Exception as e:
                log.warning(f"  Invalid embedded polygon {name}: {e}")

        # 2. 載入外部 GeoJSON (如有)
        external_path = Path(geojson_path) if geojson_path else Path("data/global_mpa_rfmo.geojson")
        if external_path.exists():
            try:
                with open(external_path, "r", encoding="utf-8") as f:
                    geojson = json.load(f)

                features = geojson.get("features", [])
                for feat in features:
                    try:
                        geom = shape(feat["geometry"])
                        if geom.is_valid:
                            props = feat.get("properties", {})
                            polygons.append(geom)
                            zone_meta.append({
                                "name": props.get("NAME", props.get("name", "Unknown MPA")),
                                "violation": props.get("DESIG_ENG", "MPA_ZONE"),
                                "description": props.get("ORIG_NAME", ""),
                            })
                    except Exception:
                        continue

                log.info(f"  Legal: loaded {len(features)} zones from {external_path}")
            except Exception as e:
                log.warning(f"  Legal: GeoJSON load error: {e}")

        # 3. 建立 STRtree
        if polygons:
            self._tree = STRtree(polygons)
            self._zones = zone_meta
            self._polygons = polygons
            log.info(f"  Legal: STRtree built with {len(polygons)} zones (O(log n) lookup)")
        else:
            log.warning("  Legal: no zones loaded, all points will be legal")

    def _load_zones_fallback(self):
        """無 Shapely 時的後備方案 (BBox 近似)"""
        self._zones = []
        for name, violation_type, description, coords in EMBEDDED_ZONES:
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            self._zones.append({
                "name": name,
                "violation": violation_type,
                "description": description,
                "bbox": (min(lons), min(lats), max(lons), max(lats)),
                "coords": coords,
            })
        log.info(f"  Legal: fallback mode with {len(self._zones)} zones (BBox + ray-casting)")

    def check_point(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        檢查座標是否落入禁漁區。

        Args:
            lat, lon: 座標

        Returns:
            {
                "legal": bool,
                "violation": str or None,  # "MPA_ZONE", "RFMO_SEASONAL", etc.
                "zone_name": str or None,
                "description": str or None,
            }
        """
        result = {
            "legal": True,
            "violation": None,
            "zone_name": None,
            "description": None,
        }

        if self._shapely_available and self._tree is not None:
            return self._check_point_strtree(lat, lon)
        elif self._zones:
            return self._check_point_fallback(lat, lon)

        return result

    def _check_point_strtree(self, lat: float, lon: float) -> Dict:
        """STRtree 毫秒級查詢"""
        from shapely.geometry import Point

        point = Point(lon, lat)  # GeoJSON: (lon, lat)
        result = {
            "legal": True,
            "violation": None,
            "zone_name": None,
            "description": None,
        }

        # STRtree.query 找出可能相交的多邊形索引
        candidate_idxs = self._tree.query(point)

        for idx in candidate_idxs:
            poly = self._polygons[idx]
            if poly.contains(point):
                meta = self._zones[idx]
                result["legal"] = False
                result["violation"] = meta["violation"]
                result["zone_name"] = meta["name"]
                result["description"] = meta["description"]
                return result

        return result

    def _check_point_fallback(self, lat: float, lon: float) -> Dict:
        """無 Shapely 時的 ray-casting 後備"""
        result = {
            "legal": True,
            "violation": None,
            "zone_name": None,
            "description": None,
        }

        for zone in self._zones:
            bbox = zone.get("bbox")
            if bbox:
                if not (bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]):
                    continue

            # Ray-casting point-in-polygon
            coords = zone.get("coords", [])
            if coords and self._ray_casting(lat, lon, coords):
                result["legal"] = False
                result["violation"] = zone["violation"]
                result["zone_name"] = zone["name"]
                result["description"] = zone["description"]
                return result

        return result

    @staticmethod
    def _ray_casting(lat: float, lon: float, polygon: List[Tuple]) -> bool:
        """Ray-casting algorithm for point-in-polygon (fallback)"""
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]  # (lon, lat) GeoJSON order
            xj, yj = polygon[j]
            if ((yi > lat) != (yj > lat)) and \
               (lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-10) + xi):
                inside = not inside
            j = i
        return inside

    def filter_hotspots(
        self, hotspots: List[Dict], remove_illegal: bool = True
    ) -> List[Dict]:
        """
        批次檢查 hotspots 合法性。

        Args:
            hotspots: 熱點列表，每個需有 lat, lon
            remove_illegal: True=剔除非法, False=僅標記

        Returns:
            經過合法性標記/剔除的 hotspots
        """
        result = []
        n_illegal = 0

        for h in hotspots:
            check = self.check_point(h.get("lat", 0), h.get("lon", 0))
            h["legal"] = check["legal"]
            h["legal_violation"] = check["violation"]
            h["legal_zone"] = check["zone_name"]

            if not check["legal"]:
                n_illegal += 1
                if remove_illegal:
                    continue

            result.append(h)

        if n_illegal > 0:
            action = "removed" if remove_illegal else "flagged"
            log.warning(
                f"  Legal: {n_illegal}/{len(hotspots)} hotspots {action} "
                f"(MPA/RFMO violations)"
            )

        return result

    def get_all_zones_geojson(self) -> Dict:
        """輸出所有禁漁區為 GeoJSON (供 Dashboard 展示)"""
        features = []
        for name, violation_type, description, coords in EMBEDDED_ZONES:
            features.append({
                "type": "Feature",
                "properties": {
                    "name": name,
                    "violation": violation_type,
                    "description": description,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords],
                },
            })
        return {"type": "FeatureCollection", "features": features}


# ═══════════════════════════════════════════════════
# Module-level singleton (啟動時初始化一次)
# ═══════════════════════════════════════════════════
_checker_instance: Optional[RegulationChecker] = None


def get_checker() -> RegulationChecker:
    """Get or create the singleton RegulationChecker."""
    global _checker_instance
    if _checker_instance is None:
        _checker_instance = RegulationChecker()
    return _checker_instance


def check_point(lat: float, lon: float) -> Dict[str, Any]:
    """Convenience function: check single point legality."""
    return get_checker().check_point(lat, lon)


def filter_hotspots(hotspots: List[Dict], remove_illegal: bool = True) -> List[Dict]:
    """Convenience function: filter hotspots by legality."""
    return get_checker().filter_hotspots(hotspots, remove_illegal)
