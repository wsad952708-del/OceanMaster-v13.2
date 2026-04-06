"""
OceanMaster — 陸地遮罩模組 (Shared Land Mask)
===============================================
[v13.2-P2] 統一陸地 bbox 定義，消除 main_v10_3.py 和 ai_fusion.py 的重複。

Usage:
    from engine.land_mask import KNOWN_LAND_BBOXES, near_known_land
"""

from typing import List, Tuple

# 已知陸地 bounding boxes (lat_min, lat_max, lon_min, lon_max)
# 寧可多排也不漏排 — 商業遠洋漁場不應落在這些區域內
KNOWN_LAND_BBOXES: List[Tuple[float, float, float, float]] = [
    # ===== 菲律賓 =====
    (14.0, 18.8, 119.7, 123.0),   # 呂宋
    (11.5, 14.0, 119.8, 122.5),   # 民都洛
    (18.3, 20.5, 120.3, 122.5),   # 巴丹+巴布延
    (13.0, 14.5, 123.0, 124.5),   # Catanduanes
    (10.5, 14.0, 122.5, 125.5),   # Samar+Leyte
    (9.0, 11.0, 123.0, 124.5),    # Bohol+Cebu
    (9.0, 11.5, 121.5, 123.5),    # Negros+Panay
    (7.5, 10.0, 121.5, 127.0),    # 民答那峨
    (5.5, 8.0, 124.5, 127.5),     # 民答那峨南
    (4.5, 7.5, 118.5, 123.0),     # 蘇祿群島
    (7.0, 13.0, 117.0, 121.0),    # 巴拉望
    (5.5, 8.5, 116.0, 119.0),     # 蘇祿海西側
    # ===== 台灣 =====
    (21.8, 25.4, 119.8, 122.2),
    (23.5, 25.8, 119.3, 120.5),
    # ===== 日本 =====
    (30.5, 35.5, 129.0, 142.0),
    (26.0, 30.5, 126.0, 131.5),
    (24.0, 26.5, 122.5, 128.5),
    # ===== 中國 =====
    (18.0, 35.5, 108.0, 120.5),
    # ===== 韓國 =====
    (33.0, 38.5, 124.5, 130.5),
    # ===== 印尼 =====
    (-1.0, 5.5, 118.0, 128.5),
    (-8.5, 0.0, 114.0, 141.0),
    (0.5, 7.5, 104.0, 118.5),
    (-5.0, 1.0, 127.0, 141.0),
    # ===== PNG =====
    (-10.0, 0.0, 141.0, 156.0),
    # ===== 越南 =====
    (8.0, 23.5, 102.0, 110.0),
    # ===== 馬來西亞 =====
    (0.8, 7.5, 99.5, 119.5),
    # ===== 密克/帛琉 =====
    (2.0, 10.0, 130.5, 139.5),
    # ===== 澳洲北 =====
    (-20.0, -10.0, 130.0, 155.0),
]

# 預設緩衝距離
DEFAULT_BUFFER_DEG = 0.25   # ~28km (main pipeline 用)
FUSION_BUFFER_DEG = 0.15    # ~17km (ai_fusion hotspot 驗證用，較寬鬆)


def near_known_land(lat: float, lon: float,
                    buffer_deg: float = DEFAULT_BUFFER_DEG) -> bool:
    """檢查座標是否在已知陸地 bbox + buffer 範圍內。

    Args:
        lat: 緯度
        lon: 經度
        buffer_deg: 緩衝距離 (度)，預設 0.25° (~28km)

    Returns:
        True 如果座標在任一陸地 bbox + buffer 內
    """
    for lat1, lat2, lon1, lon2 in KNOWN_LAND_BBOXES:
        if (lat1 - buffer_deg <= lat <= lat2 + buffer_deg and
                lon1 - buffer_deg <= lon <= lon2 + buffer_deg):
            return True
    return False
