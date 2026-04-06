"""
OceanMaster v13.2 — 省油避流航線規劃
=====================================
🟠 重要升級：從熱點列表 → 實際可用航線

功能:
  1. 最短距離航線（大圓航線）
  2. 省油避流航線（避開逆流，利用順流）
  3. 安全航線（避開高浪區、颱風路徑）
  4. 多熱點巡航規劃（TSP 近似求解）

原理:
  - A* 搜索算法在海流場上的應用
  - 考慮洋流對實際航速 (SOG) 的影響
  - 燃油消耗 = f(速度, 逆流, 浪高)

參考: Zermelo 航行問題 (time-optimal routing in current fields)
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timezone
import heapq

log = logging.getLogger("OceanMaster.Navigation")


class RoutePoint:
    """航路點"""
    def __init__(self, lat: float, lon: float, name: str = ""):
        self.lat = lat
        self.lon = lon
        self.name = name


class RouteResult:
    """航線結果"""
    def __init__(self):
        self.waypoints: List[Dict] = []
        self.total_distance_nm: float = 0
        self.total_time_hours: float = 0
        self.fuel_savings_pct: float = 0
        self.route_type: str = ""
        self.warnings: List[str] = []


class FuelOptimalRouter:
    """
    省油避流航線規劃器

    算法：在離散化的海洋網格上做 A* 搜索
    - 節點 = 每個 0.25° 格點
    - 邊成本 = 航行時間（考慮洋流加減速）
    - 啟發函數 = 大圓距離 / 最大速度
    """

    def __init__(
        self,
        vessel_speed_kts: float = 10.0,
        fuel_rate_per_hour: float = 100.0,  # 公升/小時
    ):
        self.vessel_speed_kts = vessel_speed_kts
        self.fuel_rate = fuel_rate_per_hour

    def plan_route(
        self,
        start: RoutePoint,
        destination: RoutePoint,
        ocean_currents: Optional[Dict[str, np.ndarray]] = None,
        weather: Optional[Dict] = None,
        eez_checker: Optional[Any] = None,
    ) -> RouteResult:
        """
        規劃最佳航線

        如果有洋流數據 → A* 省油路線
        如果沒有 → 大圓航線 + 分段路徑點
        """
        result = RouteResult()

        # 基礎大圓距離
        great_circle_nm = self._great_circle_distance(
            start.lat, start.lon, destination.lat, destination.lon
        )

        if ocean_currents is not None and "u" in ocean_currents:
            # A* 省油航線
            log.info(f"規劃省油航線: {start.name} → {destination.name}")
            result = self._astar_route(
                start, destination, ocean_currents, eez_checker
            )
            result.route_type = "fuel_optimal"

            # 計算省油比例
            direct_time = great_circle_nm / self.vessel_speed_kts
            if direct_time > 0:
                result.fuel_savings_pct = max(
                    0, (1 - result.total_time_hours / direct_time) * 100
                )
        else:
            # 大圓航線
            result = self._great_circle_route(start, destination)
            result.route_type = "great_circle"

        log.info(
            f"航線規劃完成: {result.total_distance_nm:.0f}nm, "
            f"{result.total_time_hours:.1f}h, "
            f"省油 {result.fuel_savings_pct:.1f}%"
        )

        return result

    def plan_multi_hotspot_tour(
        self,
        start: RoutePoint,
        hotspots: List[Dict[str, Any]],
        max_hotspots: int = 5,
        ocean_currents: Optional[Dict] = None,
    ) -> List[RouteResult]:
        """
        多熱點巡航規劃（TSP 近似解）

        使用貪心最近鄰算法 + 2-opt 改善
        """
        if not hotspots:
            return []

        # 取前 N 個熱點
        top_hotspots = hotspots[:max_hotspots]

        # 貪心最近鄰：從起點開始，每次選最近的未訪問熱點
        visited = [False] * len(top_hotspots)
        order = []
        current = start

        for _ in range(len(top_hotspots)):
            best_idx = -1
            best_dist = float("inf")

            for i, h in enumerate(top_hotspots):
                if visited[i]:
                    continue
                dist = self._great_circle_distance(
                    current.lat, current.lon, h["lat"], h["lon"]
                )
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx >= 0:
                visited[best_idx] = True
                order.append(best_idx)
                h = top_hotspots[best_idx]
                current = RoutePoint(h["lat"], h["lon"])

        # 2-opt 改善
        order = self._two_opt_improve(start, top_hotspots, order)

        # 為每段規劃航線
        routes = []
        current = start
        for idx in order:
            h = top_hotspots[idx]
            dest = RoutePoint(
                h["lat"], h["lon"],
                f"#{h.get('rank', idx+1)} {h.get('species', 'unknown')}"
            )
            route = self.plan_route(current, dest, ocean_currents)
            routes.append(route)
            current = dest

        return routes

    def _astar_route(
        self,
        start: RoutePoint,
        destination: RoutePoint,
        currents: Dict[str, np.ndarray],
        eez_checker: Optional[Any] = None,
    ) -> RouteResult:
        """A* 搜索省油航線"""
        u = currents.get("u")
        v = currents.get("v")
        lat_arr = currents.get("lat")
        lon_arr = currents.get("lon")

        if u is None or lat_arr is None:
            return self._great_circle_route(start, destination)

        ny, nx = u.shape
        resolution = abs(lat_arr[1] - lat_arr[0]) if len(lat_arr) > 1 else 1.0

        # 起點/終點網格索引
        si = np.argmin(np.abs(lat_arr - start.lat))
        sj = np.argmin(np.abs(lon_arr - start.lon))
        ei = np.argmin(np.abs(lat_arr - destination.lat))
        ej = np.argmin(np.abs(lon_arr - destination.lon))

        # A* 搜索
        # 狀態: (cost, i, j)
        # 8方向移動
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]

        open_set = [(0.0, si, sj)]
        g_cost = np.full((ny, nx), np.inf)
        g_cost[si, sj] = 0
        came_from = {}

        while open_set:
            cost, ci, cj = heapq.heappop(open_set)

            if ci == ei and cj == ej:
                break

            if cost > g_cost[ci, cj]:
                continue

            for di, dj in directions:
                ni, nj = ci + di, cj + dj
                if not (0 <= ni < ny and 0 <= nj < nx):
                    continue

                # 計算這段航行的時間成本
                seg_dist_nm = resolution * 60  # 度 → 海浬
                if di != 0 and dj != 0:
                    seg_dist_nm *= 1.414  # 對角線

                # 洋流對航速的影響
                # 船速 + 順流分量 = 實際對地速度
                curr_u = float(np.nan_to_num(u[ni, nj], nan=0))
                curr_v = float(np.nan_to_num(v[ni, nj], nan=0))

                # 航向角度
                heading = np.arctan2(dj, -di)  # 注意座標系
                curr_along = curr_u * np.cos(heading) + curr_v * np.sin(heading)

                # m/s → 節
                curr_along_kts = curr_along * 1.94384

                effective_speed = max(
                    self.vessel_speed_kts + curr_along_kts,
                    0.5  # 最低有效航速
                )

                seg_time = seg_dist_nm / effective_speed

                # EEZ 碰撞懲罰
                if eez_checker is not None:
                    lat_n = float(lat_arr[ni])
                    lon_n = float(lon_arr[nj])
                    if eez_checker.is_restricted(lat_n, lon_n):
                        seg_time += 999  # 重大懲罰

                new_cost = g_cost[ci, cj] + seg_time
                if new_cost < g_cost[ni, nj]:
                    g_cost[ni, nj] = new_cost
                    came_from[(ni, nj)] = (ci, cj)

                    # 啟發函數: 大圓距離 / 最大速度
                    h = self._great_circle_distance(
                        float(lat_arr[ni]), float(lon_arr[nj]),
                        destination.lat, destination.lon
                    ) / (self.vessel_speed_kts * 1.2)

                    heapq.heappush(open_set, (new_cost + h, ni, nj))

        # 回溯路徑
        result = RouteResult()
        path = []
        current = (ei, ej)
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append((si, sj))
        path.reverse()

        # 轉換為路徑點（每 N 個格點取一個）
        step = max(1, len(path) // 10)
        total_dist = 0
        prev_lat, prev_lon = start.lat, start.lon

        for idx in range(0, len(path), step):
            i, j = path[idx]
            lat_wp = float(lat_arr[i])
            lon_wp = float(lon_arr[j])

            seg_dist = self._great_circle_distance(prev_lat, prev_lon, lat_wp, lon_wp)
            total_dist += seg_dist

            result.waypoints.append({
                "lat": lat_wp,
                "lon": lon_wp,
                "distance_from_prev_nm": round(seg_dist, 1),
                "cumulative_nm": round(total_dist, 1),
            })

            prev_lat, prev_lon = lat_wp, lon_wp

        # 確保終點在最後
        if path:
            last_i, last_j = path[-1]
            last_lat = float(lat_arr[last_i])
            last_lon = float(lon_arr[last_j])
            seg_dist = self._great_circle_distance(prev_lat, prev_lon, last_lat, last_lon)
            total_dist += seg_dist

        result.total_distance_nm = total_dist
        result.total_time_hours = float(g_cost[ei, ej])
        result.fuel_savings_pct = 0  # 在 plan_route 中計算

        return result

    def _great_circle_route(
        self, start: RoutePoint, dest: RoutePoint, n_waypoints: int = 10
    ) -> RouteResult:
        """大圓航線（無洋流數據時使用）"""
        result = RouteResult()

        total_nm = self._great_circle_distance(
            start.lat, start.lon, dest.lat, dest.lon
        )

        for i in range(n_waypoints + 1):
            frac = i / n_waypoints
            lat = start.lat + frac * (dest.lat - start.lat)
            lon = start.lon + frac * (dest.lon - start.lon)

            result.waypoints.append({
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "distance_from_prev_nm": round(total_nm / n_waypoints, 1),
                "cumulative_nm": round(total_nm * frac, 1),
            })

        result.total_distance_nm = total_nm
        result.total_time_hours = total_nm / self.vessel_speed_kts

        return result

    def _great_circle_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """大圓距離（海浬）"""
        r = 3440.065  # 地球半徑（海浬）
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return 2 * r * np.arcsin(np.sqrt(a))

    def _two_opt_improve(
        self, start: RoutePoint, hotspots: List[Dict], order: List[int]
    ) -> List[int]:
        """2-opt 改善 TSP 路徑"""
        n = len(order)
        if n < 3:
            return order

        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    # 計算交換前後的距離差
                    h_i = hotspots[order[i]]
                    h_j = hotspots[order[j]]
                    h_i1 = hotspots[order[i+1]] if i+1 < n else h_i
                    h_j1 = hotspots[order[(j+1) % n]] if j+1 < n else h_j

                    d_before = (
                        self._great_circle_distance(h_i["lat"], h_i["lon"], h_i1["lat"], h_i1["lon"])
                        + self._great_circle_distance(h_j["lat"], h_j["lon"], h_j1["lat"], h_j1["lon"])
                    )
                    d_after = (
                        self._great_circle_distance(h_i["lat"], h_i["lon"], h_j["lat"], h_j["lon"])
                        + self._great_circle_distance(h_i1["lat"], h_i1["lon"], h_j1["lat"], h_j1["lon"])
                    )

                    if d_after < d_before:
                        order[i+1:j+1] = order[i+1:j+1][::-1]
                        improved = True

        return order
