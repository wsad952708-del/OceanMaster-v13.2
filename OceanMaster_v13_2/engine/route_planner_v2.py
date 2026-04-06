import numpy as np
import logging
import heapq
import time
from typing import Dict, Any, List, Tuple, Optional

log = logging.getLogger("OceanMaster.Route")

# ── 預設參數 ──
KAOHSIUNG_PORT = (22.61, 120.28)   # 高雄前鎮漁港
EEZ_BUFFER_NM = 10  # 海浬安全緩衝
GRID_RESOLUTION = 0.1  # 度 (~11km)

DEFAULT_VESSEL = {
    "name": "100GT 鮪延繩釣",
    "speed_knots": 10.0,
    "fuel_rate_ton_per_hour": 0.18,
    "diesel_price_usd_per_ton": 1000.0,
    "operation_hours_per_day": 12.0,   # 作業 + 轉場
    "crew_cost_usd_per_day": 200.0,    # 船員日薪
    "ice_bait_usd_per_trip": 500.0,    # 冰+餌固定成本
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """大圓距離 (km)"""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2) ** 2)
    return float(R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


# ── A* 尋路核心 ──────────────────────────────────────────────

def _build_obstacle_grid(lat_min, lat_max, lon_min, lon_max, res, buffer_nm):
    """
    建立障礙物網格。True = 禁航。
    """
    try:
        from engine.eez.eez_checker import EEZChecker
    except ImportError:
        return None, None, None

    lats = np.arange(lat_min, lat_max + res, res)
    lons = np.arange(lon_min, lon_max + res, res)
    n_lat, n_lon = len(lats), len(lons)
    blocked = np.zeros((n_lat, n_lon), dtype=bool)

    for i in range(n_lat):
        for j in range(n_lon):
            blocked[i, j] = EEZChecker.is_in_restricted_zone(
                float(lats[i]), float(lons[j]), buffer_nm
            )
    return blocked, lats, lons


def _astar_path(
    start: Tuple[float, float],
    end: Tuple[float, float],
    buffer_nm: float = EEZ_BUFFER_NM,
    res: float = GRID_RESOLUTION,
    u_grid: Optional[np.ndarray] = None,  # [v15.4] east current m/s
    v_grid: Optional[np.ndarray] = None,  # [v15.4] north current m/s
    current_lats: Optional[np.ndarray] = None,
    current_lons: Optional[np.ndarray] = None,
    vessel_speed_kmh: float = 18.52,  # 10 knots default
) -> Tuple[List[Tuple[float, float]], float]:
    """
    A* 尋路: 回傳 (waypoints, total_distance_km)。
    [v13.2] 自適應解析度 + 5 秒 timeout 保護。
    [v15.4] 海流感知路由：考慮海流對航速/油耗的影響。
    若無法建立網格或目標不可達或超時，退化為大圓直線。
    """
    direct_dist = haversine_km(start[0], start[1], end[0], end[1])

    # [v13.2] 自適應解析度 — 遠洋用粗網格避免效能爆炸
    if direct_dist > 2000:
        res = 0.5   # ~55km, 遠洋
    elif direct_dist > 1000:
        res = 0.25  # ~27km, 中程
    # else: 0.1° (~11km, 近海) — 保持預設

    # 建立網格邊界 (加 padding 讓繞路有空間)
    pad = max(2.0, direct_dist / 111.0 * 0.3)  # 至少 2° 或 30% 距離的 padding
    lat_min = min(start[0], end[0]) - pad
    lat_max = max(start[0], end[0]) + pad
    lon_min = min(start[1], end[1]) - pad
    lon_max = max(start[1], end[1]) + pad

    result = _build_obstacle_grid(lat_min, lat_max, lon_min, lon_max, res, buffer_nm)
    if result is None or result[0] is None:
        log.warning("  A* fallback: EEZChecker not available, using direct route")
        return [start, end], direct_dist

    blocked, lats, lons = result
    n_lat, n_lon = len(lats), len(lons)

    # [v13.2-R7] Grid size safety — auto-coarsen if too large
    if n_lat * n_lon > 100_000:
        log.warning(f"  A* grid too large ({n_lat}x{n_lon}={n_lat*n_lon}), "
                    f"coarsening resolution from {res}° to {res*2}°")
        res *= 2
        result = _build_obstacle_grid(lat_min, lat_max, lon_min, lon_max, res, buffer_nm)
        if result is None or result[0] is None:
            return [start, end], direct_dist
        blocked, lats, lons = result
        n_lat, n_lon = len(lats), len(lons)

    # [v15.4] Build current-effect lookup for fuel-aware routing
    has_currents = (u_grid is not None and v_grid is not None and
                    current_lats is not None and current_lons is not None)
    curr_u = None
    curr_v = None
    if has_currents:
        try:
            from scipy.interpolate import RegularGridInterpolator
            u_interp = RegularGridInterpolator(
                (current_lats, current_lons), u_grid,
                method='nearest', bounds_error=False, fill_value=0.0
            )
            v_interp = RegularGridInterpolator(
                (current_lats, current_lons), v_grid,
                method='nearest', bounds_error=False, fill_value=0.0
            )
            # Pre-compute current vectors on the A* grid
            curr_u = np.zeros((n_lat, n_lon), dtype=np.float32)
            curr_v = np.zeros((n_lat, n_lon), dtype=np.float32)
            for i in range(n_lat):
                for j in range(n_lon):
                    pt = np.array([float(lats[i]), float(lons[j])])
                    curr_u[i, j] = float(u_interp(pt))
                    curr_v[i, j] = float(v_interp(pt))
            log.info(f"  ⛽ Current-aware routing: grid {n_lat}×{n_lon}, "
                     f"max current {np.sqrt(curr_u**2+curr_v**2).max():.2f} m/s")
        except Exception as e:
            log.warning(f"  Current-aware routing failed, using distance: {e}")
            has_currents = False

    # 找最近的網格索引
    def nearest(lat, lon):
        i = int(round((lat - lat_min) / res))
        j = int(round((lon - lon_min) / res))
        i = max(0, min(i, n_lat - 1))
        j = max(0, min(j, n_lon - 1))
        return i, j

    si, sj = nearest(start[0], start[1])
    ei, ej = nearest(end[0], end[1])

    # 起終點本身被擋住 → 放行 (港口或熱點在 EEZ 內)
    blocked[si, sj] = False
    blocked[ei, ej] = False

    # 8-way 鄰居
    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # A* open set: (f_score, g_score, i, j)
    open_set = [(0.0, 0.0, si, sj)]
    came_from = {}
    g_score = np.full((n_lat, n_lon), np.inf)
    g_score[si, sj] = 0.0

    def h(i, j):
        return haversine_km(float(lats[i]), float(lons[j]),
                            float(lats[ei]), float(lons[ej]))

    def fuel_step_cost(ci, cj, ni, nj):
        """
        [v15.4] Fuel-adjusted step cost.
        If moving against the current → effective speed drops → more fuel → higher cost.
        If moving with the current → effective speed rises → less fuel → lower cost.
        """
        dist_km = haversine_km(
            float(lats[ci]), float(lons[cj]),
            float(lats[ni]), float(lons[nj])
        )
        if not has_currents or curr_u is None:
            return dist_km

        # Ship heading vector (unit, in m/s scale doesn't matter, just direction)
        dlat = float(lats[ni] - lats[ci])
        dlon = float(lons[nj] - lons[cj]) * np.cos(np.radians(lats[ci]))
        norm = max(np.sqrt(dlat**2 + dlon**2), 1e-10)
        h_north = dlat / norm  # heading north component
        h_east = dlon / norm   # heading east component

        # Current at midpoint
        mi, mj = (ci + ni) // 2, (cj + nj) // 2
        cu = float(curr_u[mi, mj])  # east m/s
        cv = float(curr_v[mi, mj])  # north m/s

        # Project current onto heading: positive = following, negative = opposing
        current_proj_ms = cu * h_east + cv * h_north  # m/s along heading

        # Effective speed = vessel_speed + current projection
        current_proj_kmh = current_proj_ms * 3.6  # m/s → km/h
        effective_speed = max(vessel_speed_kmh + current_proj_kmh, vessel_speed_kmh * 0.3)

        # Fuel cost ∝ distance / effective_speed × vessel_speed (relative time)
        fuel_factor = vessel_speed_kmh / effective_speed
        return dist_km * fuel_factor

    found = False
    iterations = 0
    max_iterations = min(n_lat * n_lon * 2, 50_000)  # [v13.2-R7] hard cap at 50K
    t_start = time.time()  # [v13.2] timeout 保護

    while open_set and iterations < max_iterations:
        iterations += 1
        # [v13.2] 5 秒 timeout — 超時 fallback 大圓直線
        if iterations % 500 == 0 and (time.time() - t_start) > 5.0:
            log.warning(f"  A* timeout after {time.time()-t_start:.1f}s ({iterations} iters), using direct route")
            return [start, end], direct_dist
        f, g, ci, cj = heapq.heappop(open_set)

        if ci == ei and cj == ej:
            found = True
            break

        if g > g_score[ci, cj]:
            continue  # stale entry

        for di, dj in DIRS:
            ni, nj = ci + di, cj + dj
            if 0 <= ni < n_lat and 0 <= nj < n_lon and not blocked[ni, nj]:
                step_cost = fuel_step_cost(ci, cj, ni, nj)
                ng = g + step_cost
                if ng < g_score[ni, nj]:
                    g_score[ni, nj] = ng
                    f_new = ng + h(ni, nj)
                    came_from[(ni, nj)] = (ci, cj)
                    heapq.heappush(open_set, (f_new, ng, ni, nj))

    if not found:
        log.warning("  A* fallback: no valid path found, using direct route")
        return [start, end], direct_dist

    # 回溯路徑
    path_ij = [(ei, ej)]
    cur = (ei, ej)
    while cur in came_from:
        cur = came_from[cur]
        path_ij.append(cur)
    path_ij.reverse()

    # 轉座標 + 計算總距離
    waypoints = [(round(float(lats[i]), 2), round(float(lons[j]), 2))
                 for i, j in path_ij]
    total_dist = sum(
        haversine_km(waypoints[k][0], waypoints[k][1],
                     waypoints[k+1][0], waypoints[k+1][1])
        for k in range(len(waypoints) - 1)
    )

    # [v15.4] Calculate fuel savings vs direct route
    if has_currents:
        direct_fuel_cost = sum(
            fuel_step_cost(si, sj, ei, ej)
            for _ in [0]
        )
        log.info(f"  ⛽ Current-aware route: {len(waypoints)} waypoints, "
                 f"{total_dist:.1f}km (direct: {direct_dist:.1f}km)")
    else:
        log.info(f"  A* route: {len(waypoints)} waypoints, "
                 f"{total_dist:.1f}km (direct: {direct_dist:.1f}km, "
                 f"detour: +{total_dist - direct_dist:.1f}km)")

    return waypoints, total_dist




def compute_route_cost(
    hotspot_lat: float,
    hotspot_lon: float,
    port: Tuple[float, float] = KAOHSIUNG_PORT,
    vessel: Optional[Dict] = None,
    round_trip: bool = True,
    operation_days: int = 3,
    wave_height_m: float = 1.0,
    current_speed_ms: float = 0.0,
    current_dir_deg: float = 0.0,
) -> Dict[str, Any]:
    """
    計算從港口到熱點的航行成本 (v13.2: A* 自適應繞行 + 海況修正)。

    Parameters (v13.2 新增)
    ----------
    wave_height_m : float
        有效波高 (m), 預設 1.0m (BN 3)
    current_speed_ms : float
        作用海流速度 (m/s), 預設 0.0
    current_dir_deg : float
        海流方向 (度, 0=北), 預設 0.0
    """
    v = vessel or DEFAULT_VESSEL

    # ── [v13.1] A* EEZ 安全繞行 ──
    waypoints, one_way_dist = _astar_path(
        port, (hotspot_lat, hotspot_lon),
        buffer_nm=EEZ_BUFFER_NM, res=GRID_RESOLUTION,
    )
    direct_dist = haversine_km(port[0], port[1], hotspot_lat, hotspot_lon)
    crossed_eezs = []

    # 檢查原始直線是否穿越限制區
    try:
        from engine.eez.eez_checker import EEZChecker
        num_check = max(3, int(direct_dist / 50.0))
        lats_c = np.linspace(port[0], hotspot_lat, num_check)
        lons_c = np.linspace(port[1], hotspot_lon, num_check)
        for lt, ln in zip(lats_c, lons_c):
            res_eez = EEZChecker.check_point(float(lt), float(ln))
            if not res_eez["is_high_seas"]:
                name = res_eez["eez_name"]
                info = EEZChecker.get_zone_info(name)
                if info.get("authority") != "台灣漁業署" and info.get("license_required"):
                    if name not in crossed_eezs:
                        crossed_eezs.append(name)
    except ImportError as e:
        log.debug(f"[降級] engine/route_planner_v2.py: {e}")

    if round_trip:
        total_dist = one_way_dist * 2
    else:
        total_dist = one_way_dist

    speed_kmh = v["speed_knots"] * 1.852
    transit_hours = total_dist / max(speed_kmh, 1.0)
    transit_days = transit_hours / 24.0

    # ── [v13.2] 海況油耗修正 ──
    # 浪高修正: BN > 1m 每增加 1m 油耗增 15%
    sea_state_factor = 1.0 + 0.15 * max(0.0, wave_height_m - 1.0)
    # 海流修正: 順流減油耗, 逆流增油耗
    heading_deg = _bearing(port[0], port[1], hotspot_lat, hotspot_lon)
    angle_diff = np.radians(current_dir_deg - heading_deg)
    current_effect_kmh = current_speed_ms * 3.6 * np.cos(angle_diff)  # m/s → km/h projected
    current_factor = 1.0 - current_effect_kmh / max(speed_kmh, 1.0)
    current_factor = float(np.clip(current_factor, 0.7, 1.5))  # cap: ±30-50%

    fuel_rate_adjusted = v["fuel_rate_ton_per_hour"] * sea_state_factor * current_factor

    # 燃油
    fuel_transit = transit_hours * fuel_rate_adjusted
    fuel_operation = operation_days * v["operation_hours_per_day"] * v["fuel_rate_ton_per_hour"] * 0.6
    fuel_total = fuel_transit + fuel_operation

    fuel_cost = fuel_total * v["diesel_price_usd_per_ton"]
    total_days = transit_days + operation_days
    crew_cost = total_days * v["crew_cost_usd_per_day"]
    fixed_cost = v["ice_bait_usd_per_trip"]

    total_cost = fuel_cost + crew_cost + fixed_cost
    cost_per_day = total_cost / max(total_days, 0.1)

    # 方位角
    bearing = _bearing(port[0], port[1], hotspot_lat, hotspot_lon)

    return {
        "distance_km": round(one_way_dist, 1),
        "direct_distance_km": round(direct_dist, 1),
        "detour_km": round(one_way_dist - direct_dist, 1),
        "transit_hours": round(transit_hours, 1),
        "transit_days": round(transit_days, 1),
        "total_days": round(total_days, 1),
        "operation_days": operation_days,
        "fuel_transit_ton": round(fuel_transit, 2),
        "fuel_operation_ton": round(fuel_operation, 2),
        "fuel_total_ton": round(fuel_total, 2),
        "fuel_cost_usd": round(fuel_cost, 0),
        "crew_cost_usd": round(crew_cost, 0),
        "fixed_cost_usd": round(fixed_cost, 0),
        "total_cost_usd": round(total_cost, 0),
        "cost_per_day_usd": round(cost_per_day, 0),
        "bearing_deg": round(bearing, 1),
        "bearing_compass": _compass(bearing),
        "vessel_type": v["name"],
        "round_trip": round_trip,
        "safe_waypoints": waypoints,
        "eez_buffer_nm": EEZ_BUFFER_NM,
        "eez_avoidance_method": "A*" if len(waypoints) > 2 else "direct",
        "crossed_unauthorized_eez": crossed_eezs,
        "weather_hazards": _check_route_weather(waypoints),
    }


def compute_all_routes(
    hotspots: List[Dict],
    port: Tuple[float, float] = KAOHSIUNG_PORT,
    vessel: Optional[Dict] = None,
    operation_days: int = 3,
) -> List[Dict]:
    """
    批量計算所有熱點的路線成本。

    Parameters
    ----------
    hotspots : list of dict
        每個 dict 至少有 lat, lon
    """
    results = []
    for h in hotspots:
        lat = h.get("lat", 0)
        lon = h.get("lon", 0)
        if lat == 0 and lon == 0:
            continue
        route = compute_route_cost(lat, lon, port, vessel,
                                    operation_days=operation_days)
        route["hotspot"] = h
        results.append(route)

    # 排序: 總成本由低到高
    results.sort(key=lambda r: r["total_cost_usd"])

    if results:
        cheapest = results[0]
        most_expensive = results[-1]
        log.info(f"  Route costs: cheapest={cheapest['distance_km']:.0f}km "
                 f"${cheapest['total_cost_usd']:.0f}, "
                 f"farthest={most_expensive['distance_km']:.0f}km "
                 f"${most_expensive['total_cost_usd']:.0f}")

    return results


def _bearing(lat1, lon1, lat2, lon2):
    """初始方位角 (度)"""
    dlon = np.radians(lon2 - lon1)
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def _compass(deg):
    """方位角轉羅盤方向"""
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = int((deg + 11.25) / 22.5) % 16
    return dirs[idx]


def _check_route_weather(
    waypoints: List[Tuple[float, float]],
    hs_threshold: float = 3.0,
) -> List[Dict[str, Any]]:
    """
    [v18] Check 3-day sea state forecast along route waypoints.

    Uses Open-Meteo Marine API (free, no key).
    Tags segments with Hs > threshold as hazardous.

    Ref: ZeroNorth/Sofar weather-aware voyage optimization concept.

    Returns:
        List of hazard dicts: [{lat, lon, day, Hs, warning}]
    """
    if len(waypoints) < 2:
        return []

    # Sample up to 5 waypoints to avoid API flooding
    step = max(1, len(waypoints) // 5)
    sample_pts = waypoints[::step]
    if waypoints[-1] not in sample_pts:
        sample_pts.append(waypoints[-1])

    hazards = []
    try:
        import urllib.request
        import json

        for lat, lon in sample_pts:
            url = (
                f"https://marine-api.open-meteo.com/v1/marine?"
                f"latitude={lat}&longitude={lon}"
                f"&hourly=wave_height&forecast_days=3&timezone=UTC"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "OceanMaster/15.3"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())

            hourly_hs = data.get("hourly", {}).get("wave_height", [])
            times = data.get("hourly", {}).get("time", [])

            for i, hs in enumerate(hourly_hs):
                if hs is not None and hs > hs_threshold:
                    day = i // 24 + 1
                    hazards.append({
                        "lat": round(lat, 2),
                        "lon": round(lon, 2),
                        "day": day,
                        "Hs_m": round(hs, 1),
                        "warning": f"⚠️ Day{day} Hs={hs:.1f}m > {hs_threshold}m",
                    })
                    break  # one hazard per waypoint is enough

        if hazards:
            log.info(f"  🌊 Weather hazards: {len(hazards)} waypoints with Hs>{hs_threshold}m")

    except Exception as e:
        log.debug(f"  Route weather check skipped: {e}")

    return hazards
