"""
OceanMaster v13.2 — War Room Dashboard
======================================
Three-panel layout:
  Left  (280px): Hotspot ranking cards
  Center (flex): Leaflet.js interactive map
  Right (320px): Detail panel on click

All CSS/JS inline in a single self-contained HTML file.
"""

import json
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("OceanMaster.HTMLMap")

# ═══════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════
SP_ZH = {
    "skipjack": "鰹魚", "yellowfin": "黃鰭鮪", "bigeye": "大目鮪",
    "albacore": "長鰭鮪", "squid_todarodes": "赤魷", "squid_ommastrephes": "劍尖魷",
    "japanese_flying_squid": "日本魷", "pacific_saury": "秋刀魚",
    "mahi_mahi": "鬼頭刀", "blue_marlin": "旗魚", "mackerel_scad": "竹筴魚",
    "neon_flying_squid": "赤魷",
}

SP_COLORS = {
    "skipjack": "#2ecc71", "yellowfin": "#f39c12", "bigeye": "#3498db",
    "albacore": "#e74c3c", "squid_todarodes": "#9b59b6", "squid_ommastrephes": "#e67e22",
    "japanese_flying_squid": "#9b59b6", "pacific_saury": "#1abc9c",
    "mahi_mahi": "#e67e22", "blue_marlin": "#2c3e50", "mackerel_scad": "#16a085",
    "neon_flying_squid": "#9b59b6",
}

# 前鎮漁港 (高雄遠洋鮪延繩釣母港)
HOME_PORT = (22.56, 120.31)
HOME_PORT_NAME = "前鎮漁港"
DEFAULT_SPEED_KN = 10

# ═══════════════════════════════════════════
# v15: Real feature lookup from CMEMS data
# ═══════════════════════════════════════════
_V15_LOOKUP = {}


def _load_v15_features():
    """Load true_features_all.csv into a lookup dict (lazy, once)."""
    global _V15_LOOKUP
    if _V15_LOOKUP:
        return
    csv_path = Path("data/true_features/true_features_all.csv")
    if not csv_path.exists():
        log.warning("v15 features not found: %s", csv_path)
        return
    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            key = (int(row["month"]), float(row["lat5c"]), float(row["lon5c"]))
            _V15_LOOKUP[key] = {
                "mld": row.get("mld"),
                "ssh_variance": row.get("ssh_variance"),
                "sst_real": row.get("sst"),
                "chl_real": row.get("chl"),
                "do_surface": row.get("do_surface"),
                "current_speed": row.get("current_speed"),
            }
        log.info("v15 features loaded: %d cell-month entries", len(_V15_LOOKUP))
    except Exception as e:
        log.warning("Failed to load v15 features: %s", e)


def _get_v15_for_hotspot(lat, lon, month=None):
    """Lookup v15 real features for a hotspot location."""
    _load_v15_features()
    if not _V15_LOOKUP:
        return None
    lat5c = np.floor(lat / 5) * 5 + 2.5
    lon5c = np.floor(lon / 5) * 5 + 2.5
    if month is None:
        month = datetime.now().month
    return _V15_LOOKUP.get((month, lat5c, lon5c))


def _eddy_label(ssh_var):
    """Classify eddy activity from SSH spatial variance."""
    if ssh_var is None or (isinstance(ssh_var, float) and math.isnan(ssh_var)):
        return "N/A"
    if ssh_var >= 0.05:
        return "高"
    elif ssh_var >= 0.02:
        return "中"
    else:
        return "低"


def _haversine_nm(lat1, lon1, lat2, lon2):
    """Great-circle distance in nautical miles."""
    R_NM = 3440.065
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return 2 * R_NM * math.asin(math.sqrt(a))


def _s(v, default=0):
    """Sanitize NaN/Inf → default."""
    try:
        return default if (v is None or math.isnan(v) or math.isinf(v)) else v
    except TypeError:
        return v if v is not None else default


# ═══════════════════════════════════════════
# Build data structures for the 3-panel UI
# ═══════════════════════════════════════════



def _build_hotspot_records(hotspots):
    """Build enriched hotspot records with v15 data and travel info."""
    records = []
    for i, h in enumerate(hotspots):
        hid = f"hs_{i}"
        sp = h.get("species", "unknown")
        score = h.get("score", 0)
        score_display = h.get("score_display", score)
        pct = int(round(score_display * 100))
        hsi_str = f"{score_display:.2f}"  # honest HSI value
        hlat, hlon = h.get("lat", 0), h.get("lon", 0)

        # Travel from 前鎮漁港
        dist_nm = _haversine_nm(HOME_PORT[0], HOME_PORT[1], hlat, hlon)
        total_hours = dist_nm / DEFAULT_SPEED_KN
        t_days, t_rem = divmod(total_hours, 24)

        # v15 real features
        v15 = _get_v15_for_hotspot(hlat, hlon)
        mld = None
        ssh_var = None
        sst_real = None
        chl_real = None
        do_val = None
        current_spd = None
        eddy_lbl = "N/A"
        if v15:
            mld = v15.get("mld")
            if mld is not None and isinstance(mld, float) and math.isnan(mld):
                mld = None
            ssh_var = v15.get("ssh_variance")
            if ssh_var is not None and isinstance(ssh_var, float) and math.isnan(ssh_var):
                ssh_var = None
            sst_real = v15.get("sst_real")
            if sst_real is not None and isinstance(sst_real, float) and math.isnan(sst_real):
                sst_real = None
            chl_real = v15.get("chl_real")
            if chl_real is not None and isinstance(chl_real, float) and math.isnan(chl_real):
                chl_real = None
            do_val = v15.get("do_surface")
            if do_val is not None and isinstance(do_val, float) and math.isnan(do_val):
                do_val = None
            current_spd = v15.get("current_speed")
            if current_spd is not None and isinstance(current_spd, float) and math.isnan(current_spd):
                current_spd = None
            eddy_lbl = _eddy_label(ssh_var)

        # 3-tier marker
        if pct >= 70:
            marker_color = "#ff4757"
            marker_radius = 11
            tier = "high"
        elif pct >= 50:
            marker_color = "#ffd24d"
            marker_radius = 8
            tier = "mid"
        else:
            marker_color = "#2ed573"
            marker_radius = 6
            tier = "low"

        eddy_high = bool(ssh_var is not None and ssh_var >= 0.05)

        rec = {
            "id": hid,
            "rank": i + 1,
            "species": sp,
            "species_zh": SP_ZH.get(sp, sp),
            "sp_color": SP_COLORS.get(sp, "#999"),
            "lat": round(hlat, 2),
            "lon": round(hlon, 2),
            "score": round(score, 3),
            "pct": pct,
            "hsi_str": hsi_str,
            "tier": tier,
            "marker_color": marker_color,
            "marker_radius": marker_radius,
            "eddy_high": eddy_high,
            "eddy_label": eddy_lbl,
            "season_warning": h.get("season_warning", False),
            # Popup data
            "phi": round(_s(h.get("phi", 0)), 2),
            "sst": round(_s(h.get("sst", 0)), 1),
            "npp": round(_s(h.get("npp", 0)), 0),
            "chl": round(_s(h.get("chl", 0)), 3),
            "do_surface": round(_s(h.get("do_surface", h.get("do", 0))), 1),
            "hook_depth_label": h.get("hook_depth_label", "N/A"),
            "best_fishing_time": h.get("best_fishing_time", "N/A"),
            "fishing_method": h.get("fishing_method", "N/A"),
            "eez": h.get("eez", "公海"),
            "cpue_index": round(_s(h.get("cpue_index", 0)), 0),
            "cpue_ci_low": round(_s(h.get("cpue_ci_low", 0)), 0),
            "cpue_ci_high": round(_s(h.get("cpue_ci_high", 0)), 0),
            "confidence": h.get("cpue_confidence", h.get("confidence", "Medium")),
            "gfw_validation": h.get("gfw_validation", "N/A"),
            "primary_species_prob": h.get("primary_species_prob", ""),
            # Travel
            "dist_nm": round(dist_nm, 0),
            "travel_days": int(t_days),
            "travel_hours": int(t_rem),
            # v15 real features
            "mld": round(mld, 1) if mld is not None else None,
            "ssh_variance": round(ssh_var, 4) if ssh_var is not None else None,
            "sst_real": round(sst_real, 1) if sst_real is not None else None,
            "chl_real": round(chl_real, 4) if chl_real is not None else None,
            "do_real": round(do_val, 1) if do_val is not None else None,
            "current_speed": round(current_spd, 3) if current_spd is not None else None,
            # v16 NRT data
            "ssh_anomaly": round(h.get("ssh_anomaly", 0), 4) if h.get("ssh_anomaly") is not None else None,
            "eddy_type": h.get("eddy_type", ""),
            "chl_front_strength": round(h.get("chl_front_strength", 0), 3) if h.get("chl_front_strength") is not None else None,
            "wind_speed_ms": h.get("wind_speed_ms"),
            "wind_warning": h.get("wind_warning", False),
            "bottom_temp_c": h.get("bottom_temp_c"),
            # [v17] 雙軌 HSI + 數據新鮮度
            "raw_hsi": round(_s(h.get("raw_hsi", score_display)), 3),
            "data_freshness": h.get("data_freshness", {}),
            # [海鷹/蒼鷺] 新特徵
            "t100": round(_s(h.get("t100", 0)), 1) if h.get("t100") is not None else None,
            "gradient_strength": round(_s(h.get("gradient_strength", 0)), 3) if h.get("gradient_strength") is not None else None,
            "delta_t": round(_s(h.get("delta_t_surface_100", 0)), 1) if h.get("delta_t_surface_100") is not None else None,
            "chl_lag15d": round(_s(h.get("chl_lag15d", 0)), 4) if h.get("chl_lag15d") is not None else None,
            # ═══ v16 Captain Dashboard: 新增欄位 ═══
            # 安全
            "safety_level": h.get("safety_level", "SAFE"),
            "safety_emoji": h.get("safety_emoji", "🟢"),
            "safety_reason": h.get("safety_reason", "海況正常"),
            "safety_score": round(_s(h.get("safety_score", 1.0)), 2),
            # 海流剪切
            "current_shear_ms": round(_s(h.get("current_shear_ms", 0)), 3),
            "shear_risk_level": h.get("shear_risk_level", 0),
            "shear_warning": h.get("shear_warning", ""),
            # 月相
            "lunar_phase": h.get("lunar_phase", ""),
            "lunar_fullness": round(_s(h.get("lunar_fullness", 0)), 2),
            "lunar_fishing_impact": h.get("lunar_fishing_impact", ""),
            "lunar_factor": round(_s(h.get("lunar_factor", 1.0)), 4),
            # 航線成本
            "route_cost_usd": round(_s(h.get("route_cost_usd", 0)), 0),
            "fuel_ton": round(_s(h.get("fuel_ton", 0)), 1),
            "transit_days": round(_s(h.get("transit_days", 0)), 1),
            "distance_km": round(_s(h.get("distance_km", 0)), 0),
            "bearing": h.get("bearing", ""),
            # 多魚種機率
            "species_probs": h.get("species_probs", {}),
            "primary_species_prob": h.get("primary_species_prob", ""),
            "species_uncertainty": h.get("species_uncertainty", ""),
            # SHAP 解釋
            "shap_values": h.get("shap_values", {}),
            "explain": h.get("explain", ""),
            # 生態
            "depth_m": round(_s(h.get("depth_m", 0)), 0),
            "z20_m": round(_s(h.get("z20_m", 0)), 1),
            "mld_m": round(_s(h.get("mld_m", 0)), 1),
            "feeding_index": round(_s(h.get("feeding_index", 0)), 2),
            "greenfish_hsi": round(_s(h.get("greenfish_hsi", 0)), 1),
            "dvm_depth_m": h.get("dvm_depth_m", 0),
            "convergence": round(_s(h.get("convergence", 0)), 2),
            "zoo_proxy": round(_s(h.get("zoo_proxy", 0)), 3),
            # EEZ
            "eez_code": h.get("eez_code", ""),
            "eez_authorized": h.get("eez_authorized", True),
            # ML
            "ml_cpue_kg_day": round(_s(h.get("ml_cpue_kg_day", 0)), 1),
            # 風速
            "wind_speed_ms": round(_s(h.get("wind_speed_ms", 0)), 1),
            "wind_warning": h.get("wind_warning", False),
        }
        records.append(rec)
    return records


def generate_html_map(
    hotspots: List[Dict],
    output_path: str,
    vessel_pos: tuple = (22.56, 120.31),
    timestamp: str = "",
    area: Optional[Dict] = None,
    forecast_result: Optional[Dict] = None,
    gfw_density=None,
    lats=None,
    lons=None,
):
    """Generate v16 Professional Captain Dashboard HTML."""
    from engine.dashboard_template import get_css, get_html_structure, get_javascript

    records = _build_hotspot_records(hotspots)
    records_json = json.dumps(records, ensure_ascii=False)

    # Map center
    if hotspots:
        center_lat = sum(h["lat"] for h in hotspots) / len(hotspots)
        center_lon = sum(h["lon"] for h in hotspots) / len(hotspots)
    else:
        center_lat, center_lon = 22, 130

    ts_display = timestamp[:19] if timestamp else datetime.now().strftime("%Y-%m-%d %H:%M")

    # GFW heatmap data
    gfw_pts_js = "[]"
    if gfw_density is not None and lats is not None and lons is not None:
        pts = []
        ny, nx = gfw_density.shape
        step = max(1, int(np.sqrt(ny * nx / 3000)))
        for iy in range(0, ny, step):
            for ix in range(0, nx, step):
                val = float(gfw_density[iy, ix])
                if val > 0.05:
                    la = float(lats[min(iy, len(lats)-1)])
                    lo = float(lons[min(ix, len(lons)-1)])
                    pts.append(f"[{la:.2f},{lo:.2f},{val:.3f}]")
        gfw_pts_js = "[" + ",".join(pts) + "]"

    # Safety stats
    n_total = len(records)
    n_safe = sum(1 for r in records if r.get("safety_level") == "SAFE")
    n_caution = sum(1 for r in records if r.get("safety_level") == "CAUTION")
    n_avoid = sum(1 for r in records if r.get("safety_level") == "AVOID")

    # Lunar phase from first hotspot
    lunar_phase_str = records[0].get("lunar_phase", "N/A") if records else "N/A"

    # ENSO from first hotspot
    enso_mod = hotspots[0].get("enso_hsi_modifier", 0) if hotspots else 0
    if abs(enso_mod) < 0.01:
        enso_str = "ENSO 中性"
    elif enso_mod > 0:
        enso_str = "La Niña"
    else:
        enso_str = "El Niño"

    # Typhoon detection
    has_typhoon = any(h.get("safety_reason", "").find("颱風") >= 0 for h in hotspots)

    # EEZ boundaries — uses L_.eez (matches dashboard_template.py variable name)
    eez_geojson_js = """
// Taiwan EEZ (粗線高亮)
L.polyline([[21,118],[26,118],[26,123],[21,123],[21,118]], {color:'#00aaff',weight:2.5,opacity:0.7,dashArray:'10,5'}).addTo(L_.eez);
L.marker([23.5,120.5], {icon:L.divIcon({html:'<span style="font-size:11px;color:#00aaff;font-weight:600">🇹🇼 台灣EEZ</span>',className:'',iconAnchor:[5,5]})}).addTo(L_.eez);
// Japan EEZ
L.polyline([[24,122],[24,155],[46,155],[46,130],[30,122],[24,122]], {color:'#e05050',weight:2.5,opacity:0.6,dashArray:'10,5'}).addTo(L_.eez);
L.marker([33,140], {icon:L.divIcon({html:'<span style="font-size:11px;color:#e05050;font-weight:600">🇯🇵 日本EEZ</span>',className:'',iconAnchor:[5,5]})}).addTo(L_.eez);
// Philippines EEZ
L.polyline([[5,116],[5,128],[21,128],[21,118],[18,116],[5,116]], {color:'#e0a020',weight:2.5,opacity:0.6,dashArray:'10,5'}).addTo(L_.eez);
L.marker([12,122], {icon:L.divIcon({html:'<span style="font-size:11px;color:#e0a020;font-weight:600">🇵🇭 菲律賓EEZ</span>',className:'',iconAnchor:[5,5]})}).addTo(L_.eez);
// High seas
L.marker([15,155], {icon:L.divIcon({html:'<span style="font-size:11px;color:#4a6a8a;font-weight:600">🌊 公海 (WCPFC)</span>',className:'',iconAnchor:[20,5]})}).addTo(L_.eez);
"""

    css = get_css()
    body = get_html_structure(
        ts_display, vessel_pos[0], vessel_pos[1], HOME_PORT_NAME,
        n_total, n_safe, n_caution, n_avoid,
        lunar_phase_str, enso_str, has_typhoon,
    )
    js = get_javascript(
        records_json, gfw_pts_js, center_lat, center_lon,
        vessel_pos[0], vessel_pos[1], HOME_PORT_NAME, eez_geojson_js,
    )

    html = f'''<!DOCTYPE html>
<html lang="zh-TW"><head><meta charset="utf-8">
<title>OceanMaster v13.2 Captain Dashboard</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Noto+Sans+TC:wght@300;400;500;700&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<style>{css}</style></head><body>
{body}
<script>{js}</script></body></html>'''

    Path(output_path).write_text(html, encoding="utf-8")
    size_kb = Path(output_path).stat().st_size / 1024
    log.info(f"  War Room HTML: {output_path} ({size_kb:.1f} KB, {len(hotspots)} hotspots)")
    return output_path

