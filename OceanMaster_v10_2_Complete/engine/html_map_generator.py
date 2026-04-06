"""
OceanMaster v10.2.2 — HTML 互動地圖生成
=========================================
使用 Leaflet.js 生成自包含 HTML 地圖。
管線 Step 9 自動呼叫，替代手寫靜態 HTML。
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("OceanMaster.HTMLMap")

SP_ZH = {
    "skipjack": "鰹魚", "yellowfin": "黃鰭鮪", "bigeye": "大目鮪",
    "albacore": "長鰭鮪", "squid_todarodes": "赤魷", "squid_ommastrephes": "劍尖魷",
}

SP_COLORS = {
    "skipjack": "#2ecc71", "yellowfin": "#f39c12", "bigeye": "#3498db",
    "albacore": "#e74c3c", "squid_todarodes": "#9b59b6", "squid_ommastrephes": "#e67e22",
}

GRADE_COLORS = {
    "best": "#e74c3c", "good": "#f39c12", "fair": "#f1c40f", "low": "#95a5a6",
}


def generate_html_map(
    hotspots: List[Dict],
    output_path: str,
    vessel_pos: tuple = (25.13, 121.74),
    timestamp: str = "",
    area: Optional[Dict] = None,
):
    """生成自包含 Leaflet HTML 地圖"""

    # 按物種分組
    species_groups = {}
    for h in hotspots:
        sp = h.get("species", "unknown")
        species_groups.setdefault(sp, []).append(h)

    # 生成 JS 數據
    js_data_lines = []
    for h in hotspots:
        sp = h.get("species", "unknown")
        sp_zh = SP_ZH.get(sp, sp)
        score = h.get("score", 0)
        pct = int(round(score * 100))

        if score >= 0.75:
            grade_color = GRADE_COLORS["best"]
            grade_label = "最佳🔴"
        elif score >= 0.60:
            grade_color = GRADE_COLORS["good"]
            grade_label = "優秀🟠"
        elif score >= 0.45:
            grade_color = GRADE_COLORS["fair"]
            grade_label = "良好🟡"
        else:
            grade_color = GRADE_COLORS["low"]
            grade_label = "偏低⚪"

        phi = h.get("phi", 0)
        sst = h.get("sst", 0)
        npp = h.get("npp", 0)
        chl = h.get("chl", 0)
        do = h.get("do_surface", h.get("do", 0))
        front = h.get("front_persistence", h.get("front_pct", 0))
        if isinstance(front, float) and front <= 1: front = int(front * 100)
        eddy = h.get("eddy_edge", h.get("eddy_pct", 0))
        if isinstance(eddy, float) and eddy <= 1: eddy = int(eddy * 100)
        conv = h.get("convergence", h.get("conv_pct", 0))
        if isinstance(conv, float) and conv <= 1: conv = int(conv * 100)
        conv = min(conv, 100)

        forage = h.get("forage_index", h.get("forage_pct", 0))
        if isinstance(forage, float) and forage <= 1: forage = int(forage * 100)
        thermal = h.get("h_thermal", h.get("thermal_pct", 0))
        if isinstance(thermal, float) and thermal <= 1: thermal = int(thermal * 100)
        depth = h.get("depth_m", 0)
        eez = h.get("eez", "公海")
        sp_color = SP_COLORS.get(sp, "#999")

        popup = (
            f"<b>#{hotspots.index(h)+1} {sp_zh} ({pct}%)</b><br>"
            f"<span style='color:{grade_color}'>{grade_label}</span> HSI: {pct}%<br>"
            f"Φ: {phi:.2f} | SST: {sst:.1f}°C<br>"
            f"NPP: {npp:.0f} | Chl: {chl:.3f} | DO: {do:.1f}<br>"
            f"鋒面: {front}% | 渦旋: {eddy}% | 匯聚: {conv}%<br>"
            f"水域: {eez}"
        )
        dist = h.get('distance_nm', 0)
        travel_h = h.get('travel_hours', 0)
        if dist > 0:
            popup += f"<br>📍 距離: {dist:.0f} nm ({travel_h:.1f}hr @10kn)"

        rank = hotspots.index(h) + 1
        js_data_lines.append(
            f"{{lat:{h['lat']:.2f},lng:{h['lon']:.2f},name:'#{rank} {sp_zh} ({pct}%)',"
            f"species:'{sp_zh}',hsi:{pct},phi:{phi:.2f},sst:{sst:.1f},"
            f"npp:{npp:.0f},chl:{chl:.3f},do_:{do:.1f},"
            f"front:{front},eddy:{eddy},conv:{conv},"
            f"forage:{forage},thermal:{thermal},depth:{depth:.1f},"
            f"eez:'{eez}',popup:`{popup}`,spColor:'{sp_color}',gradeColor:'{grade_color}'}}"
        )

    js_data = ",\n".join(js_data_lines)

    # 物種 legend
    legend_items = []
    for sp, group in species_groups.items():
        sp_zh = SP_ZH.get(sp, sp)
        color = SP_COLORS.get(sp, "#999")
        legend_items.append(f"<span style='color:{color};font-weight:bold'>●</span> {sp_zh} ({len(group)})")

    legend_html = " &nbsp;|&nbsp; ".join(legend_items)

    # 地圖中心
    if hotspots:
        center_lat = sum(h["lat"] for h in hotspots) / len(hotspots)
        center_lon = sum(h["lon"] for h in hotspots) / len(hotspots)
    else:
        center_lat, center_lon = 25, 140

    html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>OceanMaster v10.2 漁場地圖</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
body{{margin:0;font-family:'Segoe UI',sans-serif}}
#map{{height:100vh;width:100%}}
.info-box{{position:absolute;top:10px;left:50px;z-index:1000;background:rgba(0,0,0,.85);color:#fff;
  padding:12px 18px;border-radius:8px;font-size:13px;max-width:400px}}
.info-box h3{{margin:0 0 6px;color:#3498db}}
.legend{{position:absolute;bottom:30px;left:50px;z-index:1000;background:rgba(0,0,0,.8);color:#fff;
  padding:8px 14px;border-radius:6px;font-size:12px}}
.grade-bar{{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:4px}}
</style></head><body>
<div id="map"></div>
<div class="info-box">
  <h3>🎣 OceanMaster v10.2</h3>
  <div>分析時間: {timestamp[:19]}</div>
  <div style="margin-top:4px">{legend_html}</div>
  <div style="margin-top:4px;color:#aaa;font-size:11px">Φ=Deutsch 2015 代謝指數(真實計算) | SEAPODYM 棲息地</div>
</div>
<div class="legend">
  <span class="grade-bar" style="background:#e74c3c"></span>最佳(≥75%)
  <span class="grade-bar" style="background:#f39c12;margin-left:8px"></span>優秀(60-75%)
  <span class="grade-bar" style="background:#f1c40f;margin-left:8px"></span>良好(45-60%)
  <span class="grade-bar" style="background:#95a5a6;margin-left:8px"></span>偏低(&lt;45%)
</div>
<script>
const data = [
{js_data}
];

const map = L.map('map').setView([{center_lat:.2f},{center_lon:.2f}], 5);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}@2x.png',{{
  attribution:'&copy; CARTO',maxZoom:18
}}).addTo(map);

data.forEach(d => {{
  const marker = L.circleMarker([d.lat, d.lng], {{
    radius: Math.max(5, d.hsi / 8),
    fillColor: d.gradeColor,
    color: d.spColor,
    weight: 2,
    opacity: 0.9,
    fillOpacity: 0.7
  }}).addTo(map);
  marker.bindPopup(d.popup, {{maxWidth:300}});
}});

// Vessel position
L.marker([{vessel_pos[0]},{vessel_pos[1]}], {{
  icon: L.divIcon({{html:'⛵',className:'',iconSize:[20,20]}})
}}).addTo(map).bindPopup('母港 / 船位');
</script></body></html>'''

    Path(output_path).write_text(html, encoding="utf-8")
    size_kb = Path(output_path).stat().st_size / 1024
    log.info(f"  HTML Map: {output_path} ({size_kb:.1f} KB, {len(hotspots)} hotspots)")
