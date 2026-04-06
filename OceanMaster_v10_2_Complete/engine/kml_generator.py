"""
OceanMaster v10.2.2 — KML 漁場地圖生成
========================================
v10.2.2 修正:
  - generate_kml() 簽名對齊 main_v10_2.py 呼叫方式
  - 支援兩種呼叫: (hotspots, lats, lons, ...) 或 (fusion_result_dict)
  - Φ 顯示真實值（不截斷）
  - convergence clip 到 100%
  - HTTPS icon URLs
  - 商業級描述框（Φ/SST/NPP/鋒面/渦旋/EEZ）

圖標:
  🔴 紅 = 最佳 (HSI ≥ 75%)
  🟠 橙 = 優秀 (60-75%)
  🟡 黃 = 良好 (45-60%)
  ⚪ 白 = 普通 (< 45%)
"""

import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging
import zipfile

log = logging.getLogger("OceanMaster.KML")

# 物種中文名
SP_ZH = {
    "skipjack": "鰹魚", "yellowfin": "黃鰭鮪", "bigeye": "大目鮪",
    "albacore": "長鰭鮪", "squid_todarodes": "赤魷", "squid_ommastrephes": "劍尖魷",
    "squid": "魷魚",
}

# Φ 臨界值 (Deutsch 2015)
PHI_CRIT = {
    "skipjack": 3.0, "yellowfin": 3.0, "bigeye": 3.5,
    "albacore": 3.5, "squid_todarodes": 2.8, "squid_ommastrephes": 2.8,
}


def generate_kml(
    hotspots_or_fusion: Union[List[Dict], Dict[str, Any]],
    lats: Optional[np.ndarray] = None,
    lons: Optional[np.ndarray] = None,
    vessel_lat: float = 25.13,
    vessel_lon: float = 121.74,
    output_path: str = None,
    # 舊版相容參數
    sst_data: Optional[Dict] = None,
    front_data: Optional[Dict] = None,
    output_dir: str = "output",
    filename: str = None,
) -> str:
    """
    生成 v10.2 商業級 KML

    支援兩種呼叫方式:
      1. generate_kml(hotspot_list, lats, lons, output_path=...) — v10.2 main
      2. generate_kml(fusion_result_dict, output_dir=...) — v8 相容
    """
    # ── 統一為 hotspot list ──
    if isinstance(hotspots_or_fusion, dict):
        hotspots = hotspots_or_fusion.get("combined_hotspots", [])
        area = hotspots_or_fusion.get("area", {})
    else:
        hotspots = hotspots_or_fusion
        area = {}
        if lats is not None and lons is not None:
            area = {
                "lat_min": float(lats.min()), "lat_max": float(lats.max()),
                "lon_min": float(lons.min()), "lon_max": float(lons.max()),
            }

    # ── 輸出路徑 ──
    if output_path:
        filepath = Path(output_path)
    elif filename:
        filepath = Path(output_dir) / filename
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        filepath = Path(output_dir) / f"OceanMaster_v102_{ts}.kml"

    filepath.parent.mkdir(parents=True, exist_ok=True)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── 按物種分組 ──
    species_groups = {}
    for h in hotspots:
        sp = h.get("species", "unknown")
        species_groups.setdefault(sp, []).append(h)

    # ── 生成 KML ──
    parts = [_header(now_str, area)]

    # 漁場熱點
    parts.append('<Folder><name>🎯 v10.2 漁場熱點</name><open>1</open>')
    parts.append(f'<description>Φ(Deutsch2015真實計算) + SEAPODYM棲息地 + 營養鏈\n'
                 f'Φ不截斷 + EEZ地理校正 + HSI幾何平均\n'
                 f'生成: {now_str}</description>')

    for sp, sp_hotspots in species_groups.items():
        sp_zh = SP_ZH.get(sp, sp)
        parts.append(f'<Folder><name>{sp_zh}</name>')

        for idx, h in enumerate(sp_hotspots, 1):
            parts.append(_hotspot_placemark(h, idx, sp, sp_zh))

        parts.append('</Folder>')

    parts.append('</Folder>')

    # 分析範圍
    if area:
        lat_min = area.get("lat_min", 5)
        lat_max = area.get("lat_max", 35)
        lon_min = area.get("lon_min", 120)
        lon_max = area.get("lon_max", 175)
        parts.append(f'''<Folder><name>分析範圍</name><Placemark><styleUrl>#s_area</styleUrl>
<Polygon><outerBoundaryIs><LinearRing><coordinates>
{lon_min},{lat_min},0 {lon_max},{lat_min},0 {lon_max},{lat_max},0 {lon_min},{lat_max},0 {lon_min},{lat_min},0
</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark></Folder>''')

    parts.append(_footer())

    kml_content = "\n".join(parts)
    filepath.write_text(kml_content, encoding="utf-8")
    size_kb = filepath.stat().st_size / 1024
    log.info(f"  KML: {filepath} ({size_kb:.1f} KB, {len(hotspots)} hotspots)")

    # 超 100KB → KMZ
    if size_kb > 100:
        kmz_path = filepath.with_suffix(".kmz")
        with zipfile.ZipFile(str(kmz_path), "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(str(filepath), "doc.kml")
        log.info(f"  KMZ: {kmz_path} ({kmz_path.stat().st_size/1024:.1f} KB)")
        return str(kmz_path)

    return str(filepath)


def _header(timestamp: str, area: dict) -> str:
    lat_c = (area.get("lat_min", 15) + area.get("lat_max", 30)) / 2
    lon_c = (area.get("lon_min", 130) + area.get("lon_max", 155)) / 2
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
<name>🎣 OceanMaster v10.2 漁場預測</name>
<description>生成: {timestamp}
⚠️ 本預測僅供參考，實際作業請依據船長經驗和即時觀測</description>
<Style id="s_best"><IconStyle><scale>1.3</scale><Icon><href>https://maps.google.com/mapfiles/kml/paddle/red-circle.png</href></Icon></IconStyle></Style>
<Style id="s_good"><IconStyle><scale>1.1</scale><Icon><href>https://maps.google.com/mapfiles/kml/paddle/orange-circle.png</href></Icon></IconStyle></Style>
<Style id="s_fair"><IconStyle><scale>0.9</scale><Icon><href>https://maps.google.com/mapfiles/kml/paddle/ylw-circle.png</href></Icon></IconStyle></Style>
<Style id="s_low"><IconStyle><scale>0.7</scale><Icon><href>https://maps.google.com/mapfiles/kml/paddle/wht-circle.png</href></Icon></IconStyle></Style>
<Style id="s_area"><PolyStyle><color>20ffffff</color><outline>1</outline></PolyStyle><LineStyle><color>ff0088ff</color><width>1</width></LineStyle></Style>
<LookAt><longitude>{lon_c}</longitude><latitude>{lat_c}</latitude><range>2500000</range></LookAt>'''


def _hotspot_placemark(h: dict, idx: int, sp: str, sp_zh: str) -> str:
    score = h.get("score", 0)
    lat = h.get("lat", 0)
    lon = h.get("lon", 0)
    pct = int(round(score * 100))

    # 等級 + 樣式
    if score >= 0.75:
        style, grade, grade_label = "s_best", "🔴", "最佳"
    elif score >= 0.60:
        style, grade, grade_label = "s_good", "🟠", "優秀"
    elif score >= 0.45:
        style, grade, grade_label = "s_fair", "🟡", "良好"
    else:
        style, grade, grade_label = "s_low", "⚪", "偏低"

    # Φ 資訊 (不截斷)
    phi = h.get("phi", None)
    phi_crit = PHI_CRIT.get(sp, 3.0)
    if phi is not None:
        phi_status = "✅充足" if phi >= phi_crit else "⚠️偏低"
        phi_line = f"🧬 代謝指數Φ: {phi:.2f} {phi_status} (Φ_crit={phi_crit})\nΦ真實計算: Deutsch 2015 (非截斷)"
    else:
        phi_line = "🧬 Φ: 無數據"

    # SST
    sst = h.get("sst", None)
    thermal = h.get("h_thermal", h.get("thermal_pct", None))
    sst_line = f"🌡️ SST: {sst:.1f}°C" if sst else "🌡️ SST: N/A"
    if thermal is not None:
        t_pct = thermal * 100 if thermal <= 1 else thermal
        sst_line += f" (適宜: {t_pct:.0f}%)"

    # 生態參數
    npp = h.get("npp", None)
    chl = h.get("chl", None)
    do = h.get("do_surface", h.get("do", None))
    eco_parts = []
    if npp: eco_parts.append(f"NPP: {npp:.0f} mgC/m²/d")
    if chl: eco_parts.append(f"Chl: {chl:.3f}")
    if do: eco_parts.append(f"DO: {do:.1f} ml/L")
    eco_line = "🌿 " + " | ".join(eco_parts) if eco_parts else ""

    # 鋒面/渦旋/匯聚
    front = h.get("front_persistence", h.get("front_pct", None))
    eddy = h.get("eddy_edge", h.get("eddy_pct", None))
    conv = h.get("convergence", h.get("conv_pct", None))
    phys_parts = []
    if front is not None:
        f_pct = front * 100 if front <= 1 else front
        phys_parts.append(f"鋒面: {f_pct:.0f}%")
    if eddy is not None:
        e_pct = eddy * 100 if eddy <= 1 else eddy
        phys_parts.append(f"渦旋: {e_pct:.0f}%")
    if conv is not None:
        c_pct = conv * 100 if conv <= 1 else conv
        c_pct = min(c_pct, 100)  # clip convergence ≤100%
        phys_parts.append(f"海流匯聚: {c_pct:.0f}%")
    phys_line = "🌊 " + " | ".join(phys_parts) if phys_parts else ""

    # 深度
    depth = h.get("depth_m", None)
    depth_line = f"🏔️ 深度: {depth:.0f}m" if depth else ""

    # EEZ
    eez = h.get("eez", "公海")
    eez_line = f"⚓ 水域: {eez}"
    eez_info_obj = None
    try:
        from engine.eez.eez_checker import EEZChecker
        eez_info_obj = EEZChecker.get_zone_info(eez)
    except Exception:
        pass
    if eez_info_obj and eez != "公海":
        eez_line += f"\nEEZ驗證: {eez_info_obj.get('note', '')}"

    # 距離
    dist = h.get("distance_nm", None)
    dist_line = f"📍 距離: {dist:.0f} nm ({h.get('travel_hours', 0):.1f}hr)" if dist else ""

    # SHAP 說明
    explain = h.get("explain", "")
    explain_line = f"📊 {explain}" if explain else ""

    # 組合描述
    desc_parts = [
        f"【v10.2 商業級分析】{grade}{grade_label}",
        f"═══ 綜合 HSI: {pct}% ═══",
        phi_line,
        sst_line,
    ]
    for line in [eco_line, phys_line, depth_line, eez_line, dist_line, explain_line]:
        if line:
            desc_parts.append(line)

    desc = "\n".join(desc_parts)

    return f'''<Placemark><name>#{idx} {sp_zh} ({pct}%)</name>
<description><![CDATA[{desc}]]></description>
<styleUrl>#{style}</styleUrl>
<Point><coordinates>{lon},{lat},0</coordinates></Point></Placemark>'''


def _footer() -> str:
    return '</Document>\n</kml>'
