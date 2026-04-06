"""
OceanMaster v9.0 — GeoJSON 輸出 + 衛星 Email 傳送
====================================================
🟡 補充功能

格式:
  - GeoJSON: OpenCPN, QGIS 等導航軟體
  - Simplified JSON: 衛星 Email 壓縮傳輸 (<10KB)
"""

import json
import logging
import smtplib
import gzip
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

log = logging.getLogger("OceanMaster.Output")


def generate_geojson(
    fusion_result: Dict[str, Any],
    route_result: Optional[Any] = None,
    eez_checks: Optional[List[Dict]] = None,
    output_dir: str = "output",
) -> str:
    """
    生成 GeoJSON 檔案 (OpenCPN 可直接載入)

    Features:
    - 漁場熱點 (Point + 評分/物種/建議)
    - 建議航線 (LineString)
    - EEZ 邊界警告 (Polygon)
    """
    features = []

    # ── 1. 熱點 ──
    hotspots = fusion_result.get("combined_hotspots", [])
    for h in hotspots:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [h["lon"], h["lat"]],
            },
            "properties": {
                "name": f"#{h.get('rank', '?')} {h.get('species', 'unknown')}",
                "score": round(h.get("score", 0) * 100),
                "species": h.get("species"),
                "distance_nm": h.get("distance_nm", 0),
                "travel_hours": h.get("travel_hours", 0),
                "eez": h.get("eez_name", "公海"),
                "safety": h.get("safety", ""),
                "marker-color": _score_to_color(h.get("score", 0)),
                "marker-size": "large" if h.get("score", 0) > 0.8 else "medium",
                "marker-symbol": "harbor",
            },
        }
        features.append(feature)

    # ── 2. 航線 ──
    if route_result and hasattr(route_result, "waypoints"):
        coords = [[wp["lon"], wp["lat"]] for wp in route_result.waypoints]
        if coords:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords,
                },
                "properties": {
                    "name": "建議航線",
                    "type": route_result.route_type,
                    "distance_nm": round(route_result.total_distance_nm, 1),
                    "time_hours": round(route_result.total_time_hours, 1),
                    "fuel_savings_pct": round(route_result.fuel_savings_pct, 1),
                    "stroke": "#4fc3f7",
                    "stroke-width": 3,
                    "stroke-opacity": 0.8,
                },
            })

    # ── 3. EEZ 警告點 ──
    if eez_checks:
        for check in eez_checks:
            if check.get("severity") in ("warning", "danger"):
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [check["lon"], check["lat"]],
                    },
                    "properties": {
                        "name": f"⚠ {check.get('eez_name', 'EEZ')}",
                        "warning": check.get("warning"),
                        "severity": check.get("severity"),
                        "marker-color": "#ff0000" if check["severity"] == "danger" else "#ff9800",
                        "marker-symbol": "danger",
                    },
                })

    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "generator": "OceanMaster v9.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_hotspots": len(hotspots),
        },
    }

    # 保存
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    filepath = output_path / f"OceanMaster_{timestamp}.geojson"

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)

    size_kb = filepath.stat().st_size / 1024
    log.info(f"GeoJSON 生成: {filepath} ({size_kb:.1f}KB)")

    return str(filepath)


def generate_satellite_email_payload(
    hotspots: List[Dict],
    max_size_kb: int = 10,
) -> str:
    """
    生成極度壓縮的衛星 Email 載荷

    目標: < 10KB（衛星通信限制）
    格式: 純文字，每行一個熱點
    """
    lines = [
        f"OM v9 | {datetime.now(timezone.utc).strftime('%Y%m%d %H%M')} UTC",
        f"TOP {len(hotspots)} HOTSPOTS",
        "---",
    ]

    for h in hotspots[:15]:  # 最多 15 個
        sp_short = {
            "skipjack": "SKJ", "yellowfin": "YFT",
            "bigeye": "BET", "squid": "SQD",
        }.get(h.get("species", ""), "UNK")

        line = (
            f"#{h.get('rank', '?'):2d} {sp_short} "
            f"{h['lat']:6.2f}N {h['lon']:7.2f}E "
            f"S:{h.get('score', 0):.0%} "
            f"D:{h.get('distance_nm', 0):.0f}nm "
            f"T:{h.get('travel_hours', 0):.0f}h"
        )

        eez = h.get("eez_name", "")
        if eez and eez != "公海":
            line += f" ⚠{eez}"

        lines.append(line)

    lines.append("---")
    lines.append("Safe fishing!")

    payload = "\n".join(lines)

    # 確保在大小限制內
    while len(payload.encode()) > max_size_kb * 1024:
        lines.pop(-3)  # 移除最後一個熱點
        payload = "\n".join(lines)

    return payload


async def send_satellite_email(
    payload: str,
    to_email: str,
    smtp_host: str = "smtp.gmail.com",
    smtp_port: int = 587,
    username: str = "",
    password: str = "",
    subject: str = "OceanMaster Fishing Report",
) -> bool:
    """
    透過 SMTP 發送衛星 Email

    船上衛星通信系統（Iridium GO, Fleet One 等）
    接收純文字 Email 就好
    """
    if not username or not to_email:
        log.warning("Email 設定不完整，跳過衛星傳送")
        return False

    try:
        msg = MIMEMultipart()
        msg["From"] = username
        msg["To"] = to_email
        msg["Subject"] = subject

        msg.attach(MIMEText(payload, "plain", "utf-8"))

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(username, password)
            server.send_message(msg)

        log.info(f"衛星 Email 已發送至 {to_email}")
        return True

    except Exception as e:
        log.error(f"衛星 Email 發送失敗: {e}")
        return False


def _score_to_color(score: float) -> str:
    if score >= 0.8:
        return "#e53935"  # 紅色
    elif score >= 0.6:
        return "#ff9800"  # 橘色
    else:
        return "#fdd835"  # 黃色
