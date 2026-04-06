"""
OceanMaster Web 服務 v13.2 — 漁船專用即時漁場儀表板
====================================================
提供：
  1. 即時互動地圖 (3-tab dashboard)
  2. 熱點排名列表
  3. SHAP 因子分解 (/api/v1/explain)
  4. 回測對比 (/api/v1/backtest)
  5. 海況總覽 (/api/sea_conditions)
  6. 自動更新 + 離線降級
  7. API Key 認證 (T8)
  8. lifespan (T11, non-deprecated)
  9. CORS + 輸入驗證 (v12)
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Query, Depends, HTTPException, Security
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware  # [v12-fix] CORS support
import asyncio
import time as _time  # [v12-enhance] rate limiter clock
import json
import os
from dotenv import load_dotenv
load_dotenv()  # [v13.2] 自動載入 .env (CMEMS_USER, CMEMS_PASS, etc.)
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
import logging
import numpy as np

# [v15.3-refactor] 共用安全模組 — 統一 rate limiter, sanitizer, session token
from engine.shared_security import (
    rate_check as _rate_check,
    sanitize_for_json as _sanitize_for_json,
    generate_session_token as _shared_generate_session_token,
    verify_session_token as _shared_verify_session_token,
)

# 導入主管線
import sys
sys.path.append(str(Path(__file__).parent))
from main_v10_3 import OceanMasterPipeline, VERSION

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("OceanMaster.Web")


# [v10.5] UTF-8 JSON Response for proper CJK character rendering
class UTF8JSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        # [v15.3-refactor] sanitize_for_json 從 shared_security import，不再有 NameError 風險
        content = _sanitize_for_json(content)
        return json.dumps(
            content, ensure_ascii=False, allow_nan=False, default=str
        ).encode("utf-8")

# ── T8: API Key Authentication ──
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
API_KEY = os.environ.get("OCEANMASTER_API_KEY", "dev-key-change-me")

# [v13.2-R4-P1] Production guard: refuse to start with default key
if API_KEY == "dev-key-change-me" and os.environ.get("OCEANMASTER_ENV") == "production":
    raise SystemExit("FATAL: OCEANMASTER_API_KEY must be changed from default in production")

# [v15.3-refactor] Session token — 使用 shared_security 模組
import secrets
_SESSION_SECRET = os.environ.get("SESSION_SECRET", secrets.token_hex(32))
_SESSION_TTL = 86400  # 24 hours


def _generate_session_token() -> str:
    """生成 HMAC session token (24hr TTL), 注入到 dashboard HTML 取代 master key"""
    return _shared_generate_session_token(_SESSION_SECRET, _SESSION_TTL)


def _verify_session_token(token: str) -> bool:
    """驗證 session token 是否有效 (未過期 + 簽章正確)"""
    return _shared_verify_session_token(token, _SESSION_SECRET, _SESSION_TTL)


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """T8: 驗證 API key — 接受 master key 或 session token"""
    if api_key == API_KEY:
        return  # master key (curl / 後端呼叫)
    if _verify_session_token(api_key):
        return  # session token (dashboard 前端)
    raise HTTPException(status_code=403, detail="Invalid API key")


# ═══════════════════════════════════════════════════
# [v15.3-refactor] Rate limiter — 使用 shared_security 模組
# _rate_check 已從 engine.shared_security import
# ═══════════════════════════════════════════════════


# 靜態檔案 (dashboard.js etc.)
WEB_DIR = Path(__file__).parent / "web"

# ── SHAP 因子 → 中文 label + 學術引用 對照表 ──
FACTOR_MAP = {
    "sst":              {"label": "海表溫度適宜",       "ref": "Boyce et al. 2008"},
    "sst_gradient":     {"label": "海洋鋒面活躍",       "ref": "Druon et al. 2012, PLoS ONE"},
    "front_strength":   {"label": "溫度鋒面強度",       "ref": "Druon et al. 2012, PLoS ONE"},
    "front_persistence":{"label": "穩定鋒面持續",       "ref": "Druon et al. 2012, PLoS ONE"},
    "dist_to_front":    {"label": "靠近海洋鋒面",       "ref": "Druon et al. 2012, PLoS ONE"},
    "dist_to_eddy":     {"label": "中尺度渦旋邊緣",     "ref": "Scales et al. 2014, Prog. Ocean."},
    "eddy_edge":        {"label": "渦旋邊緣聚集",       "ref": "Scales et al. 2014, Prog. Ocean."},
    "ftle":             {"label": "拉格朗日聚集線",      "ref": "d'Ovidio et al. 2004, GRL"},
    "ftle_ridge":       {"label": "FTLE 脊線活躍",      "ref": "d'Ovidio et al. 2004, GRL"},
    "thermocline_depth":{"label": "溫躍層深度適宜",     "ref": "Lehodey et al. 2008, Deep-Sea Res."},
    "d20_depth":        {"label": "20°C 等溫線深度",    "ref": "Lehodey et al. 2008, Deep-Sea Res."},
    "mld":              {"label": "混合層深度",          "ref": "Lehodey et al. 2008, Deep-Sea Res."},
    "chl_log":          {"label": "葉綠素濃度 (餌料)",  "ref": "Behrenfeld & Falkowski 1997"},
    "chl_gradient":     {"label": "葉綠素梯度 (鋒面)",  "ref": "Behrenfeld & Falkowski 1997"},
    "chl_30d_anomaly":  {"label": "葉綠素異常升高",     "ref": "Behrenfeld & Falkowski 1997"},
    "dist_to_seamount": {"label": "海底山地形抬升",     "ref": "GEBCO"},
    "dist_to_shelf_break":{"label":"大陸棚邊緣",        "ref": "GEBCO"},
    "bathy_depth":      {"label": "水深適宜",           "ref": "GEBCO"},
    "bathy_slope":      {"label": "海底地形坡度",       "ref": "GEBCO"},
    "moon_phase":       {"label": "月相有利",           "ref": "Bigelow et al. 1999"},
    "sst_7d_trend":     {"label": "SST 趨勢上升",      "ref": "GreenFish 8-day concept"},
    "ssh":              {"label": "海面高度異常",       "ref": "Polovina et al. 2001"},
    "current_speed":    {"label": "洋流速度",           "ref": "Kimura et al. 2010"},
    "h_thermal":        {"label": "溫度適宜度",         "ref": "SEAPODYM"},
    "phi_viability":    {"label": "代謝可行性 (Φ)",     "ref": "Deutsch et al. 2015, Science"},
    "h_feeding":        {"label": "餌料可及性",         "ref": "Lehodey et al. 2008"},
}

# ─── Fallback Demo Hotspots (when external APIs unavailable) ───
# [v15.3-fix] All ecological fields included so frontend detail cards display correctly
_DEMO_HOTSPOTS = [
    {"rank": 1, "lat": 24.5, "lon": 141.2, "species": "yellowfin", "score": 0.92, "hsi": 0.92, "is_demo": True,
     "sst": 27.3, "chl": 0.28, "do_ml": 4.8, "do_surface": 4.8, "ssh": 0.12,
     "phi": 4.2, "phi_viability": 4.2, "confidence": 0.88,
     "depth_m": 4850, "npp": 3200, "z20_m": 185, "mld_m": 65, "dvm_depth_m": 95,
     "feeding_index": 0.72, "zoo_proxy": 0.45, "omz_net_effect": 0.0,
     "forage_pct": 72, "front_persistence": 0.65, "eddy_edge": 0.3, "eke": 0.15,
     "lunar_phase": "上弦月", "lunar_illumination": 0.48, "eez": "公海",
     "analysis": "黑潮暖流與冷渦旋交匯形成強溫度鋒面，葉綠素適中，代謝指數Φ=4.2（適宜），SST 27.3°C 位於黃鰭鮪最適溫度範圍"},
    {"rank": 2, "lat": 18.8, "lon": 151.5, "species": "yellowfin", "score": 0.89, "hsi": 0.89, "is_demo": True,
     "sst": 28.1, "chl": 0.35, "do_ml": 4.5, "do_surface": 4.5, "ssh": 0.08,
     "phi": 3.9, "phi_viability": 3.9, "confidence": 0.85,
     "depth_m": 5200, "npp": 2800, "z20_m": 210, "mld_m": 55, "dvm_depth_m": 90,
     "feeding_index": 0.65, "zoo_proxy": 0.38, "omz_net_effect": 0.0,
     "forage_pct": 65, "front_persistence": 0.42, "eddy_edge": 0.1, "eke": 0.08,
     "lunar_phase": "上弦月", "lunar_illumination": 0.48, "eez": "公海",
     "analysis": "赤道逆流北緣，海底山地形抬升帶來深層營養鹽，浮游生物豐富區"},
    {"rank": 3, "lat": 28.2, "lon": 134.8, "species": "bigeye", "score": 0.87, "hsi": 0.87, "is_demo": True,
     "sst": 24.5, "chl": 0.42, "do_ml": 4.1, "do_surface": 4.1, "ssh": -0.05,
     "phi": 3.6, "phi_viability": 3.6, "confidence": 0.82,
     "depth_m": 3800, "npp": 3500, "z20_m": 120, "mld_m": 80, "dvm_depth_m": 130,
     "feeding_index": 0.58, "zoo_proxy": 0.52, "omz_net_effect": 0.0,
     "forage_pct": 58, "front_persistence": 0.78, "eddy_edge": 0.55, "eke": 0.22,
     "lunar_phase": "上弦月", "lunar_illumination": 0.48, "eez": "日本 EEZ",
     "analysis": "冷渦旋邊緣，溫躍層抬升至80m，大目鮪深潛覓食理想區，FTLE聚集線明顯"},
    {"rank": 4, "lat": 12.5, "lon": 145.3, "species": "skipjack", "score": 0.85, "hsi": 0.85, "is_demo": True,
     "sst": 29.2, "chl": 0.18, "do_ml": 4.3, "do_surface": 4.3, "ssh": 0.15,
     "phi": 3.4, "phi_viability": 3.4, "confidence": 0.80,
     "depth_m": 4200, "npp": 1800, "z20_m": 250, "mld_m": 45, "dvm_depth_m": 60,
     "feeding_index": 0.45, "zoo_proxy": 0.25, "omz_net_effect": 0.0,
     "forage_pct": 45, "front_persistence": 0.28, "eddy_edge": 0.05, "eke": 0.05,
     "lunar_phase": "上弦月", "lunar_illumination": 0.48, "eez": "密克羅尼西亞 EEZ",
     "analysis": "暖池核心區，正柱魚群聚集，表層水溫29°C正魚偏好高溫帶"},
    {"rank": 5, "lat": 32.1, "lon": 142.5, "species": "albacore", "score": 0.83, "hsi": 0.83, "is_demo": True,
     "sst": 20.8, "chl": 0.55, "do_ml": 5.2, "do_surface": 5.2, "ssh": -0.08,
     "phi": 4.5, "phi_viability": 4.5, "confidence": 0.79,
     "depth_m": 5800, "npp": 4200, "z20_m": 95, "mld_m": 110, "dvm_depth_m": 150,
     "feeding_index": 0.82, "zoo_proxy": 0.68, "omz_net_effect": 0.0,
     "forage_pct": 82, "front_persistence": 0.85, "eddy_edge": 0.15, "eke": 0.18,
     "lunar_phase": "上弦月", "lunar_illumination": 0.48, "eez": "日本 EEZ",
     "analysis": "黑潮延伸體北側，親潮交匯帶，Chl-a 0.55 mg/m³ 高生產力區，長鰭鮪偏好冷水環境"},
    {"rank": 6, "lat": 22.3, "lon": 155.7, "species": "yellowfin", "score": 0.81, "hsi": 0.81, "is_demo": True,
     "sst": 26.8, "chl": 0.22, "do_ml": 4.6, "do_surface": 4.6, "ssh": 0.10,
     "phi": 4.0, "phi_viability": 4.0, "confidence": 0.77,
     "depth_m": 5100, "npp": 2200, "z20_m": 200, "mld_m": 60, "dvm_depth_m": 85,
     "feeding_index": 0.55, "zoo_proxy": 0.32, "omz_net_effect": 0.0,
     "forage_pct": 55, "front_persistence": 0.52, "eddy_edge": 0.2, "eke": 0.12,
     "lunar_phase": "上弦月", "lunar_illumination": 0.48, "eez": "公海",
     "analysis": "副熱帶逆流帶，溫鹽鋒面與渦旋邊緣共同作用，餌料魚聚集"},
    {"rank": 7, "lat": 38.5, "lon": 148.2, "species": "neon_flying_squid", "score": 0.90, "hsi": 0.90, "is_demo": True,
     "sst": 17.5, "chl": 1.20, "do_ml": 5.8, "do_surface": 5.8, "ssh": -0.12,
     "phi": 3.8, "phi_viability": 3.8, "confidence": 0.86,
     "depth_m": 6200, "npp": 5500, "z20_m": 75, "mld_m": 95, "dvm_depth_m": 110,
     "feeding_index": 0.88, "zoo_proxy": 0.85, "omz_net_effect": 0.0,
     "forage_pct": 88, "front_persistence": 0.92, "eddy_edge": 0.35, "eke": 0.25,
     "lunar_phase": "新月", "lunar_illumination": 0.05, "eez": "公海",
     "analysis": "親潮-黑潮混合水域，新月期間（月光極低），適合燈火誘魚，Chl-a 1.2 mg/m³ 高餌料密度"},
    {"rank": 8, "lat": 40.2, "lon": 152.8, "species": "neon_flying_squid", "score": 0.86, "hsi": 0.86, "is_demo": True,
     "sst": 16.2, "chl": 1.50, "do_ml": 6.0, "do_surface": 6.0, "ssh": -0.15,
     "phi": 3.5, "phi_viability": 3.5, "confidence": 0.83,
     "depth_m": 5500, "npp": 6200, "z20_m": 65, "mld_m": 105, "dvm_depth_m": 120,
     "feeding_index": 0.90, "zoo_proxy": 0.92, "omz_net_effect": 0.0,
     "forage_pct": 90, "front_persistence": 0.88, "eddy_edge": 0.18, "eke": 0.20,
     "lunar_phase": "新月", "lunar_illumination": 0.05, "eez": "公海",
     "analysis": "北太平洋亞極地鋒面，赤魷產卵洄游路線，DVM活躍夜間升至表層"},
    {"rank": 9, "lat": 35.8, "lon": 139.5, "species": "japanese_flying_squid", "score": 0.84, "hsi": 0.84, "is_demo": True,
     "sst": 14.8, "chl": 2.10, "do_ml": 5.5, "do_surface": 5.5, "ssh": -0.10,
     "phi": 3.3, "phi_viability": 3.3, "confidence": 0.81,
     "depth_m": 1200, "npp": 7500, "z20_m": 55, "mld_m": 85, "dvm_depth_m": 75,
     "feeding_index": 0.92, "zoo_proxy": 0.95, "omz_net_effect": 0.0,
     "forage_pct": 92, "front_persistence": 0.75, "eddy_edge": 0.1, "eke": 0.10,
     "lunar_phase": "新月", "lunar_illumination": 0.05, "eez": "日本 EEZ",
     "analysis": "日本近海，冬季產卵群集中區，Chl-a 2.1 表示春初浮游植物增殖期"},
    {"rank": 10, "lat": 15.2, "lon": 138.6, "species": "bigeye", "score": 0.80, "hsi": 0.80, "is_demo": True,
     "sst": 25.8, "chl": 0.38, "do_ml": 3.9, "do_surface": 3.9, "ssh": 0.05,
     "phi": 3.2, "phi_viability": 3.2, "confidence": 0.76,
     "depth_m": 4600, "npp": 3000, "z20_m": 160, "mld_m": 50, "dvm_depth_m": 140,
     "feeding_index": 0.48, "zoo_proxy": 0.40, "omz_net_effect": 0.12,
     "forage_pct": 48, "front_persistence": 0.35, "eddy_edge": 0.08, "eke": 0.06,
     "lunar_phase": "上弦月", "lunar_illumination": 0.48, "eez": "密克羅尼西亞 EEZ",
     "analysis": "馬里亞納海溝西側，深水湧升區，大目鮪深潛覓食，代謝指數偏低但溫躍層結構適宜"},
    {"rank": 11, "lat": 20.5, "lon": 165.3, "species": "skipjack", "score": 0.78, "hsi": 0.78, "is_demo": True,
     "sst": 28.5, "chl": 0.15, "do_ml": 4.4, "do_surface": 4.4, "ssh": 0.18,
     "phi": 3.7, "phi_viability": 3.7, "confidence": 0.74,
     "depth_m": 4800, "npp": 1500, "z20_m": 240, "mld_m": 40, "dvm_depth_m": 55,
     "feeding_index": 0.38, "zoo_proxy": 0.20, "omz_net_effect": 0.0,
     "forage_pct": 38, "front_persistence": 0.22, "eddy_edge": 0.02, "eke": 0.03,
     "lunar_phase": "上弦月", "lunar_illumination": 0.48, "eez": "馬紹爾群島 EEZ",
     "analysis": "ENSO中性期暖池東擴，正鰹追隨暖水與餌料魚東移"},
    {"rank": 12, "lat": 26.8, "lon": 128.5, "species": "yellowfin", "score": 0.76, "hsi": 0.76, "is_demo": True,
     "sst": 26.2, "chl": 0.32, "do_ml": 4.7, "do_surface": 4.7, "ssh": 0.06,
     "phi": 3.8, "phi_viability": 3.8, "confidence": 0.72,
     "depth_m": 1800, "npp": 2600, "z20_m": 175, "mld_m": 70, "dvm_depth_m": 80,
     "feeding_index": 0.62, "zoo_proxy": 0.42, "omz_net_effect": 0.0,
     "forage_pct": 62, "front_persistence": 0.55, "eddy_edge": 0.12, "eke": 0.09,
     "lunar_phase": "上弦月", "lunar_illumination": 0.48, "eez": "日本 EEZ",
     "analysis": "沖繩以南黑潮主軸，營養鹽從東海輸入，浮游動物充足，混合層深度適中"},
]

# Auto-compute distance and travel time from Kaohsiung Port (22.61N, 120.28E)
import math as _math
_KH_LAT, _KH_LON = 22.61, 120.28
_CRUISE_KNOTS = 12  # 遠洋延繩釣船巡航速度
for _h in _DEMO_HOTSPOTS:
    _dlat = _math.radians(_h["lat"] - _KH_LAT)
    _dlon = _math.radians(_h["lon"] - _KH_LON)
    _a = _math.sin(_dlat / 2) ** 2 + _math.cos(_math.radians(_KH_LAT)) * _math.cos(_math.radians(_h["lat"])) * _math.sin(_dlon / 2) ** 2
    _nm = 2 * _math.asin(_math.sqrt(_a)) * 3440.065  # Earth radius in nm
    _h["distance_nm"] = round(_nm, 1)
    _h["travel_hours"] = round(_nm / _CRUISE_KNOTS, 1)

# [v15.3-refactor] _sanitize_for_json 已從 engine.shared_security import

# ─── Global State ───
latest_data = {
    "hotspots": _DEMO_HOTSPOTS,
    "last_update": datetime.now(timezone.utc).isoformat(),
    "is_updating": False,
    "data_timestamp": datetime.now(timezone.utc).isoformat(),
    "analysis_summary": {"mode": "demo_fallback", "note": "使用合成示範資料（外部API無法連線時自動啟用）"},
}

# [v13.2-P1] Concurrency guards
_update_lock = asyncio.Lock()  # P1-2: prevent concurrent background updates
_analyze_sem = asyncio.Semaphore(1)  # P1-4: at most 1 concurrent analyze
_analyze_cache = {"result": None, "ts": 0.0}  # P1-4: 30-min result cache


# ═══════════════════════════════════════════════════
# Background Task
# ═══════════════════════════════════════════════════
async def update_data_background():
    while True:
        # [v13.2-P1] Skip if previous update still running
        if _update_lock.locked():
            log.warning("⚠️ Previous update still running, skipping this cycle")
            await asyncio.sleep(3600)
            continue
        async with _update_lock:
            try:
                log.info("🔄 開始背景更新（最多等30分鐘）...")
                latest_data["is_updating"] = True

                pipeline = OceanMasterPipeline(
                    lat_range=(5, 35), lon_range=(120, 175),
                    vessel_pos=(22.61, 120.28), output_dir="output",  # 高雄港
                )

                # [v13.2-fix] S-2: 直接 await async pipeline — 不再巢狀事件迴圈
                try:
                    results = await asyncio.wait_for(
                        pipeline.run_full(),
                        timeout=1800  # 30 分鐘
                    )
                except asyncio.TimeoutError:
                    log.warning("⏰ Pipeline 30分鐘 timeout — 保留上次成功數據")
                    latest_data["is_updating"] = False
                    # [v13.2] 不回退 demo，保留上次成功的真實數據
                    if latest_data.get("analysis_summary", {}).get("mode") != "demo_fallback":
                        latest_data["analysis_summary"]["note"] = "管線超時30分鐘，保留上次真實數據"
                    await asyncio.sleep(3600)  # 1小時後重試（非6小時）
                    continue

                # [v13.2-P1] Build new_data dict first, then atomic update
                new_data = {}
                out_dir = Path("output")
                hotspot_files = sorted(out_dir.glob("hotspots_*.json"), reverse=True)
                hp_file = hotspot_files[0] if hotspot_files else out_dir / "hotspots.json"

                if hp_file.exists():
                    with open(hp_file, 'r', encoding='utf-8') as f:
                        new_hotspots = json.load(f)
                    if new_hotspots and len(new_hotspots) > 0:
                        new_data["hotspots"] = new_hotspots
                        new_data["last_update"] = datetime.now(timezone.utc).isoformat()
                        new_data["data_timestamp"] = datetime.now(timezone.utc).isoformat()
                        log.info(f"✅ 更新完成: {len(new_hotspots)} 個熱點（真實資料）")
                    else:
                        log.warning("⚠️ 分析完但無熱點，保留 demo 資料")
                else:
                    log.warning(f"⚠️ 找不到 hotspot 檔案，保留 demo 資料")

                new_data["analysis_summary"] = results.get("analysis_summary", {})
                # [v14.0] Fleet Operations data
                new_data["optimized_route"] = results.get("optimized_route", {})
                new_data["current_shear"] = results.get("current_shear", {})
                new_data["is_updating"] = False
                # Atomic update
                latest_data.update(new_data)

                # [v18] Update latest.json manifest for dashboard pollData
                geojson_files = sorted(out_dir.glob("OceanMaster_*.geojson"), reverse=True)
                if geojson_files:
                    manifest = {
                        "file": geojson_files[0].name,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "n_hotspots": len(new_data.get("hotspots", [])),
                    }
                    with open(out_dir / "latest.json", "w", encoding="utf-8") as f:
                        json.dump(manifest, f, ensure_ascii=False)
                    log.info(f"📄 latest.json → {geojson_files[0].name}")
            except Exception as e:
                log.error(f"❌ 更新失敗: {e}")
                latest_data["is_updating"] = False

        await asyncio.sleep(21600)


# ═══════════════════════════════════════════════════
# T11: lifespan (replaces deprecated @app.on_event)
# ═══════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    """T11: Modern lifespan handler (non-deprecated)."""
    log.info(f"🚀 啟動 OceanMaster Web 服務 v{VERSION}...")

    out_dir = Path("output")

    # [v10.4] SQLite catch_reports DB 初始化
    _init_catch_db()

    # [商用] 優先讀 hotspots.json (非時間戳版，管線每次都寫最新)
    hp_file = out_dir / "hotspots.json"
    if not hp_file.exists():
        hotspot_files = sorted(out_dir.glob("hotspots_*.json"), reverse=True)
        hp_file = hotspot_files[0] if hotspot_files else hp_file

    if hp_file.exists():
        with open(hp_file, 'r', encoding='utf-8') as f:
            latest_data["hotspots"] = json.load(f)
        latest_data["last_update"] = datetime.fromtimestamp(
            hp_file.stat().st_mtime
        ).isoformat()
        latest_data["data_timestamp"] = latest_data["last_update"]
        log.info(f"📊 載入現有數據: {len(latest_data['hotspots'])} 個熱點 (from {hp_file.name})")

    # [商用] 冷啟動讀取 analysis_summary — 讓 sea_conditions API 立即有值
    summary_file = out_dir / "analysis_summary.json"
    if summary_file.exists():
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                latest_data["analysis_summary"] = json.load(f)
            log.info(f"📊 載入 analysis_summary: {list(latest_data['analysis_summary'].keys())}")
        except Exception as e:
            log.warning(f"  analysis_summary load: {e}")

    # [v13.2-P1] Create shared httpx client
    import httpx as _httpx
    app.state.http_client = _httpx.AsyncClient(
        timeout=_httpx.Timeout(30, connect=10),
        limits=_httpx.Limits(max_connections=20, max_keepalive_connections=10),
    )
    log.info("✅ Shared httpx.AsyncClient created")

    asyncio.create_task(update_data_background())
    log.info("✅ 背景更新任務已啟動 (每 6 小時)")

    yield

    # [v13.2-P1] Close shared client on shutdown
    await app.state.http_client.aclose()
    log.info("👋 關閉 OceanMaster 服務")


app = FastAPI(
    title=f"OceanMaster v{VERSION}",
    lifespan=lifespan,
    default_response_class=UTF8JSONResponse,
)

# [audit-fix] S-3: CORS — 預設限制為 localhost，生產環境透過 CORS_ORIGINS 設定
_cors_origins = os.environ.get("CORS_ORIGINS", "http://localhost:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

# Mount static files after app creation
if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")

# [v10.5] Captain PWA
CAPTAIN_DIR = Path("static/captain")
if CAPTAIN_DIR.exists():
    app.mount("/captain", StaticFiles(directory=str(CAPTAIN_DIR), html=True), name="captain")

# [v18] Pipeline output — GeoJSON/KML accessible by dashboard
OUTPUT_STATIC = Path("output")
if OUTPUT_STATIC.exists():
    app.mount("/output", StaticFiles(directory=str(OUTPUT_STATIC)), name="output")
# ═══════════════════════════════════════════════════
# Pages (Public — no auth needed)
# ═══════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
async def root():
    """主頁：3-tab 儀表板 (API key injected at serve time)"""
    dashboard = WEB_DIR / "dashboard.html"
    if dashboard.exists():
        html = dashboard.read_text(encoding="utf-8")
        # [v13.2-fix] S-4: 注入 session token 取代 master key
        html = html.replace("dev-key-change-me", _generate_session_token())
        return HTMLResponse(content=html)
    return HTMLResponse(content=f"<h1>OceanMaster v{VERSION}</h1><p>Dashboard not found</p>")

@app.get("/dashboard.js")
async def serve_js():
    """Serve dashboard JS"""
    js = WEB_DIR / "dashboard.js"
    if js.exists():
        return FileResponse(str(js), media_type="application/javascript")
    return JSONResponse({"error": "JS not found"}, status_code=404)

@app.get("/health")
async def health():
    """Health check — public, no auth needed. [v13.2-P2] data freshness aware."""
    freshness = None
    try:
        dt = datetime.fromisoformat(latest_data["data_timestamp"].replace("Z", "+00:00"))
        freshness = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
    except Exception as e:
        log.debug(f"[降級] web_server.py: {e}")

    status = "healthy"
    if freshness is None or freshness > 24:
        status = "stale"
    elif latest_data.get("analysis_summary", {}).get("mode") == "demo_fallback":
        status = "degraded"

    return JSONResponse({
        "status": status, "version": VERSION,
        "hotspot_count": len(latest_data["hotspots"]),
        "data_freshness_hours": round(freshness, 1) if freshness else None,
    })


# ═══════════════════════════════════════════════════
# Core API (T8: requires API key)
# ═══════════════════════════════════════════════════
@app.get("/api/hotspots", dependencies=[Depends(verify_api_key)])
async def get_hotspots(species: str = Query(None, description="物種過濾")):
    """熱點 + 新鮮度欄位 (A3)"""
    freshness_hours = None
    if latest_data["data_timestamp"]:
        try:
            dt = datetime.fromisoformat(latest_data["data_timestamp"].replace("Z", "+00:00"))
            freshness_hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
        except Exception as e:
            log.debug(f"[降級] web_server.py: {e}")

    hs = latest_data["hotspots"]
    if species:
        hs = [h for h in hs if h.get("species") == species]

    # [v13.2] NaN/Inf 清除 — 防止 JSON 序列化崩潰
    hs = _sanitize_for_json(hs)

    return JSONResponse(_sanitize_for_json({
        "hotspots": hs,
        "last_update": latest_data["last_update"],
        "is_updating": latest_data["is_updating"],
        "data_timestamp": latest_data["data_timestamp"],
        "data_freshness": round(freshness_hours, 1) if freshness_hours else None,
        "data_sources": latest_data.get("analysis_summary", {}).get("data_sources", {}),
        "optimized_route": latest_data.get("optimized_route", {}),
        "current_shear": latest_data.get("current_shear", {}),
        "version": VERSION,
    }))


# [v13.2-R7] Public hotspot summary — no API key required
# Returns limited info (count + top 3) for dashboards that lack auth
@app.get("/api/hotspots/public")
async def get_hotspots_public():
    """Public hotspot summary (no auth, limited data)."""
    hs = latest_data["hotspots"]
    top3 = []
    for h in hs[:3]:
        top3.append({
            "lat": round(h.get("lat", 0), 2),
            "lon": round(h.get("lon", 0), 2),
            "species": h.get("species", ""),
            "score": round(h.get("score", 0), 2),
        })
    return JSONResponse({
        "total_hotspots": len(hs),
        "top_hotspots": top3,
        "last_update": latest_data["last_update"],
        "note": "Full data requires API key via /api/hotspots",
    })

@app.get("/api/analyze", dependencies=[Depends(verify_api_key)])
async def api_analyze(
    request: Request,
    lat_min: float = Query(5, ge=-90, le=90, description="南界緯度"),
    lat_max: float = Query(35, ge=-90, le=90, description="北界緯度"),
    lon_min: float = Query(120, ge=-180, le=180, description="西界經度"),
    lon_max: float = Query(175, ge=-180, le=180, description="東界經度"),
    vessel_lat: float = Query(22.61, ge=-90, le=90, description="船位緯度"),
    vessel_lon: float = Query(120.28, ge=-180, le=180, description="船位經度"),
    date: str = None,
):
    """執行即時或歷史分析 [v13.2-P1] with semaphore + rate check + cache"""
    # [v13.2-P1] Rate check
    ip = request.client.host if request.client else "unknown"
    if not _rate_check(ip):
        raise HTTPException(429, "Rate limit exceeded")

    # [v13.2-P1] Check 30-min cache (only for default params)
    now = _time.time()
    if (date is None and lat_min == 5 and lat_max == 35 and lon_min == 120 and lon_max == 175
            and _analyze_cache["result"] is not None
            and now - _analyze_cache["ts"] < 1800):
        return JSONResponse(_analyze_cache["result"])

    # [v13.2-P1] Semaphore: at most 1 concurrent analysis
    if _analyze_sem.locked():
        raise HTTPException(503, "Analysis already running, please retry later")
    async with _analyze_sem:
        pipeline = OceanMasterPipeline(
            lat_range=(lat_min, lat_max), lon_range=(lon_min, lon_max),
            vessel_pos=(vessel_lat, vessel_lon), target_date=date,
        )
        results = await pipeline.run_full()
        clean = {k: v for k, v in results.items() if not isinstance(v, np.ndarray)}
        # Cache default-param results
        if date is None and lat_min == 5 and lat_max == 35:
            _analyze_cache["result"] = clean
            _analyze_cache["ts"] = _time.time()
        return JSONResponse(clean)

@app.get("/api/status", dependencies=[Depends(verify_api_key)])
async def get_status():
    return JSONResponse({
        "status": "running", "version": VERSION,
        "last_update": latest_data["last_update"],
        "is_updating": latest_data["is_updating"],
        "hotspot_count": len(latest_data["hotspots"]),
    })


# ═══════════════════════════════════════════════════
# /api/sea_conditions (A1 + A4) — requires key
# ═══════════════════════════════════════════════════
@app.get("/api/sea_conditions", dependencies=[Depends(verify_api_key)])
async def get_sea_conditions():
    """海況總覽 — Tab 3 數據源"""
    hs = latest_data["hotspots"]
    summary = latest_data.get("analysis_summary", {})

    ssts = [h.get("sst") for h in hs if h.get("sst") is not None]
    oni = summary.get("enso_oni", 0)

    return {
        "moon_phase": hs[0].get("lunar_phase", "--") if hs else "--",
        "lunar_factor": hs[0].get("lunar_factor", 1.0) if hs else 1.0,
        "sst_min": round(min(ssts), 1) if ssts else None,
        "sst_max": round(max(ssts), 1) if ssts else None,
        "oni": oni,
        "wind_speed": summary.get("wind_speed"),
        "wind_dir": summary.get("wind_dir"),
        "current_speed": summary.get("current_speed"),
        "current_dir": summary.get("current_dir"),
        "data_sources": summary.get("data_sources", {}),
        "greenfish_lite": summary.get("greenfish_lite", False),
        "ml_ensemble": summary.get("ml_ensemble", False),
    }


# [v10.5] Alias: dashboard JS calls /api/sea-conditions (dash)
@app.get("/api/sea-conditions", dependencies=[Depends(verify_api_key)])
async def get_sea_conditions_alias():
    return await get_sea_conditions()


# ═══════════════════════════════════════════════════
# /api/v1/explain (SHAP) — requires key
# ═══════════════════════════════════════════════════
@app.get("/api/v1/explain", dependencies=[Depends(verify_api_key)])
async def explain_hotspot(hotspot_id: int = 0):
    """
    回傳單一漁場的 SHAP 特徵貢獻完整分解。
    Maps raw SHAP feature names → 中文 label + 學術引用。
    """
    hs = latest_data["hotspots"]
    if hotspot_id < 0 or hotspot_id >= len(hs):
        return JSONResponse({"error": "hotspot_id out of range"}, status_code=400)

    h = hs[hotspot_id]
    shap_raw = h.get("shap_values", {})

    # Build factor list with labels and references
    factors = []
    for feat, val in sorted(shap_raw.items(), key=lambda x: abs(x[1]), reverse=True):
        info = FACTOR_MAP.get(feat, {"label": feat, "ref": ""})
        factors.append({
            "feature": feat,
            "label": info["label"],
            "contribution": round(val, 4),
            "reference": info["ref"],
        })

    # If no SHAP, build from available fields
    if not factors:
        field_map = [
            ("h_thermal", "溫度適宜度"), ("phi_viability", "代謝可行性"),
            ("h_feeding", "餌料可及性"), ("front_persistence", "穩定鋒面"),
            ("eddy_edge", "渦旋邊緣"),
        ]
        for fk, label in field_map:
            v = h.get(fk)
            if v is not None and v > 0:
                info = FACTOR_MAP.get(fk, {"label": label, "ref": ""})
                factors.append({
                    "feature": fk, "label": info["label"],
                    "contribution": round(v, 4), "reference": info["ref"],
                })

    # Normalize contributions to sum=1
    total = sum(abs(f["contribution"]) for f in factors) or 1
    for f in factors:
        f["contribution"] = round(f["contribution"] / total, 4)

    return JSONResponse({
        "hotspot_id": hotspot_id,
        "species": h.get("species", ""),
        "hsi": h.get("score", h.get("hsi", 0)),
        "shap_factors": factors[:8],
        "reason_summary": h.get("explain", ""),
        "shap_method": h.get("shap_method", "weight-based"),
    })


# ═══════════════════════════════════════════════════
# /api/v1/backtest — requires key
# ═══════════════════════════════════════════════════
@app.get("/api/v1/backtest", dependencies=[Depends(verify_api_key)])
async def backtest_endpoint(
    date: str = Query(..., description="YYYY-MM-DD"),
    species: str = Query("YFT", description="YFT|BET|SKJ|ALB"),
):
    """
    回傳模型回測結果 vs 實際 GFW 漁船分布。
    Phase 6 dry run 後接入真實 backtest.py 邏輯。
    """
    try:
        from backtest import run_backtest
        import argparse
        # 轉換物種代碼
        sp_map = {"YFT": "yellowfin", "BET": "bigeye", "SKJ": "skipjack", "ALB": "albacore"}
        sp = sp_map.get(species.upper(), species.lower())

        # 簡化執行: 取該日期的預測
        pipeline = OceanMasterPipeline(
            lat_range=(15, 28), lon_range=(120, 135),
            vessel_pos=(22.61, 120.28), target_date=date,  # 高雄港
        )
        results = await pipeline.run_full()
        predicted = [h for h in results.get("hotspots", []) if h.get("species") == sp]

        return JSONResponse({
            "date": date,
            "species": species,
            "predicted_hotspots": predicted[:10],
            "actual_gfw_effort": [],  # Phase 6 接入
            "overlap_score": None,    # Phase 6 計算
            "status": "predicted_only",
        })
    except Exception as e:
        log.error(f"Backtest error: {e}", exc_info=True)
        return JSONResponse({
            "date": date, "species": species,
            "error": "Internal server error", "status": "unavailable",
        })


# ═══════════════════════════════════════════════════
# /api/typhoon-status (v10.3) — requires key
# ═══════════════════════════════════════════════════
@app.get("/api/typhoon-status", dependencies=[Depends(verify_api_key)])
async def get_typhoon_status():
    """活躍颱風 + 危險海域 GeoJSON"""
    try:
        from engine.typhoon_tracker import TyphoonTracker
        tracker = TyphoonTracker()
        alerts = await tracker.fetch_active_typhoons()
        geojson = tracker.to_geojson(alerts)
        return JSONResponse({
            "active_typhoons": len(alerts),
            "alerts": [
                {
                    "id": a.typhoon_id,
                    "typhoon_id": a.typhoon_id,
                    "name": a.name,
                    "lat": a.lat, "lon": a.lon,
                    "intensity_kt": a.intensity_kt,
                    "category": a.category,
                    "source": a.source,
                    "forecast_track": [
                        {
                            "hour": fp.hour,
                            "lat": fp.lat, "lon": fp.lon,
                            "intensity_kt": fp.intensity_kt,
                            "category": fp.category,
                        }
                        for fp in (a.forecast_track or [])
                    ],
                }
                for a in alerts
            ],
            "geojson": geojson,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        log.error(f"Typhoon status error: {e}", exc_info=True)
        return JSONResponse({
            "active_typhoons": 0,
            "alerts": [],
            "error": "Internal server error",
        })


# [v12-fix] 原 /api/system-health 重複定義已合併至 L685 的 lineage 版本
# 參考: Removed duplicate endpoint (was lines 456-489 in v10.4)


# ═══════════════════════════════════════════════════
# [v12-enhance] /api/v1/food_chain — 食物鏈完整剖析
# ═══════════════════════════════════════════════════
@app.get("/api/v1/food_chain", dependencies=[Depends(verify_api_key)])
async def food_chain(
    request: Request,  # [v12-phase11-ratelimit]
    lat: float = Query(25.0, ge=-90, le=90, description="緯度"),
    lon: float = Query(130.0, ge=-180, le=180, description="經度"),
    species: str = Query("yellowfin", description="物種 key"),
):
    """
    [v12] 回傳指定位置的食物鏈生態指標。
    包含 DVM 深度、浮游動物代理、OMZ 壓縮/增益、月相等。
    [v13.2] 從 pipeline 結果讀取真實值，不再用 placeholder。
    """
    if not _rate_check(request.client.host if request.client else "unknown"):  # [v12-phase11-ratelimit]
        raise HTTPException(429, "Rate limit exceeded")

    try:
        # [v13.2] 從 latest_data 找最近的 hotspot 讀真實值
        best_h = None
        best_dist = float("inf")
        for h in latest_data.get("hotspots", []):
            h_lat = h.get("lat", 0)
            h_lon = h.get("lon", 0)
            d = (h_lat - lat) ** 2 + (h_lon - lon) ** 2
            if d < best_dist:
                best_dist = d
                best_h = h

        # DVM 深度
        dvm_depth = 0.0
        if best_h and best_h.get("dvm_depth_m"):
            dvm_depth = best_h["dvm_depth_m"]
        else:
            try:
                from engine.dvm_model import DVMModel
                from engine.species_params import DVM_PARAMS
                dvm_info = DVM_PARAMS.get(species, {})
                day_range = dvm_info.get("day_depth_range", (50, 200))
                night_range = dvm_info.get("night_depth_range", (0, 50))
                hour = datetime.now(timezone.utc).hour
                import math
                day_w = 0.5 * (1 + math.cos(2 * math.pi * (hour - 12) / 24))
                dvm_depth = day_w * sum(day_range)/2 + (1-day_w) * sum(night_range)/2
            except Exception:
                dvm_depth = 100.0

        # 月相
        if best_h and best_h.get("lunar_phase"):
            lunar_phase = best_h["lunar_phase"]
            lunar_illum = best_h.get("lunar_illumination", 0.0)
        else:
            lunar_phase = "未知"
            lunar_illum = 0.0
            try:
                from engine.lunar_model import LunarPhaseEngine
                le = LunarPhaseEngine()
                moon = le.compute_moon_phase(datetime.now(timezone.utc))
                lunar_phase = moon.get("phase_name", "未知")
                lunar_illum = moon.get("illumination", 0.0)
            except Exception as e:
                log.debug(f"[降級] web_server.py: {e}")

        # 浮游動物代理
        zoo_proxy = best_h.get("zoo_proxy", 0.0) if best_h else 0.0

        # OMZ 效應
        omz_compression = best_h.get("omz_net_effect", 0.0) if best_h else 0.0

        return JSONResponse({
            "lat": lat, "lon": lon, "species": species,
            "dvm_depth_m": round(dvm_depth, 1),
            "lunar_phase": lunar_phase,
            "lunar_illumination": round(lunar_illum, 3),
            "zooplankton_proxy": round(zoo_proxy, 3),
            "omz_compression": round(omz_compression, 3),
            "omz_edge_enrichment": round(max(0, omz_compression), 3),
            "food_chain_summary": {
                "primary_producers": "Chl-a → size-fractionated (Brewin 2010)",
                "secondary_producers": f"Zooplankton proxy {zoo_proxy*100:.0f}% (pico/nano/micro weighted)",
                "grazers_to_prey": f"DVM depth ~{dvm_depth:.0f}m ({species})",
                "lunar_effect": f"{lunar_phase} ({lunar_illum*100:.0f}% 光照)",
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        log.error(f"Food chain error: {e}", exc_info=True)
        return JSONResponse({"error": "Internal server error"}, status_code=500)


# ═══════════════════════════════════════════════════
# [v12-squid] /api/v1/squid_jigging_forecast
# ═══════════════════════════════════════════════════
@app.get("/api/v1/squid_jigging_forecast", dependencies=[Depends(verify_api_key)])
async def squid_jigging_forecast(
    lat: float = Query(25.0, description="緯度"),
    lon: float = Query(130.0, description="經度"),
    species: str = Query("neon_flying_squid", description="魷魚物種")
):
    """
    魷魚特製 API (結合月相)
    """
    try:
        from engine.lunar_model import LunarPhaseEngine
        import numpy as np
        le = LunarPhaseEngine()
        moon = le.compute_moon_phase(datetime.now(timezone.utc))
        moon_illum = moon.get("illumination", 0.5)
        # 月相指數: 新月(0.0)最佳(1.0)，滿月(1.0)最差(0.0). illumination 即為此值 (0=新月, 1.0=滿月)
        lunar_index = 1.0 - moon_illum

        return JSONResponse({
            "lat": lat, "lon": lon, "species": species,
            "moon_phase": moon_illum,
            "moon_phase_name": moon.get("phase_name", "未知"),
            "lunar_jigging_index": round(lunar_index, 3),
            "recommendation": "極佳" if lunar_index > 0.8 else "普通" if lunar_index > 0.4 else "不佳",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        log.error(f"Squid forecast error: {e}", exc_info=True)
        return JSONResponse({"error": "Internal server error"}, status_code=500)


# ═══════════════════════════════════════════════════
# [v12-enhance] /api/v1/dvm_profile — 24hr DVM 深度剖面
# ═══════════════════════════════════════════════════
@app.get("/api/v1/dvm_profile", dependencies=[Depends(verify_api_key)])
async def dvm_profile(
    request: Request,  # [v12-phase11-ratelimit]
    species: str = Query("yellowfin", description="物種 key"),
    lat: float = Query(25.0, ge=-90, le=90, description="緯度 (影響日照時長)"),
):
    """
    [v12] 回傳 24 小時 DVM 深度剖面。
    使用 cosine 日夜過渡模型 (v12 dvm_model.py)。
    """
    if not _rate_check(request.client.host if request.client else "unknown"):  # [v12-phase11-ratelimit]
        raise HTTPException(429, "Rate limit exceeded")

    try:
        import math
        from engine.species_params import DVM_PARAMS

        dvm_info = DVM_PARAMS.get(species, {
            "day_depth_range": (50, 200),
            "night_depth_range": (0, 50),
        })
        day_range = dvm_info.get("day_depth_range", (50, 200))
        night_range = dvm_info.get("night_depth_range", (0, 50))
        day_mid = sum(day_range) / 2
        night_mid = sum(night_range) / 2

        profile = []
        for hour in range(24):
            day_w = 0.5 * (1 + math.cos(2 * math.pi * (hour - 12) / 24))
            depth = day_w * day_mid + (1 - day_w) * night_mid
            profile.append({
                "hour_utc": hour,
                "depth_m": round(depth, 1),
                "day_weight": round(day_w, 3),
                "phase": "day" if day_w > 0.5 else "night",
            })

        return JSONResponse({
            "species": species,
            "lat": lat,
            "day_depth_range": list(day_range),
            "night_depth_range": list(night_range),
            "source": dvm_info.get("source", "unknown"),
            "profile_24h": profile,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        log.error(f"DVM profile error: {e}", exc_info=True)
        return JSONResponse({"error": "Internal server error"}, status_code=500)


# ═══════════════════════════════════════════════════
# /api/v1/report_catch (v10.4) — 漁獲回報 API
# ═══════════════════════════════════════════════════
CATCH_DB_PATH = Path("data/catch_reports.db")


def _init_catch_db():
    """Initialize SQLite catch reports database. [v13.2-P2] Uses context manager."""
    CATCH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(CATCH_DB_PATH)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS catch_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vessel_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                target_species TEXT NOT NULL,
                actual_cpue_kg_day REAL NOT NULL,
                predicted_cpue REAL,
                env_snapshot TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_catch_species
            ON catch_reports(target_species, timestamp)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_catch_vessel
            ON catch_reports(vessel_id, timestamp)
        """)
        conn.commit()
    log.info(f"  Catch DB: initialized at {CATCH_DB_PATH}")


@app.post("/api/v1/report_catch", dependencies=[Depends(verify_api_key)])
async def report_catch(request: Request):
    """
    漁獲回報 API — Ground Truth 數據飛輪。

    Request body:
    {
        "vessel_id": "CT6-1234",
        "timestamp": "2026-02-14T08:00:00Z",
        "lat": 24.5, "lon": 140.2,
        "target_species": "yellowfin",
        "actual_cpue_kg_day": 85.0,
        "predicted_cpue": 72.3,
        "env_snapshot": { "sst": 28.1, "sss": 34.8, "eke": 0.012, ... }
    }
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    # Required fields
    required = ["vessel_id", "timestamp", "lat", "lon",
                "target_species", "actual_cpue_kg_day"]
    missing = [f for f in required if f not in body]
    if missing:
        return JSONResponse(
            {"error": f"Missing required fields: {missing}"},
            status_code=400,
        )

    # Validate
    try:
        lat = float(body["lat"])
        lon = float(body["lon"])
        cpue = float(body["actual_cpue_kg_day"])
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):  # [v13.2-fix] A-5
            raise ValueError("lat/lon out of range")
        if cpue < 0:
            raise ValueError("CPUE cannot be negative")
    except (ValueError, TypeError) as e:
        return JSONResponse({"error": "Invalid input parameters"}, status_code=400)

    # env_snapshot → JSON string
    env_snapshot = body.get("env_snapshot")
    env_json = json.dumps(env_snapshot) if env_snapshot else None

    # [v13.2-fix] A-7: SQLite context manager — 避免連線洩漏
    try:
        with sqlite3.connect(str(CATCH_DB_PATH)) as conn:
            conn.execute(
                """
                INSERT INTO catch_reports
                    (vessel_id, timestamp, lat, lon, target_species,
                     actual_cpue_kg_day, predicted_cpue, env_snapshot)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    body["vessel_id"],
                    body["timestamp"],
                    lat, lon,
                    body["target_species"],
                    cpue,
                    body.get("predicted_cpue"),
                    env_json,
                ),
            )
            conn.commit()
            report_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            total = conn.execute("SELECT COUNT(*) FROM catch_reports").fetchone()[0]

        log.info(
            f"  Catch report #{report_id}: {body['vessel_id']} "
            f"{body['target_species']} {cpue:.1f}kg/day @ ({lat:.2f},{lon:.2f})"
        )

        return JSONResponse({
            "status": "ok",
            "report_id": report_id,
            "total_reports": total,
            "message": f"Report saved. {total} total reports in database."
                       + (f" ({500 - total} more needed for retraining.)" if total < 500 else ""),
        })

    except Exception as e:
        log.error(f"  Catch report DB error: {e}", exc_info=True)
        return JSONResponse({"error": "Internal server error"}, status_code=500)


@app.get("/api/v1/catch_stats", dependencies=[Depends(verify_api_key)])
async def get_catch_stats():
    """漁獲回報統計 — 總數、各魚種分佈、最近回報"""
    # [v13.2-fix] A-7: SQLite context manager
    try:
        with sqlite3.connect(str(CATCH_DB_PATH)) as conn:
            total = conn.execute("SELECT COUNT(*) FROM catch_reports").fetchone()[0]

            species_stats = conn.execute(
                "SELECT target_species, COUNT(*), AVG(actual_cpue_kg_day) "
                "FROM catch_reports GROUP BY target_species"
            ).fetchall()

            recent = conn.execute(
                "SELECT vessel_id, timestamp, target_species, actual_cpue_kg_day "
                "FROM catch_reports ORDER BY created_at DESC LIMIT 10"
            ).fetchall()

        return JSONResponse({
            "total_reports": total,
            "retrain_ready": total >= 500,
            "species": [
                {"species": s[0], "count": s[1], "avg_cpue": round(s[2], 1)}
                for s in species_stats
            ],
            "recent": [
                {"vessel": r[0], "time": r[1], "species": r[2], "cpue": r[3]}
                for r in recent
            ],
        })
    except Exception as e:
        log.error(f"Catch stats error: {e}", exc_info=True)
        return JSONResponse({"total_reports": 0, "error": "Internal server error"})


# ═══════════════════════════════════════════════════
# [v10.5] Data Lineage & System Health Dashboard
# ═══════════════════════════════════════════════════
# In-memory lineage tracker (updated by pipeline runs)
_lineage_log = {
    "sources": {},           # source_name → {"last_success": ts, "consecutive_fallback": 0}
    "nan_blockage_pct": 0.0, # % of grid cells blocked due to NaN
    "total_runs": 0,
    "total_fallbacks": 0,
}


def update_lineage(source: str, is_fallback: bool, nan_pct: float = 0.0):
    """Pipeline 每次 fetch 後呼叫，更新血統記錄"""
    if source not in _lineage_log["sources"]:
        _lineage_log["sources"][source] = {
            "last_success": None, "consecutive_fallback": 0,
            "total_calls": 0, "fallback_calls": 0,
        }
    entry = _lineage_log["sources"][source]
    entry["total_calls"] += 1
    if is_fallback:
        entry["fallback_calls"] += 1
        entry["consecutive_fallback"] += 1
        _lineage_log["total_fallbacks"] += 1
    else:
        entry["consecutive_fallback"] = 0
        entry["last_success"] = datetime.now(timezone.utc).isoformat()
    _lineage_log["total_runs"] += 1
    _lineage_log["nan_blockage_pct"] = nan_pct


@app.get("/api/system-health", dependencies=[Depends(verify_api_key)])
async def system_health():
    """v10.5 真實性溯源監控面板 — 資料血統健康度"""
    sources_health = {}
    for src, info in _lineage_log["sources"].items():
        total = max(info["total_calls"], 1)
        fb_rate = round(info["fallback_calls"] / total * 100, 1)
        consec = info["consecutive_fallback"]
        status = "🟢" if consec == 0 else ("🟡" if consec < 3 else "🔴")
        sources_health[src] = {
            "fallback_rate_pct": fb_rate,
            "consecutive_fallback_days": consec,
            "last_success": info["last_success"],
            "total_calls": total,
            "status": status,
        }

    return JSONResponse({
        "status": "healthy" if _lineage_log["total_fallbacks"] == 0 else "degraded",
        "nan_blockage_rate_pct": round(_lineage_log["nan_blockage_pct"], 1),
        "total_pipeline_runs": _lineage_log["total_runs"],
        "total_fallback_events": _lineage_log["total_fallbacks"],
        "sources": sources_health,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

# ═══════════════════════════════════════════════════
# [v12-phase8] 食物鏈時間線 API
# ═══════════════════════════════════════════════════
@app.get("/api/v1/food_chain_timeline", dependencies=[Depends(verify_api_key)])
async def food_chain_timeline(
    request: Request,
    lat: float = Query(25.0, ge=-90, le=90, description="緯度"),
    lon: float = Query(130.0, ge=-180, le=180, description="經度"),
    species: str = Query("yellowfin", description="物種 key"),
    forecast_days: int = Query(8, ge=1, le=30, description="預報天數"),
):
    """
    [v12-phase8] 完整食物鏈級聯時間線。
    回傳從浮游植物爆發到鮪魚到達的完整營養級級聯預測。
    """
    if not _rate_check(request.client.host if request.client else "unknown"):
        raise HTTPException(429, "Rate limit exceeded")

    try:
        from engine.food_chain_predictor import FoodChainPredictor
        fcp = FoodChainPredictor()

        # 從現有熱點數據中查找最近的環境數據
        ocean_data = _find_nearest_ocean_data(lat, lon)

        timeline = fcp.compute_food_chain_timeline(
            lat=lat, lon=lon, date=None,
            species=species, ocean_data=ocean_data,
        )
        timeline["forecast_days"] = forecast_days
        return JSONResponse(timeline)

    except Exception as e:
        log.error(f"Food chain timeline error: {e}", exc_info=True)
        return JSONResponse({"error": "Internal server error"}, status_code=500)


@app.get("/api/v1/feeding_windows", dependencies=[Depends(verify_api_key)])
async def feeding_windows(
    request: Request,
    lat: float = Query(25.0, ge=-90, le=90, description="緯度"),
    lon: float = Query(130.0, ge=-180, le=180, description="經度"),
    species: str = Query("yellowfin", description="物種 key"),
):
    """
    [v12-phase8] 最佳進食時段。
    回傳未來 3 天的最佳捕獲時段 (基於 DVM 晨昏模式)。
    """
    if not _rate_check(request.client.host if request.client else "unknown"):
        raise HTTPException(429, "Rate limit exceeded")

    try:
        from engine.fish_behavior_model import FishBehaviorModel
        fbm = FishBehaviorModel()

        # 取得月相
        lunar_phase = None
        try:
            from engine.lunar_model import LunarPhaseEngine
            le = LunarPhaseEngine()
            lunar_phase = le.compute_moon_phase(datetime.now(timezone.utc))
        except Exception as e:
            log.debug(f"[降級] web_server.py: {e}")

        result = fbm.estimate_feeding_windows(
            lat=lat, lon=lon, date=datetime.now(timezone.utc),
            species=species, lunar_phase=lunar_phase,
        )
        return JSONResponse(result)

    except Exception as e:
        log.error(f"Feeding windows error: {e}", exc_info=True)
        return JSONResponse({"error": "Internal server error"}, status_code=500)


@app.get("/api/v1/fish_movement", dependencies=[Depends(verify_api_key)])
async def fish_movement(
    request: Request,
    lat: float = Query(25.0, ge=-90, le=90, description="緯度"),
    lon: float = Query(130.0, ge=-180, le=180, description="經度"),
    species: str = Query("yellowfin", description="物種 key"),
):
    """
    [v12-phase8] 鮪魚遷移預測。
    回傳未來 3 天的預測遷移方向與速度。
    """
    if not _rate_check(request.client.host if request.client else "unknown"):
        raise HTTPException(429, "Rate limit exceeded")

    try:
        from engine.fish_behavior_model import FishBehaviorModel
        fbm = FishBehaviorModel()

        # 從熱點資料推斷 SST 梯度
        ocean_data = _find_nearest_ocean_data(lat, lon)
        sst_gradient = None
        current_vectors = None
        if ocean_data:
            sst_gradient = ocean_data.get("sst_gradient")
            current_vectors = ocean_data.get("current_vectors")

        month = datetime.now(timezone.utc).month
        result = fbm.predict_migration_direction(
            sst_gradient=sst_gradient,
            current_vectors=current_vectors,
            historical_pattern=None,
            month=month,
            species=species,
            lat=lat, lon=lon,
        )

        # 附加駐留時間估計
        sst = ocean_data.get("sst", 27.0) if ocean_data else 27.0
        eddy = ocean_data.get("eddy_strength", 0) > 0.3 if ocean_data else False
        dvm = ocean_data.get("dvm_depth", 100) if ocean_data else 100.0
        residence = fbm.estimate_school_residence_time(species, sst, eddy, dvm)
        result["residence_time"] = residence

        return JSONResponse(result)

    except Exception as e:
        log.error(f"Fish movement error: {e}", exc_info=True)
        return JSONResponse({"error": "Internal server error"}, status_code=500)


def _find_nearest_ocean_data(lat: float, lon: float) -> dict:
    """[v12-phase8] 從現有熱點數據找最近的環境資料"""
    hs = latest_data.get("hotspots", [])
    if not hs:
        return {}

    best = None
    best_dist = float("inf")
    for h in hs:
        hlat = h.get("lat", 0)
        hlon = h.get("lon", 0)
        d = (hlat - lat) ** 2 + (hlon - lon) ** 2
        if d < best_dist:
            best_dist = d
            best = h

    if best is None:
        return {}

    return {
        "sst": best.get("sst", 27.0),
        "chl": best.get("chl", 0.3),
        "chl_7d": best.get("chl_7d", [0.2, 0.22, 0.25, 0.28, 0.30, 0.28, 0.30]),
        "chl_clim": best.get("chl_climatology", 0.2),
        "phi": best.get("phi", best.get("phi_viability", 4.0)),
        "front_strength": best.get("front_strength", best.get("front_persistence", 0.3)),
        "eddy_strength": best.get("eddy_edge", best.get("eddy_strength", 0.2)),
        "dvm_depth": best.get("dvm_depth", 100.0),
    }


# ═══════════════════════════════════════════════════
# [v12-phase8] Dashboard v2 路由
# ═══════════════════════════════════════════════════
@app.get("/dashboard/v2", response_class=HTMLResponse)
async def dashboard_v2():
    """[v12-phase8] 專業儀表板 v2"""
    dash_v2 = WEB_DIR / "dashboard_v2.html"
    if dash_v2.exists():
        html = dash_v2.read_text(encoding="utf-8")
        # [v13.2-fix] S-4: 注入 session token 取代 master key
        html = html.replace("dev-key-change-me", _generate_session_token())
        return HTMLResponse(content=html)
    return HTMLResponse(content=f"<h1>OceanMaster v{VERSION}</h1><p>Dashboard v2 not found</p>")


@app.get("/dashboard_v2.js")
async def serve_v2_js():
    """[v12-phase8] Serve dashboard v2 JS"""
    js = WEB_DIR / "dashboard_v2.js"
    if js.exists():
        return FileResponse(str(js), media_type="application/javascript")
    return JSONResponse({"error": "JS not found"}, status_code=404)


@app.get("/dashboard_v2.css")
async def serve_v2_css():
    """[v12-phase8] Serve dashboard v2 CSS"""
    css = WEB_DIR / "dashboard_v2.css"
    if css.exists():
        return FileResponse(str(css), media_type="text/css")
    return JSONResponse({"error": "CSS not found"}, status_code=404)


# ═══════════════════════════════════════════════════
# [v13.2] /api/v1/weekly_briefing — Claude AI 作業建議
# ═══════════════════════════════════════════════════
_briefing_cache: dict = {"data": None, "ts": 0}  # DataCache TTL 6hr
_BRIEFING_TTL = 6 * 3600  # 6 hours


def _build_briefing_prompt(hotspots: list, summary: dict) -> str:
    """組裝給 Claude 的 prompt"""
    top3 = hotspots[:3] if hotspots else []

    hotspot_text = ""
    for i, h in enumerate(top3, 1):
        hotspot_text += (
            f"\n熱點 #{i}: {h.get('species','unknown')} "
            f"({h.get('lat',0):.1f}°N, {h.get('lon',0):.1f}°E)\n"
            f"  HSI: {h.get('hsi', h.get('score',0)):.2f}, "
            f"SST: {h.get('sst','N/A')}°C, "
            f"CHL: {h.get('chl','N/A')} mg/m³, "
            f"距高雄: {h.get('distance_nm','N/A')} nm, "
            f"航程: {h.get('travel_hours','N/A')} hr\n"
            f"  分析: {h.get('analysis','')}\n"
        )

    species_list = list({h.get("species", "") for h in hotspots if h.get("species")})

    ssts = [h.get("sst") for h in hotspots if h.get("sst") is not None]
    sst_range = f"{min(ssts):.1f}-{max(ssts):.1f}°C" if ssts else "N/A"

    oni = summary.get("enso_oni", 0)
    enso_phase = "El Niño" if oni > 0.5 else "La Niña" if oni < -0.5 else "中性"

    prompt = f"""你是 OceanMaster 遠洋漁場分析系統的 AI 作業顧問。
請根據以下最新海況數據，為台灣遠洋漁船船長撰寫本週作業建議簡報。

## 系統數據

### 前 3 名熱點
{hotspot_text}

### 海況概要
- SST 範圍: {sst_range}
- ENSO: ONI={oni:.2f} ({enso_phase})
- 風速: {summary.get('wind_speed', 'N/A')} m/s
- 洋流: {summary.get('current_speed', 'N/A')} m/s
- 可用物種: {', '.join(species_list)}
- 總熱點數: {len(hotspots)}

## 輸出要求

請用**繁體中文**撰寫，語氣專業但親切，像是給船長的實用建議。

### 必須包含：
1. **本週最佳漁場**（前3名熱點的作業建議，包含推薦作業水深、時段）
2. **主要海況特徵**（黑潮、鋒面、渦旋、ENSO 影響等）
3. **安全警告**（若有颱風、強流、大浪風險）
4. **推薦物種**（根據當前海況，哪些魚種最值得投入）
5. **航線建議**（從高雄港出發的建議航線）

### 格式：
- 簡潔有力，每段不超過3句
- 數據引用具體數值
- 結尾附上「資料來源: OceanMaster v{VERSION}, OISST/VIIRS/CMEMS」
"""
    return prompt


@app.get("/api/v1/weekly_briefing", dependencies=[Depends(verify_api_key)])
async def weekly_briefing(
    request: Request,
    format: str = Query("json", description="輸出格式: json 或 text"),
):
    """
    [v13.2] AI 週報 — Claude 生成的台灣遠洋船長作業建議

    使用 Claude claude-sonnet-4-6 分析最新海況數據，產出繁體中文作業簡報。
    DataCache TTL: 6 小時
    """
    if not _rate_check(request.client.host if request.client else "unknown"):
        raise HTTPException(429, "Rate limit exceeded")

    now = _time.time()

    # ── DataCache 6hr ──
    if (_briefing_cache["data"] is not None
            and now - _briefing_cache["ts"] < _BRIEFING_TTL):
        cached = _briefing_cache["data"]
        cached["from_cache"] = True
        if format == "text":
            return JSONResponse({"briefing": cached.get("briefing_text", ""), "from_cache": True})
        return JSONResponse(cached)

    # ── 取得最新數據 ──
    hotspots = latest_data.get("hotspots", [])
    summary = latest_data.get("analysis_summary", {})

    if not hotspots:
        return JSONResponse({"error": "尚無漁場數據，請稍後再試"}, status_code=503)

    # ── 呼叫 Claude API ──
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        # .env fallback
        env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

    if not api_key:
        # Fallback: 無 API Key 時用規則生成簡報
        return _generate_rule_based_briefing(hotspots, summary, format)

    prompt = _build_briefing_prompt(hotspots, summary)

    try:
        import httpx
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-6-20250514",
                    "max_tokens": 2000,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )

        if resp.status_code != 200:
            log.warning(f"  Claude API error {resp.status_code}: {resp.text[:200]}")
            return _generate_rule_based_briefing(hotspots, summary, format)

        claude_data = resp.json()
        briefing_text = claude_data.get("content", [{}])[0].get("text", "")

        result = {
            "briefing_text": briefing_text,
            "top_hotspots": [
                {
                    "rank": i + 1,
                    "species": h.get("species", ""),
                    "lat": h.get("lat", 0),
                    "lon": h.get("lon", 0),
                    "hsi": h.get("hsi", h.get("score", 0)),
                    "sst": h.get("sst"),
                    "distance_nm": h.get("distance_nm"),
                    "travel_hours": h.get("travel_hours"),
                }
                for i, h in enumerate(hotspots[:3])
            ],
            "sea_conditions": {
                "sst_range": f"{min(s for s in [h.get('sst') for h in hotspots] if s):.1f}-"
                             f"{max(s for s in [h.get('sst') for h in hotspots] if s):.1f}°C"
                             if any(h.get("sst") for h in hotspots) else "N/A",
                "enso_oni": summary.get("enso_oni", 0),
                "wind_speed": summary.get("wind_speed"),
            },
            "recommended_species": list({h.get("species") for h in hotspots[:5] if h.get("species")}),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": "claude-sonnet-4-6-20250514",
            "from_cache": False,
        }

        # Cache
        _briefing_cache["data"] = result
        _briefing_cache["ts"] = now

        if format == "text":
            return JSONResponse({"briefing": briefing_text, "from_cache": False})
        return JSONResponse(result)

    except Exception as e:
        log.warning(f"  Claude API call failed: {e}")
        return _generate_rule_based_briefing(hotspots, summary, format)


def _generate_rule_based_briefing(hotspots: list, summary: dict, fmt: str) -> JSONResponse:
    """Fallback: 無 Claude API 時用規則生成簡報"""
    top3 = hotspots[:3]
    species_set = list({h.get("species", "") for h in top3 if h.get("species")})
    ssts = [h.get("sst") for h in hotspots if h.get("sst") is not None]

    lines = ["═══ OceanMaster 本週作業建議 ═══\n"]

    lines.append("【最佳漁場】")
    for i, h in enumerate(top3, 1):
        lines.append(
            f"  #{i} {h.get('species','?')} — "
            f"({h.get('lat',0):.1f}°N, {h.get('lon',0):.1f}°E) "
            f"HSI={h.get('hsi', h.get('score',0)):.2f}, "
            f"SST={h.get('sst','?')}°C, "
            f"距港 {h.get('distance_nm','?')}nm"
        )

    lines.append("\n【海況特徵】")
    if ssts:
        lines.append(f"  SST 範圍: {min(ssts):.1f}-{max(ssts):.1f}°C")
    oni = summary.get("enso_oni", 0)
    enso = "El Niño" if oni > 0.5 else "La Niña" if oni < -0.5 else "中性"
    lines.append(f"  ENSO: ONI={oni:.2f} ({enso})")

    lines.append("\n【安全警告】")
    lines.append("  請注意即時氣象預報，確認作業海域無颱風警報。")

    lines.append(f"\n【推薦物種】{', '.join(species_set)}")
    lines.append(f"\n資料來源: OceanMaster v{VERSION}, OISST/VIIRS/CMEMS")

    briefing_text = "\n".join(lines)

    result = {
        "briefing_text": briefing_text,
        "top_hotspots": [
            {
                "rank": i + 1,
                "species": h.get("species", ""),
                "lat": h.get("lat", 0),
                "lon": h.get("lon", 0),
                "hsi": h.get("hsi", h.get("score", 0)),
            }
            for i, h in enumerate(top3)
        ],
        "recommended_species": species_set,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "rule-based (no API key)",
        "from_cache": False,
    }

    if fmt == "text":
        return JSONResponse({"briefing": briefing_text, "from_cache": False})
    return JSONResponse(result)


# ═══════════════════════════════════════════════════
# [v15.2] /api/v1/sat_briefing — Low-bandwidth satellite text briefing
# ═══════════════════════════════════════════════════

@app.get("/api/v1/sat_briefing")
async def sat_briefing(
    request: Request,
    format: str = Query("text", description="Output: text (plain ASCII) or json"),
    save: bool = Query(False, description="Save briefing to output/ directory"),
):
    """
    [v15.2] V3.0 #3: Pure-ASCII daily briefing for satellite transmission.

    Designed for Iridium SBD / Inmarsat C:
      - Pure 7-bit ASCII, no Unicode
      - Fixed-width 60 columns
      - Total size < 5KB
      - No API key required (satellite terminals have no browser)
    """
    hotspots = latest_data.get("hotspots", [])
    summary = latest_data.get("analysis_summary", {})

    if not hotspots:
        if format == "text":
            return Response(
                "OCEANMASTER: No data available. Try again later.\n",
                media_type="text/plain; charset=ascii",
            )
        return JSONResponse({"error": "No data available"}, status_code=503)

    try:
        from engine.text_briefing import generate_sat_briefing, save_briefing_file
    except ImportError:
        return JSONResponse({"error": "text_briefing module not available"}, status_code=500)

    text = generate_sat_briefing(hotspots, summary)

    if save:
        try:
            path = save_briefing_file(text)
            log.info(f"Briefing saved to {path}")
        except Exception as e:
            log.warning(f"Failed to save briefing: {e}")

    if format == "text":
        return Response(text, media_type="text/plain; charset=ascii")

    return JSONResponse({
        "briefing_text": text,
        "size_bytes": len(text.encode("ascii", "replace")),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "format": "ascii-60col",
    })


# ═══════════════════════════════════════════════════
# Legacy
# ═══════════════════════════════════════════════════
@app.get("/map.html")
async def get_map():
    # [v13.2-fix] B-12: 動態找最新的 Map.html（不再硬編碼 v10_2）
    out = Path("output")
    maps = sorted(out.glob("OceanMaster_*_Map.html"), reverse=True)
    if maps:
        return FileResponse(str(maps[0]))
    return JSONResponse({"error": "地圖檔案不存在"}, status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

