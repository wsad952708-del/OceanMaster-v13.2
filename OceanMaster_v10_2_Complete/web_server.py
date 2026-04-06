"""
OceanMaster Web 服務 v12 — 漁船專用即時漁場儀表板  # [v12-fix] version bump
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
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
import logging
import numpy as np

# 導入主管線
import sys
sys.path.append(str(Path(__file__).parent))
from main_v10_3 import OceanMasterPipeline, VERSION

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("OceanMaster.Web")


# [v10.5] UTF-8 JSON Response for proper CJK character rendering
class UTF8JSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(
            content, ensure_ascii=False, allow_nan=False, default=str
        ).encode("utf-8")

# ── T8: API Key Authentication ──
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
API_KEY = os.environ.get("OCEANMASTER_API_KEY", "dev-key-change-me")


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """T8: Verify API key for protected endpoints."""
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


# ═══════════════════════════════════════════════════
# [v12-enhance] Simple sliding-window rate limiter
# 100 requests / 60 seconds per IP  (no Redis needed)
# ═══════════════════════════════════════════════════
_RATE_LIMIT = int(os.environ.get("OCEANMASTER_RATE_LIMIT", "100"))
_RATE_WINDOW = 60  # seconds
_request_log: dict = {}  # ip -> [timestamps]


def _rate_check(ip: str) -> bool:
    """Return True if the request should be allowed."""
    now = _time.time()
    timestamps = _request_log.get(ip, [])
    # prune old entries
    timestamps = [t for t in timestamps if now - t < _RATE_WINDOW]
    if len(timestamps) >= _RATE_LIMIT:
        _request_log[ip] = timestamps
        return False
    timestamps.append(now)
    _request_log[ip] = timestamps
    return True


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

# ─── Global State ───
latest_data = {
    "hotspots": [],
    "last_update": None,
    "is_updating": False,
    "data_timestamp": None,
    "analysis_summary": {},
}


# ═══════════════════════════════════════════════════
# Background Task
# ═══════════════════════════════════════════════════
async def update_data_background():
    while True:
        try:
            log.info("🔄 開始背景更新...")
            latest_data["is_updating"] = True

            pipeline = OceanMasterPipeline(
                lat_range=(5, 35), lon_range=(120, 175),
                vessel_pos=(25.13, 121.74), output_dir="output",
            )
            # [商用] run_in_executor 避免阻塞 uvicorn event loop
            # pipeline.run_full() 內含大量同步計算 (numpy/HTTP)
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,  # default ThreadPoolExecutor
                lambda: asyncio.run(pipeline.run_full())
            )

            # Read latest hotspots (timestamped files)
            out_dir = Path("output")
            hotspot_files = sorted(out_dir.glob("hotspots_*.json"), reverse=True)
            hp_file = hotspot_files[0] if hotspot_files else out_dir / "hotspots.json"

            if hp_file.exists():
                with open(hp_file, 'r', encoding='utf-8') as f:
                    latest_data["hotspots"] = json.load(f)
                latest_data["last_update"] = datetime.now(timezone.utc).isoformat()
                latest_data["data_timestamp"] = datetime.now(timezone.utc).isoformat()
                log.info(f"✅ 更新完成: {len(latest_data['hotspots'])} 個熱點")

            latest_data["analysis_summary"] = results.get("analysis_summary", {})
            latest_data["is_updating"] = False
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
    log.info("🚀 啟動 OceanMaster Web 服務 v12...")

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

    asyncio.create_task(update_data_background())
    log.info("✅ 背景更新任務已啟動 (每 6 小時)")

    yield

    log.info("👋 關閉 OceanMaster 服務")


app = FastAPI(
    title=f"OceanMaster v{VERSION}",
    lifespan=lifespan,
    default_response_class=UTF8JSONResponse,
)

# [v12-fix] CORS middleware for cross-origin dashboard deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files after app creation
if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")

# [v10.5] Captain PWA
CAPTAIN_DIR = Path("static/captain")
if CAPTAIN_DIR.exists():
    app.mount("/captain", StaticFiles(directory=str(CAPTAIN_DIR), html=True), name="captain")


# ═══════════════════════════════════════════════════
# Pages (Public — no auth needed)
# ═══════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
async def root():
    """主頁：3-tab 儀表板 (API key injected at serve time)"""
    dashboard = WEB_DIR / "dashboard.html"
    if dashboard.exists():
        html = dashboard.read_text(encoding="utf-8")
        # Inject actual API key into meta tag so dashboard.js uses it
        html = html.replace("dev-key-change-me", API_KEY)
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
    """Health check — public, no auth needed."""
    return JSONResponse({
        "status": "healthy", "version": VERSION,
        "hotspot_count": len(latest_data["hotspots"]),
    })


# ═══════════════════════════════════════════════════
# Core API (T8: requires API key)
# ═══════════════════════════════════════════════════
@app.get("/api/hotspots", dependencies=[Depends(verify_api_key)])
async def get_hotspots():
    """熱點 + 新鮮度欄位 (A3)"""
    freshness_hours = None
    if latest_data["data_timestamp"]:
        try:
            dt = datetime.fromisoformat(latest_data["data_timestamp"].replace("Z", "+00:00"))
            freshness_hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
        except Exception:
            pass

    return JSONResponse({
        "hotspots": latest_data["hotspots"],
        "last_update": latest_data["last_update"],
        "is_updating": latest_data["is_updating"],
        "data_timestamp": latest_data["data_timestamp"],
        "data_freshness": round(freshness_hours, 1) if freshness_hours else None,
        "version": VERSION,
    })

@app.get("/api/analyze", dependencies=[Depends(verify_api_key)])
async def api_analyze(
    request: Request,
    lat_min: float = 5, lat_max: float = 35,
    lon_min: float = 120, lon_max: float = 175,
    vessel_lat: float = 25.13, vessel_lon: float = 121.74,
    date: str = None,
):
    """執行即時或歷史分析"""
    pipeline = OceanMasterPipeline(
        lat_range=(lat_min, lat_max), lon_range=(lon_min, lon_max),
        vessel_pos=(vessel_lat, vessel_lon), target_date=date,
    )
    results = await pipeline.run_full()
    clean = {k: v for k, v in results.items() if not isinstance(v, np.ndarray)}
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
            vessel_pos=(25.13, 121.74), target_date=date,
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
        return JSONResponse({
            "date": date, "species": species,
            "error": str(e), "status": "unavailable",
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
                    "name": a.name,
                    "lat": a.lat, "lon": a.lon,
                    "intensity_kt": a.intensity_kt,
                    "category": a.category,
                    "source": a.source,
                }
                for a in alerts
            ],
            "geojson": geojson,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        return JSONResponse({
            "active_typhoons": 0,
            "alerts": [],
            "error": str(e),
        })


# [v12-fix] 原 /api/system-health 重複定義已合併至 L685 的 lineage 版本
# 參考: Removed duplicate endpoint (was lines 456-489 in v10.4)


# ═══════════════════════════════════════════════════
# [v12-enhance] /api/v1/food_chain — 食物鏈完整剖析
# ═══════════════════════════════════════════════════
@app.get("/api/v1/food_chain", dependencies=[Depends(verify_api_key)])
async def food_chain(
    lat: float = Query(25.0, ge=-90, le=90, description="緯度"),
    lon: float = Query(130.0, ge=-180, le=180, description="經度"),
    species: str = Query("yellowfin", description="物種 key"),
):
    """
    [v12] 回傳指定位置的食物鏈生態指標。
    包含 DVM 深度、浮游動物代理、OMZ 壓縮/增益、月相等。
    """
    if not _rate_check(str(lat) + str(lon)):  # light rate check
        raise HTTPException(429, "Rate limit exceeded")

    try:
        # 取得最新 pipeline 結果 (if available)
        result = {}
        if hasattr(app.state, 'pipeline') and app.state.pipeline:
            p = app.state.pipeline
        else:
            p = None

        # DVM 深度
        dvm_depth = 0.0
        try:
            from engine.dvm_model import DVMModel
            from engine.species_params import DVM_PARAMS
            dvm = DVMModel()
            dvm_info = DVM_PARAMS.get(species, {})
            day_range = dvm_info.get("day_depth_range", (50, 200))
            night_range = dvm_info.get("night_depth_range", (0, 50))
            hour = datetime.now(timezone.utc).hour
            # simple day/night interpolation
            import math
            day_w = 0.5 * (1 + math.cos(2 * math.pi * (hour - 12) / 24))
            dvm_depth = day_w * sum(day_range)/2 + (1-day_w) * sum(night_range)/2
        except Exception:
            dvm_depth = 100.0  # fallback

        # 月相
        lunar_phase = "未知"
        lunar_illum = 0.0
        try:
            from engine.lunar_model import LunarEngine
            le = LunarEngine()
            moon = le.compute(datetime.now(timezone.utc))
            lunar_phase = moon.get("phase_name", "未知")
            lunar_illum = moon.get("illumination", 0.0)
        except Exception:
            pass

        # 浮游動物代理
        zoo_proxy = 0.0
        try:
            from engine.zooplankton_proxy import ZooplanktonProxy
            zp = ZooplanktonProxy()
            # rough estimate without full ocean data
            zoo_proxy = 0.5  # placeholder: real value requires CHL grid
        except Exception:
            pass

        return JSONResponse({
            "lat": lat, "lon": lon, "species": species,
            "dvm_depth_m": round(dvm_depth, 1),
            "lunar_phase": lunar_phase,
            "lunar_illumination": round(lunar_illum, 3),
            "zooplankton_proxy": round(zoo_proxy, 3),
            "omz_compression": 0.0,  # requires full DO grid
            "omz_edge_enrichment": 0.0,
            "food_chain_summary": {
                "primary_producers": "Chl-a → size-fractionated (Brewin 2010)",
                "secondary_producers": "Zooplankton proxy (pico/nano/micro weighted)",
                "grazers_to_prey": f"DVM depth ~{dvm_depth:.0f}m ({species})",
                "lunar_effect": f"{lunar_phase} ({lunar_illum*100:.0f}% 光照)",
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ═══════════════════════════════════════════════════
# [v12-enhance] /api/v1/dvm_profile — 24hr DVM 深度剖面
# ═══════════════════════════════════════════════════
@app.get("/api/v1/dvm_profile", dependencies=[Depends(verify_api_key)])
async def dvm_profile(
    species: str = Query("yellowfin", description="物種 key"),
    lat: float = Query(25.0, ge=-90, le=90, description="緯度 (影響日照時長)"),
):
    """
    [v12] 回傳 24 小時 DVM 深度剖面。
    使用 cosine 日夜過渡模型 (v12 dvm_model.py)。
    """
    if not _rate_check(f"dvm_{species}"):
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
        return JSONResponse({"error": str(e)}, status_code=500)


# ═══════════════════════════════════════════════════
# /api/v1/report_catch (v10.4) — 漁獲回報 API
# ═══════════════════════════════════════════════════
CATCH_DB_PATH = Path("data/catch_reports.db")


def _init_catch_db():
    """Initialize SQLite catch reports database."""
    CATCH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CATCH_DB_PATH))
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
    conn.close()
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
        if not (-90 <= lat <= 90 and -180 <= lon <= 360):
            raise ValueError("lat/lon out of range")
        if cpue < 0:
            raise ValueError("CPUE cannot be negative")
    except (ValueError, TypeError) as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    # env_snapshot → JSON string
    env_snapshot = body.get("env_snapshot")
    env_json = json.dumps(env_snapshot) if env_snapshot else None

    # Insert into SQLite
    try:
        conn = sqlite3.connect(str(CATCH_DB_PATH))
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
        conn.close()

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
        log.error(f"  Catch report DB error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/v1/catch_stats", dependencies=[Depends(verify_api_key)])
async def get_catch_stats():
    """漁獲回報統計 — 總數、各魚種分佈、最近回報"""
    try:
        conn = sqlite3.connect(str(CATCH_DB_PATH))
        total = conn.execute("SELECT COUNT(*) FROM catch_reports").fetchone()[0]

        species_stats = conn.execute(
            "SELECT target_species, COUNT(*), AVG(actual_cpue_kg_day) "
            "FROM catch_reports GROUP BY target_species"
        ).fetchall()

        recent = conn.execute(
            "SELECT vessel_id, timestamp, target_species, actual_cpue_kg_day "
            "FROM catch_reports ORDER BY created_at DESC LIMIT 10"
        ).fetchall()
        conn.close()

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
        return JSONResponse({"total_reports": 0, "error": str(e)})


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
        log.error(f"Food chain timeline error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


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
        except Exception:
            pass

        result = fbm.estimate_feeding_windows(
            lat=lat, lon=lon, date=datetime.now(timezone.utc),
            species=species, lunar_phase=lunar_phase,
        )
        return JSONResponse(result)

    except Exception as e:
        log.error(f"Feeding windows error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


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
        log.error(f"Fish movement error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


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
        html = html.replace("dev-key-change-me", API_KEY)
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
# Legacy
# ═══════════════════════════════════════════════════
@app.get("/map.html")
async def get_map():
    map_file = "output/OceanMaster_v10_2_Map.html"
    if os.path.exists(map_file):
        return FileResponse(map_file)
    return JSONResponse({"error": "地圖檔案不存在"}, status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

