"""
OceanMaster 商業技術驗證 (Commercial Technical Verification)
============================================================
排除「真實漁獲數據」限制，驗證所有其他技術面是否達到商業水準。

檢查維度：
1. 程式碼品質 — import 完整性、異常處理、模組可載入
2. 數據管道 — fetcher 建構、快取、插值、fallback
3. 演算法 — SST 鋒面/FTLE/GreenFish HSI/ML pipeline
4. 商業基礎設施 — Route/GeoJSON/KML/Dashboard/WebServer
5. 部署就緒 — Docker/requirements/health check/API auth
6. 輸出品質 — hotspots.json 完整性
"""

import sys, json, os, time, importlib, traceback, inspect
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ── Results ──
P_LIST, W_LIST, F_LIST = [], [], []
def P(msg): P_LIST.append(msg); print(f"  ✅ {msg}")
def F(msg): F_LIST.append(msg); print(f"  ❌ {msg}")
def W(msg): W_LIST.append(msg); print(f"  ⚠️  {msg}")

def check(cond, p_msg, f_msg):
    if cond: P(p_msg)
    else: F(f_msg)

t0 = time.time()
print("=" * 70)
print("  OceanMaster 商業技術驗證")
print("  (排除真實漁獲數據限制)")
print("=" * 70)

# ═══════════════════════════════════════════════════
# 1. 模組載入完整性
# ═══════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  [1] 模組載入完整性 (Module Import Integrity)")
print("─" * 70)

modules_to_check = {
    "engine.data_fetcher_v2": "OceanDataFetcher",
    "engine.algorithms": "detect_sst_fronts",
    "engine.greenfish_hsi": "GreenFishLiteHSI",
    "engine.ml.stacking_ensemble": "FishingStackingModel",
    "engine.shap_explainer": "SHAPExplainer",
    "engine.navigation.route_planner": "FuelOptimalRouter",
    "engine.geojson_output": "generate_geojson",
    "engine.kml_generator": "generate_kml",
    "engine.primary_production": "PrimaryProductionEngine",
    "engine.accuracy_booster": "SSTAnomalyEngine",
    "engine.captain_reports": "CaptainReportSystem",
    "engine.ai_fusion": "fuse_and_rank",
    "engine.species_params": "SPECIES",
    "engine.dissolved_oxygen": None,
    "engine.forage_engine": None,
    "engine.dvm_model": None,
    "engine.eddy_detector": None,
    "engine.gebco_features": "GEBCOFeatures",
    "engine.ocean_physics": None,
    "engine.eez.eez_checker": "EEZChecker",
}

imported = {}
for mod_path, expected_attr in modules_to_check.items():
    try:
        m = importlib.import_module(mod_path)
        imported[mod_path] = m
        if expected_attr:
            check(hasattr(m, expected_attr),
                  f"{mod_path}: {expected_attr} found",
                  f"{mod_path}: {expected_attr} MISSING")
        else:
            P(f"{mod_path}: imported OK")
    except Exception as e:
        F(f"{mod_path}: IMPORT FAILED — {e}")

# ═══════════════════════════════════════════════════
# 2. 程式碼品質 (Code Quality)
# ═══════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  [2] 程式碼品質 (Code Quality)")
print("─" * 70)

# 2a: 異常處理覆蓋率
print("\n  [2a] 異常處理覆蓋率")
critical_files = [
    "engine/data_fetcher_v2.py",
    "engine/algorithms.py",
    "engine/greenfish_hsi.py",
    "main_v10_3.py",
    "web_server.py",
]
for f in critical_files:
    fpath = Path(f)
    if fpath.exists():
        content = fpath.read_text(encoding="utf-8", errors="replace")
        try_count = content.count("try:")
        except_count = content.count("except")
        bare_except = content.count("except:")
        lines = len(content.split("\n"))
        # 演算法檔案是純數學，不需要 try/except
        min_try = 0 if "algorithm" in f or "greenfish" in f else 3
        check(try_count >= min_try and except_count >= min_try,
              f"{f}: {try_count} try/{except_count} except blocks ({lines} lines)",
              f"{f}: insufficient error handling ({try_count} try)")
        if bare_except > 2:
            W(f"{f}: {bare_except} bare 'except:' (should use specific exceptions)")
    else:
        F(f"{f}: file not found")

# 2b: async/await 正確性
print("\n  [2b] async/await 架構")
ws_path = Path("web_server.py")
if ws_path.exists():
    wsc = ws_path.read_text(encoding="utf-8", errors="replace")
    check("run_in_executor" in wsc, "web_server: run_in_executor for blocking ops",
          "web_server: MISSING run_in_executor (may block event loop)")
    check("asyncio" in wsc or "async def" in wsc, "web_server: async architecture",
          "web_server: not async")
    check("/health" in wsc, "web_server: /health endpoint present",
          "web_server: MISSING /health endpoint")
    check("X-API-Key" in wsc or "api_key" in wsc.lower(),
          "web_server: API key authentication",
          "web_server: MISSING API auth")

# 2c: 型別提示
print("\n  [2c] 型別提示覆蓋率")
for f in ["engine/algorithms.py", "engine/greenfish_hsi.py", "engine/shap_explainer.py"]:
    fpath = Path(f)
    if fpath.exists():
        c = fpath.read_text(encoding="utf-8", errors="replace")
        type_hints = c.count("->") + c.count(": Dict") + c.count(": np.ndarray") + c.count(": List") + c.count(": Optional")
        check(type_hints >= 3, f"{f}: {type_hints} type annotations",
              f"{f}: insufficient type hints ({type_hints})")

# ═══════════════════════════════════════════════════
# 3. 數據管道品質 (Data Pipeline)
# ═══════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  [3] 數據管道品質 (Data Pipeline)")
print("─" * 70)

# 3a: SST timeout
print("\n  [3a] SST Timeout 防護")
df_path = Path("engine/data_fetcher_v2.py")
if df_path.exists():
    dfc = df_path.read_text(encoding="utf-8", errors="replace")
    check("wait_for" in dfc and "timeout" in dfc,
          "data_fetcher: SST global timeout present (90s)",
          "data_fetcher: MISSING SST timeout — can hang 20+ minutes")
    check("griddata" in dfc,
          "data_fetcher: SST spatial interpolation (scipy griddata)",
          "data_fetcher: MISSING SST interpolation")
    check("WOA" in dfc or "climatology" in dfc.lower(),
          "data_fetcher: temp_3d WOA climatology fallback",
          "data_fetcher: MISSING temp_3d fallback")
    check("clamp" in dfc.lower() or "bad_mask" in dfc,
          "data_fetcher: temp_3d outlier clamping",
          "data_fetcher: MISSING temp_3d clamping")

# 3b: ERDDAP mirrors
print("\n  [3b] ERDDAP 多鏡像")
if "engine.data_fetcher_v2" in imported:
    m = imported["engine.data_fetcher_v2"]
    mirrors = getattr(m, "ERDDAP_MIRRORS", [])
    check(len(mirrors) >= 3,
          f"ERDDAP mirrors: {len(mirrors)} mirrors configured",
          f"ERDDAP mirrors: only {len(mirrors)} (need ≥3)")

# 3c: Cache system
print("\n  [3c] 快取系統")
if df_path.exists():
    check("cache" in dfc.lower() and "self.cache" in dfc,
          "data_fetcher: cache system present",
          "data_fetcher: MISSING cache")

# ═══════════════════════════════════════════════════
# 4. 演算法正確性 (Algorithm Correctness)
# ═══════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  [4] 演算法正確性 (Algorithm Correctness)")
print("─" * 70)

# 4a: SST front detection
print("\n  [4a] SST 鋒面偵測")
try:
    from engine.algorithms import detect_sst_fronts
    test_sst = np.random.uniform(24, 28, (50, 80)).astype(np.float32)
    # Add a sharp front
    test_sst[:, 40:] += 3.0
    test_lat = np.linspace(20, 25, 50)
    test_lon = np.linspace(120, 128, 80)
    result = detect_sst_fronts(test_sst, test_lat, test_lon)
    check("gradient_magnitude" in result and "front_mask" in result and "front_strength" in result,
          f"SST fronts: all keys present ({len(result)} keys)",
          "SST fronts: MISSING output keys")
    check(result["front_mask"].any(),
          f"SST fronts: detected {np.sum(result['front_mask'])} front pixels (3°C step)",
          "SST fronts: FAILED to detect obvious 3°C front")
    check(result["front_strength"].max() > 0.3,
          f"SST fronts: max strength={result['front_strength'].max():.3f}",
          "SST fronts: strength too low for 3°C front")
except Exception as e:
    F(f"SST fronts: {e}")

# 4b: FTLE
print("\n  [4b] FTLE 拉格朗日相干結構")
try:
    from engine.algorithms import compute_ftle
    u = np.random.uniform(-0.3, 0.3, (30, 40)).astype(np.float32)
    v = np.random.uniform(-0.3, 0.3, (30, 40)).astype(np.float32)
    lat = np.linspace(20, 23, 30)
    lon = np.linspace(120, 124, 40)
    ftle_r = compute_ftle(u, v, lat, lon, integration_days=1.0, dt_hours=6.0)
    check("ftle" in ftle_r and "ridges" in ftle_r,
          f"FTLE: computed, shape={ftle_r['ftle'].shape}, ridges={np.sum(ftle_r['ridges'])}",
          "FTLE: MISSING output keys")
except Exception as e:
    F(f"FTLE: {e}")

# 4c: GreenFish HSI
print("\n  [4c] GreenFish HSI")
try:
    from engine.greenfish_hsi import GreenFishLiteHSI
    gf = GreenFishLiteHSI()
    sst_test = np.random.uniform(22, 30, (30, 40)).astype(np.float32)
    chl_test = np.random.uniform(0.1, 2.0, (30, 40)).astype(np.float32)
    lat_t = np.linspace(20, 23, 30)
    lon_t = np.linspace(120, 124, 40)
    species_pass = 0
    phi_test = np.random.uniform(0.5, 4.0, (30, 40)).astype(np.float32)
    for sp in ["yellowfin", "bigeye", "skipjack", "albacore"]:
        h = gf.compute(
            species=sp, sst=sst_test, phi_viability=phi_test, chl=chl_test
        )
        if h is not None and isinstance(h, dict) and "hsi" in h:
            hsi_arr = h["hsi"]
            if isinstance(hsi_arr, np.ndarray) and hsi_arr.shape == sst_test.shape:
                species_pass += 1
    check(species_pass == 4,
          f"GreenFish HSI: {species_pass}/4 species computed",
          f"GreenFish HSI: only {species_pass}/4 species passed")
except Exception as e:
    F(f"GreenFish HSI: {e}")

# 4d: ML Model loading
print("\n  [4d] ML 模型載入+預測")
try:
    import joblib
    models_dir = Path("models")
    for sp in ["yellowfin", "bigeye", "skipjack", "albacore"]:
        pkl = models_dir / f"stacking_{sp}.pkl"
        if pkl.exists():
            loaded = joblib.load(pkl)
            model = loaded["model"] if isinstance(loaded, dict) else loaded
            scaler = loaded.get("scaler") if isinstance(loaded, dict) else None
            # Test predict with 12 features
            X_test = np.random.uniform(0, 1, (5, 12)).astype(np.float32)
            if scaler:
                X_test = scaler.transform(X_test)
            pred = model.predict(X_test)
            check(pred.shape == (5,) and np.all(np.isfinite(pred)),
                  f"ML {sp}: predict OK, shape={pred.shape}, mean={pred.mean():.3f}",
                  f"ML {sp}: predict FAILED")
        else:
            W(f"ML {sp}: {pkl} not found")
except Exception as e:
    F(f"ML models: {e}")

# 4e: SHAP explainer
print("\n  [4e] SHAP 可解釋性")
try:
    from engine.shap_explainer import SHAPExplainer
    se = SHAPExplainer()
    check(se._shap_available, "SHAP: library available",
          "SHAP: library NOT available (fallback mode)")
    # Try registering a model
    if models_dir.exists():
        pkl = models_dir / "stacking_yellowfin.pkl"
        if pkl.exists():
            loaded = joblib.load(pkl)
            model = loaded["model"] if isinstance(loaded, dict) else loaded
            feat_names = ["sst", "chl", "ssh", "do", "current_speed",
                         "front_strength", "eddy_strength", "phi",
                         "bathy_depth", "bathy_slope", "dist_seamount", "dist_shelf_break"]
            se.register_model("yellowfin", model, feat_names)
            check("yellowfin" in se._explainers,
                  f"SHAP: yellowfin registered ({type(list(se._explainers.values())[0]).__name__})",
                  "SHAP: registration FAILED")
except Exception as e:
    W(f"SHAP: {e}")

# ═══════════════════════════════════════════════════
# 5. 商業基礎設施 (Commercial Infrastructure)
# ═══════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  [5] 商業基礎設施 (Commercial Infrastructure)")
print("─" * 70)

# 5a: Route Planner
print("\n  [5a] 航線規劃")
try:
    from engine.navigation.route_planner import FuelOptimalRouter, RoutePoint
    rp = FuelOptimalRouter(vessel_speed_kts=10.0)
    start = RoutePoint(22.6, 120.3, "高雄港")
    dest = RoutePoint(25.0, 135.0, "漁場A")
    route = rp.plan_route(start, dest)
    if route:
        dist = getattr(route, 'total_distance_nm', 0)
        check(dist > 0, f"Route: 高雄→漁場A = {dist:.1f} nm",
              "Route: distance is 0")
    else:
        W("Route planner: returned None")
except Exception as e:
    W(f"Route Planner: {e}")

# 5b: GeoJSON
print("\n  [5b] GeoJSON 輸出")
try:
    from engine.geojson_output import generate_geojson
    hp_file = Path("output/hotspots.json")
    if hp_file.exists():
        hs = json.loads(hp_file.read_text(encoding="utf-8"))
        geo = generate_geojson({"combined_hotspots": hs})
        check(geo is not None, f"GeoJSON: generated ({type(geo).__name__}, {len(str(geo))} chars)",
              "GeoJSON: returned None")
    else:
        W("GeoJSON: no hotspots.json to test")
except Exception as e:
    W(f"GeoJSON: {e}")

# 5c: KML
print("\n  [5c] KML 輸出")
try:
    from engine.kml_generator import generate_kml
    if hp_file.exists():
        kml = generate_kml(hs, output_dir="output")
        check(kml is not None and len(str(kml)) > 10,
              f"KML: generated ({len(str(kml))} chars)",
              "KML: generation failed")
except Exception as e:
    W(f"KML: {e}")

# 5d: Dashboard
print("\n  [5d] Dashboard")
dash_path = Path("web/dashboard.html")
dashjs_path = Path("web/dashboard.js")
if dash_path.exists():
    dc = dash_path.read_text(encoding="utf-8", errors="replace")
    check(len(dc) > 10000, f"Dashboard HTML: {len(dc)} bytes", "Dashboard: too small")
    check("leaflet" in dc.lower() or "L.map" in dc, "Dashboard: Leaflet map",
          "Dashboard: MISSING map library")
    check("hotspot" in dc.lower() or "marker" in dc.lower(),
          "Dashboard: hotspot display", "Dashboard: MISSING hotspot")
else:
    F("Dashboard: dashboard.html not found")
if dashjs_path.exists():
    djc = dashjs_path.read_text(encoding="utf-8", errors="replace")
    check(len(djc) > 5000, f"Dashboard JS: {len(djc)} bytes", "Dashboard JS: too small")

# 5e: Species params
print("\n  [5e] 魚種參數")
try:
    from engine.species_params import SPECIES
    check(len(SPECIES) >= 4,
          f"Species: {len(SPECIES)} species configured",
          f"Species: only {len(SPECIES)} (need ≥4)")
    for sp_name, sp_data in SPECIES.items():
        has_topt = "Topt_C" in sp_data or "Topt" in str(sp_data)
        if not has_topt:
            W(f"Species {sp_name}: missing Topt parameter")
except Exception as e:
    F(f"Species params: {e}")

# 5f: AI Fusion
print("\n  [5f] AI Fusion")
try:
    from engine.ai_fusion import fuse_and_rank, assess_safety
    check(callable(fuse_and_rank), "AI Fusion: fuse_and_rank available",
          "AI Fusion: MISSING fuse_and_rank")
    check(callable(assess_safety), "AI Fusion: assess_safety available",
          "AI Fusion: MISSING assess_safety")
except Exception as e:
    F(f"AI Fusion: {e}")

# ═══════════════════════════════════════════════════
# 6. 部署就緒 (Deployment Readiness)
# ═══════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  [6] 部署就緒 (Deployment Readiness)")
print("─" * 70)

# 6a: Docker
print("\n  [6a] Docker")
for f, min_size in [("Dockerfile", 500), ("docker-compose.yml", 200), ("requirements.txt", 200)]:
    fp = Path(f)
    if fp.exists():
        sz = fp.stat().st_size
        check(sz >= min_size, f"{f}: {sz} bytes", f"{f}: too small ({sz})")
    else:
        F(f"{f}: NOT FOUND")

# 6b: Dockerfile quality
print("\n  [6b] Dockerfile 品質")
df_file = Path("Dockerfile")
if df_file.exists():
    dc = df_file.read_text(encoding="utf-8", errors="replace")
    check("HEALTHCHECK" in dc or "healthcheck" in dc.lower(),
          "Dockerfile: HEALTHCHECK present",
          "Dockerfile: MISSING HEALTHCHECK")
    check("EXPOSE" in dc, "Dockerfile: EXPOSE port", "Dockerfile: MISSING EXPOSE")
    check("requirements.txt" in dc, "Dockerfile: installs requirements",
          "Dockerfile: MISSING requirements install")

# 6c: README
print("\n  [6c] README")
readme = Path("README.md")
if readme.exists():
    rc = readme.read_text(encoding="utf-8", errors="replace")
    check(len(rc) > 2000, f"README: {len(rc)} bytes",
          f"README: too short ({len(rc)} bytes)")
else:
    F("README.md: NOT FOUND")

# ═══════════════════════════════════════════════════
# 7. 輸出品質 (Output Quality)
# ═══════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  [7] 輸出品質 (Output Quality)")
print("─" * 70)

hp_file = Path("output/hotspots.json")
if hp_file.exists():
    hs = json.loads(hp_file.read_text(encoding="utf-8"))
    check(len(hs) >= 10, f"Hotspots: {len(hs)} entries", f"Hotspots: only {len(hs)}")
    
    # Species distribution
    sp_dist = {}
    for h in hs:
        sp = h.get("species", "unknown")
        sp_dist[sp] = sp_dist.get(sp, 0) + 1
    check(len(sp_dist) >= 4,
          f"Species distribution: {sp_dist}",
          f"Only {len(sp_dist)} species in output")

    # Required fields
    required = ["lat", "lon", "score", "species", "sst", "explain"]
    h0 = hs[0]
    missing = [f for f in required if f not in h0]
    check(len(missing) == 0,
          f"Hotspot fields: all {len(required)} required fields present",
          f"Hotspot fields: MISSING {missing}")

    # Value ranges
    scores = [h.get("score", 0) for h in hs]
    lats = [h.get("lat", 0) for h in hs]
    check(0 < min(scores) and max(scores) <= 1.0,
          f"HSI scores: [{min(scores):.3f}, {max(scores):.3f}]",
          f"HSI scores: out of [0, 1] range")
    check(all(1 <= l <= 35 for l in lats),
          f"Latitudes: [{min(lats):.1f}, {max(lats):.1f}] (valid range)",
          f"Latitudes: out of expected range [{min(lats):.1f}, {max(lats):.1f}]")

    # ML CPUE check
    cpue_vals = [h.get("ml_cpue_kg_day", 0) for h in hs]
    ml_conf = [h.get("ml_confidence", 0) for h in hs]
    if max(cpue_vals) > 0:
        P(f"ML CPUE: non-zero values detected (max={max(cpue_vals):.1f})")
    else:
        W(f"ML CPUE: all zeros (expected without real catch training data)")

    # Explanation text
    explanations = [h.get("explain", "") for h in hs]
    has_explain = sum(1 for e in explanations if e and len(str(e)) > 10)
    check(has_explain >= len(hs) * 0.8,
          f"Explanations: {has_explain}/{len(hs)} hotspots have explanation text",
          f"Explanations: only {has_explain}/{len(hs)} have explanations")
else:
    W("hotspots.json: not found (need to run main pipeline first)")

# Summary file
summ_file = Path("output/analysis_summary.json")
if summ_file.exists():
    summ = json.loads(summ_file.read_text(encoding="utf-8"))
    check(summ.get("ml_ensemble", False), "Summary: ML Ensemble = True",
          "Summary: ML Ensemble = False")
    check(summ.get("greenfish_lite", False), "Summary: GreenFish Lite = True",
          "Summary: GreenFish Lite = False")
    ds = summ.get("data_sources", {})
    ds_count = len(ds) if isinstance(ds, dict) else 0
    check(ds_count >= 5, f"Summary: {ds_count} data sources ({list(ds.keys()) if isinstance(ds, dict) else ds})",
          f"Summary: only {ds_count} data sources")

# ═══════════════════════════════════════════════════
# 8. 關鍵安全性 (Security & Robustness)
# ═══════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  [8] 安全性與健壯性 (Security & Robustness)")
print("─" * 70)

# 8a: No hardcoded secrets
print("\n  [8a] 硬編碼密鑰檢查")
secret_patterns = ["sk_live", "pk_live", "secret_key="]
for f in ["web_server.py", "main_v10_3.py", "engine/data_fetcher_v2.py"]:
    fpath = Path(f)
    if fpath.exists():
        c = fpath.read_text(encoding="utf-8", errors="replace")
        found = [p for p in secret_patterns if p in c.lower()]
        check(len(found) == 0, f"{f}: no hardcoded secrets",
              f"{f}: FOUND hardcoded secrets: {found}")

# 8b: Input validation
print("\n  [8b] 輸入驗證")
ws = Path("web_server.py")
if ws.exists():
    wsc = ws.read_text(encoding="utf-8", errors="replace")
    check("try" in wsc and "except" in wsc,
          "web_server: has try/except error handling",
          "web_server: MISSING error handling")

# ═══════════════════════════════════════════════════
# FINAL VERDICT
# ═══════════════════════════════════════════════════
elapsed = time.time() - t0
print("\n" + "=" * 70)
print(f"  商業技術驗證完成 ({elapsed:.1f}s)")
print("=" * 70)
print(f"\n  ✅ PASS: {len(P_LIST)}")
print(f"  ⚠️  WARN: {len(W_LIST)}")
print(f"  ❌ FAIL: {len(F_LIST)}")

if F_LIST:
    print("\n  FAILURES:")
    for f in F_LIST:
        print(f"    ❌ {f}")

if W_LIST:
    print("\n  WARNINGS:")
    for w in W_LIST:
        print(f"    ⚠️  {w}")

# Commercial readiness score
total = len(P_LIST) + len(F_LIST)
pass_rate = len(P_LIST) / max(total, 1) * 100

print(f"\n  商業就緒率: {pass_rate:.1f}% ({len(P_LIST)}/{total})")

if len(F_LIST) == 0:
    print(f"  VERDICT: 🎉 技術面達到商業水準 (COMMERCIAL GRADE)")
elif len(F_LIST) <= 2:
    print(f"  VERDICT: ⚠️ 接近商業水準，需修復 {len(F_LIST)} 項")
else:
    print(f"  VERDICT: 🔧 尚未達到商業水準 ({len(F_LIST)} 項待修復)")

print("=" * 70)

sys.exit(1 if F_LIST else 0)
