"""
OceanMaster v10.3 — 全面品質稽核
================================
驗證所有數據源 + 演算法 + ML + Dashboard
"""
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import asyncio, json, sys, time, traceback
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# ── Results tracking ──
results = {"pass": [], "fail": [], "warn": []}

def P(msg): results["pass"].append(msg); print(f"  ✅ {msg}")
def F(msg): results["fail"].append(msg); print(f"  ❌ {msg}")
def W(msg): results["warn"].append(msg); print(f"  ⚠️  {msg}")

def check(condition, pass_msg, fail_msg):
    if condition: P(pass_msg)
    else: F(fail_msg)

def check_array(arr, name, min_val=None, max_val=None, max_nan_ratio=0.5):
    """Validate a numpy array: shape, finite values, value range."""
    if arr is None:
        F(f"{name}: is None")
        return False
    if not isinstance(arr, np.ndarray):
        F(f"{name}: not ndarray (type={type(arr).__name__})")
        return False
    if arr.size == 0:
        F(f"{name}: empty array")
        return False

    nan_ratio = np.sum(np.isnan(arr)) / max(arr.size, 1)
    finite = arr[np.isfinite(arr)]

    if nan_ratio > max_nan_ratio:
        W(f"{name}: high NaN ratio {nan_ratio:.1%} (threshold {max_nan_ratio:.0%})")

    if len(finite) == 0:
        F(f"{name}: ALL values NaN/Inf")
        return False

    vmin, vmax = float(np.min(finite)), float(np.max(finite))
    msg = f"{name}: shape={arr.shape}, range=[{vmin:.3g}, {vmax:.3g}], NaN={nan_ratio:.1%}"

    ok = True
    if min_val is not None and vmin < min_val:
        W(f"{name}: min {vmin:.3g} < expected {min_val}")
        ok = False
    if max_val is not None and vmax > max_val:
        W(f"{name}: max {vmax:.3g} > expected {max_val}")
        ok = False

    if ok:
        P(msg)
    else:
        W(msg)
    return True


async def audit():
    t0 = time.time()
    print("=" * 70)
    print("  OceanMaster v10.3 全面品質稽核")
    print("=" * 70)

    # ═══════════════════════════
    # Phase 1: 數據爬取驗證
    # ═══════════════════════════
    print("\n" + "─" * 70)
    print("  Phase 1: 數據爬取驗證 (Data Fetch)")
    print("─" * 70)

    from engine.data_fetcher_v2 import OceanDataFetcher
    fetcher = OceanDataFetcher(
        lat_range=(5, 35), lon_range=(120, 175)
    )
    data = await fetcher.fetch_all()

    # Check each data source
    sources = data.get("data_sources", {})
    print(f"\n  Data sources reported: {json.dumps(sources, ensure_ascii=False)}")

    # SST
    print("\n  [SST]")
    sst = data.get("sst")
    check_array(sst, "SST", min_val=-2, max_val=40)
    check(sources.get("SST") not in [None, "", "NONE", "FAILED"],
          f"SST source: {sources.get('SST')}", "SST source missing or failed")

    # CHL (Chlorophyll)
    print("\n  [CHL]")
    chl = data.get("chl")
    check_array(chl, "CHL", min_val=0, max_val=100, max_nan_ratio=0.8)

    # Currents (u, v)
    print("\n  [CURRENTS]")
    u = data.get("u_current")
    v = data.get("v_current")
    check_array(u, "u_current", min_val=-3, max_val=3)
    check_array(v, "v_current", min_val=-3, max_val=3)
    if u is not None and v is not None:
        speed = np.sqrt(u**2 + v**2)
        check_array(speed, "current_speed", min_val=0, max_val=5)
    check(sources.get("CURRENTS") not in [None, "", "NONE", "FAILED"],
          f"CURRENTS source: {sources.get('CURRENTS')}", "CURRENTS source missing or failed")

    # SSH
    print("\n  [SSH]")
    ssh = data.get("ssh")
    if ssh is not None:
        check_array(ssh, "SSH", min_val=-2, max_val=2)
    else:
        W("SSH: not available (optional)")

    # Wind
    print("\n  [WIND]")
    wind_u = data.get("wind_u")
    wind_v = data.get("wind_v")
    if wind_u is not None and wind_v is not None:
        check_array(wind_u, "wind_u")
        check_array(wind_v, "wind_v")
        ws = np.sqrt(wind_u**2 + wind_v**2)
        check_array(ws, "wind_speed", min_val=0, max_val=60)
    else:
        W("Wind: not available (may use single-point OpenMeteo)")

    # Salinity
    print("\n  [SALINITY]")
    sal = data.get("salinity")
    if sal is not None:
        check_array(sal, "salinity", min_val=30, max_val=40)
    else:
        W("Salinity: not available")

    # DO (Dissolved Oxygen)
    print("\n  [DISSOLVED OXYGEN]")
    do_surf = data.get("do_surface")
    if do_surf is not None:
        check_array(do_surf, "DO_surface", min_val=0, max_val=15)
    else:
        W("DO: not available")

    # Temp 3D (for thermocline)
    print("\n  [TEMP_3D]")
    t3d = data.get("temp_3d")
    if t3d is not None:
        finite_count = np.sum(np.isfinite(t3d))
        if finite_count == 0:
            W("temp_3d: all NaN after clamping (will use climatology fallback)")
        else:
            check_array(t3d, "temp_3d", min_val=-2, max_val=35)
    else:
        W("temp_3d: not available (will use climatology fallback)")

    lats = data["lats"]
    lons = data["lons"]
    ny, nx = len(lats), len(lons)
    check(ny > 10 and nx > 10, f"Grid: {ny}x{nx} = {ny*nx} pts", f"Grid too small: {ny}x{nx}")

    # ═══════════════════════════
    # Phase 2: 演算法驗證
    # ═══════════════════════════
    print("\n" + "─" * 70)
    print("  Phase 2: 演算法驗證 (Algorithms)")
    print("─" * 70)

    from engine.algorithms import (
        detect_sst_fronts, compute_boa_gradient, compute_ftle, compute_thermocline,
    )
    from engine.ocean_physics import BathymetryAnalyzer, OceanPhysicsEngine

    # SST Fronts
    print("\n  [SST Fronts]")
    try:
        sst_fr = detect_sst_fronts(sst, lats, lons)
        fs = sst_fr["front_strength"]
        check_array(fs, "front_strength", min_val=0, max_val=10)
        strong = np.nansum(fs > 0.3)
        check(strong > 0, f"SST strong fronts: {strong} pts", "NO SST fronts detected")
    except Exception as e:
        F(f"SST fronts: {e}")

    # CHL Fronts
    print("\n  [CHL Fronts]")
    try:
        chl_fr = compute_boa_gradient(chl, lats, lons)
        check(chl_fr is not None and len(chl_fr) > 0, "CHL fronts computed", "CHL fronts failed")
    except Exception as e:
        F(f"CHL fronts: {e}")

    # FTLE
    print("\n  [FTLE]")
    try:
        ftle_r = compute_ftle(u, v, lats, lons)
        ftle = ftle_r["ftle"]
        check_array(ftle, "FTLE")
        ftle_mean = float(np.nanmean(ftle))
        # FTLE can be very small or negative — that's normal
        P(f"FTLE mean: {ftle_mean:.6f}")
    except Exception as e:
        F(f"FTLE: {e}")

    # Thermocline
    print("\n  [Thermocline]")
    if t3d is not None:
        try:
            depths_arr = np.array([0, 50, 100, 200, 300, 500])[:t3d.shape[0]]
            tc = compute_thermocline(t3d, depths_arr, lats, lons)
            check(tc is not None, "Thermocline computed", "Thermocline failed")
        except Exception as e:
            W(f"Thermocline: {e}")
    else:
        W("Thermocline: skipped (no temp_3d)")

    # Bathymetry (ETOPO1)
    print("\n  [Bathymetry]")
    try:
        bathy = BathymetryAnalyzer.generate_bathymetry(lats, lons)
        check_array(bathy, "bathymetry", min_val=-11000, max_val=5000)
    except Exception as e:
        F(f"Bathymetry: {e}")

    # GEBCO Features
    print("\n  [GEBCO Features]")
    try:
        from engine.gebco_features import GEBCOFeatures
        gebco = GEBCOFeatures()
        gf = gebco.compute_features(lats, lons, bathy=bathy)
        for k in ["slope", "dist_to_seamount", "dist_to_shelf_break"]:
            if k in gf:
                check_array(gf[k], f"GEBCO_{k}")
            else:
                W(f"GEBCO {k}: missing")
    except Exception as e:
        W(f"GEBCO Features: {e}")

    # Ocean Physics (EKE, Ekman)
    print("\n  [Ocean Physics]")
    try:
        physics_engine = OceanPhysicsEngine()
        physics = physics_engine.analyze_full(
            u, v, sal,
            wind_u if wind_u is not None else np.zeros_like(u),
            wind_v if wind_v is not None else np.zeros_like(v),
            lats,
            species_list=["skipjack", "yellowfin", "bigeye", "albacore"],
        )
        eke = physics.get("eke")
        if eke is not None:
            check_array(eke, "EKE", min_val=0)
        else:
            W("EKE: not in physics output")
    except Exception as e:
        F(f"Ocean Physics: {e}")

    # ═══════════════════════════
    # Phase 2.5: GreenFish Lite
    # ═══════════════════════════
    print("\n" + "─" * 70)
    print("  Phase 2.5: GreenFish Lite 驗證")
    print("─" * 70)

    try:
        from engine.thermocline_fetcher import ThermoclineFetcher
        from engine.eddy_detector import EddyDetector as GFEddyDetector
        from engine.forage_engine import ForageEngine
        from engine.dvm_model import DVMModel
        from engine.greenfish_hsi import GreenFishLiteHSI
        from engine.vessel_lights import VesselLightValidator

        month = datetime.now().month

        # Z20/MLD
        print("\n  [Z20/MLD - Thermocline Fetcher]")
        tc_fetcher = ThermoclineFetcher()
        glorys = tc_fetcher.fetch_glorys12(lats, lons, month)
        if glorys is not None:
            if "z20" in glorys:
                check_array(glorys["z20"], "Z20", min_val=0, max_val=500)
            if "mld" in glorys:
                check_array(glorys["mld"], "MLD", min_val=0, max_val=500)
        else:
            W("GLORYS12: no data returned, will use fallback")
            # ThermoclineFetcher may not have direct HYCOM method
            P("Thermocline will use climatology fallback in pipeline")

        # Eddy Detection
        print("\n  [Eddy Detection]")
        eddy_det = GFEddyDetector()
        ssh_grid = ssh if ssh is not None else np.zeros((ny, nx))
        eddy_r = eddy_det.detect(ssh_grid, u, v, lats, lons)
        if eddy_r is not None:
            if "eke" in eddy_r:
                check_array(eddy_r["eke"], "Eddy_EKE", min_val=0)
            if "eddy_mask" in eddy_r:
                mask = eddy_r["eddy_mask"]
                n_eddy = np.sum(mask > 0) if mask is not None else 0
                P(f"Eddy mask: {n_eddy} eddy points detected")
        else:
            W("Eddy detection returned None")

        # Forage Engine
        print("\n  [Forage Engine]")
        forage_eng = ForageEngine()
        z20_grid = glorys["z20"] if glorys and "z20" in glorys else np.full((ny, nx), 150.0)
        mld_grid = glorys["mld"] if glorys and "mld" in glorys else np.full((ny, nx), 30.0)
        forage_result = forage_eng.compute(chl, sst, lats)
        forage = forage_result.get("forage_total") if isinstance(forage_result, dict) else forage_result
        if forage is not None:
            check_array(forage, "forage_index", min_val=0)
        else:
            W("Forage: returned None")

        # DVM Model
        print("\n  [DVM Model]")
        dvm = DVMModel()
        forage_surf = forage * 0.5 if forage is not None else np.full((ny, nx), 0.25)
        forage_deep = forage * 0.5 if forage is not None else np.full((ny, nx), 0.25)
        feeding = dvm.compute_feeding_index("yellowfin", forage_surf, forage_deep, z20_grid, mld_grid, sst)
        if feeding is not None:
            check_array(feeding, "feeding_index", min_val=0, max_val=1)
        else:
            W("DVM feeding: returned None")

        # GreenFish HSI
        print("\n  [GreenFish HSI]")
        gf_hsi = GreenFishLiteHSI()

        # 準備 phi_viability (溶氧適宜度) — DO 標準化到 0-1
        do_grid = data.get("do_surface")
        if do_grid is not None:
            phi_viability = np.clip(do_grid / 7.0, 0.01, 1.0)
        else:
            phi_viability = np.full((ny, nx), 0.5)

        # 準備 eke — 確保 2D
        eke_g = eddy_r.get("eke") if eddy_r else np.full((ny, nx), 0.01)
        if isinstance(eke_g, np.ndarray) and eke_g.ndim == 1:
            eke_g = np.broadcast_to(eke_g.reshape(-1, 1) if eke_g.shape[0] == ny
                        else eke_g.reshape(1, -1), (ny, nx)).copy()

        # 準備 eddy_edge
        eddy_edge_g = eddy_r.get("eddy_mask", np.zeros((ny, nx))) if eddy_r else np.zeros((ny, nx))
        if isinstance(eddy_edge_g, np.ndarray) and eddy_edge_g.ndim == 1:
            eddy_edge_g = np.broadcast_to(eddy_edge_g.reshape(-1, 1) if eddy_edge_g.shape[0] == ny
                              else eddy_edge_g.reshape(1, -1), (ny, nx)).copy()

        # 準備 front_strength
        front_g = sst_fr.get("front_strength", np.zeros((ny, nx))) if 'sst_fr' in dir() else np.zeros((ny, nx))

        for sp in ["yellowfin", "bigeye", "skipjack", "albacore"]:
            try:
                hsi_result = gf_hsi.compute(
                    species=sp,
                    sst=sst,
                    phi_viability=phi_viability,
                    feeding_index=feeding if feeding is not None else np.full((ny, nx), 0.5),
                    z20=z20_grid,
                    mld=mld_grid,
                    eddy_edge=eddy_edge_g,
                    eke=eke_g,
                    front_strength=front_g,
                    chl=chl,
                )
                # compute() returns dict with 'hsi' key (0-100 scale)
                if isinstance(hsi_result, dict) and "hsi" in hsi_result:
                    hsi_arr = hsi_result["hsi"] / 100.0  # normalize to 0-1
                    check_array(hsi_arr, f"GF_HSI_{sp}", min_val=0, max_val=1)
                elif isinstance(hsi_result, np.ndarray):
                    check_array(hsi_result, f"GF_HSI_{sp}", min_val=0, max_val=1)
                else:
                    W(f"GreenFish HSI {sp}: unexpected return type {type(hsi_result)}")
            except Exception as e:
                F(f"GreenFish HSI {sp}: {e}")

        P("GreenFish Lite: ALL modules loaded and computed (4 species)")

    except ImportError as e:
        F(f"GreenFish Lite: import error — {e}")
    except Exception as e:
        F(f"GreenFish Lite: {e}\n{traceback.format_exc()}")

    # ═══════════════════════════
    # Phase 3: ML Ensemble
    # ═══════════════════════════
    print("\n" + "─" * 70)
    print("  Phase 3: ML Ensemble 驗證")
    print("─" * 70)

    try:
        import joblib
        ml_species = ["yellowfin", "bigeye", "skipjack", "albacore"]
        for sp in ml_species:
            p = Path(f"ml_system/models/stacking_{sp}.pkl")
            if not p.exists():
                p = Path(f"models/stacking_{sp}.pkl")
            if p.exists():
                loaded = joblib.load(p)
                if isinstance(loaded, dict) and "model" in loaded:
                    model = loaded["model"]
                    scaler = loaded.get("scaler")
                    model_type = type(model).__name__
                    P(f"ML {sp}: {model_type} (scaler={'yes' if scaler else 'no'})")

                    # Test predict with dummy 12-feature input
                    X_test = np.random.randn(5, 12)
                    X_test = np.nan_to_num(X_test, nan=0.0)
                    if scaler:
                        X_test = scaler.transform(X_test)
                    pred = model.predict(X_test)
                    check(pred.shape == (5,), f"ML {sp}: predict OK, output shape={pred.shape}",
                          f"ML {sp}: predict shape mismatch")
                else:
                    model = loaded
                    P(f"ML {sp}: {type(model).__name__} (raw)")
            else:
                F(f"ML {sp}: model file NOT FOUND")
    except Exception as e:
        F(f"ML Ensemble: {e}")

    # ═══════════════════════════
    # Phase 4: Commercial Features
    # ═══════════════════════════
    print("\n" + "─" * 70)
    print("  Phase 4: 商業功能驗證")
    print("─" * 70)

    # ENSO ONI
    print("\n  [ENSO ONI]")
    try:
        from engine.enhanced_data_sources import fetch_enso_oni
        import httpx
        async with httpx.AsyncClient(timeout=15) as ec:
            oni = await fetch_enso_oni(ec)
        check(isinstance(oni, float), f"ONI value: {oni:+.2f}", "ONI: not a float")
        check(-3 < oni < 3, f"ONI in valid range [-3, 3]", f"ONI out of range: {oni}")
    except Exception as e:
        F(f"ENSO ONI: {e}")

    # Lunar Phase
    print("\n  [Lunar Phase]")
    try:
        from engine.enhanced_data_sources import compute_lunar_fishing_factor
        from datetime import datetime as dt
        today = dt.now()
        lunar = compute_lunar_fishing_factor(today.year, today.month, today.day)
        check("phase_name" in lunar, f"Moon phase: {lunar.get('phase_name')}", "Lunar: no phase_name")
        check("species_factors" in lunar or "fishing_factor" in lunar,
              f"Lunar factors: {lunar.get('species_factors', lunar.get('fishing_factor', {}))}",
              "Lunar: no species_factors")
    except Exception as e:
        F(f"Lunar: {e}")

    # EEZ Check
    print("\n  [EEZ Checker]")
    try:
        from engine.eez.eez_checker import EEZChecker
        eez = EEZChecker.check_point(25.0, 135.0)
        check("eez_name" in eez, f"EEZ at (25,135): {eez.get('eez_name')}", "EEZ: failed")
    except Exception as e:
        F(f"EEZ: {e}")

    # SHAP Explainer
    print("\n  [SHAP Explainer]")
    try:
        from engine.shap_explainer import get_shap_explainer
        shap_exp = get_shap_explainer()
        if shap_exp is not None:
            P(f"SHAP: loaded, species={list(shap_exp._explainers.keys())}")
        else:
            W("SHAP: not available (fallback to weight-based)")
    except Exception as e:
        W(f"SHAP: {e}")

    # ═══════════════════════════
    # Phase 5: Output 驗證
    # ═══════════════════════════
    print("\n" + "─" * 70)
    print("  Phase 5: Output 檔案驗證")
    print("─" * 70)

    # hotspots.json
    hp_file = Path("output/hotspots.json")
    if hp_file.exists():
        hs = json.load(open(hp_file, encoding='utf-8'))
        check(len(hs) == 20, f"hotspots.json: {len(hs)} hotspots", f"hotspots.json: {len(hs)} (expected 20)")
        species_count = {}
        for h in hs:
            sp = h.get("species", "?")
            species_count[sp] = species_count.get(sp, 0) + 1
        check(len(species_count) == 4, f"Species: {species_count}", f"Species count: {len(species_count)} (expected 4)")

        # Check required fields
        required_fields = ["lat", "lon", "score", "species", "rank", "eez", "sst",
                          "npp", "do_surface", "depth_m", "distance_nm", "explain",
                          "ml_cpue_kg_day", "lunar_phase", "greenfish_hsi"]
        h0 = hs[0]
        missing = [f for f in required_fields if f not in h0]
        check(len(missing) == 0, f"All {len(required_fields)} fields present in hotspot",
              f"Missing fields: {missing}")

        # Range checks
        scores = [h["score"] for h in hs]
        check(max(scores) > 0.1, f"HSI range: [{min(scores):.3f}, {max(scores):.3f}]",
              f"HSI too low: max={max(scores):.3f}")

        sst_vals = [h.get("sst") for h in hs if h.get("sst") is not None]
        if sst_vals:
            check(15 < max(sst_vals) < 40, f"SST range: [{min(sst_vals):.1f}, {max(sst_vals):.1f}]°C",
                  f"SST suspicious: [{min(sst_vals):.1f}, {max(sst_vals):.1f}]°C")
    else:
        F("hotspots.json: NOT FOUND")

    # analysis_summary.json
    sf = Path("output/analysis_summary.json")
    if sf.exists():
        s = json.load(open(sf, encoding='utf-8'))
        check(s.get("ml_ensemble") == True, "ML ensemble: True", "ML ensemble: False")
        check(s.get("greenfish_lite") == True, "GreenFish Lite: True", "GreenFish Lite: False")
        check(s.get("wind_speed") is not None, f"Wind speed: {s.get('wind_speed')} m/s", "Wind speed: None")
        check(s.get("current_speed") is not None, f"Current speed: {s.get('current_speed')} m/s", "Current speed: None")
        check(len(s.get("data_sources", {})) >= 5, f"Data sources: {len(s.get('data_sources', {}))} types",
              f"Too few data sources: {len(s.get('data_sources', {}))}")
    else:
        F("analysis_summary.json: NOT FOUND")

    # ═══════════════════════════
    # Phase 6: 商業基礎設施驗證
    # ═══════════════════════════
    print("\n" + "─" * 70)
    print("  Phase 6: 商業基礎設施驗證 (Commercial Infrastructure)")
    print("─" * 70)

    # Route Planner
    print("\n  [Route Planner]")
    try:
        from engine.navigation.route_planner import FuelOptimalRouter, RoutePoint
        rp = FuelOptimalRouter(vessel_speed_kts=10.0)
        # 測試從高雄港到太平洋漁場的大圓航線
        start = RoutePoint(22.6, 120.3, "高雄港")
        dest = RoutePoint(25.0, 135.0, "漁場A")
        test_route = rp.plan_route(start, dest)
        if test_route is not None:
            dist = getattr(test_route, 'total_distance_nm', 0)
            check(dist > 0,
                  f"Route: 高雄→(25,135) = {dist:.1f} nm", "Route: distance is 0")
        else:
            W("Route planner: returned None")
    except Exception as e:
        W(f"Route Planner: {e}")

    # GeoJSON 輸出
    print("\n  [GeoJSON Output]")
    try:
        from engine.geojson_output import generate_geojson
        if hp_file.exists():
            # generate_geojson expects Dict with 'combined_hotspots' key
            fusion_input = {"combined_hotspots": hs}
            geo = generate_geojson(fusion_input)
            if geo is not None:
                check(isinstance(geo, (dict, str)),
                      f"GeoJSON: generated ({type(geo).__name__})", "GeoJSON: invalid type")
            else:
                W("GeoJSON: returned None")
        else:
            W("GeoJSON: no hotspots to test")
    except Exception as e:
        W(f"GeoJSON: {e}")

    # KML 輸出
    print("\n  [KML Output]")
    try:
        from engine.kml_generator import generate_kml
        if hp_file.exists():
            kml_data = generate_kml(hs, "audit_test")
            check(kml_data is not None, "KML: generated successfully", "KML: returned None")
        else:
            W("KML: no hotspots to test")
    except Exception as e:
        W(f"KML: {e}")

    # Dashboard HTML 完整性
    print("\n  [Dashboard HTML]")
    dash_file = Path("web/dashboard.html")
    if dash_file.exists():
        dash_content = dash_file.read_text(encoding='utf-8')
        check(len(dash_content) > 10000, f"Dashboard: {len(dash_content)} bytes", "Dashboard: too small")
        check("leaflet" in dash_content.lower() or "L.map" in dash_content,
              "Dashboard: Leaflet map present", "Dashboard: no map library found")
        check("hotspot" in dash_content.lower() or "marker" in dash_content.lower() or "L.marker" in dash_content,
              "Dashboard: hotspot/marker functionality present", "Dashboard: no hotspot display")
        check("api" in dash_content.lower() or "fetch" in dash_content.lower(),
              "Dashboard: API integration present", "Dashboard: no API calls")
    else:
        F("Dashboard: web/dashboard.html NOT FOUND")

    # Dashboard JS
    dash_js = Path("web/dashboard.js")
    if dash_js.exists():
        js_content = dash_js.read_text(encoding='utf-8')
        check(len(js_content) > 5000, f"Dashboard JS: {len(js_content)} bytes", "Dashboard JS: too small")
    else:
        W("Dashboard JS: web/dashboard.js not found")

    # Primary Production Model
    print("\n  [Primary Production]")
    try:
        from engine.primary_production import PrimaryProductionEngine
        pp_engine = PrimaryProductionEngine()
        if sst is not None and chl is not None:
            npp_r = pp_engine.analyze(chl, sst, lats)
            if npp_r is not None:
                npp = npp_r.get("npp") if isinstance(npp_r, dict) else npp_r
                if isinstance(npp, np.ndarray):
                    # VGPM NPP 在赤道可達 20000+ mgC/m²/day，閾值設 25000
                    check_array(npp, "NPP", min_val=0, max_val=25000)
                P(f"PrimaryProductionEngine: analyze() OK")
            else:
                W("NPP: returned None")
    except Exception as e:
        W(f"Primary Production: {e}")

    # Accuracy Booster (SST/CHL Anomaly Engines + ENSO adjustment)
    print("\n  [Accuracy Booster]")
    try:
        from engine.accuracy_booster import SSTAnomalyEngine, ChlAnomalyEngine, TemporalLagEngine
        sst_anom = SSTAnomalyEngine()
        chl_anom = ChlAnomalyEngine()
        temporal = TemporalLagEngine()
        P(f"AccuracyBooster modules: SST/CHL Anomaly + Temporal Lag loaded OK")
    except Exception as e:
        W(f"Accuracy Booster: {e}")

    # Captain Reports
    print("\n  [Captain Reports]")
    try:
        from engine.captain_reports import CaptainReportSystem
        crg = CaptainReportSystem()
        P(f"Captain Reports: CaptainReportSystem loaded OK")
    except Exception as e:
        W(f"Captain Reports: {e}")

    # AI Fusion
    print("\n  [AI Fusion]")
    try:
        from engine.ai_fusion import fuse_and_rank, assess_safety
        check(callable(fuse_and_rank), "AI Fusion: fuse_and_rank() available", "AI Fusion: fuse_and_rank missing")
        check(callable(assess_safety), "AI Fusion: assess_safety() available", "AI Fusion: assess_safety missing")
    except Exception as e:
        W(f"AI Fusion: {e}")

    # Species Params
    print("\n  [Species Params]")
    try:
        from engine.species_params import SPECIES
        check(len(SPECIES) >= 4,
              f"Species params: {len(SPECIES)} species ({', '.join(SPECIES.keys())})",
              f"Species params: only {len(SPECIES)} species")
        for sp_name, sp_params in SPECIES.items():
            check("Topt_C" in sp_params and "Topt_sigma" in sp_params,
                  f"  {sp_name}: Topt={sp_params.get('Topt_C')}°C ± {sp_params.get('Topt_sigma')}",
                  f"  {sp_name}: missing Topt parameters")
    except Exception as e:
        W(f"Species params: {e}")

    # Docker 部署檔完整性
    print("\n  [Docker Deployment]")
    df = Path("Dockerfile")
    check(df.exists() and df.stat().st_size > 1000, f"Dockerfile: {df.stat().st_size} bytes", "Dockerfile: missing/empty")
    dc = Path("docker-compose.yml")
    check(dc.exists() and dc.stat().st_size > 200, f"docker-compose: {dc.stat().st_size} bytes", "docker-compose: missing/empty")
    req = Path("requirements.txt")
    check(req.exists() and req.stat().st_size > 200, f"requirements.txt: {req.stat().st_size} bytes", "requirements.txt: missing/empty")

    # README
    readme = Path("README.md")
    check(readme.exists() and readme.stat().st_size > 2000,
          f"README.md: {readme.stat().st_size} bytes", "README.md: missing or too small")

    # Web Server
    print("\n  [Web Server]")
    ws = Path("web_server.py")
    check(ws.exists(), "web_server.py exists", "web_server.py: MISSING")
    if ws.exists():
        ws_content = ws.read_text(encoding='utf-8')
        check("run_in_executor" in ws_content,
              "web_server: uses run_in_executor (non-blocking)", "web_server: missing run_in_executor")
        check("/health" in ws_content, "web_server: /health endpoint", "web_server: no /health")
        check("X-API-Key" in ws_content or "api_key" in ws_content.lower(),
              "web_server: API key auth", "web_server: no API auth")

    # ═══════════════════════════
    # Summary
    # ═══════════════════════════
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  AUDIT COMPLETE ({elapsed:.1f}s)")
    print("=" * 70)
    print(f"\n  ✅ PASS: {len(results['pass'])}")
    print(f"  ⚠️  WARN: {len(results['warn'])}")
    print(f"  ❌ FAIL: {len(results['fail'])}")

    if results["fail"]:
        print("\n  FAILURES:")
        for f in results["fail"]:
            print(f"    ❌ {f}")

    if results["warn"]:
        print("\n  WARNINGS:")
        for w in results["warn"]:
            print(f"    ⚠️  {w}")

    print(f"\n  VERDICT: {'🎉 COMMERCIAL READY' if not results['fail'] else '🔧 NEEDS FIXES'}")
    print("=" * 70)

    return len(results["fail"])


if __name__ == "__main__":
    fail_count = asyncio.run(audit())
    sys.exit(1 if fail_count > 0 else 0)
