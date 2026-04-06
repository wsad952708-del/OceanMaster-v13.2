"""
OceanMaster v13.2 — 深層七層全系統檢驗
=====================================
L1: Import        — 所有模組能 import
L2: Init          — 初始化 + 方法存在
L3: Edge/NaN      — 邊界值, NaN, 空輸入
L4: Physics       — 物理合理性
L5: RealData      — 真實 CMEMS/MUR 數據
L6: Determinism   — 同輸入同輸出
L7: CrossModule   — 跨模組資料流
"""
import sys, os, time, traceback
os.chdir(r'c:\Users\user\Desktop\好像快好了\OceanMaster_v13_2')
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import asyncio

t_start = time.time()
PASS = 0; FAIL = 0; ERRORS = []

def T(layer, name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        ERRORS.append(f"L{layer} {name}: {detail}")
        print(f"  ❌ L{layer} {name}: {detail}")

def H(t):
    print(f"\n{'═'*55}\n  {t}\n{'═'*55}")

# ═══════════════════════════════════════════════════════
#  L1: IMPORT — 所有模組
# ═══════════════════════════════════════════════════════
H("L1: IMPORT (所有引擎模組)")
modules = [
    ("algorithms", "engine.algorithms"),
    ("ai_fusion", "engine.ai_fusion"),
    ("forecast_hsi", "engine.forecast_hsi"),
    ("hook_depth", "engine.hook_depth"),
    ("hsi_models", "engine.hsi_models"),
    ("route_planner_v2", "engine.route_planner_v2"),
    ("fuel_predictor", "engine.fuel_predictor"),
    ("carbon_calculator", "engine.carbon_calculator"),
    ("roi_calculator", "engine.roi_calculator"),
    ("departure_optimizer", "engine.departure_optimizer"),
    ("safety_checker", "engine.safety_checker"),
    ("okubo_weiss", "engine.okubo_weiss"),
    ("eddy_detector", "engine.eddy_detector"),
    ("eddy_maturity", "engine.eddy_maturity"),
    ("sst_change_detector", "engine.sst_change_detector"),
    ("ocean_color_fronts", "engine.ocean_color_fronts"),
    ("convergence_detector", "engine.convergence_detector"),
    ("dvm_model", "engine.dvm_model"),
    ("lunar_model", "engine.lunar_model"),
    ("competition_penalty", "engine.competition_penalty"),
    ("species_params", "engine.species_params"),
    ("ocean_srgan", "engine.ocean_srgan"),
    ("shap_explainer", "engine.shap_explainer"),
    ("gfw_heatmap", "engine.gfw_heatmap"),
    ("wave_fetcher", "engine.wave_fetcher"),
    ("dissolved_oxygen", "engine.dissolved_oxygen"),
    ("pressure_features", "engine.pressure_features"),
    ("thermocline_fetcher", "engine.thermocline_fetcher"),
    ("gebco_features", "engine.gebco_features"),
    ("npz_model", "engine.npz_model"),
    ("catch_composition", "engine.catch_composition"),
    ("longline_drift", "engine.longline_drift"),
    ("fleet_commander", "engine.fleet_commander"),
    ("market_optimizer", "engine.market_optimizer"),
    ("mur_sst_loader", "engine.mur_sst_loader"),
    ("convlstm", "engine.ml.convlstm_predictor"),
    ("transfish", "engine.ml.transfish"),
    ("unet", "engine.ml.unet_fishing"),
    ("cloud_removal", "engine.cloud_removal"),
    ("stacking", "engine.ml.stacking_ensemble"),
]
imported = {}
for name, mod_path in modules:
    try:
        imported[name] = __import__(mod_path, fromlist=[name])
        T(1, name, True)
    except Exception as e:
        T(1, name, False, str(e)[:60])

l1_pass = PASS
print(f"  L1 total: {PASS}/{len(modules)}")

# ═══════════════════════════════════════════════════════
#  L2: INIT + METHODS
# ═══════════════════════════════════════════════════════
H("L2: INIT + METHOD CHECK")
checks = [
    ("OkuboWeiss", lambda: imported["okubo_weiss"].OkuboWeissAnalyzer(), ["compute"]),
    ("EddyDetector", lambda: imported["eddy_detector"].EddyDetector(), ["detect"]),
    ("EddyMaturity", lambda: imported["eddy_maturity"].EddyMaturityEstimator(), ["compute"]),
    ("SSTChange", lambda: imported["sst_change_detector"].SSTChangeDetector(), ["compute"]),
    ("FuelPredictor", lambda: imported["fuel_predictor"].FuelPredictor(), ["predict"]),
    ("CarbonCalc", lambda: imported["carbon_calculator"].CarbonCalculator(), ["calculate"]),
    ("ROICalc", lambda: imported["roi_calculator"].ROICalculator(), ["calculate"]),
    ("DVM", lambda: imported["dvm_model"].DVMModel(), ["compute_accessibility"]),
    ("OceanSRGAN", lambda: imported["ocean_srgan"].OceanSRGAN(), ["super_resolve"]),
    ("DepartureOpt", lambda: imported["departure_optimizer"].DepartureOptimizer(), ["optimize"]),
    ("ConvLSTMCell", lambda: imported["convlstm"].ConvLSTMCell(1, 8), ["forward", "__call__"]),
]
for name, init_fn, methods in checks:
    try:
        obj = init_fn()
        for m in methods:
            T(2, f"{name}.{m}", hasattr(obj, m), f"missing method {m}")
    except Exception as e:
        T(2, name, False, str(e)[:60])

# ═══════════════════════════════════════════════════════
#  L3: EDGE / NaN / BOUNDARY
# ═══════════════════════════════════════════════════════
H("L3: EDGE CASES + NaN")

# NaN SST
sst_nan = np.full((10, 10), np.nan, dtype=np.float32)
try:
    from engine.hsi_models import compute_hsi_yellowfin
    r = compute_hsi_yellowfin(sst=sst_nan)
    hsi = r["hsi"]
    T(3, "HSI with NaN SST", np.all(np.isfinite(hsi)) or np.all(np.isnan(hsi)), "should handle NaN")
except Exception as e:
    T(3, "HSI NaN", False, str(e)[:60])

# Zero-size array
try:
    from engine.algorithms import detect_sst_fronts
    r = detect_sst_fronts(np.ones((3,3), dtype=np.float32)*25, np.array([20,21,22.0]), np.array([120,121,122.0]))
    T(3, "Fronts 3x3 grid", True, "small grid OK")
except Exception as e:
    T(3, "Fronts small", False, str(e)[:60])

# Extreme SST
try:
    from engine.hook_depth import compute_hook_depth
    r = compute_hook_depth("bigeye", sst=5.0, hour_utc=12)
    T(3, "HookDepth cold SST=5", r["hook_depth_optimal"] > 0, f"depth={r['hook_depth_optimal']}")
    r2 = compute_hook_depth("bigeye", sst=35.0, hour_utc=12)
    T(3, "HookDepth hot SST=35", r2["hook_depth_optimal"] > 0, f"depth={r2['hook_depth_optimal']}")
except Exception as e:
    T(3, "HookDepth extreme", False, str(e)[:60])

# Fuel negative wind
try:
    from engine.fuel_predictor import FuelPredictor
    fp = FuelPredictor()
    r = fp.predict(wave_height=0.0, wind_speed_ms=0.0, wind_angle_deg=0)
    T(3, "Fuel zero conditions", r["fuel_l_per_nm"] > 0, f"{r['fuel_l_per_nm']:.1f}")
    r2 = fp.predict(wave_height=5.0, wind_speed_ms=25.0, wind_angle_deg=180)
    T(3, "Fuel extreme storm", r2["fuel_l_per_nm"] > r["fuel_l_per_nm"], "storm > calm")
except Exception as e:
    T(3, "Fuel edge", False, str(e)[:60])

# Carbon zero fuel
try:
    from engine.carbon_calculator import CarbonCalculator
    r = CarbonCalculator().calculate(fuel_consumed_l=0, distance_nm=0)
    T(3, "Carbon zero", r["co2_kg"] == 0, f"co2={r['co2_kg']}")
except Exception as e:
    T(3, "Carbon zero", False, str(e)[:60])

# ═══════════════════════════════════════════════════════
#  L4: PHYSICS VALIDATION
# ═══════════════════════════════════════════════════════
H("L4: PHYSICS")

# Fuel: headwind > tailwind
try:
    fp = FuelPredictor()
    tail = fp.predict(wave_height=1.5, wind_speed_ms=10, wind_angle_deg=0)
    head = fp.predict(wave_height=1.5, wind_speed_ms=10, wind_angle_deg=180)
    T(4, "Fuel: head > tail", head["fuel_l_per_nm"] > tail["fuel_l_per_nm"],
      f"head={head['fuel_l_per_nm']:.1f} tail={tail['fuel_l_per_nm']:.1f}")
except Exception as e:
    T(4, "Fuel physics", False, str(e)[:60])

# Hook depth: night shallower for bigeye
try:
    day = compute_hook_depth("bigeye", sst=25, hour_utc=12)
    night = compute_hook_depth("bigeye", sst=25, hour_utc=0)
    T(4, "HookDepth: bigeye night<day", night["hook_depth_optimal"] < day["hook_depth_optimal"],
      f"night={night['hook_depth_optimal']}m day={day['hook_depth_optimal']}m")
except Exception as e:
    T(4, "Hook physics", False, str(e)[:60])

# Full moon deeper
try:
    new_moon = compute_hook_depth("bigeye", sst=25, hour_utc=22, lunar_illumination=0.0)
    full_moon = compute_hook_depth("bigeye", sst=25, hour_utc=22, lunar_illumination=1.0)
    T(4, "HookDepth: full>new moon", full_moon["hook_depth_optimal"] >= new_moon["hook_depth_optimal"],
      f"full={full_moon['hook_depth_optimal']}m new={new_moon['hook_depth_optimal']}m")
except Exception as e:
    T(4, "Hook moon", False, str(e)[:60])

# Carbon proportional
try:
    cc = CarbonCalculator()
    r1 = cc.calculate(fuel_consumed_l=1000, distance_nm=500)
    r2 = cc.calculate(fuel_consumed_l=2000, distance_nm=500)
    T(4, "Carbon: 2x fuel=2x CO2", abs(r2["co2_kg"]/r1["co2_kg"] - 2.0) < 0.01,
      f"ratio={r2['co2_kg']/r1['co2_kg']:.3f}")
except Exception as e:
    T(4, "Carbon linear", False, str(e)[:60])

# Safety: high waves = unsafe
try:
    from engine.safety_checker import is_safe_for_fishing
    safe = is_safe_for_fishing(lat=25, lon=130, wave_height=1.0, wind_speed=5)
    unsafe = is_safe_for_fishing(lat=25, lon=130, wave_height=6.0, wind_speed=30)
    T(4, "Safety: storm=unsafe", safe.score > unsafe.score, f"calm={safe.score:.1f} storm={unsafe.score:.1f}")
except Exception as e:
    T(4, "Safety physics", False, str(e)[:60])

# ═══════════════════════════════════════════════════════
#  L5: REAL DATA
# ═══════════════════════════════════════════════════════
H("L5: REAL DATA")
import xarray as xr

# MUR SST 1km
try:
    from engine.mur_sst_loader import load_mur_sst
    sst_mur, lats_m, lons_m = load_mur_sst()
    T(5, "MUR SST 1km load", sst_mur.shape[0] > 100, f"shape={sst_mur.shape}")
except Exception as e:
    T(5, "MUR SST", False, str(e)[:60])

# CMEMS SST
try:
    ds = xr.open_dataset("C:/tmp/L2_cache/cmems_sst.nc")
    sst = np.nan_to_num(ds["thetao"].isel(depth=0, time=-1).values.astype(np.float32), nan=25.0)
    lats = ds.latitude.values; lons = ds.longitude.values
    T(5, "CMEMS SST", sst.shape == (181, 301), f"shape={sst.shape}")
    ds.close()
except Exception as e:
    T(5, "CMEMS SST", False, str(e)[:60])

# CMEMS current
try:
    ds = xr.open_dataset("C:/tmp/L2_cache/cmems_phy.nc")
    u = np.nan_to_num(ds["uo"].isel(depth=0, time=-1).values.astype(np.float32))
    v = np.nan_to_num(ds["vo"].isel(depth=0, time=-1).values.astype(np.float32))
    T(5, "CMEMS current", u.shape == v.shape, f"u={u.shape}")
    ds.close()
except Exception as e:
    T(5, "CMEMS current", False, str(e)[:60])

# SSH
try:
    ds = xr.open_dataset("C:/tmp/L2_cache/cmems_ssh.nc")
    ssh = ds["zos"].isel(time=-1).values.astype(np.float32)
    if ssh.ndim == 3: ssh = ssh[0]
    T(5, "CMEMS SSH", ssh.shape[0] > 10, f"shape={ssh.shape}")
    ds.close()
except Exception as e:
    T(5, "SSH", False, str(e)[:60])

# 3D temp
try:
    ds = xr.open_dataset("C:/tmp/L2_cache/cmems_3d_deep.nc")
    depths = ds.depth.values
    T(5, "3D depth profile", len(depths) > 10, f"{len(depths)} levels, max={depths[-1]:.0f}m")
    ds.close()
except Exception as e:
    T(5, "3D depth", False, str(e)[:60])

# Fronts on real SST
try:
    f = detect_sst_fronts(sst, lats, lons)
    grad = f.get("gradient", f.get("front_strength", np.zeros_like(sst)))
    n = int(np.sum(grad > 0.01))
    T(5, "Fronts on CMEMS", n > 0, f"{n} front pixels")
except Exception as e:
    T(5, "Fronts real", False, str(e)[:60])

# Fronts on MUR
try:
    f = detect_sst_fronts(sst_mur, lats_m, lons_m)
    grad = f.get("gradient", f.get("front_strength", np.zeros_like(sst_mur)))
    n_mur = int(np.sum(grad > 0.01))
    T(5, "Fronts on MUR 1km", n_mur > n, f"{n_mur} vs {n} (9km)")
except Exception as e:
    T(5, "Fronts MUR", False, str(e)[:60])

# FTLE on real
try:
    from engine.algorithms import compute_ftle
    r = compute_ftle(u[::4,::4], v[::4,::4], lats[::4], lons[::4])
    T(5, "FTLE real data", r["ftle"].shape[0] > 0, f"shape={r['ftle'].shape}")
except Exception as e:
    T(5, "FTLE real", False, str(e)[:60])

# OW on real
try:
    from engine.okubo_weiss import OkuboWeissAnalyzer
    r = OkuboWeissAnalyzer().compute(u, v, lats, lons)
    T(5, "OW real data", r["strain_pct"] > 0, f"strain={r['strain_pct']:.1f}%")
except Exception as e:
    T(5, "OW real", False, str(e)[:60])

# Eddy with SSH
try:
    from engine.eddy_detector import EddyDetector
    r = EddyDetector().detect(ssh=ssh, u_geo=u, v_geo=v, lats=lats, lons=lons)
    T(5, "Eddy with SSH", True, f"keys={list(r.keys())[:3]}")
except Exception as e:
    T(5, "Eddy real", False, str(e)[:60])

# ═══════════════════════════════════════════════════════
#  L6: DETERMINISM
# ═══════════════════════════════════════════════════════
H("L6: DETERMINISM")

try:
    r1 = compute_hook_depth("bigeye", sst=25, hour_utc=10)
    r2 = compute_hook_depth("bigeye", sst=25, hour_utc=10)
    T(6, "HookDepth deterministic", r1["hook_depth_optimal"] == r2["hook_depth_optimal"])
except Exception as e:
    T(6, "HookDepth det", False, str(e)[:60])

try:
    r1 = FuelPredictor().predict(wave_height=2, wind_speed_ms=10, wind_angle_deg=90)
    r2 = FuelPredictor().predict(wave_height=2, wind_speed_ms=10, wind_angle_deg=90)
    T(6, "Fuel deterministic", r1["fuel_l_per_nm"] == r2["fuel_l_per_nm"])
except Exception as e:
    T(6, "Fuel det", False, str(e)[:60])

try:
    r1 = CarbonCalculator().calculate(fuel_consumed_l=1000, distance_nm=500)
    r2 = CarbonCalculator().calculate(fuel_consumed_l=1000, distance_nm=500)
    T(6, "Carbon deterministic", r1["co2_kg"] == r2["co2_kg"])
except Exception as e:
    T(6, "Carbon det", False, str(e)[:60])

try:
    from engine.hsi_models import compute_hsi_yellowfin
    r1 = compute_hsi_yellowfin(sst=sst)
    r2 = compute_hsi_yellowfin(sst=sst)
    T(6, "HSI deterministic", np.array_equal(r1["hsi"], r2["hsi"]))
except Exception as e:
    T(6, "HSI det", False, str(e)[:60])

# ═══════════════════════════════════════════════════════
#  L7: CROSS-MODULE INTEGRATION
# ═══════════════════════════════════════════════════════
H("L7: CROSS-MODULE PIPELINE")

try:
    # HSI → Fronts → OW → Hook → Fuel → Carbon → ROI
    hsi = compute_hsi_yellowfin(sst=sst)["hsi"]
    f = detect_sst_fronts(sst, lats, lons)
    ow = OkuboWeissAnalyzer().compute(u, v, lats, lons)
    hd = compute_hook_depth("yellowfin", sst=float(np.nanmean(sst)), hour_utc=10)
    fuel = FuelPredictor().predict(wave_height=1.5, wind_speed_ms=8, wind_angle_deg=90)
    carbon = CarbonCalculator().calculate(fuel_consumed_l=fuel["fuel_l_per_nm"]*500, distance_nm=500)
    roi = imported["roi_calculator"].ROICalculator().calculate(fuel_saved_l=500, time_saved_hours=12)

    T(7, "Pipeline HSI→Fronts→OW→Hook→Fuel→Carbon→ROI", True,
      f"HSI={np.nanmax(hsi):.2f}, hook={hd['hook_depth_optimal']}m, fuel={fuel['fuel_l_per_nm']:.1f}L/nm, CO2={carbon['co2_kg']:.0f}kg, ROI=NT${roi['total_savings_twd']:,.0f}")
except Exception as e:
    T(7, "Full pipeline", False, str(e)[:80])

# MUR SST → Front → HSI
try:
    f_mur = detect_sst_fronts(sst_mur, lats_m, lons_m)
    hsi_mur = compute_hsi_yellowfin(sst=sst_mur)["hsi"]
    T(7, "MUR→Fronts→HSI", True,
      f"MUR fronts={int(np.sum(f_mur.get('gradient',np.zeros(1))>0.01)):,}, HSI max={np.nanmax(hsi_mur):.3f}")
except Exception as e:
    T(7, "MUR pipeline", False, str(e)[:60])

# DVM + Hook + Safety
try:
    dvm = imported["dvm_model"].DVMModel()
    acc = dvm.compute_accessibility(species="bigeye", sst=25, mld=80, z20=200, is_daytime=False)
    hd = compute_hook_depth("bigeye", sst=25, hour_utc=22, lunar_illumination=0.9)
    safe = is_safe_for_fishing(lat=25, lon=130, wave_height=1.5, wind_speed=8)
    T(7, "DVM→Hook→Safety", True,
      f"acc={float(acc):.2f}, depth={hd['hook_depth_optimal']}m, safety={safe.level.value}")
except Exception as e:
    T(7, "DVM+Hook+Safety", False, str(e)[:60])

# WaveFetcher async
try:
    from engine.wave_fetcher import WaveFetcher
    wf = WaveFetcher()
    r = asyncio.run(wf.fetch_wave_height((22,24),(128,132),np.arange(22,24.5,0.5),np.arange(128,132.5,0.5)))
    T(7, "WaveFetcher cascade", True, f"source={r['source']}, shape={r['wave_height'].shape}")
except Exception as e:
    T(7, "WaveFetcher", False, str(e)[:60])

# ═══════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════
elapsed = time.time() - t_start
print(f"\n{'═'*55}")
print(f"  DEEP 7-LAYER VERIFICATION COMPLETE")
print(f"  Total: {PASS}/{PASS+FAIL} passed ({PASS/(PASS+FAIL)*100:.1f}%)")
print(f"  Time:  {elapsed:.1f}s")
print(f"  L1 Import:     {l1_pass}/{len(modules)}")
print(f"  L2-L7 Tests:   {PASS-l1_pass}/{PASS+FAIL-len(modules)}")
print(f"{'═'*55}")
if ERRORS:
    print(f"\n  FAILURES ({len(ERRORS)}):")
    for e in ERRORS:
        print(f"    ❌ {e}")
else:
    print("\n  *** ALL TESTS PASSED — ZERO FAILURES ***")
print(f"{'═'*55}")
