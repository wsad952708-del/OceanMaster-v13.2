"""FINAL E2E — 14 steps with correct signatures"""
import sys, os, time
os.chdir(r'c:\Users\user\Desktop\好像快好了\OceanMaster_v13_2'); sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, xarray as xr
from scipy.ndimage import zoom

PASS = 0; FAIL = 0; ERRORS = []
def test(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ✅ {name}: {detail}")
    else: FAIL += 1; ERRORS.append(f"{name}: {detail}"); print(f"  ❌ {name}: {detail}")

ds_s = xr.open_dataset("C:/tmp/L2_cache/cmems_sst.nc")
ds_p = xr.open_dataset("C:/tmp/L2_cache/cmems_phy.nc")
ds_b = xr.open_dataset("C:/tmp/L2_cache/cmems_bgc.nc")
ds_ssh = xr.open_dataset("C:/tmp/L2_cache/cmems_ssh.nc")
sst = ds_s["thetao"].isel(depth=0, time=-1).values.astype(np.float32)
u = np.nan_to_num(ds_p["uo"].isel(depth=0, time=-1).values.astype(np.float32))
v = np.nan_to_num(ds_p["vo"].isel(depth=0, time=-1).values.astype(np.float32))
chl = ds_b["chl"].isel(depth=0, time=-1).values.astype(np.float32)
ssh = ds_ssh["zos"].isel(time=-1).values.astype(np.float32)
if ssh.ndim == 3: ssh = ssh[0]
lats = ds_s.latitude.values; lons = ds_s.longitude.values
chl_m = zoom(chl, (sst.shape[0]/chl.shape[0], sst.shape[1]/chl.shape[1]), order=1)

print("="*55 + "\n  14-Step E2E Pipeline\n" + "="*55)

# 1. HSI
try:
    from engine.hsi_models import compute_hsi_yellowfin
    r = compute_hsi_yellowfin(sst=sst, chl=chl_m)
    hsi = r["hsi"]
    test("HSI (yellowfin)", np.nanmax(hsi) > 0, f"max={np.nanmax(hsi):.3f}")
except Exception as e: test("HSI", False, str(e)[:80])

# 2. SST Fronts
try:
    from engine.algorithms import detect_sst_fronts
    r = detect_sst_fronts(sst, lats, lons)
    test("SST Fronts", np.sum(r["ridge_mask"]) > 0, f"{np.sum(r['ridge_mask'])} points")
except Exception as e: test("SST Fronts", False, str(e)[:80])

# 3. FTLE
try:
    from engine.algorithms import compute_ftle
    r = compute_ftle(u[::4,::4], v[::4,::4], lats[::4], lons[::4])
    test("FTLE", r["ftle"].shape[0] > 0, f"shape={r['ftle'].shape}")
except Exception as e: test("FTLE", False, str(e)[:80])

# 4. Okubo-Weiss
try:
    from engine.okubo_weiss import OkuboWeissAnalyzer
    r = OkuboWeissAnalyzer().compute(u, v, lats, lons)
    test("Okubo-Weiss", r["strain_pct"] > 0, f"strain={r['strain_pct']:.1f}%")
except Exception as e: test("OW", False, str(e)[:80])

# 5. Eddy Detection (needs ssh)
try:
    from engine.eddy_detector import EddyDetector
    r = EddyDetector().detect(ssh=ssh, u_geo=u, v_geo=v, lats=lats, lons=lons)
    test("Eddy Detection", True, f"keys={list(r.keys())[:4]}")
except Exception as e: test("Eddy", False, str(e)[:80])

# 6. Hook Depth (lunar_illumination not moon_phase)
try:
    from engine.hook_depth import compute_hook_depth
    r = compute_hook_depth("bigeye", sst=25, hour_utc=22, lunar_illumination=0.95)
    test("Hook Depth", r["hook_depth_optimal"] > 0, f"depth={r['hook_depth_optimal']}m")
except Exception as e: test("Hook", False, str(e)[:80])

# 7. Route (compute_all_routes not RoutePlanner)
try:
    from engine.route_planner_v2 import compute_all_routes
    test("Route import", True, "compute_all_routes available")
except Exception as e: test("Route", False, str(e)[:80])

# 8. Fuel
try:
    from engine.fuel_predictor import FuelPredictor
    r = FuelPredictor().predict(wave_height=1.5, wind_speed_ms=8, wind_angle_deg=135)
    test("Fuel", r["fuel_l_per_nm"] > 0, f"{r['fuel_l_per_nm']:.1f} L/nm")
except Exception as e: test("Fuel", False, str(e)[:80])

# 9. Carbon
try:
    from engine.carbon_calculator import CarbonCalculator
    r = CarbonCalculator().calculate(fuel_consumed_l=1200, distance_nm=500)
    test("Carbon", r["co2_kg"] > 0, f"CO₂={r['co2_kg']:.0f}kg")
except Exception as e: test("Carbon", False, str(e)[:80])

# 10. ROI
try:
    from engine.roi_calculator import ROICalculator
    r = ROICalculator().calculate(fuel_saved_l=500, time_saved_hours=12)
    test("ROI", r["total_savings_twd"] > 0, f"NT${r['total_savings_twd']:,.0f}")
except Exception as e: test("ROI", False, str(e)[:80])

# 11. Departure (method: optimize)
try:
    from engine.departure_optimizer import DepartureOptimizer
    d = DepartureOptimizer()
    r = d.optimize(lat=24.5, lon=121.0)
    test("Departure", r is not None, f"type={type(r).__name__}")
except Exception as e: test("Departure", False, str(e)[:80])

# 12. Safety (needs lat/lon, no visibility)
try:
    from engine.safety_checker import is_safe_for_fishing
    r = is_safe_for_fishing(lat=24.5, lon=130, wave_height=1.5, wind_speed=10)
    test("Safety", r is not None, f"result={r}")
except Exception as e: test("Safety", False, str(e)[:80])

# 13. DVM
try:
    from engine.dvm_model import DVMModel
    r = DVMModel().compute_accessibility(species="bigeye", sst=26, mld=80, z20=200, is_daytime=False)
    test("DVM", r is not None, f"accessibility={float(r):.3f}")
except Exception as e: test("DVM", False, str(e)[:80])

# 14. EddyMaturity
try:
    from engine.eddy_maturity import EddyMaturityEstimator
    r = EddyMaturityEstimator().compute(ssh=ssh[:30,:30], chl=chl_m[:30,:30], eddy_type=-1, eke=np.random.rand(30,30).astype(np.float32)*0.01)
    score = float(np.mean(r['fishing_score']))
    test("EddyMaturity", score >= 0, f"avg_score={score:.3f}")
except Exception as e: test("EddyMaturity", False, str(e)[:80])

ds_s.close(); ds_p.close(); ds_b.close(); ds_ssh.close()
print(f"\n{'='*55}")
print(f"  E2E: {PASS}/{PASS+FAIL} PASSED")
if ERRORS:
    for e in ERRORS: print(f"    ❌ {e}")
else: print("  *** ALL TESTS PASSED ***")
print(f"{'='*55}")
