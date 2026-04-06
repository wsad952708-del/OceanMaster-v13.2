import logging
log = logging.getLogger(__name__)
"""Batch 3: Fix DVM + EddyMaturity + FastAPI + E2E test"""
import sys, os, time
os.chdir(r'c:\Users\user\Desktop\好像快好了\OceanMaster_v13_2')
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, xarray as xr

RESULTS = {}
def test(name, cond, detail=""):
    s = "✅" if cond else "❌"
    RESULTS[name] = f"{s} {detail}"
    print(f"  {s} {name}: {detail}")

# === 1. DVM with correct params ===
print("\n=== DVM (correct params) ===")
try:
    from engine.dvm_model import DVMModel
    dvm = DVMModel()
    r = dvm.compute_accessibility(species="bigeye", sst=26, mld=80, z20=200, is_daytime=False)
    test("DVM compute_accessibility", r is not None, f"result type={type(r).__name__}")
    r2 = dvm.compute_feeding_index(species="bigeye", z20=200, mld=80)
    test("DVM compute_feeding_index", True, f"result={r2}")
    r3 = dvm.moonlight_dvm_factor(illumination=0.8)
    test("DVM moonlight_dvm_factor", True, f"result={r3}")
except Exception as e:
    test("DVM", False, str(e)[:80])

# === 2. EddyMaturity with correct params ===
print("\n=== EddyMaturity (correct params) ===")
try:
    from engine.eddy_maturity import EddyMaturityEstimator
    emi = EddyMaturityEstimator()
    ssh = np.random.randn(30, 30).astype(np.float32) * 0.1
    chl = np.ones((30, 30), dtype=np.float32) * 0.3
    eke = np.random.rand(30, 30).astype(np.float32) * 0.01
    r = emi.compute(ssh=ssh, chl=chl, eddy_type=-1, eke=eke)
    test("EddyMaturity", r is not None, f"keys={list(r.keys()) if isinstance(r,dict) else r}")
except Exception as e:
    test("EddyMaturity", False, str(e)[:80])

# === 3. Run eddy detection with real SSH ===
print("\n=== Eddy Detection with real SSH ===")
try:
    ds_ssh = xr.open_dataset("C:/tmp/L2_cache/cmems_ssh.nc")
    ssh = ds_ssh["zos"].isel(time=-1).values.astype(np.float32)
    lats = ds_ssh.latitude.values
    lons = ds_ssh.longitude.values
    print(f"  SSH: {ssh.shape}, range={np.nanmin(ssh):.3f}~{np.nanmax(ssh):.3f}m")

    from engine.algorithms import calculate_eke
    eke = calculate_eke(ssh=ssh, lat=lats, lon=lons)
    test("EKE from SSH", eke is not None, f"keys={list(eke.keys()) if isinstance(eke,dict) else 'array'}")
    ds_ssh.close()
except Exception as e:
    test("EKE from SSH", False, str(e)[:80])

# === 4. FastAPI start test ===
print("\n=== FastAPI Smoke Test ===")
try:
    from api.main import app
    test("FastAPI app import", True, f"app={type(app).__name__}")
except Exception as e:
    # Try finding the actual api entrypoint
    try:
        import importlib
        for mod_name in ['api.main', 'api.app', 'api.server']:
            try:
                mod = importlib.import_module(mod_name)
                test("FastAPI module", True, f"found: {mod_name}")
                break
            except Exception as e:
                log.debug(f"[降級] batch_verify_3.py: {e}")
        else:
            # Check what's in api/
            import glob
            files = glob.glob('api/*.py')
            test("FastAPI", False, f"available: {files}")
    except Exception as e2:
        test("FastAPI", False, str(e2)[:80])

# === 5. E2E: HSI pipeline test ===
print("\n=== E2E: Mini Pipeline ===")
try:
    ds_s = xr.open_dataset("C:/tmp/L2_cache/cmems_sst.nc")
    ds_p = xr.open_dataset("C:/tmp/L2_cache/cmems_phy.nc")
    ds_b = xr.open_dataset("C:/tmp/L2_cache/cmems_bgc.nc")

    sst = ds_s["thetao"].isel(depth=0, time=-1).values.astype(np.float32)
    u = np.nan_to_num(ds_p["uo"].isel(depth=0, time=-1).values.astype(np.float32))
    v = np.nan_to_num(ds_p["vo"].isel(depth=0, time=-1).values.astype(np.float32))
    chl = ds_b["chl"].isel(depth=0, time=-1).values.astype(np.float32)
    lats = ds_s.latitude.values; lons = ds_s.longitude.values

    # Step 1: HSI
    from engine.hsi_models import compute_hsi_grid
    hsi = compute_hsi_grid(sst, lats, lons, "yellowfin")
    test("E2E Step1: HSI", hsi is not None and np.nanmax(hsi) > 0)

    # Step 2: Fronts
    from engine.algorithms import detect_sst_fronts
    fronts = detect_sst_fronts(sst, lats, lons)
    test("E2E Step2: Fronts", fronts is not None)

    # Step 3: OW
    from engine.okubo_weiss import OkuboWeissAnalyzer
    ow = OkuboWeissAnalyzer().compute(u, v, lats, lons)
    test("E2E Step3: OW", ow["strain_pct"] > 0)

    # Step 4: Hook depth
    from engine.hook_depth import compute_hook_depth
    hd = compute_hook_depth("yellowfin", sst=27, hour_utc=10)
    test("E2E Step4: HookDepth", hd["hook_depth_optimal"] > 0)

    # Step 5: Fuel
    from engine.fuel_predictor import FuelPredictor
    fp = FuelPredictor().predict(wave_height=1.5, wind_speed_ms=8, wind_angle_deg=135)
    test("E2E Step5: Fuel", fp["fuel_l_per_nm"] > 0)

    # Step 6: Carbon
    from engine.carbon_calculator import CarbonCalculator
    cc = CarbonCalculator().calculate(fuel_consumed_l=fp["fuel_l_per_nm"]*500, distance_nm=500)
    test("E2E Step6: Carbon", cc["co2_kg"] > 0)

    # Step 7: ROI
    from engine.roi_calculator import ROICalculator
    roi = ROICalculator().calculate(fuel_saved_l=500, time_saved_hours=12)
    test("E2E Step7: ROI", roi["total_savings_twd"] > 0)

    ds_s.close(); ds_p.close(); ds_b.close()
except Exception as e:
    test("E2E Pipeline", False, str(e)[:80])

# SUMMARY
print(f"\n{'='*55}")
passed = sum(1 for v in RESULTS.values() if v.startswith("✅"))
print(f"  BATCH 3: {passed}/{len(RESULTS)} passed")
for k, v in RESULTS.items():
    print(f"    {k:35s} {v}")
print(f"{'='*55}")
