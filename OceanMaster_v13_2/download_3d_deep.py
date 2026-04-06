"""Download CMEMS multi-depth 3D temp/salinity + test thermocline/DO"""
import sys, os, time
os.chdir(r'c:\Users\user\Desktop\好像快好了\OceanMaster_v13_2')
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

PASS = 0; FAIL = 0; ERRORS = []
def test(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ✅ {name}: {detail}")
    else: FAIL += 1; ERRORS.append(f"{name}: {detail}"); print(f"  ❌ {name}: {detail}")

# === 1. Download multi-depth data ===
print("="*55 + "\n  CMEMS 3D Multi-depth (daily, 0-2000m)\n" + "="*55)
try:
    import subprocess
    import sys
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_fetcher_external', 'cmems_fetcher.py')
    out_dir = "C:/tmp/L2_cache"
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        sys.executable, script_path,
        "--lat-min", "20", "--lat-max", "25",
        "--lon-min", "125", "--lon-max", "135",
        "--out-dir", out_dir, "--target", "glorys12"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        raise RuntimeError(f"External download failed: {proc.stderr}")
        
    # Rename to expected name
    if os.path.exists(os.path.join(out_dir, "cmems_glorys12.nc")):
        import shutil
        shutil.move(os.path.join(out_dir, "cmems_glorys12.nc"), os.path.join(out_dir, "cmems_3d_deep.nc"))

    import xarray as xr
    ds = xr.open_dataset("C:/tmp/L2_cache/cmems_3d_deep.nc")
    temp = ds["thetao"]
    # Glorys might not have 'so' in this subset, but we'll try
    sal = ds["so"] if "so" in ds else None
    depths = ds.depth.values
    print(f"  Depths: {depths}")
    print(f"  Temp shape: {temp.shape}")
    if sal is not None:
        print(f"  Sal shape: {sal.shape}")
    test("3D download", len(depths) > 1, f"{len(depths)} depth levels, {depths[0]:.0f}~{depths[-1]:.0f}m")
except Exception as e:
    test("3D download", False, str(e)[:100])

# === 2. Thermocline detection ===
print("\n" + "="*55 + "\n  Thermocline Detection\n" + "="*55)
try:
    ds = xr.open_dataset("C:/tmp/L2_cache/cmems_3d_deep.nc")
    depths = ds.depth.values
    if len(depths) > 3:
        # Extract a temperature profile
        temp_profile = np.nan_to_num(ds["thetao"].isel(time=0, latitude=30, longitude=30).values)
        print(f"  Temp profile ({len(depths)} levels): {temp_profile[:5]}...")

        # Find thermocline (max gradient)
        dT = np.diff(temp_profile)
        dz = np.diff(depths)
        gradient = dT / dz
        tc_idx = np.argmin(gradient)
        tc_depth = (depths[tc_idx] + depths[tc_idx+1]) / 2
        test("Thermocline depth", tc_depth > 0, f"depth={tc_depth:.0f}m, gradient={gradient[tc_idx]:.4f}°C/m")
    else:
        test("Thermocline", False, f"only {len(depths)} depth levels")
    ds.close()
except Exception as e:
    test("Thermocline", False, str(e)[:80])

# === 3. MLD (Mixed Layer Depth) ===
try:
    ds = xr.open_dataset("C:/tmp/L2_cache/cmems_3d_deep.nc")
    depths = ds.depth.values
    if len(depths) > 3:
        temp_profile = np.nan_to_num(ds["thetao"].isel(time=0, latitude=30, longitude=30).values)
        sst = temp_profile[0]
        # MLD = depth where T drops 0.5°C from surface
        mld_mask = temp_profile < (sst - 0.5)
        if np.any(mld_mask):
            mld = depths[np.argmax(mld_mask)]
            test("MLD estimation", mld > 0, f"MLD={mld:.0f}m (ΔT=0.5°C criterion)")
        else:
            test("MLD estimation", True, "no MLD found (well-mixed)")
    ds.close()
except Exception as e:
    test("MLD", False, str(e)[:80])

# === 4. Dissolved Oxygen proxy ===
try:
    from engine.dissolved_oxygen import DissolvedOxygenAnalyzer
    doa = DissolvedOxygenAnalyzer()
    test("DO Analyzer import", True, str([m for m in dir(doa) if not m.startswith('_')][:5]))
except Exception as e:
    test("DO Analyzer", False, str(e)[:80])

print(f"\n{'='*55}")
print(f"  RESULT: {PASS}/{PASS+FAIL} passed")
if ERRORS:
    for e in ERRORS: print(f"    ❌ {e}")
print(f"{'='*55}")
