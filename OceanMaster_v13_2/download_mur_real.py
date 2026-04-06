"""Download REAL MUR SST 1km from NASA PODAAC"""
import sys, os, time
os.chdir(r'c:\Users\user\Desktop\好像快好了\OceanMaster_v13_2')
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

os.environ['EARTHDATA_USERNAME'] = 'wsad952701@gmail.com'
os.environ['EARTHDATA_PASSWORD'] = 'Flame854685123.'

print("="*55 + "\n  MUR SST 1km — Real NASA Data\n" + "="*55)

import earthaccess
auth = earthaccess.login(strategy="environment")
print(f"Auth: {auth}")

# Search for latest MUR SST
results = earthaccess.search_data(
    short_name="MUR-JPL-L4-GLOB-v4.1",
    temporal=("2026-02-24", "2026-02-27"),
    count=1
)
print(f"Found: {len(results)} granules")

if results:
    print("Downloading...")
    os.makedirs("C:/tmp/L2_cache/mur/", exist_ok=True)
    files = earthaccess.download(results, "C:/tmp/L2_cache/mur/")
    print(f"Files: {files}")

    # Load and subset to our region
    import xarray as xr
    ds = xr.open_dataset(files[0])
    print(f"Variables: {list(ds.data_vars)}")
    print(f"Full shape: {ds['analysed_sst'].shape}")

    # Subset to 20-25N, 125-135E
    sst = ds['analysed_sst'].sel(lat=slice(20, 25), lon=slice(125, 135))
    print(f"Subset shape: {sst.shape}")
    sst_vals = sst.values.astype(np.float32)
    if sst_vals.ndim == 3: sst_vals = sst_vals[0]
    # MUR is in Kelvin
    if np.nanmean(sst_vals) > 200: sst_vals -= 273.15
    print(f"SST range: {np.nanmin(sst_vals):.1f} ~ {np.nanmax(sst_vals):.1f} °C")
    print(f"Resolution: {sst_vals.shape} ({sst_vals.shape[0]}×{sst_vals.shape[1]})")

    # Save subset
    lats = sst.lat.values
    lons = sst.lon.values
    ds_out = xr.Dataset(
        {'analysed_sst': (['lat', 'lon'], np.nan_to_num(sst_vals, nan=25.0))},
        coords={'lat': lats, 'lon': lons}
    )
    ds_out.to_netcdf("C:/tmp/L2_cache/mur_sst_1km.nc")
    print(f"Saved: C:/tmp/L2_cache/mur_sst_1km.nc")
    ds.close()

    # Front detection
    from engine.algorithms import detect_sst_fronts
    sst_clean = np.nan_to_num(sst_vals, nan=25.0)
    f = detect_sst_fronts(sst_clean, lats, lons)
    grad = f.get("gradient", f.get("front_strength", np.zeros_like(sst_clean)))
    n = int(np.sum(grad > 0.01))
    print(f"\nFront detection: {n:,} front pixels")
    print(f"Gradient max: {np.nanmax(grad):.4f} °C/km")

    print(f"\n{'='*55}")
    print(f"  ✅ REAL MUR SST 1km READY")
    print(f"  {sst_vals.shape[0]}×{sst_vals.shape[1]} pixels, {n:,} fronts")
    print(f"{'='*55}")
else:
    print("No granules found — trying different date range...")
    results = earthaccess.search_data(
        short_name="MUR-JPL-L4-GLOB-v4.1",
        temporal=("2026-02-20", "2026-02-28"),
        count=3
    )
    print(f"Extended search: {len(results)} granules")
    if results:
        for r in results:
            print(f"  {r}")
