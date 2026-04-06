"""
Batch execution: downloads + API pulls + method fixes + URL pings
All in one script for efficiency
"""
import sys, os, time, json
sys.stdout.reconfigure(encoding='utf-8')
os.chdir(r'c:\Users\user\Desktop\好像快好了\OceanMaster_v13_2')
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()

import numpy as np
t0 = time.time()
RESULTS = {}

def test(name, cond, detail=""):
    status = "✅" if cond else "❌"
    RESULTS[name] = f"{status} {detail}"
    print(f"  {status} {name}: {detail}")

# ═══ 1. CMEMS SSH 下載 ═══
print("\n=== 1. CMEMS SSH ===")
try:
    import subprocess
    import xarray as xr
    import os
    
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_fetcher_external', 'cmems_fetcher.py')
    out_dir = "C:/tmp/L2_cache"
    os.makedirs(out_dir, exist_ok=True)
    
    cmd = [
        "python", script_path,
        "--lat-min", "15", "--lat-max", "30",
        "--lon-min", "120", "--lon-max", "145",
        "--out-dir", out_dir, "--target", "ssh"
    ]
    
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"External CMEMS script failed: {proc.stderr}")
        
    ds = xr.open_dataset(os.path.join(out_dir, "cmems_ssh.nc"))
    ssh = ds["zos"].isel(time=-1).values
    test("CMEMS SSH", True, f"shape={ssh.shape}, range={np.nanmin(ssh):.3f}~{np.nanmax(ssh):.3f}m")
    ds.close()
except Exception as e:
    test("CMEMS SSH", False, str(e)[:80])

# ═══ 2. VIIRS 漁火 ═══
print("\n=== 2. VIIRS 漁火 ===")
try:
    import urllib.request
    firms_key = os.getenv("FIRMS_API_KEY", "")
    if firms_key:
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{firms_key}/VIIRS_SNPP_NRT/120,15,145,30/3"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read().decode('utf-8')
            lines = data.strip().split('\n')
            test("VIIRS 漁火", len(lines) > 1, f"{len(lines)-1} 個熱點")
    else:
        test("VIIRS 漁火", False, "FIRMS_API_KEY not set")
except Exception as e:
    test("VIIRS 漁火", False, str(e)[:80])

# ═══ 3. GFW AIS ═══
print("\n=== 3. GFW AIS ===")
try:
    gfw_key = os.getenv("GFW_API_KEY", "")
    if gfw_key:
        url = "https://gateway.api.globalfishingwatch.org/v3/4wings/report"
        headers = {"Authorization": f"Bearer {gfw_key}", "Content-Type": "application/json"}
        body = json.dumps({
            "region": {"dataset": "public-eez-areas", "id": 8492},
            "datasets": ["public-global-fishing-effort:latest"],
            "date-range": ["2026-02-01", "2026-02-28"],
            "spatial-resolution": "low",
            "temporal-resolution": "monthly",
            "group-by": "flag"
        }).encode()
        req = urllib.request.Request(url, data=body, headers=headers, method='POST')
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
            test("GFW AIS", True, f"response keys: {list(result.keys())[:5]}")
    else:
        test("GFW AIS", False, "GFW_API_KEY not set")
except Exception as e:
    test("GFW AIS", False, str(e)[:80])

# ═══ 4. DVM 方法名修正 ═══
print("\n=== 4. DVM 方法名驗證 ===")
try:
    from engine.dvm_model import DVMModel
    dvm = DVMModel()
    methods = [m for m in dir(dvm) if not m.startswith('_')]
    r = dvm.compute_accessibility(sst=26, mld=80, z20=200, species="bigeye", hour_utc=14)
    test("DVM compute_accessibility", r is not None, f"result={r}")
except Exception as e:
    # Try with different param names
    try:
        import inspect
        sig = inspect.signature(dvm.compute_accessibility)
        test("DVM", False, f"params: {list(sig.parameters.keys())}")
    except Exception:
        test("DVM", False, str(e)[:80])

# ═══ 5. 渦旋成熟指數 ═══
print("\n=== 5. 渦旋成熟指數 ===")
try:
    from engine.eddy_maturity import EddyMaturityEstimator
    import inspect
    sig = inspect.signature(EddyMaturityEstimator.compute)
    params = list(sig.parameters.keys())
    print(f"  Params: {params}")
    emi = EddyMaturityEstimator()
    # Try with correct params
    kwargs = {}
    if 'chl' in params: kwargs['chl'] = np.ones((10,10)) * 0.3
    if 'eddy_age_days' in params: kwargs['eddy_age_days'] = 10
    if 'sst_anomaly' in params: kwargs['sst_anomaly'] = np.ones((10,10)) * -1.0
    if 'chl_anomaly' in params: kwargs['chl_anomaly'] = np.ones((10,10)) * 0.5
    r = emi.compute(**kwargs)
    test("EddyMaturity", r is not None, f"keys={list(r.keys()) if isinstance(r,dict) else type(r)}")
except Exception as e:
    test("EddyMaturity", False, str(e)[:80])

# ═══ 6. URL 驗活 ═══
print("\n=== 6. URL Endpoint 驗活 ===")
endpoints = [
    ("HYCOM THREDDS", "https://tds.hycom.org/thredds/catalog.html"),
    ("WOA ERDDAP", "https://www.ncei.noaa.gov/erddap/info/index.html"),
    ("NOAA ERDDAP", "https://coastwatch.pfeg.noaa.gov/erddap/info/index.html"),
    ("Open-Meteo Marine", "https://marine-api.open-meteo.com/v1/marine?latitude=25&longitude=130&hourly=wave_height&forecast_days=1"),
    ("NOAA ONI (ENSO)", "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"),
    ("IBTrACS (JTWC)", "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.WP.list.v04r01.csv"),
]
for name, url in endpoints:
    try:
        req = urllib.request.Request(url, method='GET')
        req.add_header('User-Agent', 'OceanMaster/15.3')
        with urllib.request.urlopen(req, timeout=10) as resp:
            code = resp.getcode()
            size = len(resp.read(1024))
            test(f"URL:{name}", code == 200, f"HTTP {code}, {size}+ bytes")
    except Exception as e:
        test(f"URL:{name}", False, str(e)[:60])

# ═══ 7. FastAPI ═══
print("\n=== 7. FastAPI 模組檢查 ===")
try:
    from api import main as api_main
    test("FastAPI import", True, "module loaded")
except Exception as e:
    # Try alternative
    try:
        import importlib.util
        spec = importlib.util.find_spec("fastapi")
        test("FastAPI package", spec is not None, "fastapi installed" if spec else "NOT installed")
    except Exception:
        test("FastAPI", False, str(e)[:80])

# ═══ SUMMARY ═══
elapsed = time.time() - t0
print(f"\n{'='*55}")
passed = sum(1 for v in RESULTS.values() if v.startswith("✅"))
print(f"  BATCH 1: {passed}/{len(RESULTS)} passed ({elapsed:.1f}s)")
for k, v in RESULTS.items():
    print(f"    {k:30s} {v}")
print(f"{'='*55}")
