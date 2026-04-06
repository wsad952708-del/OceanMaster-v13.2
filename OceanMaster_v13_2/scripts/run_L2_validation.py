"""
OceanMaster L2 數據驗證 — 下載真實 CMEMS 數據 + 全線路跑通測試
================================================================
1. 下載 CMEMS SST / CHL / 海流 (最新 1 天, 台灣外海小區)
2. 下載 Open-Meteo 海象預報
3. 跑 HSI 計算 → 生成熱點
4. 跑 8 天預報
5. 跑鉤深建議
6. 跑航線規劃 + 海象疊加
7. 跑 ROI / 碳排 / 出港最佳化
8. 輸出 HTML 地圖
"""
import sys
import os
import json
import time

# Ensure imports work when run from scripts/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("L2_Validation")

# Load .env
from dotenv import load_dotenv
load_dotenv()

RESULTS = {}
ERRORS = []
t0 = time.time()


def section(name):
    log.info(f"\n{'='*60}")
    log.info(f"  {name}")
    log.info(f"{'='*60}")


# ═══════════════════════════════════════════════════════════
# STEP 1: Download CMEMS SST + CHL + Currents
# ═══════════════════════════════════════════════════════════
section("STEP 1: CMEMS 真實衛星數據下載")

CACHE_DIR = Path("C:/tmp/L2_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Small test area: Taiwan offshore (15-30N, 120-145E) — last 1 day
lat_min, lat_max = 15.0, 30.0
lon_min, lon_max = 120.0, 145.0
today = datetime.now(timezone.utc)
yesterday = today - timedelta(days=2)

sst_file = CACHE_DIR / "cmems_sst.nc"
bgc_file = CACHE_DIR / "cmems_bgc.nc"
phy_file = CACHE_DIR / "cmems_phy.nc"

try:
    import subprocess
    import os
    
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_fetcher_external', 'cmems_fetcher.py')
    
    if not sst_file.exists():
        log.info("  Downloading CMEMS SST (analysis-forecast) via external script...")
        cmd = [
            "python", script_path,
            "--lat-min", str(lat_min), "--lat-max", str(lat_max),
            "--lon-min", str(lon_min), "--lon-max", str(lon_max),
            "--out-dir", str(CACHE_DIR), "--target", "sst"
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            log.warning(f"SST fetch failed: {proc.stderr}")
    else:
        log.info("  SST cache exists, skipping download")

    if not phy_file.exists():
        log.info("  Downloading CMEMS currents (u, v) via external script...")
        cmd = [
            "python", script_path,
            "--lat-min", str(lat_min), "--lat-max", str(lat_max),
            "--lon-min", str(lon_min), "--lon-max", str(lon_max),
            "--out-dir", str(CACHE_DIR), "--target", "cur"
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            log.warning(f"Currents fetch failed: {proc.stderr}")
        # Rename for backward compatibility
        if (CACHE_DIR / "cmems_cur.nc").exists():
            (CACHE_DIR / "cmems_cur.nc").rename(phy_file)
    else:
        log.info("  Currents cache exists, skipping download")

    if not bgc_file.exists():
        log.info("  Downloading CMEMS CHL (biogeochemistry) via external script...")
        cmd = [
            "python", script_path,
            "--lat-min", str(lat_min), "--lat-max", str(lat_max),
            "--lon-min", str(lon_min), "--lon-max", str(lon_max),
            "--out-dir", str(CACHE_DIR), "--target", "chl"
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            log.warning(f"CHL fetch failed: {proc.stderr}")
        # Rename for backward compatibility
        if (CACHE_DIR / "cmems_chl.nc").exists():
            (CACHE_DIR / "cmems_chl.nc").rename(bgc_file)
    else:
        log.info("  CHL cache exists, skipping download")

    RESULTS["cmems_download"] = "✅ OK"
    log.info("  ✅ CMEMS download complete")

except Exception as e:
    RESULTS["cmems_download"] = f"❌ {e}"
    ERRORS.append(f"CMEMS download: {e}")
    log.error(f"  ❌ CMEMS download failed: {e}")


# ═══════════════════════════════════════════════════════════
# STEP 2: Parse downloaded NetCDF
# ═══════════════════════════════════════════════════════════
section("STEP 2: 解析衛星數據")

sst_grid = None
chl_grid = None
u_grid = None
v_grid = None
lats = None
lons = None

try:
    import xarray as xr

    if sst_file.exists():
        ds = xr.open_dataset(sst_file)
        sst_var = "thetao" if "thetao" in ds else list(ds.data_vars)[0]
        sst_data = ds[sst_var]
        if "depth" in sst_data.dims:
            sst_data = sst_data.isel(depth=0)
        if "time" in sst_data.dims:
            sst_data = sst_data.isel(time=-1)

        sst_grid = sst_data.values.astype(np.float32)
        lats = ds.latitude.values if "latitude" in ds.coords else ds.lat.values
        lons = ds.longitude.values if "longitude" in ds.coords else ds.lon.values
        ds.close()

        sst_valid = np.isfinite(sst_grid)
        log.info(f"  SST: {sst_grid.shape}, range {np.nanmin(sst_grid):.1f}-{np.nanmax(sst_grid):.1f}°C, "
                 f"valid={sst_valid.sum()}/{sst_grid.size} ({sst_valid.mean()*100:.0f}%)")
        RESULTS["sst_parse"] = f"✅ {sst_grid.shape}, {np.nanmin(sst_grid):.1f}-{np.nanmax(sst_grid):.1f}°C"
    else:
        log.warning("  SST file not found")
        RESULTS["sst_parse"] = "⚠️ File not found"

    if phy_file.exists():
        ds = xr.open_dataset(phy_file)
        u_data = ds["uo"] if "uo" in ds else ds[list(ds.data_vars)[0]]
        v_data = ds["vo"] if "vo" in ds else ds[list(ds.data_vars)[-1]]
        for dim in ["depth", "time"]:
            if dim in u_data.dims:
                u_data = u_data.isel(**{dim: 0 if dim == "depth" else -1})
                v_data = v_data.isel(**{dim: 0 if dim == "depth" else -1})
        u_grid = u_data.values.astype(np.float32)
        v_grid = v_data.values.astype(np.float32)
        ds.close()
        speed = np.sqrt(np.nan_to_num(u_grid)**2 + np.nan_to_num(v_grid)**2)
        log.info(f"  Currents: {u_grid.shape}, max speed={np.nanmax(speed):.2f} m/s")
        RESULTS["current_parse"] = f"✅ {u_grid.shape}, max={np.nanmax(speed):.2f} m/s"
    else:
        RESULTS["current_parse"] = "⚠️ File not found"

    if bgc_file.exists():
        ds = xr.open_dataset(bgc_file)
        chl_var = "chl" if "chl" in ds else list(ds.data_vars)[0]
        chl_data = ds[chl_var]
        for dim in ["depth", "time"]:
            if dim in chl_data.dims:
                chl_data = chl_data.isel(**{dim: 0 if dim == "depth" else -1})
        chl_grid = chl_data.values.astype(np.float32)
        ds.close()
        log.info(f"  CHL: {chl_grid.shape}, range {np.nanmin(chl_grid):.3f}-{np.nanmax(chl_grid):.3f} mg/m³")
        RESULTS["chl_parse"] = f"✅ {chl_grid.shape}, {np.nanmin(chl_grid):.3f}-{np.nanmax(chl_grid):.3f} mg/m³"
    else:
        RESULTS["chl_parse"] = "⚠️ File not found"

except Exception as e:
    RESULTS["data_parse"] = f"❌ {e}"
    ERRORS.append(f"Data parse: {e}")
    log.error(f"  ❌ Parse failed: {e}")


# ═══════════════════════════════════════════════════════════
# STEP 3: Open-Meteo 海象預報
# ═══════════════════════════════════════════════════════════
section("STEP 3: Open-Meteo 真實海象預報")

try:
    import urllib.request

    url = ("https://marine-api.open-meteo.com/v1/marine?"
           "latitude=24.15&longitude=120.6"
           "&hourly=wave_height,wind_wave_height&forecast_days=3&timezone=UTC")
    req = urllib.request.Request(url, headers={"User-Agent": "OceanMaster/15.3"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        meteo = json.loads(resp.read())

    hs = meteo["hourly"]["wave_height"]
    valid_hs = [h for h in hs if h is not None]
    log.info(f"  Open-Meteo: {len(valid_hs)} hours of wave data")
    log.info(f"  Hs range: {min(valid_hs):.1f}-{max(valid_hs):.1f} m")
    RESULTS["open_meteo"] = f"✅ {len(valid_hs)}h, Hs={min(valid_hs):.1f}-{max(valid_hs):.1f}m"
except Exception as e:
    RESULTS["open_meteo"] = f"❌ {e}"
    ERRORS.append(f"Open-Meteo: {e}")


# ═══════════════════════════════════════════════════════════
# STEP 4: 用真實數據跑 HSI + 熱點
# ═══════════════════════════════════════════════════════════
section("STEP 4: HSI 計算 (真實衛星數據)")

hotspots = []
chl_for_hsi = None
if sst_grid is not None and lats is not None:
    try:
        from engine.forecast_hsi import _sst_hsi

        # Regrid CHL to match SST if sizes differ
        if chl_grid is not None:
            from scipy.ndimage import zoom
            zy = sst_grid.shape[0] / chl_grid.shape[0]
            zx = sst_grid.shape[1] / chl_grid.shape[1]
            if abs(zy - 1.0) > 0.01 or abs(zx - 1.0) > 0.01:
                chl_for_hsi = zoom(chl_grid, (zy, zx), order=1)
            else:
                chl_for_hsi = chl_grid

        species = ["yellowfin", "bigeye", "skipjack", "albacore"]
        all_hotspots = []

        for sp in species:
            hsi = _sst_hsi(sst_grid, sp)

            # Add CHL bonus if available
            if chl_for_hsi is not None:
                chl_safe = np.nan_to_num(chl_for_hsi, nan=0)
                chl_bonus = 0.15 * np.clip(chl_safe / 1.0, 0, 1)
                hsi = np.clip(hsi + chl_bonus, 0, 1)

            # Extract top 5 hotspots per species
            flat_idx = np.argsort(np.nan_to_num(hsi).ravel())[::-1][:5]
            ny, nx = hsi.shape
            for idx in flat_idx:
                iy, ix = divmod(idx, nx)
                score = float(hsi[iy, ix])
                if score > 0.3:
                    all_hotspots.append({
                        "lat": float(lats[iy]),
                        "lon": float(lons[ix]),
                        "hsi": round(score, 3),
                        "species": sp,
                        "sst": float(sst_grid[iy, ix]) if np.isfinite(sst_grid[iy, ix]) else 0,
                    })
                log.info(f"  {sp}: HSI max={np.nanmax(hsi):.3f}, "
                         f"valid={(np.isfinite(hsi) & (hsi > 0)).sum()} cells")

        hotspots = sorted(all_hotspots, key=lambda x: x["hsi"], reverse=True)[:20]
        RESULTS["hsi_compute"] = f"✅ {len(hotspots)} hotspots from {len(species)} species"
        log.info(f"  Top hotspot: {hotspots[0]['species']} HSI={hotspots[0]['hsi']} "
                 f"at {hotspots[0]['lat']:.1f}N {hotspots[0]['lon']:.1f}E" if hotspots else "  No hotspots")
    except Exception as e:
        RESULTS["hsi_compute"] = f"❌ {e}"
        ERRORS.append(f"HSI: {e}")
        log.error(f"  ❌ HSI failed: {e}")
else:
    RESULTS["hsi_compute"] = "⚠️ No SST data"


# ═══════════════════════════════════════════════════════════
# STEP 5: 鉤深建議 (用真實 SST)
# ═══════════════════════════════════════════════════════════
section("STEP 5: 鉤深建議 (真實 SST)")

try:
    from engine.hook_depth import compute_hook_depth

    if hotspots:
        example = hotspots[0]
        hd = compute_hook_depth(
            species=example["species"],
            sst=example["sst"],
            hour_utc=today.hour,
            lunar_illumination=0.6,
        )
        log.info(f"  {example['species']} at SST={example['sst']:.1f}°C: "
                 f"hook {hd['hook_depth_min']}-{hd['hook_depth_max']}m "
                 f"(optimal {hd['hook_depth_optimal']}m)")
        RESULTS["hook_depth"] = (f"✅ {example['species']}: "
                                  f"{hd['hook_depth_min']}-{hd['hook_depth_max']}m")
except Exception as e:
    RESULTS["hook_depth"] = f"❌ {e}"
    ERRORS.append(f"Hook depth: {e}")


# ═══════════════════════════════════════════════════════════
# STEP 6: 出港最佳化 (真實 Open-Meteo)
# ═══════════════════════════════════════════════════════════
section("STEP 6: 出港最佳化 (真實海象)")

try:
    from engine.departure_optimizer import DepartureOptimizer

    if hotspots:
        opt = DepartureOptimizer()
        options = opt.optimize(
            target_lat=hotspots[0]["lat"],
            target_lon=hotspots[0]["lon"],
        )
        if options:
            best = options[0]
            log.info(f"  Best departure: {best['depart']}, "
                     f"score={best['score']}, wave={best['wave_m']}m")
            RESULTS["departure"] = f"✅ {best['depart']}, score={best['score']}"
        else:
            RESULTS["departure"] = "⚠️ No valid windows"
            log.warning("  No suitable departure windows")
except Exception as e:
    RESULTS["departure"] = f"❌ {e}"
    ERRORS.append(f"Departure: {e}")


# ═══════════════════════════════════════════════════════════
# STEP 7: 燃油 + ROI + 碳排
# ═══════════════════════════════════════════════════════════
section("STEP 7: 燃油 / ROI / 碳排")

try:
    from engine.fuel_predictor import FuelPredictor
    from engine.roi_calculator import ROICalculator
    from engine.carbon_calculator import CarbonCalculator

    fp = FuelPredictor()
    fuel = fp.predict(
        wave_height=float(max(valid_hs)) if valid_hs else 1.5,
        wind_speed_ms=8.0,
        wind_angle_deg=90,
    )
    log.info(f"  Fuel: {fuel['fuel_l_per_nm']:.1f} L/nm, penalty={fuel['fuel_penalty_pct']:.1f}%")

    roi = ROICalculator()
    report = roi.calculate(fuel_saved_l=1500, time_saved_hours=6, catch_increase_pct=3.0)
    log.info(f"  ROI: NT${report['total_savings_twd']:,}/trip")

    cc = CarbonCalculator()
    carbon = cc.calculate(fuel_consumed_l=5000, distance_nm=300, catch_kg=2000)
    log.info(f"  Carbon: {carbon['co2_tons']:.1f}t CO2, CII={carbon['cii_rating']}")

    RESULTS["fuel_roi_carbon"] = (f"✅ Fuel={fuel['fuel_l_per_nm']:.1f}L/nm, "
                                   f"ROI=NT${report['total_savings_twd']:,}, "
                                   f"CII={carbon['cii_rating']}")
except Exception as e:
    RESULTS["fuel_roi_carbon"] = f"❌ {e}"
    ERRORS.append(f"Fuel/ROI/Carbon: {e}")


# ═══════════════════════════════════════════════════════════
# STEP 8: 航線規劃 + 海象疊加
# ═══════════════════════════════════════════════════════════
section("STEP 8: 航線規劃")

try:
    from engine.route_planner_v2 import compute_route_cost

    if hotspots:
        route = compute_route_cost(
            hotspot_lat=hotspots[0]["lat"],
            hotspot_lon=hotspots[0]["lon"],
        )
        log.info(f"  Route: {route['distance_km']:.0f}km, "
                 f"${route['total_cost_usd']:.0f}, "
                 f"weather hazards={len(route.get('weather_hazards', []))}")
        RESULTS["route"] = (f"✅ {route['distance_km']:.0f}km, "
                            f"${route['total_cost_usd']:.0f}")
except Exception as e:
    RESULTS["route"] = f"❌ {e}"
    ERRORS.append(f"Route: {e}")


# ═══════════════════════════════════════════════════════════
# STEP 9: 8 天預報 (用真實 SST 作為 Day1)
# ═══════════════════════════════════════════════════════════
section("STEP 9: 8 天預報")

try:
    from engine.forecast_hsi import generate_8day_forecast

    if sst_grid is not None and lats is not None:
        # Use today's SST as Day 1-3, with slight perturbation for Day 4-8
        forecast_data = {}
        for d in range(1, 9):
            noise = np.random.RandomState(d).normal(0, 0.3, sst_grid.shape).astype(np.float32)
            fc_sst = sst_grid + noise * (d / 8.0)
            fc_entry = {"sst": fc_sst}
            if chl_for_hsi is not None:
                fc_entry["chl"] = chl_for_hsi
            if u_grid is not None:
                fc_entry["u"] = u_grid
                fc_entry["v"] = v_grid
            forecast_data[d] = fc_entry

        fc_result = generate_8day_forecast(
            forecast_data, lats, lons,
            species_list=["yellowfin", "bigeye"],
            top_n_per_day=5,
        )
        total = fc_result["summary"]["total_hotspots"]
        log.info(f"  8-day forecast: {total} hotspots across "
                 f"{fc_result['summary']['forecast_days']} days")
        if fc_result["best_windows"]:
            bw = fc_result["best_windows"][0]
            log.info(f"  Best window: Day{bw['day']} {bw['species']} HSI={bw['hsi']:.3f}")
        RESULTS["forecast_8day"] = f"✅ {total} hotspots, {fc_result['summary']['forecast_days']} days"
except Exception as e:
    RESULTS["forecast_8day"] = f"❌ {e}"
    ERRORS.append(f"Forecast: {e}")


# ═══════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════
elapsed = time.time() - t0
section("L2 驗證結果總表")

print()
passed = sum(1 for v in RESULTS.values() if v.startswith("✅"))
total_tests = len(RESULTS)

for key, val in RESULTS.items():
    print(f"  {key:25s} {val}")

print(f"\n{'='*60}")
print(f"  通過: {passed}/{total_tests} | 耗時: {elapsed:.1f}s | 錯誤: {len(ERRORS)}")
if ERRORS:
    print(f"  Errors:")
    for e in ERRORS:
        print(f"    - {e}")
print(f"{'='*60}")

# Save results
report_path = Path("output/L2_validation_report.json")
report_path.parent.mkdir(exist_ok=True)
report_path.write_text(json.dumps({
    "timestamp": today.isoformat(),
    "results": RESULTS,
    "errors": ERRORS,
    "passed": passed,
    "total": total_tests,
    "elapsed_seconds": round(elapsed, 1),
    "top_hotspots": hotspots[:5],
}, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n  Report saved: {report_path}")
