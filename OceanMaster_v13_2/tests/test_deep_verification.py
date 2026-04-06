"""
OceanMaster 深層多層驗證 (Deep Multi-Layer Verification)
========================================================
Layer 1: Import + signature verification
Layer 2: Edge cases (NaN, zero, extreme values)
Layer 3: Physics sanity (direction, monotonicity, bounds)
Layer 4: Cross-module integration (data flows between modules)
Layer 5: Real CMEMS data pipeline end-to-end
Layer 6: Determinism (same input = same output)
Layer 7: Backward compatibility (old callers don't break)
"""
import sys, time, traceback
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')

t0 = time.time()
PASS = 0
FAIL = 0
ERRORS = []

def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        ERRORS.append(f"{name}: {detail}")
        print(f"  [FAIL] {name} -- {detail}")

def heading(title):
    print(f"\n{'='*60}")
    print(f"  LAYER: {title}")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════
heading("1. IMPORT + SIGNATURE")
# ═══════════════════════════════════════════════════════════

try:
    from engine.fuel_predictor import FuelPredictor
    from engine.hook_depth import compute_hook_depth, enrich_hotspots_with_hook_depth
    from engine.departure_optimizer import DepartureOptimizer
    from engine.carbon_calculator import CarbonCalculator, CIIAnnualTracker
    from engine.roi_calculator import ROICalculator
    from engine.competition_penalty import CompetitionPenalty
    from engine.forecast_hsi import generate_8day_forecast
    from engine.route_planner_v2 import compute_route_cost, _check_route_weather
    from engine.lagrangian_advection import BioAdvectionEngine
    from engine.ai_fusion import fuse_and_rank
    from engine.algorithms import compute_ftle, calculate_eke
    from engine.okubo_weiss import OkuboWeissAnalyzer
    from engine.eddy_detector import EddyDetector
    test("All 13 modules import", True)
except Exception as e:
    test("All 13 modules import", False, str(e))

# Check key signatures have expected params
import inspect
sig = inspect.signature(compute_hook_depth)
test("hook_depth has lunar_illumination param", "lunar_illumination" in sig.parameters)
test("hook_depth lunar default=0.5", sig.parameters["lunar_illumination"].default == 0.5)

sig2 = inspect.signature(FuelPredictor.predict)
test("fuel_predictor has wind_angle_deg param", "wind_angle_deg" in sig2.parameters)

test("CIIAnnualTracker has add_voyage", hasattr(CIIAnnualTracker, "add_voyage"))
test("BioAdvectionEngine has predict_lcs_drift", hasattr(BioAdvectionEngine, "predict_lcs_drift"))


# ═══════════════════════════════════════════════════════════
heading("2. EDGE CASES (NaN, zero, extreme)")
# ═══════════════════════════════════════════════════════════

fp = FuelPredictor()

# Zero everything
r = fp.predict(wave_height=0, wave_period=0, wind_speed_ms=0, wind_angle_deg=0)
test("Fuel: all-zero no crash", r["fuel_penalty_pct"] == 0.0)

# Extreme wave
r = fp.predict(wave_height=15, wave_period=3, wind_speed_ms=30, wind_angle_deg=180)
test("Fuel: extreme values no crash", r["fuel_l_per_nm"] > 0)
test("Fuel: extreme penalty > 100%", r["fuel_penalty_pct"] > 100, f"got {r['fuel_penalty_pct']}")

# Negative wind speed (should not crash)
r = fp.predict(wave_height=1, wind_speed_ms=-5)
test("Fuel: negative wind no crash", r["fuel_l_per_nm"] > 0)

# Hook depth edge cases
hd = compute_hook_depth("bigeye", sst=0, mld=0, z20=0)
test("HookDepth: SST=0 no crash", hd["hook_depth_optimal"] >= 0)

hd = compute_hook_depth("bigeye", sst=35, mld=200, z20=500, lunar_illumination=1.0, hour_utc=20)
test("HookDepth: extreme warm deep no crash", hd["hook_depth_optimal"] > 0)

hd = compute_hook_depth("unknown_species", sst=25)
test("HookDepth: unknown species no crash", hd["hook_depth_optimal"] > 0)

# CII tracker - empty
t = CIIAnnualTracker(500)
s = t.get_status()
test("CII: empty tracker no crash", s["pct_used"] == 0.0)

# CII tracker - negative quota
t2 = CIIAnnualTracker(0)
t2.add_voyage(100)
s2 = t2.get_status()
test("CII: zero quota no crash/div-by-zero", True)

# ROI: negative savings
roi = ROICalculator()
r = roi.calculate(fuel_saved_l=-1000, time_saved_hours=-5)
test("ROI: negative savings no crash", True)

# CompetitionPenalty edge
cp = CompetitionPenalty()
f = cp.compute_fatigue_penalty(0, 0)
test("Fatigue: zero vessels zero days = 0", f == 0.0)
f = cp.compute_fatigue_penalty(100, 30)
test("Fatigue: extreme values capped", f <= 0.30, f"got {f}")

# Okubo-Weiss with constant field (no gradient)
ow = OkuboWeissAnalyzer()
uniform_u = np.ones((50, 50), dtype=np.float32) * 0.5
uniform_v = np.ones((50, 50), dtype=np.float32) * 0.3
lats_t = np.linspace(20, 25, 50)
lons_t = np.linspace(120, 125, 50)
r = ow.compute(uniform_u, uniform_v, lats_t, lons_t)
test("OW: uniform field = no strain/rotation", r["strain_pct"] == 0, f"strain={r['strain_pct']}")


# ═══════════════════════════════════════════════════════════
heading("3. PHYSICS SANITY")
# ═══════════════════════════════════════════════════════════

# Wind direction monotonicity: tailwind < beam < headwind
fuel_0 = fp.predict(wave_height=0, wind_speed_ms=15, wind_angle_deg=0)["fuel_l_per_nm"]
fuel_90 = fp.predict(wave_height=0, wind_speed_ms=15, wind_angle_deg=90)["fuel_l_per_nm"]
fuel_180 = fp.predict(wave_height=0, wind_speed_ms=15, wind_angle_deg=180)["fuel_l_per_nm"]
test("Fuel: tailwind < beam < headwind",
     fuel_0 < fuel_90 < fuel_180,
     f"tail={fuel_0:.2f} beam={fuel_90:.2f} head={fuel_180:.2f}")

# Wave height monotonicity
fuel_w0 = fp.predict(wave_height=0)["fuel_l_per_nm"]
fuel_w2 = fp.predict(wave_height=2)["fuel_l_per_nm"]
fuel_w5 = fp.predict(wave_height=5)["fuel_l_per_nm"]
test("Fuel: more waves = more fuel",
     fuel_w0 <= fuel_w2 <= fuel_w5,
     f"w0={fuel_w0:.2f} w2={fuel_w2:.2f} w5={fuel_w5:.2f}")

# Calm sea = base consumption
test("Fuel: calm = base (45 L/nm)", fuel_w0 == 45.0, f"got {fuel_w0}")

# Hook depth: night full moon > night new moon
full_night = compute_hook_depth("bigeye", sst=26, hour_utc=14, lunar_illumination=1.0)
new_night = compute_hook_depth("bigeye", sst=26, hour_utc=14, lunar_illumination=0.0)
test("HookDepth: full moon night deeper",
     full_night["hook_depth_optimal"] > new_night["hook_depth_optimal"],
     f"full={full_night['hook_depth_optimal']} new={new_night['hook_depth_optimal']}")

# Hook depth: day ignores lunar
full_day = compute_hook_depth("bigeye", sst=26, hour_utc=4, lunar_illumination=1.0)
new_day = compute_hook_depth("bigeye", sst=26, hour_utc=4, lunar_illumination=0.0)
test("HookDepth: day ignores lunar",
     full_day["hook_depth_optimal"] == new_day["hook_depth_optimal"],
     f"full_day={full_day['hook_depth_optimal']} new_day={new_day['hook_depth_optimal']}")

# Hook depth: bigeye deeper than skipjack
be = compute_hook_depth("bigeye", sst=26)
sk = compute_hook_depth("skipjack", sst=26)
test("HookDepth: bigeye deeper than skipjack",
     be["hook_depth_optimal"] > sk["hook_depth_optimal"],
     f"bigeye={be['hook_depth_optimal']} skip={sk['hook_depth_optimal']}")

# Carbon: more fuel = more CO2
cc = CarbonCalculator()
c1 = cc.calculate(fuel_consumed_l=1000, distance_nm=100)
c2 = cc.calculate(fuel_consumed_l=5000, distance_nm=100)
test("Carbon: more fuel = more CO2", c2["co2_kg"] > c1["co2_kg"])
test("Carbon: CO2 = fuel * 3.206", abs(c1["co2_kg"] - 1000 * 3.206) < 0.1)

# CII: more voyages = higher pct_used
t3 = CIIAnnualTracker(500)
t3.add_voyage(50)
s1 = t3.get_status()["pct_used"]
t3.add_voyage(50)
s2 = t3.get_status()["pct_used"]
test("CII: more voyages = higher pct", s2 > s1)

# Fatigue: more days = more penalty
f3 = cp.compute_fatigue_penalty(10, 3)
f5 = cp.compute_fatigue_penalty(10, 5)
f7 = cp.compute_fatigue_penalty(10, 7)
test("Fatigue: monotonic with days", f3 <= f5 <= f7, f"3d={f3} 5d={f5} 7d={f7}")


# ═══════════════════════════════════════════════════════════
heading("4. CROSS-MODULE INTEGRATION")
# ═══════════════════════════════════════════════════════════

# Hook depth → ROI (depth info feeds into operational decisions)
hd = compute_hook_depth("yellowfin", sst=27, hour_utc=8)
test("HookDepth: returns all required keys",
     all(k in hd for k in ["hook_depth_min", "hook_depth_max", "hook_depth_optimal",
                            "best_time", "method", "confidence"]))

# Fuel → Carbon (fuel output feeds into carbon calculator)
fuel_result = fp.predict(wave_height=2, wind_speed_ms=10, wind_angle_deg=135)
distance = 500  # nm
total_fuel = fuel_result["fuel_l_per_nm"] * distance
carbon = cc.calculate(fuel_consumed_l=total_fuel, distance_nm=distance)
test("Fuel→Carbon integration", carbon["co2_kg"] > 0 and carbon["cii_rating"] in "ABCDE")

# Carbon → CII tracker
tracker = CIIAnnualTracker(500)
status = tracker.add_voyage(carbon["co2_tons"], distance)
test("Carbon→CII integration", status["pct_used"] > 0)

# enrich_hotspots passes lunar to compute_hook_depth
mock_hotspots = [{
    "lat": 25.0, "lon": 130.0, "species": "bigeye",
    "sst": 26.0, "lunar_illumination": 0.8,
}]
enriched = enrich_hotspots_with_hook_depth(mock_hotspots, month=2)
test("enrich_hotspots: returns hook_depth_label",
     "hook_depth_label" in enriched[0],
     f"keys={list(enriched[0].keys())}")

# Okubo-Weiss → Eddy Detector (same u/v → consistent results)
ow_result = OkuboWeissAnalyzer().compute(
    np.random.RandomState(42).randn(50, 50).astype(np.float32) * 0.1,
    np.random.RandomState(43).randn(50, 50).astype(np.float32) * 0.1,
    np.linspace(20, 25, 50), np.linspace(120, 125, 50)
)
ed_result = EddyDetector().classify_okubo_weiss(
    np.random.RandomState(42).randn(50, 50).astype(np.float32) * 0.1,
    np.random.RandomState(43).randn(50, 50).astype(np.float32) * 0.1,
    np.linspace(20, 25, 50), np.linspace(120, 125, 50)
)
test("OW vs EddyDetector: both produce strain/vorticity",
     ow_result["strain_pct"] > 0 and ed_result["pct_strain"] > 0)


# ═══════════════════════════════════════════════════════════
heading("5. REAL CMEMS DATA PIPELINE")
# ═══════════════════════════════════════════════════════════

try:
    import xarray as xr
    CACHE = "C:/tmp/L2_cache"
    ds_sst = xr.open_dataset(f"{CACHE}/cmems_sst.nc")
    ds_phy = xr.open_dataset(f"{CACHE}/cmems_phy.nc")
    ds_bgc = xr.open_dataset(f"{CACHE}/cmems_bgc.nc")

    real_sst = ds_sst["thetao"].isel(depth=0, time=-1).values.astype(np.float32)
    real_u = np.nan_to_num(ds_phy["uo"].isel(depth=0, time=-1).values.astype(np.float32))
    real_v = np.nan_to_num(ds_phy["vo"].isel(depth=0, time=-1).values.astype(np.float32))
    real_chl = ds_bgc["chl"].isel(depth=0, time=-1).values.astype(np.float32)
    real_lats = ds_sst.latitude.values
    real_lons = ds_sst.longitude.values

    test("Real data: SST shape valid", real_sst.shape[0] > 10 and real_sst.shape[1] > 10)
    test("Real data: SST range 5-35C",
         5 < np.nanmin(real_sst) < 15 and 25 < np.nanmax(real_sst) < 35,
         f"range={np.nanmin(real_sst):.1f}-{np.nanmax(real_sst):.1f}")
    test("Real data: CHL > 0", np.nanmin(real_chl) > 0, f"min={np.nanmin(real_chl):.4f}")

    # FTLE with real data
    ftle = compute_ftle(real_u[::4, ::4], real_v[::4, ::4],
                        real_lats[::4], real_lons[::4], integration_days=2)
    ftle_grid = ftle if isinstance(ftle, np.ndarray) else ftle.get("ftle", ftle.get("ftle_field"))
    test("Real FTLE: computed", ftle_grid is not None and ftle_grid.shape[0] > 5)

    # OW with real data
    ow_real = OkuboWeissAnalyzer().compute(real_u, real_v, real_lats, real_lons)
    test("Real OW: has structure", ow_real["strain_pct"] > 0 or ow_real["rotation_pct"] > 0)

    # Eddy bio with real data
    from scipy.ndimage import zoom
    chl_matched = zoom(real_chl, (real_u.shape[0]/real_chl.shape[0],
                                   real_u.shape[1]/real_chl.shape[1]), order=1)
    ow_class = ow_real["ow_classification"]
    eddy_core = np.abs(ow_class).astype(np.float32)
    eddy_type = ow_class.astype(np.float32)
    from scipy.ndimage import binary_dilation
    core_mask = (np.abs(ow_class) > 0).astype(bool)
    eddy_edge = (binary_dilation(core_mask, iterations=2) & ~core_mask).astype(np.float32)
    bio = EddyDetector.compute_eddy_biological_enrichment(
        eddy_core, eddy_type, chl_matched, real_sst, eddy_edge, "bigeye"
    )
    test("Real EddyBio: computed", bio.shape == real_sst.shape)
    test("Real EddyBio: range 0-1", 0 <= np.nanmin(bio) and np.nanmax(bio) <= 1.0)

    ds_sst.close(); ds_phy.close(); ds_bgc.close()

except FileNotFoundError:
    test("Real data: CMEMS files exist", False, "Files not found in C:/tmp/L2_cache")
except Exception as e:
    test("Real data pipeline", False, str(e))


# ═══════════════════════════════════════════════════════════
heading("6. DETERMINISM")
# ═══════════════════════════════════════════════════════════

# Same input → same output
r1 = fp.predict(wave_height=2, wind_speed_ms=10, wind_angle_deg=135)
r2 = fp.predict(wave_height=2, wind_speed_ms=10, wind_angle_deg=135)
test("Fuel: deterministic", r1 == r2)

hd1 = compute_hook_depth("bigeye", sst=26, hour_utc=14, lunar_illumination=0.7)
hd2 = compute_hook_depth("bigeye", sst=26, hour_utc=14, lunar_illumination=0.7)
test("HookDepth: deterministic", hd1 == hd2)

c1 = cc.calculate(fuel_consumed_l=3000, distance_nm=200)
c2 = cc.calculate(fuel_consumed_l=3000, distance_nm=200)
test("Carbon: deterministic", c1 == c2)


# ═══════════════════════════════════════════════════════════
heading("7. BACKWARD COMPATIBILITY")
# ═══════════════════════════════════════════════════════════

# Old callers that don't pass lunar_illumination should still work
hd_old = compute_hook_depth("yellowfin", sst=28)
test("HookDepth: old call (no lunar) works", hd_old["hook_depth_optimal"] > 0)

# Old callers that don't pass wind_angle_deg
r_old = fp.predict(wave_height=2, wind_speed_ms=10)
test("Fuel: old call (no wind_angle) works", r_old["fuel_l_per_nm"] > 0)

# enrich_hotspots without lunar_illumination in hotspot dict
old_hotspots = [{"lat": 25.0, "lon": 130.0, "species": "yellowfin", "sst": 27.0}]
enriched_old = enrich_hotspots_with_hook_depth(old_hotspots, month=6)
test("enrich: old format without lunar works", "hook_depth_label" in enriched_old[0])

# CarbonCalculator without catch_kg
c_no_catch = cc.calculate(fuel_consumed_l=2000, distance_nm=150)
test("Carbon: without catch_kg works", c_no_catch["co2_per_kg_catch"] == 0.0)

# ROI with no args
r_zero = roi.calculate()
test("ROI: empty call = zero savings", r_zero["total_savings_twd"] == 0)


# ═══════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════
elapsed = time.time() - t0
print(f"\n{'='*60}")
print(f"  DEEP MULTI-LAYER VERIFICATION COMPLETE")
print(f"{'='*60}")
print(f"  PASSED: {PASS}")
print(f"  FAILED: {FAIL}")
print(f"  TOTAL:  {PASS + FAIL}")
print(f"  TIME:   {elapsed:.1f}s")
if ERRORS:
    print(f"\n  FAILURES:")
    for e in ERRORS:
        print(f"    - {e}")
else:
    print(f"\n  *** ALL {PASS} TESTS PASSED ***")
print(f"{'='*60}")
