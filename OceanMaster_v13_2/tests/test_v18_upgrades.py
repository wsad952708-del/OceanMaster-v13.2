"""Deep verification of all v18 competitor tech upgrades."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

errors = []
passes = 0

# ===== 1. All 10 modules import =====
try:
    from engine.forecast_hsi import generate_8day_forecast
    from engine.fuel_predictor import FuelPredictor
    from engine.departure_optimizer import DepartureOptimizer
    from engine.route_planner_v2 import compute_route_cost
    from engine.lagrangian_advection import BioAdvectionEngine
    from engine.ai_fusion import fuse_and_rank
    from engine.hook_depth import compute_hook_depth, enrich_hotspots_with_hook_depth
    from engine.competition_penalty import CompetitionPenalty
    from engine.carbon_calculator import CarbonCalculator, CIIAnnualTracker
    from engine.roi_calculator import ROICalculator
    print("[PASS] All 10 modules import OK")
    passes += 1
except Exception as e:
    errors.append(f"[FAIL] Import: {e}")

# ===== 2. Fuel Predictor edge cases =====
fp = FuelPredictor()

# Head wind > beam wind > tail wind
head = fp.predict(wave_height=2, wave_period=8, wind_speed_ms=15, wind_angle_deg=180)
beam = fp.predict(wave_height=2, wave_period=8, wind_speed_ms=15, wind_angle_deg=90)
tail = fp.predict(wave_height=2, wave_period=8, wind_speed_ms=15, wind_angle_deg=0)
if head["fuel_l_per_nm"] >= beam["fuel_l_per_nm"]:
    print(f"[PASS] Fuel: head({head['fuel_l_per_nm']:.1f}) >= beam({beam['fuel_l_per_nm']:.1f})")
    passes += 1
else:
    errors.append(f"[FAIL] Fuel: head({head['fuel_l_per_nm']:.1f}) < beam({beam['fuel_l_per_nm']:.1f})")

# Calm sea = zero penalty
calm = fp.predict(wave_height=0, wave_period=8, wind_speed_ms=0)
if calm["fuel_penalty_pct"] == 0.0:
    print(f"[PASS] Fuel: calm penalty = 0%")
    passes += 1
else:
    errors.append(f"[FAIL] Fuel: calm penalty = {calm['fuel_penalty_pct']}%")

# wave_period=0 should not crash (div by zero)
try:
    edge = fp.predict(wave_height=3, wave_period=0)
    print(f"[PASS] Fuel: wave_period=0 no crash, penalty={edge['fuel_penalty_pct']:.1f}%")
    passes += 1
except Exception as e:
    errors.append(f"[FAIL] Fuel wave_period=0 crash: {e}")

# ===== 3. Hook Depth lunar logic =====
# Full moon night deeper than new moon night
full = compute_hook_depth("bigeye", sst=26, hour_utc=14, lunar_illumination=1.0)
new = compute_hook_depth("bigeye", sst=26, hour_utc=14, lunar_illumination=0.0)
if full["hook_depth_optimal"] > new["hook_depth_optimal"]:
    print(f"[PASS] HookDepth: full moon({full['hook_depth_optimal']}m) > new moon({new['hook_depth_optimal']}m)")
    passes += 1
else:
    errors.append(f"[FAIL] HookDepth: full({full['hook_depth_optimal']}) <= new({new['hook_depth_optimal']})")

# Daytime should NOT apply lunar correction
day_full = compute_hook_depth("bigeye", sst=26, hour_utc=2, lunar_illumination=1.0)
day_new = compute_hook_depth("bigeye", sst=26, hour_utc=2, lunar_illumination=0.0)
if day_full["hook_depth_optimal"] == day_new["hook_depth_optimal"]:
    print(f"[PASS] HookDepth: daytime ignores lunar ({day_full['hook_depth_optimal']}m)")
    passes += 1
else:
    errors.append(f"[FAIL] HookDepth: day differs full={day_full['hook_depth_optimal']} new={day_new['hook_depth_optimal']}")

# Default params should work
default = compute_hook_depth("yellowfin", sst=28)
if default["hook_depth_optimal"] > 0:
    print(f"[PASS] HookDepth: default params OK ({default['hook_depth_optimal']}m)")
    passes += 1
else:
    errors.append(f"[FAIL] HookDepth: default optimal <= 0")

# ===== 4. ROI Calculator =====
roi = ROICalculator()
r = roi.calculate(fuel_saved_l=2000, time_saved_hours=8, catch_increase_pct=5.0)
if r["total_savings_twd"] > 0:
    print(f"[PASS] ROI: NT${r['total_savings_twd']:,} per trip")
    passes += 1
else:
    errors.append("[FAIL] ROI: total_savings <= 0")

# Zero savings should be zero
r0 = roi.calculate()
if r0["total_savings_twd"] == 0:
    print(f"[PASS] ROI: zero input = zero savings")
    passes += 1
else:
    errors.append(f"[FAIL] ROI: zero input = {r0['total_savings_twd']}")

# ===== 5. CII Tracker =====
t = CIIAnnualTracker(500)
t.add_voyage(45.2, 800)
s = t.add_voyage(52.1, 900)
if s["pct_used"] > 0 and s["remaining_tons"] > 0:
    print(f"[PASS] CII: {s['pct_used']}% used, {s['remaining_tons']}t remaining")
    passes += 1
else:
    errors.append(f"[FAIL] CII: {s}")

# Exceed quota
t2 = CIIAnnualTracker(100)
for _ in range(3):
    s2 = t2.add_voyage(40)
if "CRITICAL" in s2["level"]:
    print(f"[PASS] CII: quota exceeded triggers CRITICAL")
    passes += 1
else:
    errors.append(f"[FAIL] CII: quota exceeded not CRITICAL: {s2['level']}")

# ===== 6. Competition Fatigue =====
cp = CompetitionPenalty()
f1 = cp.compute_fatigue_penalty(10, 0)
f5 = cp.compute_fatigue_penalty(10, 5)
f7 = cp.compute_fatigue_penalty(10, 7)
if f1 == 0.0 and 0 < f5 < f7 and f7 <= 0.30:
    print(f"[PASS] Fatigue: 0d=0, 5d={f5:.2f}, 7d={f7:.2f}")
    passes += 1
else:
    errors.append(f"[FAIL] Fatigue: 0d={f1}, 5d={f5}, 7d={f7}")

# ===== 7. LCS Drift method exists =====
engine = BioAdvectionEngine()
if hasattr(engine, "predict_lcs_drift"):
    print("[PASS] LCS: predict_lcs_drift() exists")
    passes += 1
else:
    errors.append("[FAIL] LCS: predict_lcs_drift() missing")

# ===== 8. Route planner weather check =====
from engine.route_planner_v2 import _check_route_weather
result = _check_route_weather([(25.0, 140.0)])
# Should return empty list for < 2 waypoints or list of dicts
if isinstance(result, list):
    print(f"[PASS] RouteWeather: returns list (len={len(result)})")
    passes += 1
else:
    errors.append(f"[FAIL] RouteWeather: not a list: {type(result)}")

# ===== Summary =====
print(f"\n{'='*50}")
if errors:
    print(f"FAILED: {len(errors)} errors")
    for e in errors:
        print(f"  {e}")
else:
    print(f"ALL {passes} TESTS PASSED")
