"""OceanMaster v10.3 — Import Verification"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("  OceanMaster v10.3 — Module Import Verification")
print("=" * 60)

modules = [
    ("engine.base_fetcher", ["BaseFetcher", "DataSource", "CascadeResult"]),
    ("engine.weather_fetcher", ["WeatherFetcher"]),
    ("engine.typhoon_tracker", ["TyphoonTracker", "TyphoonAlert"]),
    ("engine.safety_checker", ["is_safe_for_fishing", "SafetyLevel",
                               "assess_grid_safety", "filter_hotspots_by_safety"]),
    ("engine.wave_fetcher", ["WaveFetcher"]),
    ("engine.pressure_features", ["compute_pressure_features"]),
    ("engine.tchp_calculator", ["compute_tchp", "tchp_alert_message"]),
    ("engine.typhoon_ml_features", ["TyphoonFeatureEngineer"]),
    ("engine.rainfall_fetcher", ["RainfallFetcher"]),
    ("engine.mld_forecast", ["MLDForecaster"]),
]

passed = 0
failed = 0
for mod_name, symbols in modules:
    try:
        mod = __import__(mod_name, fromlist=symbols)
        for s in symbols:
            assert hasattr(mod, s), f"{s} not found in {mod_name}"
        print(f"  PASS  {mod_name} ({len(symbols)} symbols)")
        passed += 1
    except Exception as e:
        print(f"  FAIL  {mod_name}: {e}")
        failed += 1

print()

# Quick functional test
print("--- Functional Tests ---")
try:
    import numpy as np
    from engine.pressure_features import compute_pressure_features

    p = np.random.uniform(1005, 1020, (5, 5)).astype(np.float32)
    lats = np.linspace(20, 25, 5)
    lons = np.linspace(130, 135, 5)
    result = compute_pressure_features(p, lats, lons)
    assert "pressure_gradient" in result
    assert "low_pressure_bonus" in result
    print("  PASS  pressure_features functional test")
except Exception as e:
    print(f"  FAIL  pressure_features: {e}")
    failed += 1

try:
    from engine.tchp_calculator import compute_tchp
    depths = np.array([0, 10, 20, 50, 100, 200, 300])
    temp = np.zeros((7, 3, 3), dtype=np.float32)
    temp[0] = 29.0; temp[1] = 28.5; temp[2] = 27.5
    temp[3] = 25.0; temp[4] = 18.0; temp[5] = 10.0; temp[6] = 5.0
    result = compute_tchp(temp, depths)
    assert "tchp" in result
    assert "d26" in result
    print(f"  PASS  tchp_calculator: TCHP={np.nanmean(result['tchp']):.1f} kJ/cm2, D26={np.nanmean(result['d26']):.0f}m")
except Exception as e:
    print(f"  FAIL  tchp_calculator: {e}")
    failed += 1

try:
    from engine.safety_checker import is_safe_for_fishing, SafetyLevel
    result = is_safe_for_fishing(25.0, 140.0, wave_height=3.5, wind_speed=20.0)
    # SafetyResult is a namedtuple/dataclass, access .level
    level_str = str(result.level) if hasattr(result, 'level') else str(result)
    print(f"  PASS  safety_checker: level={level_str}, reason={result.reason}")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"  FAIL  safety_checker: {e}")
    failed += 1

print()
print(f"=== Results: {passed}/{len(modules)} imports + functional tests ===")
if failed:
    print(f"  {failed} failures")
    sys.exit(1)
else:
    print("  ALL PASSED!")
