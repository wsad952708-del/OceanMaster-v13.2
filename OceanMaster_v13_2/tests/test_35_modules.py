"""Fixed 35-module test with correct class/function names"""
import sys, time
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
t0 = time.time(); PASS = 0; FAIL = 0; ERRORS = []

def test(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  [PASS] {name}")
    else: FAIL += 1; ERRORS.append(f"{name}: {detail}"); print(f"  [FAIL] {name} -- {detail}")

def H(t): print(f"\n{'='*55}\n  {t}\n{'='*55}")

# Load CMEMS
import xarray as xr
from scipy.ndimage import zoom
ds_s = xr.open_dataset("C:/tmp/L2_cache/cmems_sst.nc")
ds_p = xr.open_dataset("C:/tmp/L2_cache/cmems_phy.nc")
ds_b = xr.open_dataset("C:/tmp/L2_cache/cmems_bgc.nc")
sst = ds_s["thetao"].isel(depth=0, time=-1).values.astype(np.float32)
u = np.nan_to_num(ds_p["uo"].isel(depth=0, time=-1).values.astype(np.float32))
v = np.nan_to_num(ds_p["vo"].isel(depth=0, time=-1).values.astype(np.float32))
chl = ds_b["chl"].isel(depth=0, time=-1).values.astype(np.float32)
lats = ds_s.latitude.values; lons = ds_s.longitude.values
chl_m = zoom(chl, (sst.shape[0]/chl.shape[0], sst.shape[1]/chl.shape[1]), order=1)

H("A. 物理演算法 (8)")
try:
    from engine.eddy_maturity import EddyMaturityEstimator
    r = EddyMaturityEstimator().compute(chl=chl_m, sst=sst, eddy_age_days=10)
    test("eddy_maturity", r is not None)
except Exception as e: test("eddy_maturity", False, str(e)[:80])

try:
    from engine.convergence_detector import ConvergenceDetector
    r = ConvergenceDetector().compute(u, v, lats, lons)
    test("convergence_detector", r is not None)
except Exception as e: test("convergence_detector", False, str(e)[:80])

try:
    from engine.ocean_color_fronts import OceanColorFrontDetector
    r = OceanColorFrontDetector().detect(chl_m, lats, lons)
    test("ocean_color_fronts", r is not None)
except Exception as e: test("ocean_color_fronts", False, str(e)[:80])

try:
    from engine.sst_change_detector import SSTChangeDetector
    scd = SSTChangeDetector()
    sst2 = sst + np.random.randn(*sst.shape).astype(np.float32) * 0.5
    r = scd.compute(sst, sst2, lats, lons)
    test("sst_change_detector", r is not None)
except Exception as e: test("sst_change_detector", False, str(e)[:80])

try:
    from engine.habitat_compression import HabitatCompressionAnalyzer
    do = np.ones_like(sst) * 5.0
    r = HabitatCompressionAnalyzer().compute(sst=sst, dissolved_oxygen=do, species="bigeye")
    test("habitat_compression", r is not None)
except Exception as e: test("habitat_compression", False, str(e)[:80])

try:
    from engine.ocean_diagnostics import SpicinessAnalyzer, StratificationAnalyzer
    test("ocean_diagnostics: import", True)
except Exception as e: test("ocean_diagnostics", False, str(e)[:80])

try:
    from engine.marine_heatwave import MarineHeatwaveDetector
    clim = np.ones_like(sst) * 24.0
    r = MarineHeatwaveDetector().detect(sst, clim)
    test("marine_heatwave", r is not None)
except Exception as e: test("marine_heatwave", False, str(e)[:80])

try:
    from engine.water_mass_classifier import WaterMassClassifier
    sal = np.ones_like(sst) * 34.5
    r = WaterMassClassifier().classify(sst, sal)
    test("water_mass_classifier", r is not None)
except Exception as e: test("water_mass_classifier", False, str(e)[:80])

H("B. 生物模型 (6)")
try:
    from engine.fish_behavior_model import FishBehaviorModel
    fbm = FishBehaviorModel()
    test("fish_behavior: import", True)
    test("fish_behavior: has methods", hasattr(fbm, 'estimate_feeding_windows'))
except Exception as e: test("fish_behavior", False, str(e)[:80])

try:
    from engine.dvm_model import DVMModel
    dvm = DVMModel()
    r = dvm.compute_accessibility(sst=26, mld=80, z20=200, species="bigeye", hour=14)
    test("dvm_model", r is not None)
except Exception as e: test("dvm_model", False, str(e)[:80])

try:
    from engine.food_chain_predictor import FoodChainPredictor
    test("food_chain: import", True)
except Exception as e: test("food_chain", False, str(e)[:80])

try:
    from engine.zooplankton_proxy import ZooplanktonProxy
    zp = ZooplanktonProxy()
    r = zp.estimate_simple(chl_mg=0.5, sst=26)
    test("zooplankton_proxy", r is not None)
except Exception as e: test("zooplankton_proxy", False, str(e)[:80])

try:
    from engine.forage_engine import ForageEngine
    test("forage_engine: import", True)
except Exception as e: test("forage_engine", False, str(e)[:80])

try:
    from engine.npz_model import NPZModel
    test("npz_model: import", True)
except Exception as e: test("npz_model", False, str(e)[:80])

H("C. AI 融合 (4)")
try:
    from engine.commercial_core_v2 import SEAPODYMHabitatEngine, EddyEdgeDetector
    test("commercial_core_v2: import", True)
except Exception as e: test("commercial_core_v2", False, str(e)[:80])

try:
    from engine.hsi_dynamic_weights import compute_dynamic_weights
    test("hsi_dynamic_weights: import", True)
except Exception as e: test("hsi_dynamic_weights", False, str(e)[:80])

try:
    from engine.hotspot_downscaler import downscale_hotspots
    test("hotspot_downscaler: import", True)
except Exception as e: test("hotspot_downscaler", False, str(e)[:80])

try:
    from engine.greenfish_hsi import GreenFishLiteHSI
    gf = GreenFishLiteHSI()
    r = gf.compute(sst=sst, chl=chl_m, species="yellowfin")
    test("greenfish_hsi", r is not None)
except Exception as e: test("greenfish_hsi", False, str(e)[:80])

H("D. 作業決策 (3)")
try:
    from engine.longline_drift import LonglineDriftPredictor
    ldp = LonglineDriftPredictor()
    r = ldp.predict_drift(lat=24, lon=125, u_current=0.3, v_current=-0.1, duration_hours=8)
    test("longline_drift", r is not None)
except Exception as e: test("longline_drift", False, str(e)[:80])

try:
    from engine.fleet_commander import FleetCommander
    test("fleet_commander: import", True)
except Exception as e: test("fleet_commander", False, str(e)[:80])

try:
    from engine.market_optimizer import MarketOptimizer
    test("market_optimizer: import", True)
except Exception as e: test("market_optimizer", False, str(e)[:80])

H("E. 安全 + 輸出 (5)")
try:
    from engine.safety_checker import is_safe_for_fishing
    test("safety_checker: import", True)
except Exception as e: test("safety_checker", False, str(e)[:80])

try:
    from engine.geojson_output import generate_geojson
    test("geojson_output: import", True)
except Exception as e: test("geojson_output", False, str(e)[:80])

try:
    from engine.text_briefing import generate_sat_briefing
    test("text_briefing: import", True)
except Exception as e: test("text_briefing", False, str(e)[:80])

try:
    from engine.lunar_model import LunarPhaseEngine
    lpe = LunarPhaseEngine()
    r = lpe.compute_moon_phase(2026, 2, 28)
    test("lunar_model", r is not None)
except Exception as e: test("lunar_model", False, str(e)[:80])

try:
    from engine.catch_composition import CatchCompositionForecaster
    ccf = CatchCompositionForecaster()
    r = ccf.forecast(sst=27, lat=22, month=3)
    test("catch_composition", r is not None)
except Exception as e: test("catch_composition", False, str(e)[:80])

H("F. 其他 (9)")
try:
    from engine.tactical_features import lunar_phase_index, compute_eddy_index
    test("tactical_features: import", True)
except Exception as e: test("tactical_features", False, str(e)[:80])

try:
    from engine.migration_corridor import MigrationCorridorPredictor
    test("migration_corridor: import", True)
except Exception as e: test("migration_corridor", False, str(e)[:80])

try:
    from engine.fishery_market import fetch_fishery_market_data
    test("fishery_market: import", True)
except Exception as e: test("fishery_market", False, str(e)[:80])

try:
    from engine.enso_calibrator import ENSOCalibrator
    test("enso_calibrator: import", True)
except Exception as e: test("enso_calibrator", False, str(e)[:80])

try:
    from engine.kuroshio_engine import KuroshioEngine
    test("kuroshio_engine: import", True)
except Exception as e: test("kuroshio_engine", False, str(e)[:80])

try:
    from engine.species_params import get_species, SPECIES
    test("species_params", len(SPECIES) >= 5, f"count={len(SPECIES)}")
except Exception as e: test("species_params", False, str(e)[:80])

try:
    from engine.cpue_estimator import estimate_relative_cpue
    test("cpue_estimator: import", True)
except Exception as e: test("cpue_estimator", False, str(e)[:80])

try:
    from engine.env_cpue_estimator import EnvironmentalCPUEEstimator
    r = EnvironmentalCPUEEstimator().compute(sst=27, chl=0.3, species="yellowfin")
    test("env_cpue_estimator", r is not None)
except Exception as e: test("env_cpue_estimator", False, str(e)[:80])

try:
    from engine.ml.stacking_ensemble import FishingStackingModel
    fsm = FishingStackingModel(species="yellowfin")
    test("stacking_ensemble: import+init", True)
    test("stacking_ensemble: has predict", hasattr(fsm, 'predict'))
except Exception as e: test("stacking_ensemble", False, str(e)[:80])

ds_s.close(); ds_p.close(); ds_b.close()
elapsed = time.time() - t0
print(f"\n{'='*55}")
print(f"  RESULT: {PASS} PASS, {FAIL} FAIL ({elapsed:.1f}s)")
if ERRORS:
    for e in ERRORS: print(f"    - {e}")
else: print(f"  *** ALL {PASS} TESTS PASSED ***")
print(f"{'='*55}")
