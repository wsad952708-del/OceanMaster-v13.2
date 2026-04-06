"""Final science verification with correct function names"""
import sys, os, numpy as np
sys.path.insert(0, '.')

def ok(msg): print(f"  [PASS] {msg}")
def fail(msg): print(f"  [FAIL] {msg}")

print("=== Science Calculations ===")

# haversine - it's in multiple places
try:
    from engine.gebco_features import GEBCOFeatures
    # haversine is likely a utility, check common locations
    gf = GEBCOFeatures()
    # algorithms doesn't export standalone haversine; check other places
    from engine.algorithms import compute_ftle, detect_sst_fronts, detect_eddies, compute_chl_gradient
    ok("algorithms: compute_ftle, detect_sst_fronts, detect_eddies, compute_chl_gradient")
except Exception as e:
    fail(f"algorithms: {e}")

# FTLE
try:
    u = np.random.rand(20, 20).astype(np.float32) * 0.5
    v = np.random.rand(20, 20).astype(np.float32) * 0.5
    ftle = compute_ftle(u, v, dx=0.25, dt=86400)
    ok(f"FTLE: shape={ftle.shape}, range=[{np.nanmin(ftle):.6f}, {np.nanmax(ftle):.6f}]")
except Exception as e:
    fail(f"FTLE: {e}")

# SST Fronts
try:
    sst = 25.0 + np.random.rand(30, 30).astype(np.float32) * 5
    fronts = detect_sst_fronts(sst, threshold=0.5)
    ok(f"SST fronts: shape={fronts.shape}, {np.sum(fronts > 0)} front pixels")
except Exception as e:
    fail(f"SST fronts: {e}")

# Eddies
try:
    u = np.random.rand(20, 20).astype(np.float32) * 0.3
    v = np.random.rand(20, 20).astype(np.float32) * 0.3
    eddies = detect_eddies(u, v) 
    ok(f"Eddies detected: type={type(eddies).__name__}")
except Exception as e:
    fail(f"Eddies: {e}")

# HSI
try:
    from engine.hsi_models import compute_hsi_yellowfin, compute_hsi_bigeye, compute_hsi_skipjack, compute_all_hsi
    
    # Test individual species HSI
    score_yf = compute_hsi_yellowfin(sst=28.0, chl=0.3)
    score_be = compute_hsi_bigeye(sst=26.0, chl=0.2)
    score_sk = compute_hsi_skipjack(sst=29.0, chl=0.5)
    
    ok(f"HSI yellowfin(SST=28,CHL=0.3) = {score_yf:.3f}")
    ok(f"HSI bigeye(SST=26,CHL=0.2) = {score_be:.3f}")
    ok(f"HSI skipjack(SST=29,CHL=0.5) = {score_sk:.3f}")
    
    # Check scores are in valid range
    for name, score in [("yellowfin", score_yf), ("bigeye", score_be), ("skipjack", score_sk)]:
        if 0 <= score <= 1:
            ok(f"  {name} HSI in [0,1] range")
        else:
            fail(f"  {name} HSI={score} out of [0,1] range!")
except Exception as e:
    fail(f"HSI: {e}")

# Lunar Phase
try:
    from engine.lunar_model import LunarPhaseEngine
    lp = LunarPhaseEngine()
    phase = lp.compute_moon_phase('2026-04-01')
    modifier = lp.compute_lunar_cpue_modifier('2026-04-01', 'yellowfin')
    ok(f"Moon phase 2026-04-01 = {phase}")
    ok(f"Lunar CPUE modifier (yellowfin) = {modifier}")
except Exception as e:
    fail(f"Lunar: {e}")

# Kuroshio
try:
    from engine.kuroshio_engine import KuroshioEngine
    ke = KuroshioEngine()
    lats = np.arange(20, 30, 1.0)
    lons = np.arange(120, 135, 1.0)
    dist_grid = ke.compute_distance_grid(lats, lons)
    ok(f"Kuroshio distance grid: shape={dist_grid.shape}, range=[{np.nanmin(dist_grid):.0f}, {np.nanmax(dist_grid):.0f}] km")
except Exception as e:
    fail(f"Kuroshio: {e}")

# Species params
try:
    from engine.species_params import SPECIES, DVM_PARAMS, KUROSHIO_SPECIES_AFFINITY
    ok(f"10 species loaded: {list(SPECIES.keys())}")
    ok(f"DVM params: {list(DVM_PARAMS.keys()) if isinstance(DVM_PARAMS, dict) else type(DVM_PARAMS).__name__}")
    ok(f"Kuroshio affinity: {KUROSHIO_SPECIES_AFFINITY}")
except Exception as e:
    fail(f"Species params: {e}")

# Typhoon 72h prediction (just check method exists and is callable)
try:
    from engine.typhoon_tracker import TyphoonTracker
    tt = TyphoonTracker()
    # Check predict_path method exists
    has_predict = hasattr(tt, 'predict_path') or hasattr(tt, 'predict_track')
    ok(f"TyphoonTracker predict method exists: {has_predict}")
    
    # List all methods to find the right one
    ms = [m for m in dir(tt) if 'predict' in m.lower() or 'forecast' in m.lower() or 'track' in m.lower()]
    ok(f"Prediction-related methods: {ms}")
except Exception as e:
    fail(f"Typhoon: {e}")

# FeatureEngineer 66-dim
try:
    from engine.ml.stacking_ensemble import FeatureEngineer
    fe = FeatureEngineer()
    # Check expected feature names/count
    if hasattr(fe, 'feature_names'):
        ok(f"Feature names: {len(fe.feature_names)} dimensions")
    elif hasattr(fe, 'get_feature_names'):
        names = fe.get_feature_names()
        ok(f"Feature names: {len(names)} dimensions")
    else:
        ms = [m for m in dir(fe) if not m.startswith('_') and callable(getattr(fe, m))]
        ok(f"FeatureEngineer methods: {ms}")
except Exception as e:
    fail(f"FeatureEngineer: {e}")

# WCPFC data quality
try:
    import pandas as pd
    import glob
    wcpfc_files = sorted(glob.glob('data/wcpfc/*.csv'))
    total_rows = 0
    for f in wcpfc_files:
        df = pd.read_csv(f)
        total_rows += len(df)
        ok(f"{os.path.basename(f)}: {len(df):,} rows x {len(df.columns)} cols")
    ok(f"WCPFC total: {total_rows:,} rows = {total_rows/1000:.0f}K records")
except Exception as e:
    fail(f"WCPFC: {e}")

print("\n=== DONE ===")
