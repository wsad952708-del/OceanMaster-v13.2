"""
Train ML-12 fallback models (12-feature).
Generates stacking models compatible with main_v10_3.py L1212-1234 fallback path.

Output:
  models/ml12_stacking_{sp}.pkl      → dict {model, scaler, species}
  models/ml12_stacking_{sp}_meta.json
"""
import numpy as np
import logging
import pickle
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from engine.ml.stacking_ensemble import SpatialBlockCV

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ML12-Train")

# 12 features matching main_v10_3.py L1212-1216
FEATURES_12 = [
    "sst", "chl", "ssh", "do", "current_speed",
    "front_strength", "eddy_strength", "phi",
    "bathy_depth", "bathy_slope", "dist_seamount", "dist_shelf_break",
]

# Species-specific parameter ranges
SPECIES_PARAMS = {
    "yellowfin": {"sst_range": (24, 31), "chl_range": (0.05, 0.5), "cpue_median": 45, "cpue_scale": 25},
    "bigeye":    {"sst_range": (15, 25), "chl_range": (0.1, 0.6),  "cpue_median": 35, "cpue_scale": 20},
    "skipjack":  {"sst_range": (26, 32), "chl_range": (0.1, 0.4),  "cpue_median": 100,"cpue_scale": 40},
    "albacore":  {"sst_range": (15, 22), "chl_range": (0.2, 1.0),  "cpue_median": 32, "cpue_scale": 15},
}


def generate_12feat_data(species: str, n: int = 2000, seed: int = 42) -> tuple:
    """Generate synthetic 12-feature training data with physically-plausible CPUE."""
    rng = np.random.RandomState(seed + sum(ord(c) for c in species))
    params = SPECIES_PARAMS[species]

    sst = rng.uniform(*params["sst_range"], n)
    chl = np.exp(rng.uniform(np.log(params["chl_range"][0]), np.log(params["chl_range"][1]), n))
    ssh = rng.normal(0, 0.1, n)
    do = rng.uniform(3.5, 7.0, n)
    current_speed = rng.exponential(0.25, n) + 0.05
    front_strength = rng.exponential(0.02, n)
    eddy_strength = rng.exponential(0.01, n) * 100
    phi = rng.uniform(1.0, 8.0, n)
    bathy_depth = -rng.uniform(500, 5000, n)
    bathy_slope = rng.exponential(1.5, n)
    dist_seamount = rng.exponential(150, n)
    dist_shelf_break = rng.exponential(100, n)

    X = np.column_stack([
        sst, chl, ssh, do, current_speed,
        front_strength, eddy_strength, phi,
        bathy_depth, bathy_slope, dist_seamount, dist_shelf_break,
    ])

    # CPUE = f(thermal, feeding, fronts, phi, bathymetry noise)
    sst_opt, sst_sigma = params["sst_range"][0] + (params["sst_range"][1]-params["sst_range"][0])*0.6, 3.0
    thermal = np.exp(-0.5 * ((sst - sst_opt) / sst_sigma) ** 2)
    chl_log = np.log10(np.maximum(chl, 0.001))
    feeding = 1.0 / (1.0 + np.exp(-3.0 * (chl_log + 0.5)))
    front_bonus = 1.0 + 2.0 * np.clip(front_strength / 0.05, 0, 1)
    phi_factor = np.clip((phi - 1.5) / 4.0, 0.1, 1.0)
    bathy_factor = np.clip(-bathy_depth / 3000, 0.3, 1.0)
    seamount_bonus = 1.0 + 0.3 * np.exp(-dist_seamount / 50)

    cpue = (
        params["cpue_median"]
        * (0.3 * thermal + 0.25 * feeding + 0.2 * phi_factor + 0.15 * front_bonus * 0.5 + 0.1 * bathy_factor)
        * seamount_bonus
    )
    noise = rng.normal(1.0, 0.15, n)
    cpue = np.clip(cpue * noise, 0, params["cpue_median"] * 4)

    # Synthetic lat/lon for SpatialBlockCV (WCPFC range)
    syn_lats = rng.uniform(-10, 35, n)   # 10S-35N
    syn_lons = rng.uniform(120, 175, n)  # 120E-175E

    return X, cpue, FEATURES_12, syn_lats, syn_lons


def train_12feat_model(species: str, n_samples: int = 2000):
    """Train and save a 12-feature ML model."""
    log.info(f"{'='*60}")
    log.info(f"  Training 12-feature model: {species.upper()}")
    log.info(f"{'='*60}")
    t0 = time.time()

    X, y, feature_names, syn_lats, syn_lons = generate_12feat_data(species, n_samples)
    assert X.shape[1] == 12

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_train_s = np.nan_to_num(X_train_s, nan=0, posinf=0, neginf=0)
    X_test_s = np.nan_to_num(X_test_s, nan=0, posinf=0, neginf=0)

    # Build stacking model
    try:
        import xgboost as xgb
        has_xgb = True
    except ImportError:
        has_xgb = False
    try:
        import lightgbm as lgb
        has_lgb = True
    except ImportError:
        has_lgb = False

    estimators = [
        ("rf", RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=10, n_jobs=-1, random_state=42)),
        ("gbr", GradientBoostingRegressor(n_estimators=150, max_depth=6, learning_rate=0.05, subsample=0.8, random_state=42)),
        ("et", ExtraTreesRegressor(n_estimators=200, max_depth=12, min_samples_leaf=10, n_jobs=-1, random_state=42)),
    ]
    if has_xgb:
        estimators.append(("xgb", xgb.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, n_jobs=-1, random_state=42)))
    if has_lgb:
        estimators.append(("lgb", lgb.LGBMRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, n_jobs=-1, random_state=42, verbose=-1)))

    model = StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]),
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
    )

    log.info(f"  Fitting {len(estimators)} base learners + RidgeCV meta...")
    model.fit(X_train_s, y_train)

    # Evaluate
    y_pred = model.predict(X_test_s)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    log.info(f"  R²={r2:.4f}  MAE={mae:.2f}  RMSE={rmse:.2f}")

    # Spatial Block CV (honest R²)
    X_all_s = scaler.transform(np.nan_to_num(X, nan=0, posinf=0, neginf=0))
    spatial_cv = SpatialBlockCV(n_blocks=5, buffer_km=50)
    sp_result = spatial_cv.cross_val_score(model, X_all_s, y, syn_lats, syn_lons)
    log.info(f"  SpatialBlockCV R²={sp_result['r2_mean']:.4f} +/- {sp_result['r2_std']:.4f}")

    # Save as dict {model, scaler, species} — compatible with main_v10_3.py L401
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    out_path = model_dir / f"ml12_stacking_{species}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "species": species}, f)
    log.info(f"  Saved: {out_path} ({out_path.stat().st_size // 1024} KB)")

    # HMAC sign
    try:
        from engine.ml.stacking_ensemble import _sign_model_file
        _sign_model_file(out_path)
        log.info(f"  HMAC signed: {out_path}")
    except Exception as e:
        log.warning(f"  HMAC signing skipped: {e}")

    # Meta
    meta = {
        "species": species, "n_features": 12, "feature_names": feature_names,
        "r2": float(r2), "mae": float(mae), "rmse": float(rmse),
        "cv_r2": float(cv_r2.mean()), "cv_r2_std": float(cv_r2.std()),
        "n_train": int(X_train.shape[0]), "n_test": int(X_test.shape[0]),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "base_learners": [name for name, _ in estimators],
    }
    meta_path = model_dir / f"ml12_stacking_{species}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    log.info(f"  Done in {elapsed:.1f}s\n")
    return meta


if __name__ == "__main__":
    all_results = {}
    for sp in ["yellowfin", "bigeye", "skipjack", "albacore"]:
        all_results[sp] = train_12feat_model(sp)

    print("\n" + "=" * 70)
    print(f"{'Species':<12} {'R²':>8} {'MAE':>8} {'RMSE':>8} {'CV R²':>10}")
    print("-" * 70)
    for sp, m in all_results.items():
        print(f"{sp:<12} {m['r2']:>8.4f} {m['mae']:>8.2f} {m['rmse']:>8.2f} {m['cv_r2']:>10.4f}")
    print("=" * 70)
