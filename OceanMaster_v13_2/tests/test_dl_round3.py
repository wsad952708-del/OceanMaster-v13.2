"""Round 3 regression tests: nan handling, save_weights mkdir, warnings."""
import numpy as np
import sys
import os
import warnings
import tempfile

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} -- {detail}")

print("=== Round 3 Regression Tests ===\n")

# 1. normalize_field all-NaN produces NO warnings
from engine.ml.dl_data_pipeline import normalize_field, compute_time_statistics
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = normalize_field(np.full((5, 5), np.nan), "sst", mode="per_sample")
    runtime_warns = [x for x in w if issubclass(x.category, RuntimeWarning)]
    check("normalize all-NaN no RuntimeWarning",
          len(runtime_warns) == 0,
          f"Got {len(runtime_warns)} warnings: {[str(x.message) for x in runtime_warns]}")

check("normalize all-NaN output is zeros", np.allclose(result, 0.0))

# 2. normalize_field single value (std=0)
single = np.full((5, 5), 25.0)
norm_single = normalize_field(single, "sst", mode="per_sample")
check("normalize constant field no NaN", not np.any(np.isnan(norm_single)))

# 3. np.nan_to_num with explicit nan kwarg in time stats
daily = [np.full((3, 3), np.nan) for _ in range(5)]
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    ts = compute_time_statistics(daily)
    check("time_stats all-NaN grids no crash", True)
check("time_stats mean is 0 for nan input", np.allclose(ts["mean"], 0.0))

# 4. U-Net save_weights to nonexistent dir
import torch
from engine.ml.unet_fishing import UNetFishingPredictor
p = UNetFishingPredictor(use_cbam=False)
save_dir = os.path.join(tempfile.gettempdir(), "oceanmaster_test_r3", "deep", "nested")
save_path = os.path.join(save_dir, "model.pt")
# Clean up first
if os.path.exists(save_path):
    os.remove(save_path)
try:
    p.save_weights(save_path)
    check("U-Net save_weights mkdir", os.path.exists(save_path))
except Exception as e:
    check("U-Net save_weights mkdir", False, str(e))

# 5. ConvLSTM save_weights to nonexistent dir
from engine.ml.convlstm_predictor import ConvLSTMPredictor
cl = ConvLSTMPredictor(hidden_channels=[16])
save_path2 = os.path.join(tempfile.gettempdir(), "oceanmaster_test_r3", "convlstm", "model.pt")
try:
    cl.save_weights(save_path2)
    check("ConvLSTM save_weights mkdir", os.path.exists(save_path2))
except Exception as e:
    check("ConvLSTM save_weights mkdir", False, str(e))

# 6. Load weights round-trip
p2 = UNetFishingPredictor(use_cbam=False, model_path=save_path)
check("U-Net load_weights round-trip", p2.is_trained is True)

# 7. ConvLSTM load weights round-trip
cl2 = ConvLSTMPredictor(hidden_channels=[16], model_path=save_path2)
check("ConvLSTM load_weights round-trip", cl2.is_trained is True)

# 8. Forward pass after load gives same result
dev = next(p.get_model().parameters()).device
feat = {"sst": np.random.rand(32, 32).astype(np.float32) * 30}
r1 = p.predict(feat, np.linspace(20, 40, 32), np.linspace(130, 160, 32))
r2 = p2.predict(feat, np.linspace(20, 40, 32), np.linspace(130, 160, 32))
check("U-Net save/load gives same output",
      np.allclose(r1["fishing_probability"], r2["fishing_probability"], atol=1e-5),
      f"max diff={np.max(np.abs(r1['fishing_probability'] - r2['fishing_probability'])):.6f}")

# 9. Trainer with mixed precision inputs (float64 target, float32 features)
from engine.ml.dl_data_pipeline import create_unet_dataset
samples = []
for _ in range(4):
    f = {"sst": np.random.rand(16, 16).astype(np.float64) * 30}  # NOTE: float64
    t = np.random.rand(16, 16).astype(np.float64)  # NOTE: float64
    samples.append((f, t))
try:
    ds = create_unet_dataset(samples, ["sst"])
    x, y = ds[0]
    check("Dataset handles float64 input", x.dtype == torch.float32,
          f"dtype={x.dtype}")
except Exception as e:
    check("Dataset handles float64 input", False, str(e))

# 10. CPUE density with negative CPUE (edge case - bad data)
from engine.ml.dl_data_pipeline import cpue_to_density_field
pts_bad = [{"lat": 25.0, "lon": 135.0, "cpue": -5.0}]
lats = np.linspace(20, 30, 10)
lons = np.linspace(130, 140, 10)
d = cpue_to_density_field(pts_bad, lats, lons)
check("CPUE density with negative CPUE doesn't crash", d.shape == (10, 10))

# 11. Syrjala with 1x1 array
from engine.ml.syrjala_test import syrjala_test
s = syrjala_test(np.array([[1.0]]), np.array([[1.0]]), n_permutations=9)
check("Syrjala 1x1 array", 0 <= s["p_value"] <= 1)

# 12. TZCF with equatorial data (neither clearly NH nor SH)
from engine.tzcf_tracker import TZCFTracker
lats_eq = np.linspace(-10, 10, 40)
lons_eq = np.linspace(140, 160, 30)
chl_eq = np.random.rand(40, 30) * 0.5
r_eq = TZCFTracker(0.2).compute(chl_eq, lats_eq, lons_eq)
check("TZCF equatorial no crash", True)

# 13. Thermotaxis with sst_field=None (pure random walk only)
from engine.lagrangian_advection import _compute_thermotaxis
lat_pts = np.array([25.0, 30.0], dtype=np.float64)
lon_pts = np.array([140.0, 145.0], dtype=np.float64)
dx, dy = _compute_thermotaxis(
    lat_pts, lon_pts,
    None, None, None, None, None, None, None,
    sst_field=None,  # No SST -> random walk only
    random_walk_std=0.01,
)
check("Thermotaxis sst=None (random only)", dx.shape == (2,) and dy.shape == (2,))

# 14. DataLoader pin_memory
from engine.ml.dl_data_pipeline import create_dataloader
ds2 = create_unet_dataset(samples[:2], ["sst"])
loader = create_dataloader(ds2, batch_size=2)
for batch in loader:
    X, Y = batch
    check("DataLoader batch shape X", X.shape == (2, 1, 16, 16), f"{X.shape}")
    check("DataLoader batch shape Y", Y.shape == (2, 1, 16, 16), f"{Y.shape}")
    break

# Cleanup
import shutil
test_dir = os.path.join(tempfile.gettempdir(), "oceanmaster_test_r3")
if os.path.exists(test_dir):
    shutil.rmtree(test_dir)

print(f"\n{'='*50}")
print(f"  Round 3: {PASS} passed, {FAIL} failed")
print(f"{'='*50}")
if FAIL > 0:
    sys.exit(1)
else:
    print("  ALL ROUND 3 TESTS PASSED")
