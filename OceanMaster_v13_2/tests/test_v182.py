from datetime import datetime
from engine.satellite.historical_fetcher import (
    HistoricalFeatureExtractor,
    detect_sst_front,
    detect_marine_heatwave,
    gfw_proxy_label,
    compute_tuna_arrival_doy,
    GFW_PROXY_WEIGHT,
)

ext = HistoricalFeatureExtractor(use_cmems=False, use_erddap=False)
f = ext.extract_features(datetime(2024, 3, 15), 25.0, 135.0)

new_feats = [
    "front_intensity", "front_distance_km",
    "mhw_active", "mhw_intensity", "mhw_duration_days",
    "gfw_proxy_score", "tuna_arrival_doy",
]
print("=== v18.2 New Features ===")
for k in new_feats:
    print(f"  {k:25s} = {f.get(k, 'MISSING')}")

total = len([k for k in f if not k.startswith("_")])
print(f"\nTotal features: {total}")
print(f"GFW_PROXY_WEIGHT: {GFW_PROXY_WEIGHT}")

# Test tuna_arrival_doy
arr = compute_tuna_arrival_doy(60)
print(f"\ntuna_arrival_doy(bloom_start=60): {arr} (expected: 102)")

# Test front detection
front = detect_sst_front(25.0, [24.0, 25.5, 25.1, 26.2])
print(f"front_intensity: {front['front_intensity']:.3f}")
print(f"front_distance: {front['front_distance_km']}")

# Test MHW
mhw = detect_marine_heatwave(30.0, 27.0, 29.0)
print(f"\nMHW (SST=30, clim=27, 90th=29):")
print(f"  active={mhw['mhw_active']}, intensity={mhw['mhw_intensity']:.2f}")
