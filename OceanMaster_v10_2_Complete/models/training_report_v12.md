# OceanMaster v12 — ML Training Report

> Generated: 2026-02-20 17:01 UTC

---

## Summary

| Species | R² | RMSE | MAE | Acc ±20% | Acc ±30% | N Samples |
|---------|-----|------|-----|----------|----------|-----------|
| yellowfin | 0.8826 | 0.0636 | 0.0293 | 56.2% | 69.2% | 400 |
| bigeye | 0.9036 | 0.0740 | 0.0424 | 47.5% | 62.0% | 400 |
| albacore | 0.9019 | 0.0677 | 0.0351 | 58.7% | 74.3% | 300 |
| skipjack | 0.9084 | 0.0615 | 0.0276 | 62.0% | 71.3% | 300 |

---

## Yellowfin

- **R²**: 0.8826
- **RMSE**: 0.0636
- **MAE**: 0.0293
- **Accuracy (±20%)**: 56.2%
- **Accuracy (±30%)**: 69.2%
- **Mean Bias**: -0.0054
- **Prediction range**: P10=0.000, P50=0.008, P90=0.356

### Top Features

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | sst | 0.7778 |
| 2 | day_of_year_sin | 0.0493 |
| 3 | season_sin | 0.0394 |
| 4 | sst_x_chl | 0.0325 |
| 5 | chl_log | 0.0144 |
| 6 | chl_local_mean | 0.0127 |
| 7 | moon_phase | 0.0126 |
| 8 | season_cos | 0.0066 |
| 9 | chl_gradient | 0.0062 |
| 10 | day_of_year_cos | 0.0047 |

---

## Bigeye

- **R²**: 0.9036
- **RMSE**: 0.0740
- **MAE**: 0.0424
- **Accuracy (±20%)**: 47.5%
- **Accuracy (±30%)**: 62.0%
- **Mean Bias**: 0.0039
- **Prediction range**: P10=0.000, P50=0.084, P90=0.496

### Top Features

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | sst | 0.7247 |
| 2 | season_sin | 0.0540 |
| 3 | day_of_year_sin | 0.0487 |
| 4 | chl_local_mean | 0.0481 |
| 5 | sst_x_chl | 0.0278 |
| 6 | chl_log | 0.0188 |
| 7 | season_cos | 0.0175 |
| 8 | day_of_year_cos | 0.0157 |
| 9 | moon_phase | 0.0061 |
| 10 | dist_to_shelf_break | 0.0039 |

---

## Albacore

- **R²**: 0.9019
- **RMSE**: 0.0677
- **MAE**: 0.0351
- **Accuracy (±20%)**: 58.7%
- **Accuracy (±30%)**: 74.3%
- **Mean Bias**: -0.0044
- **Prediction range**: P10=0.000, P50=0.012, P90=0.459

### Top Features

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | sst | 0.8031 |
| 2 | season_sin | 0.0298 |
| 3 | day_of_year_sin | 0.0248 |
| 4 | season_cos | 0.0184 |
| 5 | sst_x_chl | 0.0179 |
| 6 | day_of_year_cos | 0.0157 |
| 7 | chl_local_mean | 0.0137 |
| 8 | chl_log | 0.0122 |
| 9 | moon_phase | 0.0105 |
| 10 | enso_oni | 0.0049 |

---

## Skipjack

- **R²**: 0.9084
- **RMSE**: 0.0615
- **MAE**: 0.0276
- **Accuracy (±20%)**: 62.0%
- **Accuracy (±30%)**: 71.3%
- **Mean Bias**: -0.0071
- **Prediction range**: P10=0.000, P50=0.000, P90=0.371

### Top Features

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | sst | 0.8002 |
| 2 | chl_log | 0.0502 |
| 3 | day_of_year_sin | 0.0369 |
| 4 | season_sin | 0.0364 |
| 5 | chl_local_mean | 0.0275 |
| 6 | sst_x_chl | 0.0137 |
| 7 | ais_fishing_density | 0.0037 |
| 8 | day_of_year_cos | 0.0025 |
| 9 | season_cos | 0.0024 |
| 10 | moon_phase | 0.0021 |

---

## Data Source

> **Synthetic data** — trained on scientifically realistic generated data.
> Swap for real FAO/RFMO CPUE data for production accuracy.
> Expected R² with real data: 0.40-0.55 (higher noise, more realistic).
