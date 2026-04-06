# OceanMaster v15.3 — ML Training Report

> Generated: 2026-03-01 09:50 UTC

---

## Summary

| Species | R² | RMSE | MAE | Acc ±20% | Acc ±30% | N Samples |
|---------|-----|------|-----|----------|----------|-----------|
| skipjack | 0.8809 | 0.0719 | 0.0334 | 57.7% | 66.7% | 300 |

---

## Skipjack

- **R²**: 0.8809
- **RMSE**: 0.0719
- **MAE**: 0.0334
- **Accuracy (±20%)**: 57.7%
- **Accuracy (±30%)**: 66.7%
- **Mean Bias**: -0.0009
- **Prediction range**: P10=0.000, P50=0.018, P90=0.369

### Top Features

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | sst | 0.7072 |
| 2 | day_of_year_sin | 0.0568 |
| 3 | season_sin | 0.0476 |
| 4 | chl_log | 0.0455 |
| 5 | sst_seasonal_derivative | 0.0305 |
| 6 | sst_x_chl | 0.0268 |
| 7 | season_cos | 0.0108 |
| 8 | t100 | 0.0108 |
| 9 | day_of_year_cos | 0.0092 |
| 10 | delta_t_surface_100 | 0.0077 |

---

## Data Source

> **Synthetic data** — trained on scientifically realistic generated data.
> Swap for real FAO/RFMO CPUE data for production accuracy.
> Expected R² with real data: 0.40-0.55 (higher noise, more realistic).
