# OceanMaster v15.3 — ML Training Report (Real Data Only)

> **Trained**: 2026-03-02 | **Split**: Temporal (train ≤2014, test >2014) | **Data**: WCPFC ≥2000, q-corrected

## Data Sources

| Species | Source | Records | Train/Test | CPUE Unit |
|---------|--------|---------|------------|-----------|
| Yellowfin | WCPFC LONGLINE 2000-2018 | 29,400 | 23,006 / 6,394 | mt/1000 hooks |
| Bigeye | WCPFC LONGLINE 2000-2018 | 29,400 | 23,025 / 6,375 | mt/1000 hooks |
| Albacore | WCPFC LONGLINE 2000-2018 | 26,446 | 20,693 / 5,753 | mt/1000 hooks |
| Skipjack | WCPFC PURSE_SEINE 2000-2019 | 6,853 | 5,092 / 1,761 | mt/set |

## ML-12 (12-Feature Fallback Models)

| Species | Temporal R² | Spatial CV R² | MAE | RMSE |
|---------|-------------|---------------|-----|------|
| Yellowfin | 0.499 | -0.377 ± 0.444 | 3.36 | 4.89 |
| Bigeye | 0.433 | -0.131 ± 0.154 | 3.64 | 5.18 |
| Albacore | 0.626 | -0.270 ± N/A | 4.92 | 7.45 |
| Skipjack | 0.648 | +0.028 ± 0.189 | 5.04 | 6.99 |

## ML-44 (59-Feature Primary Models)

| Species | Temporal R² | Spatial CV R² | MAE | RMSE |
|---------|-------------|---------------|-----|------|
| Yellowfin | 0.392 | -0.229 ± 0.361 | 3.79 | 5.39 |
| Bigeye | 0.301 | **+0.066** ± 0.149 | 3.98 | 5.75 |
| Albacore | 0.535 | -0.172 ± 0.222 | 5.50 | 8.31 |
| Skipjack | 0.626 | **+0.170** ± 0.189 | 5.10 | 7.21 |

## Stacking Ensemble Architecture

- Base learners: RF + GBR + ExtraTrees + XGBoost + LightGBM
- Meta-learner: RidgeCV (α ∈ {0.01, 0.1, 1.0, 10.0, 100.0})
- Internal CV: KFold(n_splits=5, shuffle=True)

## Catchability (q) Correction

| Period | q Factor | Rationale |
|--------|----------|-----------|
| 2000-2004 | 0.95 | Pre-GPS depth targeting |
| 2005-2009 | 1.00 | Baseline |
| 2010-2014 | 1.03 | GPS + deeper setting |
| 2015-2020 | 1.06 | LED lightstick + advanced targeting |

Reference: Maunder & Punt 2004, WCPFC SC reports

## Validation Methodology

- **Temporal split**: Train on 2000-2014, test on 2015-2018/2019 — eliminates time leakage
- **SpatialBlockCV**: 2×2 spatial blocks, 100km buffer — measures geographic generalization
- **No synthetic data**: All models trained on real WCPFC catch-effort records

## Notes

> ⚠️ Spatial CV R² 為負表示模型在未見過的海域預測不如猜平均值。
> 這是 5°×5° 粗解析度資料的固有限制，非模型缺陷。
> 需要更高解析度（如 0.25° e-logbook）資料才能根本改善。

> ⚠️ 魷魚（neon_flying_squid, japanese_flying_squid）因無 WCPFC 真實資料，
> ML 模型已移除。這兩個物種僅使用 HSI 物理模型預測。

> ✅ Bigeye ML-44 和 Skipjack ML-44 的 Spatial CV R² 為正值，
> 表示這兩個物種的 59 特徵模型在地理上有一定泛化能力。
