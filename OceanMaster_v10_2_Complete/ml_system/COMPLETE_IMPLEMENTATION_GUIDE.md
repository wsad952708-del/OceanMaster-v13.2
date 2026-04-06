# OceanMaster ML Training & Backtest System — Complete Guide

## System Overview

This package provides a **complete, runnable ML training and backtesting pipeline** for OceanMaster. It bridges the gap identified in the technical assessment: the original system had ML architecture but no training data, no trained weights, and no validation.

**This system delivers:**
- Synthetic CPUE data generator grounded in real fisheries science
- Feature engineering pipeline (26 features including Φ, SEAPODYM HSI)
- Stacking Ensemble training (RF + GBR + ExtraTrees + Ridge → RidgeCV meta)
- 4-strategy backtesting (walk-forward, rolling window, spatial, seasonal)
- Science + ML fusion predictor with confidence estimation
- FastAPI deployment package

---

## Verified Performance Results

Results from running the pipeline on 2,000 synthetic yellowfin tuna records:

### Training Metrics
| Metric | Value |
|--------|-------|
| Train R² | 0.926 |
| Cross-Validation R² | 0.760 ± 0.020 |
| Test R² | 0.782 |
| Test MAE | 5.52 kg/day |
| Test RMSE | 7.70 kg/day |
| Accuracy (±20%) | 69.0% |
| Accuracy (±30%) | 85.8% |
| Spearman Rank Correlation | 0.910 |

### Backtest Results (4 Strategies)
| Strategy | Avg R² | Avg MAE | Avg Acc±20% | Folds |
|----------|--------|---------|-------------|-------|
| Walk-Forward (2020-2024) | 0.776 ± 0.032 | 5.82 kg/day | 66.7% | 5 |
| Rolling Window (24mo/6mo) | 0.723 ± 0.058 | 6.46 kg/day | 61.0% | 15 |
| Spatial Leave-One-Out | 0.674 ± 0.128 | 6.47 kg/day | 59.8% | 4 |
| Seasonal Leave-One-Out | 0.369 ± 0.183 | 8.98 kg/day | 49.4% | 4 |
| **Overall Average** | **0.636** | **6.93 kg/day** | **59.2%** | — |

### Model Comparison
| Model | Test R² | Test MAE |
|-------|---------|----------|
| Random Forest | 0.777 | 5.64 |
| Gradient Boosting | 0.770 | 5.63 |
| Extra Trees | 0.780 | 5.57 |
| Ridge (baseline) | 0.672 | 7.09 |
| **Stacking Ensemble** | **0.782** | **5.52** |

---

## File Structure

```
ml_system/
├── run_pipeline.py                    # Master runner (start here)
├── historical_data_collector.py       # Synthetic CPUE data generator
├── feature_engineering_pipeline.py    # 26-feature engineering pipeline
├── oceanmaster_ml_trainer_v2.py       # Stacking Ensemble trainer
├── backtest_engine.py                 # 4-strategy backtest engine
├── integrated_predictor.py            # Science + ML fusion predictor
├── deployment_package.py              # FastAPI server + Dockerfile
├── test_suite.py                      # 11 verification tests
├── data/
│   ├── synthetic_cpue_yellowfin.csv   # Training data (2000 records)
│   └── ...
├── models/
│   ├── stacking_yellowfin.pkl         # Trained ensemble model
│   ├── feature_pipeline_yellowfin.pkl # Fitted scaler + feature config
│   ├── training_metrics_yellowfin.json
│   └── ...
└── reports/
    ├── training_report_yellowfin.txt  # Human-readable training report
    ├── training_results_yellowfin.json
    ├── backtest_results_yellowfin.json
    └── pipeline_results.json          # Master results
```

---

## Quick Start

### 1. Run the Full Pipeline (recommended)
```bash
cd oceanmaster
python3 ml_system/run_pipeline.py
```

This will:
1. Generate 2,000 synthetic CPUE records
2. Engineer 26 ML features
3. Train Stacking Ensemble
4. Evaluate on held-out test set
5. Run integration tests with the predictor

### 2. Run with Backtest (slower, more thorough)
```bash
python3 ml_system/run_pipeline.py --species yellowfin --samples 2000
# (without --skip-backtest, backtesting runs automatically)
```

### 3. Train All Species
```bash
python3 ml_system/run_pipeline.py --all-species --samples 2000
```

### 4. Run Tests Only
```bash
python3 ml_system/test_suite.py
```

### 5. Run Backtest Only (uses pre-generated data)
```bash
python3 ml_system/run_pipeline.py --backtest-only
```

---

## How to Use with Real FAO Data

When you obtain real FAO CPUE data, replace the synthetic data:

### Step 1: Download FAO Data
Visit: https://www.fao.org/fishery/statistics-query/en/capture/capture_quantity
- Region: Area 71 (Western Central Pacific)
- Species: Yellowfin tuna
- Years: 2015-2024
- Format: CSV

### Step 2: Format Your CSV
Required columns (minimum):
```csv
year,month,lat,lon,cpue_kg_per_day,species
2020,1,15.5,131.0,45.6,yellowfin
2020,2,12.3,142.5,32.1,yellowfin
```

Optional columns (will be computed if missing):
```
sst, chl, ssh, do, current_speed, front_strength, eddy_strength, phi, hsi
```

### Step 3: Run Training with Real Data
```python
import pandas as pd
from feature_engineering_pipeline import FeatureEngineeringPipeline
from oceanmaster_ml_trainer_v2 import OceanMasterMLTrainer, TrainingConfig
from sklearn.model_selection import train_test_split

# Load your data
df = pd.read_csv("your_fao_data.csv")

# Feature engineering
pipeline = FeatureEngineeringPipeline(species="yellowfin")
X, y, names = pipeline.transform(df, fit=True, include_target=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
config = TrainingConfig(species="yellowfin")
trainer = OceanMasterMLTrainer(config)
trainer.train(X_train, y_train, names)
metrics = trainer.evaluate(X_test, y_test)

# Save
trainer.save("models/")
pipeline.save("models/feature_pipeline_yellowfin.pkl")
```

---

## Synthetic Data Methodology

Since real FAO CPUE data requires manual download, this system generates scientifically-grounded synthetic data. Here's how it works:

### CPUE Model
```
CPUE = median_CPUE × (0.5 × Φ_score + 0.3 × HSI_score + 0.2 × eddy_score)
       × front_bonus × season_mod × spatial_pref × noise
```

### Scientific Basis
- **Φ (Metabolic Index)**: From Deutsch 2015 (Science), implemented exactly as in `commercial_core_v2.py`
- **SEAPODYM HSI**: Simplified version of the SPC model used by WCPFC
- **CPUE Ranges**: Based on WCPFC public statistics (yellowfin: 5-180 kg/day, median ~45)
- **Seasonal Patterns**: Literature-based peak months per species
- **Spatial Patterns**: Known fishing grounds (Kuroshio, Western Warm Pool)
- **Noise**: 20% multiplicative lognormal noise (realistic fishing variability)

### Limitations
- Synthetic data has known, clean relationships (R² will be higher than with real data)
- Real-world CPUE is affected by gear type, vessel size, captain skill
- The 26-feature set may need adjustment for real data
- With real FAO data, expect R² of 0.4-0.6 (still useful for commercial prediction)

---

## Integration with OceanMaster Main Pipeline

### Adding ML to main_v10_2.py

```python
# At the top of main_v10_2.py, add:
import joblib
from pathlib import Path

# In the pipeline class __init__, add:
try:
    self.ml_model = joblib.load('ml_system/models/stacking_yellowfin.pkl')
    self.ml_pipeline = joblib.load('ml_system/models/feature_pipeline_yellowfin.pkl')
    self.ml_available = True
    log.info("✅ ML model loaded")
except FileNotFoundError:
    self.ml_available = False
    log.info("⚠️ ML model not found, using science-only mode")

# After HSI computation, add ML fusion:
if self.ml_available:
    # See integrated_predictor.py for full fusion logic
    from ml_system.integrated_predictor import IntegratedPredictor
    predictor = IntegratedPredictor(species=sp)
    predictor.ml_model = self.ml_model
    predictor.feature_pipeline = self.ml_pipeline
    predictor.ml_available = True
    # ... use predictor.predict() for each grid cell
```

---

## Deployment

### Local API Server
```bash
pip install fastapi uvicorn
python3 ml_system/deployment_package.py
# Server starts at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Docker
```bash
python3 ml_system/deployment_package.py --dockerfile
docker build -f Dockerfile.ml -t oceanmaster-ml .
docker run -p 8000:8000 oceanmaster-ml
```

### API Endpoints
```
POST /predict         — Single location CPUE prediction
POST /predict/grid    — Grid prediction for a region
GET  /health          — Health check
GET  /model/info      — Model metadata
GET  /species         — Supported species
```

---

## Next Steps for Production

1. **Replace synthetic data with real FAO CPUE** — This is the single most impactful improvement
2. **Add Global Fishing Watch data** — Free API, adds 5-10% accuracy
3. **Collect captain reports** — Use `engine/captain_reports.py` for real-time learning
4. **Expand species** — Run pipeline for bigeye, skipjack, albacore
5. **Deploy and validate** — Get 3-5 fishermen to test for 3 months
6. **Publish** — Submit to Remote Sensing or Fisheries Research journal

---

## Dependencies

**Required** (already in most Python environments):
- scikit-learn >= 1.2
- numpy >= 1.24
- pandas >= 2.0
- scipy >= 1.10
- joblib >= 1.2

**Optional** (improves performance):
- xgboost — Adds XGBoost base learner
- lightgbm — Adds LightGBM base learner
- fastapi + uvicorn — API server
- shap — Feature interpretability

The system gracefully degrades without optional dependencies.
