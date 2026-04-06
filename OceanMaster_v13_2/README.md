# 🌊 OceanMaster v13.2 — AI-Powered Tuna & Squid Fishing Ground Prediction System

> Predict optimal fishing grounds using satellite oceanography,
> trophic cascade modeling, and machine learning ensemble methods.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

OceanMaster is an end-to-end prediction system that identifies the most productive tuna fishing grounds across the Western & Central Pacific Ocean. It ingests data from **14+ satellite sources** (CMEMS, NOAA CoastWatch, NASA OceanColor, HYCOM, WOA2023), computes full **trophic cascade timelines** (phytoplankton → zooplankton → baitfish → tuna), and ranks hotspots using a **hybrid science-ML fusion** approach.

The system covers **6 species** — 4 tuna (yellowfin, bigeye, albacore, skipjack) and 2 squid (neon flying squid, Japanese flying squid) — with species-specific habitat models grounded in peer-reviewed fisheries oceanography. Squid models additionally incorporate **lunar phase** and **VIIRS nightlight** factors critical to jigging operations. A **stacking ensemble** (Random Forest + XGBoost + LightGBM + CatBoost + Ridge meta-learner) provides ML-based CPUE predictions, fused at a **80/20 ratio** (80% science-based HSI + 20% ML) as the ML generalization remains limited.

A professional **dark-theme dashboard** built with Leaflet.js and D3.js delivers interactive maps, trophic cascade timelines, SHAP explainability modals, and time-slider playback — designed for use on vessel bridges.

### Key Differentiators

- **Full trophic cascade**: Not just SST + Chl-a — models the entire food chain timing from bloom to tuna arrival
- **Hybrid prediction**: Science-based HSI ensures physical plausibility; ML captures non-linear patterns
- **SHAP explainability**: Every prediction comes with feature attribution — no black boxes
- **66-feature model**: Incorporates SST, Chl-a fractions, DVM, lunar phase, OMZ, eddies, fronts, ENSO, Kuroshio distance, Taiwan Strait flag, and more
- **Metabolic Index (Φ)**: Oxygen-limited habitat modeling per Deutsch et al. 2015

---

## Screenshots

> Dashboard v2: Full-screen Leaflet map + 4-panel layout with dark theme  
> Route: `http://localhost:8000/dashboard/v2`

---

## Key Features

### 🔬 Scientific Engine

| Module | Method | Reference |
|--------|--------|-----------|
| Habitat Suitability Index (HSI) | SEAPODYM feeding habitat model | Lehodey et al. 2008 |
| Metabolic Index Φ | O₂-limited habitat envelope | Deutsch et al. 2015 |
| Frontal Detection (BOA) | Gradient → Thinning → Connection | Belkin & O'Reilly 2009 |
| Eddy Detection | Okubo-Weiss cyclonic/anticyclonic | Isern-Fontanet 2006 |
| Chl-a Size Fractionation | Pico/nano/micro phytoplankton | Brewin et al. 2010 |
| Diel Vertical Migration | Lunar suppression model | Benoit-Bird et al. 2009 |
| OMZ Compression | Edge enrichment effects | Stramma et al. 2012 |
| Lagrangian Advection | Particle retention zones | FTLE-based |

### 🧬 Food Chain Prediction Engine (Phase 8)

| Feature | Description | Reference |
|---------|-------------|-----------|
| Bloom Phenology | Satellite-detected bloom stage & timing | Brody et al. 2013 |
| Zooplankton Response Lag | Temperature-dependent trophic transfer delay | Henson et al. 2009 |
| Tuna Arrival Windows | Species-specific aggregation prediction | Precioso et al. 2022 (TUN-AI) |
| Feeding Window Prediction | DVM-based dawn/dusk/night windows | Schaefer & Fuller 2007 |
| Migration Direction | SST gradient + Lagrangian advection | Lehodey et al. 2008 |
| School Residence Time | Eddy-retention estimation | — |

### 🤖 Machine Learning

- **Stacking Ensemble**: Random Forest + XGBoost + LightGBM + CatBoost + Ridge meta-learner (up to 6 base learners)
- **66-feature model** (v13.2: +kuroshio_distance, +taiwan_strait_flag) with automatic legacy fallback
- **Hybrid prediction**: 80% science-based HSI + 20% ML prediction (masked by Φ metabolic index)
- **SHAP explainability** for every prediction
- **Safe model loading** with HMAC-SHA256 signature verification before pickle.load

#### Training Results (Synthetic Data Baseline)

| Species | R² | RMSE | MAE | Acc ±20% | Acc ±30% | N Samples |
|---------|-----|------|-----|----------|----------|-----------|
| Yellowfin | 0.8826 | 0.0636 | 0.0293 | 56.2% | 69.2% | 400 |
| Bigeye | 0.9036 | 0.0740 | 0.0424 | 47.5% | 62.0% | 400 |
| Albacore | 0.9019 | 0.0677 | 0.0351 | 58.7% | 74.3% | 300 |
| Skipjack | 0.9084 | 0.0615 | 0.0276 | 62.0% | 71.3% | 300 |

**Top predictive features**: SST (0.77), season_sin, day_of_year_sin, chl_log, sst_x_chl, moon_phase

> ⚠️ **Note**: These results are trained on **scientifically realistic synthetic data**, not real CPUE catch records.
> Expected R² with real FAO/RFMO data: 0.40–0.55 (higher noise, more realistic distribution).
> See [FAO_WCPFC_Data_Guide.md](docs/FAO_WCPFC_Data_Guide.md) for data acquisition instructions.

### 📊 Dashboard v2

- **Leaflet.js** interactive map with heatmap, markers, front lines, eddy circles
- **D3.js** trophic cascade timeline visualization
- Species selector with real-time data refresh
- Time slider: -24h to +8d with playback animation
- Layer toggles: SST, Chl-a, fronts, eddies, hotspot markers
- SHAP modal for prediction explainability
- Dark theme optimized for bridge use
- Mobile responsive with glassmorphism design

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     14+ Satellite Data Sources                  │
│   CMEMS · NOAA CoastWatch · NASA OceanColor · HYCOM · WOA2023  │
│   VIIRS · GFW AIS · Argo · GEBCO · Open-Meteo · ONI           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                    ┌────▼────┐
                    │ Fetchers │  data_fetcher_v2.py + data_sources.py
                    │ + Cache  │  CircuitBreaker, TTL-based caching
                    └────┬────┘
                         │
            ┌────────────▼────────────┐
            │    Scientific Engine     │
            │  HSI · Φ · Fronts ·     │
            │  Eddies · DVM · OMZ ·   │
            │  Zooplankton · Lunar    │
            └────────────┬────────────┘
                         │
        ┌────────────────▼────────────────┐
        │     Food Chain Predictor         │
        │  Bloom → Zoo lag → Tuna arrival  │
        │  + Feeding windows + Migration   │
        └────────────────┬────────────────┘
                         │
              ┌──────────▼──────────┐
              │    ML Ensemble       │
              │  RF+XGB+LGBM+Cat+Ri │
              │  66 features · SHAP │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │   Hybrid Fusion      │
              │  80% HSI + 20% ML   │
              │  × Φ metabolic mask │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │   FastAPI Server     │
              │  API + Dashboard v2  │
              │  Auth · Rate Limit   │
              └─────────────────────┘
```

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Backend** | Python 3.10+, FastAPI, Uvicorn |
| **ML** | scikit-learn, XGBoost, LightGBM, CatBoost (optional), SHAP |
| **Data** | NumPy, SciPy, Pandas, xarray, netCDF4, httpx |
| **Frontend** | Leaflet.js, D3.js, Chart.js, Google Fonts (Inter, Noto Sans TC) |
| **Satellite APIs** | CMEMS (Copernicus), NOAA CoastWatch ERDDAP, NASA OceanColor, HYCOM, WOA2023 |
| **Deployment** | Docker, docker-compose, Nginx, systemd |

---

## Quick Start

### Option 1: Docker (Recommended)

```bash
git clone https://your-repo-host.com/your-org/oceanmaster.git
cd oceanmaster
cp .env.example .env
# Edit .env with your CMEMS credentials
docker-compose -f docker-compose.production.yml up -d --build
# Open http://localhost:8000/dashboard/v2
```

### Option 2: Local Installation

```bash
git clone https://your-repo-host.com/your-org/oceanmaster.git
cd oceanmaster
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your CMEMS credentials
python web_server.py
# Open http://localhost:8000/dashboard/v2
```

### Option 3: Production Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for full production setup with Nginx, SSL, systemd, and cron jobs.

---

## Configuration

### Required Environment Variables (.env)

```bash
# Copernicus Marine Service (free account)
# Register: https://data.marine.copernicus.eu/register
CMEMS_USERNAME=your_cmems_username
CMEMS_PASSWORD=your_cmems_password

# API key for authenticated endpoints
OCEANMASTER_API_KEY=your_secret_key

# Optional
COASTWATCH_API_KEY=your_key       # NOAA CoastWatch
GFW_API_KEY=your_key              # Global Fishing Watch
```

See [.env.example](.env.example) for full configuration options.

### Species Configuration (config.py)

Species parameters are managed via `engine/species_params.py` (Single Source of Truth), with HSI-specific overrides in `config.py`. Supported species:

| Key | Species | SST Optimal | Depth Range |
|-----|---------|-------------|-------------|
| `yellowfin` | 黃鰭鮪 Yellowfin | 28°C | 0–400m |
| `bigeye` | 大目鮪 Bigeye | 21°C | 100–500m |
| `albacore` | 長鰭鮪 Albacore | 19°C | 50–300m |
| `skipjack` | 正鰹 Skipjack | 29°C | 0–200m |
| `neon_flying_squid` | 🦑 北太赤魷 Neon Flying Squid | 18°C | 0–300m |
| `japanese_flying_squid` | 🦑 日本魷魚 Japanese Flying Squid | 17°C | 0–200m |

---

## API Documentation

### Public Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard v1 (legacy) |
| GET | `/dashboard/v2` | Dashboard v2 (professional) |
| GET | `/health` | Health check |

### Core API (Requires `X-API-Key` header)

> ⚠️ 所有 Core API 端點需在 request header 中放入 `X-API-Key: <your-key>`。
> 預設開發用金鑰：`dev-key-change-me`（請在生產環境更換）。

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/hotspots` | Top fishing hotspots with freshness indicator |
| GET | `/api/analyze` | Run real-time or historical analysis |
| GET | `/api/sea_conditions` | Current SST, Chl-a, currents overview |
| GET | `/api/status` | Pipeline status |
| GET | `/api/typhoon-status` | Active typhoons + danger zones |

### Food Chain & Behavior API (v12)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/food_chain` | DVM + lunar + zoo + OMZ ecological indicators |
| GET | `/api/v1/food_chain_timeline` | Full trophic cascade timeline |
| GET | `/api/v1/feeding_windows` | Optimal fishing time windows |
| GET | `/api/v1/fish_movement` | Tuna migration prediction |
| GET | `/api/v1/dvm_profile` | 24-hour DVM depth profile |
| GET | `/api/v1/squid_jigging_forecast` | 🦑 Squid jigging lunar forecast |
| GET | `/api/v1/explain` | SHAP feature attribution for a hotspot |

### Data Collection API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/report_catch` | Submit catch report (ground truth) |
| GET | `/api/v1/catch_stats` | Catch report statistics |

### Backtest API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/backtest?date=YYYY-MM-DD&species=YFT` | Model backtest vs GFW |

### System API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/system-health` | Data lineage & source health |

### Response Example

```json
{
  "food_chain_score": 0.72,
  "bloom_stage": "peak",
  "zoo_peak_eta_days": 3,
  "tuna_arrival_window": {"min_days": 5, "max_days": 10},
  "optimal_fishing_date": "2026-02-25",
  "confidence": 0.68,
  "recommendation": "Bloom peaking, zooplankton aggregating. Tuna expected in 5-10 days."
}
```

---

## ML Training

### Train with synthetic data (default)

```bash
python train_and_validate.py --all
```

### Train with real FAO/WCPFC data

```bash
# Place CPUE CSV in data/fao_cpue.csv
python train_and_validate.py --all --data-path data/fao_cpue.csv
```

### Train individual species

```bash
python train_and_validate.py --species yellowfin
```

Models are saved to `models/stacking_<species>.pkl` with metadata JSON and scaler files.

---

## Data Sources

| Data | Source | Resolution | Latency |
|------|--------|------------|---------|
| SST | NOAA OISST v2.1 / JPL MUR | 0.25° / 0.01° | ~1 day |
| Chl-a | MODIS Aqua / VIIRS | 4 km | ~1 day |
| Currents | OSCAR / CMEMS / HYCOM | 0.33° / 0.08° | ~5 days |
| Dissolved O₂ | WOA2023 THREDDS | 1° | Climatology |
| 3D Temp/Sal | HYCOM GLBv0.08 | 0.08° | ~1 day |
| BGC | CMEMS Global BGC | 0.25° | ~1 day |
| Night Lights | VIIRS Nightfire / DNB | 375 m | ~12 hours |
| Argo Profiles | Argo Floats | Point | ~12 hours |
| ENSO | NOAA CPC ONI | Global | Monthly |
| Bathymetry | ETOPO1 / ETOPO2022 | 1' / 15" | Static |
| NPP | Oregon State VGPM/CbPM2 | 1/6° | Monthly |
| AIS Fishing | Global Fishing Watch | — | ~12 hours |
| Sea State | Open-Meteo Marine | — | ~3 hours |
| Typhoons | JTWC via API | — | ~6 hours |

---

## Project Structure

```
OceanMaster_v13_2/
├── main_v10_3.py              # Main pipeline: data → HSI → ML → hotspots
├── web_server.py              # FastAPI server (all API endpoints)
├── config.py                  # Global configuration & data source URLs
├── train_and_validate.py      # ML training & validation script
├── backtest.py                # Backtesting framework
├── engine/                    # Core scientific engine
│   ├── data_fetcher_v2.py     #   Satellite data fetcher
│   ├── commercial_core_v2.py  #   HSI / Metabolic Index Φ
│   ├── algorithms.py          #   BOA frontal detection
│   ├── eddy_detector.py       #   Okubo-Weiss eddy classification
│   ├── dvm_model.py           #   Diel Vertical Migration + lunar
│   ├── zooplankton_proxy.py   #   Zoo proxy + Chl-a fractionation
│   ├── omz_model.py           #   OMZ compression/enrichment
│   ├── food_chain_predictor.py#   Trophic cascade timing
│   ├── fish_behavior_model.py #   Feeding windows + migration
│   ├── species_params.py      #   Species parameters (SSoT)
│   ├── shap_explainer.py      #   SHAP explainability
│   ├── ml/                    #   ML subsystem
│   │   ├── stacking_ensemble.py   Stacking ensemble + safe_model_load
│   │   ├── synthetic_training_data.py  Training data generator
│   │   ├── validation.py      #   Validation framework
│   │   └── fao_data_loader.py #   FAO/RFMO data interface
│   ├── infra.py               #   Cache, retry, circuit breaker
│   └── data_sources.py        #   CMEMS/Argo/ONI/GEBCO fetchers
├── models/                    # Trained ML models (.pkl + meta)
├── web/                       # Frontend
│   ├── dashboard_v2.html      #   Professional dashboard
│   ├── dashboard_v2.js        #   Map + D3 timeline + SHAP
│   └── dashboard_v2.css       #   Dark theme
├── deploy/                    # Deployment configs
│   ├── oceanmaster.service    #   systemd unit
│   └── nginx-oceanmaster.conf #   Nginx reverse proxy
├── Dockerfile.production      # Optimized multi-stage Docker build
├── docker-compose.production.yml
├── requirements.txt           # Python dependencies
├── DEPLOYMENT.md              # Production deployment guide
├── CHANGELOG_v12.md           # Version history
└── .env.example               # Environment variable template
```

---

## ⚠️ Known Limitations & Honest Disclosures  <!-- [v12-phase11-disclosure] -->

> **Transparency is important.** The following limitations should be understood before operational use.

1. **Synthetic Data Baseline**: Current R² values (0.88–0.91) are from synthetic training data where CPUE was generated from known SST-species relationships. Real-world performance with WCPFC data will be significantly lower (expected R² = 0.35–0.55). **SST feature importance (0.72–0.80) reflects the data generation process, not true ecological signal strength.**

2. **No Field Validation**: Predictions have NOT been verified against actual fishing outcomes. HSI scores are habitat suitability indices, not catch probability predictions.

3. **Internet Required**: All satellite data requires active internet connection. Offline operation uses cached data (configurable TTL).

4. **Prediction Accuracy**: R² scores should NOT be cited to buyers as real-world performance. Retrain with FAO/WCPFC data first. See `data/README_數據說明.md`.

5. **Regional Scope**: Optimized for Western Central Pacific (WCPFC Convention Area, 120°E–180°, 10°S–40°N). Performance in other oceans is untested.

---

## Scientific References

1. Deutsch, C. et al. (2015). Climate change tightens a metabolic constraint on marine habitats. *Science*, 348(6239), 1132-1135.
2. Lehodey, P. et al. (2008). A spatial ecosystem and populations dynamics model (SEAPODYM). *Progress in Oceanography*, 78(4), 304-318.
3. Belkin, I.M. & O'Reilly, J.E. (2009). An algorithm for oceanic front detection in chlorophyll and SST satellite imagery. *JGR-Oceans*, 114(C7).
4. Brewin, R.J.W. et al. (2010). A three-component model of phytoplankton size class for the Atlantic Ocean. *Ecological Modelling*, 221(11), 1472-1483.
5. Benoit-Bird, K.J. et al. (2009). Cooperative prey herding by the pelagic dolphins. *JASA*, 125(1), 125-137.
6. Stramma, L. et al. (2012). Expanding oxygen-minimum zones in the tropical oceans. *Nature Climate Change*, 2, 33-37.
7. Brody, S.R. et al. (2013). A comparison of methods for determining phytoplankton bloom initiation. *JGR-Oceans*, 118(5), 2345-2357.
8. Henson, S.A. et al. (2009). Phytoplankton-zooplankton seasonal coupling and the pelagic marine carbon cycle. *GBC*, 23(2).
9. Precioso, D. et al. (2022). TUN-AI: Tuna biomass estimation with Machine Learning models trained on oceanography and fisheries data. *Fisheries Research*, 245, 106139.
10. Schaefer, K.M. & Fuller, D.W. (2007). Vertical movement patterns of skipjack tuna in the eastern equatorial Pacific Ocean. *Fishery Bulletin*, 105(3), 301-311.
11. Isern-Fontanet, J. et al. (2006). Vortices of the Mediterranean Sea: An altimetric perspective. *JPC*, 36(1), 87-103.

---

## License

MIT License

---

## Changelog

See [CHANGELOG_v12.md](CHANGELOG_v12.md) for complete version history.

---

# 🌊 OceanMaster v13.2 — 繁體中文摘要

**台灣遠洋鮪魚漁場預測系統** — 結合衛星海洋數據、營養級級聯模型與機器學習的智慧漁場預測平台。

### 功能亮點

| 功能 | 說明 |
|------|------|
| 🛰️ 即時衛星數據 | 14+ 來源：CMEMS、NOAA、NASA、HYCOM、WOA2023 |
| 🔬 科學引擎 | HSI、代謝指數 Φ、BOA 鋒面、OW 渦旋、DVM、OMZ |
| 🧬 食物鏈預測 | 浮游植物爆發 → 浮游動物響應 → 餌料魚 → 鮪魚到達 |
| 🤖 ML 集成學習 | Stacking Ensemble (RF+XGB+LGBM) + SHAP 可解釋性 |
| 📊 專業儀表板 | Leaflet 互動地圖 + D3 食物鏈時間軸 + 暗色主題 |
| 🐟🦑 6 物種 | 黃鰭鮪、大目鮪、正鰹、長鰭鮪、北太赤魷、日本魷魚 |

### 快速開始

```bash
# Docker 部署
cp .env.example .env        # 填入 CMEMS 帳號
docker-compose -f docker-compose.production.yml up -d
# 開啟 http://localhost:8000/dashboard/v2

# 本地執行
pip install -r requirements.txt
python web_server.py
```

### 使用指南

詳細漁船操作指南請參考 [漁船使用指南.md](漁船使用指南.md)。

---

**Made with 🌊 for 台灣遠洋鮪漁船隊**
