# OceanMaster v12 — CHANGELOG

> Commercial-Grade Upgrade from v11 → v12

---

## Phase 1: Critical Bug Fixes & Cleanup

| Change | File | Description |
|--------|------|-------------|
| Version bump | `web_server.py` | API title → v12 |
| CORS middleware | `web_server.py` | Cross-origin dashboard support |
| Duplicate removal | `web_server.py` | `/api/system-health` consolidated |
| SPECIES_PARAMS unification | `config.py` | Delegates to `engine/species_params.py` SSoT |
| Duplicate source removal | `config.py` | WOA2023_SOURCES / VGPM_SOURCES dedup |

---

## Phase 2: Biology & Physics Module Enhancements

| Change | File | Description |
|--------|------|-------------|
| BOA gradient | `engine/algorithms.py` | `compute_boa_gradient()` — median→Sobel→percentile |
| Okubo-Weiss classification | `engine/eddy_detector.py` | `classify_okubo_weiss()` — vorticity/strain/background regimes |
| DVM moonlight suppression | `engine/dvm_model.py` | Cosine day/night transition, HYCOM temp fallback |
| Chl-a size fractionation | `engine/zooplankton_proxy.py` | Brewin 2010 pico/nano/micro split |
| OMZ edge enrichment | `engine/omz_model.py` | Compression + enrichment gradient effects |

---

## Phase 3: Dashboard & Frontend

| Change | File | Description |
|--------|------|-------------|
| Version unification | `dashboard.html`, `dashboard.js`, `README.md` | All references → v12 |
| Food chain panel | `dashboard.js` | SHAP modal: lunar, DVM, zoo, OMZ indicators |

---

## Phase 4: ML System

| Change | File | Description |
|--------|------|-------------|
| Feature count guard | `engine/ml/stacking_ensemble.py` | 44-feature assertion at class level |
| `safe_model_load()` | `engine/ml/stacking_ensemble.py` | 5-point validation: pickle, species, scaler, feature count, meta |
| CatBoost learner | `engine/ml/stacking_ensemble.py` | 5th base learner in stacking ensemble (optional) |
| Version header | `engine/ml/stacking_ensemble.py` | v9.0 → v12 |

---

## Phase 5: Web Server & API

| Change | File | Description |
|--------|------|-------------|
| `/api/v1/food_chain` | `web_server.py` | DVM + lunar + zoo + OMZ ecological indicators |
| `/api/v1/dvm_profile` | `web_server.py` | 24-hour DVM depth profile (cosine model) |
| Rate limiter | `web_server.py` | Sliding-window 100 req/min per IP (in-memory) |

---

## Phase 6: Infrastructure

| Change | File | Description |
|--------|------|-------------|
| CircuitBreaker | `engine/infra.py` | Three-state breaker (CLOSED→OPEN→HALF_OPEN) |
| Cache quality check | `engine/infra.py` | Reject all-NaN / >95% NaN data from cache |
| Dockerfile v12 | `Dockerfile` | Version labels, JSON healthcheck, start-period |

---

## API Endpoints Summary (New in v12)

```
GET /api/v1/food_chain?lat=25&lon=130&species=yellowfin
GET /api/v1/dvm_profile?species=yellowfin&lat=25
```

## Breaking Changes

- None. All v11 APIs remain backward-compatible.

## Dependencies (Optional)

- `catboost` — optional, enables 5th base learner in ML stacking

## Phase 7: Cross-Validation Fixes (Gemini Audit)

| Change | File | Description |
|--------|------|-------------|
| Metabolic Index Sign Error | `engine/commercial_core_v2.py` | Fixed critical bug where metabolic demand incorrectly decreased with temperature instead of increasing (1/T_K - 1/TREF_K -> 1/TREF_K - 1/T_K). |
| Deprecated Import Fix | `engine/cmems_ssh.py` | Updated legacy `data_fetcher` import to `data_fetcher_v2`. |
| Leftover Version Strings | `web_server.py` | Updated old `v10.4` startup logs to `v12`. |

---

## Phase 8: Professional Dashboard & Food Chain Prediction Engine

### New Modules

| File | Description |
|------|-------------|
| `engine/food_chain_predictor.py` | Trophic cascade timing model: Bloom → Zooplankton → Baitfish → Tuna (Henson 2009, Brody 2013, Platt 2003, Precioso 2022) |
| `engine/fish_behavior_model.py` | Tuna behavior: school residence time, DVM feeding windows, migration direction (Schaefer & Fuller 2007, Lehodey 2008) |

### New API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/v1/food_chain_timeline` | Full trophic cascade timeline for a location |
| `GET /api/v1/feeding_windows` | Optimal feeding time windows (DVM dawn/dusk/night) |
| `GET /api/v1/fish_movement` | Predicted tuna migration direction + school residence |

### Dashboard v2

| File | Description |
|------|-------------|
| `web/dashboard_v2.html` | Full-screen Leaflet map + 4-panel layout |
| `web/dashboard_v2.js` | Map layers, D3 food chain timeline, SHAP modal, time slider |
| `web/dashboard_v2.css` | Dark theme, glassmorphism, mobile-responsive |
| Route: `/dashboard/v2` | Served via `web_server.py` |

### Config Changes

| Change | File | Description |
|--------|------|-------------|
| `FOOD_CHAIN_TIMING` | `config.py` | Per-species trophic cascade lag parameters |

### Dependencies (CDN)

- Leaflet.js 1.9.4 (map)
- Leaflet.heat 0.2.0 (heatmap plugin)
- D3.js 7 (timeline visualization)
- Chart.js 4.4 (SHAP bar charts)
- Google Fonts: Inter + Noto Sans TC

---

## Phase 9: ML Training Pipeline & FAO Data Integration  `# [v12-phase9]`

### New Modules

| File | Description |
|------|-------------|
| `engine/ml/synthetic_training_data.py` | Scientifically realistic CPUE synthetic data generator (44 features, species-specific SST/food/metabolic models) |
| `engine/ml/fao_data_loader.py` | FAO/RFMO CPUE data loader (interface ready for real data) |
| `engine/ml/validation.py` | Prediction validation framework: R², RMSE, MAE, accuracy ±20%, SHAP importance, markdown report generation |
| `train_and_validate.py` | End-to-end training script: generate data → validate 44-feature alignment → train FishingStackingModel → cross-validate → save models |

### Training Results (Synthetic Data Baseline)

| Species | R² | RMSE | Acc ±20% | Acc ±30% | Training Time |
|---------|-----|------|----------|----------|---------------|
| Yellowfin | 0.8867 | 0.0756 | 57.2% | 66.8% | 72s |
| Bigeye | 0.8566 | 0.0811 | 42.0% | 56.2% | 54s |
| Albacore | 0.8986 | 0.0832 | 53.3% | 65.7% | 45s |
| Skipjack | 0.8508 | 0.0760 | 65.7% | 74.0% | 41s |

Top features: SST (0.77), season_sin, day_of_year_sin, chl_log, sst_x_chl, moon_phase

### Pipeline Integration

| Change | File | Description |
|--------|------|-------------|
| 44-feature ML loading | `main_v10_3.py` | `FishingStackingModel.safe_model_load()` with 5-point validation → 44-feature prediction |
| Hybrid fusion | `main_v10_3.py` | 40% science HSI + 60% ML CPUE norm, with Φ metabolic mask |
| Legacy fallback | `main_v10_3.py` | 12-feature `ml_system/models/` path preserved as fallback |

### Model Files

| File | Size |
|------|------|
| `models/stacking_yellowfin.pkl` | 4.2 MB |
| `models/stacking_bigeye.pkl` | 4.9 MB |
| `models/stacking_albacore.pkl` | 3.9 MB |
| `models/stacking_skipjack.pkl` | 3.1 MB |
| `models/training_report_v12.md` | Training metrics report |

---

## Phase 10: Commercialization & Deployment Readiness  `# [v12-phase10]`

### New Files

| File | Description |
|------|-------------|
| `README.md` | Complete professional rewrite — bilingual (EN + 繁中), full API docs, training results, architecture diagram, limitations disclosure |
| `Dockerfile.production` | Multi-stage build with non-root user (`appuser`), selective COPY, health check |
| `docker-compose.production.yml` | Production compose with resource limits (4GB/2CPU), structured logging, volume mounts |
| `DEPLOYMENT.md` | Full deployment guide: Docker Compose / Bare Metal + Nginx + SSL + Cron + Monitoring + Troubleshooting |
| `deploy/oceanmaster.service` | systemd unit with security hardening (NoNewPrivileges, ProtectSystem) |
| `deploy/nginx-oceanmaster.conf` | Nginx reverse proxy with rate limiting, security headers, gzip |
| `.env.example` | Environment variable template with all configurable options |

### Modified Files

| File | Change |
|------|--------|
| `.dockerignore` | Added temp files, audit artifacts, compliance dir exclusions |
| `CHANGELOG_v12.md` | Phase 10 section added |

### Deployment Target

- Production Docker image: non-root, health-checked, resource-limited
- Bare metal: systemd + Nginx + Let's Encrypt SSL
- Automated: cron for 6-hourly predictions, weekly retrain, daily cache cleanup
