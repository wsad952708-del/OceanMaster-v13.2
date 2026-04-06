"""
Retrain ML-44 (59-feature) and ML-12 (12-feature) models using real WCPFC data.

Data sources:
  - LONGLINE.CSV (156,212 rows, 1950-2018) → yellowfin, bigeye, albacore
  - PURSE_SEINE.CSV (30,025 rows, 1967-2018) → skipjack

Key improvements (v15.3-real):
  - Year filter: only use >= 2000 to reduce catchability variation
  - Temporal split: train <= 2014, test 2015-2018 (no time leakage)
  - Catchability (q) correction by 5-year period
  - SpatialBlockCV n_blocks=2 for honest spatial generalization metric

Output:
  models/stacking_{species}.pkl          → 59-feature model
  models/ml12_stacking_{species}.pkl     → 12-feature fallback model
"""
import csv
import json
import time
import logging
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    StackingRegressor, ExtraTreesRegressor,
)
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from engine.ml.stacking_ensemble import SpatialBlockCV

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("WCPFC-Train")

WCPFC_CSV = Path("data/wcpfc/LONGLINE.CSV")
PURSE_SEINE_CSV = Path("data/wcpfc/PURSE_SEINE.CSV")
MIN_YEAR = 2000  # Only use 2000+ to reduce catchability variation
TRAIN_CUTOFF_YEAR = 2014  # Train <= 2014, test >= 2015

SPECIES_COLS = {
    "yellowfin": "yft_c",
    "bigeye":    "bet_c",
    "albacore":  "alb_c",
}

# Catchability correction factors by 5-year period
# Accounts for technological improvements in gear efficiency
# Reference: Maunder & Punt 2004, WCPFC SC reports
Q_FACTORS = {
    (2000, 2004): 0.95,   # Pre-GPS depth targeting
    (2005, 2009): 1.00,   # Baseline period
    (2010, 2014): 1.03,   # GPS + deeper setting
    (2015, 2020): 1.06,   # LED lightstick + advanced targeting
}


def _get_q_factor(yy: int) -> float:
    """Get catchability correction factor for a given year."""
    for (y_lo, y_hi), q in Q_FACTORS.items():
        if y_lo <= yy <= y_hi:
            return q
    return 1.0


def parse_coord(s):
    s = s.strip().strip('"')
    if not s:
        return 0.0
    d = s[-1].upper()
    v = float(s[:-1])
    return -v if d in ('S', 'W') else v


def load_wcpfc_data(species: str, min_year: int = MIN_YEAR):
    """Load and process WCPFC LONGLINE data for a species.
    
    Applies:
    - Year filter (>= min_year)
    - Catchability (q) correction by 5-year period
    """
    col = SPECIES_COLS[species]
    rows = []
    skipped_year = 0
    with open(WCPFC_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                yy = int(row["yy"])
                if yy < min_year:
                    skipped_year += 1
                    continue
                catch = float(row.get(col, 0) or 0)
                hhooks = float(row.get("hhooks", 0) or 0)
                if catch <= 0 or hhooks <= 0:
                    continue
                lat = parse_coord(row["lat5"])
                lon = parse_coord(row["lon5"])
                mm = int(row["mm"])
                cpue_raw = catch / (hhooks / 1000.0)  # mt per 1000 hooks
                q = _get_q_factor(yy)
                cpue = cpue_raw / q  # q-corrected CPUE
                rows.append({"lat": lat, "lon": lon, "mm": mm, "yy": yy,
                             "cpue": cpue, "catch": catch, "hooks": hhooks})
            except (ValueError, KeyError):
                continue
    log.info(f"  {species}: {len(rows)} valid CPUE records from LONGLINE.CSV "
             f"(year>={min_year}, skipped {skipped_year} older records, q-corrected)")
    return rows


def load_purse_seine_data(species: str = "skipjack", min_year: int = MIN_YEAR):
    """Load and process WCPFC PURSE_SEINE data for skipjack.
    
    CPUE = total_catch (mt) / total_sets
    Applies year filter and q-correction.
    """
    if not PURSE_SEINE_CSV.exists():
        log.error(f"  PURSE_SEINE.CSV not found: {PURSE_SEINE_CSV}")
        return []
    
    # Catch columns by set type
    catch_cols = [f"skj_c_{t}" for t in ["una", "log", "dfad", "afad", "oth"]]
    sets_cols = [f"sets_{t}" for t in ["una", "log", "dfad", "afad", "oth"]]
    
    rows = []
    skipped_year = 0
    with open(PURSE_SEINE_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                yy = int(row["yy"])
                if yy < min_year:
                    skipped_year += 1
                    continue
                total_catch = sum(float(row.get(c, 0) or 0) for c in catch_cols)
                total_sets = sum(float(row.get(c, 0) or 0) for c in sets_cols)
                if total_catch <= 0 or total_sets <= 0:
                    continue
                lat = parse_coord(row["lat5"])
                lon = parse_coord(row["lon5"])
                mm = int(row["mm"])
                cpue_raw = total_catch / total_sets  # mt per set
                q = _get_q_factor(yy)
                cpue = cpue_raw / q
                rows.append({"lat": lat, "lon": lon, "mm": mm, "yy": yy,
                             "cpue": cpue, "catch": total_catch, "sets": total_sets})
            except (ValueError, KeyError):
                continue
    log.info(f"  {species}: {len(rows)} valid CPUE records from PURSE_SEINE.CSV "
             f"(year>={min_year}, skipped {skipped_year} older records, q-corrected)")
    return rows


def derive_12_features(lat, lon, mm):
    """Derive 12 climatological proxy features from location + month.

    [v13.2-audit] Improved formulas with scientific basis:
    - SST: nonlinear latitude + Kuroshio warm tongue + seasonal
    - CHL: equatorial upwelling high-CHL belt + spring bloom amplitude
    - SSH: steric height anomaly + Kuroshio SSH boost
    - DO: Henry's Law (kept — correct physics)
    - Front: derived from SST spatial gradient proxy
    - Bathy: multi-zone depth + island arc slope + known seamount chains
    """
    lat_abs = abs(lat)

    # ── SST: Nonlinear latitude gradient + Kuroshio warm tongue ──
    # Base: polynomial fit to WOA annual mean SST by latitude
    sst_base = 29.5 - 0.25 * lat_abs - 0.004 * lat_abs ** 2
    # Kuroshio warm tongue: +2~3°C along 125-145°E, 20-35°N
    kuroshio_boost = 0.0
    if 18 <= lat <= 38 and 122 <= lon <= 150:
        kuro_center_lon = 128 + 0.5 * (lat - 20)  # axis shifts east with latitude
        dist_kuro = abs(lon - kuro_center_lon)
        kuroshio_boost = 2.5 * np.exp(-(dist_kuro ** 2) / (2 * 8 ** 2))
    # Seasonal: NH warmest in Aug, amplitude increases with latitude
    seasonal_amp = 1.0 + 0.08 * lat_abs  # larger swing at high lat
    sst_seasonal = seasonal_amp * np.cos(2 * np.pi * (mm - 8) / 12)
    # South hemisphere: flip season
    if lat < 0:
        sst_seasonal = seasonal_amp * np.cos(2 * np.pi * (mm - 2) / 12)
    sst = np.clip(sst_base + kuroshio_boost + sst_seasonal, 2, 32)

    # ── CHL: Equatorial upwelling + spring bloom ──
    # Equatorial Pacific upwelling belt (high CHL at 0-10°N, 150-180°E)
    equatorial_chl = 0.0
    if lat_abs < 12 and 140 <= lon <= 180:
        equatorial_chl = 0.3 * np.exp(-((lat_abs - 5) ** 2) / (2 * 5 ** 2))
    # Base: log-linear with latitude (subtropical gyre = low CHL)
    # Subtropical gyre minimum at ~25°N
    gyre_suppression = -0.15 * np.exp(-((lat_abs - 25) ** 2) / (2 * 8 ** 2))
    chl_base = 0.08 + 0.015 * lat_abs + equatorial_chl + gyre_suppression
    # Spring bloom: amplitude stronger at higher latitudes
    bloom_amp = 0.02 + 0.012 * max(0, lat_abs - 20)
    chl_seasonal = bloom_amp * np.cos(2 * np.pi * (mm - 4) / 12)
    if lat < 0:
        chl_seasonal = bloom_amp * np.cos(2 * np.pi * (mm - 10) / 12)
    chl = np.clip(chl_base + chl_seasonal, 0.01, 5.0)

    # ── SSH: Steric height + Kuroshio dynamic height ──
    ssh_base = -0.003 * lat_abs  # decreasing with latitude
    ssh_seasonal = 0.08 * np.sin(2 * np.pi * (mm - 6) / 12)
    # Kuroshio: SSH +0.3-0.5m on the warm side
    ssh_kuroshio = 0.0
    if 18 <= lat <= 38 and 122 <= lon <= 150:
        ssh_kuroshio = 0.25 * np.exp(-((lon - (128 + 0.5 * (lat - 20))) ** 2) / (2 * 6 ** 2))
    ssh = ssh_base + ssh_seasonal + ssh_kuroshio

    # ── DO: Henry's Law — cold water holds more O2 (correct, kept) ──
    do = np.clip(8.0 - 0.1 * sst, 2.0, 8.0)

    # ── Current speed: Western boundary current + equatorial ──
    cs_base = 0.15
    # Kuroshio Current: 0.3-1.0 m/s
    if 18 <= lat <= 38 and 122 <= lon <= 150:
        kuro_axis = 128 + 0.5 * (lat - 20)
        cs_base = 0.15 + 0.45 * np.exp(-((lon - kuro_axis) ** 2) / (2 * 5 ** 2))
    # North Equatorial Current: westward 0.2-0.4 m/s at 8-18°N
    elif 8 <= lat <= 18 and lon > 130:
        cs_base = 0.25
    # Equatorial Counter Current
    elif 2 <= lat <= 8 and lon > 140:
        cs_base = 0.3
    current_speed = cs_base + 0.05 * np.sin(2 * np.pi * mm / 12)

    # ── Front strength: derived from SST meridional gradient proxy ──
    # Fronts are strongest where SST changes rapidly:
    # - Kuroshio-Oyashio transition zone (~35°N)
    # - Subtropical front (~25°N)
    # - Equatorial front (~5°N)
    front_sst_gradient = abs(-0.25 - 0.008 * lat_abs)  # dSST/dlat proxy
    # Kuroshio front enhancement
    front_kuroshio = 0.0
    if 30 <= lat <= 40 and 130 <= lon <= 170:
        front_kuroshio = 0.04 * np.exp(-((lat - 35) ** 2) / (2 * 3 ** 2))
    front_strength = np.clip(front_sst_gradient + front_kuroshio, 0.0, 0.15)

    # ── EKE proxy: Western boundary + eddy-rich zones ──
    eke_base = 0.005
    # Kuroshio extension: highest EKE in Pacific
    if 30 <= lat <= 40 and 140 <= lon <= 170:
        eke_base = 0.02 + 0.015 * np.exp(-((lat - 35) ** 2) / (2 * 3 ** 2))
    # South China Sea mesoscale eddies
    elif 10 <= lat <= 22 and 110 <= lon <= 122:
        eke_base = 0.012
    eddy = eke_base

    # ── Phi (metabolic viability): Deutsch 2015 — correct, kept ──
    phi = np.clip(3.0 + 1.5 * np.exp(-((sst - 25) / 8) ** 2), 1.0, 8.0)

    # ── Bathymetry: Multi-zone approximation ──
    bathy_depth = -4500.0  # default deep Pacific
    # Continental shelves
    if 120 <= lon <= 130 and 20 <= lat <= 35:
        bathy_depth = -200 - 1800 * min(1, max(0, (lon - 120) / 10))  # ECS shelf→deep
    elif lat_abs < 5 and 100 <= lon <= 130:
        bathy_depth = -2000.0  # Indonesian shelf edge
    elif 30 <= lat <= 40 and 130 <= lon <= 135:
        bathy_depth = -1500.0  # Japan nearshore
    elif lat_abs > 35:
        bathy_depth = -5000.0  # deep Pacific basins

    # ── Bathy slope: Island arc / trench / ridge zones ──
    # Steep slopes near: Ryukyu arc (24-30°N, 124-130°E), Mariana trench,
    # Izu-Bonin arc (25-35°N, 139-142°E)
    bathy_slope = 0.5  # gentle default
    if 24 <= lat <= 30 and 124 <= lon <= 130:
        bathy_slope = 4.0  # Ryukyu arc
    elif 25 <= lat <= 35 and 139 <= lon <= 142:
        bathy_slope = 3.5  # Izu-Bonin arc
    elif 10 <= lat <= 20 and 143 <= lon <= 148:
        bathy_slope = 5.0  # Mariana trench
    elif 120 <= lon <= 125 and 20 <= lat <= 28:
        bathy_slope = 3.0  # Taiwan east (steep)

    # ── Distance to nearest seamount (known chains) ──
    # Emperor Seamount Chain: ~30-50°N, 170°E-170°W
    # Marcus-Necker Ridge: ~20-30°N, 155-180°E
    # Kyushu-Palau Ridge: ~10-30°N, 134-136°E
    _seamounts = [
        (32, 172), (36, 171), (40, 170), (44, 168),  # Emperor chain
        (24, 160), (26, 165), (28, 170),              # Marcus-Necker
        (15, 135), (20, 135), (25, 135),              # Kyushu-Palau Ridge
        (22, 138), (25, 140),                         # Ogasawara
    ]
    dist_seamount = 500.0  # default far
    for sm_lat, sm_lon in _seamounts:
        d = np.sqrt(((lat - sm_lat) * 111) ** 2 +
                    ((lon - sm_lon) * 111 * np.cos(np.radians(lat))) ** 2)
        if d < dist_seamount:
            dist_seamount = d

    # ── Distance to shelf break ──
    # Continental shelf break: ~200m isobath
    # Approximate: closer near Asia continent (lon < 130)
    shelf_break_lon = 122 + 0.3 * (lat - 20) if 15 <= lat <= 40 else 130
    dist_shelf_break = max(0, abs(lon - shelf_break_lon) * 111 *
                           np.cos(np.radians(lat)))
    if lon < shelf_break_lon:
        dist_shelf_break = max(0, (shelf_break_lon - lon) * 50)  # on shelf, closer

    return [sst, chl, ssh, do, current_speed, front_strength,
            eddy, phi, bathy_depth, bathy_slope, dist_seamount, dist_shelf_break]


def _derive_t100_features(sst, lat):
    """[海鷹] Derive T100/gradient features from climatological relationship.

    T100 (100m water temperature) decreases faster with latitude than SST
    because the thermocline is shallower at higher latitudes.
    Source: WOA climatology patterns.
    """
    lat_abs = abs(lat)
    # T100: tropical offset ~8°C, temperate offset ~12°C
    t100_offset = 8 + 0.12 * lat_abs + 0.002 * lat_abs ** 2
    t100 = max(4.0, sst - t100_offset)
    # Gradient: ΔT/100m (°C/m) — steeper in tropics
    gradient_strength = np.clip((sst - t100) / 100.0, 0.01, 0.35)
    delta_t = sst - t100
    # CHL lag: use current CHL value as proxy (no temporal data available)
    # Add latitude-dependent anomaly to simulate 15-day lag effect
    chl_lag15d_log = -0.5 + 0.008 * lat_abs
    return t100, gradient_strength, delta_t, chl_lag15d_log


def _calc_moon_phase(yy, mm):
    """Calculate approximate moon phase from year/month.

    Uses known new moon reference (2000-01-06) + synodic month.
    Returns illumination: 0=new moon, 0.5=first/last quarter, 1.0=full moon.
    """
    # Approximate mid-month date
    from datetime import date
    ref = date(2000, 1, 6)  # known new moon
    target = date(int(yy), int(mm), 15)
    days = (target - ref).days
    cycle = 29.53058867
    phase = (days % cycle) / cycle
    # illumination: 0 at new moon (phase=0), 1 at full moon (phase=0.5)
    illumination = 0.5 * (1 - np.cos(2 * np.pi * phase))
    return illumination


def derive_features(lat, lon, mm, yy):
    """Derive features matching FeatureEngineer.FEATURE_NAMES.

    [v13.2-audit] Major fixes:
    - All features now have genuine spatial/temporal variance
    - Removed constants (was: current_dir=23.2°, moon_phase=0.5, etc.)
    - Added biology-based hidden features (moonlight DVM, post-storm bloom)
    - FTLE derived from current×front interaction (not sin(lon))
    - Front distances derived from SST proxy gradient structure
    """
    f12 = derive_12_features(lat, lon, mm)
    sst, chl, ssh, do_val, cs = f12[0], f12[1], f12[2], f12[3], f12[4]
    front, eddy, phi = f12[5], f12[6], f12[7]
    bathy_d, bathy_s, dist_sm, dist_sb = f12[8], f12[9], f12[10], f12[11]

    lat_abs = abs(lat)
    doy = int(mm * 30.4)  # approx day of year

    # [海鷹] T100/gradient features
    t100_v, grad_v, dt_v, chl_lag_v = _derive_t100_features(sst, lat)

    # ── Derived physical features ──

    # Current direction: varies by major current system
    # Kuroshio: NE (~45°), NEC: W (~270°), ECC: E (~90°), Oyashio: SW (~225°)
    if 18 <= lat <= 38 and 122 <= lon <= 150:
        current_dir = 45 + 10 * np.sin(np.radians(lat * 2))  # Kuroshio NE
    elif 8 <= lat <= 18 and lon > 130:
        current_dir = 270  # North Equatorial Current (westward)
    elif 2 <= lat <= 8 and lon > 140:
        current_dir = 90   # Equatorial Counter Current (eastward)
    elif lat > 38 and 140 <= lon <= 170:
        current_dir = 225  # Oyashio (SW)
    else:
        current_dir = 180 + 30 * np.sin(np.radians(lon))  # variable

    # SST gradient: proxy from meridional gradient + front strength
    sst_gradient = front * 8.0 + 0.01 * lat_abs

    # CHL gradient: derived from CHL spatial variability proxy
    chl_gradient = np.clip(chl * 0.15 + 0.005 * lat_abs, 0.001, 0.5)

    # FTLE: Lagrangian coherent structure proxy
    # FTLE ∝ current_speed × grad_front (stirring + front = material lines)
    # Higher near strong currents with nearby fronts
    ftle_val = np.clip(cs * front * 50 + eddy * 0.5 + 0.003, 0.001, 0.1)
    ftle_ridge = np.clip(ftle_val * 0.7 + front * 0.3 + 0.001, 0.001, 0.08)

    # Thermocline: seasonal (correct, kept)
    thermo = 80 + 40 * np.cos(2 * np.pi * (mm - 3) / 12)
    d20 = thermo + 15 + 5 * np.cos(2 * np.pi * (mm - 4) / 12)
    mld_val = 50 + 30 * np.cos(2 * np.pi * (mm - 1) / 12)

    # Distance to front: inversely related to front_strength
    # Strong front → you're near a front → small distance
    dist_to_front = np.clip(100 * (1 - front / 0.15), 5, 300)

    # Distance to eddy: inversely related to EKE
    dist_to_eddy = np.clip(200 * (1 - eddy / 0.035), 10, 500)

    # Moon phase from year/month (not constant anymore!)
    moon_phase = _calc_moon_phase(yy, mm)

    # Kuroshio distance (correct, kept)
    kuroshio_dist = (max(0, np.sqrt((lat - 27) ** 2 + (lon - 135) ** 2) - 5)
                     if 15 <= lat <= 40 and 120 <= lon <= 160 else 50.0)

    # ── Time features (correct, kept) ──
    season_sin = np.sin(2 * np.pi * mm / 12)
    season_cos = np.cos(2 * np.pi * mm / 12)
    doy_sin = np.sin(2 * np.pi * doy / 365)
    doy_cos = np.cos(2 * np.pi * doy / 365)

    # ── Interaction features ──
    sst_x_chl = sst * np.log10(max(chl, 0.001))
    front_x_ftle = front * ftle_val
    ssh_x_thermo = ssh * thermo

    # SST local std: proxy from 2nd derivative of SST field (curvature)
    # High curvature = high spatial variability
    sst_local_std = np.clip(abs(front * 10) + eddy * 20 + 0.1, 0.05, 3.0)

    # CHL local mean: current CHL is already local proxy
    chl_local_mean = chl

    # Current local mean: slight smoothing proxy
    current_local_mean = cs * 0.95

    # AIS/VIIRS: truly unavailable in historical data, keep as 0
    ais_density = 0.0
    viirs_density = 0.0

    # Lunar CPUE modifier: derived from actual moon phase
    # New moon → better for longline (~1.08×), full moon → worse (~0.92×)
    # Based on Bigelow et al. (1999) and Poisson et al. (2010)
    lunar_cpue_mod = 1.0 + 0.08 * np.cos(2 * np.pi * moon_phase)

    # Zooplankton index: CHL → phytoplankton → zooplankton (food chain, correct)
    zoo_index = 0.3 + 0.2 * chl

    # Spawning season (correct, kept)
    spawning = 1.0 if mm in (4, 5, 6, 7, 8) else 0.0

    # ENSO ONI: approximate from year
    # Major El Niños: 1997-98, 2002-03, 2009-10, 2015-16
    _el_nino_years = {1997: 2.0, 1998: 1.0, 2002: 1.0, 2003: 0.5,
                      2009: 1.0, 2010: -1.0, 2015: 2.0, 2016: 0.5,
                      2004: -0.5, 2005: -0.5, 2007: -1.0, 2008: -1.0,
                      2011: -1.0, 2012: -0.5}
    enso_oni = _el_nino_years.get(int(yy), 0.0)

    # OMZ compression (correct, kept)
    omz_compress = max(0, 1.0 - do_val / 4.0)

    # Eddy enrichment: EKE × CHL interaction (eddy brings nutrients)
    eddy_enrich = eddy * chl * 100

    # DVM accessible depth (correct, kept)
    dvm_depth = min(300, thermo + 50)

    # Salinity front: Kuroshio/Oyashio boundary (35°N) and river plumes
    sal_front = 0.0
    if 32 <= lat <= 38 and 140 <= lon <= 170:
        sal_front = 0.08 * np.exp(-((lat - 35) ** 2) / (2 * 2 ** 2))
    elif 120 <= lon <= 124 and 20 <= lat <= 28:
        sal_front = 0.05  # Taiwan freshwater influence

    # Productivity front (correct concept, now uses fixed front)
    prod_front = front * chl * 10

    # Habitat compression ratio (correct, kept)
    hab_compress = mld_val / max(thermo, 1)

    # Taiwan strait flag (correct, kept)
    ts_flag = 1.0 if (23 <= lat <= 26 and 119 <= lon <= 122) else 0.0

    # ── NEW: Biology-based hidden features ──

    # [NEW-1] Moonlight DVM suppression factor
    # Benoit-Bird et al. 2009, Drazen et al. 2011:
    # Full moon suppresses micronekton surfacing → 表層餌料減少 30-50%
    # → 延繩釣深放鉤反而更好; 圍網/集魚燈效果差
    # Factor: 1.0 = no suppression (new moon), 0.5 = max suppression (full moon)
    moonlight_dvm_suppression = 1.0 - 0.4 * moon_phase

    # [NEW-2] Post-storm CHL bloom potential
    # Zhao et al. 2017, Lin 2012: 颱風攪拌混合層 → 深層營養鹽上表面
    # → 3-7天後浮游植物爆發 → 7-14天後魚群聚集
    # Proxy: seasonal typhoon activity × ocean thermal energy
    if mm in (7, 8, 9, 10) and 15 <= lat <= 30 and 120 <= lon <= 160:
        typhoon_season_strength = {7: 0.4, 8: 0.8, 9: 1.0, 10: 0.6}[mm]
        ocean_heat = max(0, (sst - 26) / 4)
        post_storm_bloom = np.clip(typhoon_season_strength * ocean_heat * 0.5, 0, 1)
    else:
        post_storm_bloom = 0.0

    # [NEW-3] Seamount × Current interaction (Taylor column effect)
    # Pitcher et al. 2007, Rowden et al. 2010:
    # 海流流過海底山 → 形成 Taylor column 渦旋 → 營養鹽上湧 → 生物量 3-5 倍
    # 效果大小 ∝ 1/距離 × 流速。越近海底山 + 越強海流 = 越強上湧
    seamount_interaction = np.clip(
        max(0, 1 - dist_sm / 300) * cs * 8, 0, 1
    )

    # [NEW-4] Prey thermocline trap index
    # Mann & Lazier 2006: 溫躍層是「獵物天花板」
    # 溫度梯度越陡 → 浮游動物越集中在溫躍層附近 → 掠食者聚集
    # 淺 MLD + 強梯度 = 獵物最集中（容易被捕食）
    prey_trap = np.clip(
        grad_v * 50 * min(1, 80 / max(mld_val, 10)), 0, 1
    )

    # [NEW-5] Tidal mixing index (spring/neap cycle)
    # Simpson & Hunter 1974, Sharples 2008:
    # 大潮（新月+滿月）→ 潮汐混合增強 → 營養鹽上湧 → 餌料增加
    # 與月光效應方向相同但機制不同（潮汐力 vs 光照）
    # cos(2π*moon) = 1 at new moon and full moon (both spring tides)
    tidal_mixing = 0.5 + 0.5 * np.cos(4 * np.pi * moon_phase)

    # [NEW-6] SST seasonal rate of change (dSST/dt)
    # Podesta et al. 1993, Zainuddin et al. 2006:
    # 急速升溫/降溫 → 觸發魚群遷移。漁船應追蹤溫變前緣。
    # 用 SST 對月份的微分近似
    seasonal_amp_val = 1.0 + 0.08 * lat_abs
    sst_rate = seasonal_amp_val * (2 * np.pi / 12) * np.sin(2 * np.pi * (mm - 8) / 12)
    if lat < 0:
        sst_rate = seasonal_amp_val * (2 * np.pi / 12) * np.sin(2 * np.pi * (mm - 2) / 12)

    # [NEW-7] Convergence zone proxy
    # Bakun 2006, Olson et al. 1994:
    # 海流收斂帶 = 餌料、碎屑、浮游生物聚集的主要物理機制
    # 收斂強度 ∝ 海流剪切 (current speed × front_strength)
    # 鋒面附近 + 強海流 = 收斂帶最強
    convergence_proxy = np.clip(cs * front * 30, 0, 1)

    # [NEW-8] Dawn/dusk twilight feeding window (hours)
    # 大多數鮪魚在晨昏時段覓食最活躍 (hook_depth.py, fish_behavior_model.py)
    # 窗口長度取決於太陽仰角變化率，隨緯度和季節變化：
    # - 赤道：快速日出日落，晨昏窗口短 (~1.5 hr)
    # - 高緯夏季：緩慢日出日落，晨昏窗口長 (~3 hr)
    # 簡化公式：twilight ∝ 1 + tan(declination) * tan(latitude)
    solar_decl = 23.44 * np.sin(2 * np.pi * (doy - 81) / 365)
    twilight_factor = 1 + 0.3 * abs(np.tan(np.radians(solar_decl)) *
                                    np.tan(np.radians(lat)))
    dawn_twilight_hrs = np.clip(1.5 * twilight_factor, 1.0, 4.0)

    # ── Bathy roughness proxy (sync with FeatureEngineer: BUG-3 fix) ──
    # Rough seafloor = more turbulent upwelling = more nutrients
    # Near seamounts/ridges/trenches → rougher
    bathy_roughness = np.clip(bathy_s * 0.3 + seamount_interaction * 0.5, 0, 3.0)

    features = np.array([
        sst,                            # 0: sst
        np.log10(max(chl, 0.001)),      # 1: chl_log
        ssh,                            # 2: ssh
        cs,                             # 3: current_speed
        current_dir,                    # 4: current_dir
        sst_gradient,                   # 5: sst_gradient
        chl_gradient,                   # 6: chl_gradient
        front,                          # 7: front_strength
        ftle_val,                       # 8: ftle
        ftle_ridge,                     # 9: ftle_ridge
        thermo,                         # 10: thermocline_depth
        d20,                            # 11: d20_depth
        mld_val,                        # 12: mld
        dist_to_front,                  # 13: dist_to_front
        dist_to_eddy,                   # 14: dist_to_eddy
        dist_sm,                        # 15: dist_to_seamount
        dist_sb,                        # 16: dist_to_shelf_break
        bathy_d,                        # 17: bathy_depth
        bathy_s,                        # 18: bathy_slope
        bathy_roughness,                # 19: bathy_roughness [BUG-3 sync]
        moon_phase,                     # 20: moon_phase
        season_sin,                     # 21: season_sin
        season_cos,                     # 22: season_cos
        doy_sin,                        # 23: day_of_year_sin
        doy_cos,                        # 24: day_of_year_cos
        0.0,                            # 25: sst_7d_trend (unavailable)
        chl * 0.1 * np.sin(2 * np.pi * (mm - 4) / 12),  # 26: chl_30d_anomaly
        sst_x_chl,                      # 27: sst_x_chl
        front_x_ftle,                   # 28: front_x_ftle
        ssh_x_thermo,                   # 29: ssh_x_thermo
        sst_local_std,                  # 30: sst_local_std
        chl_local_mean,                 # 31: chl_local_mean
        current_local_mean,             # 32: current_local_mean
        ais_density,                    # 33: ais_fishing_density
        viirs_density,                  # 34: viirs_light_density
        lunar_cpue_mod,                 # 35: lunar_cpue_modifier
        zoo_index,                      # 36: zooplankton_index
        spawning,                       # 37: spawning_season
        enso_oni,                       # 38: enso_oni
        omz_compress,                   # 39: omz_compression
        eddy_enrich,                    # 40: eddy_enrichment
        dvm_depth,                      # 41: dvm_accessible_depth
        sal_front,                      # 42: salinity_front_strength
        prod_front,                     # 43: productivity_front
        hab_compress,                   # 44: habitat_compression_ratio
        kuroshio_dist,                  # 45: kuroshio_distance
        ts_flag,                        # 46: taiwan_strait_flag
        t100_v,                         # 47: t100
        grad_v,                         # 48: gradient_strength
        dt_v,                           # 49: delta_t_surface_100
        chl_lag_v,                      # 50: chl_lag15d
        # ── v13.2-audit 新增 8 個科學衍生特徵 ──
        moonlight_dvm_suppression,      # 51: moonlight_dvm_suppression
        post_storm_bloom,               # 52: post_storm_chl_bloom
        seamount_interaction,           # 53: seamount_current_interaction
        prey_trap,                      # 54: prey_thermocline_trap
        tidal_mixing,                   # 55: tidal_mixing_index
        sst_rate,                       # 56: sst_seasonal_derivative
        convergence_proxy,              # 57: convergence_proxy
        dawn_twilight_hrs,              # 58: dawn_twilight_hours
    ], dtype=np.float64)

    return features, None  # feature_names not needed


def build_stacking_model():
    """Build a StackingRegressor ensemble."""
    estimators = [
        ("rf", RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=10, n_jobs=-1, random_state=42)),
        ("gbr", GradientBoostingRegressor(n_estimators=150, max_depth=6, learning_rate=0.05, subsample=0.8, random_state=42)),
        ("et", ExtraTreesRegressor(n_estimators=200, max_depth=12, min_samples_leaf=10, n_jobs=-1, random_state=42)),
    ]
    if HAS_XGB:
        estimators.append(("xgb", xgb.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, n_jobs=-1, random_state=42)))
    if HAS_LGB:
        estimators.append(("lgb", lgb.LGBMRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, n_jobs=-1, random_state=42, verbose=-1)))

    return StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]),
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
    )


def train_species(species: str, n_feat: int = 12):
    """Train one species model with temporal split and q-correction."""
    log.info(f"\n{'='*60}")
    log.info(f"  Training {n_feat}-feature model: {species.upper()}")
    log.info(f"{'='*60}")
    t0 = time.time()

    # Load WCPFC data (year >= 2000, q-corrected)
    if species == "skipjack":
        rows = load_purse_seine_data(species)
        data_source = f"WCPFC_PURSE_SEINE_{MIN_YEAR}-2018"
        cpue_unit = "mt_per_set"
    else:
        rows = load_wcpfc_data(species)
        data_source = f"WCPFC_LONGLINE_{MIN_YEAR}-2018"
        cpue_unit = "mt_per_1000_hooks"
    
    if len(rows) < 100:
        log.warning(f"  Too few records ({len(rows)}), skipping")
        return None

    # Subsample if too many rows (speed)
    if len(rows) > 30000:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(rows), 30000, replace=False)
        rows = [rows[i] for i in idx]
        log.info(f"  Subsampled to {len(rows)} records")

    # Build feature matrix
    if n_feat == 12:
        X = np.array([derive_12_features(r["lat"], r["lon"], r["mm"]) for r in rows])
    else:
        result = [derive_features(r["lat"], r["lon"], r["mm"], r["yy"]) for r in rows]
        X = np.array([r[0] for r in result])
        feature_names = result[0][1]

    y = np.array([r["cpue"] for r in rows])
    years_all = np.array([r["yy"] for r in rows])
    lats_all = np.array([r["lat"] for r in rows])
    lons_all = np.array([r["lon"] for r in rows])

    # Clean outliers: remove top/bottom 1%
    p1, p99 = np.percentile(y, [1, 99])
    mask = (y >= p1) & (y <= p99)
    X, y = X[mask], y[mask]
    years_all = years_all[mask]
    lats_all, lons_all = lats_all[mask], lons_all[mask]
    log.info(f"  After outlier removal: {len(y)} samples, CPUE range [{y.min():.2f}, {y.max():.2f}]")

    # === TEMPORAL SPLIT: train <= 2014, test >= 2015 ===
    mask_train = years_all <= TRAIN_CUTOFF_YEAR
    mask_test = years_all > TRAIN_CUTOFF_YEAR
    
    if mask_test.sum() < 50:
        log.warning(f"  Too few test samples ({mask_test.sum()}), falling back to 80/20 temporal")
        # fallback: use last 20% of years
        sorted_years = np.sort(np.unique(years_all))
        cutoff_idx = int(len(sorted_years) * 0.8)
        cutoff_year = sorted_years[cutoff_idx]
        mask_train = years_all <= cutoff_year
        mask_test = years_all > cutoff_year
        log.info(f"  Temporal fallback cutoff: {cutoff_year}")
    
    X_train, X_test = X[mask_train], X[mask_test]
    y_train, y_test = y[mask_train], y[mask_test]
    lat_train, lat_test = lats_all[mask_train], lats_all[mask_test]
    lon_train, lon_test = lons_all[mask_train], lons_all[mask_test]
    
    log.info(f"  Temporal split: train={len(y_train)} (<={TRAIN_CUTOFF_YEAR}), "
             f"test={len(y_test)} (>{TRAIN_CUTOFF_YEAR})")
    log.info(f"  Train years: {int(years_all[mask_train].min())}-{int(years_all[mask_train].max())}")
    log.info(f"  Test years:  {int(years_all[mask_test].min())}-{int(years_all[mask_test].max())}")

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_train_s = np.nan_to_num(X_train_s, nan=0, posinf=0, neginf=0)
    X_test_s = np.nan_to_num(X_test_s, nan=0, posinf=0, neginf=0)

    # Train
    model = build_stacking_model()
    log.info(f"  Fitting stacking ensemble...")
    model.fit(X_train_s, y_train)

    # Evaluate (holdout)
    y_pred = model.predict(X_test_s)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    log.info(f"  Holdout R2={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")

    # Spatial Block CV evaluation (honest R²)
    # Use n_blocks=2 for larger spatial blocks (more honest with coarse 5° data)
    X_all_s = scaler.transform(np.nan_to_num(X, nan=0, posinf=0, neginf=0))
    spatial_cv = SpatialBlockCV(n_blocks=2, buffer_km=100)
    sp_result = spatial_cv.cross_val_score(model, X_all_s, y, lats_all, lons_all)
    log.info(f"  SpatialBlockCV R2={sp_result['r2_mean']:.4f} +/- {sp_result['r2_std']:.4f} "
             f"({sp_result['n_folds']} folds)")

    # Save
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    if n_feat == 12:
        out_path = model_dir / f"ml12_stacking_{species}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump({"model": model, "scaler": scaler, "species": species}, f)
    else:
        out_path = model_dir / f"stacking_{species}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump({"model": model, "scaler": scaler, "species": species}, f)

    log.info(f"  Saved: {out_path} ({out_path.stat().st_size // 1024} KB)")

    # HMAC sign
    try:
        from engine.ml.stacking_ensemble import _sign_model_file
        _sign_model_file(out_path)
        log.info(f"  HMAC signed")
    except Exception as e:
        log.debug(f"[降級] train_wcpfc.py: {e}")

    # Meta
    meta = {
        "species": species, "n_features": n_feat,
        "data_source": data_source,
        "split_method": f"temporal (train<={TRAIN_CUTOFF_YEAR}, test>{TRAIN_CUTOFF_YEAR})",
        "q_correction": True,
        "min_year": MIN_YEAR,
        "n_train": int(X_train.shape[0]), "n_test": int(X_test.shape[0]),
        "r2_holdout_temporal": float(r2), "r2_spatial_cv": float(sp_result['r2_mean']),
        "r2_spatial_cv_std": float(sp_result['r2_std']),
        "mae": float(mae), "rmse": float(rmse),
        "cpue_unit": cpue_unit,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = out_path.with_suffix(".pkl").with_name(out_path.stem + "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    log.info(f"  Done in {elapsed:.1f}s")
    return meta


if __name__ == "__main__":
    # All 4 species with real WCPFC data (no synthetic training)
    species_list = ["yellowfin", "bigeye", "albacore", "skipjack"]
    all_results = {"ml12": {}, "ml44": {}}

    log.info("\n" + "=" * 70)
    log.info("  OceanMaster v13.2 — Real Data Training (No Synthetic)")
    log.info(f"  Data filter: year >= {MIN_YEAR}, q-corrected")
    log.info(f"  Split: temporal (train <= {TRAIN_CUTOFF_YEAR}, test > {TRAIN_CUTOFF_YEAR})")
    log.info("=" * 70)

    # Train ML-12
    log.info("\n" + "#" * 60)
    log.info("#  ML-12 (12-feature) Training with WCPFC Real Data")
    log.info("#" * 60)
    for sp in species_list:
        meta = train_species(sp, n_feat=12)
        if meta:
            all_results["ml12"][sp] = meta

    # Train ML-44
    log.info("\n" + "#" * 60)
    log.info("#  ML-44 (59-feature) Training with WCPFC Real Data")
    log.info("#" * 60)
    for sp in species_list:
        meta = train_species(sp, n_feat=59)
        if meta:
            all_results["ml44"][sp] = meta

    # Summary
    print("\n" + "=" * 100)
    print(f"{'Model':<8} {'Species':<12} {'R2_temp':>8} {'R2_spCV':>8} {'MAE':>10} {'RMSE':>10} {'N_train':>10} {'N_test':>10}")
    print("-" * 100)
    for model_type in ["ml12", "ml44"]:
        for sp, m in all_results[model_type].items():
            r2_t = m.get('r2_holdout_temporal', m.get('r2_holdout', 0))
            print(f"{model_type:<8} {sp:<12} {r2_t:>8.4f} {m['r2_spatial_cv']:>8.4f} "
                  f"{m['mae']:>10.4f} {m['rmse']:>10.4f} {m['n_train']:>10} {m['n_test']:>10}")
    print("=" * 100)
    print(f"\nSplit: temporal (train <= {TRAIN_CUTOFF_YEAR}, test > {TRAIN_CUTOFF_YEAR})")
    print(f"Data: year >= {MIN_YEAR}, q-corrected")
    print(f"SpatialBlockCV: n_blocks=2, buffer=100km")
