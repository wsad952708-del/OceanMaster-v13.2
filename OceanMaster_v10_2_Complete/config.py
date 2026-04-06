"""
OceanMaster v12 — 全域設定  # [v12-fix]
==============================
所有數據源 URL（已查證可用）、物種參數、排程設定
v10 新增: WOA2023 DO、Oregon State VGPM、ETOPO1、CMEMS BGC
v12 修正: 移除重複定義，SPECIES_PARAMS 統一為 species_params.py 代理
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class AreaConfig:
    lat_min: float = -10.0
    lat_max: float = 40.0
    lon_min: float = 120.0
    lon_max: float = 180.0
    resolution: float = 0.25

ERDDAP_SOURCES = {
    "sst_oisst": {
        "url": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg",
        "var": "sst", "desc": "NOAA OISST v2.1 (0.25°, 每日)",
    },
    "sst_mur": {
        "url": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41",
        "var": "analysed_sst", "desc": "JPL MUR SST L4 (0.01°)",
    },
    "chlorophyll": {
        "url": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMH1chla1day",
        "var": "chlorophyll", "desc": "MODIS Aqua Chl-a (4km, 每日)",
    },
    "wind_u": {
        "url": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdQMwindmday",
        "var": "x_wind", "desc": "QuikSCAT 月均風 U",
    },
    "etopo1": {
        "url": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180",
        "var": "altitude", "desc": "ETOPO1 海底地形 (1弧分)",
    },
}

CMEMS_SOURCES = {
    "ocean_physics": {
        "dataset_id": "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        "variables": ["thetao", "so", "uo", "vo", "zos"],
        "desc": "Copernicus 全球海洋物理",
    },
    "ocean_bgc": {
        "dataset_id": "cmems_mod_glo_bgc_anfc_0.25deg_P1D-m",
        "variables": ["chl", "o2", "no3", "nppv", "phyc"],
        "desc": "Copernicus 全球海洋生化（DO/NPP）",
    },
}

HYCOM_SOURCES = {
    "hycom_3d": {
        "url": "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0",
        "variables": ["water_temp", "salinity", "water_u", "water_v"],
        "depths": [0, 50, 100, 150, 200, 300, 500],
        "desc": "HYCOM GLBv0.08 3D (0.08°)",
    },
}

OPEN_METEO_URL = "https://marine-api.open-meteo.com/v1/marine"

GFW_API = {
    "base_url": "https://gateway.api.globalfishingwatch.org/v3",
    "endpoints": {
        "vessels": "/vessels/search",
        "events": "/events",
        "map_tiles": "/4wings/tile/heatmap/{z}/{x}/{y}",
    },
}

# ─── v10 數據源 (已查證 URL, v12 移除重複) ────────────────
# [v12-fix] 原有兩組 WOA2023_SOURCES / VGPM_SOURCES 定義，現統一為一組

WOA2023_SOURCES = {
    "dissolved_oxygen": {
        "thredds": "https://www.ncei.noaa.gov/thredds-ocean/dodsC/ncei/woa/oxygen/all/1.00/woa23_all_o{month:02d}_01.nc",
        "variable": "o_an",
        "unit": "μmol/kg",
        "convert_to_mll": 1/44.66,
        "desc": "WOA2023 溶解氧氣候態 (1°, 102層, 月平均)",
    },
    "nutrients": {
        "phosphate": "https://www.ncei.noaa.gov/thredds-ocean/dodsC/ncei/woa/phosphate/all/1.00/woa23_all_p00_01.nc",
        "nitrate": "https://www.ncei.noaa.gov/thredds-ocean/dodsC/ncei/woa/nitrate/all/1.00/woa23_all_n00_01.nc",
        "desc": "WOA2023 營養鹽氣候態",
    },
}

VGPM_SOURCES = {
    "standard_vgpm": {
        "base_url": "https://orca.science.oregonstate.edu/1080.by.2160.monthly.hdf.vgpm.m.chl.m.sst.php",
        "xyz_url": "https://orca.science.oregonstate.edu/1080.by.2160.monthly.xyz.vgpm.m.chl.m.sst4.php",
        "resolution": "1/6 degree (1080x2160)",
        "unit": "mgC/m²/day",
        "desc": "Oregon State VGPM 標準產品 (Behrenfeld & Falkowski 1997)",
        "citation": "Behrenfeld & Falkowski 1997, L&O 42:1-20",
    },
    "eppley_vgpm": {
        "base_url": "https://orca.science.oregonstate.edu/1080.by.2160.8day.hdf.eppley.m.chl.m.sst.php",
        "desc": "Eppley-VGPM (溫度依賴性增強版)",
    },
    "cbpm2": {
        "base_url": "https://orca.science.oregonstate.edu/1080.by.2160.monthly.hdf.cbpm2.m.php",
        "desc": "CbPM2 碳基生產力模型 (光譜解析+深度解析)",
    },
}

ETOPO_SOURCES = {
    "etopo1_erddap": {
        "url": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180",
        "variable": "altitude",
        "resolution": "1 arc-minute (0.0167°)",
        "desc": "ETOPO1 全球地形 via ERDDAP (可子集下載)",
    },
    "etopo2022_erddap": {
        "url": "https://oceanwatch.pifsc.noaa.gov/erddap/griddap/ETOPO_2022_v1_15s",
        "variable": "z",
        "resolution": "15 arc-second",
        "desc": "ETOPO 2022 (更新版, 15弧秒)",
    },
}

VIIRS_SOURCES = {
    "nightfire": {
        "url": "https://eogdata.mines.edu/nighttime_light/nightly/rade9d",
        "desc": "VIIRS Nightfire 夜間光源",
    },
    "dnb_gibs": {
        "url": "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/"
               "VIIRS_SNPP_DayNightBand_At_Sensor_Radiance/default/"
               "{date}/500m/{z}/{y}/{x}.png",
        "desc": "NASA GIBS VIIRS DNB",
    },
}

# ═══════════════════════════════════════════════════════
# [v12-fix] SPECIES_PARAMS — 從 species_params.py 自動建構
# species_params.py 是單一真實來源 (SSoT); 此處僅補充 HSI 特有欄位
# ═══════════════════════════════════════════════════════

# HSI-specific fields not in species_params.py
_HSI_OVERRIDES = {
    "skipjack": {
        "sst_optimal": 29.0, "sst_sigma": 2.5, "sst_range": (26.0, 32.0),
        "chl_optimal": 0.25, "chl_sigma": 0.15, "chl_range": (0.05, 0.5),
        "ssh_range": (0.48, 0.58), "depth_range": (0, 200),
        "do_min": 3.5, "do_preferred": 4.5,
        "salinity_range": (34.0, 36.0), "salinity_optimal": 35.0,
        "front_attraction": 0.8, "ftle_weight": 0.6,
        "moon_sensitivity": 0.3, "forage_layer": "epipelagic",
    },
    "yellowfin": {
        "sst_optimal": 28.0, "sst_sigma": 2.0, "sst_range": (24.0, 31.0),
        "chl_optimal": 0.2, "chl_sigma": 0.1, "chl_range": (0.05, 0.4),
        "ssh_range": (-0.05, 0.15), "depth_range": (0, 400),
        "do_min": 3.0, "do_preferred": 4.0,
        "salinity_range": (33.5, 36.5), "salinity_optimal": 35.0,
        "front_attraction": 0.7, "ftle_weight": 0.5,
        "thermocline_importance": 0.8, "forage_layer": "mixed",
    },
    "bigeye": {
        "sst_optimal": 21.0, "sst_sigma": 3.0, "sst_range": (15.0, 29.0),
        "sst_surface_range": (26.0, 30.0),
        "chl_optimal": 0.15, "chl_sigma": 0.1, "chl_range": (0.05, 0.3),
        "ssh_range": (-0.10, 0.03), "depth_range": (100, 500),
        "do_min": 1.5, "do_preferred": 3.0,
        "salinity_range": (33.0, 36.5), "salinity_optimal": 34.8,
        "thermocline_importance": 1.0, "forage_layer": "mesopelagic",
        "hook_depth_tip": "鉤子下在溫躍層下緣（200-350m）",
    },
    "albacore": {
        "sst_optimal": 19.0, "sst_sigma": 2.0, "sst_range": (15.0, 22.0),
        "chl_optimal": 0.15, "chl_sigma": 0.1, "chl_range": (0.05, 0.3),
        "ssh_range": (0.24, 0.84), "depth_range": (50, 300),
        "do_min": 2.5, "do_preferred": 3.5,
        "salinity_range": (33.5, 36.0), "salinity_optimal": 34.5,
        "eddy_attraction": 0.7, "forage_layer": "mixed",
    },
    "squid_todarodes": {
        "name_zh": "日本魷魚", "name_en": "Japanese Common Squid",
        "sst_optimal": 17.0, "sst_sigma": 3.0, "sst_range": (12.0, 22.0),
        "chl_optimal": 0.8, "chl_sigma": 0.5, "chl_range": (0.2, 3.0),
        "depth_range": (0, 200), "do_min": 2.0, "do_preferred": 3.0,
        "moon_sensitivity": 0.9, "viirs_weight": 1.0,
        "forage_layer": "epipelagic",
    },
    "squid_ommastrephes": {
        "name_zh": "北太赤魷", "name_en": "Neon Flying Squid",
        "sst_optimal": 18.0, "sst_sigma": 3.0, "sst_range": (14.0, 24.0),
        "chl_optimal": 0.3, "chl_sigma": 0.2, "chl_range": (0.1, 1.0),
        "ssh_range": (-0.05, 0.05), "depth_range": (0, 300),
        "do_min": 2.0, "do_preferred": 3.0,
        "moon_sensitivity": 0.8, "viirs_weight": 0.9,
        "forage_layer": "mixed",
    },
}


def _build_species_params() -> Dict:
    """[v12-fix] 從 species_params.py 合併 HSI 欄位，建構統一 SPECIES_PARAMS"""
    try:
        from engine.species_params import SPECIES
        merged = {}
        # 先複製 species_params.py 中的核心參數
        for key, bio in SPECIES.items():
            merged[key] = dict(bio)
            # 覆蓋/補充 HSI 特有欄位
            if key in _HSI_OVERRIDES:
                merged[key].update(_HSI_OVERRIDES[key])
        # 補充 species_params.py 不含的物種 (魷魚等)
        for key, hsi in _HSI_OVERRIDES.items():
            if key not in merged:
                merged[key] = dict(hsi)
        return merged
    except ImportError:
        # Fallback: 如果 species_params.py 不可用，使用 HSI overrides 本身
        return dict(_HSI_OVERRIDES)


SPECIES_PARAMS = _build_species_params()  # [v12-fix] 統一來源

# ═══════════════════════════════════════════════════════
# [v12-phase8] 食物鏈級聯時序參數
# 營養級級聯延遲 (天) — 用於 FoodChainPredictor
# References: Henson 2009, Platt 2003, Precioso 2022
# ═══════════════════════════════════════════════════════
FOOD_CHAIN_TIMING = {
    "yellowfin": {
        "zoo_lag_warm": (5, 10),
        "zoo_lag_cool": (10, 18),
        "zoo_lag_cold": (18, 30),
        "arrival_min": 3,
        "arrival_max": 7,
    },
    "bigeye": {
        "zoo_lag_warm": (5, 10),
        "zoo_lag_cool": (10, 18),
        "zoo_lag_cold": (18, 30),
        "arrival_min": 5,
        "arrival_max": 10,
    },
    "albacore": {
        "zoo_lag_warm": (5, 10),
        "zoo_lag_cool": (10, 18),
        "zoo_lag_cold": (18, 30),
        "arrival_min": 7,
        "arrival_max": 14,
    },
    "skipjack": {
        "zoo_lag_warm": (5, 10),
        "zoo_lag_cool": (10, 18),
        "zoo_lag_cold": (18, 30),
        "arrival_min": 1,
        "arrival_max": 5,
    },
}

SCHEDULE = {
    "fetch_sst":            {"hours": 6,  "jitter": 1800},
    "fetch_chlorophyll":    {"hours": 6,  "jitter": 1800},
    "fetch_hycom_3d":       {"hours": 12, "jitter": 3600},
    "fetch_ocean_current":  {"hours": 6,  "jitter": 1800},
    "fetch_open_meteo":     {"hours": 3,  "jitter": 600},
    "fetch_gfw_ais":        {"hours": 12, "jitter": 3600},
    "fetch_viirs_dnb":      {"hours": 24, "jitter": 7200},
    "fetch_cmems_bgc":      {"hours": 24, "jitter": 3600},
    "fetch_woa_do":         {"hours": 720, "jitter": 0},
    "fetch_etopo1":         {"hours": 8760, "jitter": 0},
    "run_mega_analysis":    {"hours": 6,  "jitter": 1800},
}

RATE_LIMIT = {
    "request_delay": 2.0, "max_retries": 3,
    "backoff_base": 30, "cache_fallback": True,
}

OUTPUT = {
    "kml_dir": "output/", "max_kml_size_kb": 100,
    "geojson_dir": "output/", "top_n_hotspots": 20,
    "route_waypoints": 10,
}

