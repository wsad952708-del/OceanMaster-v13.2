"""
OceanMaster — 物種生物學參數集中管理
=====================================
所有魚種的溫度偏好、深度行為、氧耐受、食物偏好集中於此。
參數值來自學術論文，附來源標注。

使用:
  from engine.species_params import SPECIES, get_species, DVM_PARAMS, Z20_PREFS
"""

import logging

log = logging.getLogger("OceanMaster.SpeciesParams")

# ═══════════════════════════════════════════════════════
# 核心物種代謝參數 (Deutsch 2015 + 文獻整理)
# ═══════════════════════════════════════════════════════
SPECIES = {
    "yellowfin": {
        "name_zh": "黃鰭鮪",
        "name_en": "Yellowfin tuna",
        "scientific": "Thunnus albacares",
        # 代謝 (Deutsch 2015)
        "Eo": 0.40,
        "Pcrit_kPa": 4.5,
        "phi_crit": 3.0,
        # 溫度偏好 (Brill 1994, Lehodey 2008)
        "Topt_C": 26.0,           # 向後相容 (表層偏好)
        "Topt_surface_C": 26.0,   # [v11] 夜間表層覓食溫度
        "Topt_deep_C": 20.0,      # [v11] 日間深層溫度 (淺 DVM, ~50-250m)
        "Topt_deep_sigma": 4.0,
        "Topt_sigma": 3.5,
        "T_min": 18.0,
        "T_max": 31.0,
        "T_spawning": (26, 30),
        "T_spawning_months": [4, 5, 6, 7, 8, 9],  # [v11] 產卵季 (熱帶全年偏夏)
        # CHL 偏好 (mg/m³)
        "chl_optimal": (0.1, 0.5),
        "chl_ref": "Lehodey 2008",
    },
    "bigeye": {
        "name_zh": "大目鮪",
        "name_en": "Bigeye tuna",
        "scientific": "Thunnus obesus",
        "Eo": 0.35,
        "Pcrit_kPa": 3.8,
        "phi_crit": 2.5,
        # [v11] 修正 Bug #4: 分離表層/深層溫度偏好 (Schaefer & Fuller 2010)
        # 原始 Topt_C=18.0 僅反映深層偏好，SST 計算用表層偏好更準確
        "Topt_C": 18.0,           # 保留向後相容
        "Topt_surface_C": 26.0,   # [v11] 夜間上浮表層 SST 偏好 (Schaefer & Fuller 2010)
        "Topt_deep_C": 14.0,      # [v11] 日間深層覓食 (200-600m, 10-18°C, 中位 ~14°C)
        "Topt_deep_sigma": 4.0,
        "Topt_sigma": 5.0,
        "T_min": 9.0,
        "T_max": 29.0,
        "T_spawning": (26, 29),
        "T_spawning_months": [4, 5, 6, 7, 8, 9, 10],
        "chl_optimal": (0.1, 0.3),
        "chl_ref": "Musyl 2003, SEAPODYM, Schaefer & Fuller 2010",
    },
    "skipjack": {
        "name_zh": "正鰹",
        "name_en": "Skipjack tuna",
        "scientific": "Katsuwonus pelamis",
        "Eo": 0.45,
        "Pcrit_kPa": 5.5,
        "phi_crit": 3.5,
        "Topt_C": 28.0,
        "Topt_sigma": 2.5,
        "T_min": 20.0,
        "T_max": 33.0,
        "T_spawning": (28, 30),
        "T_spawning_months": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # 全年產卵
        "chl_optimal": (0.2, 1.0),
        "chl_ref": "Brill 1994, Lehodey 2010",
    },
    "albacore": {
        "name_zh": "長鰭鮪",
        "name_en": "Albacore tuna",
        "scientific": "Thunnus alalunga",
        "Eo": 0.38,
        "Pcrit_kPa": 4.0,
        "phi_crit": 2.8,
        "Topt_C": 18.5,
        "Topt_surface_C": 18.5,   # [v11] 長鰭偏好冷水表層
        "Topt_deep_C": 14.0,
        "Topt_deep_sigma": 3.5,
        "Topt_sigma": 4.0,
        "T_min": 12.0,
        "T_max": 25.0,
        "T_spawning": (24, 28),
        "T_spawning_months": [3, 4, 5, 6, 7, 8],
        "chl_optimal": (0.1, 0.4),
        "chl_ref": "Williams 2014",
    },
}

# ═══════════════════════════════════════════════════════
# 垂直遷移行為 (DVM) 參數
# ═══════════════════════════════════════════════════════
DVM_PARAMS = {
    "yellowfin": {
        "day_depth_range": (50, 250),
        "night_depth_range": (0, 50),
        "max_dive_depth": 300,
        "temp_min_tolerance": 15.0,
        "optimal_feeding_depth": (20, 150),
        "source": "Schaefer et al. 2007, 2011",
    },
    "bigeye": {
        "day_depth_range": (200, 600),
        "night_depth_range": (0, 100),
        "max_dive_depth": 700,
        "temp_min_tolerance": 9.0,
        "optimal_feeding_depth": (150, 400),
        "source": "Musyl et al. 2003, Schaefer & Fuller 2010",
    },
    "skipjack": {
        "day_depth_range": (20, 200),
        "night_depth_range": (0, 30),
        "max_dive_depth": 250,
        "temp_min_tolerance": 20.0,
        "optimal_feeding_depth": (10, 100),
        "source": "Brill et al. 1999",
    },
    "albacore": {
        "day_depth_range": (100, 300),
        "night_depth_range": (0, 80),
        "max_dive_depth": 400,
        "temp_min_tolerance": 12.0,
        "optimal_feeding_depth": (50, 250),
        "source": "Williams et al. 2014",
    },
}

# ═══════════════════════════════════════════════════════
# Z20 (20°C 等溫線深度) 偏好
# ═══════════════════════════════════════════════════════
Z20_PREFS = {
    "yellowfin": {"optimal": 150, "sigma": 50, "note": "淺溫躍層=表層餌料多"},
    "bigeye":    {"optimal": 300, "sigma": 80, "note": "深溫躍層=深層覓食空間大"},
    "skipjack":  {"optimal": 120, "sigma": 40, "note": "嚴格在MLD上方活動"},
    "albacore":  {"optimal": 200, "sigma": 60, "note": "中深層溫躍層附近活動"},
}

# ═══════════════════════════════════════════════════════
# VGPM 餌場參數
# ═══════════════════════════════════════════════════════
FORAGE_PARAMS = {
    "epsilon": 0.04,     # 營養轉換效率 (Iverson 1990)
    "vgpm_coeffs": [     # Pb_opt 多項式係數 (Behrenfeld & Falkowski 1997)
        -3.27e-8, 3.4132e-6, -1.348e-4, 2.46e-3,
        -2.05e-2, 6.17e-2, 2.749e-1, 1.2956
    ],
}


def get_species(name: str) -> dict:
    """取得物種參數，支援中英文名"""
    if name in SPECIES:
        return SPECIES[name]
    for k, v in SPECIES.items():
        if v["name_zh"] == name or v["name_en"].lower() == name.lower():
            return v
    raise KeyError(f"Unknown species: {name}")


def list_species() -> list:
    """列出所有支援的物種"""
    return [(k, v["name_zh"]) for k, v in SPECIES.items()]
