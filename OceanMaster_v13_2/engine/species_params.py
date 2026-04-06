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
        # Eo=0.30 — Deutsch 2015 deep-diving species conservative extrapolation
        # (unified with commercial_core_v2.py METABOLIC_TRAITS for consistency)
        "Eo": 0.30,
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
    "neon_flying_squid": {
        "name_zh": "赤魷",
        "name_en": "Neon Flying Squid",
        "scientific": "Ommastrephes bartramii",
        "Eo": 0.50,
        "Pcrit_kPa": 6.0,
        "phi_crit": 4.0,
        "Topt_C": 18.0,
        "Topt_surface_C": 18.0,
        "Topt_deep_C": 6.0,
        "Topt_deep_sigma": 3.0,
        "Topt_sigma": 3.5,
        "T_min": 12.0,
        "T_max": 24.0,
        "T_spawning": (16, 20),
        "T_spawning_months": [7, 8, 9, 10, 11],
        "moon_sensitivity": "extreme",
        "dvm_strength": "extreme",
        "gear_type": "squid_jigging",
        "chl_optimal": (0.4, 3.0),
        "chl_ref": "Bower 2004",
    },
    "japanese_flying_squid": {
        "name_zh": "日本魷",
        "name_en": "Japanese Flying Squid",
        "scientific": "Todarodes pacificus",
        "Eo": 0.50,
        "Pcrit_kPa": 6.0,
        "phi_crit": 4.0,
        "Topt_C": 15.0,
        "Topt_surface_C": 15.0,
        "Topt_deep_C": 5.0,
        "Topt_deep_sigma": 3.0,
        "Topt_sigma": 3.0,
        "T_min": 8.0,
        "T_max": 22.0,
        "T_spawning": (15, 23),
        "T_spawning_months": [12, 1, 2],
        "moon_sensitivity": "extreme",
        "dvm_strength": "extreme",
        "gear_type": "squid_jigging",
        "chl_optimal": (0.5, 5.0),
        "chl_ref": "Kidokoro 2010",
    },
    # ── v13 台灣近海/遠洋新增魚種 ──────────────────────
    "mahi_mahi": {
        "name_zh": "鬼頭刀",
        "name_en": "Mahi-mahi",
        "scientific": "Coryphaena hippurus",
        # 代謝 (Palko et al. 1982, Benetti et al. 2010)
        "Eo": 0.42,
        "Pcrit_kPa": 4.0,
        "phi_crit": 3.2,
        "Topt_C": 27.0,
        "Topt_surface_C": 27.0,
        "Topt_deep_C": 24.0,           # 極少深潛, 表層捕食者
        "Topt_deep_sigma": 3.0,
        "Topt_sigma": 3.0,
        "T_min": 21.0,
        "T_max": 33.0,
        "T_spawning": (25, 30),
        "T_spawning_months": [4, 5, 6, 7, 8, 9, 10],  # 台灣 4-10 月
        "chl_optimal": (0.15, 0.6),
        "chl_ref": "Palko 1982, Farrell 2014",
        "gear_type": "trolling",
        "fad_association": "extreme",    # [UNVALIDATED] 極高FAD關聯
        "trophic_level": 4.4,
        "prey": ["flying_fish", "squid", "small_pelagics"],
    },
    "blue_marlin": {
        "name_zh": "旗魚",
        "name_en": "Blue Marlin",
        "scientific": "Makaira nigricans",
        # 代謝 (Brill 1996, Prince & Goodyear 2006)
        "Eo": 0.38,
        "Pcrit_kPa": 3.8,
        "phi_crit": 2.8,
        "Topt_C": 26.0,
        "Topt_surface_C": 26.0,
        "Topt_deep_C": 18.0,           # 溫躍層覓食 (可短暫深潛至 800m)
        "Topt_deep_sigma": 5.0,
        "Topt_sigma": 3.5,
        "T_min": 20.0,
        "T_max": 31.0,
        "T_spawning": (26, 29),
        "T_spawning_months": [5, 6, 7, 8, 9],  # 台灣東部 5-9 月
        "chl_optimal": (0.1, 0.4),
        "chl_ref": "Nakamura 1985, Prince & Goodyear 2006",
        "gear_type": "longline",
        "trophic_level": 4.5,
        "prey": ["skipjack", "flying_fish", "squid", "mahi_mahi"],
    },
    "mackerel_scad": {
        "name_zh": "竹筴魚",
        "name_en": "Mackerel Scad",
        "scientific": "Decapterus maruadsi",
        # 代謝 (FAO species catalog, 台灣水試所)
        "Eo": 0.35,
        "Pcrit_kPa": 3.5,
        "phi_crit": 3.0,
        "Topt_C": 24.0,
        "Topt_surface_C": 24.0,
        "Topt_deep_C": 20.0,
        "Topt_deep_sigma": 3.0,
        "Topt_sigma": 4.0,
        "T_min": 16.0,
        "T_max": 30.0,
        "T_spawning": (22, 27),
        "T_spawning_months": [3, 4, 5, 6, 7],  # 台灣近海 3-7 月
        "chl_optimal": (0.3, 2.0),
        "chl_ref": "台灣水試所, FAO Species Catalog",
        "gear_type": "purse_seine",
        "trophic_level": 3.2,
        "prey": ["zooplankton", "small_crustaceans"],
        "schooling": "extreme",          # 大群群游
    },
    "pacific_saury": {
        "name_zh": "秋刀魚",
        "name_en": "Pacific Saury",
        "scientific": "Cololabis saira",
        # 代謝 (Suyama et al. 2006, Watanabe & Oozeki 2015)
        "Eo": 0.45,
        "Pcrit_kPa": 5.0,
        "phi_crit": 3.5,
        "Topt_C": 15.0,
        "Topt_surface_C": 15.0,
        "Topt_deep_C": 10.0,
        "Topt_deep_sigma": 3.0,
        "Topt_sigma": 3.0,
        "T_min": 8.0,
        "T_max": 22.0,
        "T_spawning": (14, 20),
        "T_spawning_months": [10, 11, 12, 1, 2],  # 秋冬洄游台灣海域
        "chl_optimal": (0.3, 3.0),
        "chl_ref": "Suyama 2006, Watanabe & Oozeki 2015",
        "gear_type": "stick_held_dip_net",  # 棒受網
        "moon_sensitivity": "high",      # 集魚燈作業
        "dvm_strength": "moderate",
        "trophic_level": 3.1,
        "prey": ["copepods", "euphausiids", "zooplankton"],
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
    "neon_flying_squid": {
        "day_depth_range": (200, 600),
        "night_depth_range": (0, 50),
        "max_dive_depth": 800,
        "temp_min_tolerance": 4.0,
        "optimal_feeding_depth": (0, 50),
        "source": "Bower 2004",
    },
    "japanese_flying_squid": {
        "day_depth_range": (100, 400),
        "night_depth_range": (0, 30),
        "max_dive_depth": 500,
        "temp_min_tolerance": 4.0,
        "optimal_feeding_depth": (0, 30),
        "source": "Kidokoro 2010",
    },
    # ── v13 新增 ──
    "mahi_mahi": {
        "day_depth_range": (0, 40),
        "night_depth_range": (0, 20),
        "max_dive_depth": 85,
        "temp_min_tolerance": 21.0,
        "optimal_feeding_depth": (0, 30),
        "source": "Palko 1982, Merten et al. 2014",
    },
    "blue_marlin": {
        "day_depth_range": (50, 300),
        "night_depth_range": (0, 50),
        "max_dive_depth": 800,
        "temp_min_tolerance": 14.0,
        "optimal_feeding_depth": (30, 200),
        "source": "Prince & Goodyear 2006, Block et al. 1992",
    },
    "mackerel_scad": {
        "day_depth_range": (30, 150),
        "night_depth_range": (0, 30),
        "max_dive_depth": 200,
        "temp_min_tolerance": 16.0,
        "optimal_feeding_depth": (10, 80),
        "source": "FAO Species Catalog, 台灣水試所",
    },
    "pacific_saury": {
        "day_depth_range": (20, 100),
        "night_depth_range": (0, 15),
        "max_dive_depth": 150,
        "temp_min_tolerance": 8.0,
        "optimal_feeding_depth": (0, 30),
        "source": "Suyama 2006, Watanabe & Oozeki 2015",
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
    "neon_flying_squid": {"optimal": 80, "sigma": 30, "note": "夜間在溫躍層上方覓食"},
    "japanese_flying_squid": {"optimal": 50, "sigma": 20, "note": "夜間在極淺表層覓食"},
    # ── v13 新增 ──
    "mahi_mahi": {"optimal": 100, "sigma": 40, "note": "表層覓食者, Z20淺=暖水層厚=有利"},
    "blue_marlin": {"optimal": 180, "sigma": 60, "note": "溫躍層附近覓食, Z20中等偏好"},
    "mackerel_scad": {"optimal": 80, "sigma": 30, "note": "近岸淺層, Z20影響小"},
    "pacific_saury": {"optimal": 60, "sigma": 25, "note": "追蹤親潮/黑潮交匯的淺溫躍層"},
}

# ═══════════════════════════════════════════════════════
# [v13] 黑潮 (Kuroshio) 物種親和度
# 0=無關, 1=高度依賴黑潮
# ═══════════════════════════════════════════════════════
KUROSHIO_SPECIES_AFFINITY = {
    "mahi_mahi": 0.90,         # 黑潮暖水邊緣是鬼頭刀主要漁場
    "blue_marlin": 0.85,       # 旗魚追蹤黑潮暖流覓食
    "yellowfin": 0.70,         # 黃鰭鮪在黑潮主軸兩側活動
    "bigeye": 0.50,            # 大目鮪深層活動, 黑潮影響較間接
    "skipjack": 0.65,          # 正鰹在暖池-黑潮交匯處聚集
    "albacore": 0.45,          # 長鰭偏好冷水, 在黑潮北界出沒
    "mackerel_scad": 0.55,     # 竹筴魚在黑潮支流的近岸湧升區
    "pacific_saury": 0.40,     # 秋刀魚在親潮-黑潮交匯 (混合水域)
    "neon_flying_squid": 0.35, # 赤魷在北太平洋過渡帶
    "japanese_flying_squid": 0.30,
    "squid_todarodes": 0.30,
    "squid_ommastrephes": 0.35,
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
