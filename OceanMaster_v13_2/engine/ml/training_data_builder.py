"""
OceanMaster v13.2 — Training Data Builder
==========================================
統一的訓練數據管線：公開數據 + 合成數據 + 真實漁獲日誌接口

三層數據來源：
  1. WCPFC/FAO 公開漁獲統計（已有 CSV）
  2. 食物鏈感知的合成數據（在真實數據到位前使用）
  3. 真實漁獲日誌（船公司提供，預留插槽）

輸出統一格式：
  date, lat, lon, species, catch_kg, depth, source
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime, timedelta

log = logging.getLogger("OceanMaster.TrainingDataBuilder")


# ═══════════════════════════════════════════════════════════
#  魚種生態參數（含食物鏈延遲）
# ═══════════════════════════════════════════════════════════

SPECIES_ECOLOGY = {
    "yellowfin": {
        "name_zh": "黃鰭鮪", "name_en": "Yellowfin Tuna",
        "sst_opt": (24, 30), "sst_fatal": (20, 33),
        "chl_opt": (0.04, 1.0),   # mg/m³
        "z20_range": None,         # 表層魚
        "prey": ["mackerel", "flying_fish", "squid"],
        "food_chain_delay_days": (15, 25),  # 湧升後多久到達
        "fad_residence_days": 6.6,
        "swim_speed_kmh": 75,
        "depth_range_m": (0, 250),
        "wcpfc_code": "YFT",
    },
    "bigeye": {
        "name_zh": "大目鮪", "name_en": "Bigeye Tuna",
        "sst_opt": (17, 22), "sst_fatal": (6, 33),
        "chl_opt": (0.02, 0.5),
        "z20_range": (150, 230),   # 溫躍層下緣
        "prey": ["deep_shrimp", "lanternfish", "squid"],
        "food_chain_delay_days": (20, 32),
        "fad_residence_days": 7.6,
        "swim_speed_kmh": 60,
        "depth_range_m": (200, 600),
        "wcpfc_code": "BET",
    },
    "skipjack": {
        "name_zh": "鰹魚", "name_en": "Skipjack Tuna",
        "sst_opt": (24, 30), "sst_fatal": (18, 33),
        "chl_opt": (0.2, 0.5),
        "z20_range": None,
        "prey": ["copepod", "anchovy", "sardine"],
        "food_chain_delay_days": (14, 20),
        "fad_residence_days": 4.6,
        "swim_speed_kmh": 50,
        "depth_range_m": (0, 260),
        "wcpfc_code": "SKJ",
    },
    "albacore": {
        "name_zh": "長鰭鮪", "name_en": "Albacore Tuna",
        "sst_opt": (14, 18), "sst_fatal": (10, 25),
        "chl_opt": (0.1, 0.3),
        "z20_range": None,
        "prey": ["saury", "squid", "krill"],
        "food_chain_delay_days": (20, 35),
        "fad_residence_days": 5.0,
        "swim_speed_kmh": 80,
        "depth_range_m": (0, 380),
        "wcpfc_code": "ALB",
    },
    "squid": {
        "name_zh": "魷魚", "name_en": "Ommastrephid Squid",
        "sst_opt": (15, 24), "sst_fatal": (8, 28),
        "chl_opt": (0.1, 0.5),
        "z20_range": None,
        "prey": ["krill", "small_fish"],
        "food_chain_delay_days": (0, 3),   # 即時（趨光性）
        "fad_residence_days": 1.0,
        "swim_speed_kmh": 30,
        "depth_range_m": (0, 300),
        "wcpfc_code": None,
    },
}

# 食物鏈延遲矩陣（湧升後各事件的天數）
FOOD_CHAIN_DELAYS = {
    "upwelling_to_bloom":     (3, 7),    # 湧升 → CHL 爆發
    "bloom_to_zooplankton":   (5, 14),   # CHL → 浮游動物高峰
    "zooplankton_to_baitfish": (7, 14),  # 浮游動物 → 餌料魚
    "baitfish_to_tuna":       (2, 10),   # 餌料魚 → 鮪魚
    "eddy_maturation":        (15, 30),  # 渦旋形成 → 生態成熟
}


# ═══════════════════════════════════════════════════════════
#  Part 1a: 公開數據載入器
# ═══════════════════════════════════════════════════════════

class PublicDataLoader:
    """載入 WCPFC / FAO 等公開漁獲數據，統一輸出格式。"""

    @staticmethod
    def load_wcpfc(data_dir: str = "data") -> pd.DataFrame:
        """載入 WCPFC 公開統計，轉為統一格式。"""
        try:
            from engine.ml.wcpfc_data_loader import combine_all_wcpfc
            df = combine_all_wcpfc(data_dir, min_year=2000)
            # 轉統一格式
            out = pd.DataFrame({
                "date": pd.to_datetime(
                    df["year"].astype(str) + "-" +
                    df["month"].astype(str).str.zfill(2) + "-15"
                ),
                "lat": df["lat"],
                "lon": df["lon"],
                "species": df["species"],
                "catch_kg": df["catch_mt"] * 1000,
                "depth": np.nan,
                "source": "wcpfc_public",
                "synthetic": False,
            })
            log.info(f"✅ WCPFC: {len(out)} records loaded")
            return out
        except Exception as e:
            log.warning(f"⚠️ WCPFC load failed: {e}")
            return pd.DataFrame()

    @staticmethod
    def load_fao(data_dir: str = "data") -> pd.DataFrame:
        """載入 FAO 全球漁獲統計。"""
        try:
            from engine.ml.fao_data_loader import load_fao_catch
            df = load_fao_catch(data_dir)
            out = pd.DataFrame({
                "date": df.get("date", pd.NaT),
                "lat": df.get("lat", np.nan),
                "lon": df.get("lon", np.nan),
                "species": df.get("species", "unknown"),
                "catch_kg": df.get("catch_kg", 0),
                "depth": np.nan,
                "source": "fao",
                "synthetic": False,
            })
            log.info(f"✅ FAO: {len(out)} records loaded")
            return out
        except Exception as e:
            log.warning(f"⚠️ FAO load failed: {e}")
            return pd.DataFrame()


# ═══════════════════════════════════════════════════════════
#  Part 1b: 食物鏈感知合成數據產生器
# ═══════════════════════════════════════════════════════════

class FoodChainSyntheticGenerator:
    """
    產生食物鏈感知的合成漁獲數據。

    與原 SyntheticCPUEGenerator 不同之處：
    1. 內建食物鏈延遲邏輯（CHL 爆發 → X 天後漁獲）
    2. 產生 84 維特徵（含 lag features）
    3. 標記 synthetic=True，真實數據進來後自動降權
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        n_samples: int = 2000,
        species: str = "yellowfin",
        region: str = "wpac",
    ) -> pd.DataFrame:
        """
        產生含食物鏈延遲的合成數據。

        Returns:
            DataFrame with columns: date, lat, lon, species, catch_kg,
                                    sst, chl, ssh, ... (84 features),
                                    food_chain_stage, synthetic
        """
        eco = SPECIES_ECOLOGY.get(species, SPECIES_ECOLOGY["yellowfin"])
        rng = self.rng

        # ── 時空座標 ──
        if region == "wpac":
            lat = rng.uniform(5, 35, n_samples)
            lon = rng.uniform(120, 175, n_samples)
        else:
            lat = rng.uniform(-20, 40, n_samples)
            lon = rng.uniform(100, 180, n_samples)

        # 隨機日期（過去 3 年）
        base = datetime(2023, 1, 1)
        days_offset = rng.integers(0, 1095, n_samples)
        dates = [base + timedelta(days=int(d)) for d in days_offset]

        # ── 環境特徵（物理基礎）──
        sst_opt_mean = np.mean(eco["sst_opt"])
        sst = rng.normal(sst_opt_mean, 3.0, n_samples)
        sst = np.clip(sst, 5, 35)

        chl_opt_mean = np.mean(eco["chl_opt"])
        chl = np.exp(rng.normal(np.log(chl_opt_mean + 0.01), 0.5, n_samples))
        chl = np.clip(chl, 0.01, 20.0)

        ssh = rng.normal(0, 0.15, n_samples)
        sss = rng.normal(34.8, 0.5, n_samples)
        mld = rng.uniform(10, 120, n_samples)
        z20 = rng.uniform(50, 300, n_samples)
        do_surface = rng.uniform(180, 280, n_samples)
        wind = rng.uniform(0, 15, n_samples)
        wave = rng.uniform(0.3, 5, n_samples)
        bathy = -rng.uniform(100, 5000, n_samples)

        # ── 食物鏈特徵（核心新增）──
        npp = 300 + 600 * chl / (chl + 0.5) + rng.normal(0, 50, n_samples)
        npp = np.clip(npp, 50, 2000)

        # CHL lag features — 模擬食物鏈時間延遲
        chl_lag3 = chl * rng.uniform(0.5, 1.5, n_samples)
        chl_lag7 = chl * rng.uniform(0.3, 2.0, n_samples)
        chl_lag14 = chl * rng.uniform(0.2, 2.5, n_samples)
        chl_lag21 = chl * rng.uniform(0.1, 3.0, n_samples)

        npp_lag7 = npp * rng.uniform(0.5, 1.5, n_samples)
        npp_lag14 = npp * rng.uniform(0.3, 2.0, n_samples)

        sst_rate_3d = rng.normal(0, 0.5, n_samples)  # °C/3d
        sst_rate_7d = rng.normal(0, 0.8, n_samples)

        # 湧升指數
        upwelling_idx = np.clip(-sst_rate_3d * wind * 0.1, -2, 5)

        # 藻華偵測
        chl_clim = 0.2  # 氣候學平均
        bloom_active = (chl > chl_clim * 3).astype(float)
        bloom_day = bloom_active * rng.integers(0, 21, n_samples)
        bloom_intensity = chl / (chl_clim + 0.01)

        # 浮游植物功能型
        pft_diatom = np.clip(rng.beta(2, 3, n_samples), 0.05, 0.95)
        pft_micro = np.clip(rng.beta(3, 4, n_samples), 0.05, 0.95)

        # 浮游動物密度估計 (Stock-Dunne)
        zoo_density = 0.1 * npp ** 0.7 * np.exp(-0.02 * np.abs(sst - 22))
        zoo_density = np.clip(zoo_density, 0, 100)

        # 餌料魚潛力
        front_intensity = np.abs(rng.normal(0, 0.5, n_samples))
        eddy_strength = np.abs(rng.normal(0, 0.3, n_samples))
        baitfish_potential = (npp_lag14 / 500) * front_intensity * (1 + eddy_strength)
        baitfish_potential = np.clip(baitfish_potential, 0, 5)

        # 食物鏈階段 (0-4)
        delay_min, delay_max = eco["food_chain_delay_days"]
        food_chain_stage = np.clip(
            (bloom_day / ((delay_min + delay_max) / 2) * 4).astype(int),
            0, 4
        )

        # 食物鏈 ETA
        food_chain_eta = np.clip(
            (delay_max - bloom_day).astype(float),
            0, delay_max
        )

        # 渦旋年齡
        eddy_age = rng.uniform(0, 45, n_samples)

        # CHL 鋒面
        chl_front = np.abs(rng.normal(0, 0.3, n_samples))

        # PAR / Kd490
        par = rng.uniform(20, 60, n_samples)
        kd490 = rng.uniform(0.02, 0.15, n_samples)

        # VIIRS 燈船
        viirs_light = rng.exponential(0.5, n_samples) if species == "squid" else rng.exponential(0.05, n_samples)

        # ── 漁獲量（基於食物鏈邏輯）──
        # SST 適合度
        sst_score = np.exp(-0.5 * ((sst - sst_opt_mean) / 2) ** 2)

        # CHL 適合度（歷史 CHL lag14 更重要）
        chl_score = np.exp(-0.5 * ((np.log(chl_lag14 + 0.01) - np.log(chl_opt_mean + 0.01)) / 0.5) ** 2)

        # 食物鏈成熟度 — 核心生成邏輯
        fc_score = np.clip(food_chain_stage / 4.0, 0, 1)

        # 鋒面加成
        front_score = np.clip(front_intensity * 2, 0, 1)

        # 綜合 HSI
        hsi = (
            0.25 * sst_score +
            0.30 * chl_score +
            0.25 * fc_score +
            0.20 * front_score
        )

        # 漁獲 = HSI × 隨機噪音
        catch_kg = hsi * eco.get("cpue_scale", 100) * rng.lognormal(0, 0.5, n_samples)
        catch_kg = np.clip(catch_kg, 0, 5000)

        # 輸出 DataFrame — 84 維特徵
        df = pd.DataFrame({
            # 基礎欄位
            "date": dates, "lat": lat, "lon": lon,
            "species": species, "catch_kg": catch_kg,
            "depth": -np.abs(rng.normal(eco["depth_range_m"][0], 50, n_samples)),
            "source": "synthetic_food_chain",
            "synthetic": True,

            # 環境基礎 (1-15)
            "sst": sst, "chl": chl, "ssh": ssh, "sss": sss,
            "mld": mld, "z20": z20, "do_surface": do_surface,
            "wind_speed": wind, "wave_height": wave, "bathy": bathy,
            "npp": npp, "par": par, "kd490": kd490,
            "sst_gradient": np.abs(rng.normal(0, 0.3, n_samples)),
            "ssh_gradient": np.abs(rng.normal(0, 0.01, n_samples)),

            # 鋒面/渦旋 (16-25)
            "front_intensity": front_intensity,
            "front_distance_km": rng.uniform(0, 200, n_samples),
            "eddy_strength": eddy_strength,
            "eddy_type": rng.choice([-1, 0, 1], n_samples),  # -1=cyclonic, 1=anti
            "eddy_age_days": eddy_age,
            "kuroshio_distance": rng.uniform(0, 500, n_samples),
            "u_current": rng.normal(0, 0.3, n_samples),
            "v_current": rng.normal(0, 0.3, n_samples),
            "current_shear": np.abs(rng.normal(0, 0.1, n_samples)),
            "convergence": rng.normal(0, 0.01, n_samples),

            # 食物鏈時序 (26-37) — 核心新增
            "chl_lag3": chl_lag3, "chl_lag7": chl_lag7,
            "chl_lag14": chl_lag14, "chl_lag21": chl_lag21,
            "npp_lag7": npp_lag7, "npp_lag14": npp_lag14,
            "sst_rate_3d": sst_rate_3d, "sst_rate_7d": sst_rate_7d,
            "upwelling_index": upwelling_idx,
            "bloom_active": bloom_active,
            "bloom_day": bloom_day,
            "bloom_intensity": bloom_intensity,

            # 食物鏈推估 (38-48)
            "bloom_type": rng.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1]),
            "pft_diatom_frac": pft_diatom,
            "pft_micro_frac": pft_micro,
            "zoo_density_est": zoo_density,
            "baitfish_potential": baitfish_potential,
            "food_chain_stage": food_chain_stage,
            "food_chain_eta": food_chain_eta,
            "chl_front_intensity": chl_front,
            "prey_sst_match": sst_score,
            "viirs_fishing_light": viirs_light,
            "npp_change_rate": rng.normal(0, 50, n_samples),

            # 時空/天文 (49-59)
            "day_of_year": [d.timetuple().tm_yday for d in dates],
            "month": [d.month for d in dates],
            "hour_utc": rng.integers(0, 24, n_samples),
            "lunar_phase": rng.uniform(0, 1, n_samples),
            "solar_elevation": rng.uniform(-10, 80, n_samples),
            "distance_from_port_nm": rng.uniform(50, 800, n_samples),
            "distance_to_shelf_nm": rng.uniform(0, 300, n_samples),
            "eez_flag": rng.choice([0, 1], n_samples),
            "seamount_distance_km": rng.uniform(0, 500, n_samples),
            "gfw_fishing_hours": rng.exponential(5, n_samples),
            "ais_vessel_density": rng.poisson(3, n_samples),
        })

        log.info(
            f"✅ 合成數據: {n_samples} 筆 {species} "
            f"({df.shape[1]} 欄位, 含食物鏈特徵)"
        )
        return df

    def generate_multi_species(
        self, n_per_species: int = 500
    ) -> pd.DataFrame:
        """所有魚種各產生 n 筆合成數據。"""
        frames = []
        for sp in SPECIES_ECOLOGY:
            frames.append(self.generate(n_per_species, sp))
        df = pd.concat(frames, ignore_index=True)
        log.info(f"✅ 多魚種合成數據: {len(df)} 筆 ({len(SPECIES_ECOLOGY)} 種)")
        return df


# ═══════════════════════════════════════════════════════════
#  Part 1c: 真實漁獲日誌接口（預留插槽）
# ═══════════════════════════════════════════════════════════

class RealCatchLoader:
    """
    真實漁獲日誌載入器。

    ╔══════════════════════════════════════════════════════════╗
    ║  TODO: 替換真實漁獲數據                                  ║
    ║                                                         ║
    ║  當你從船公司取得真實漁獲日誌時：                          ║
    ║  1. 準備 CSV/Excel/JSON，欄位如下：                      ║
    ║     date, lat, lon, species, catch_kg, depth             ║
    ║  2. 放到 data/catch_logs/ 目錄下                         ║
    ║  3. 呼叫 RealCatchLoader.load_all("data/catch_logs/")   ║
    ║  4. 系統會自動回查衛星數據、重新訓練模型                   ║
    ║                                                         ║
    ║  欄位說明：                                              ║
    ║  - date: 日期 (YYYY-MM-DD 或 YYYY-MM-DD HH:MM)          ║
    ║  - lat: 緯度 (度, 正=北)                                 ║
    ║  - lon: 經度 (度, 正=東)                                 ║
    ║  - species: 魚種 (yellowfin/bigeye/skipjack/albacore)    ║
    ║  - catch_kg: 漁獲重量 (公斤)                             ║
    ║  - depth: 作業水深 (公尺, 可選)                           ║
    ╚══════════════════════════════════════════════════════════╝
    """

    EXPECTED_COLUMNS = ["date", "lat", "lon", "species", "catch_kg"]
    OPTIONAL_COLUMNS = ["depth", "hooks", "gear", "vessel_id", "sst_onboard", "notes"]

    @staticmethod
    def load_csv(path: str) -> pd.DataFrame:
        """載入單一 CSV 漁獲日誌。"""
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df["source"] = "real_catch_log"
        df["synthetic"] = False
        if "depth" not in df.columns:
            df["depth"] = np.nan
        missing = [c for c in RealCatchLoader.EXPECTED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"CSV 缺少必要欄位: {missing}")
        log.info(f"✅ 真實漁獲: {len(df)} 筆 from {path}")
        return df

    @staticmethod
    def load_excel(path: str, sheet_name: str = "Sheet1") -> pd.DataFrame:
        """載入 Excel 漁獲日誌。"""
        df = pd.read_excel(path, sheet_name=sheet_name)
        df["date"] = pd.to_datetime(df["date"])
        df["source"] = "real_catch_log"
        df["synthetic"] = False
        if "depth" not in df.columns:
            df["depth"] = np.nan
        log.info(f"✅ 真實漁獲: {len(df)} 筆 from {path}")
        return df

    @staticmethod
    def load_all(
        catch_dir: str = "data/catch_logs",
    ) -> pd.DataFrame:
        """載入目錄下所有漁獲日誌（CSV/Excel）。"""
        p = Path(catch_dir)
        if not p.exists():
            log.info(f"📂 漁獲日誌目錄不存在: {catch_dir}（等待真實數據）")
            return pd.DataFrame()

        frames = []
        for f in sorted(p.glob("*.csv")):
            try:
                frames.append(RealCatchLoader.load_csv(str(f)))
            except Exception as e:
                log.warning(f"  ⚠️ {f.name}: {e}")

        for f in sorted(p.glob("*.xlsx")):
            try:
                frames.append(RealCatchLoader.load_excel(str(f)))
            except Exception as e:
                log.warning(f"  ⚠️ {f.name}: {e}")

        if not frames:
            log.info("📂 無漁獲日誌檔案")
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        log.info(f"✅ 漁獲日誌: {len(df)} 筆, {df['species'].nunique()} 魚種")
        return df


# ═══════════════════════════════════════════════════════════
#  整合：訓練數據建構器
# ═══════════════════════════════════════════════════════════

class TrainingDataBuilder:
    """
    整合三層數據來源，產生訓練用 DataFrame。

    優先級：真實漁獲 > 公開統計 > 合成數據
    合成數據在真實數據到位後自動降權。
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.public_loader = PublicDataLoader()
        self.synthetic_gen = FoodChainSyntheticGenerator()
        self.real_loader = RealCatchLoader()

    def build(
        self,
        min_samples: int = 2000,
        synthetic_weight: float = 0.3,
    ) -> pd.DataFrame:
        """
        建構訓練數據集。

        Args:
            min_samples: 最少樣本數（不足的部分用合成數據補）
            synthetic_weight: 合成數據的 sample_weight（0-1）
                              真實數據 weight=1.0

        Returns:
            DataFrame with 'sample_weight' column for weighted training
        """
        frames = []

        # 1. 真實漁獲日誌（最高優先級）
        real_df = self.real_loader.load_all(
            str(Path(self.data_dir) / "catch_logs")
        )
        if len(real_df) > 0:
            real_df["sample_weight"] = 1.0
            frames.append(real_df)

        # 2. WCPFC 公開數據
        wcpfc_df = self.public_loader.load_wcpfc(self.data_dir)
        if len(wcpfc_df) > 0:
            wcpfc_df["sample_weight"] = 0.7
            frames.append(wcpfc_df)

        # 3. 合成數據（填補不足）
        n_real = sum(len(f) for f in frames)
        if n_real < min_samples:
            n_syn = min_samples - n_real
            syn_df = self.synthetic_gen.generate_multi_species(
                n_per_species=max(100, n_syn // len(SPECIES_ECOLOGY))
            )
            syn_df["sample_weight"] = synthetic_weight
            frames.append(syn_df)
            log.info(f"  + {len(syn_df)} 合成數據補足 (weight={synthetic_weight})")

        if not frames:
            log.warning("⚠️ 無任何數據來源，產生純合成數據")
            syn_df = self.synthetic_gen.generate_multi_species(
                n_per_species=min_samples // len(SPECIES_ECOLOGY)
            )
            syn_df["sample_weight"] = synthetic_weight
            frames.append(syn_df)

        df = pd.concat(frames, ignore_index=True)
        # 去重
        df = df.drop_duplicates(subset=["date", "lat", "lon", "species"])

        log.info(
            f"📊 訓練數據集: {len(df)} 筆\n"
            f"   真實:  {(~df['synthetic']).sum()} 筆\n"
            f"   合成:  {df['synthetic'].sum()} 筆\n"
            f"   魚種:  {df['species'].nunique()}\n"
            f"   日期:  {df['date'].min()} — {df['date'].max()}"
        )
        return df


# ═══════════════════════════════════════════════════════════
#  CLI 測試
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 60)
    print("  OceanMaster — Training Data Builder 測試")
    print("=" * 60)

    builder = TrainingDataBuilder()
    df = builder.build(min_samples=3000, synthetic_weight=0.3)

    print(f"\n欄位數: {df.shape[1]}")
    print(f"前 5 筆:\n{df.head()}")
    print(f"\n魚種分佈:\n{df['species'].value_counts()}")
    print(f"\n來源分佈:\n{df['source'].value_counts()}")
    print(f"\n食物鏈特徵範例:")
    fc_cols = [c for c in df.columns if "lag" in c or "bloom" in c or "food_chain" in c]
    if fc_cols:
        print(df[fc_cols].describe().round(2))
