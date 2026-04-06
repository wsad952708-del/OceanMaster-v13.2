#!/usr/bin/env python3
"""
OceanMaster — 真實漁獲數據轉換工具
====================================
將 WCPFC 或自有漁獲數據轉換成 ML 訓練格式。

支援的輸入格式:
  A) WCPFC 公開數據 (CSV)
  B) 自有漁船日誌 (CSV)
  C) 通用格式 (只要有 year, month, lat, lon, cpue 或 catch+effort)

用法:
  python prepare_real_data.py --input your_data.csv --format wcpfc
  python prepare_real_data.py --input logbook.csv --format custom
  python prepare_real_data.py --input data.csv --format auto

輸出:
  ml_system/data/simulated_cpue_yellowfin.csv  (ML訓練用)
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
log = logging.getLogger("DataPrep")

# ═══════════════════════════════════════════════════
#  環境模擬器 (用WOA氣候態估算環境變量)
# ═══════════════════════════════════════════════════

class ClimatologyEstimator:
    """
    用氣候態統計估算環境變量。
    當沒有即時衛星數據時，用這些估算值作為特徵。
    未來接上真實衛星數據後，精度會再提升。
    """

    def estimate_sst(self, lat, lon, month):
        """WOA氣候態SST估算"""
        base = 30.0 - 0.35 * abs(lat - 5.0)
        seasonal_amp = 0.5 + 0.15 * abs(lat - 15)
        seasonal = seasonal_amp * np.cos(2 * np.pi * (month - 2) / 12)
        lon_effect = 0.5 * np.exp(-((lon - 145) / 30) ** 2)
        return np.clip(base + seasonal + lon_effect, 15.0, 33.0)

    def estimate_chl(self, lat, lon, month):
        """MODIS氣候態Chl-a估算"""
        base = 0.15
        lat_effect = 0.02 * abs(lat - 10)
        coast = 0.2 if (lon < 130 or lon > 170) else 0.0
        bloom = 0.05 * np.exp(-((month - 4) / 2.5) ** 2)
        return np.clip(base + lat_effect + coast + bloom, 0.01, 5.0)

    def estimate_ssh(self, lat, lon, month):
        base = 0.05 * np.sin(2 * np.pi * lat / 40)
        seasonal = 0.03 * np.cos(2 * np.pi * (month - 1) / 12)
        return base + seasonal

    def estimate_do(self, lat, lon, month, sst):
        do_base = 8.0 - 0.15 * sst + 0.02 * abs(lat)
        if 5 < lat < 15 and 130 < lon < 160:
            do_base -= 1.5
        return np.clip(do_base, 1.0, 8.0)

    def estimate_current_speed(self, lat, lon):
        base = 0.2 + 0.1 * np.sin(2 * np.pi * lat / 30)
        if 25 < lat < 35 and 130 < lon < 150:
            base += 0.3  # 黑潮
        return np.clip(base, 0.01, 1.5)

    def estimate_front_strength(self, lat, lon):
        if 20 < lat < 30:
            return np.clip(0.08 + np.random.normal(0, 0.02), 0, 0.2)
        elif 10 < lat < 20:
            return np.clip(0.05 + np.random.normal(0, 0.02), 0, 0.2)
        return np.clip(0.02 + np.random.normal(0, 0.01), 0, 0.2)

    def estimate_eddy_strength(self, lat, lon):
        base = 100 + 50 * np.sin(2 * np.pi * (lat - 10) / 25)
        return np.clip(base + np.random.normal(0, 20), 10, 500)


# ═══════════════════════════════════════════════════
#  代謝指數 & HSI 計算
# ═══════════════════════════════════════════════════

METABOLIC_TRAITS = {
    "yellowfin": {"Eo": 0.40, "Pcrit_kPa": 4.8, "Topt_C": 28.0, "Topt_sigma": 2.0},
    "bigeye":    {"Eo": 0.35, "Pcrit_kPa": 3.5, "Topt_C": 18.0, "Topt_sigma": 3.0},
    "skipjack":  {"Eo": 0.45, "Pcrit_kPa": 5.5, "Topt_C": 26.0, "Topt_sigma": 3.5},
    "albacore":  {"Eo": 0.38, "Pcrit_kPa": 4.0, "Topt_C": 20.0, "Topt_sigma": 3.0},
}

KB = 8.617e-5
TREF_K = 288.15

def compute_phi(sst_c, do_ml_l, species="yellowfin"):
    traits = METABOLIC_TRAITS.get(species, METABOLIC_TRAITS["yellowfin"])
    T_K = sst_c + 273.15
    do_kpa = do_ml_l * 1.42 * 0.2095 * 101.325 / 100
    supply = do_kpa * np.exp(-traits["Eo"] / KB * (1/T_K - 1/TREF_K))
    demand = traits["Pcrit_kPa"]
    return max(supply / max(demand, 0.01), 0.01)

def compute_hsi(sst_c, chl, species="yellowfin"):
    traits = METABOLIC_TRAITS.get(species, METABOLIC_TRAITS["yellowfin"])
    thermal = np.exp(-0.5 * ((sst_c - traits["Topt_C"]) / traits["Topt_sigma"]) ** 2)
    feeding = np.tanh(chl / 0.3)
    return np.sqrt(thermal * feeding)


# ═══════════════════════════════════════════════════
#  WCPFC 數據解析器
# ═══════════════════════════════════════════════════

SPECIES_CODE_MAP = {
    # WCPFC column names → our species names
    "yft_c": "yellowfin", "yft_n": "yellowfin",
    "bet_c": "bigeye",    "bet_n": "bigeye",
    "skj_c": "skipjack",  "skj_n": "skipjack",
    "alb_c": "albacore",  "alb_n": "albacore",
}


def parse_wcpfc(filepath, target_species="yellowfin"):
    """
    解析 WCPFC 公開數據格式。
    
    WCPFC 典型欄位:
      yy, mm, lat5, lon5, hhooks, yft_c, bet_c, skj_c, alb_c
    
    或:
      year, month, lat_short, lon_short, hooks, catch_mt, effort
    """
    log.info(f"Reading WCPFC data from: {filepath}")
    df = pd.read_csv(filepath)
    
    # 標準化欄位名稱 (小寫)
    df.columns = [c.strip().lower() for c in df.columns]
    
    log.info(f"  Columns found: {list(df.columns)}")
    log.info(f"  Rows: {len(df)}")
    
    # 年月
    year_col = _find_col(df, ["yy", "year", "yr"])
    month_col = _find_col(df, ["mm", "month", "mon"])
    
    if year_col is None or month_col is None:
        raise ValueError("找不到年份(year/yy)或月份(month/mm)欄位")
    
    # 經緯度
    lat_col = _find_col(df, ["lat5", "lat_short", "lat", "latitude"])
    lon_col = _find_col(df, ["lon5", "lon_short", "lon", "longitude"])
    
    if lat_col is None or lon_col is None:
        raise ValueError("找不到經緯度欄位")
    
    # 漁獲量
    species_abbrev = {"yellowfin": "yft", "bigeye": "bet", 
                      "skipjack": "skj", "albacore": "alb"}
    abbrev = species_abbrev.get(target_species, "yft")
    
    catch_col = _find_col(df, [f"{abbrev}_c", f"{abbrev}_mt", f"{abbrev}_catch",
                                "catch", "catch_mt", "catch_kg"])
    effort_col = _find_col(df, ["hhooks", "hooks", "effort", "days", "sets",
                                 "fishing_days", "effort_hooks"])
    
    if catch_col is None:
        raise ValueError(f"找不到 {target_species} 的漁獲量欄位 "
                         f"(嘗試過: {abbrev}_c, catch, catch_mt)")
    
    log.info(f"  Using: year={year_col}, month={month_col}, "
             f"lat={lat_col}, lon={lon_col}, catch={catch_col}, effort={effort_col}")
    
    # 建構輸出
    result = pd.DataFrame()
    result["year"] = df[year_col].astype(int)
    result["month"] = df[month_col].astype(int)
    result["lat"] = df[lat_col].astype(float)
    result["lon"] = df[lon_col].astype(float)
    
    # 計算 CPUE
    catch = df[catch_col].astype(float)
    
    if effort_col:
        effort = df[effort_col].astype(float)
        # 如果是 hundred hooks，轉成天數估算 (1000 hooks ≈ 1 fishing day for longline)
        if "hooks" in effort_col.lower() or "hhooks" in effort_col.lower():
            if "hhooks" in effort_col.lower():
                effort = effort * 100  # hundred hooks → hooks
            # 1000 hooks per day (typical longline)
            effort_days = effort / 1000
        else:
            effort_days = effort
        
        effort_days = effort_days.replace(0, np.nan)
        result["cpue_kg_per_day"] = (catch / effort_days).round(2)
    else:
        # 沒有effort，直接用catch當代理
        log.warning("  ⚠ 沒有找到effort欄位，用catch直接作為CPUE代理")
        result["cpue_kg_per_day"] = catch.round(2)
    
    # 過濾
    result = result.dropna(subset=["cpue_kg_per_day"])
    result = result[result["cpue_kg_per_day"] > 0]
    
    # 過濾研究區域
    result = result[
        (result["lat"] >= 5) & (result["lat"] <= 35) &
        (result["lon"] >= 120) & (result["lon"] <= 175)
    ]
    
    result["species"] = target_species
    
    log.info(f"  Valid records: {len(result)}")
    log.info(f"  CPUE range: {result['cpue_kg_per_day'].min():.1f} - "
             f"{result['cpue_kg_per_day'].max():.1f} kg/day")
    log.info(f"  CPUE median: {result['cpue_kg_per_day'].median():.1f} kg/day")
    
    return result


def parse_custom(filepath, target_species="yellowfin"):
    """
    解析通用格式 CSV。
    
    最低要求欄位: year, month, lat, lon, cpue (或 catch + effort)
    """
    log.info(f"Reading custom data from: {filepath}")
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns]
    
    log.info(f"  Columns: {list(df.columns)}")
    
    result = pd.DataFrame()
    
    # 年月
    year_col = _find_col(df, ["year", "yy", "yr"])
    month_col = _find_col(df, ["month", "mm", "mon"])
    
    if year_col:
        result["year"] = df[year_col].astype(int)
    else:
        raise ValueError("需要 year 欄位")
    
    if month_col:
        result["month"] = df[month_col].astype(int)
    else:
        result["month"] = 6  # 預設6月
    
    # 經緯度
    lat_col = _find_col(df, ["lat", "latitude", "lat5"])
    lon_col = _find_col(df, ["lon", "longitude", "lon5", "lng"])
    result["lat"] = df[lat_col].astype(float)
    result["lon"] = df[lon_col].astype(float)
    
    # CPUE
    cpue_col = _find_col(df, ["cpue", "cpue_kg_per_day", "cpue_kg", "catch_per_day",
                               "cpue_mt", "catch_rate"])
    if cpue_col:
        result["cpue_kg_per_day"] = df[cpue_col].astype(float)
    else:
        catch_col = _find_col(df, ["catch", "catch_mt", "catch_kg", "weight"])
        effort_col = _find_col(df, ["effort", "days", "fishing_days", "hooks", "sets"])
        if catch_col and effort_col:
            effort = df[effort_col].astype(float).replace(0, np.nan)
            result["cpue_kg_per_day"] = (df[catch_col].astype(float) / effort).round(2)
        else:
            raise ValueError("需要 cpue 欄位，或 catch + effort 欄位")
    
    result["species"] = target_species
    result = result.dropna(subset=["cpue_kg_per_day"])
    result = result[result["cpue_kg_per_day"] > 0]
    
    log.info(f"  Valid records: {len(result)}")
    return result


def _find_col(df, candidates):
    """在 DataFrame 中搜尋匹配的欄位名稱"""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ═══════════════════════════════════════════════════
#  環境數據補全 (用氣候態)
# ═══════════════════════════════════════════════════

def enrich_with_environment(df, species="yellowfin"):
    """
    為漁獲數據添加環境變量特徵。
    
    目前用氣候態估算。未來可以替換為:
    - ERA5 再分析數據 (SST, wind)
    - MODIS Level-3 Chl-a
    - WOA 2023 DO, temperature profiles
    - AVISO SSH, currents
    """
    log.info("Enriching with environmental data (climatology)...")
    
    est = ClimatologyEstimator()
    np.random.seed(42)
    
    env_data = []
    for _, row in df.iterrows():
        lat, lon, month = row["lat"], row["lon"], int(row["month"])
        
        sst = est.estimate_sst(lat, lon, month)
        chl = est.estimate_chl(lat, lon, month)
        ssh = est.estimate_ssh(lat, lon, month)
        do = est.estimate_do(lat, lon, month, sst)
        cs = est.estimate_current_speed(lat, lon)
        fs = est.estimate_front_strength(lat, lon)
        es = est.estimate_eddy_strength(lat, lon)
        phi = compute_phi(sst, do, species)
        hsi = compute_hsi(sst, chl, species)
        
        env_data.append({
            "sst": round(sst, 2),
            "chl": round(chl, 4),
            "ssh": round(ssh, 4),
            "do": round(do, 2),
            "current_speed": round(cs, 3),
            "front_strength": round(fs, 4),
            "eddy_strength": round(es, 1),
            "phi": round(phi, 3),
            "hsi": round(hsi, 4),
        })
    
    env_df = pd.DataFrame(env_data)
    result = pd.concat([df.reset_index(drop=True), env_df], axis=1)
    
    # 添加時間欄位
    result["day_of_year"] = result.apply(
        lambda r: datetime(int(r["year"]), int(r["month"]), 15).timetuple().tm_yday,
        axis=1
    )
    result["date"] = result.apply(
        lambda r: f"{int(r['year'])}-{int(r['month']):02d}-15",
        axis=1
    )
    
    # 排序欄位 (與合成數據一致)
    cols = ["date", "year", "month", "day_of_year", "lat", "lon",
            "sst", "chl", "ssh", "do", "current_speed",
            "front_strength", "eddy_strength", "phi", "hsi",
            "cpue_kg_per_day", "species"]
    
    result = result[cols].sort_values(["year", "month"]).reset_index(drop=True)
    
    log.info(f"  Enriched {len(result)} records with {len(env_data[0])} environmental features")
    
    return result


# ═══════════════════════════════════════════════════
#  自動偵測格式
# ═══════════════════════════════════════════════════

def detect_format(filepath):
    """嘗試自動偵測CSV格式"""
    df = pd.read_csv(filepath, nrows=5)
    cols = [c.strip().lower() for c in df.columns]
    
    # WCPFC 特徵: 有 yft_c, bet_c 或 hhooks
    if any(c in cols for c in ["yft_c", "bet_c", "hhooks", "lat5", "lon5"]):
        return "wcpfc"
    
    # 通用格式
    return "custom"


# ═══════════════════════════════════════════════════
#  主程式
# ═══════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="轉換真實漁獲數據為ML訓練格式")
    parser.add_argument("--input", "-i", required=True, help="輸入CSV路徑")
    parser.add_argument("--format", "-f", default="auto", 
                        choices=["auto", "wcpfc", "custom"],
                        help="數據格式 (auto=自動偵測)")
    parser.add_argument("--species", "-s", default="yellowfin",
                        help="目標魚種 (yellowfin/bigeye/skipjack/albacore)")
    parser.add_argument("--output", "-o", default=None,
                        help="輸出路徑 (預設: ml_system/data/simulated_cpue_{species}.csv)")
    args = parser.parse_args()
    
    # 偵測格式
    fmt = args.format
    if fmt == "auto":
        fmt = detect_format(args.input)
        log.info(f"Auto-detected format: {fmt}")
    
    # 解析
    if fmt == "wcpfc":
        df = parse_wcpfc(args.input, args.species)
    else:
        df = parse_custom(args.input, args.species)
    
    if len(df) < 50:
        log.warning(f"⚠ 只有 {len(df)} 條記錄，建議至少 200+ 條才能訓練出有意義的模型")
    
    # 補全環境數據
    df = enrich_with_environment(df, args.species)
    
    # 儲存
    output_path = args.output or f"ml_system/data/simulated_cpue_{args.species}.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 70)
    print(f"✅ 轉換完成!")
    print(f"   記錄數: {len(df)}")
    print(f"   輸出: {output_path}")
    print(f"   CPUE: {df['cpue_kg_per_day'].median():.1f} kg/day (中位數)")
    print(f"   範圍: {df['cpue_kg_per_day'].min():.1f} - {df['cpue_kg_per_day'].max():.1f}")
    print(f"   年份: {int(df['year'].min())} - {int(df['year'].max())}")
    print("=" * 70)
    print(f"\n下一步: 用真實數據重新訓練模型:")
    print(f"  python run_pipeline.py --data-path {output_path}")
    
    return df


if __name__ == "__main__":
    main()
