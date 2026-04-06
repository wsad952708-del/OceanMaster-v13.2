"""
OceanMaster v13.2 — WCPFC Real Data Loader  # [v12-phase11-wcpfc]
================================================================
Parse actual WCPFC public domain CSV files (longline & purse seine).
Handles LAT5/LON5 coordinate format, HHOOKS/DAYS effort, species columns.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, List

log = logging.getLogger("OceanMaster.WCPFC")


# ── Coordinate Parsing ──────────────────────────────

def _parse_lat5(val: str) -> float:
    """'15N' → 15.0, '05S' → -5.0, '00N' → 0.0"""
    val = str(val).strip().strip('"')
    if not val:
        return np.nan
    direction = val[-1].upper()
    try:
        deg = float(val[:-1])
    except ValueError:
        return np.nan
    return deg if direction == "N" else -deg


def _parse_lon5(val: str) -> float:
    """'120E' → 120.0, '170W' → -170.0"""
    val = str(val).strip().strip('"')
    if not val:
        return np.nan
    direction = val[-1].upper()
    try:
        deg = float(val[:-1])
    except ValueError:
        return np.nan
    return deg if direction == "E" else -deg


# ── Longline Loader ─────────────────────────────────

def load_longline_monthly(csv_path: str, min_year: int = 2000) -> pd.DataFrame:
    """
    Load WCPFC_L_PUBLIC_BY_YY_MM_FLAG.csv → cleaned DataFrame.

    Columns: YY, MM, flag_code, LAT5, LON5, cwp_grid, HHOOKS,
             ALB_C, ALB_N, YFT_C, YFT_N, BET_C, BET_N, ...

    HHOOKS = hundreds of hooks. Effort = HHOOKS * 100.
    *_C = catch in metric tons. *_N = number of fish.

    Returns long-form DataFrame:
        year, month, lat, lon, species, catch_mt, effort_hooks, cpue
    """
    log.info(f"Loading WCPFC longline: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str)
    df.columns = [c.strip().strip('"') for c in df.columns]

    # Parse coordinates
    df["year"] = pd.to_numeric(df["YY"], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["MM"], errors="coerce").astype("Int64")
    df["lat"] = df["LAT5"].apply(_parse_lat5)
    df["lon"] = df["LON5"].apply(_parse_lon5)
    df["hhooks"] = pd.to_numeric(df["HHOOKS"], errors="coerce").fillna(0)

    # Filter
    df = df[df["year"] >= min_year].copy()
    df = df[df["hhooks"] > 0].copy()  # 3-vessel rule: 0 = redacted  # [v12-phase11-wcpfc]

    # Species catch columns (metric tons)
    species_cols = {
        "ALB_C": "albacore",
        "YFT_C": "yellowfin",
        "BET_C": "bigeye",
    }

    records = []
    for idx, row in df.iterrows():
        effort_hooks = row["hhooks"] * 100  # HHOOKS = hundreds of hooks
        for col, species in species_cols.items():
            if col not in df.columns:
                continue
            catch = pd.to_numeric(row.get(col, 0), errors="coerce")
            if pd.isna(catch) or catch <= 0:
                continue
            cpue = catch / (effort_hooks / 1000)  # MT per 1000 hooks
            records.append({
                "year": row["year"],
                "month": row["month"],
                "lat": row["lat"] + 2.5,  # center of 5° cell
                "lon": row["lon"] + 2.5,
                "species": species,
                "catch_mt": float(catch),
                "effort_hooks": float(effort_hooks),
                "cpue": float(cpue),
                "gear": "longline",
            })

    result = pd.DataFrame(records)
    log.info(f"  Longline: {len(result)} records "
             f"({result['species'].value_counts().to_dict()})")
    return result


# ── Purse Seine Loader ──────────────────────────────

def load_purse_seine_monthly(csv_path: str, min_year: int = 2000) -> pd.DataFrame:
    """
    Load WCPFC_S_PUBLIC_BY_YY_MM.csv → cleaned DataFrame.

    Columns: YY, MM, LAT5, LON5, cwp_grid, DAYS, SETS_UNA, ...,
             SKJ_C_UNA, YFT_C_UNA, BET_C_UNA, ...

    DAYS = fishing days. Catch split by set type (UNA=unassociated, LOG, DFAD, AFAD, OTH).

    Returns long-form DataFrame:
        year, month, lat, lon, species, catch_mt, effort_days, cpue, gear
    """
    log.info(f"Loading WCPFC purse seine: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str)
    df.columns = [c.strip().strip('"') for c in df.columns]

    df["year"] = pd.to_numeric(df["YY"], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["MM"], errors="coerce").astype("Int64")
    df["lat"] = df["LAT5"].apply(_parse_lat5)
    df["lon"] = df["LON5"].apply(_parse_lon5)
    df["days"] = pd.to_numeric(df["DAYS"], errors="coerce").fillna(0)

    df = df[df["year"] >= min_year].copy()
    df = df[df["days"] > 0].copy()  # 3-vessel rule  # [v12-phase11-wcpfc]

    # Species: sum all set types (UNA + LOG + DFAD + AFAD + OTH)
    species_prefixes = {
        "SKJ": "skipjack",
        "YFT": "yellowfin",
        "BET": "bigeye",
    }
    set_types = ["UNA", "LOG", "DFAD", "AFAD", "OTH"]

    records = []
    for idx, row in df.iterrows():
        days = row["days"]
        for prefix, species in species_prefixes.items():
            total_catch = 0.0
            for st in set_types:
                col = f"{prefix}_C_{st}"
                if col in df.columns:
                    val = pd.to_numeric(row.get(col, 0), errors="coerce")
                    if not pd.isna(val):
                        total_catch += val

            if total_catch <= 0:
                continue

            cpue = total_catch / days  # MT per day
            records.append({
                "year": row["year"],
                "month": row["month"],
                "lat": row["lat"] + 2.5,
                "lon": row["lon"] + 2.5,
                "species": species,
                "catch_mt": float(total_catch),
                "effort_days": float(days),
                "cpue": float(cpue),
                "gear": "purse_seine",
            })

    result = pd.DataFrame(records)
    log.info(f"  Purse seine: {len(result)} records "
             f"({result['species'].value_counts().to_dict()})")
    return result


# ── Combined Loader ─────────────────────────────────

def combine_all_wcpfc(
    data_dir: str,
    min_year: int = 2000,
) -> pd.DataFrame:
    """
    Auto-detect and load all WCPFC CSV files from a directory.

    Searches for:
    - WCPFC_L_*.csv → longline
    - WCPFC_S_*.csv → purse seine

    Returns unified DataFrame with columns:
        year, month, lat, lon, species, catch_mt, cpue, gear
    """
    data_path = Path(data_dir)
    frames: List[pd.DataFrame] = []

    # Search recursively
    for csv_file in sorted(data_path.rglob("*.csv")):
        name = csv_file.name.upper()
        if "COVERAGE" in name:
            continue  # skip coverage metadata files

        if "WCPFC_L_" in name and "PUBLIC" in name:
            try:
                df = load_longline_monthly(str(csv_file), min_year=min_year)
                if len(df) > 0:
                    frames.append(df)
            except Exception as e:
                log.warning(f"  Skip {csv_file.name}: {e}")

        elif "WCPFC_S_" in name and "PUBLIC" in name:
            try:
                df = load_purse_seine_monthly(str(csv_file), min_year=min_year)
                if len(df) > 0:
                    frames.append(df)
            except Exception as e:
                log.warning(f"  Skip {csv_file.name}: {e}")

    if not frames:
        raise FileNotFoundError(
            f"No WCPFC CSV files found in {data_dir}. "
            "Expected WCPFC_L_*.csv (longline) or WCPFC_S_*.csv (purse seine)."
        )

    combined = pd.concat(frames, ignore_index=True)
    log.info(f"  Combined WCPFC: {len(combined)} total records, "
             f"species: {combined['species'].value_counts().to_dict()}")
    return combined


# ── CLI ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if len(sys.argv) < 2:
        print("Usage: python wcpfc_data_loader.py <data_dir>")
        print("  data_dir: path to directory containing WCPFC CSV files")
        sys.exit(1)

    data_dir = sys.argv[1]
    df = combine_all_wcpfc(data_dir, min_year=2000)
    print(f"\n{'='*60}")
    print(f"Total records: {len(df)}")
    print(f"\nSpecies breakdown:")
    print(df.groupby("species").agg(
        records=("cpue", "count"),
        mean_cpue=("cpue", "mean"),
        median_cpue=("cpue", "median"),
    ).round(4))
    print(f"\nYear range: {df['year'].min()} — {df['year'].max()}")
    print(f"Gear types: {df['gear'].value_counts().to_dict()}")
