"""
Parse WCPFC LONGLINE.CSV → historical_prior_longline.json

Output format:
{
  "meta": { "source": "WCPFC", "records": N, ... },
  "prior": {
    "YFT": { "1": [{"lat": 5.0, "lon": 140.0, "cpue": 2.3}, ...], "2": [...], ..., "12": [...] },
    "BET": { ... },
    "ALB": { ... }
  }
}

CPUE = catch_weight_mt / (hhooks / 1000)  → mt per 1000 hooks
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

LONGLINE_CSV = Path("data/wcpfc/LONGLINE.CSV")
OUTPUT = Path("data/wcpfc/historical_prior_longline.json")

# Species code → standard name mapping
SPECIES_MAP = {
    "yft": "yellowfin",
    "bet": "bigeye",
    "alb": "albacore",
    "swo": "swordfish",
    "mls": "striped_marlin",
    "blm": "black_marlin",
    "bum": "blue_marlin",
}


def parse_coord(s: str) -> float:
    """Convert WCPFC lat5/lon5 (e.g. '05N', '140E') to decimal degrees."""
    s = s.strip().strip('"')
    if not s:
        return None
    direction = s[-1].upper()
    value = float(s[:-1])
    if direction in ('S', 'W'):
        value = -value
    return value


def main():
    # ── Read LONGLINE.CSV ──
    print(f"Reading {LONGLINE_CSV} ...")
    rows = []
    with open(LONGLINE_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"  Total rows: {len(rows)}")

    # ── Aggregate: per (species, month, lat5, lon5) → sum(catch), sum(hhooks), count ──
    # Structure: agg[species][month][(lat, lon)] = {"catch": sum, "hooks": sum, "count": n}
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"catch": 0.0, "hooks": 0.0, "count": 0})))
    n_parsed = 0
    n_skipped = 0

    for row in rows:
        try:
            mm = int(row["mm"])
            lat = parse_coord(row["lat5"])
            lon = parse_coord(row["lon5"])
            hhooks = float(row.get("hhooks", 0) or 0)

            if lat is None or lon is None:
                n_skipped += 1
                continue

            # Parse each species
            for code, name in SPECIES_MAP.items():
                catch_col = f"{code}_c"
                if catch_col in row:
                    catch = float(row.get(catch_col, 0) or 0)
                    if catch > 0 or hhooks > 0:
                        entry = agg[name][mm][(lat, lon)]
                        entry["catch"] += catch
                        entry["hooks"] += hhooks
                        entry["count"] += 1

            n_parsed += 1
        except (ValueError, KeyError) as e:
            n_skipped += 1

    print(f"  Parsed: {n_parsed}, Skipped: {n_skipped}")

    # ── Compute average CPUE per (species, month, grid_cell) ──
    prior = {}
    total_cells = 0
    for species, months in agg.items():
        prior[species] = {}
        for mm, cells in months.items():
            month_data = []
            for (lat, lon), vals in cells.items():
                if vals["hooks"] > 0:
                    # CPUE = mt / (1000 hooks)
                    cpue = vals["catch"] / (vals["hooks"] / 1000.0) if vals["hooks"] > 0 else 0
                else:
                    cpue = 0
                if cpue > 0:
                    month_data.append({
                        "lat": lat,
                        "lon": lon,
                        "cpue": round(cpue, 4),
                        "n_years": vals["count"],
                    })
            # Sort by CPUE descending
            month_data.sort(key=lambda x: x["cpue"], reverse=True)
            prior[species][str(mm)] = month_data
            total_cells += len(month_data)

    # ── Summary ──
    print(f"\n  Species breakdown:")
    for sp, months in prior.items():
        total = sum(len(v) for v in months.values())
        print(f"    {sp}: {total} grid-month cells")
    print(f"  Total grid-month cells: {total_cells}")

    # ── Save ──
    output = {
        "meta": {
            "source": "WCPFC Scientific Committee",
            "file": "LONGLINE.CSV",
            "total_rows": len(rows),
            "parsed": n_parsed,
            "species": list(prior.keys()),
            "grid_resolution_deg": 5,
            "cpue_unit": "mt_per_1000_hooks",
        },
        "prior": prior,
    }

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    size_mb = OUTPUT.stat().st_size / (1024 * 1024)
    print(f"\n  Saved: {OUTPUT} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
