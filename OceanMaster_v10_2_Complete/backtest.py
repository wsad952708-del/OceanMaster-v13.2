#!/usr/bin/env python3
"""
OceanMaster v10.3 — 回測框架
============================
用歷史日期重跑預測，與實際漁獲比對。

用法:
  python backtest.py --start 2023-01 --end 2023-12
  python backtest.py --start 2024-01 --end 2024-06 --wcpfc data/wcpfc/

輸出:
  output/backtest/ 目錄下產生:
    - backtest_report_{start}_{end}.json  — 完整報告
    - backtest_summary_{start}_{end}.txt  — 可讀摘要
"""

import sys
import os
import json
import asyncio
import argparse
import logging
import time
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("OceanMaster.Backtest")


def parse_month_range(start_str: str, end_str: str):
    """解析 YYYY-MM 格式的月份範圍"""
    sy, sm = map(int, start_str.split("-"))
    ey, em = map(int, end_str.split("-"))
    months = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def load_wcpfc_data(wcpfc_dir: str) -> dict:
    """
    載入 WCPFC 公開數據。
    返回 {(year, month): DataFrame-like list of records}
    """
    import pandas as pd
    wcpfc_path = Path(wcpfc_dir)

    if not wcpfc_path.exists():
        log.info(f"  WCPFC 目錄不存在: {wcpfc_dir}")
        return {}

    records = {}
    csv_files = list(wcpfc_path.glob("*.csv"))
    if not csv_files:
        log.info(f"  WCPFC 目錄中沒有 CSV 檔案")
        return {}

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # WCPFC format: yy, mm, lat5, lon5, hhooks, alb_c, yft_c, bet_c, skj_c
            for _, row in df.iterrows():
                yr = int(row.get("yy", row.get("year", 0)))
                mo = int(row.get("mm", row.get("month", 0)))
                if yr < 100:
                    yr += 2000
                if (yr, mo) not in records:
                    records[(yr, mo)] = []

                lat = float(row.get("lat5", row.get("lat", 0)))
                lon = float(row.get("lon5", row.get("lon", 0)))
                hooks = float(row.get("hhooks", row.get("effort", 1))) * 100

                # CPUE per species
                for sp_col, sp_name in [
                    ("alb_c", "albacore"), ("yft_c", "yellowfin"),
                    ("bet_c", "bigeye"), ("skj_c", "skipjack"),
                ]:
                    catch = float(row.get(sp_col, 0))
                    if catch > 0 and hooks > 0:
                        records[(yr, mo)].append({
                            "lat": lat, "lon": lon,
                            "species": sp_name,
                            "catch": catch,
                            "effort": hooks,
                            "cpue": catch / hooks * 1000,  # per 1000 hooks
                        })
        except Exception as e:
            log.warning(f"  WCPFC 讀取錯誤 {csv_file.name}: {e}")

    log.info(f"  WCPFC 數據: {len(records)} 個月, "
             f"{sum(len(v) for v in records.values())} 筆記錄")
    return records


def load_fishing_json(json_path: str) -> dict:
    """嘗試載入 捕魚(18).json 備用數據"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            log.info(f"  JSON 數據: {len(data)} 筆記錄")
            return {"records": data}
        elif isinstance(data, dict):
            log.info(f"  JSON 數據: {list(data.keys())[:5]}")
            return data
        return {}
    except Exception as e:
        log.warning(f"  JSON 讀取錯誤: {e}")
        return {}


def compute_lift_ratio(
    hotspots: list,
    actual_records: list,
    top_n: int = 10,
) -> dict:
    """
    計算 Lift Ratio = top_n 熱點區域平均 CPUE / 全區域平均 CPUE

    Args:
        hotspots: 預測的熱點列表 (需要 lat, lon, species)
        actual_records: 實際漁獲記錄 (需要 lat, lon, cpue)
        top_n: 取前幾個熱點

    Returns:
        {"lift_ratio": float, "hotspot_cpue": float,
         "baseline_cpue": float, "n_matched": int}
    """
    if not actual_records or not hotspots:
        return {"lift_ratio": None, "note": "無足夠數據計算"}

    # 全區域平均 CPUE
    all_cpue = [r["cpue"] for r in actual_records if r.get("cpue", 0) > 0]
    if not all_cpue:
        return {"lift_ratio": None, "note": "實際 CPUE 全為零"}

    baseline_cpue = np.mean(all_cpue)

    # Top-N 熱點區域的實際 CPUE
    # 每個熱點匹配 ±2.5° 內的實際漁獲
    hotspot_cpues = []
    n_matched = 0

    for h in hotspots[:top_n]:
        h_lat, h_lon = h.get("lat", 0), h.get("lon", 0)
        # 找距離熱點 ±2.5° 的記錄
        for r in actual_records:
            if (abs(r["lat"] - h_lat) <= 2.5 and
                abs(r["lon"] - h_lon) <= 2.5 and
                r.get("cpue", 0) > 0):
                hotspot_cpues.append(r["cpue"])
                n_matched += 1

    if not hotspot_cpues:
        return {"lift_ratio": None, "hotspot_cpue": 0,
                "baseline_cpue": float(baseline_cpue),
                "n_matched": 0,
                "note": "熱點區域無匹配的實際漁獲"}

    hotspot_cpue = np.mean(hotspot_cpues)
    lift = hotspot_cpue / baseline_cpue if baseline_cpue > 0 else 0

    return {
        "lift_ratio": round(float(lift), 3),
        "hotspot_cpue": round(float(hotspot_cpue), 2),
        "baseline_cpue": round(float(baseline_cpue), 2),
        "n_matched": n_matched,
        "top_n": top_n,
    }


async def run_backtest(args):
    """執行完整回測"""
    from main_v10_3 import OceanMasterPipeline

    months = parse_month_range(args.start, args.end)
    log.info(f"\n{'='*65}")
    log.info(f"  OceanMaster 回測: {args.start} → {args.end}")
    log.info(f"  共 {len(months)} 個月")
    log.info(f"{'='*65}")

    # 載入驗證數據
    wcpfc_data = {}
    fishing_json = {}

    if args.wcpfc and Path(args.wcpfc).exists():
        wcpfc_data = load_wcpfc_data(args.wcpfc)
    else:
        log.info("  WCPFC 數據未提供或不存在")

    # 檢查 捕魚(18).json
    fishing_json_path = Path(__file__).parent.parent / "捕魚 (18).json"
    if fishing_json_path.exists():
        fishing_json = load_fishing_json(str(fishing_json_path))
    else:
        # 也嘗試在同級目錄
        for p in [
            Path(__file__).parent / "捕魚 (18).json",
            Path("捕魚 (18).json"),
        ]:
            if p.exists():
                fishing_json = load_fishing_json(str(p))
                break

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "start": args.start,
        "end": args.end,
        "total_months": len(months),
        "monthly_results": [],
        "wcpfc_available": bool(wcpfc_data),
        "fishing_json_available": bool(fishing_json),
    }

    lift_ratios = []
    total_hotspots = 0
    t0 = time.time()

    for i, (year, month) in enumerate(months):
        date_str = f"{year}-{month:02d}-15"
        log.info(f"\n{'━'*50}")
        log.info(f"  [{i+1}/{len(months)}] 分析 {year}-{month:02d} (日期: {date_str})")
        log.info(f"{'━'*50}")

        month_result = {
            "year": year,
            "month": month,
            "date": date_str,
        }

        try:
            pipeline = OceanMasterPipeline(
                lat_range=(args.lat_min, args.lat_max),
                lon_range=(args.lon_min, args.lon_max),
                vessel_pos=(args.vessel_lat, args.vessel_lon),
                output_dir=str(output_dir / f"backtest_{year}_{month:02d}"),
                target_date=date_str,
            )

            r = await pipeline.run_full()
            hotspots = r.get("hotspots", [])
            total_hotspots += len(hotspots)

            month_result["n_hotspots"] = len(hotspots)
            month_result["elapsed_s"] = round(r.get("elapsed_seconds", 0), 1)
            month_result["data_sources"] = r.get("data_sources", {})

            # Top 5 hotspots summary
            month_result["top5"] = [
                {
                    "rank": j + 1,
                    "lat": h.get("lat"),
                    "lon": h.get("lon"),
                    "species": h.get("species"),
                    "score": h.get("score"),
                }
                for j, h in enumerate(hotspots[:5])
            ]

            # 驗證: 計算 lift ratio
            actual = None
            validation_source = None

            if (year, month) in wcpfc_data:
                actual = wcpfc_data[(year, month)]
                validation_source = "WCPFC"
            elif fishing_json and "records" in fishing_json:
                # 嘗試從 JSON 找同月數據
                json_records = [
                    r for r in fishing_json["records"]
                    if r.get("year") == year and r.get("month") == month
                ]
                if json_records:
                    actual = json_records
                    validation_source = "捕魚JSON"

            if actual:
                lift = compute_lift_ratio(hotspots, actual)
                month_result["validation"] = {
                    "source": validation_source,
                    "n_actual_records": len(actual),
                    **lift,
                }
                if lift.get("lift_ratio") is not None:
                    lift_ratios.append(lift["lift_ratio"])
                    log.info(f"  ✅ Lift Ratio = {lift['lift_ratio']:.2f}x "
                             f"({validation_source}, {lift['n_matched']} matched)")
            else:
                month_result["validation"] = {
                    "source": None,
                    "note": "無驗證數據，待補充",
                }
                log.info("  ⚠️ 無驗證數據，待補充")

            month_result["status"] = "success"

        except Exception as e:
            log.error(f"  ❌ 分析失敗: {e}")
            month_result["status"] = "failed"
            month_result["error"] = str(e)

        results["monthly_results"].append(month_result)

    # ─── 彙總統計 ───
    elapsed = time.time() - t0
    successful = [m for m in results["monthly_results"] if m["status"] == "success"]

    results["summary"] = {
        "total_elapsed_s": round(elapsed, 1),
        "successful_months": len(successful),
        "failed_months": len(months) - len(successful),
        "total_hotspots": total_hotspots,
        "avg_hotspots_per_month": round(
            total_hotspots / max(len(successful), 1), 1
        ),
    }

    if lift_ratios:
        results["summary"]["lift_ratio"] = {
            "mean": round(float(np.mean(lift_ratios)), 3),
            "median": round(float(np.median(lift_ratios)), 3),
            "min": round(float(np.min(lift_ratios)), 3),
            "max": round(float(np.max(lift_ratios)), 3),
            "n_months_validated": len(lift_ratios),
            "months_above_1x": sum(1 for lr in lift_ratios if lr > 1.0),
        }
        log.info(f"\n  📊 Lift Ratio 平均: {results['summary']['lift_ratio']['mean']:.2f}x")
        log.info(f"  📊 {results['summary']['lift_ratio']['months_above_1x']}/{len(lift_ratios)} 個月 > 1.0x")
    else:
        results["summary"]["lift_ratio"] = {
            "note": "無驗證數據可計算 lift ratio",
        }

    # ─── 儸出報告 ───
    report_path = output_dir / f"backtest_report_{args.start}_{args.end}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    log.info(f"\n  📄 報告: {report_path}")

    # ─── 文字摘要 ───
    summary_path = output_dir / f"backtest_summary_{args.start}_{args.end}.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"OceanMaster v10.3 回測報告\n")
        f.write(f"{'='*50}\n")
        f.write(f"期間: {args.start} → {args.end}\n")
        f.write(f"範圍: {args.lat_min}-{args.lat_max}N, {args.lon_min}-{args.lon_max}E\n")
        f.write(f"總耗時: {elapsed:.0f}s\n\n")

        f.write(f"成功月份: {len(successful)}/{len(months)}\n")
        f.write(f"總熱點數: {total_hotspots}\n")
        f.write(f"平均熱點/月: {results['summary']['avg_hotspots_per_month']}\n\n")

        if lift_ratios:
            lr = results["summary"]["lift_ratio"]
            f.write(f"Lift Ratio 驗證結果\n")
            f.write(f"{'-'*30}\n")
            f.write(f"  平均: {lr['mean']:.2f}x\n")
            f.write(f"  中位數: {lr['median']:.2f}x\n")
            f.write(f"  範圍: {lr['min']:.2f}x - {lr['max']:.2f}x\n")
            f.write(f"  > 1.0x 月份: {lr['months_above_1x']}/{lr['n_months_validated']}\n\n")

            # 評價
            if lr["mean"] >= 2.0:
                f.write("  ✅ 預測品質: 優 (lift ≥ 2.0)\n")
            elif lr["mean"] >= 1.5:
                f.write("  ✅ 預測品質: 良 (lift ≥ 1.5)\n")
            elif lr["mean"] >= 1.0:
                f.write("  ⚠️ 預測品質: 可 (lift ≥ 1.0)\n")
            else:
                f.write("  ❌ 預測品質: 差 (lift < 1.0)\n")
        else:
            f.write("⚠️ 無驗證數據\n")
            f.write("請提供 WCPFC CSV 或漁獲 JSON 以啟用 lift ratio 驗證。\n")

        f.write(f"\n月份詳情:\n")
        f.write(f"{'-'*50}\n")
        for m in results["monthly_results"]:
            v = m.get("validation", {})
            lr_str = f"lift={v.get('lift_ratio', 'N/A')}" if v.get("lift_ratio") else v.get("note", "N/A")
            f.write(f"  {m['year']}-{m['month']:02d}: "
                    f"{'✅' if m['status'] == 'success' else '❌'} "
                    f"{m.get('n_hotspots', 0)} hotspots, {lr_str}\n")

    log.info(f"  📄 摘要: {summary_path}")

    log.info(f"\n{'='*65}")
    log.info(f"  回測完成: {elapsed:.0f}s, {len(successful)}/{len(months)} 成功")
    log.info(f"{'='*65}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="OceanMaster v10.3 回測框架"
    )
    parser.add_argument("--start", required=True,
                        help="開始月份 YYYY-MM")
    parser.add_argument("--end", required=True,
                        help="結束月份 YYYY-MM")
    parser.add_argument("--wcpfc", default="data/wcpfc",
                        help="WCPFC CSV 目錄")
    parser.add_argument("--output", default="output/backtest",
                        help="輸出目錄")
    parser.add_argument("--lat-min", type=float, default=5)
    parser.add_argument("--lat-max", type=float, default=35)
    parser.add_argument("--lon-min", type=float, default=120)
    parser.add_argument("--lon-max", type=float, default=175)
    parser.add_argument("--vessel-lat", type=float, default=25.13)
    parser.add_argument("--vessel-lon", type=float, default=121.74)
    args = parser.parse_args()

    asyncio.run(run_backtest(args))


if __name__ == "__main__":
    main()
